"""Speculative decoding with a Sillage memory as the drafter.

The claim under test (axe 2 / paper 5): the same fixed-size state that lowers
perplexity on text the model has read can also *accelerate* generation, by
proposing continuations that the augmented model then verifies in a single
batched forward pass. Greedy verification makes the output token-for-token
IDENTICAL to normal greedy decoding of the augmented model -- speculation can
only change speed, never content. The acceptance rate is the model-free
quantity; wall-clock speedup is what a user feels.

Drafting sources, in order of confidence (no language model involved):
  1. the cold store: exact 4-gram -> successor distribution (paper 3);
  2. the fast n-gram matrix M_G: argmax of the retrieval scores, only above
     the same abstention threshold the readout already uses (papers 1-2).
The draft chain advances the sliding key with its own proposals and stops as
soon as neither source is confident. A snapshot/replay of the key state keeps
the memory honest when drafts are rejected.

Everything here READS the memory; nothing is ever written (like `complete`).
"""

import os
import sys
import time

import numpy as np

# Resolve the sillage package: the repository root when this file lives in
# spec/, the directory itself in a flat kit, else the pip-installed package.
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break
from sillage.core import D_K, NGRAM, SillageMemory  # noqa: E402


# --------------------------------------------------------------- KV cache ---

def cache_len(past):
    """Sequence length already held by the cache."""
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    return int(past[0][0].shape[2])


def crop_cache(past, n):
    """Keep the first n positions of the KV cache (rejection rewind)."""
    if past is None or cache_len(past) <= n:
        return past
    if hasattr(past, "crop"):
        past.crop(n)
        return past
    return tuple((k[:, :, :n, :], v[:, :, :n, :]) for k, v in past)


# --------------------------------------------------------------- drafters ---

class SillageDrafter:
    """Draft next tokens from the memory alone (cold store, then M_G).

    The DRAFT gate is deliberately looser than the READOUT's abstention
    threshold: a wrong draft costs one wasted verification lane, never a
    wrong output, so the drafter can afford to speak where the readout
    would stay silent. thr_q is the reservoir quantile of the draft gate
    (the readout keeps its own q75 untouched).
    """

    def __init__(self, mem, gamma=6, p_cold_min=0.35, thr_q=0.75,
                 mg_max_run=2):
        self.mem = mem
        self.gamma = gamma
        self.p_cold_min = p_cold_min
        self.mg_max_run = mg_max_run   # long chains only from the cold store
        res = mem.res_G if len(mem.res_G) >= 500 else None
        self.thrG = (float(np.quantile(res, thr_q)) if res is not None
                     else mem.thresholds()[0])
        self.last_sources = []

    def snapshot(self):
        m = self.mem
        return m._graw.copy(), list(m._hist)

    def restore(self, snap):
        self.mem._graw, self.mem._hist = snap[0].copy(), list(snap[1])

    def scores_now(self):
        """M_G scores for the position after the tokens bound so far."""
        m = self.mem
        q = m._graw / np.sqrt(D_K)
        u, s = m.scores(m.M, q)
        return s

    def propose(self):
        """Draft up to gamma tokens; returns (drafts, per-position readouts).

        readouts[i] = (sG, pc) aligned with the position that PREDICTS
        drafts[i]; a final extra entry covers the bonus position after the
        last draft, so the verifier can emit one corrected/bonus token per
        round with no extra memory reads.
        """
        m = self.mem
        drafts, readouts = [], []
        self.last_sources = []
        mg_run = 0
        for _ in range(self.gamma):
            sG = self.scores_now()
            pc = m.cold_lookup()
            readouts.append((sG, pc))
            tok, src = None, None
            if pc:
                t_best, p_best = max(pc.items(), key=lambda kv: kv[1])
                if p_best >= self.p_cold_min:
                    tok, src = int(t_best), "cold"
                    mg_run = 0
            if tok is None and float(sG.max()) >= self.thrG \
                    and mg_run < self.mg_max_run:
                tok, src = int(np.argmax(sG)), "mg"
                mg_run += 1
            if tok is None:
                break
            drafts.append(tok)
            self.last_sources.append(src)
            m.step_key(tok)
        readouts.append((self.scores_now(), m.cold_lookup()))
        return drafts, readouts

    def advance(self, tok):
        self.mem.step_key(int(tok))


class PromptLookupDrafter:
    """Same speculation, no persistence: n-gram match in the CURRENT context.

    The ablation that separates 'speculation helps' from 'a persistent
    cross-session memory helps': this drafter can only copy from the prompt
    and from what was generated in this very call (prompt-lookup decoding).
    It still needs the memory object -- only to serve the verifier the same
    (sG, pc) readouts, so the TARGET distribution is identical across
    drafters and the outputs stay comparable token for token.
    """

    def __init__(self, mem, gamma=6, min_match=4):
        self.mem = mem
        self.gamma = gamma
        self.min_match = min_match
        self.context = []            # prompt + accepted tokens, maintained
        self.thrG, _ = mem.thresholds()

    def snapshot(self):
        m = self.mem
        return m._graw.copy(), list(m._hist), len(self.context)

    def restore(self, snap):
        self.mem._graw, self.mem._hist = snap[0].copy(), list(snap[1])
        del self.context[snap[2]:]

    def scores_now(self):
        m = self.mem
        q = m._graw / np.sqrt(D_K)
        _, s = m.scores(m.M, q)
        return s

    def _lookup(self):
        """Continuation after the previous occurrence of the last n-gram."""
        c, n = self.context, self.min_match
        if len(c) <= n:
            return None
        tail = c[-n:]
        for start in range(len(c) - n - 1, -1, -1):
            if c[start:start + n] == tail:
                nxt = start + n
                if nxt < len(c):
                    return c[nxt]
                return None
        return None

    def propose(self):
        m = self.mem
        drafts, readouts = [], []
        for _ in range(self.gamma):
            readouts.append((self.scores_now(), m.cold_lookup()))
            tok = self._lookup()
            if tok is None:
                break
            drafts.append(int(tok))
            self.context.append(int(tok))
            m.step_key(int(tok))
        readouts.append((self.scores_now(), m.cold_lookup()))
        return drafts, readouts

    def advance(self, tok):
        self.context.append(int(tok))
        self.mem.step_key(int(tok))


# --------------------------------------------------------------- decoding ---

class SpeculativeSillage:
    """Greedy generation for a frozen LM + Sillage state, with or without
    speculation. The augmented target distribution is computed exactly as in
    `sillage.runtime.Sillage.complete` (adapter, then score-level mixing with
    abstention, then cold store); greedy verification therefore reproduces
    the plain decoding output bit for bit.
    """

    def __init__(self, state_dir, device="cpu", target_hub=None,
                 memory_in_target=True, fastweights=None, dtype="float32"):
        """target_hub: generate with ANOTHER model than the one that wrote
        the memory -- it must share the tokenizer (e.g. the Qwen3 family).
        memory_in_target: mix the memory into the target's distribution
        (the augmented model) or leave the target vanilla (pure-acceleration
        setting: the memory then only drafts, and cannot change the output).
        fastweights=False is REQUIRED cross-model: the adapter was trained
        against the reading model's hidden geometry.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(os.cpu_count() or 4)
        self.mem = SillageMemory(state_dir, fastweights=fastweights)
        self.memory_in_target = memory_in_target
        self.tok = AutoTokenizer.from_pretrained(self.mem.hub)
        hub = target_hub or self.mem.hub
        self.target_hub = hub
        td = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(hub, dtype=td)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.eos = getattr(self.tok, "eos_token_id", -1)

    # ----------------------------------------------------------- target ----
    def _target_token(self, logits, h, sG, pc, thrG):
        """argmax of the target at one position (never writes)."""
        if not self.memory_in_target:
            return int(np.argmax(logits))
        m = self.mem
        la, _ = m.adapt(logits, h)
        p = np.exp(la - la.max())
        p /= p.sum()
        p = m.mix_full(p, sG, None, pc, thrG, None)
        return int(np.argmax(p))

    def _forward(self, ids, past):
        t = self.torch
        with t.no_grad():
            out = self.model(
                t.tensor([ids], device=self.device), past_key_values=past,
                use_cache=True, output_hidden_states=True)
        logits = out.logits[0].float().cpu().numpy()
        hs = out.hidden_states[-1][0].float().cpu().numpy()
        return logits, hs, out.past_key_values

    def _prime(self, prompt_ids):
        """Bind the prompt into the key state; prefill the KV cache."""
        m = self.mem
        m.new_stream()
        for t in prompt_ids[:-1]:
            m.step_key(int(t))
        logits, hs, past = self._forward(prompt_ids, None)
        if logits.shape[-1] != m.vocab:
            raise SystemExit(
                f"target vocabulary ({logits.shape[-1]}) does not match the "
                f"memory's token space ({m.vocab}): cross-model speculation "
                f"needs a shared tokenizer.")
        return logits[-1], hs[-1], past

    # --------------------------------------------------------- baseline ----
    def generate_plain(self, prompt, n=48):
        """One forward per token -- the reference output and reference time."""
        m = self.mem
        ids = self.tok.encode(prompt)
        thrG, _ = m.thresholds()
        stats = {"forwards": 1, "tokens": 0}
        logits, h, past = self._prime(ids)
        out_ids = []
        last = ids[-1]
        for _ in range(n):
            m.step_key(int(last))
            q = m._graw / np.sqrt(D_K)
            _, sG = m.scores(m.M, q)
            pc = m.cold_lookup()
            nxt = self._target_token(logits, h, sG, pc, thrG)
            out_ids.append(nxt)
            stats["tokens"] += 1
            if nxt == self.eos:
                break
            lg, hh, past = self._forward([nxt], past)
            stats["forwards"] += 1
            logits, h = lg[-1], hh[-1]
            last = nxt
        return out_ids, stats

    # ------------------------------------------------------- speculative ----
    def generate_spec(self, prompt, drafter_cls, n=48, **dr_kw):
        """Speculative loop: draft, verify in one forward, accept prefix."""
        m = self.mem
        ids = self.tok.encode(prompt)
        thrG, _ = m.thresholds()
        drafter = drafter_cls(m, **dr_kw)
        if isinstance(drafter, PromptLookupDrafter):
            # ids[-1] is bound by the first advance() at the top of the loop
            drafter.context = list(ids[:-1])
        stats = {"forwards": 1, "tokens": 0, "rounds": 0,
                 "drafted": 0, "accepted": 0,
                 "drafted_cold": 0, "acc_cold": 0,
                 "drafted_mg": 0, "acc_mg": 0}
        logits, h, past = self._prime(ids)
        out_ids = []
        last = ids[-1]
        pending = (logits, h)     # logits/hidden that predict the next token
        while len(out_ids) < n:
            drafter.advance(last)             # bind the last emitted token
            snap = drafter.snapshot()
            drafts, readouts = drafter.propose()
            sources = list(getattr(drafter, "last_sources", []))
            stats["rounds"] += 1
            stats["drafted"] += len(drafts)
            for s in sources:
                stats["drafted_" + s] += 1
            if not drafts:
                # nothing confident: fall back to one plain step
                sG, pc = readouts[0]
                nxt = self._target_token(pending[0], pending[1], sG, pc, thrG)
                out_ids.append(nxt)
                stats["tokens"] += 1
                if nxt == self.eos or len(out_ids) >= n:
                    break
                lg, hh, past = self._forward([nxt], past)
                stats["forwards"] += 1
                pending = (lg[-1], hh[-1])
                last = nxt
                continue
            # one forward over the drafted chunk verifies every position
            base = cache_len(past)
            lg, hh, past = self._forward(drafts, past)
            stats["forwards"] += 1
            # position i is predicted by: pending (i=0) else lg[i-1]
            accepted = 0
            emitted = None
            for i in range(len(drafts) + 1):
                logit_i = pending[0] if i == 0 else lg[i - 1]
                h_i = pending[1] if i == 0 else hh[i - 1]
                sG, pc = readouts[i]
                t_i = self._target_token(logit_i, h_i, sG, pc, thrG)
                if i < len(drafts) and t_i == drafts[i]:
                    accepted += 1
                    continue
                emitted = t_i        # correction (i<len) or free bonus (i==len)
                break
            stats["accepted"] += accepted
            for s in sources[:accepted]:
                stats["acc_" + s] += 1
            keep = out_ids[:]
            keep.extend(drafts[:accepted])
            keep.append(emitted)
            out_ids = keep[:n]
            stats["tokens"] = len(out_ids)
            # rewind key state to the accepted prefix, then bind it back
            drafter.restore(snap)
            for t in drafts[:accepted]:
                drafter.advance(t)
            # cache holds base+len(drafts); accepted prefix stays, rest goes
            past = crop_cache(past, base + accepted)
            if emitted == self.eos or len(out_ids) >= n:
                break
            # the emitted token has not been forwarded yet: do it now, so the
            # next round starts from fresh logits (same role as `pending`)
            lg2, hh2, past = self._forward([emitted], past)
            stats["forwards"] += 1
            pending = (lg2[-1], hh2[-1])
            last = emitted
        return out_ids, stats


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out, stats = fn(*a, **kw)
    stats["seconds"] = time.perf_counter() - t0
    return out, stats
