"""GPU-resident readout for the Sillage speculative drafter.

The Kaggle v5 runs proved the science (acceptance 75-87% at three target
sizes, 16-token verification as cheap as 1 token) and exposed the
bottleneck: the numpy readout -- V@u, softmaxes and mixing over a
151,936-token vocabulary -- costs ~130 ms/token on a 2-core Kaggle CPU,
dwarfing the 44 ms GPU forward. This module keeps that entire hot path on
the model's device.

Faithfulness: T, V and Rf are generated with the SAME numpy seeds as
sillage.core and moved to the device once, so the state's trained adapter A
and the Hebbian matrix M are reused unchanged. The target distribution
(adapter, score-level mixing with abstention, cold store) mirrors
`sillage.runtime.complete`; plain and speculative decoding share it, so
outputs are identical within this engine by construction.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break
from sillage.core import (CAP, COLD_MIN_COUNT, D_K, D_V, LAM_C, NGRAM,  # noqa
                          SillageMemory)

from sillage_drafter import (PromptLookupDrafter, SillageDrafter,  # noqa
                             cache_len, crop_cache)


class TorchEngine:
    """Frozen LM + Sillage state, hot path on the device."""

    def __init__(self, state_dir, device="cuda", target_hub=None,
                 memory_in_target=True, fastweights=None, dtype="float16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(os.cpu_count() or 4)
        self.mem = SillageMemory(state_dir, fastweights=fastweights)
        m = self.mem
        self.memory_in_target = memory_in_target
        self.tok = AutoTokenizer.from_pretrained(m.hub)
        hub = target_hub or m.hub
        self.target_hub = hub
        td = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(hub, dtype=td)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.eos = getattr(self.tok, "eos_token_id", -1)

        # ---- device-resident copies of the state (seeds identical) ----
        dev = torch.device(device)
        self.T = torch.from_numpy(np.ascontiguousarray(m.T)).to(dev)  # int8
        self.V = torch.from_numpy(m.V).to(dev, torch.float16)
        self.M = torch.from_numpy(m.M).to(dev, torch.float32)
        self.A = (torch.from_numpy(m.A).to(dev, torch.float32)
                  if m.fastweights else None)
        self._Rf = None
        self.beta_G = float(m.beta_G)
        self.lam_G = float(m.lam_G)
        thrG, _ = m.thresholds()
        self.thrG = float(thrG)
        res = m.res_G if len(m.res_G) >= 500 else None
        self.thr_draft = (float(np.quantile(res, 0.75)) if res is not None
                          else self.thrG)
        self.graw = None
        self.hist = []

    def refresh_settings(self):
        """Re-read (beta, lam, threshold) from the memory (post-override)."""
        m = self.mem
        self.beta_G, self.lam_G = float(m.beta_G), float(m.lam_G)
        self.thrG = float(m.thresholds()[0])

    # ------------------------------------------------------------- keys ----
    def new_stream(self):
        self.graw = self.torch.ones(D_K, dtype=self.torch.int8,
                                    device=self.device)
        self.hist = []

    def step_key(self, tok):
        t = self.torch
        self.graw = t.roll(self.graw, 1) * self.T[tok]
        self.hist.append(int(tok))
        if len(self.hist) > NGRAM:
            old = self.hist.pop(0)
            self.graw = self.graw * t.roll(self.T[old], NGRAM)

    def snapshot(self):
        return self.graw.clone(), list(self.hist)

    def restore(self, snap):
        self.graw, self.hist = snap[0].clone(), list(snap[1])

    def scores_now(self):
        """[vocab] fp32 scores of the memory for the next position."""
        q = self.graw.to(self.torch.float32) / (D_K ** 0.5)
        u = q @ self.M                                   # [D_V]
        un = self.torch.linalg.vector_norm(u) + 1e-8
        s = (self.V @ u.to(self.torch.float16)).to(self.torch.float32) / un
        return s

    def cold_lookup(self):
        if len(self.hist) < NGRAM:
            return None
        gram = np.array(self.hist[-NGRAM:], dtype=np.int32).tobytes()
        slot = self.mem.cold.get(gram)
        if slot is None or sum(slot[1].values()) < COLD_MIN_COUNT:
            return None
        tot = sum(slot[1].values())
        return {t: c / tot for t, c in slot[1].items()}

    # ----------------------------------------------------------- target ----
    def _phi(self, h):
        t = self.torch
        if self._Rf is None:
            rng = np.random.default_rng(7010)          # SEED_R of core.py
            rf = (rng.normal(size=(h.shape[-1], 16))
                  / np.sqrt(h.shape[-1])).astype(np.float32)
            self._Rf = t.from_numpy(rf).to(self.device)
        v = h.to(t.float32) @ self._Rf
        return v / (t.linalg.vector_norm(v) + 1e-8)

    def target_rows(self, logits_rows, h_rows, s_rows, colds):
        """Batched target argmax for a block of positions (the verify path).

        logits_rows: [k, vocab] device tensor (model dtype); h_rows: [k, d]
        or None; s_rows: [k, vocab] fp32; colds: list of k dicts or None.
        Returns k python ints.
        """
        t = self.torch
        lb = logits_rows.to(t.float32)
        if not self.memory_in_target:
            return t.argmax(lb, dim=-1).tolist()
        if self.A is not None and h_rows is not None:
            phi = h_rows.to(t.float32) @ self._phi_matrix(h_rows.shape[-1])
            phi = phi / (t.linalg.vector_norm(phi, dim=-1, keepdim=True)
                         + 1e-8)
            lb = lb + phi @ self.A.T
        p = t.softmax(lb, dim=-1)
        pm = t.softmax(self.beta_G * s_rows, dim=-1)
        gate = (s_rows.max(dim=-1).values >= self.thrG).to(t.float32)
        lam = self.lam_G * gate
        p = lam.unsqueeze(1) * pm + (1 - lam).unsqueeze(1) * p
        for i, pc in enumerate(colds):
            if pc:
                p[i] *= (1 - LAM_C)
                idx = t.tensor(list(pc.keys()), device=self.device,
                               dtype=t.long)
                val = t.tensor(list(pc.values()), device=self.device,
                               dtype=t.float32)
                p[i].index_add_(0, idx, LAM_C * val)
        return t.argmax(p, dim=-1).tolist()

    def _phi_matrix(self, hidden_dim):
        if self._Rf is None:
            rng = np.random.default_rng(7010)
            rf = (rng.normal(size=(hidden_dim, 16))
                  / np.sqrt(hidden_dim)).astype(np.float32)
            self._Rf = self.torch.from_numpy(rf).to(self.device)
        return self._Rf

    # ------------------------------------------------------------ model ----
    def _forward(self, ids, past):
        t = self.torch
        need_h = self.A is not None and self.memory_in_target
        with t.no_grad():
            out = self.model(t.tensor([ids], device=self.device),
                             past_key_values=past, use_cache=True,
                             output_hidden_states=need_h)
        hs = out.hidden_states[-1][0] if need_h else None
        return out.logits[0], hs, out.past_key_values

    def _prime(self, ids):
        self.new_stream()
        for tk in ids[:-1]:
            self.step_key(int(tk))
        logits, hs, past = self._forward(ids, None)
        if logits.shape[-1] != self.mem.vocab:
            raise SystemExit("vocabulaire cible != espace de tokens de la "
                             "memoire : tokenizer non partage.")
        return logits, hs, past

    # --------------------------------------------------------- decoding ----
    def generate_plain(self, prompt, n=48):
        ids = self.tok.encode(prompt)
        stats = {"forwards": 1, "tokens": 0}
        logits, hs, past = self._prime(ids)
        pend_l = logits[-1:].contiguous()
        pend_h = hs[-1:].contiguous() if hs is not None else None
        out_ids = []
        last = ids[-1]
        for _ in range(n):
            self.step_key(int(last))
            s = self.scores_now().unsqueeze(0)
            pc = self.cold_lookup()
            nxt = self.target_rows(pend_l, pend_h, s, [pc])[0]
            out_ids.append(nxt)
            stats["tokens"] += 1
            if nxt == self.eos:
                break
            lg, hh, past = self._forward([nxt], past)
            stats["forwards"] += 1
            pend_l = lg[-1:].contiguous()
            pend_h = hh[-1:].contiguous() if hh is not None else None
            last = nxt
        return out_ids, stats

    def _draft_sillage(self, gamma, p_cold_min=0.35, mg_max=2):
        drafts, s_rows, colds, sources = [], [], [], []
        mg_run = 0
        for _ in range(gamma):
            s = self.scores_now()
            pc = self.cold_lookup()
            s_rows.append(s)
            colds.append(pc)
            tok, src = None, None
            if pc:
                t_best, p_best = max(pc.items(), key=lambda kv: kv[1])
                if p_best >= p_cold_min:
                    tok, src = int(t_best), "cold"
                    mg_run = 0
            if tok is None and float(s.max()) >= self.thr_draft \
                    and mg_run < mg_max:
                tok, src = int(self.torch.argmax(s)), "mg"
                mg_run += 1
            if tok is None:
                break
            drafts.append(tok)
            sources.append(src)
            self.step_key(tok)
        s_rows.append(self.scores_now())
        colds.append(self.cold_lookup())
        return drafts, s_rows, colds, sources

    def _draft_pld(self, gamma, context, min_match=4):
        drafts, s_rows, colds = [], [], []
        for _ in range(gamma):
            s_rows.append(self.scores_now())
            colds.append(self.cold_lookup())
            tok = None
            c = context
            if len(c) > min_match:
                tail = c[-min_match:]
                for start in range(len(c) - min_match - 1, -1, -1):
                    if c[start:start + min_match] == tail:
                        nxt = start + min_match
                        if nxt < len(c):
                            tok = c[nxt]
                        break
            if tok is None:
                break
            drafts.append(int(tok))
            context.append(int(tok))
            self.step_key(int(tok))
        s_rows.append(self.scores_now())
        colds.append(self.cold_lookup())
        return drafts, s_rows, colds, ["pld"] * len(drafts)

    def generate_spec(self, prompt, drafter_cls, n=48, gamma=8, **_):
        """Fused speculative loop: ONE forward per round.

        Each round's verification chunk is [last_emitted] + drafts, so the
        forward that verifies the block also produces the logits the next
        round needs -- the separate emitted-token forward of the two-pass
        loop disappears. Targets are computed identically, so the output is
        unchanged; only forwards/round drops from 2 to 1.
        """
        t = self.torch
        pld = drafter_cls is PromptLookupDrafter
        ids = self.tok.encode(prompt)
        context = list(ids[:-1]) if pld else None
        stats = {"forwards": 0, "tokens": 0, "rounds": 0,
                 "drafted": 0, "accepted": 0,
                 "drafted_cold": 0, "acc_cold": 0,
                 "drafted_mg": 0, "acc_mg": 0}
        # prime the key state on ids[:-1] and the KV cache likewise: ids[-1]
        # opens the first verification chunk instead of being prefilled
        self.new_stream()
        for tk in ids[:-1]:
            self.step_key(int(tk))
        past = None
        if len(ids) > 1:
            _, _, past = self._forward(ids[:-1], None)
            stats["forwards"] += 1
        checked = False
        out_ids = []
        last = ids[-1]
        while len(out_ids) < n:
            self.step_key(int(last))
            if pld:
                context.append(int(last))
            snap = self.snapshot()
            ctx_len = len(context) if pld else 0
            if pld:
                drafts, s_rows, colds, sources = self._draft_pld(
                    gamma, context)
            else:
                drafts, s_rows, colds, sources = self._draft_sillage(gamma)
            stats["rounds"] += 1
            stats["drafted"] += len(drafts)
            for src in sources:
                if src in ("cold", "mg"):
                    stats["drafted_" + src] += 1
            base = cache_len(past)
            chunk = [int(last)] + drafts
            lg, hh, past = self._forward(chunk, past)
            stats["forwards"] += 1
            if not checked:
                if lg.shape[-1] != self.mem.vocab:
                    raise SystemExit(
                        "vocabulaire cible != espace de tokens de la "
                        "memoire : tokenizer non partage.")
                checked = True
            k = len(drafts) + 1
            s_rows_t = t.stack(s_rows[:k], dim=0)
            targets = self.target_rows(lg[:k], hh[:k] if hh is not None
                                       else None, s_rows_t, colds[:k])
            accepted = 0
            emitted = None
            for i in range(k):
                if i < len(drafts) and targets[i] == drafts[i]:
                    accepted += 1
                    continue
                emitted = targets[i]
                break
            stats["accepted"] += accepted
            for src in sources[:accepted]:
                if src in ("cold", "mg"):
                    stats["acc_" + src] += 1
            out_ids.extend(drafts[:accepted])
            out_ids.append(emitted)
            out_ids = out_ids[:n]
            stats["tokens"] = len(out_ids)
            self.restore(snap)
            if pld:
                del context[ctx_len:]
            for tk in drafts[:accepted]:
                self.step_key(int(tk))
                if pld:
                    context.append(int(tk))
            # chunk contributed 1+len(drafts) cache entries; keep last +
            # the accepted prefix, drop the rejected tail
            past = crop_cache(past, base + 1 + accepted)
            if emitted == self.eos or len(out_ids) >= n:
                break
            last = emitted
        return out_ids, stats
