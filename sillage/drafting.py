"""Speculative decoding from the memory itself (paper 5).

`complete --fast` drafts the next tokens from the state alone -- the cold
store first (exact continuations), then the n-gram matrix above the same
abstention threshold the readout already uses -- and verifies a whole block
with ONE forward pass of the frozen model. Greedy verification reproduces
plain greedy decoding token for token, whatever the drafter proposes: the
flag can only change speed, never content. Nothing is ever written.

The loop is the fused single-forward shape measured in paper 5 (x1.63-1.98
on a T4 at 70-87% draft acceptance on text the memory knows; on a CPU the
verification of k tokens costs nearly k times one token, so the gain there
is bounded by per-call overhead -- x1.4 measured at best, and ~x1.0 with a
small overhead on text the memory never read, where the drafter abstains).

Exactness has one moving part worth naming: the semantic tier's centering
mean `mu` advances at every scored position. The fast path therefore
snapshots it per round and replays it over the accepted prefix, so the
state trajectory -- and hence every score -- matches plain decoding
exactly. fp16 targets can still resolve argmax near-ties differently
between chunked and one-token forwards (paper 5, negative result 3); in
float32, the default everywhere in this tool, outputs are identical.
"""

import numpy as np

from .core import D_K

GAMMA = 8              # draft lane length
P_COLD_MIN = 0.35      # cold-store confidence to draft
MG_MAX_RUN = 2         # consecutive matrix-only drafts


def _cache_len(past):
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    return int(past[0][0].shape[2])


def _crop(past, n):
    if past is None or _cache_len(past) <= n:
        return past
    if hasattr(past, "crop"):
        past.crop(n)
        return past
    return tuple((k[:, :, :n, :], v[:, :, :n, :]) for k, v in past)


class _Drafter:
    """Proposals from the state alone; snapshot/rewind keeps it honest."""

    def __init__(self, mem, gamma=GAMMA):
        self.mem = mem
        self.gamma = gamma
        self.thr = mem.thresholds()[0]      # the readout's own gate

    def snapshot(self):
        m = self.mem
        return m._graw.copy(), list(m._hist)

    def restore(self, snap):
        self.mem._graw, self.mem._hist = snap[0].copy(), list(snap[1])

    def propose(self):
        """Up to gamma tokens plus per-position (sG, cold) readouts.

        readouts[i] belongs to the position that PREDICTS drafts[i]; one
        extra entry covers the bonus position after the last draft.
        """
        m = self.mem
        drafts, readouts = [], []
        mg_run = 0
        for _ in range(self.gamma):
            q = m._graw / np.sqrt(D_K)
            _, sG = m.scores(m.M, q)
            pc = m.cold_lookup()
            readouts.append((sG, pc))
            tok = None
            if pc:
                t_best, p_best = max(pc.items(), key=lambda kv: kv[1])
                if p_best >= P_COLD_MIN:
                    tok = int(t_best)
                    mg_run = 0
            if tok is None and float(sG.max()) >= self.thr \
                    and mg_run < MG_MAX_RUN:
                tok = int(np.argmax(sG))
                mg_run += 1
            if tok is None:
                break
            drafts.append(tok)
            m.step_key(tok)
        q = m._graw / np.sqrt(D_K)
        _, sG = m.scores(m.M, q)
        readouts.append((sG, m.cold_lookup()))
        return drafts, readouts


def complete_fast(rt, prompt, n=40, gamma=GAMMA):
    """Greedy completion, speculatively -- output identical to `complete`.

    `rt` is the Sillage runtime (frozen model + memory). Returns
    (text, stats); stats carries forwards/drafted/accepted for the curious.
    """
    import torch
    tok, model = rt.load_model()
    mem = rt.mem
    ids = tok.encode(prompt)
    mem.new_stream()
    for t in ids[:-1]:
        mem.step_key(int(t))
    thrG, thrS = mem.thresholds()
    need_h = mem.semantic or mem.fastweights
    drafter = _Drafter(mem, gamma)
    stats = {"forwards": 0, "rounds": 0, "drafted": 0, "accepted": 0}
    past = None
    if len(ids) > 1:
        with torch.no_grad():
            out = model(torch.tensor([ids[:-1]], device=rt.device),
                        use_cache=True)
        past = out.past_key_values
        stats["forwards"] += 1
    out_ids = []
    last = ids[-1]
    while len(out_ids) < n:
        mem.step_key(int(last))
        snap = drafter.snapshot()
        drafts, readouts = drafter.propose()
        stats["rounds"] += 1
        stats["drafted"] += len(drafts)
        base = _cache_len(past)
        chunk = [int(last)] + drafts
        with torch.no_grad():
            out = model(torch.tensor([chunk], device=rt.device),
                        past_key_values=past, use_cache=True,
                        output_hidden_states=need_h)
        past = out.past_key_values
        stats["forwards"] += 1
        logits = out.logits[0].float().cpu().numpy()
        mem.set_vocab(logits.shape[-1])
        hs = (out.hidden_states[-1][0].float().cpu().numpy()
              if need_h else None)
        # the semantic mean advances per scored position: snapshot it, and
        # replay only the accepted prefix afterwards, so state == plain
        mu_snap = (None if mem.mu is None else mem.mu.copy(), mem.mu_n)
        k = len(drafts) + 1
        accepted, emitted = 0, None
        for i in range(k):
            lb = logits[i]
            h = hs[i] if hs is not None else None
            la, _ = mem.adapt(lb, h)
            p_base = np.exp(la - la.max())
            p_base /= p_base.sum()
            sG, pc = readouts[i]
            sS = None
            if mem.semantic:
                qS = mem.sem_key(h)
                _, sS = mem.scores(mem.MS, qS)
            p = mem.mix_full(p_base, sG, sS, pc, thrG, thrS)
            t_i = int(np.argmax(p))
            if i < len(drafts) and t_i == drafts[i]:
                accepted += 1
                continue
            emitted = t_i
            break
        stats["accepted"] += accepted
        # rewind the key to the accepted prefix and the semantic mean to the
        # positions plain decoding would actually have scored
        drafter.restore(snap)
        for t in drafts[:accepted]:
            mem.step_key(int(t))
        if mem.semantic and hs is not None:
            mem.mu = mu_snap[0]
            mem.mu_n = mu_snap[1]
            for i in range(accepted + 1):
                mem.sem_key(hs[i])
        out_ids.extend(drafts[:accepted])
        out_ids.append(emitted)
        out_ids = out_ids[:n]
        past = _crop(past, base + 1 + accepted)
        if emitted == getattr(tok, "eos_token_id", -1) or len(out_ids) >= n:
            break
        last = emitted
    stats["tokens"] = len(out_ids)
    return tok.decode(out_ids), stats
