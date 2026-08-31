"""Fast ingestion: build the memory without the pricing tax (paper 7).

`read_text` prices the memory at every token --- reported perplexity,
two vocabulary-sized tier readouts, score mixing --- and pays two
rank-1 outer products per write. The write path proper needs none of
the pricing: the gate is one scalar the frozen model already computed.
Profiled on Qwen3 dimensions, the outer products alone are 11.4 of
13.3 ms/token (pure memory traffic).

`ingest_text` drops the pricing and applies the writes in 64-token
blocks, one GEMM per matrix. Exact BY CONSTRUCTION: cold-store
admissions, successor counts, surprise masses, gate statistics, token
counters, aging --- none reads the matrices. Bounded and declared:
trace amplitudes when the same 4-gram+successor repeats within one
block (the sqrt coefficient is computed against the block-start
matrix); the gate by float rounding when computed on the GPU; the
abstention reservoirs, sampled one token in `res_every` (their rolling
quantile is unchanged on a stationary stream; the tiers stay silent
for the first ~500 samples either way). Greedy completions on the
validation document are identical to `read_text`'s (paper 7,
reproduction appendix).

The rank-16 adapter does not learn during a fast read: its delta rule
consumes the full probability vector at every position and is
inherently sequential (and it stores no facts --- paper 6). The
adapter itself is untouched and still serves at generation time.
Readout calibration also needs the full tier scores, so a model whose
readout is still calibrating must be read normally first.
"""

import time

import numpy as np

from .core import CAP, NGRAM, PRUNE_MARGIN

WINDOW, STRIDE = 1024, 512


def blocked_write(mem, Qg, Qs, toks, g_vec, grams):
    """Apply one block of writes: one GEMM per matrix, then the exact
    sequential cold-store / counter / aging updates. Pure numpy."""
    Vt = mem.V[toks]
    Ug = Qg @ mem.M
    aG = np.clip((Ug * Vt).sum(1), 0.0, None)
    coefG = (np.sqrt(aG * aG + g_vec) - aG).astype(np.float32)
    Us = None
    if Qs is not None:
        Us = Qs @ mem.MS
        aS = np.clip((Us * Vt).sum(1), 0.0, None)
        coefS = (np.sqrt(aS * aS + g_vec) - aS).astype(np.float32)
    mem.M += Qg.T @ (coefG[:, None] * Vt)
    if Qs is not None:
        mem.MS += Qs.T @ (coefS[:, None] * Vt)
    for k in range(len(toks)):
        gk = float(g_vec[k])
        tk = int(toks[k])
        gr = grams[k]
        if gr is not None:
            # the same margin the sequential path uses: without it a
            # long fast ingest runs far past the cap (measured 2.99x on
            # a 2000-gram cap), because eviction used to live only in
            # save() and this loop does not go through write_all
            if len(mem.cold) > mem.cold_max * PRUNE_MARGIN:
                mem.prune_cold()
            slot = mem.cold.setdefault(gr, [0.0, {}, {}])
            if len(slot) == 2:          # pre-1.2 slot: migrate in place
                slot.append({t: float(c) for t, c in slot[1].items()})
            slot[0] += gk
            slot[1][tk] = slot[1].get(tk, 0) + 1
            slot[2][tk] = slot[2].get(tk, 0.0) + gk
        mem.g_sum += gk
        mem.g_cnt += 1
        mem.tokens += 1
        mem.decay_step()
    return Ug, Us


def ingest_text(s, text, name="<ingest>", block=64, res_every=8,
                quiet=True, between_windows=None):
    """Stream one text through the memory, writes only. ~40x read_text
    on long documents. The state is NOT saved -- call s.save() after
    the stream, like read_text."""
    import torch
    tok, model = s.load_model()
    mem = s.mem
    if mem.collecting():
        raise SystemExit(
            "fast read: this model's readout is still calibrating -- "
            "run a normal read first (calibration needs the tier "
            "scores that fast ingestion skips).")
    ids = np.array(tok.encode(text), dtype=np.int64)
    n = len(ids) - 1
    if n < 1:
        return {"file": name, "tokens": 0}
    s.resolve_sem2(ids)
    mem.new_stream()
    # paper 8's tier keys on an early layer and consolidates at the end
    # of the document, which is exactly this path's shape; the classic
    # tier keys on the last layer and writes in the block
    sem2 = mem.sem2_layer if mem.semantic else None
    sem = mem.semantic and sem2 is None
    anchors, sem2_buf, null_buf = [], [], []
    anchor_idx, prev_kept, g_prev, null_stride = None, False, 0.0, 4
    nsp = {}

    def flush_sem2():
        nonlocal anchor_idx
        mem.sem2_flush(anchors, sem2_buf, null_buf)
        last = None if anchor_idx is None else anchors[anchor_idx]
        anchors.clear()
        sem2_buf.clear()
        null_buf.clear()
        anchor_idx = None
        if last is not None:
            anchors.append(last)
            anchor_idx = 0

    use_gpu = "cuda" in str(s.device or "")
    Vg = None
    x = torch.tensor(ids, device=s.device)
    a = cnt = 0
    t0 = time.time()
    with torch.no_grad():
        while a < n:
            w = min(WINDOW, len(ids) - a)
            need_h = sem or sem2 is not None
            out = model(x[a:a + w].unsqueeze(0),
                        output_hidden_states=need_h)
            lg = out.logits[0].float()
            mem.set_vocab(lg.shape[-1])
            hi = min(w, n - a)
            tr = x[a + 1:a + hi + 1]
            lp_t = torch.log_softmax(lg[:hi], dim=-1).gather(
                1, tr.unsqueeze(1))[:, 0].cpu().numpy()
            if use_gpu and Vg is None:
                Vg = torch.tensor(mem.V, device=s.device)
            hs = (out.hidden_states[-1][0].float().cpu().numpy()
                  if sem else None)
            hs2 = (out.hidden_states[sem2][0].float().cpu().numpy()
                   if sem2 is not None else None)
            lo = 0 if a == 0 else WINDOW - STRIDE
            pos = list(range(lo, hi))
            b0 = 0
            while b0 < len(pos):
                blk = pos[b0:b0 + block]
                B = len(blk)
                Qg = np.empty((B, mem.M.shape[0]), np.float32)
                grams = []
                Qs = (np.empty((B, mem.MS.shape[0]), np.float32)
                      if sem else None)
                toks = np.array([int(ids[a + i + 1]) for i in blk])
                g_vec = np.clip(-lp_t[np.array(blk)], 0.0, CAP)
                for k, i in enumerate(blk):
                    Qg[k] = np.asarray(mem.step_key(int(ids[a + i])),
                                       dtype=np.float32)
                    grams.append(
                        np.array(mem._hist[-NGRAM:],
                                 dtype=np.int32).tobytes()
                        if len(mem._hist) >= NGRAM else None)
                    if sem:
                        Qs[k] = np.asarray(mem.sem_key(hs[i]),
                                           dtype=np.float32)
                    elif sem2 is not None:
                        # same rules as the normal read: anchor on the
                        # last surprising token, keep writes that
                        # surprise or that finish a kept word, and defer
                        # everything to the flush
                        mem.sem2_observe(hs2[i])
                        if g_prev >= 2.5:
                            anchors.append(hs2[i].copy())
                            anchor_idx = len(anchors) - 1
                            if len(anchors) >= 8192:
                                flush_sem2()
                        tk, g2 = int(toks[k]), float(g_vec[k])
                        if tk not in nsp:
                            d2 = tok.decode([tk])
                            nsp[tk] = (len(d2) > 0
                                       and not d2[0].isspace())
                        keep = g2 >= 0.5 or (prev_kept and nsp[tk])
                        if anchor_idx is not None and keep:
                            sem2_buf.append((anchor_idx, tk, g2))
                        elif (cnt + k) % null_stride == 0:
                            null_buf.append(hs2[i].copy())
                            if len(null_buf) > 4096:
                                del null_buf[::2]
                                null_stride *= 2
                        prev_kept = keep
                        g_prev = g2
                Ug, Us = blocked_write(mem, Qg, Qs, toks, g_vec, grams)
                sel = [k for k in range(B)
                       if (cnt + k) % res_every == 0]
                if sel:
                    pairs = [(Ug, mem.res_G)]
                    if sem:
                        pairs.append((Us, mem.res_S))
                    for U, res in pairs:
                        nn = np.linalg.norm(U[sel], axis=1) + 1e-8
                        if use_gpu:
                            Ut = torch.tensor(U[sel], device=s.device)
                            mx = ((Ut @ Vg.T).max(dim=1).values
                                  .cpu().numpy())
                        else:
                            mx = (U[sel] @ mem.V.T).max(axis=1)
                        res.extend((mx / nn).tolist())
                cnt += B
                b0 += block
                if not quiet and cnt % 4096 < block:
                    rate = cnt / max(1e-6, time.time() - t0)
                    print(f"  ... {cnt}/{n} tokens ({rate:.0f} tok/s)",
                          flush=True)
                if between_windows is not None:
                    # a server holds the state's lock while ingesting;
                    # yielding after each 64-token block, not only at the
                    # window boundary, is what keeps a conversation alive
                    # during a read that fits in a single window
                    between_windows()
            if between_windows is not None:
                between_windows()          # and once per window, as before
            if a + w >= len(ids):
                break
            a += STRIDE
    if sem2 is not None:
        flush_sem2()
    mem.res_G = mem.res_G[-5000:]
    mem.res_S = mem.res_S[-5000:]
    mins = (time.time() - t0) / 60
    rec = {"file": name, "tokens": int(cnt),
           "date": time.strftime("%Y-%m-%d %H:%M"),
           "minutes": round(mins, 2),
           "tok_per_s": round(cnt / max(1e-6, mins * 60), 1),
           "ppl_frozen": None, "ppl_fastweights": None,
           "ppl_with_memory": None}
    mem.log["files"].append({k: rec[k] for k in
                             ("file", "tokens", "date", "ppl_frozen",
                              "ppl_fastweights", "ppl_with_memory")})
    return rec
