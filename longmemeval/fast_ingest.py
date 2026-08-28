"""Fast ingestion: build the memory state of a long text WITHOUT the
per-token vocabulary-sized readout that `read_text` runs to report
perplexity (candidate feature for 1.3.0, born from the LongMemEval
campaign the way `--fast` was born from paper 5).

The write path of `read_text` needs, per token: the gate (the frozen
model's own surprise, one scalar from the logits), the 4-gram key, the
retrieved vector u = M.T @ q (256 dims), the semantic key when that
tier is on, and the cold-store update. None of that is
vocabulary-sized. What IS vocabulary-sized in `read_text` -- the
log-softmax for reported perplexity, the two `V @ u` tier readouts,
the score mixing -- exists to price the memory, not to build it.

This module replays the write path VERBATIM (same numpy expressions on
the same float32 logits, so gate and traces are bit-identical) and
keeps only one vocabulary product per active tier for the abstention
reservoirs (`res_G`/`res_S`, persisted, they set the thresholds used
at generation time). On CPU that product goes through `mem.scores`
exactly like `read_text` (bit-exact states, verified by
test_fast_ingest.py); on CUDA it runs against a GPU-resident copy of V
(declared tolerance ~1e-6 on reservoir values only -- they feed
quantiles).

Deliberately NOT done here: readout calibration (`collect` needs the
full tier scores) -- fast ingestion is for models whose readout is
already set (the presets, or a state calibrated by a normal read).
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break
sys.path.insert(0, HERE)

from sillage.core import CAP

WINDOW, STRIDE = 1024, 512


def fast_ingest(s, text, name="<ingest>", res_on_gpu=None, quiet=True,
                gate="exact", res_every=None):
    """Stream `text` through every memory tier of `s`, writes only.

    gate="exact"  the surprise gate is computed with read_text's exact
                  numpy arithmetic on the transferred float32 logits --
                  states are bit-identical to a normal read.
    gate="torch"  log_softmax runs where the logits already are (the
                  GPU) and only one scalar per position is gathered;
                  the gate differs from exact by float rounding (~1e-6
                  nats), so cold COUNTS and admissions are identical
                  while trace amplitudes drift at the same order --
                  declared tolerance, verified by test_fast_ingest.py.
                  This removes the last vocabulary-sized CPU op when
                  the adapter is off.

    res_every=K samples the abstention reservoirs one token in K
    (default: 1 in exact mode -- every token, like read_text -- and 8
    in torch mode). The reservoirs feed a rolling QUANTILE over their
    last 5000 samples; sampling widens the window those samples span,
    it does not bias the estimator on a stationary stream -- declared
    tolerance on the thresholds, verified by test_fast_ingest.py. The
    per-token GPU round-trip for the reservoir product is the dominant
    ingestion cost, so K divides it directly.

    Returns a small record (tokens, minutes, tokens/s). The state is
    NOT saved -- call s.save() when the stream is done, like read_text.
    """
    import torch
    tok, model = s.load_model()
    mem = s.mem
    if mem.collecting():
        raise SystemExit(
            "fast_ingest: this model's readout is still calibrating -- "
            "run a normal read first (calibration needs the tier scores "
            "that fast ingestion skips).")
    ids = np.array(tok.encode(text), dtype=np.int64)
    n = len(ids) - 1
    if n < 1:
        return {"file": name, "tokens": 0}
    mem.new_stream()
    need_h = mem.semantic or mem.fastweights
    use_gpu = (res_on_gpu if res_on_gpu is not None
               else "cuda" in str(s.device or ""))
    torch_gate = (gate == "torch")
    need_np_logits = mem.fastweights or not torch_gate
    if res_every is None:
        res_every = 8 if torch_gate else 1
    Vg = None
    x = torch.tensor(ids, device=s.device)
    a = cnt = 0
    t0 = time.time()
    with torch.no_grad():
        while a < n:
            w = min(WINDOW, len(ids) - a)
            out = model(x[a:a + w].unsqueeze(0),
                        output_hidden_states=need_h)
            lg = out.logits[0].float()
            mem.set_vocab(lg.shape[-1])
            if torch_gate:
                hi = min(w, n - a)
                tr = x[a + 1:a + hi + 1]
                lps = torch.log_softmax(lg[:hi], dim=-1)
                lp_t = lps.gather(1, tr.unsqueeze(1))[:, 0].cpu().numpy()
            logits = lg.cpu().numpy() if need_np_logits else None
            if use_gpu and Vg is None:
                Vg = torch.tensor(mem.V, device=s.device)
            hs = (out.hidden_states[-1][0].float().cpu().numpy()
                  if need_h else None)
            lo = 0 if a == 0 else WINDOW - STRIDE
            for i in range(lo, w):
                j = a + i
                if j >= n:
                    break
                truth = int(ids[j + 1])
                lb = logits[i] if need_np_logits else None
                if torch_gate:
                    lp = float(lp_t[i])
                else:
                    # gate: verbatim read_text arithmetic -> bit-exact g
                    mx = lb.max()
                    lp = float(lb[truth]
                               - (mx + np.log(np.exp(lb - mx).sum())))
                phi = p_ad = None
                if mem.fastweights:
                    la, phi = mem.adapt(lb, hs[i] if need_h else None)
                    if phi is not None:
                        m2 = la.max()
                        p_ad = np.exp(la - m2)
                        p_ad /= p_ad.sum()
                qG = mem.step_key(int(ids[j]))
                if torch_gate:
                    qG = np.asarray(qG, dtype=np.float32)
                take = (cnt % res_every == 0)
                if use_gpu:
                    uG = mem.M.T @ qG
                    if take:
                        un = float(np.linalg.norm(uG)) + 1e-8
                        ug_t = torch.tensor(
                            np.asarray(uG, dtype=np.float32),
                            device=s.device)
                        mem.res_G.append(float((Vg @ ug_t).max()) / un)
                else:
                    uG, sG = mem.scores(mem.M, qG)
                    if take:
                        mem.res_G.append(float(sG.max()))
                qS = uS = None
                if mem.semantic:
                    qS = mem.sem_key(hs[i])
                    if torch_gate:
                        qS = np.asarray(qS, dtype=np.float32)
                    if use_gpu:
                        uS = mem.MS.T @ qS
                        if take:
                            un = float(np.linalg.norm(uS)) + 1e-8
                            us_t = torch.tensor(
                                np.asarray(uS, dtype=np.float32),
                                device=s.device)
                            mem.res_S.append(float((Vg @ us_t).max()) / un)
                    else:
                        uS, sS = mem.scores(mem.MS, qS)
                        if take:
                            mem.res_S.append(float(sS.max()))
                g = min(CAP, max(0.0, -lp))
                mem.write_all(qG, uG, qS, uS, truth, g, phi, p_ad)
                cnt += 1
                if not quiet and cnt % 2000 == 0:
                    rate = cnt / max(1e-6, time.time() - t0)
                    print(f"  ... {cnt}/{n} tokens ({rate:.0f} tok/s, "
                          f"{(n - cnt) / rate / 60:.1f} min left)",
                          flush=True)
            if a + w >= len(ids):
                break
            a += STRIDE
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


def fast_ingest_blocked(s, text, name="<ingest>", block=64, res_every=8,
                        quiet=True):
    """Blocked-GEMM ingestion: the write path applied in blocks of
    `block` tokens against matrices frozen at block start -- one GEMM
    per matrix per block instead of one rank-1 outer product per token
    (which the profile shows is 11.4 of the 13.3 ms/token: pure memory
    traffic).

    EXACT by construction: the surprise gate (GPU log-softmax), the
    cold store -- admissions, counts, masses -- the gate statistics,
    the token counter and the aging, none of which read the matrices.
    Declared tolerance: trace amplitudes under-escalate when the SAME
    4-gram+successor repeats within one block (the sqrt coefficient is
    computed against the block-start matrix); reservoirs are sampled
    one token in `res_every` and read the block-start matrix too.
    Behavioral equality and bounds are verified by test_fast_ingest.py.
    Requires the adapter off (its delta rule is inherently sequential).
    """
    import torch
    from sillage.core import NGRAM
    tok, model = s.load_model()
    mem = s.mem
    if mem.fastweights:
        raise SystemExit("fast_ingest_blocked: the adapter must be off "
                         "(fastweights=False).")
    if mem.collecting():
        raise SystemExit(
            "fast_ingest_blocked: this model's readout is still "
            "calibrating -- run a normal read first.")
    ids = np.array(tok.encode(text), dtype=np.int64)
    n = len(ids) - 1
    if n < 1:
        return {"file": name, "tokens": 0}
    mem.new_stream()
    sem = mem.semantic
    use_gpu = "cuda" in str(s.device or "")
    Vg = None
    x = torch.tensor(ids, device=s.device)
    a = cnt = 0
    t0 = time.time()
    D_K = mem.M.shape[0]
    with torch.no_grad():
        while a < n:
            w = min(WINDOW, len(ids) - a)
            out = model(x[a:a + w].unsqueeze(0), output_hidden_states=sem)
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
            lo = 0 if a == 0 else WINDOW - STRIDE
            pos = list(range(lo, hi))
            b0 = 0
            while b0 < len(pos):
                blk = pos[b0:b0 + block]
                B = len(blk)
                Qg = np.empty((B, D_K), np.float32)
                grams = []
                Qs = (np.empty((B, mem.MS.shape[0]), np.float32)
                      if sem else None)
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
                toks = np.array([int(ids[a + i + 1]) for i in blk])
                g_vec = np.clip(-lp_t[np.array(blk)], 0.0, CAP)
                Vt = mem.V[toks]
                Ug = Qg @ mem.M
                aG = np.clip((Ug * Vt).sum(1), 0.0, None)
                coefG = (np.sqrt(aG * aG + g_vec) - aG).astype(np.float32)
                if sem:
                    Us = Qs @ mem.MS
                    aS = np.clip((Us * Vt).sum(1), 0.0, None)
                    coefS = (np.sqrt(aS * aS + g_vec)
                             - aS).astype(np.float32)
                sel = [k for k in range(B) if (cnt + k) % res_every == 0]
                if sel:
                    for U, res in (((Ug, mem.res_G),) +
                                   (((Us, mem.res_S),) if sem else ())):
                        nn = np.linalg.norm(U[sel], axis=1) + 1e-8
                        if use_gpu:
                            Ut = torch.tensor(U[sel], device=s.device)
                            mx = ((Ut @ Vg.T).max(dim=1).values
                                  .cpu().numpy())
                        else:
                            mx = (U[sel] @ mem.V.T).max(axis=1)
                        res.extend((mx / nn).tolist())
                mem.M += Qg.T @ (coefG[:, None] * Vt)
                if sem:
                    mem.MS += Qs.T @ (coefS[:, None] * Vt)
                for k in range(B):
                    gr = grams[k]
                    gk = float(g_vec[k])
                    tk = int(toks[k])
                    if gr is not None:
                        slot = mem.cold.setdefault(gr, [0.0, {}, {}])
                        if len(slot) == 2:
                            slot.append({t2: float(c) for t2, c
                                         in slot[1].items()})
                        slot[0] += gk
                        slot[1][tk] = slot[1].get(tk, 0) + 1
                        slot[2][tk] = slot[2].get(tk, 0.0) + gk
                    mem.g_sum += gk
                    mem.g_cnt += 1
                    mem.tokens += 1
                    mem.decay_step()
                cnt += B
                b0 += block
                if not quiet and cnt % 4096 < block:
                    rate = cnt / max(1e-6, time.time() - t0)
                    print(f"  ... {cnt}/{n} tokens ({rate:.0f} tok/s)",
                          flush=True)
            if a + w >= len(ids):
                break
            a += STRIDE
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
