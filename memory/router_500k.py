"""Router at 500k tokens: does the mixture of memories hold at long horizon,
and does leaky decay (both matrices) restore it under saturation?

Protocol identical to exp_500k (scoring every 4th position, writes at every
position, model-surprise gate); the semantic block uses the multi-resolution
nested bands and frozen whitening.

Usage: python router_500k.py <tolstoy|bible> [decay]
Output: results/router500k_<domain>[_decay].json
"""


# --- repo bootstrap: run this script from anywhere ---
import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "requirements.txt")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
for _sub in ("", "pipeline", "memory", "fastweights", "eval", "figures"):
    _p = _os.path.join(_d, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_d)
# --- end bootstrap ---

import json
import sys
import time

import numpy as np

from exp_500k import SEG, eval_positions
from memories import BETAS, CAP
from sillage_router import (B_LIST_MULTI, greedy_tune, paired_ci)
from sillage_semantic import (B_BITS, D_BAND, D_G, D_V, EPS_EIG, L_BANDS,
                           NGRAM, WHITEN_FIT, band_vector_cache)

DECAY_HALF_LIFE = 100_000
DECAY_EVERY = 64


def pass_500k(ids, H, LP, vals, epos, decay, vocab=50257, hv_seed=7001):
    rngV = np.random.default_rng(hv_seed)
    rngT = np.random.default_rng(hv_seed + 1)
    rngW = np.random.default_rng(hv_seed + 2)
    V = ((rngV.integers(0, 2, size=(vocab, D_V)) * 2.0 - 1.0)
         / np.sqrt(D_V)).astype(np.float32)
    T = (rngT.integers(0, 2, size=(vocab, D_G), dtype=np.int8) * 2 - 1)
    Wh = rngW.normal(size=(H.shape[1], L_BANDS * B_BITS)).astype(np.float32)
    getv = band_vector_cache()
    n_groups = len(B_LIST_MULTI)
    D_S = n_groups * L_BANDS * D_BAND
    M_G = np.zeros((D_G, D_V), dtype=np.float32)
    M_S = np.zeros((D_S, D_V), dtype=np.float32)
    n = len(vals)
    is_eval = np.zeros(n, dtype=bool)
    is_eval[epos] = True
    ne = len(epos)
    stats = {b: {"s_true": np.zeros(ne, np.float32),
                 "smax": np.zeros(ne, np.float32),
                 "lse": np.zeros((ne, len(BETAS)), np.float32),
                 "sumsq": np.zeros(ne, np.float32)} for b in ("G", "S")}
    betas = np.array(BETAS, dtype=np.float32)
    g_raw = np.ones(D_G, dtype=np.float32)
    invG = 1.0 / np.sqrt(D_G)
    mu, W_zca = None, None
    pw2 = 2 ** np.arange(B_BITS)
    gamma = 0.5 ** (DECAY_EVERY / DECAY_HALF_LIFE)
    ei = 0
    t0 = time.time()
    for j in range(n):
        g_raw = np.roll(g_raw, 1)
        g_raw *= T[ids[j]]
        if j >= NGRAM:
            g_raw *= np.roll(T[ids[j - NGRAM]], NGRAM)
        qG = g_raw * invG
        uG = M_G.T @ qG
        if j == WHITEN_FIT:
            X = H[:j].astype(np.float32)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            mu = X.mean(0)
            C = X - mu
            cov = (C.T @ C) / len(C)
            w, U = np.linalg.eigh(cov)
            W_zca = (U @ np.diag(1.0 / np.sqrt(w + EPS_EIG)) @ U.T
                     ).astype(np.float32)
        qS, uS = None, None
        if W_zca is not None:
            h = H[j].astype(np.float32)
            h /= np.linalg.norm(h) + 1e-8
            z = (h - mu) @ W_zca
            bits = ((z @ Wh) > 0).reshape(L_BANDS, B_BITS)
            qS = np.empty(D_S, dtype=np.float32)
            scale = 1.0 / np.sqrt(n_groups * L_BANDS * D_BAND)
            slot = 0
            for gi, b in enumerate(B_LIST_MULTI):
                for k in range(L_BANDS):
                    pat = int(bits[k, :b] @ pw2[:b])
                    qS[slot * D_BAND:(slot + 1) * D_BAND] = \
                        scale * getv(gi * L_BANDS + k, pat)
                    slot += 1
            uS = M_S.T @ qS
        if is_eval[j]:
            for block, u in (("G", uG),) + ((("S", uS),) if uS is not None
                                            else ()):
                un = float(np.linalg.norm(u)) + 1e-8
                s = (V @ u) / un
                d = stats[block]
                d["s_true"][ei] = s[vals[j]]
                d["smax"][ei] = s.max()
                m = betas[:, None] * s[None, :]
                mx = m.max(axis=1)
                d["lse"][ei] = mx + np.log(np.exp(m - mx[:, None]).sum(axis=1))
                r = np.maximum(s, 0.0)
                d["sumsq"][ei] = float((r * r).sum())
            ei += 1
        g = min(CAP, max(0.0, -float(LP[j])))
        if decay and j % DECAY_EVERY == 0 and j > 0:
            M_G *= gamma
            M_S *= gamma
        a = max(0.0, float(uG @ V[vals[j]]))
        M_G += (np.sqrt(a * a + g) - a) * qG[:, None] * V[vals[j]][None, :]
        if uS is not None:
            a = max(0.0, float(uS @ V[vals[j]]))
            M_S += (np.sqrt(a * a + g) - a) * qS[:, None] * V[vals[j]][None, :]
        if j % 100_000 == 0 and j > 0:
            print(f"  ... {j}/{n} ({(time.time()-t0)/60:.0f} min)", flush=True)
    return stats


def main():
    domain = sys.argv[1]
    decay = "decay" in sys.argv[2:]
    ids = np.load(f"data/{domain}_ids.npy")
    H = np.load(f"dumps/{domain}_h.npy", mmap_mode="r")
    LP = np.load(f"dumps/{domain}_lp.npy")
    vals = ids[1:].astype(np.int64)
    n = len(vals)
    epos = eval_positions(n)
    lp_e = LP[epos]
    dev_e = epos < int(0.2 * n)
    test_e = ~dev_e
    base_nll = float(-lp_e[test_e].mean())
    stats = pass_500k(ids, H, LP, vals, epos, decay)
    r = greedy_tune(stats, lp_e, dev_e, test_e)
    g_test = base_nll - float(-np.log(
        np.maximum(r["p_gonly"][test_e], 1e-30)).mean())
    r_test = base_nll - float(-np.log(
        np.maximum(r["p_router"][test_e], 1e-30)).mean())
    pc = paired_ci(r["p_gonly"], r["p_router"], test_e)
    segs = {}
    for name, p in (("g_only", r["p_gonly"]), ("router", r["p_router"])):
        d = (-lp_e) - (-np.log(np.maximum(p, 1e-30)))
        segs[name] = [float(d[(epos >= s0) & (epos < s0 + SEG)].mean())
                      for s0 in range(0, n, SEG)]
    out = {"domain": domain, "decay": decay, "base_nll_test": base_nll,
           "g_only": {"dnll_test": float(g_test), **r["G"],
                      "segments": segs["g_only"]},
           "router": {"dnll_test": float(r_test), **r["S"],
                      "segments": segs["router"]},
           "paired_router_vs_gonly": pc}
    print(f"{domain}{'_decay' if decay else ''}: G-only {g_test:+.4f} | "
          f"ROUTER {r_test:+.4f} | paired {pc['mean']:+.4f} CI {pc['ci95']} "
          f"P={pc['p_better']:.3f}", flush=True)
    suffix = "_decay" if decay else ""
    with open(f"results/router500k_{domain}{suffix}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
