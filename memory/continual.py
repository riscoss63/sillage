"""Continual stream A -> B -> A (relativity 15k | alice 15k | relativity 15k).

Question: does a memory built on domain A survive the interlude B, and what
does forgetting (decay) trade for adaptivity?

Methods (hyperparameters tuned on segment A1 only, then frozen):
  base            frozen GPT-2
  knn             unbounded kNN-LM
  knn_cap         byte-capped kNN-LM (BHD budget, surprise-kept)
  bhd             sparse BHD memory, no decay
  bhd_decay       sparse BHD memory, per-token decay with 8k-token half-life

Reported: NLL per segment (A1 dev, B, A2), relative to base.
Output: results/continual.json
"""


# --- repo bootstrap (added by reorganize.py) ---
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
import os

import numpy as np

from memories import (BETAS, BHD_BYTES, CAP, LAMS, THRESH_Q, W_SPARSE,
                      bhd_p_true, bhd_pass, knn_capped_neighbors,
                      knn_neighbors, knn_p_true, load_domain,
                      unigram_cache_p_true)

SEG = 15_000
HALF_LIFE = 8_000


def tune_then_segments(p_mem_fn, grid, LP, seg_masks):
    p_base = np.exp(LP)
    a1 = seg_masks["A1"]
    best = None
    for cfg in grid:
        p_mem, conf = p_mem_fn(cfg)
        for tq in THRESH_Q:
            if tq is None:
                mask = np.ones(len(LP), dtype=bool)
            else:
                fin = np.isfinite(conf[a1])
                if not fin.any():
                    continue
                thr = np.quantile(conf[a1][fin], tq)
                mask = conf >= thr
            for lam in LAMS:
                p = np.where(mask, lam * p_mem + (1 - lam) * p_base, p_base)
                nll_dev = float(-np.log(np.maximum(p[a1], 1e-30)).mean())
                if best is None or nll_dev < best["nll_A1"]:
                    best = {"cfg": str(cfg), "lam": lam, "thresh_q": tq,
                            "nll_A1": nll_dev}
                    for name, m in seg_masks.items():
                        best[f"nll_{name}"] = float(
                            -np.log(np.maximum(p[m], 1e-30)).mean())
    return best


def main():
    ids, H, LP, vals = load_domain("aba")
    n = len(H)
    seg_of_pred = np.minimum((np.arange(n) + 1) // SEG, 2)  # segment of x[j+1]
    seg_masks = {"A1": seg_of_pred == 0, "B": seg_of_pred == 1,
                 "A2": seg_of_pred == 2}
    g = np.clip(-LP, 0.0, CAP)
    out = {"segments": {k: float(-LP[m].mean())
                        for k, m in seg_masks.items()}}
    print("base NLL per segment:", out["segments"], flush=True)

    grid_knn = [(k, tau) for k in (8, 16, 32) for tau in (0.5, 1.0, 3.0)]

    p_uni = unigram_cache_p_true(ids, vals)
    out["cache_unigram"] = tune_then_segments(
        lambda c: (p_uni, p_uni), [("uni",)], LP, seg_masks)

    nd, nv = knn_neighbors(H, vals)
    out["knn"] = tune_then_segments(
        lambda c: knn_p_true(nd, nv, vals, c[0], c[1]), grid_knn, LP, seg_masks)

    cap_entries = BHD_BYTES // (768 * 2 + 4 + 4)
    nd2, nv2 = knn_capped_neighbors(H, vals, g, cap_entries)
    out["knn_cap"] = tune_then_segments(
        lambda c: knn_p_true(nd2, nv2, vals, c[0], c[1]), grid_knn, LP,
        seg_masks)

    grid_bhd = [(i,) for i in range(len(BETAS))]
    for name, decay in [("bhd", 1.0),
                        ("bhd_decay", 0.5 ** (1.0 / HALF_LIFE))]:
        s_true, smax, lse = bhd_pass(H, vals, g, sparse_w=W_SPARSE,
                                     decay_per_token=decay)
        out[name] = tune_then_segments(
            lambda c: (bhd_p_true(s_true, lse, c[0], BETAS[c[0]]), smax),
            grid_bhd, LP, seg_masks)

    for m in ["cache_unigram", "knn", "knn_cap", "bhd", "bhd_decay"]:
        r = out[m]
        print(f"{m}: A1 {r['nll_A1']:.4f} | B {r['nll_B']:.4f} | "
              f"A2 {r['nll_A2']:.4f} (cfg={r['cfg']} lam={r['lam']} "
              f"thr={r['thresh_q']})", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/continual.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
