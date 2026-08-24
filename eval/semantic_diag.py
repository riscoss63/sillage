"""Step-1 diagnostics for gradient-free SEMANTIC keys (measure before build).

Q1  Does whitening fix the hidden-state similarity geometry?
    Metric: AUC separating same-next-token pairs from random pairs by
    cosine, raw-centered vs ZCA-whitened (fit on the dev 20%), plus the
    random-pair p95 tail that killed the linear readout.

Q2  Do banded SimHash codes on whitened states give small, precise,
    causal candidate sets?  (LSH bands = the discrete analog of top-k, and
    each band pattern can become a VSA symbol -> Hebbian-compatible key.)
    Metrics per b (bits/band, L=32 bands): mean candidate-set size,
    precision P(candidate shares the next token), lift over base rate,
    coverage of the true top-32 whitened-cosine neighbors, and
    P(same next token | >= m bands match).

Reference ceiling: precision of the true top-32 neighbors (what kNN uses).

Usage: python semantic_diag.py <prefix:gpt2|qwen> <domain> [max_positions]
Output: results/semantic_diag_<prefix><domain>.json
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
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

L_BANDS = 32
B_LIST = [8, 12, 16]
N_PAIRS = 10_000
N_QUERIES = 2_000
MIN_GAP = 16
EPS_EIG = 1e-3


def auc(pos, neg):
    u = stats.mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (len(pos) * len(neg)))


def main(which, domain, cap=100_000):
    prefix = "" if which == "gpt2" else "q_"
    ids = np.load(f"data/{prefix}{domain}_ids.npy")
    H = np.load(f"dumps/{prefix}{domain}_h.npy", mmap_mode="r")
    n = min(len(H), cap)
    H = np.array(H[:n], dtype=np.float32)
    vals = ids[1:n + 1].astype(np.int64)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
    n_dev = int(0.2 * n)
    rng = np.random.default_rng(0)
    out = {"model": which, "domain": domain, "n": int(n)}

    # ---- whitening fitted on dev only (causal) ------------------------------
    mu = H[:n_dev].mean(0)
    C = H[:n_dev] - mu
    cov = (C.T @ C) / len(C)
    w, U = np.linalg.eigh(cov)
    W_zca = U @ np.diag(1.0 / np.sqrt(w + EPS_EIG)) @ U.T
    Z = (H - mu) @ W_zca
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    Craw = H - H[:n_dev].mean(0)
    Craw /= np.linalg.norm(Craw, axis=1, keepdims=True) + 1e-8

    # ---- pair geometry ------------------------------------------------------
    lo = n_dev
    i = rng.integers(lo, n, N_PAIRS)
    j = rng.integers(lo, n, N_PAIRS)
    ok = np.abs(i - j) >= MIN_GAP
    i, j = i[ok], j[ok]
    by_tok = defaultdict(list)
    for p in range(lo, n):
        by_tok[int(vals[p])].append(p)
    pp = []
    for tok_id, lst in by_tok.items():
        if len(lst) >= 2:
            arr = np.array(lst)
            for _ in range(min(4, len(lst) - 1)):
                a, b = rng.choice(arr, 2, replace=False)
                if abs(int(a) - int(b)) >= MIN_GAP:
                    pp.append((a, b))
        if len(pp) >= N_PAIRS:
            break
    pp = np.array(pp)
    for name, X in [("raw_centered", Craw), ("whitened", Z)]:
        neg = np.sum(X[i] * X[j], 1)
        pos = np.sum(X[pp[:, 0]] * X[pp[:, 1]], 1)
        out[name] = {
            "random_mean": float(neg.mean()),
            "random_p95": float(np.quantile(neg, 0.95)),
            "same_next_mean": float(pos.mean()),
            "auc_same_vs_random": auc(pos, neg),
        }
        print(f"{name}: random mean {neg.mean():+.3f} p95 "
              f"{np.quantile(neg, 0.95):+.3f} | same-next mean "
              f"{pos.mean():+.3f} | AUC {out[name]['auc_same_vs_random']:.3f}",
              flush=True)

    # ---- reference: true top-32 neighbors (whitened cosine) -----------------
    queries = np.sort(rng.choice(np.arange(max(lo, 512), n), N_QUERIES,
                                 replace=False))
    prec_top, top_sets = [], []
    for b0 in range(0, len(queries), 256):
        b1 = min(b0 + 256, len(queries))
        hi = int(queries[b1 - 1])
        sims = Z[queries[b0:b1]] @ Z[:hi].T
        for m in range(b0, b1):
            qj = int(queries[m])
            row = sims[m - b0, :qj]
            idx = np.argpartition(-row, 32)[:32]
            top_sets.append(set(int(t) for t in idx))
            prec_top.append(float((vals[idx] == vals[qj]).mean()))
    base_rate = float(np.mean([
        (vals[:int(qj)] == vals[int(qj)]).mean() for qj in queries[:500]]))
    out["top32_reference"] = {"precision": float(np.mean(prec_top)),
                              "base_rate": base_rate}
    print(f"top-32 whitened-cos neighbors: precision "
          f"{np.mean(prec_top):.3f} (base rate {base_rate:.4f}, lift "
          f"{np.mean(prec_top) / max(base_rate, 1e-9):.1f}x)", flush=True)

    # ---- banded SimHash on whitened states ----------------------------------
    Wh = rng.normal(size=(Z.shape[1], L_BANDS * max(B_LIST))).astype(np.float32)
    bits = (Z @ Wh) > 0
    qset = set(int(q) for q in queries)
    for b in B_LIST:
        tables = [dict() for _ in range(L_BANDS)]
        sizes, precs, covers, mmatch = [], [], [], []
        same_by_m = defaultdict(list)
        for p in range(n):
            if p in qset and p >= lo:
                cand = defaultdict(int)
                for k in range(L_BANDS):
                    key = np.packbits(
                        bits[p, k * 16:k * 16 + b]).tobytes()
                    for c in tables[k].get(key, ()):
                        cand[c] += 1
                if cand:
                    cl = np.array(list(cand.keys()))
                    cm = np.array(list(cand.values()))
                    same = vals[cl] == vals[p]
                    sizes.append(len(cl))
                    precs.append(float(same.mean()))
                    covers.append(len(set(cl.tolist())
                                      & top_sets[len(sizes) - 1]) / 32
                                  if len(sizes) <= len(top_sets) else 0)
                    for m in (1, 2, 4, 8):
                        sel = cm >= m
                        if sel.any():
                            same_by_m[m].append(float(same[sel].mean()))
                    mmatch.append(float(cm.mean()))
                else:
                    sizes.append(0)
            for k in range(L_BANDS):
                key = np.packbits(bits[p, k * 16:k * 16 + b]).tobytes()
                tables[k].setdefault(key, []).append(p)
        row = {
            "mean_candidates": float(np.mean(sizes)),
            "precision": float(np.mean(precs)) if precs else 0.0,
            "lift_vs_base": float(np.mean(precs) / max(base_rate, 1e-9))
            if precs else 0.0,
            "top32_coverage": float(np.mean(covers)) if covers else 0.0,
            "precision_by_min_bands": {m: float(np.mean(v))
                                       for m, v in same_by_m.items()},
        }
        out[f"simhash_b{b}"] = row
        print(f"SimHash b={b} (L={L_BANDS}): candidates "
              f"{row['mean_candidates']:.0f} | precision {row['precision']:.3f}"
              f" (lift {row['lift_vs_base']:.1f}x) | top32 coverage "
              f"{row['top32_coverage']:.2f} | P(same|>=m bands) "
              f"{row['precision_by_min_bands']}", flush=True)

    with open(f"results/semantic_diag_{prefix}{domain}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
    main(sys.argv[1], sys.argv[2], cap)
