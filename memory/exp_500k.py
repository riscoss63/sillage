"""500k-token streams: does the fixed 4.2 MB Hebbian memory hold up against
an unbounded kNN store that grows to ~770 MB?

Protocol changes vs the 40k experiments (declared, applied to ALL methods):
  - scoring on every 4th position (>= 64) only; writes happen at EVERY
    position. 125k scored positions per stream.
  - BHD uses the model-surprise gate (the system gate needs a full-vocab
    normalization at every write; at this scale we score the vocabulary only
    at eval positions). Its contribution was +0.036 of +0.478 at 40k.
  - RAG-lite omitted (outclassed by an order of magnitude at 40k).

Methods: base | unigram cache | exact 4-gram dict | kNN-LM (unbounded) |
BHD v3 amp (no decay) | BHD v3 amp with 100k-token half-life decay |
(bible only) BHD count control.

Outputs: results/exp500k_<domain>.json with test dNLL + CI and per-50k-token
segment gain curves.

Usage: python exp_500k.py <tolstoy|bible>
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
import os
import sys
import time

import numpy as np

from memories import BETAS, CAP, LAMS, THRESH_Q, knn_p_true
from sillage_factorial import D_G, D_V

EVAL_STRIDE = 4
EVAL_START = 64
SEG = 50_000
KNN_K = 32
DECAY_HALF_LIFE = 100_000
DECAY_EVERY = 64


def eval_positions(n):
    pos = np.arange(EVAL_START, n)
    return pos[pos % EVAL_STRIDE == 2]


# --------------------------------------------------------------- BHD pass ---

def bhd_500k_pass(ids, vals, LP, epos, value="amp", decay=False,
                  vocab=50257, hv_seed=7001):
    rngV = np.random.default_rng(hv_seed)
    rngT = np.random.default_rng(hv_seed + 1)
    V = ((rngV.integers(0, 2, size=(vocab, D_V)) * 2.0 - 1.0)
         / np.sqrt(D_V)).astype(np.float32)
    T = (rngT.integers(0, 2, size=(vocab, D_G), dtype=np.int8) * 2 - 1)
    M = np.zeros((D_G, D_V), dtype=np.float32)
    n = len(vals)
    is_eval = np.zeros(n, dtype=bool)
    is_eval[epos] = True
    ne = len(epos)
    s_true = np.zeros(ne, dtype=np.float32)
    smax = np.zeros(ne, dtype=np.float32)
    lse = np.zeros((ne, len(BETAS)), dtype=np.float32)
    sumsq = np.zeros(ne, dtype=np.float32)
    betas = np.array(BETAS, dtype=np.float32)
    g_raw = np.ones(D_G, dtype=np.float32)
    inv = 1.0 / np.sqrt(D_G)
    gamma = 0.5 ** (DECAY_EVERY / DECAY_HALF_LIFE)
    ei = 0
    for j in range(n):
        g_raw = np.roll(g_raw, 1)
        g_raw *= T[ids[j]]
        if j >= 4:
            g_raw *= np.roll(T[ids[j - 4]], 4)
        q = g_raw * inv
        u = M.T @ q
        st_raw = float(u @ V[vals[j]])
        if is_eval[j]:
            un = float(np.linalg.norm(u)) + 1e-8
            s = (V @ u) / un
            s_true[ei] = s[vals[j]]
            smax[ei] = s.max()
            m = betas[:, None] * s[None, :]
            mx = m.max(axis=1)
            lse[ei] = mx + np.log(np.exp(m - mx[:, None]).sum(axis=1))
            r = np.maximum(s, 0.0)
            sumsq[ei] = float((r * r).sum())
            ei += 1
        gj = min(CAP, max(0.0, -float(LP[j])))
        if value == "count":
            coef = gj
        else:
            a = max(0.0, st_raw)
            coef = np.sqrt(a * a + gj) - a
        if decay and (j % DECAY_EVERY == 0) and j > 0:
            M *= gamma
        M += coef * q[:, None] * V[vals[j]][None, :]
    return s_true, smax, lse, sumsq


def p_true_of(cfg, s_true, lse, sumsq):
    if cfg[0] == "quad":
        return np.where(sumsq > 0,
                        np.maximum(s_true, 0.0) ** 2
                        / np.maximum(sumsq, 1e-12), 0.0)
    return np.exp(BETAS[cfg[1]] * s_true - lse[:, cfg[1]])


# ------------------------------------------------------------ kNN at 500k ---

def knn_500k(H16, vals, epos, block=256, k=KNN_K):
    n = len(vals)
    K = np.empty((n, 768), dtype=np.float32)
    step = 50_000
    for a in range(0, n, step):
        K[a:a + step] = H16[a:a + step].astype(np.float32)
    K /= np.linalg.norm(K, axis=1, keepdims=True) + 1e-8
    ne = len(epos)
    nd = np.full((ne, k), np.inf, dtype=np.float32)
    nv = np.full((ne, k), -1, dtype=np.int64)
    t0 = time.time()
    for b0 in range(0, ne, block):
        b1 = min(b0 + block, ne)
        hi = int(epos[b1 - 1])
        sims = K[epos[b0:b1]] @ K[:hi].T
        for m in range(b0, b1):
            j = int(epos[m])
            row = sims[m - b0, :j]
            kk = min(k, j)
            idx = np.argpartition(-row, kk - 1)[:kk]
            d = np.sqrt(np.maximum(2.0 - 2.0 * row[idx], 0.0))
            nd[m, :kk] = d
            nv[m, :kk] = vals[idx]
        if b0 % (block * 40) == 0:
            print(f"  knn {b0}/{ne} ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
    return nd, nv


# ------------------------------------------------- generic tuning on evals ---

def tune(p_mem_conf_fn, grid, lp_e, dev_e, test_e):
    p_base = np.exp(lp_e)
    best = None
    for cfg in grid:
        p_mem, conf = p_mem_conf_fn(cfg)
        for tq in THRESH_Q:
            if tq is None:
                mask = np.ones(len(lp_e), dtype=bool)
            else:
                fin = np.isfinite(conf[dev_e])
                if not fin.any():
                    continue
                thr = np.quantile(conf[dev_e][fin], tq)
                mask = conf >= thr
            for lam in LAMS:
                p = np.where(mask, lam * p_mem + (1 - lam) * p_base, p_base)
                nll_dev = float(-np.log(np.maximum(p[dev_e], 1e-30)).mean())
                if best is None or nll_dev < best["nll_dev"]:
                    best = {"cfg": str(cfg), "lam": lam, "thresh_q": tq,
                            "nll_dev": nll_dev,
                            "nll_test": float(-np.log(
                                np.maximum(p[test_e], 1e-30)).mean()),
                            "p": p}
    return best


def ci_and_segments(p, lp_e, epos, test_e, n_total):
    d = (-lp_e) - (-np.log(np.maximum(p, 1e-30)))
    dt = d[test_e]
    nb = len(dt) // 512
    blocks = [dt[i * 512:(i + 1) * 512] for i in range(nb)]
    rng = np.random.default_rng(3)
    means = [np.concatenate([blocks[i] for i in rng.integers(0, nb, nb)]).mean()
             for _ in range(1000)]
    segs = []
    for s0 in range(0, n_total, SEG):
        m = (epos >= s0) & (epos < s0 + SEG)
        segs.append(float(d[m].mean()) if m.any() else None)
    return ([float(np.quantile(means, 0.025)),
             float(np.quantile(means, 0.975))], segs)


def main(domain):
    ids = np.load(f"data/{domain}_ids.npy")
    H16 = np.load(f"dumps/{domain}_h.npy", mmap_mode="r")
    LP = np.load(f"dumps/{domain}_lp.npy")
    vals = ids[1:].astype(np.int64)
    n = len(vals)
    epos = eval_positions(n)
    lp_e = LP[epos]
    vals_e = vals[epos]
    dev_e = epos < int(0.2 * n)
    test_e = ~dev_e
    out = {"domain": domain, "n_tokens": int(len(ids)),
           "n_eval": int(len(epos)),
           "base_nll_test": float(-lp_e[test_e].mean()),
           "base_ppl_test": float(np.exp(-lp_e[test_e].mean()))}
    print(f"{domain}: N={len(ids)} evals={len(epos)} "
          f"base NLL {out['base_nll_test']:.4f}", flush=True)

    def report(name, best, extra=None):
        p = best.pop("p")
        gain = out["base_nll_test"] - best["nll_test"]
        ci, segs = ci_and_segments(p, lp_e, epos, test_e, n)
        out[name] = {**best, "dnll_test": float(gain), "dnll_ci95": ci,
                     "segments_dnll": segs, **(extra or {})}
        print(f"  {name}: dNLL {gain:+.4f} CI {ci} cfg={best['cfg']} "
              f"lam={best['lam']} thr={best['thresh_q']}", flush=True)

    # unigram cache
    counts = np.zeros(50257)
    p_uni = np.zeros(n, dtype=np.float32)
    tot = 0
    for j in range(n):
        counts[ids[j]] += 1.0
        tot += 1
        p_uni[j] = counts[vals[j]] / tot
    pu = p_uni[epos]
    report("cache_unigram", tune(lambda c: (pu, pu), [("uni",)],
                                 lp_e, dev_e, test_e),
           {"bytes": 50257 * 4})

    # exact 4-gram dict
    table = {}
    p_ng = np.zeros(n, dtype=np.float32)
    conf_ng = np.zeros(n, dtype=np.float32)
    n_pairs = 0
    pairs_at = {}
    for j in range(n):
        if j >= 3:
            gram = ids[j - 3: j + 1].tobytes()
            slot = table.get(gram)
            if slot:
                t = sum(slot.values())
                p_ng[j] = slot.get(int(vals[j]), 0) / t
                conf_ng[j] = t
            else:
                slot = {}
                table[gram] = slot
            if int(vals[j]) not in slot:
                n_pairs += 1
            slot[int(vals[j])] = slot.get(int(vals[j]), 0) + 1
        if (j + 1) % SEG == 0:
            pairs_at[j + 1] = n_pairs
    png, cng = p_ng[epos], conf_ng[epos]
    report("ngram_dict", tune(lambda c: (png, cng), [("dict4",)],
                              lp_e, dev_e, test_e),
           {"pairs_final": int(n_pairs), "bytes_ideal": int(n_pairs * 16),
            "pairs_growth": pairs_at})

    # BHD variants
    grid_b = [("sm", i) for i in range(len(BETAS))] + [("quad", -1)]
    variants = [("bhd_amp", "amp", False), ("bhd_amp_decay", "amp", True)]
    if domain == "bible":
        variants.append(("bhd_count", "count", False))
    for name, value, decay in variants:
        t0 = time.time()
        s_true, smax, lse, sumsq = bhd_500k_pass(ids, vals, LP, epos,
                                                 value=value, decay=decay)
        best = tune(lambda c: (p_true_of(c, s_true, lse, sumsq), smax),
                    grid_b, lp_e, dev_e, test_e)
        report(name, best, {"bytes": D_G * D_V * 4,
                            "minutes": round((time.time() - t0) / 60, 1)})

    # unbounded kNN
    t0 = time.time()
    nd, nv = knn_500k(H16, vals, epos)
    grid_k = [(k, tau) for k in (8, 16, 32) for tau in (0.5, 1.0, 3.0)]
    best = tune(lambda c: knn_p_true(nd, nv, vals_e, c[0], c[1]),
                grid_k, lp_e, dev_e, test_e)
    report("knn", best, {"bytes_final": int(n * (768 * 2 + 4)),
                         "minutes": round((time.time() - t0) / 60, 1)})

    os.makedirs("results", exist_ok=True)
    with open(f"results/exp500k_{domain}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
