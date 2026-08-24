"""Sillage-Sem: gradient-free SEMANTIC keys via whitened banded SimHash,
hybridized with the n-gram key. Built only after the diagnostics of
semantic_diag.py validated the design (whitening repairs the geometry;
band-match count is a graded precision kernel reaching P(same next | >=8
bands) ~ 0.6, above the top-32 kNN ceiling).

Key = concat( aG * q_ngram , aS * q_sem ),  aG^2 + aS^2 = 1
  q_ngram : sliding 4-gram binding (as in Sillage)
  q_sem   : L=32 bands x b=16 bits of SimHash over ZCA-whitened hidden
            states; each band's observed pattern maps (via a seeded hash) to
            a +/-1 hypervector on that band's 128-dim slot, so
            <q_sem_i, q_sem_j> = (#matching bands)/L + O(1/sqrt(128)).
            Whitening (mu, W_zca) is fitted causally on the first
            WHITEN_FIT tokens and then FROZEN (no pattern drift).

Values/gates: amplitude writes, model-surprise gate (the established Sillage
recipe). Everything else (tuning grids, splits, CIs) identical to the paper
protocol.

Usage: python sillage_semantic.py <tune|final> <gpt2|qwen> <domain> [cap]
  tune : sweep the (aG^2, aS^2) mix on the domain's dev split
  final: run one mix (env-set below after tuning) + paired test vs the
         n-gram-only control
Output: results/sillage_sem_<mode>_<prefix><domain>.json
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

from memories import (BETAS, CAP, LAMS, THRESH_Q, block_bootstrap_dnll,
                      splits)

D_G, D_V, NGRAM = 4096, 256, 4
L_BANDS, B_BITS, D_BAND = 32, 16, 128          # D_sem = 32*128 = 4096
WHITEN_FIT = 4096
EPS_EIG = 1e-3
MIXES = [(1.0, 0.0), (0.0, 1.0), (0.67, 0.33), (0.5, 0.5), (0.33, 0.67)]
FINAL_MIX = (0.5, 0.5)      # selected on the alice dev split (tune mode)


def band_vector_cache():
    cache = {}

    def get(band, pattern):
        key = (band, pattern)
        v = cache.get(key)
        if v is None:
            seed = (0x9E3779B97F4A7C15 * (band * 65537 + pattern + 1)) % 2 ** 64
            rng = np.random.default_rng(seed)
            v = (rng.integers(0, 2, size=D_BAND) * 2.0 - 1.0).astype(np.float32)
            cache[key] = v
        return v
    return get


def sem_pass(ids, H, LP, vals, aG2, aS2, vocab, hv_seed=7001):
    rngV = np.random.default_rng(hv_seed)
    rngT = np.random.default_rng(hv_seed + 1)
    rngW = np.random.default_rng(hv_seed + 2)
    V = ((rngV.integers(0, 2, size=(vocab, D_V)) * 2.0 - 1.0)
         / np.sqrt(D_V)).astype(np.float32)
    T = (rngT.integers(0, 2, size=(vocab, D_G), dtype=np.int8) * 2 - 1)
    Wh = rngW.normal(size=(H.shape[1], L_BANDS * B_BITS)).astype(np.float32)
    getv = band_vector_cache()

    D_S = L_BANDS * D_BAND
    M = np.zeros((D_G + D_S, D_V), dtype=np.float32)
    aG, aS = np.sqrt(aG2), np.sqrt(aS2)
    n = len(vals)
    s_true = np.zeros(n, dtype=np.float32)
    smax = np.zeros(n, dtype=np.float32)
    lse = np.zeros((n, len(BETAS)), dtype=np.float32)
    sumsq = np.zeros(n, dtype=np.float32)
    betas = np.array(BETAS, dtype=np.float32)
    g_raw = np.ones(D_G, dtype=np.float32)
    invG = 1.0 / np.sqrt(D_G)
    mu, W_zca = None, None
    pw2 = 2 ** np.arange(B_BITS)

    for j in range(n):
        # --- n-gram block
        g_raw = np.roll(g_raw, 1)
        g_raw *= T[ids[j]]
        if j >= NGRAM:
            g_raw *= np.roll(T[ids[j - NGRAM]], NGRAM)
        q = np.zeros(D_G + D_S, dtype=np.float32)
        q[:D_G] = aG * g_raw * invG
        # --- semantic block (after the frozen whitening fit)
        if j == WHITEN_FIT:
            X = H[:j].astype(np.float32)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            mu = X.mean(0)
            C = X - mu
            cov = (C.T @ C) / len(C)
            w, U = np.linalg.eigh(cov)
            W_zca = (U @ np.diag(1.0 / np.sqrt(w + EPS_EIG)) @ U.T
                     ).astype(np.float32)
        if W_zca is not None and aS2 > 0:
            h = H[j].astype(np.float32)
            h /= np.linalg.norm(h) + 1e-8
            z = (h - mu) @ W_zca
            bits = ((z @ Wh) > 0).reshape(L_BANDS, B_BITS)
            scale = aS / np.sqrt(L_BANDS * D_BAND)
            for k in range(L_BANDS):
                pat = int(bits[k] @ pw2)
                q[D_G + k * D_BAND: D_G + (k + 1) * D_BAND] = \
                    scale * getv(k, pat)
        nq = np.linalg.norm(q) + 1e-8
        q /= nq
        # --- read / score / write (the Sillage recipe)
        u = M.T @ q
        un = float(np.linalg.norm(u)) + 1e-8
        s = (V @ u) / un
        st = float(s[vals[j]])
        s_true[j] = st
        smax[j] = s.max()
        m = betas[:, None] * s[None, :]
        mx = m.max(axis=1)
        lse[j] = mx + np.log(np.exp(m - mx[:, None]).sum(axis=1))
        r = np.maximum(s, 0.0)
        sumsq[j] = float((r * r).sum())
        g = min(CAP, max(0.0, -float(LP[j])))
        a = max(0.0, st * un)
        M += (np.sqrt(a * a + g) - a) * q[:, None] * V[vals[j]][None, :]
    return s_true, smax, lse, sumsq


def p_true_of(cfg, s_true, lse, sumsq):
    if cfg[0] == "quad":
        return np.where(sumsq > 0, np.maximum(s_true, 0.0) ** 2
                        / np.maximum(sumsq, 1e-12), 0.0)
    return np.exp(BETAS[cfg[1]] * s_true - lse[:, cfg[1]])


def tune_eval(s_true, smax, lse, sumsq, LP, dev, test):
    p_base = np.exp(LP)
    grid = [("sm", i) for i in range(len(BETAS))] + [("quad", -1)]
    best = None
    for cfg in grid:
        p_mem = p_true_of(cfg, s_true, lse, sumsq)
        for tq in THRESH_Q:
            mask = (np.ones(len(LP), bool) if tq is None
                    else smax >= np.quantile(smax[dev], tq))
            for lam in LAMS:
                p = np.where(mask, lam * p_mem + (1 - lam) * p_base, p_base)
                nd = float(-np.log(np.maximum(p[dev], 1e-30)).mean())
                if best is None or nd < best["nll_dev"]:
                    best = {"cfg": str(cfg), "lam": lam, "thresh_q": tq,
                            "nll_dev": nd,
                            "nll_test": float(-np.log(
                                np.maximum(p[test], 1e-30)).mean()),
                            "p_test": p[test].copy()}
    return best


def load(which, domain, cap):
    prefix = "" if which == "gpt2" else "q_"
    vocab = 50257 if which == "gpt2" else 151_936
    ids = np.load(f"data/{prefix}{domain}_ids.npy")
    H = np.load(f"dumps/{prefix}{domain}_h.npy", mmap_mode="r")
    LP = np.load(f"dumps/{prefix}{domain}_lp.npy")
    n = min(len(LP), cap)
    return prefix, vocab, ids[:n + 1], H, LP[:n], ids[1:n + 1].astype(np.int64)


def main():
    mode, which, domain = sys.argv[1], sys.argv[2], sys.argv[3]
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    prefix, vocab, ids, H, LP, vals = load(which, domain, cap)
    n = len(vals)
    dev, test = splits(n)
    base_nll = float(-LP[test].mean())
    out = {"mode": mode, "model": which, "domain": domain, "n": int(n),
           "base_nll_test": base_nll}
    mixes = MIXES if mode == "tune" else [FINAL_MIX, (1.0, 0.0)]
    p_by_mix = {}
    for aG2, aS2 in mixes:
        s_true, smax, lse, sumsq = sem_pass(ids, H, LP, vals, aG2, aS2, vocab)
        best = tune_eval(s_true, smax, lse, sumsq, LP, dev, test)
        p_by_mix[(aG2, aS2)] = best.pop("p_test")
        gain = base_nll - best["nll_test"]
        out[f"aG2={aG2},aS2={aS2}"] = {**best, "dnll_test": float(gain)}
        print(f"{prefix}{domain} aG2={aG2} aS2={aS2}: dev {best['nll_dev']:.4f}"
              f" | test dNLL {gain:+.4f} (cfg={best['cfg']} lam={best['lam']}"
              f" thr={best['thresh_q']})", flush=True)
    if mode == "final" and len(mixes) == 2:
        d = (-np.log(np.maximum(p_by_mix[(1.0, 0.0)], 1e-30))
             + np.log(np.maximum(p_by_mix[FINAL_MIX], 1e-30)))
        nb = len(d) // 512
        blocks = [d[i * 512:(i + 1) * 512] for i in range(nb)]
        rng = np.random.default_rng(3)
        means = [np.concatenate([blocks[i] for i in
                                 rng.integers(0, nb, nb)]).mean()
                 for _ in range(1000)]
        out["paired_hybrid_vs_ngram"] = {
            "mean": float(d.mean()),
            "ci95": [float(np.quantile(means, 0.025)),
                     float(np.quantile(means, 0.975))],
            "p_better": float((np.array(means) > 0).mean())}
        print(f"paired hybrid vs n-gram-only: {d.mean():+.4f} "
              f"CI {out['paired_hybrid_vs_ngram']['ci95']} "
              f"P={out['paired_hybrid_vs_ngram']['p_better']:.3f}", flush=True)
    with open(f"results/sillage_sem_{mode}_{prefix}{domain}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
