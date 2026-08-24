"""Sillage-Router: a mixture of two full-strength associative memories.

The concatenated hybrid key failed the no-regression check on Manuscripts
(-0.097 nats, P=0.000): diluting the n-gram block's key weight costs exactly
where exact repetition dominates, and the optimal mix is non-stationary
within a stream. The fix is to arbitrate at the SCORE level:

  two matrices, each with its own undiluted key --
    M_G : n-gram keys (the Sillage of paper 1)
    M_S : banded-SimHash semantic keys over (whitened) hidden states,
          nested prefix resolutions b in {8,12,16} ("multi")
  per-position mixture with per-block confidence abstention:
    p = 1[G active] lam_G p_G + 1[S active] lam_S p_S + (rest) p_LM

Tuning is greedy-coordinate on dev (G exactly as in paper 1, then S with G
frozen), reported with a paired test vs G-only on identical positions.

Usage: python sillage_router.py <gpt2|qwen> <domain> [cap] [multi] [nowhiten]
  multi    : semantic bands at nested resolutions b in {8,12,16}
  nowhiten : center-normalize only (for modern models whose raw hidden-state
             geometry is already well conditioned, cf. semantic_diag)
Output: results/sillage_router_<prefix><domain>[_multi][_nw].json
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

import numpy as np

from memories import BETAS, CAP, LAMS, THRESH_Q, splits
from sillage_semantic import (B_BITS, D_BAND, D_G, D_V, EPS_EIG, L_BANDS,
                           NGRAM, WHITEN_FIT, band_vector_cache, load)

LAM_S_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
THR_S_GRID = [0.5, 0.75, 0.9]
B_LIST_SINGLE = [16]
B_LIST_MULTI = [8, 12, 16]


def router_pass(ids, H, LP, vals, vocab, hv_seed=7001,
                b_list=tuple(B_LIST_SINGLE), whiten=True):
    rngV = np.random.default_rng(hv_seed)
    rngT = np.random.default_rng(hv_seed + 1)
    rngW = np.random.default_rng(hv_seed + 2)
    V = ((rngV.integers(0, 2, size=(vocab, D_V)) * 2.0 - 1.0)
         / np.sqrt(D_V)).astype(np.float32)
    T = (rngT.integers(0, 2, size=(vocab, D_G), dtype=np.int8) * 2 - 1)
    Wh = rngW.normal(size=(H.shape[1], L_BANDS * B_BITS)).astype(np.float32)
    getv = band_vector_cache()
    n_groups = len(b_list)
    D_S = n_groups * L_BANDS * D_BAND
    M_G = np.zeros((D_G, D_V), dtype=np.float32)
    M_S = np.zeros((D_S, D_V), dtype=np.float32)
    n = len(vals)
    stats = {b: {"s_true": np.zeros(n, np.float32),
                 "smax": np.zeros(n, np.float32),
                 "lse": np.zeros((n, len(BETAS)), np.float32),
                 "sumsq": np.zeros(n, np.float32)} for b in ("G", "S")}
    betas = np.array(BETAS, dtype=np.float32)
    g_raw = np.ones(D_G, dtype=np.float32)
    invG = 1.0 / np.sqrt(D_G)
    mu, W_zca = None, None
    pw2 = 2 ** np.arange(B_BITS)

    def score_and_stats(u, block, j):
        un = float(np.linalg.norm(u)) + 1e-8
        s = (V @ u) / un
        st = float(s[vals[j]])
        d = stats[block]
        d["s_true"][j] = st
        d["smax"][j] = s.max()
        m = betas[:, None] * s[None, :]
        mx = m.max(axis=1)
        d["lse"][j] = mx + np.log(np.exp(m - mx[:, None]).sum(axis=1))
        r = np.maximum(s, 0.0)
        d["sumsq"][j] = float((r * r).sum())
        return st * un

    for j in range(n):
        g_raw = np.roll(g_raw, 1)
        g_raw *= T[ids[j]]
        if j >= NGRAM:
            g_raw *= np.roll(T[ids[j - NGRAM]], NGRAM)
        qG = g_raw * invG
        aG_raw = score_and_stats(M_G.T @ qG, "G", j)

        if j == WHITEN_FIT:
            X = H[:j].astype(np.float32)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            mu = X.mean(0)
            if whiten:
                C = X - mu
                cov = (C.T @ C) / len(C)
                w, U = np.linalg.eigh(cov)
                W_zca = (U @ np.diag(1.0 / np.sqrt(w + EPS_EIG)) @ U.T
                         ).astype(np.float32)
            else:
                W_zca = np.eye(H.shape[1], dtype=np.float32)
        qS = None
        if W_zca is not None:
            h = H[j].astype(np.float32)
            h /= np.linalg.norm(h) + 1e-8
            z = (h - mu) @ W_zca
            bits = ((z @ Wh) > 0).reshape(L_BANDS, B_BITS)
            qS = np.empty(D_S, dtype=np.float32)
            scale = 1.0 / np.sqrt(n_groups * L_BANDS * D_BAND)
            slot = 0
            for gi, b in enumerate(b_list):
                for k in range(L_BANDS):
                    pat = int(bits[k, :b] @ pw2[:b])
                    qS[slot * D_BAND:(slot + 1) * D_BAND] = \
                        scale * getv(gi * L_BANDS + k, pat)
                    slot += 1
            aS_raw = score_and_stats(M_S.T @ qS, "S", j)

        g = min(CAP, max(0.0, -float(LP[j])))
        a = max(0.0, aG_raw)
        M_G += (np.sqrt(a * a + g) - a) * qG[:, None] * V[vals[j]][None, :]
        if qS is not None:
            a = max(0.0, aS_raw)
            M_S += (np.sqrt(a * a + g) - a) * qS[:, None] * V[vals[j]][None, :]
    return stats


def p_of(d, cfg):
    if cfg[0] == "quad":
        return np.where(d["sumsq"] > 0, np.maximum(d["s_true"], 0.0) ** 2
                        / np.maximum(d["sumsq"], 1e-12), 0.0)
    return np.exp(BETAS[cfg[1]] * d["s_true"] - d["lse"][:, cfg[1]])


def greedy_tune(stats, LP, dev, test):
    """Stage 1: tune G as in paper 1. Stage 2: tune S with G frozen.
    Returns per-position mixtures and the chosen parameters."""
    n = len(LP)
    p_base = np.exp(LP)
    grid = [("sm", i) for i in range(len(BETAS))] + [("quad", -1)]
    bestG = None
    for cfg in grid:
        pG = p_of(stats["G"], cfg)
        for tq in THRESH_Q:
            mG = (np.ones(n, bool) if tq is None
                  else stats["G"]["smax"] >= np.quantile(
                      stats["G"]["smax"][dev], tq))
            for lam in LAMS:
                p = np.where(mG, lam * pG + (1 - lam) * p_base, p_base)
                nd = float(-np.log(np.maximum(p[dev], 1e-30)).mean())
                if bestG is None or nd < bestG["nll_dev"]:
                    bestG = {"cfg": cfg, "lam": lam, "tq": tq, "nll_dev": nd,
                             "pG": pG, "mG": mG}
    pG, mG, lamG = bestG["pG"], bestG["mG"], bestG["lam"]
    p_gonly = np.where(mG, lamG * pG + (1 - lamG) * p_base, p_base)

    bestS = None
    smS = stats["S"]["smax"]
    for cfg in grid:
        pS = p_of(stats["S"], cfg)
        for tq in THR_S_GRID:
            pos = smS[dev][smS[dev] > 0]
            if not len(pos):
                continue
            mS = smS >= np.quantile(pos, tq)
            for lamS in LAM_S_GRID:
                both = mG & mS
                onlyG, onlyS = mG & ~mS, ~mG & mS
                p = p_base.copy()
                p = np.where(onlyG, lamG * pG + (1 - lamG) * p_base, p)
                p = np.where(onlyS, lamS * pS + (1 - lamS) * p_base, p)
                p = np.where(both, lamG * pG + lamS * pS
                             + np.maximum(1 - lamG - lamS, 0.0) * p_base, p)
                nd = float(-np.log(np.maximum(p[dev], 1e-30)).mean())
                if bestS is None or nd < bestS["nll_dev"]:
                    bestS = {"cfg": cfg, "lam": lamS, "tq": tq,
                             "nll_dev": nd, "p": p}
    return {"p_gonly": p_gonly, "p_router": bestS["p"],
            "G": {"cfg": str(bestG["cfg"]), "lam": lamG,
                  "thr": bestG["tq"]},
            "S": {"cfg": str(bestS["cfg"]), "lam": bestS["lam"],
                  "thr": bestS["tq"]}}


def paired_ci(p_a, p_b, test):
    """Block bootstrap of NLL(a) - NLL(b) on test (positive = b better)."""
    d = (-np.log(np.maximum(p_a[test], 1e-30))
         + np.log(np.maximum(p_b[test], 1e-30)))
    nb = len(d) // 512
    blocks = [d[i * 512:(i + 1) * 512] for i in range(nb)]
    rng = np.random.default_rng(3)
    means = [np.concatenate([blocks[i] for i in rng.integers(0, nb, nb)]).mean()
             for _ in range(1000)]
    return {"mean": float(d.mean()),
            "ci95": [float(np.quantile(means, 0.025)),
                     float(np.quantile(means, 0.975))],
            "p_better": float((np.array(means) > 0).mean())}


def main():
    which, domain = sys.argv[1], sys.argv[2]
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    multi = "multi" in sys.argv[4:]
    whiten = "nowhiten" not in sys.argv[4:]
    b_list = B_LIST_MULTI if multi else B_LIST_SINGLE
    prefix, vocab, ids, H, LP, vals = load(which, domain, cap)
    n = len(vals)
    dev, test = splits(n)
    base_nll = float(-LP[test].mean())
    stats = router_pass(ids, H, LP, vals, vocab, b_list=tuple(b_list),
                        whiten=whiten)
    r = greedy_tune(stats, LP, dev, test)
    g_test = base_nll - float(-np.log(
        np.maximum(r["p_gonly"][test], 1e-30)).mean())
    r_test = base_nll - float(-np.log(
        np.maximum(r["p_router"][test], 1e-30)).mean())
    pc = paired_ci(r["p_gonly"], r["p_router"], test)
    out = {"model": which, "domain": domain, "n": int(n),
           "b_list": list(b_list), "whiten": whiten,
           "base_nll_test": base_nll,
           "g_only": {"dnll_test": float(g_test), **r["G"]},
           "router": {"dnll_test": float(r_test), **r["S"]},
           "paired_router_vs_gonly": pc}
    print(f"{prefix}{domain}{'_multi' if multi else ''}"
          f"{'' if whiten else '_nw'}: G-only {g_test:+.4f} | ROUTER "
          f"{r_test:+.4f} | paired {pc['mean']:+.4f} CI {pc['ci95']} "
          f"P={pc['p_better']:.3f}", flush=True)
    suffix = ("_multi" if multi else "") + ("" if whiten else "_nw")
    with open(f"results/sillage_router_{prefix}{domain}{suffix}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
