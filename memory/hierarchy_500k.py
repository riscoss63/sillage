"""Step 2 -- the memory hierarchy (H-Sillage): fast Hebbian tiers + a bounded
exact cold store with surprise-gated consolidation.

Tiers, routed per position at the score level (paper-2 principle):
  G : fast n-gram Hebbian matrix (4.2 MB, amplitude writes, decay)
  S : fast semantic matrix (banded SimHash, 12.6 MB, decay)
  C : cold exact 4-gram -> successor-count store, CAPACITY-BOUNDED, whose
      ADMISSION is the consolidation policy under test:
        surprise-mass : admit gram when its accumulated write-gate mass
                        (sum of clip(-ln p_LM) over its occurrences -- the
                        same free signal that gates Hebbian writes) >= theta
        count         : admit when raw occurrence count >= theta
        random-K      : a random subset of K grams, admitted from the start
      The unbounded store is the ceiling; no-C is the control.

One instrumented pass serves every policy x capacity post hoc: at each
scored position we record the fast-tier statistics plus, for the current
gram, its retrospective successor probability of the true token, its
count-so-far and mass-so-far. (The cold store answers with retrospective
counts once admitted; a strictly prospective variant is an engineering
refinement, stated in the paper.)

Routing/tuning: greedy sequential-residual mixing; block order = descending
standalone dev gain; all hyperparameters tuned on dev only.

Usage: python hierarchy_500k.py <domain> [cap]
Output: results/hierarchy_<domain>.json
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
from collections import defaultdict

import numpy as np

from exp_500k import SEG, eval_positions
from memories import BETAS, CAP, LAMS, splits
from sillage_router import B_LIST_MULTI, paired_ci
from sillage_semantic import (B_BITS, D_BAND, D_G, D_V, EPS_EIG, L_BANDS,
                           NGRAM, WHITEN_FIT, band_vector_cache)

DECAY_HALF_LIFE = 100_000
DECAY_EVERY = 64
THR_Q_G = [None, 0.25, 0.5, 0.75]     # matches paper-2 stage-1 grid
THR_Q_S = [0.5, 0.75, 0.9]            # matches paper-2 stage-2 grid
LAM_S_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
LAM_C_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85]
MIN_COUNT_C = [1, 2, 4]          # cold-store extra confidence gate


def hpass(ids, H, LP, vals, epos, vocab=50257, hv_seed=7001):
    """Fast tiers (with decay) + cold-store instrumentation."""
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
    fast = {b: {"s_true": np.zeros(ne, np.float32),
                "smax": np.zeros(ne, np.float32),
                "lse": np.zeros((ne, len(BETAS)), np.float32)}
            for b in ("G", "S")}
    cold = {"p_true": np.zeros(ne, np.float32),
            "count": np.zeros(ne, np.int32),
            "mass": np.zeros(ne, np.float32),
            "gram_id": np.zeros(ne, np.int64)}
    betas = np.array(BETAS, dtype=np.float32)
    g_raw = np.ones(D_G, dtype=np.float32)
    invG = 1.0 / np.sqrt(D_G)
    mu, W_zca = None, None
    pw2 = 2 ** np.arange(B_BITS)
    gamma = 0.5 ** (DECAY_EVERY / DECAY_HALF_LIFE)
    succ = {}                      # gram -> {tok: count}
    tot = defaultdict(int)         # gram -> count
    mass = defaultdict(float)      # gram -> surprise mass
    gram_key_of = {}
    next_gid = [0]
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
        gram = ids[max(0, j - NGRAM + 1): j + 1].tobytes() if j >= NGRAM - 1 \
            else None
        if is_eval[j]:
            for block, u in (("G", uG),) + ((("S", uS),) if uS is not None
                                            else ()):
                un = float(np.linalg.norm(u)) + 1e-8
                s = (V @ u) / un
                d = fast[block]
                d["s_true"][ei] = s[vals[j]]
                d["smax"][ei] = s.max()
                m = betas[:, None] * s[None, :]
                mx = m.max(axis=1)
                d["lse"][ei] = mx + np.log(np.exp(m - mx[:, None]).sum(axis=1))
            if gram is not None and gram in succ:
                sl = succ[gram]
                t = tot[gram]
                cold["p_true"][ei] = sl.get(int(vals[j]), 0) / t
                cold["count"][ei] = t
                cold["mass"][ei] = mass[gram]
                cold["gram_id"][ei] = gram_key_of[gram]
            ei += 1
        gj = min(CAP, max(0.0, -float(LP[j])))
        if gram is not None:
            if gram not in succ:
                succ[gram] = {}
                gram_key_of[gram] = next_gid[0]
                next_gid[0] += 1
            succ[gram][int(vals[j])] = succ[gram].get(int(vals[j]), 0) + 1
            tot[gram] += 1
            mass[gram] += gj
        if j % DECAY_EVERY == 0 and j > 0:
            M_G *= gamma
            M_S *= gamma
        a = max(0.0, float(uG @ V[vals[j]]))
        M_G += (np.sqrt(a * a + gj) - a) * qG[:, None] * V[vals[j]][None, :]
        if uS is not None:
            a = max(0.0, float(uS @ V[vals[j]]))
            M_S += (np.sqrt(a * a + gj) - a) * qS[:, None] * V[vals[j]][None, :]
        if j % 100_000 == 0 and j > 0:
            print(f"  ... {j}/{n} ({(time.time()-t0)/60:.0f} min)", flush=True)
    final_mass = np.array(list(mass.values()), dtype=np.float64)
    final_count = np.array(list(tot.values()), dtype=np.int64)
    mass_of_gid = np.zeros(next_gid[0])
    count_of_gid = np.zeros(next_gid[0], dtype=np.int64)
    for gk, gid in gram_key_of.items():
        mass_of_gid[gid] = mass[gk]
        count_of_gid[gid] = tot[gk]
    n_pairs = sum(len(s) for s in succ.values())
    return fast, cold, final_mass, final_count, mass_of_gid, count_of_gid, \
        n_pairs


def p_fast(d, cfg):
    return np.exp(BETAS[cfg[1]] * d["s_true"] - d["lse"][:, cfg[1]])


def fast_candidates(d, dev, thr_grid, lam_grid):
    """All (p_block, mask, lam, params) candidates for a fast tier."""
    out = []
    arr = d["smax"][dev]
    arr = arr[arr > 0]
    for bi in range(len(BETAS)):
        pm = p_fast(d, ("sm", bi))
        for tq in thr_grid:
            mask = (d["smax"] > -np.inf if tq is None or not len(arr)
                    else d["smax"] >= np.quantile(arr, tq))
            for lam in lam_grid:
                out.append((pm, mask, lam,
                            {"beta_idx": bi, "lam": lam, "thr_q": tq}))
    return out


def cold_candidates(cold, admit_mask):
    out = []
    for mc in MIN_COUNT_C:
        mask = admit_mask & (cold["count"] >= mc)
        for lam in LAM_C_GRID:
            out.append((cold["p_true"], mask, lam,
                        {"min_count": mc, "lam": lam}))
    return out


def greedy(blocks, LP_e, dev, test):
    """Flat-mixture greedy (paper-2 semantics generalized to N blocks):
    p = sum_i [mask_i] lam_i p_i + max(0, 1 - sum_i [mask_i] lam_i) p_base.
    Blocks are added sequentially with earlier lambdas frozen; all block
    orderings are tried and the best dev NLL wins."""
    from itertools import permutations
    p_base = np.exp(LP_e)
    best_run = None
    for order in permutations(blocks):
        acc_p = np.zeros_like(p_base)
        acc_l = np.zeros_like(p_base)
        chosen = {}
        for name, cands in order:
            best_b = None
            for pm, mask, lam, params in cands:
                add_l = np.where(mask, lam, 0.0)
                p = (acc_p + add_l * pm
                     + np.maximum(1.0 - acc_l - add_l, 0.0) * p_base)
                nd = float(-np.log(np.maximum(p[dev], 1e-30)).mean())
                if best_b is None or nd < best_b[0]:
                    best_b = (nd, add_l, pm, params)
            cur = float(-np.log(np.maximum(
                (acc_p + np.maximum(1 - acc_l, 0) * p_base)[dev],
                1e-30)).mean())
            if best_b[0] < cur - 1e-6:
                _, add_l, pm, params = best_b
                acc_p = acc_p + add_l * pm
                acc_l = acc_l + add_l
                chosen[name] = params
        p_mix = acc_p + np.maximum(1.0 - acc_l, 0.0) * p_base
        nd = float(-np.log(np.maximum(p_mix[dev], 1e-30)).mean())
        if best_run is None or nd < best_run[0]:
            nll_t = float(-np.log(np.maximum(p_mix[test], 1e-30)).mean())
            best_run = (nd, p_mix, dict(chosen), nll_t,
                        [n for n, _ in order])
    _, p_mix, chosen, nll_t, order_used = best_run
    chosen["_order"] = order_used
    return p_mix, chosen, nll_t


def main():
    domain = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    ids = np.load(f"data/{domain}_ids.npy")
    H = np.load(f"dumps/{domain}_h.npy", mmap_mode="r")
    LP = np.load(f"dumps/{domain}_lp.npy")
    n = min(len(LP), cap)
    ids = ids[: n + 1]
    LP = LP[:n]
    vals = ids[1:].astype(np.int64)
    epos = eval_positions(n)
    LP_e = LP[epos]
    dev = epos < int(0.2 * n)
    test = ~dev
    base_nll = float(-LP_e[test].mean())
    print(f"{domain}: n={n} evals={len(epos)} base NLL {base_nll:.4f}",
          flush=True)
    cache = f"results/hier_cache_{domain}.npz"
    import os
    if os.path.exists(cache):
        z = np.load(cache)
        fast = {b: {k: z[f"{b}_{k}"] for k in ("s_true", "smax", "lse")}
                for b in ("G", "S")}
        cold = {k: z[f"cold_{k}"] for k in ("p_true", "count", "mass",
                                            "gram_id")}
        mass_gid, count_gid = z["mass_gid"], z["count_gid"]
        fmass, n_pairs = z["mass_gid"], int(z["n_pairs"])
        print("  (pass loaded from cache)", flush=True)
    else:
        fast, cold, fmass, fcount, mass_gid, count_gid, n_pairs = hpass(
            ids, H, LP, vals, epos)
        np.savez_compressed(
            cache, n_pairs=n_pairs, mass_gid=mass_gid, count_gid=count_gid,
            **{f"{b}_{k}": fast[b][k] for b in ("G", "S")
               for k in ("s_true", "smax", "lse")},
            **{f"cold_{k}": cold[k] for k in ("p_true", "count", "mass",
                                              "gram_id")})
    rng = np.random.default_rng(0)
    out = {"domain": domain, "n": int(n), "base_nll_test": base_nll,
           "unbounded_grams": int(len(fmass)), "unbounded_pairs": int(n_pairs)}

    fast_blocks = [("G", fast_candidates(fast["G"], dev, THR_Q_G, LAMS)),
                   ("S", fast_candidates(fast["S"], dev, THR_Q_S,
                                         LAM_S_GRID))]

    def run_config(name, admit_mask, entries):
        blocks = list(fast_blocks)
        if admit_mask is not None:
            blocks.append(("C", cold_candidates(cold, admit_mask)))
        p, chosen, nll_t = greedy(blocks, LP_e, dev, test)
        gain = base_nll - nll_t
        out[name] = {"dnll_test": float(gain), "cold_entries": entries,
                     "params": chosen}
        print(f"  {name}: dNLL {gain:+.4f} (cold entries {entries})",
              flush=True)
        return p

    # control (no cold tier) and unbounded ceiling
    p_ctrl = run_config("no_cold", None, 0)
    p_unb = run_config("cold_unbounded",
                       np.ones(len(epos), dtype=bool), int(len(mass_gid)))
    out["paired_unbounded_vs_nocold"] = paired_ci(p_ctrl, p_unb, test)
    print(f"  paired unbounded-C vs no-C: "
          f"{out['paired_unbounded_vs_nocold']['mean']:+.4f} "
          f"P={out['paired_unbounded_vs_nocold']['p_better']:.3f}", flush=True)

    # Pareto: admission policies x capacities (EXACT top-K membership sets;
    # tiny jitter breaks the massive ties of integer counts, which would
    # otherwise inflate the effective capacity of the count policy)
    total = len(mass_gid)
    jit = rng.random(total) * 1e-6
    order_mass = np.argsort(-(mass_gid + jit))
    order_count = np.argsort(-(count_gid + jit))
    perm = rng.permutation(total)
    gid = cold["gram_id"]
    for frac in [0.001, 0.005, 0.02, 0.1, 0.3]:
        K = max(1, int(frac * total))
        for pol, order_ids in [("surprise", order_mass),
                               ("count", order_count), ("random", perm)]:
            member = np.zeros(total, dtype=bool)
            member[order_ids[:K]] = True
            adm = member[gid] & (cold["count"] >= 1)
            run_config(f"{pol}@{frac}", adm, K)
    with open(f"results/hierarchy_{domain}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
