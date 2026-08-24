"""BHD v3 — the distributional (phi_B) iteration, reconnecting to Theorem 1.

Three mechanisms, factorially tested against the v2 control:

  value encoding:
    count : M accumulates g * V[token]  -> coefficients ~ surprise-weighted
            counts of successors (phi_lin-style values; the v2 behaviour)
    amp   : read-modify-write keeps coefficients ~ sqrt(sum g) (Bhattacharyya
            amplitudes; phi_B-style values). Local update:
                a    = <u, V[t]>            (current amplitude, crosstalk incl.)
                coef = sqrt(max(a,0)^2 + g) - max(a,0)
            Readout option 'quad': p_mem ~ relu(s)^2 (Born-style), on top of
            the softmax readouts.

  write gate:
    model  : g = clip(-ln p_LM(x_{j+1}), 0, CAP)          (v2 behaviour)
    system : g = clip(-ln(0.2 p_mem + 0.8 p_LM), 0, CAP)  (surprise of the
             AUGMENTED system: the memory stops over-writing what it already
             predicts -- the closed active-inference loop)

  key scales:
    [4]      : single 4-gram sliding key (v2 behaviour)
    [2,4,8]  : multi-scale concatenated sliding keys (backoff-style coverage)

All variants share the same token/value hypervectors (paired comparisons).
Usage: python sillage_factorial.py [domain ...]
Output: results/bhd_v3_<domain>.json
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

from memories import (BETAS, CAP, block_bootstrap_dnll, load_domain, splits,
                      tune_and_eval)

D_G, D_V = 4096, 256
SYSTEM_LAM, SYSTEM_BETA_IDX = 0.2, 3          # beta = BETAS[3] = 20
VARIANTS = [
    ("count_model_n4", dict(value="count", gate="model", scales=(4,))),
    ("amp_model_n4", dict(value="amp", gate="model", scales=(4,))),
    ("count_system_n4", dict(value="count", gate="system", scales=(4,))),
    ("amp_system_n4", dict(value="amp", gate="system", scales=(4,))),
    ("count_system_n248", dict(value="count", gate="system",
                               scales=(2, 4, 8))),
    ("amp_system_n248", dict(value="amp", gate="system", scales=(2, 4, 8))),
]


def v3_pass(ids, vals, LP, value, gate, scales, vocab=50257, hv_seed=7001):
    rngV = np.random.default_rng(hv_seed)
    rngT = np.random.default_rng(hv_seed + 1)
    V = ((rngV.integers(0, 2, size=(vocab, D_V)) * 2.0 - 1.0)
         / np.sqrt(D_V)).astype(np.float32)
    T = (rngT.integers(0, 2, size=(vocab, D_G), dtype=np.int8) * 2 - 1)
    ns = len(scales)
    M = np.zeros((ns * D_G, D_V), dtype=np.float32)
    n = len(vals)
    s_true = np.zeros(n, dtype=np.float32)
    smax = np.zeros(n, dtype=np.float32)
    lse = np.zeros((n, len(BETAS)), dtype=np.float32)
    sumsq = np.zeros(n, dtype=np.float32)
    betas = np.array(BETAS, dtype=np.float32)
    p_base = np.exp(LP)
    g_raw = [np.ones(D_G, dtype=np.float32) for _ in scales]
    scale_q = 1.0 / np.sqrt(D_G * ns)

    for j in range(n):
        q = np.empty(ns * D_G, dtype=np.float32)
        for si, nn in enumerate(scales):
            g_raw[si] = np.roll(g_raw[si], 1)
            g_raw[si] *= T[ids[j]]
            if j >= nn:
                g_raw[si] *= np.roll(T[ids[j - nn]], nn)
            q[si * D_G:(si + 1) * D_G] = g_raw[si] * scale_q

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

        if gate == "model":
            gj = min(CAP, max(0.0, -float(LP[j])))
        else:
            pm = float(np.exp(BETAS[SYSTEM_BETA_IDX] * st
                              - lse[j, SYSTEM_BETA_IDX]))
            p_sys = SYSTEM_LAM * pm + (1 - SYSTEM_LAM) * float(p_base[j])
            gj = min(CAP, max(0.0, -np.log(max(p_sys, 1e-30))))

        if value == "count":
            coef = gj
        else:
            a = max(0.0, st * un)          # <u, V[true]> raw amplitude
            coef = np.sqrt(a * a + gj) - a
        M += coef * q[:, None] * V[vals[j]][None, :]
    return s_true, smax, lse, sumsq


def p_true_of(cfg, s_true, lse, sumsq):
    if cfg[0] == "quad":
        return np.where(sumsq > 0,
                        np.maximum(s_true, 0.0) ** 2 / np.maximum(sumsq, 1e-12),
                        0.0)
    return np.exp(BETAS[cfg[1]] * s_true - lse[:, cfg[1]])


def run_domain(d):
    ids, H, LP, vals = load_domain(d)
    dev, test = splits(len(H))
    out = {"domain": d, "base_nll_test": float(-LP[test].mean())}
    grid = [("sm", i) for i in range(len(BETAS))] + [("quad", -1)]
    p_control = None
    for name, kw in VARIANTS:
        t0 = time.time()
        s_true, smax, lse, sumsq = v3_pass(ids, vals, LP, **kw)
        best = tune_and_eval(
            lambda c: (p_true_of(c, s_true, lse, sumsq), smax),
            None, LP, dev, test, grid)
        p_test = best.pop("p_true_test")
        gain = out["base_nll_test"] - best["nll_test"]
        ci = block_bootstrap_dnll(p_test, LP[test])
        row = {**best, "dnll_test": float(gain), "dnll_ci95": ci,
               "bytes": int(len(kw["scales"]) * D_G * D_V * 4),
               "minutes": round((time.time() - t0) / 60, 1)}
        if name == "count_model_n4":
            p_control = p_test
        elif p_control is not None:
            dd = (-np.log(np.maximum(p_control, 1e-30))
                  + np.log(np.maximum(p_test, 1e-30)))
            nb = len(dd) // 512
            blocks = [dd[i * 512:(i + 1) * 512] for i in range(nb)]
            rng = np.random.default_rng(3)
            means = [np.concatenate([blocks[i] for i in
                                     rng.integers(0, nb, nb)]).mean()
                     for _ in range(1000)]
            row["paired_vs_control"] = {
                "mean": float(dd.mean()),
                "ci95": [float(np.quantile(means, 0.025)),
                         float(np.quantile(means, 0.975))],
                "p_better": float((np.array(means) > 0).mean()),
            }
        out[name] = row
        pv = row.get("paired_vs_control", {})
        print(f"  {d}/{name}: dNLL {gain:+.4f} CI {ci} cfg={best['cfg']} "
              f"lam={best['lam']} thr={best['thresh_q']}"
              + (f" | paired vs v2: {pv['mean']:+.4f} "
                 f"P(better)={pv['p_better']:.2f}" if pv else ""),
              flush=True)

    os.makedirs("results", exist_ok=True)
    with open(f"results/bhd_v3_{d}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    domains = sys.argv[1:] or ["bhd", "relativity", "alice"]
    for d in domains:
        print(f"=== {d} ===", flush=True)
        run_domain(d)
