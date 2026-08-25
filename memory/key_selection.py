"""BHD v2 key design: hybrid semantic + n-gram compositional key.

Motivation (measured on the bhd domain, archived in
results/semantic_diag_bhd.json): centered-cosine between GPT-2 hidden states
has a heavy overlap tail (random-pair p95 = 0.94), so a linear
Hebbian readout over semantic keys alone drowns in interference — while 35%
of positions repeat a previous 4-gram verbatim, a signal that is exactly
addressable with a near-orthogonal VSA n-gram key.

Key = concat( aS * q_S , aG * q_G ),  aS^2 + aG^2 = 1
  q_S : sparse top-w sign of R (h - mu)          (semantic, soft kernel)
  q_G : sliding 4-gram binding of token hypervectors
        g_raw <- roll(g_raw, 1) * T[x_j] * roll(T[x_{j-4}], 4)   (+/-1 ints)
        q_G = g_raw / sqrt(D_g)                  (exact-match kernel ~ delta)

Variants (aS^2, aG^2) are tuned on the DEV split of the bhd domain only;
selection is by dev NLL. Usage: python key_selection.py [domain]
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

import numpy as np

from memories import (BETAS, CAP, load_domain, splits, tune_and_eval)

D_S, W_SPARSE, D_G, NGRAM, D_V = 8192, 192, 4096, 4, 256
VARIANTS = [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.33, 0.67), (0.2, 0.8)]
V2_BYTES = (D_S + D_G) * D_V * 4


def bhd_v2_pass(H, ids, vals, g, aS2, aG2, seed=7, decay_per_token=1.0,
                vocab=50257):
    rng = np.random.default_rng(seed)
    R = (rng.integers(0, 2, size=(D_S, H.shape[1])) * 2.0 - 1.0).astype(np.float32)
    V = ((rng.integers(0, 2, size=(vocab, D_V)) * 2.0 - 1.0)
         / np.sqrt(D_V)).astype(np.float32)
    T = (rng.integers(0, 2, size=(vocab, D_G), dtype=np.int8) * 2 - 1)
    aS, aG = np.sqrt(aS2), np.sqrt(aG2)

    M_S = np.zeros((D_S, D_V), dtype=np.float32)
    M_G = np.zeros((D_G, D_V), dtype=np.float32)
    n = len(H)
    s_true = np.zeros(n, dtype=np.float32)
    smax = np.zeros(n, dtype=np.float32)
    lse = np.zeros((n, len(BETAS)), dtype=np.float32)
    mu = np.zeros(H.shape[1], dtype=np.float32)
    betas = np.array(BETAS, dtype=np.float32)
    g_raw = np.ones(D_G, dtype=np.float32)
    inv_sqrt_dg = 1.0 / np.sqrt(D_G)

    for j in range(n):
        # --- n-gram key (tokens x[j-3..j] are all visible at prediction j)
        g_raw = np.roll(g_raw, 1)
        g_raw *= T[ids[j]]
        if j >= NGRAM:
            g_raw *= np.roll(T[ids[j - NGRAM]], NGRAM)
        qG = g_raw * inv_sqrt_dg

        # --- semantic key
        c = H[j] - mu
        z = R @ c
        idx = np.argpartition(-np.abs(z), W_SPARSE - 1)[:W_SPARSE]
        qs = (np.sign(z[idx]) / np.sqrt(W_SPARSE)).astype(np.float32)

        # --- read
        u = aS * (M_S[idx].T @ qs) + aG * (M_G.T @ qG) if aS2 > 0 else \
            aG * (M_G.T @ qG)
        if aG2 == 0.0:
            u = aS * (M_S[idx].T @ qs)
        s = (V @ u) / (np.linalg.norm(u) + 1e-8)
        s_true[j] = s[vals[j]]
        smax[j] = s.max()
        m = betas[:, None] * s[None, :]
        mx = m.max(axis=1)
        lse[j] = mx + np.log(np.exp(m - mx[:, None]).sum(axis=1))

        # --- write
        if decay_per_token != 1.0:
            M_S *= decay_per_token
            M_G *= decay_per_token
        gv = (g[j] * V[vals[j]]).astype(np.float32)
        if aS2 > 0:
            M_S[idx] += (aS * qs)[:, None] * gv[None, :]
        if aG2 > 0:
            M_G += (aG * qG)[:, None] * gv[None, :]
        mu += (H[j] - mu) / (j + 1)
    return s_true, smax, lse


def p_true_of(s_true, lse, beta_idx):
    return np.exp(BETAS[beta_idx] * s_true - lse[:, beta_idx])


def main(domain="bhd"):
    ids, H, LP, vals = load_domain(domain)
    dev, test = splits(len(H))
    g = np.clip(-LP, 0.0, CAP)
    out = {"domain": domain, "base_nll_test": float(-LP[test].mean()),
           "base_nll_dev": float(-LP[dev].mean()), "v2_bytes": V2_BYTES}
    for aS2, aG2 in VARIANTS:
        s_true, smax, lse = bhd_v2_pass(H, ids, vals, g, aS2, aG2)
        best = tune_and_eval(
            lambda c: (p_true_of(s_true, lse, c[0]), smax),
            None, LP, dev, test, [(i,) for i in range(len(BETAS))])
        best.pop("p_true_test", None)
        out[f"aS2={aS2},aG2={aG2}"] = best
        print(f"aS2={aS2} aG2={aG2}: dev NLL {best['nll_dev']:.4f} "
              f"(base {out['base_nll_dev']:.4f}) | test {best['nll_test']:.4f}"
              f" (base {out['base_nll_test']:.4f}) cfg={best['cfg']} "
              f"lam={best['lam']} thr={best['thresh_q']}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open(f"results/bhd_v2_tune_{domain}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bhd")
