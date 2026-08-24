"""Final BHD v2 evaluation on all domains with the tuned key design, plus
the matched-byte capped kNN and the uniform-gating ablation.

Usage: python ngram_memory.py <aS2> <aG2>
Writes results/bhd_v2_final_<domain>.json
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

from key_selection import D_G, D_S, D_V, NGRAM, bhd_v2_pass, p_true_of
from memories import (BETAS, CAP, block_bootstrap_dnll, knn_capped_neighbors,
                      knn_p_true, load_domain, splits, tune_and_eval)


def ngram_dict_p_true(ids, vals):
    """Causal exact 4-gram -> next-token count dictionary (classical cache).
    conf = total count behind the prediction (0 when the gram is unseen)."""
    table = {}
    n = len(vals)
    p = np.zeros(n, dtype=np.float32)
    conf = np.zeros(n, dtype=np.float32)
    n_pairs = 0
    for j in range(n):
        if j >= NGRAM - 1:
            gram = ids[j - NGRAM + 1: j + 1].tobytes()
            slot = table.get(gram)
            if slot:
                tot = sum(slot.values())
                p[j] = slot.get(int(vals[j]), 0) / tot
                conf[j] = tot
            if slot is None:
                slot = {}
                table[gram] = slot
            if int(vals[j]) not in slot:
                n_pairs += 1
            slot[int(vals[j])] = slot.get(int(vals[j]), 0) + 1
    return p, conf, n_pairs


def run_domain(d, aS2, aG2):
    ids, H, LP, vals = load_domain(d)
    dev, test = splits(len(H))
    g = np.clip(-LP, 0.0, CAP)
    used_bytes = ((D_S if aS2 > 0 else 0) + (D_G if aG2 > 0 else 0)) * D_V * 4
    out = {"domain": d, "aS2": aS2, "aG2": aG2, "bytes": used_bytes,
           "base_nll_test": float(-LP[test].mean())}

    def finish(name, best):
        p_test = best.pop("p_true_test")
        gain = out["base_nll_test"] - best["nll_test"]
        ci = block_bootstrap_dnll(p_test, LP[test])
        out[name] = {**best, "dnll_test": float(gain), "dnll_ci95": ci}
        print(f"  {d}/{name}: test NLL {best['nll_test']:.4f} "
              f"(dNLL {gain:+.4f} CI {ci}) cfg={best['cfg']} "
              f"lam={best['lam']} thr={best['thresh_q']}", flush=True)

    grid_b = [(i,) for i in range(len(BETAS))]
    s_true, smax, lse = bhd_v2_pass(H, ids, vals, g, aS2, aG2)
    finish("bhd_v2", tune_and_eval(
        lambda c: (p_true_of(s_true, lse, c[0]), smax),
        None, LP, dev, test, grid_b))

    g_unif = np.full_like(g, float(g.mean()))
    s_true, smax, lse = bhd_v2_pass(H, ids, vals, g_unif, aS2, aG2)
    finish("bhd_v2_unif", tune_and_eval(
        lambda c: (p_true_of(s_true, lse, c[0]), smax),
        None, LP, dev, test, grid_b))

    cap_entries = used_bytes // (768 * 2 + 4 + 4)
    nd, nv = knn_capped_neighbors(H, vals, g, cap_entries)
    grid_k = [(k, tau) for k in (8, 16, 32) for tau in (0.5, 1.0, 3.0)]
    finish("knn_cap_matched", tune_and_eval(
        lambda c: knn_p_true(nd, nv, vals, c[0], c[1]),
        None, LP, dev, test, grid_k))

    # classical exact n-gram dict: the non-HDC twin of the winning key
    p_ng, conf_ng, n_pairs = ngram_dict_p_true(ids, vals)
    best = tune_and_eval(lambda c: (p_ng, conf_ng), None, LP, dev, test,
                         [("dict4",)])
    out["ngram_dict_pairs"] = int(n_pairs)
    out["ngram_dict_bytes_ideal"] = int(n_pairs * 16)
    finish("ngram_dict", best)

    os.makedirs("results", exist_ok=True)
    with open(f"results/bhd_v2_final_{d}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    aS2, aG2 = float(sys.argv[1]), float(sys.argv[2])
    for d in ["relativity", "alice", "bhd"]:
        run_domain(d, aS2, aG2)
