"""Paired block-bootstrap comparison of bhd_v2 vs knn_cap_matched on the
bhd domain: recompute both per-position p_true with their tuned configs,
then bootstrap the paired per-token NLL difference over 512-token blocks.

The tuned configurations are READ from results/bhd_v2_final_bhd.json (the
file ngram_memory.py writes), never hand-copied: a re-tune there propagates
here automatically instead of silently diverging from the published table.
Output: results/paired_bhd_v2_vs_knn_cap.json
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

import ast
import json
import os

import numpy as np

from key_selection import bhd_v2_pass, p_true_of
from memories import (CAP, knn_capped_neighbors, knn_p_true,
                      load_domain, splits)

tuned = json.load(open("results/bhd_v2_final_bhd.json"))
cfg_b = tuned["bhd_v2"]
cfg_k = tuned["knn_cap_matched"]
beta_idx = ast.literal_eval(cfg_b["cfg"])[0]
lam_b, thr_b = cfg_b["lam"], cfg_b["thresh_q"]
k_knn, tau_knn = ast.literal_eval(cfg_k["cfg"])
lam_k, thr_k = cfg_k["lam"], cfg_k["thresh_q"]
used_bytes = int(tuned["bytes"])

ids, H, LP, vals = load_domain("bhd")
dev, test = splits(len(H))
g = np.clip(-LP, 0.0, CAP)
p_base = np.exp(LP)

s_true, smax, lse = bhd_v2_pass(H, ids, vals, g, tuned["aS2"], tuned["aG2"])
p_mem = p_true_of(s_true, lse, beta_idx)
if thr_b is None:
    p_bhd = lam_b * p_mem + (1 - lam_b) * p_base
else:
    m_b = smax >= np.quantile(smax[dev], thr_b)
    p_bhd = np.where(m_b, lam_b * p_mem + (1 - lam_b) * p_base, p_base)

nd, nv = knn_capped_neighbors(H, vals, g, used_bytes // (768 * 2 + 4 + 4))
p_knn_raw, conf = knn_p_true(nd, nv, vals, k_knn, tau_knn)
if thr_k is None:
    p_knn = lam_k * p_knn_raw + (1 - lam_k) * p_base
else:
    thr = np.quantile(conf[dev][np.isfinite(conf[dev])], thr_k)
    p_knn = np.where(conf >= thr, lam_k * p_knn_raw + (1 - lam_k) * p_base,
                     p_base)

nll_b = -np.log(np.maximum(p_bhd[test], 1e-30))
nll_k = -np.log(np.maximum(p_knn[test], 1e-30))
d = nll_k - nll_b            # >0 means bhd_v2 better
block = 512
nb = len(d) // block
blocks = [d[i * block:(i + 1) * block] for i in range(nb)]
rng = np.random.default_rng(3)
means = [np.concatenate([blocks[i] for i in rng.integers(0, nb, nb)]).mean()
         for _ in range(2000)]
lo, hi = np.quantile(means, 0.025), np.quantile(means, 0.975)
print(f"paired dNLL (knn_cap_matched - bhd_v2) on test: {d.mean():+.4f} "
      f"CI95 [{lo:+.4f}, {hi:+.4f}]  (positive = BHD better)")
print(f"P(bhd_v2 better) = {(np.array(means) > 0).mean():.3f}")

os.makedirs("results", exist_ok=True)
with open("results/paired_bhd_v2_vs_knn_cap.json", "w") as f:
    json.dump({
        "domain": "bhd",
        "comparison": "bhd_v2 (n-gram tier) vs byte-matched capped kNN-LM, "
                      "configs read from results/bhd_v2_final_bhd.json",
        "configs": {"bhd_v2": {"cfg": cfg_b["cfg"], "lam": lam_b,
                               "thresh_q": thr_b},
                    "knn_cap": {"cfg": cfg_k["cfg"], "lam": lam_k,
                                "thresh_q": thr_k}},
        "paired_dnll_mean": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "p_bhd_v2_better": float((np.array(means) > 0).mean()),
        "n_test": int(len(d)), "bootstrap_reps": len(means),
    }, f, indent=2)
