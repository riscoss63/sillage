"""Paired block-bootstrap: BHD v3 (amp+system, n=4, 4.2 MB fixed) vs
UNBOUNDED kNN-LM (55 MB) on the bhd domain, each at its tuned config."""


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

import numpy as np

from sillage_factorial import p_true_of, v3_pass
from memories import knn_neighbors, knn_p_true, load_domain, splits

ids, H, LP, vals = load_domain("bhd")
dev, test = splits(len(H))
p_base = np.exp(LP)

# BHD v3 amp+system n4, tuned: softmax beta=40 (idx 4), lam=0.3, thr q0.75
s_true, smax, lse, sumsq = v3_pass(ids, vals, LP, value="amp", gate="system",
                                   scales=(4,))
thr_b = np.quantile(smax[dev], 0.75)
p_mem = p_true_of(("sm", 4), s_true, lse, sumsq)
p_bhd = np.where(smax >= thr_b, 0.3 * p_mem + 0.7 * p_base, p_base)

# unbounded kNN, tuned: k=32 tau=0.5 lam=0.1 thr q0.25 on -d_min
nd, nv = knn_neighbors(H, vals)
p_knn_raw, conf = knn_p_true(nd, nv, vals, 32, 0.5)
fin = np.isfinite(conf[dev])
thr_k = np.quantile(conf[dev][fin], 0.25)
p_knn = np.where(conf >= thr_k, 0.1 * p_knn_raw + 0.9 * p_base, p_base)

d = (-np.log(np.maximum(p_knn[test], 1e-30))
     + np.log(np.maximum(p_bhd[test], 1e-30)))
nb = len(d) // 512
blocks = [d[i * 512:(i + 1) * 512] for i in range(nb)]
rng = np.random.default_rng(3)
means = [np.concatenate([blocks[i] for i in rng.integers(0, nb, nb)]).mean()
         for _ in range(2000)]
print(f"paired dNLL (knn_unbounded - bhd_v3) on test: {d.mean():+.4f} "
      f"CI95 [{np.quantile(means, 0.025):+.4f}, "
      f"{np.quantile(means, 0.975):+.4f}]  (positive = BHD v3 better)")
print(f"P(bhd_v3 better than unbounded kNN) = "
      f"{(np.array(means) > 0).mean():.3f}")
