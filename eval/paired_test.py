"""Paired block-bootstrap comparison of bhd_v2 vs knn_cap_matched on the
bhd domain: recompute both per-position p_true with their tuned configs,
then bootstrap the paired per-token NLL difference over 512-token blocks.
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

import numpy as np

from key_selection import bhd_v2_pass, p_true_of
from memories import (BETAS, CAP, knn_capped_neighbors, knn_p_true,
                      load_domain, splits)

ids, H, LP, vals = load_domain("bhd")
dev, test = splits(len(H))
g = np.clip(-LP, 0.0, CAP)
p_base = np.exp(LP)

# bhd_v2 tuned config: beta idx 3 (=20), lam 0.05, no threshold
s_true, smax, lse = bhd_v2_pass(H, ids, vals, g, 0.0, 1.0)
p_bhd = 0.05 * p_true_of(s_true, lse, 3) + 0.95 * p_base

# knn_cap_matched tuned config: k=32 tau=0.5 lam=0.1 thr q0.25 on -d_min
used_bytes = 4096 * 256 * 4
nd, nv = knn_capped_neighbors(H, vals, g, used_bytes // (768 * 2 + 4 + 4))
p_knn_raw, conf = knn_p_true(nd, nv, vals, 32, 0.5)
thr = np.quantile(conf[dev][np.isfinite(conf[dev])], 0.25)
mask = conf >= thr
p_knn = np.where(mask, 0.1 * p_knn_raw + 0.9 * p_base, p_base)

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
