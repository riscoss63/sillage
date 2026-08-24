"""Capacity-scaling datum: rerun the BHD amp memory on the bible 500k stream
with D_G = 16384 (16.8 MB, 4x the default), no decay, to isolate the effect
of matrix capacity on the saturation observed at D_G = 4096.
Output: results/exp500k_bible_D16384.json
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
import time

import numpy as np

import exp_500k as E

E.D_G = 16384
DOMAIN = "bible"

ids = np.load(f"data/{DOMAIN}_ids.npy")
LP = np.load(f"dumps/{DOMAIN}_lp.npy")
vals = ids[1:].astype(np.int64)
n = len(vals)
epos = E.eval_positions(n)
lp_e = LP[epos]
dev_e = epos < int(0.2 * n)
test_e = ~dev_e
base_nll = float(-lp_e[test_e].mean())

t0 = time.time()
s_true, smax, lse, sumsq = E.bhd_500k_pass(ids, vals, LP, epos, value="amp",
                                           decay=False)
grid = [("sm", i) for i in range(len(E.BETAS))] + [("quad", -1)]
best = E.tune(lambda c: (E.p_true_of(c, s_true, lse, sumsq), smax),
              grid, lp_e, dev_e, test_e)
p = best.pop("p")
gain = base_nll - best["nll_test"]
ci, segs = E.ci_and_segments(p, lp_e, epos, test_e, n)
out = {"domain": DOMAIN, "D_G": 16384, "bytes": 16384 * 256 * 4,
       "base_nll_test": base_nll, **best, "dnll_test": float(gain),
       "dnll_ci95": ci, "segments_dnll": segs,
       "minutes": round((time.time() - t0) / 60, 1)}
with open("results/exp500k_bible_D16384.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"bible D_G=16384: dNLL {gain:+.4f} CI {ci} cfg={best['cfg']} "
      f"lam={best['lam']} thr={best['thresh_q']} "
      f"({out['minutes']} min)", flush=True)
