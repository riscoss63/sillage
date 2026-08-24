"""Grid-edge check: the Qwen manuscripts tuner selected beta = 160 (the top
of the softmax grid). Rerun the sillage_amp_system variant with the grid
extended to {320, 640} to verify no gain is left on the table.
Output: results/qwen_beta_ext.json
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

import numpy as np

import sillage_factorial as V3
import memories as MEM

EXT = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0]
V3.BETAS = EXT
MEM.BETAS = EXT

from memories import splits, tune_and_eval

ids = np.load("data/q_bhd_ids.npy")
LP = np.load("dumps/q_bhd_lp.npy")
vals = ids[1:].astype(np.int64)
dev, test = splits(len(vals))
base_nll = float(-LP[test].mean())

s_true, smax, lse, sumsq = V3.v3_pass(ids, vals, LP, value="amp",
                                      gate="system", scales=(4,),
                                      vocab=151_936)
grid = [("sm", i) for i in range(len(EXT))] + [("quad", -1)]
best = tune_and_eval(lambda c: (V3.p_true_of(c, s_true, lse, sumsq), smax),
                     None, LP, dev, test, grid)
best.pop("p_true_test", None)
gain = base_nll - best["nll_test"]
out = {"betas": EXT, **best, "dnll_test": float(gain),
       "base_nll_test": base_nll}
with open("results/qwen_beta_ext.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"q_bhd extended-beta sillage_amp_system: dNLL {gain:+.4f} "
      f"cfg={best['cfg']} lam={best['lam']} thr={best['thresh_q']}")
