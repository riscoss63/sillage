"""Diagnose the no-cold shortfall: re-tune the cached hierarchy-pass stats
with the EXACT paper-2 tuner (sillage_router.greedy_tune) and compare against
router500k_<domain>_decay.json. If the cached stats reproduce the paper-2
number, the pass is fine and hierarchy's tuner is at fault; otherwise the
pass differs."""


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

from exp_500k import eval_positions
from sillage_router import greedy_tune

domain = sys.argv[1] if len(sys.argv) > 1 else "tolstoy"
z = np.load(f"results/hier_cache_{domain}.npz")
stats = {b: {k: z[f"{b}_{k}"] for k in ("s_true", "smax", "lse")}
         for b in ("G", "S")}
for b in ("G", "S"):
    stats[b]["sumsq"] = np.zeros_like(stats[b]["s_true"])  # quad unused ref
LP = np.load(f"dumps/{domain}_lp.npy")
n = len(LP)
epos = eval_positions(n)
lp_e = LP[epos]
dev = epos < int(0.2 * n)
test = ~dev
base = float(-lp_e[test].mean())
r = greedy_tune(stats, lp_e, dev, test)
g = base - float(-np.log(np.maximum(r["p_gonly"][test], 1e-30)).mean())
rt = base - float(-np.log(np.maximum(r["p_router"][test], 1e-30)).mean())
print(f"{domain} cached-stats + paper-2 tuner: G-only {g:+.4f} | "
      f"router {rt:+.4f} | params G={r['G']} S={r['S']}")
ref = json.load(open(f"results/router500k_{domain}_decay.json"))
print(f"reference router500k_decay: G {ref['g_only']['dnll_test']:+.4f} | "
      f"router {ref['router']['dnll_test']:+.4f}")
