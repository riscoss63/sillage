"""Continual A -> B -> A with the tuned BHD v2 key (n-gram, G-only),
with and without leaky decay. Complements results/continual.json.
Output: results/continual_v2.json
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

import numpy as np

from key_selection import bhd_v2_pass, p_true_of
from continual import HALF_LIFE, tune_then_segments
from memories import BETAS, CAP, load_domain

SEG = 15_000


def main():
    ids, H, LP, vals = load_domain("aba")
    n = len(H)
    seg_of_pred = np.minimum((np.arange(n) + 1) // SEG, 2)
    seg_masks = {"A1": seg_of_pred == 0, "B": seg_of_pred == 1,
                 "A2": seg_of_pred == 2}
    g = np.clip(-LP, 0.0, CAP)
    out = {"base": {k: float(-LP[m].mean()) for k, m in seg_masks.items()}}
    grid = [(i,) for i in range(len(BETAS))]
    for name, decay in [("bhd_v2", 1.0),
                        ("bhd_v2_decay", 0.5 ** (1.0 / HALF_LIFE))]:
        s_true, smax, lse = bhd_v2_pass(H, ids, vals, g, 0.0, 1.0,
                                        decay_per_token=decay)
        out[name] = tune_then_segments(
            lambda c: (p_true_of(s_true, lse, c[0]), smax), grid, LP,
            seg_masks)
        r = out[name]
        print(f"{name}: A1 {r['nll_A1']:.4f} | B {r['nll_B']:.4f} | "
              f"A2 {r['nll_A2']:.4f} (cfg={r['cfg']} lam={r['lam']} "
              f"thr={r['thresh_q']})", flush=True)
    print("base:", out["base"])

    os.makedirs("results", exist_ok=True)
    with open("results/continual_v2.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
