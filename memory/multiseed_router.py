"""Multi-seed replication of the router (multi-resolution semantic block).

For each hypervector seed: full pass + greedy tuning (per-seed re-tuning),
reporting G-only and router test dNLL and the per-seed paired delta.

Usage: python multiseed_router.py <gpt2|qwen> <domain> [cap]
Output: results/multiseed_router_<prefix><domain>.json
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
import sys

import numpy as np

from memories import splits
from sillage_router import B_LIST_MULTI, greedy_tune, router_pass
from sillage_semantic import load

SEEDS = [11, 22, 33, 44, 55]


def main():
    which, domain = sys.argv[1], sys.argv[2]
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    prefix, vocab, ids, H, LP, vals = load(which, domain, cap)
    n = len(vals)
    dev, test = splits(n)
    base_nll = float(-LP[test].mean())
    rows = []
    for seed in SEEDS:
        stats = router_pass(ids, H, LP, vals, vocab, hv_seed=seed,
                            b_list=tuple(B_LIST_MULTI))
        r = greedy_tune(stats, LP, dev, test)
        g = base_nll - float(-np.log(
            np.maximum(r["p_gonly"][test], 1e-30)).mean())
        rt = base_nll - float(-np.log(
            np.maximum(r["p_router"][test], 1e-30)).mean())
        d = (-np.log(np.maximum(r["p_gonly"][test], 1e-30))
             + np.log(np.maximum(r["p_router"][test], 1e-30)))
        rows.append({"seed": seed, "g_only": float(g), "router": float(rt),
                     "paired": float(d.mean())})
        print(f"{prefix}{domain} seed {seed}: G {g:+.4f} | router {rt:+.4f} "
              f"| paired {d.mean():+.4f}", flush=True)

    def agg(key):
        v = np.array([r[key] for r in rows])
        return {"mean": float(v.mean()),
                "sem": float(v.std(ddof=1) / np.sqrt(len(v))),
                "all_positive": bool((v > 0).all())}

    out = {"model": which, "domain": domain, "n": int(n), "seeds": SEEDS,
           "base_nll_test": base_nll, "per_seed": rows,
           "g_only": agg("g_only"), "router": agg("router"),
           "paired": agg("paired")}
    print(f"{prefix}{domain} AGG: G {out['g_only']['mean']:+.4f}"
          f"+/-{out['g_only']['sem']:.4f} | router "
          f"{out['router']['mean']:+.4f}+/-{out['router']['sem']:.4f} | "
          f"paired {out['paired']['mean']:+.4f}+/-{out['paired']['sem']:.4f} "
          f"(all positive: {out['paired']['all_positive']})", flush=True)
    with open(f"results/multiseed_router_{prefix}{domain}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
