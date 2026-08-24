"""Multi-seed replication of the v3 headline results.

For each domain, reruns the control (count/model gate/n=4) and the winner
(amp/system gate/n=4) — plus the multi-scale winner on the two classic
domains — with 5 fresh hypervector seeds. Hyperparameters are re-tuned on
dev for every seed (full pipeline per seed). Reports mean +/- SEM of the
test dNLL and of the per-seed paired delta (winner - control).

Usage: python multiseed.py <domain>
Output: results/multiseed_<domain>.json
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
import os
import sys

import numpy as np

from sillage_factorial import p_true_of, v3_pass
from memories import BETAS, load_domain, splits, tune_and_eval

SEEDS = [11, 22, 33, 44, 55]


def run(domain):
    ids, H, LP, vals = load_domain(domain)
    dev, test = splits(len(H))
    base_nll = float(-LP[test].mean())
    grid = [("sm", i) for i in range(len(BETAS))] + [("quad", -1)]

    variants = [("control", dict(value="count", gate="model", scales=(4,))),
                ("amp_system_n4", dict(value="amp", gate="system",
                                       scales=(4,)))]
    if domain in ("relativity", "alice"):
        variants.append(("amp_system_n248", dict(value="amp", gate="system",
                                                 scales=(2, 4, 8))))

    per_seed = {name: [] for name, _ in variants}
    p_tests = {name: {} for name, _ in variants}
    for seed in SEEDS:
        for name, kw in variants:
            s_true, smax, lse, sumsq = v3_pass(ids, vals, LP, hv_seed=seed,
                                               **kw)
            best = tune_and_eval(
                lambda c: (p_true_of(c, s_true, lse, sumsq), smax),
                None, LP, dev, test, grid)
            p_tests[name][seed] = best.pop("p_true_test")
            dnll = base_nll - best["nll_test"]
            per_seed[name].append(dnll)
            print(f"{domain} seed {seed} {name}: dNLL {dnll:+.4f} "
                  f"(cfg={best['cfg']} lam={best['lam']} "
                  f"thr={best['thresh_q']})", flush=True)

    out = {"domain": domain, "base_nll_test": base_nll, "seeds": SEEDS}
    for name, _ in variants:
        arr = np.array(per_seed[name])
        out[name] = {"dnll_per_seed": arr.tolist(),
                     "dnll_mean": float(arr.mean()),
                     "dnll_sem": float(arr.std(ddof=1) / np.sqrt(len(arr)))}
    for name, _ in variants[1:]:
        deltas = []
        for seed in SEEDS:
            d = (-np.log(np.maximum(p_tests["control"][seed], 1e-30))
                 + np.log(np.maximum(p_tests[name][seed], 1e-30)))
            deltas.append(float(d.mean()))
        deltas = np.array(deltas)
        out[name]["paired_delta_vs_control"] = {
            "per_seed": deltas.tolist(),
            "mean": float(deltas.mean()),
            "sem": float(deltas.std(ddof=1) / np.sqrt(len(deltas))),
            "all_positive": bool((deltas > 0).all()),
        }
        print(f"{domain} {name} paired vs control: "
              f"{deltas.mean():+.4f} +/- {deltas.std(ddof=1)/np.sqrt(5):.4f} "
              f"(all positive: {(deltas > 0).all()})", flush=True)

    os.makedirs("results", exist_ok=True)
    with open(f"results/multiseed_{domain}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    run(sys.argv[1])
