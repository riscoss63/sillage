"""Does the gate make the memory BETTER, or make it PREDICTABLE?

Two runs of nominally the same experiment disagreed:

  interference 18k chars   shannon 96.7%   uniform 96.7%   delta  0.0
  interference 12k chars   shannon 90.0%   uniform 70.0%   delta 20.0

Both replay exactly, so neither is a fluke: the disagreement comes from
the size of the interference corpus alone. Uniform swung 27 points on
that change; Shannon swung 7.

The candidate explanation is mechanical. Under a constant gate, mass
equals occurrence count, so facts read twice carry mass 2 -- exactly the
store's median. Their survival at eviction is then a TIE-BREAK, settled
by dictionary insertion order, i.e. by how many grams happen to be
competing. Under the surprise gate they carry 10 against a median of 9:
above the pack, and safe by rank rather than by luck.

If that is right, the honest claim is neither of the two the runs
suggested. It is:

    the surprise gate does not make the memory better -- it makes it
    PREDICTABLE. Without it, what survives saturation depends on an
    arbitrary tie-break.

Registered BEFORE the run:

  R1  Across a sweep of interference sizes, uniform's recall varies more
      than Shannon's: sd(uniform) > 2 x sd(shannon).
      FALSIFIED if the two vary alike -- the tie-break story would then
      be wrong and the disagreement unexplained.
  R2  At every size, the planted facts sit AT the store's median mass
      under uniform and ABOVE it under Shannon.
  R3  Shannon's mean recall over the sweep is at least as good as
      uniform's. (Not "better": the point of this probe is variance,
      not level.)

Run:  python behav/probe_gate_stability.py
"""
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sillage.core as core                          # noqa: E402
from sillage.runtime import Sillage                  # noqa: E402
from behavioral import A_PREFIX, A_SENT, ENTS, VALS  # noqa: E402
from probe_gate_pressure import paper_body           # noqa: E402
from probe_which_surprise import read_gated          # noqa: E402

SIZES = [4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000]
N_FACTS = 30
COLD_MAX = 1200


def main():
    facts = list(zip(ENTS[:N_FACTS], VALS[:N_FACTS]))
    block = "\n".join(A_SENT.format(e=e, v=v) for e, v in facts)
    target = paper_body("sillage", 5000) + "\n\n" + block + "\n"
    res = {"sizes": SIZES, "facts": N_FACTS, "cold_max": COLD_MAX,
           "rows": []}
    print(f"{N_FACTS} facts, cap {COLD_MAX}; sweeping the interference "
          f"corpus\n", flush=True)
    print(f"{'chars':>7} {'gate':<9} {'recall':>7} {'fact mass':>10} "
          f"{'median':>7} {'above?':>7}", flush=True)

    for size in SIZES:
        interference = (paper_body("behavior", size) + "\n\n"
                        + paper_body("benchmark", size))
        for gate in ("shannon", 1.0):
            name = "shannon" if gate == "shannon" else "uniform"
            tmp = tempfile.mkdtemp(prefix="stab_")
            try:
                s = Sillage(model="gpt2", state=tmp, quiet=True,
                            fastweights=False, cold_max=COLD_MAX)
                s.load_model()
                tok = s.load_tokenizer()
                for _ in range(2):
                    read_gated(s, target, gate)
                read_gated(s, interference, gate)
                s.mem.prune_cold()
                hit = 0
                for e, v in facts:
                    out = s.complete(A_PREFIX.format(e=e), n=8)
                    hit += v.split()[0].lower() in out.lower()
                masses = []
                for e, _v in facts:
                    ids = tok.encode(A_PREFIX.format(e=e))[-core.NGRAM:]
                    g = np.array(ids, dtype=np.int32).tobytes()
                    slot = s.mem.cold.get(g)
                    masses.append(float(slot[0]) if slot else 0.0)
                med = float(np.median([sl[0]
                                       for sl in s.mem.cold.values()]))
                fact_med = float(np.median(masses))
                row = {"chars": size, "gate": name,
                       "recall": round(hit / N_FACTS, 3),
                       "fact_mass_median": round(fact_med, 2),
                       "store_median": round(med, 2),
                       "above_median": bool(fact_med > med),
                       "grams": len(s.mem.cold)}
                res["rows"].append(row)
                print(f"{size:>7} {name:<9} {row['recall']:>6.0%} "
                      f"{fact_med:>10.2f} {med:>7.2f} "
                      f"{'yes' if row['above_median'] else 'TIE':>7}",
                      flush=True)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    def vals(gate, key="recall"):
        return [r[key] for r in res["rows"] if r["gate"] == gate]

    sh, un = np.array(vals("shannon")), np.array(vals("uniform"))
    res["verdict"] = {
        "R1_sd": {"shannon": round(float(sh.std()), 4),
                  "uniform": round(float(un.std()), 4)},
        "R1_holds": bool(un.std() > 2 * sh.std()),
        "R1_range": {"shannon": [float(sh.min()), float(sh.max())],
                     "uniform": [float(un.min()), float(un.max())]},
        "R2_shannon_above_median": all(
            r["above_median"] for r in res["rows"] if r["gate"] == "shannon"),
        "R2_uniform_at_median": all(
            not r["above_median"] for r in res["rows"]
            if r["gate"] == "uniform"),
        "R3_means": {"shannon": round(float(sh.mean()), 3),
                     "uniform": round(float(un.mean()), 3)},
        "R3_holds": bool(sh.mean() >= un.mean())}
    print("\n" + json.dumps(res["verdict"], indent=1, ensure_ascii=False))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "gate_stability.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
