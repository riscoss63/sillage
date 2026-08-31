"""The same question, with enough resolution to answer it.

`probe_gate_pressure` found Shannon and uniform gating identical --
100% = 100% with the whole memory, 67% = 67% with the matrix alone --
under interference and capacity pressure, the regime where the gate is
supposed to earn its place. But three probes resolve recall only to 33%,
on one seed and one corpus. A null result at that resolution settles
nothing, and this question is about the central mechanism of paper 1.

So: thirty planted facts instead of three, two independent interference
orderings, and the same three gates. Resolution 3.3%.

What is already established and not retested here: the gate DOES what it
mechanically claims -- store median mass 9.0 under Shannon against 2.0
under uniform, a factor of 4.5, correctly ranked. The question is only
whether that changes any outcome.

Registered BEFORE the run:

  P1  Shannon retains more of the thirty facts than uniform after
      interference, by more than the 3.3% resolution.
      FALSIFIED if the difference is within one fact.
  P2  Same with the cold store emptied -- paper 1's amplitude claim
      alone.
  P3  Bayes stays worst.
  P4  If P1 and P2 both fail, the honest statement is that the surprise
      gate has no measurable behavioural effect in this regime at this
      resolution, and the project should say so.

Run:  python behav/probe_gate_power.py [--facts 30] [--seeds 2]
"""
import argparse
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sillage.runtime import Sillage                  # noqa: E402
from behavioral import A_PREFIX, A_SENT, ENTS, VALS  # noqa: E402
from probe_gate_pressure import paper_body           # noqa: E402
from probe_which_surprise import read_gated          # noqa: E402


def make_facts(n):
    return list(zip(ENTS[:n], VALS[:n]))


def recall_of(s, facts, n=8):
    hit = 0
    for e, v in facts:
        out = s.complete(A_PREFIX.format(e=e), n=n)
        hit += v.split()[0].lower() in out.lower()
    return hit / len(facts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--facts", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--cold-max", type=int, default=1200)
    a = ap.parse_args()

    facts = make_facts(a.facts)
    block = "\n".join(A_SENT.format(e=e, v=v) for e, v in facts)
    target = paper_body("sillage", 5000) + "\n\n" + block + "\n"
    pieces = [paper_body("behavior", 9000), paper_body("benchmark", 9000)]

    res = {"model": a.model, "facts": a.facts, "seeds": a.seeds,
           "cold_max": a.cold_max, "runs": []}
    print(f"{a.facts} planted facts, resolution "
          f"{100 / a.facts:.1f}%, cap {a.cold_max}\n", flush=True)

    for seed in range(a.seeds):
        order = pieces if seed == 0 else pieces[::-1]
        interference = "\n\n".join(order)
        for gate in ("shannon", "uniform", "bayes"):
            tmp = tempfile.mkdtemp(prefix="power_")
            try:
                s = Sillage(model=a.model, state=tmp, quiet=True,
                            fastweights=False, cold_max=a.cold_max)
                s.load_model()
                for _ in range(2):
                    read_gated(s, target, gate)
                before = recall_of(s, facts)
                read_gated(s, interference, gate)
                s.mem.prune_cold()
                after = recall_of(s, facts)
                cold_backup = s.mem.cold
                s.mem.cold = {}
                matrix_only = recall_of(s, facts)
                s.mem.cold = cold_backup
                row = {"seed": seed, "gate": gate,
                       "recall_before": round(before, 3),
                       "recall_after": round(after, 3),
                       "matrix_only": round(matrix_only, 3),
                       "grams": len(s.mem.cold),
                       "median_mass": round(float(np.median(
                           [sl[0] for sl in s.mem.cold.values()])), 2)}
                res["runs"].append(row)
                print(f"  seed {seed}  {gate:<9} "
                      f"{before:.0%} -> {after:.0%}  "
                      f"(matrix only {matrix_only:.0%})  "
                      f"median mass {row['median_mass']}", flush=True)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    def mean(gate, key):
        v = [r[key] for r in res["runs"] if r["gate"] == gate]
        return round(float(np.mean(v)), 3)

    d_after = mean("shannon", "recall_after") - mean("uniform",
                                                     "recall_after")
    d_matrix = mean("shannon", "matrix_only") - mean("uniform",
                                                     "matrix_only")
    one_fact = 1.0 / a.facts
    res["summary"] = {g: {k: mean(g, k) for k in
                          ("recall_before", "recall_after", "matrix_only",
                           "median_mass")}
                      for g in ("shannon", "uniform", "bayes")}
    res["verdict"] = {
        "P1_delta_after": round(d_after, 3),
        "P1_holds": d_after > one_fact,
        "P2_delta_matrix": round(d_matrix, 3),
        "P2_holds": d_matrix > one_fact,
        "P3_bayes_worst": mean("bayes", "recall_after")
        <= min(mean("shannon", "recall_after"), mean("uniform",
                                                     "recall_after")),
        "one_fact_is": round(one_fact, 3)}
    print("\n" + json.dumps(res["summary"], indent=1))
    print(json.dumps(res["verdict"], indent=1))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "gate_power.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
