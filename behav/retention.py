"""Long-horizon retention: recall of invented facts vs interference, with
and without leaky forgetting (axe 3 / paper 6).

Two arms on fresh states (gpt2 for speed): read the fact dossier once,
then add interference in +20k-token steps up to +100k, probing free recall
of all 30 facts at every checkpoint. The no-decay arm shows saturation's
behavioral face; the half-life arm shows what forgetting trades away.
writes/parameter is logged at each checkpoint (paper 1's saturation
coordinate, now paired with a behavioral metric).

    python retention.py [--model gpt2] [--steps 5] [--half-life 30000]
"""

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break
sys.path.insert(0, HERE)

from sillage import Sillage                       # noqa: E402
from behavioral import (A_PREFIX, ENTS, VALS, build_doc, filler,  # noqa
                        probe)


def run_arm(model, half_life, steps, sents_per_step, n):
    tag = f"hl{half_life}" if half_life else "nodecay"
    state = os.path.join(HERE, f".ret_state_{tag}")
    shutil.rmtree(state, ignore_errors=True)
    s = Sillage(model=model, state=state, half_life=half_life, quiet=True)
    facts = list(zip(ENTS[:30], VALS[:30]))
    curve = []

    s.read_text(build_doc(facts, seed=0), "dossier")
    rec, _ = probe(s, facts, A_PREFIX, n)
    curve.append({"interference_tokens": 0, "recall": rec,
                  "wpp": s.mem.writes_per_parameter()})
    print(f"  [{tag}] +0k : rappel {rec:.0%} "
          f"(w/p {curve[-1]['wpp']:.3f})", flush=True)

    for k in range(1, steps + 1):
        r = s.read_text(filler(1000 * k, sents_per_step), f"interf_{k}")
        rec, _ = probe(s, facts, A_PREFIX, n)
        curve.append({"interference_tokens": sum(
            c.get("added", 0) for c in curve) + r["tokens"],
            "added": r["tokens"], "recall": rec,
            "wpp": s.mem.writes_per_parameter()})
        print(f"  [{tag}] +{curve[-1]['interference_tokens']//1000}k : "
              f"rappel {rec:.0%} (w/p {curve[-1]['wpp']:.3f})", flush=True)
    shutil.rmtree(state, ignore_errors=True)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--sents", type=int, default=1500)
    ap.add_argument("--half-life", type=float, default=30000)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    R = {"model": a.model, "steps": a.steps, "sents_per_step": a.sents}
    print("== bras sans decroissance ==", flush=True)
    R["nodecay"] = run_arm(a.model, None, a.steps, a.sents, a.n)
    print(f"== bras half-life {a.half_life:.0f} ==", flush=True)
    R["decay"] = run_arm(a.model, a.half_life, a.steps, a.sents, a.n)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"retention_{a.model}.json")
    json.dump(R, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
