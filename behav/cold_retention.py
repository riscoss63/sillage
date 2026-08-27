"""Who carries retention? (axe 3 / paper 6)

Rebuilds the two retention arms (no decay / half-life 30k) and, at three
checkpoints (0, +~43k, +~110k interference tokens), decomposes recall into
four voices: full system, cold store disabled (matrix+adapter), matrix
silenced via lam_G=0 (cold+adapter), and neither (base+adapter floor --
which also asks whether the rank-16 delta-rule adapter memorized facts on
its own).

Predictions, written before running: in the decay arm at +110k the
dossier's matrix amplitudes have decayed ~16x, so no_cold should collapse
while no_matrix holds -- the P3 hierarchy as retention insurance. If
no_cold holds anyway, the hypothesis is wrong and the adapter or residual
matrix carries it.

    python cold_retention.py [--model gpt2]
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

from sillage import Sillage                                   # noqa: E402
from behavioral import (A_PREFIX, ENTS, VALS, build_doc,      # noqa
                        filler, probe)

CHECK_AFTER_STEPS = (0, 2, 5)          # dossier, +~43k, +~110k


def decompose(s, facts, n):
    mem = s.mem
    out = {}
    out["full"], _ = probe(s, facts, A_PREFIX, n)
    saved_cold = mem.cold
    mem.cold = {}
    out["no_cold"], _ = probe(s, facts, A_PREFIX, n)
    mem.cold = saved_cold
    saved_lam = mem.lam_G
    mem.lam_G = 0.0
    out["no_matrix"], _ = probe(s, facts, A_PREFIX, n)
    mem.cold = {}
    out["base_adapter"], _ = probe(s, facts, A_PREFIX, n)
    mem.cold = saved_cold
    mem.lam_G = saved_lam
    return out


def run_arm(model, half_life, steps, sents, n, facts):
    tag = f"hl{half_life:.0f}" if half_life else "nodecay"
    state = os.path.join(HERE, f".coldret_{tag}")
    shutil.rmtree(state, ignore_errors=True)
    s = Sillage(model=model, state=state, half_life=half_life, quiet=True)
    curve = []
    s.read_text(build_doc(facts, seed=0), "dossier")
    interf = 0
    for step in range(0, steps + 1):
        if step > 0:
            r = s.read_text(filler(1000 * step, sents), f"interf_{step}")
            interf += r["tokens"]
        if step in CHECK_AFTER_STEPS:
            d = decompose(s, facts, n)
            d["interference_tokens"] = interf
            d["wpp"] = s.mem.writes_per_parameter()
            curve.append(d)
            print(f"  [{tag}] +{interf//1000}k : full {d['full']:.0%} | "
                  f"sans cold {d['no_cold']:.0%} | sans matrice "
                  f"{d['no_matrix']:.0%} | base+adapt "
                  f"{d['base_adapter']:.0%}", flush=True)
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
    facts = list(zip(ENTS[:30], VALS[:30]))

    R = {"model": a.model}
    print("== bras sans decroissance ==", flush=True)
    R["nodecay"] = run_arm(a.model, None, a.steps, a.sents, a.n, facts)
    print(f"== bras half-life {a.half_life:.0f} ==", flush=True)
    R["decay"] = run_arm(a.model, a.half_life, a.steps, a.sents, a.n,
                         facts)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"cold_retention_{a.model}.json")
    json.dump(R, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
