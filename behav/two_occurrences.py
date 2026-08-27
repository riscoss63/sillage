"""The two-occurrence rule (axe 3 / paper 6): is the cold-store admission
threshold the threshold of durable memory?

Thirty facts in three groups: seen once (G1), twice (G2), three times
(G3) in one graded dossier, then +~110k tokens of interference.
Predictions, written before running: all groups ~100% at +0k; at +110k
G1 collapses (never admitted to the cold store -- COLD_MIN_COUNT = 2 --
so it rode only on the eroding matrix) while G2 and G3 hold. Direct
mechanism check: cold answerability per group expected 0/10, 10/10,
10/10; final no-cold decomposition attributes the survivals.

    python two_occurrences.py [--model gpt2]
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break
sys.path.insert(0, HERE)

from sillage import Sillage                                    # noqa: E402
from sillage.core import COLD_MIN_COUNT, NGRAM                 # noqa: E402
from behavioral import (A_PREFIX, A_SENT, ENTS, VALS, filler,  # noqa
                        probe)


def graded_doc(g1, g2, g3, seed, block=40):
    parts = [filler(seed, block)]
    for wave in (g1 + g2 + g3, g2 + g3, g3):
        for e, v in wave:
            parts.append(A_SENT.format(e=e, v=v))
        parts.append(filler(seed + len(parts), block))
    return "\n\n".join(parts)


def cold_answerable(s, facts):
    """How many facts' probe grams the cold store would actually answer."""
    tok, _ = s.load_model()
    n = 0
    for e, _v in facts:
        gram = tok.encode(A_PREFIX.format(e=e))[-NGRAM:]
        key = np.array(gram, dtype=np.int32).tobytes()
        slot = s.mem.cold.get(key)
        if slot is not None and sum(slot[1].values()) >= COLD_MIN_COUNT:
            n += 1
    return n


def report(s, groups, n, label):
    line = [label]
    out = {}
    for name, facts in groups.items():
        r, _ = probe(s, facts, A_PREFIX, n)
        out[name] = r
        line.append(f"{name} {r:.0%}")
    print("  " + " | ".join(line), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--sents", type=int, default=1500)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    facts = list(zip(ENTS[:30], VALS[:30]))
    groups = {"G1(x1)": facts[:10], "G2(x2)": facts[10:20],
              "G3(x3)": facts[20:30]}

    state = os.path.join(HERE, ".two_occ_state")
    shutil.rmtree(state, ignore_errors=True)
    s = Sillage(model=a.model, state=state, quiet=True)
    R = {"model": a.model, "checkpoints": []}

    s.read_text(graded_doc(groups["G1(x1)"], groups["G2(x2)"],
                           groups["G3(x3)"], seed=0), "dossier_gradue")
    s.save()
    cov = {name: cold_answerable(s, f) for name, f in groups.items()}
    print(f"  couverture cold (repond au gram du probe) : "
          + " | ".join(f"{k} {v}/10" for k, v in cov.items()), flush=True)
    R["cold_coverage"] = cov
    R["checkpoints"].append({"interf": 0,
                             **report(s, groups, a.n, "+0k   :")})

    interf = 0
    for k in range(1, a.steps + 1):
        r = s.read_text(filler(2000 * k, a.sents), f"interf_{k}")
        interf += r["tokens"]
        if k in (2, a.steps):
            R["checkpoints"].append(
                {"interf": interf,
                 **report(s, groups, a.n, f"+{interf//1000}k :")})

    saved_cold = s.mem.cold
    s.mem.cold = {}
    R["no_cold_final"] = report(s, groups, a.n, "final sans cold :")
    s.mem.cold = saved_cold

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"two_occurrences_{a.model}.json")
    json.dump(R, open(out, "w"), indent=2)
    print(f"saved -> {out}")
    shutil.rmtree(state, ignore_errors=True)


if __name__ == "__main__":
    main()
