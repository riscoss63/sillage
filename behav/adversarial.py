"""Adversarial interference (axe 3 / paper 6): attack stored facts at
their own addresses.

Arm B -- same template, OTHER entities (matched volume): should be
harmless, because the n-gram key includes the entity pieces; this is the
selectivity claim, tested at its edge.
Arm C -- direct collision, dosed: "The {probed E} protocol requires
further inspection." writes a competing successor at the EXACT
value-predicting gram (same entity, same relation). Doses 1, 3, 9 against
the fact's 3 original occurrences; after each dose we probe the true
value, the distractor, and neither. Final step: re-probe with the cold
store disabled, to attribute the damage (matrix vs cold).

Predictions (written before running): B ~100% recall; C degrades with
dose and flips to the distractor around d>=3 (cold-count majority),
slowed by the amplitude (sqrt-mass) encoding relative to raw counts.

    python adversarial.py [--model gpt2]
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

from sillage import Sillage                                    # noqa: E402
from behavioral import (A_PREFIX, A_SENT, ENTS, VALS, build_doc,  # noqa
                        filler, probe)

DISTRACTOR = "further inspection"
OTHER_ENTS = [e + "ex" for e in ENTS] + [e + "on" for e in ENTS]
OTHER_VALS = (VALS[15:] + VALS[:15])


def collision_doc(facts, per_fact, seed):
    parts = [filler(seed, 30)]
    for k in range(per_fact):
        for e, _v in facts:
            parts.append(A_SENT.format(e=e, v=DISTRACTOR))
        parts.append(filler(seed + k + 1, 30))
    return "\n\n".join(parts)


def other_entities_doc(reps, seed):
    parts = [filler(seed, 30)]
    for k in range(reps):
        for e, v in zip(OTHER_ENTS, OTHER_VALS + OTHER_VALS):
            parts.append(A_SENT.format(e=e, v=v))
        parts.append(filler(seed + k + 1, 30))
    return "\n\n".join(parts)


def triple_probe(s, facts, n):
    """(true%, distractor%, neither%) on the A prefix."""
    t = d = 0
    for e, v in facts:
        out = s.complete(A_PREFIX.format(e=e), n=n)
        if v.split()[0] in out:
            t += 1
        elif DISTRACTOR.split()[0] in out:
            d += 1
    k = len(facts)
    return t / k, d / k, 1 - t / k - d / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    facts = list(zip(ENTS[:30], VALS[:30]))
    R = {"model": a.model, "distractor": DISTRACTOR}

    print("== bras B : meme gabarit, autres entites (volume ~dose 9) ==",
          flush=True)
    st = os.path.join(HERE, ".adv_state_B")
    shutil.rmtree(st, ignore_errors=True)
    s = Sillage(model=a.model, state=st, quiet=True)
    s.read_text(build_doc(facts, seed=0), "dossier")
    t0, _, _ = triple_probe(s, facts, a.n)
    s.read_text(other_entities_doc(5, seed=400), "autres_entites")
    tB, dB, nB = triple_probe(s, facts, a.n)
    R["armB"] = {"before": t0, "after": tB}
    print(f"  rappel {t0:.0%} -> {tB:.0%} (attendu ~inchange)", flush=True)
    shutil.rmtree(st, ignore_errors=True)

    print("== bras C : collision directe, doses 1 / 3 / 9 ==", flush=True)
    st = os.path.join(HERE, ".adv_state_C")
    shutil.rmtree(st, ignore_errors=True)
    s = Sillage(model=a.model, state=st, quiet=True)
    s.read_text(build_doc(facts, seed=0), "dossier")
    t0, _, _ = triple_probe(s, facts, a.n)
    R["armC"] = [{"dose": 0, "true": t0, "distractor": 0.0,
                  "neither": 1 - t0}]
    print(f"  dose 0 : vrai {t0:.0%}", flush=True)
    total = 0
    for add, seed in ((1, 500), (2, 600), (6, 700)):
        s.read_text(collision_doc(facts, add, seed), f"collision_{add}")
        s.save()
        total += add
        t, d, nn = triple_probe(s, facts, a.n)
        R["armC"].append({"dose": total, "true": t, "distractor": d,
                          "neither": nn})
        print(f"  dose {total} (vs 3 occurrences du fait) : vrai {t:.0%} | "
              f"distracteur {d:.0%} | ni l'un ni l'autre {nn:.0%}",
              flush=True)

    print("== decomposition finale : cold store debranche ==", flush=True)
    saved_cold = s.mem.cold
    s.mem.cold = {}
    t, d, nn = triple_probe(s, facts, a.n)
    s.mem.cold = saved_cold
    R["no_cold_at_dose9"] = {"true": t, "distractor": d, "neither": nn}
    print(f"  matrice seule (dose 9) : vrai {t:.0%} | distracteur {d:.0%} "
          f"| ni {nn:.0%}", flush=True)
    shutil.rmtree(st, ignore_errors=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"adversarial_{a.model}.json")
    json.dump(R, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
