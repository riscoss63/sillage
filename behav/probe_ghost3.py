"""Stop guessing when the store is not sure: does precision follow?

`probe_ghost2` found the two faults, and both come from ONE decision:
the chain takes the argmax successor even when the gram has several,
with nearly equal counts.

  * it spliced sequences that were never written (2 of 200), because
    two different continuations were joined at an ambiguous gram
  * on a LATER report about the same station it offered something at
    61% of positions but was right at only 29% -- typing `2026-207` it
    proposed `114`, the previous report's number

Autocomplete is judged on precision, not coverage: a wrong ghost costs
a glance, but a feature that is wrong two times in three trains the
typist to ignore it, and then it may as well not exist.

So the chain gets a stopping rule, and the rule is chosen by
measurement rather than taste. Five candidates, from "always take the
argmax" (today) to "only continue when the store has seen exactly one
continuation".

Registered BEFORE the run:

  K1  A stricter rule raises precision. FALSIFIED if precision is flat
      across the whole sweep -- ambiguity would then not be the cause.
  K2  `unique` (continue only where the gram has ONE successor) splices
      nothing: every suggestion is a substring of what was read.
      FALSIFIED by one counter-example.
  K3  There is a rule with precision >= 60% at coverage >= 20%. That is
      the shape of a feature worth shipping: it speaks less often than
      today, and is right more often than not when it does.
      FALSIFIED if no rule reaches both.
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
from probe_readout_dial import DOC as DOC_FR         # noqa: E402
from probe_ghost2 import FOLLOW_UP                   # noqa: E402


def ghost(mem, ids, steps=8, rule=("argmax", 0.0)):
    """Follow the cold store, stopping where it is not sure enough.

    rule is (kind, parameter):
      argmax     take the most frequent successor, always (today)
      dominance  continue only while top count >= r * runner-up
      unique     continue only while the gram has ONE successor
    """
    kind, r = rule
    out, hist = [], list(ids)
    for _ in range(steps):
        if len(hist) < core.NGRAM:
            break
        gram = np.array(hist[-core.NGRAM:], dtype=np.int32).tobytes()
        slot = mem.cold.get(gram)
        if slot is None or sum(slot[1].values()) < core.COLD_MIN_COUNT:
            break
        counts = sorted(slot[1].items(), key=lambda kv: -kv[1])
        if kind == "unique" and len(counts) > 1:
            break
        if kind == "dominance" and len(counts) > 1 and \
                counts[0][1] < r * counts[1][1]:
            break
        out.append(counts[0][0])
        hist.append(counts[0][0])
    return out


RULES = [("argmax", 0.0), ("dominance", 1.5), ("dominance", 2.0),
         ("dominance", 3.0), ("unique", 0.0)]


def main():
    tmp = tempfile.mkdtemp(prefix="ghost3_")
    res = {"rules": []}
    try:
        s = Sillage(model="qwen", state=tmp, quiet=True)
        source = Sillage.reflow(DOC_FR)
        for _ in range(2):
            s.read_text(source)
        tok = s.load_tokenizer()
        mem = s.mem
        src_txt = tok.decode(tok.encode(source))
        ids_new = tok.encode(Sillage.reflow(FOLLOW_UP))
        spots = list(range(core.NGRAM, len(ids_new) - 10, 2))
        ids_src = tok.encode(source)
        spots_src = list(range(core.NGRAM, len(ids_src) - 10, 2))

        print(f"{'rule':<16} {'coverage':>9} {'precision':>10} "
              f"{'mean len':>9} {'spliced':>8}", flush=True)
        for rule in RULES:
            offered = right = 0
            lens = []
            for i in spots:
                g = ghost(mem, ids_new[:i], rule=rule)
                if len(g) < 2:
                    continue
                offered += 1
                lens.append(len(g))
                right += g == list(ids_new[i:i + len(g)])
            spliced = 0
            for i in spots_src:
                g = ghost(mem, ids_src[:i], rule=rule)
                if len(g) >= 2 and tok.decode(g) not in src_txt:
                    spliced += 1
            row = {"rule": f"{rule[0]}{('' if not rule[1] else ' ' + str(rule[1]))}",
                   "coverage": round(offered / len(spots), 3),
                   "precision": round(right / max(1, offered), 3),
                   "offered": offered, "right": right,
                   "mean_len": round(float(np.mean(lens)), 1) if lens else 0,
                   "spliced": spliced}
            res["rules"].append(row)
            print(f"{row['rule']:<16} {row['coverage']:>8.0%} "
                  f"{row['precision']:>9.0%} {row['mean_len']:>9.1f} "
                  f"{row['spliced']:>8d}", flush=True)

        best = [r for r in res["rules"]
                if r["precision"] >= 0.60 and r["coverage"] >= 0.20]
        uniq = next(r for r in res["rules"] if r["rule"] == "unique")
        flat = (max(r["precision"] for r in res["rules"])
                - min(r["precision"] for r in res["rules"])) < 0.05
        res["verdict"] = {
            "K1_precision_moves": not flat,
            "K1_range": [min(r["precision"] for r in res["rules"]),
                         max(r["precision"] for r in res["rules"])],
            "K2_unique_splices": uniq["spliced"],
            "K2_holds": uniq["spliced"] == 0,
            "K3_shippable": [r["rule"] for r in best],
            "K3_holds": bool(best)}
        print("\n" + json.dumps(res["verdict"], indent=1, ensure_ascii=False))

        # what the winner actually looks like while typing
        rule = (("dominance", 2.0) if not best else
                next(r for r in RULES if
                     f"{r[0]}{('' if not r[1] else ' ' + str(r[1]))}"
                     == best[-1]["rule"]))
        shown = []
        for i in spots:
            g = ghost(mem, ids_new[:i], rule=rule)
            if len(g) >= 2 and len(shown) < 8:
                shown.append({"typed": tok.decode(ids_new[max(0, i - 12):i]),
                              "ghost": tok.decode(g),
                              "correct": g == list(ids_new[i:i + len(g)])})
        res["samples"] = shown
        print(f"\nwith {rule}:", flush=True)
        for smp in shown:
            print(f"  {'OK ' if smp['correct'] else '.. '}"
                  f"...{smp['typed']!r} -> {smp['ghost']!r}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "ghost3.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
