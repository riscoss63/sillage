"""What happens when the cold store is full: who gets dropped?

`probe_capacity` ran to a million tokens and never lost a fact -- but it
also never triggered eviction, because the cap is enforced ONLY in
`save()`, and an ingest loop that does not persist lets the store grow
28% past it. So the headline "no degradation to 1M tokens" is true and
incomplete: it measures a store that was never pruned.

The pruning rule is one line: keep the COLD_MAX grams of highest
surprise mass. Invented facts are, by construction, the most surprising
thing in the corpus, so the rule predicts they survive while ordinary
filler is dropped. That is the "it forgets the boring parts first"
property, and it has never been measured.

This tests the MECHANISM rather than a scale: COLD_MAX is lowered so the
store saturates in minutes instead of an hour.

Registered BEFORE the run:

  E1  After save(), the store holds exactly COLD_MAX grams.
      FALSIFIED otherwise.
  E2  Planted facts survive: >= 90% of their probe grams are still in
      the store after eviction, though a large share of the store was
      dropped.
      FALSIFIED below 75% -- the surprise ranking would then not be
      protecting what matters.
  E3  Recall after eviction stays >= 80% of recall before.
      FALSIFIED below that.
  E4  The surviving store is more surprising than the one that went in:
      its median mass rises.
      FALSIFIED if it falls or does not move.

Run:  python behav/probe_eviction.py [--cap 3000]
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

import sillage.core as core                              # noqa: E402
from sillage.runtime import Sillage                      # noqa: E402
from sillage.ingest import ingest_text                   # noqa: E402
from probe_capacity import (block, entities, value_for,   # noqa: E402
                            recall, grams_present, mass_quantiles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--over", type=float, default=1.6,
                    help="fill to this multiple of the cap before saving")
    a = ap.parse_args()

    real_cap = core.COLD_MAX
    core.COLD_MAX = a.cap
    tmp = tempfile.mkdtemp(prefix="evict_")
    res = {"cap": a.cap, "shipped_cap": real_cap, "model": a.model}
    try:
        s = Sillage(model=a.model, state=tmp, quiet=True)
        s.load_model()
        tok = s.load_tokenizer()
        ents = entities(4000, tok)
        planted, tokens, seed, fi = [], 0, 0, 0
        while len(s.mem.cold) < a.cap * a.over:
            facts = [(ents[fi + j], value_for(fi + j)) for j in range(6)]
            fi += 6
            tokens += ingest_text(s, block(seed, facts), f"b{seed}",
                                  quiet=True)["tokens"]
            planted.append(facts)
            seed += 1
        old, new = planted[0], planted[-1]
        allf = [p for fs in planted for p in fs]
        before = {"grams": len(s.mem.cold), "tokens": tokens,
                  "mass_q": mass_quantiles(s.mem),
                  "recall_oldest": recall(s, old),
                  "recall_newest": recall(s, new),
                  "planted_grams_present": grams_present(s, tok, allf)}
        print(f"  before save: {before['grams']} grams "
              f"({before['grams'] / a.cap:.2f}x the cap) from "
              f"{tokens} tokens, {len(allf)} facts planted", flush=True)
        print(f"    mass quantiles (10/50/90) {before['mass_q']} | "
              f"oldest {before['recall_oldest']:.0%} newest "
              f"{before['recall_newest']:.0%} | planted grams "
              f"{before['planted_grams_present']:.0%}", flush=True)

        s.save()                                   # <- eviction happens here
        s2 = Sillage(model=a.model, state=tmp, quiet=True)
        s2.load_model()
        after = {"grams": len(s2.mem.cold),
                 "mass_q": mass_quantiles(s2.mem),
                 "recall_oldest": recall(s2, old),
                 "recall_newest": recall(s2, new),
                 "planted_grams_present": grams_present(s2, tok, allf)}
        dropped = before["grams"] - after["grams"]
        print(f"  after save : {after['grams']} grams "
              f"({dropped} dropped, {dropped / before['grams']:.0%})",
              flush=True)
        print(f"    mass quantiles (10/50/90) {after['mass_q']} | "
              f"oldest {after['recall_oldest']:.0%} newest "
              f"{after['recall_newest']:.0%} | planted grams "
              f"{after['planted_grams_present']:.0%}", flush=True)
        res.update({"before": before, "after": after, "dropped": dropped})
        res["verdict"] = {
            "E1": {"grams_after": after["grams"], "cap": a.cap,
                   "holds": after["grams"] == a.cap},
            "E2": {"planted_grams_after": after["planted_grams_present"],
                   "share_of_store_dropped": round(dropped / before["grams"],
                                                   3),
                   "holds": after["planted_grams_present"] >= 0.75},
            "E3": {"before": before["recall_oldest"],
                   "after": after["recall_oldest"],
                   "holds": after["recall_oldest"]
                   >= 0.8 * max(before["recall_oldest"], 1e-9)},
            "E4": {"median_before": before["mass_q"][1],
                   "median_after": after["mass_q"][1],
                   "holds": after["mass_q"][1] > before["mass_q"][1]}}
        print("\n" + json.dumps(res["verdict"], indent=1, ensure_ascii=False))
    finally:
        core.COLD_MAX = real_cap
        shutil.rmtree(tmp, ignore_errors=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "eviction.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
