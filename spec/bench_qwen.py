"""Cross-model speculation: a 0.6B reader's memory drafting for a 1.7B target.

The question of the session: the Sillage state was written by Qwen3-0.6B
while reading two manuscripts (16.9k tokens, PPL 10.68 -> 5.73 on the second
read). Qwen3-1.7B shares the tokenizer. Three configurations:

  A  0.6B self, memory in target      the within-model reference
  B  1.7B target, VANILLA             pure acceleration: the memory only
                                      drafts; the output is the untouched
                                      1.7B greedy continuation
  C  1.7B target, memory in target    the product configuration: the small
                                      reader's memory both improves and
                                      accelerates the bigger model
                                      (readout settings inherited from the
                                      0.6B tuning -- stated caveat)

The adapter (fast weights) is OFF for B and C: it was trained against the
0.6B's hidden geometry and cannot transfer.

Usage: python bench_qwen.py [--n 28] [--prompts 10] [--gamma 8]
"""

import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bench_drafter import UNSEEN, prefix_agreement          # noqa: E402
from sillage_drafter import (PromptLookupDrafter, SillageDrafter,  # noqa
                             SpeculativeSillage, timed)

REPO = os.path.dirname(HERE)
STATE = os.path.join(REPO, "memory_state")
# The Manuscripts stream is not redistributed: these two files exist next to
# the repository on the author's machine; point DOCS at your own documents
# to rerun the protocol on a stream your state has actually read.
DOCS = [p for p in
        (os.path.join(os.path.dirname(REPO),
                      "preprint_bhd_active_inference.txt"),
         os.path.join(os.path.dirname(REPO),
                      "PREPRINT_V2_1_BHD_ACTIVE_INFERENCE.md"))
        if os.path.exists(p)]
TARGET = "Qwen/Qwen3-1.7B"


def seen_prompts(k, words_per=24, seed=1, truth_words=60):
    text = "\n\n".join(open(p, encoding="utf-8", errors="replace").read()
                       for p in DOCS)
    words = [w for w in text.split() if w]
    rng = random.Random(seed)
    lo = int(0.05 * len(words))
    hi = int(0.97 * len(words)) - words_per - truth_words
    out = []
    while len(out) < k:
        i = rng.randrange(lo, hi)
        seg = " ".join(words[i:i + words_per])
        if re.search(r"[^\x20-\x7e]", seg):
            continue
        truth = " ".join(words[i + words_per:i + words_per + truth_words])
        out.append((seg, truth))
    return out


def run(engine, name, prompts, n, gamma, with_pld=False):
    res = {}
    ref_out = None
    methods = ["plain", "spec:sillage"] + (["spec:pld"] if with_pld else [])
    for meth in methods:
        total = {"seconds": 0.0, "tokens": 0, "forwards": 0, "drafted": 0,
                 "accepted": 0, "rounds": 0, "drafted_cold": 0,
                 "acc_cold": 0, "drafted_mg": 0, "acc_mg": 0}
        outs = []
        for p, _t in prompts:
            if meth == "plain":
                o, st = timed(engine.generate_plain, p, n=n)
            elif meth == "spec:sillage":
                o, st = timed(engine.generate_spec, p, SillageDrafter,
                              n=n, gamma=gamma)
            else:
                o, st = timed(engine.generate_spec, p, PromptLookupDrafter,
                              n=n, gamma=gamma)
            outs.append(o)
            for key in total:
                total[key] += st.get(key, 0)
        total["tok_per_s"] = total["tokens"] / max(1e-9, total["seconds"])
        if meth == "plain":
            ref_out = outs
            agr = [prefix_agreement(engine.tok.decode(o), t)
                   for o, (_, t) in zip(outs, prompts) if t]
            if agr:
                total["verbatim_prefix_words"] = agr
                print(f"    (rappel verbatim moyen "
                      f"{sum(agr)/len(agr):.1f} mots ; "
                      f"{sorted(agr, reverse=True)})")
            print(f"    plain        : {total['tok_per_s']:6.2f} tok/s")
        else:
            total["identical_outputs"] = sum(
                a == b for a, b in zip(outs, ref_out))
            total["speedup"] = (total["tok_per_s"]
                                / res["plain"]["tok_per_s"])
            acc = total["accepted"] / max(1, total["drafted"])
            print(f"    {meth:13s}: {total['tok_per_s']:6.2f} tok/s   "
                  f"x{total['speedup']:.2f}   acc {acc:.0%} "
                  f"({total['accepted']}/{total['drafted']})   "
                  f"identical {total['identical_outputs']}/{len(prompts)}")
            if total["drafted_cold"] or total["drafted_mg"]:
                print(f"                   par source: cold "
                      f"{total['acc_cold']}/{total['drafted_cold']}   "
                      f"M_G {total['acc_mg']}/{total['drafted_mg']}")
        res[meth] = total
    return res


def main():
    if not os.path.exists(STATE):
        raise SystemExit(
            "state not found (states are not shipped: a cold store would "
            "reveal what it read) -- build one:\n"
            "  sillage read doc1 doc2 --model qwen --state memory_state")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--prompts", type=int, default=10)
    a = ap.parse_args()

    if len(DOCS) < 2:
        raise SystemExit(
            "the papers' Manuscripts stream is not redistributed -- edit "
            "DOCS at the top of this script to point at two documents of "
            "your own that the state has read")
    seen = seen_prompts(a.prompts)
    unseen = [(p, None) for p in UNSEEN]
    # Record the state repo-relative: the report is committed under results/,
    # and an absolute path would publish the author's local directory layout.
    report = {"n": a.n, "gamma": a.gamma,
              "state": os.path.relpath(STATE, REPO), "target": TARGET}

    print("=== A. Qwen3-0.6B, cible augmentee (reference intra-modele) ===")
    eng = SpeculativeSillage(STATE)
    print(f"  memoire: {eng.mem.hub}, {eng.mem.tokens} tokens a vie, "
          f"{len(eng.mem.cold)} grams froids, "
          f"fastweights={eng.mem.fastweights}")
    print("  -- seen --")
    report["A_seen"] = run(eng, "A", seen, a.n, a.gamma, with_pld=True)
    print("  -- unseen --")
    report["A_unseen"] = run(eng, "A", unseen, a.n, a.gamma)
    del eng

    print("\n=== B. cible Qwen3-1.7B VANILLA (acceleration pure) ===")
    eng = SpeculativeSillage(STATE, target_hub=TARGET,
                             memory_in_target=False, fastweights=False)
    print("  -- seen --")
    report["B_seen"] = run(eng, "B", seen, a.n, a.gamma, with_pld=True)
    print("  -- unseen --")
    report["B_unseen"] = run(eng, "B", unseen, a.n, a.gamma)

    print("\n=== C. cible Qwen3-1.7B + memoire du 0.6B (config produit) ===")
    eng.memory_in_target = True
    print("  -- seen --")
    report["C_seen"] = run(eng, "C", seen, a.n, a.gamma, with_pld=True)
    print("  -- unseen --")
    report["C_unseen"] = run(eng, "C", unseen, a.n, a.gamma)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", "bench_qwen_cross.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
