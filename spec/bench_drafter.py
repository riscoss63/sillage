"""Benchmark: does the Sillage state pay for itself in tokens per second?

Three decoders, identical outputs guaranteed (greedy, same augmented target):
  plain          one forward per token (the reference)
  spec:sillage   drafts from the persistent memory (cold store + M_G)
  spec:pld       drafts by prompt-lookup in the current context only
                 (the no-persistence ablation)

Two regimes:
  seen    prompts sampled from the corpus the memory has read
          (papers_state: the four preprints, 21.8k tokens lifetime)
  unseen  prompts the memory has never read (expected: ~no acceptance,
          measured overhead -- the honest control)

Usage:  python bench_drafter.py [--n 48] [--gamma 6] [--prompts 12]
"""

import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sillage_drafter import (PromptLookupDrafter, SillageDrafter,  # noqa
                             SpeculativeSillage, timed)

REPO = os.path.dirname(HERE)
STATE = os.path.join(REPO, "papers_state", "memory")
CORPUS = os.path.join(REPO, "papers_state", "corpus.txt")

UNSEEN = [
    "The city council met on Thursday to discuss the new bicycle lanes",
    "In the morning she poured coffee, opened the window, and watched the",
    "The recipe calls for two cups of flour, a pinch of salt, and",
    "Financial markets rallied yesterday after the central bank announced",
    "The hikers followed the narrow trail along the ridge until the",
]


def seen_prompts(k, words_per=12, seed=0, truth_words=60):
    """(prompt, true continuation) pairs sampled from the read corpus."""
    text = open(CORPUS, encoding="utf-8", errors="replace").read()
    words = [w for w in text.split() if w]
    rng = random.Random(seed)
    lo = int(0.15 * len(words))
    hi = int(0.95 * len(words)) - words_per - truth_words
    out = []
    while len(out) < k:
        i = rng.randrange(lo, hi)
        seg = " ".join(words[i:i + words_per])
        if re.search(r"[^\x20-\x7e]", seg):        # keep clean ascii segments
            continue
        truth = " ".join(words[i + words_per:i + words_per + truth_words])
        out.append((seg, truth))
    return out


def prefix_agreement(generated_text, truth_text):
    """How many leading words of the generation match the source document."""
    g, t = generated_text.split(), truth_text.split()
    n = 0
    for a, b in zip(g, t):
        if a != b:
            break
        n += 1
    return n


def run_method(engine, name, prompts, n, gamma, thr_q):
    total = {"seconds": 0.0, "tokens": 0, "forwards": 0,
             "drafted": 0, "accepted": 0, "rounds": 0,
             "drafted_cold": 0, "acc_cold": 0, "drafted_mg": 0, "acc_mg": 0}
    outputs = []
    for p, _truth in prompts:
        if name == "plain":
            out, st = timed(engine.generate_plain, p, n=n)
        elif name == "spec:sillage":
            out, st = timed(engine.generate_spec, p, SillageDrafter,
                            n=n, gamma=gamma, thr_q=thr_q)
        else:
            out, st = timed(engine.generate_spec, p, PromptLookupDrafter,
                            n=n, gamma=gamma)
        outputs.append(out)
        for key in total:
            total[key] += st.get(key, 0)
    total["tok_per_s"] = total["tokens"] / max(1e-9, total["seconds"])
    total["forwards_per_token"] = total["forwards"] / max(1, total["tokens"])
    if total["drafted"]:
        total["acceptance"] = total["accepted"] / total["drafted"]
    return outputs, total


def bench(regime, prompts, engine, n, gamma, thr_q):
    print(f"\n=== regime: {regime} ({len(prompts)} prompts, "
          f"{n} tokens each) ===")
    results = {}
    ref_out, ref = run_method(engine, "plain", prompts, n, gamma, thr_q)
    results["plain"] = ref
    agr = [prefix_agreement(engine.tok.decode(o), t)
           for o, (_, t) in zip(ref_out, prompts) if t]
    if agr:
        results["verbatim_prefix_words"] = agr
        print(f"  (recall context: the augmented model regenerates the "
              f"source verbatim for {sum(agr)/len(agr):.1f} leading words "
              f"on average; per prompt {sorted(agr, reverse=True)})")
    print(f"  plain        : {ref['tok_per_s']:6.2f} tok/s   "
          f"({ref['forwards']} forwards)")
    for name in ("spec:sillage", "spec:pld"):
        outs, st = run_method(engine, name, prompts, n, gamma, thr_q)
        st["identical_outputs"] = sum(a == b for a, b in zip(outs, ref_out))
        st["speedup"] = st["tok_per_s"] / ref["tok_per_s"]
        results[name] = st
        print(f"  {name:13s}: {st['tok_per_s']:6.2f} tok/s   "
              f"x{st['speedup']:.2f}   acc "
              f"{st.get('acceptance', 0):.0%} "
              f"({st['accepted']}/{st['drafted']})   "
              f"{st['forwards_per_token']:.2f} fwd/tok   "
              f"identical {st['identical_outputs']}/{len(prompts)}")
        if st.get("drafted_cold") or st.get("drafted_mg"):
            print(f"                 par source: cold "
                  f"{st['acc_cold']}/{st['drafted_cold']} "
                  f"({st['acc_cold']/max(1,st['drafted_cold']):.0%})   "
                  f"M_G {st['acc_mg']}/{st['drafted_mg']} "
                  f"({st['acc_mg']/max(1,st['drafted_mg']):.0%})")
    return results


def main():
    if not os.path.exists(STATE):
        raise SystemExit(
            "state not found (states are not shipped: a cold store would "
            "reveal what it read) -- build one:\n"
            "  sillage papers --with-memory")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--words-per", type=int, default=24,
                    help="prompt length in words (longer = stronger "
                         "conditioning = longer verbatim runs)")
    ap.add_argument("--thr-q", type=float, default=0.75,
                    help="draft-gate quantile for the sillage drafter")
    a = ap.parse_args()

    engine = SpeculativeSillage(STATE)
    print(f"state: {STATE}")
    print(f"model: {engine.mem.hub} | lifetime tokens: {engine.mem.tokens} | "
          f"cold grams: {len(engine.mem.cold)} | draft gate q{a.thr_q:.2f}")

    report = {"n": a.n, "gamma": a.gamma, "thr_q": a.thr_q,
              "model": engine.mem.hub, "lifetime_tokens": engine.mem.tokens}
    report["seen"] = bench("seen (the four read papers)",
                           seen_prompts(a.prompts, words_per=a.words_per),
                           engine, a.n, a.gamma, a.thr_q)
    report["unseen"] = bench("unseen (control)",
                             [(p, None) for p in UNSEEN], engine,
                             a.n, a.gamma, a.thr_q)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", "bench_gpt2_papers.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
