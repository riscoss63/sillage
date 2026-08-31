"""Does the memory drafter still pay on text it has never read?

This is the last standing product direction, and the only one where the
failure mode is provably nil: speculative decoding verifies every token
against the frozen model, so `complete --fast` returns output identical
to plain decoding BY CONSTRUCTION. There is nothing to be wrong about --
only speed to gain or not gain.

Paper 5 measured 75-81% acceptance and x1.63-1.98, but on text the
memory had read. A server does not get that: it gets whatever the user
sends. If acceptance collapses off-corpus, the direction dies with it
and the honest answer becomes "leave the project as it is".

Three regimes, one state, real prose (the repository's own papers):

  seen        prompts from what was read -- the papers' own case
  same-topic  a DIFFERENT paper, same author, same vocabulary -- the
              realistic warm case for anyone serving their own domain
  unrelated   prose with nothing in common -- the cold case

Reported per regime: acceptance, and TOKENS PER FORWARD, which is what
the drafter actually buys and is architecture-independent. Wall-clock on
this CPU understates a GPU badly (the papers measured the numpy readout,
not the forward, as the CPU bottleneck), so it is recorded separately
and must not be read as the headline.

Registered BEFORE the run:

  L1  seen: acceptance >= 60%, reproducing the papers' regime.
      FALSIFIED below 45%.
  L2  same-topic: acceptance >= 30%. This is the number that decides
      the direction.
      FALSIFIED below 15% -- a drafter that misses on a user's own
      domain cannot be sold as one.
  L3  unrelated: the fast path costs at most 10% more wall-clock than
      plain. A drafter that misses must be free, not expensive.
      FALSIFIED above 25%.
  L4  Output is identical to plain decoding in every regime. This is
      the whole basis of the direction, so a single mismatch sinks it.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sillage.index import strip_latex                # noqa: E402
from sillage.runtime import Sillage                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_GEN = 24


def paper(name):
    p = os.path.join(ROOT, "papers", name, name + ".tex")
    return strip_latex(io.open(p, encoding="utf-8", errors="replace").read())


UNRELATED = """The tide reaches the foot of the cliff twice a day and
withdraws again, leaving a band of wet shingle that dries to grey by noon.
Gulls work the line methodically, turning stones with their bills. In
autumn the light comes in low from the west and the whole beach takes on
the colour of weak tea. Fishermen used to mend nets here, sitting on
upturned crates, and the older houses still have hooks set into their
gable ends for drying them. The path up to the village is cut into the
chalk and turns twice, steeply, before it reaches the churchyard wall.
"""


def prompts_from(text, tok, k=8, span=40):
    """k prompts of `span` tokens, spread across the text."""
    ids = tok.encode(text)
    out = []
    if len(ids) < span + 60:
        return out
    step = max(1, (len(ids) - span - 40) // k)
    for j in range(k):
        a = 20 + j * step
        if a + span >= len(ids):
            break
        out.append(tok.decode(ids[a:a + span]))
    return out


def measure(s, prompts, label):
    acc = drafted = fwd = toks = 0
    t_fast = t_plain = 0.0
    identical = 0
    for p in prompts:
        t0 = time.time()
        fast = s.complete(p, n=N_GEN, fast=True)
        t_fast += time.time() - t0
        st = s.mem.last_draft_stats if hasattr(s.mem, "last_draft_stats") \
            else None
        t0 = time.time()
        plain = s.complete(p, n=N_GEN)
        t_plain += time.time() - t0
        identical += fast == plain
        _ = st
    row = {"prompts": len(prompts), "identical": identical,
           "seconds_fast": round(t_fast, 1),
           "seconds_plain": round(t_plain, 1),
           "wall_speedup": round(t_plain / max(1e-9, t_fast), 2)}
    print(f"  {label:<12} identical {identical}/{len(prompts)}  "
          f"wall x{row['wall_speedup']}  "
          f"({t_plain:.0f}s plain vs {t_fast:.0f}s fast)", flush=True)
    return row


def main():
    from sillage.drafting import complete_fast

    tmp = tempfile.mkdtemp(prefix="drafter_")
    res = {"n_gen": N_GEN}
    try:
        s = Sillage(model="gpt2", state=tmp, quiet=True)
        s.load_model()
        tok = s.load_tokenizer()

        read_me = paper("sillage") + "\n\n" + paper("fastweights")
        print("reading two papers ...", flush=True)
        rec = s.read_text(read_me, "read")
        print(f"  {rec['tokens']} tokens, {len(s.mem.cold)} grams", flush=True)
        res["state"] = {"tokens": rec["tokens"], "grams": len(s.mem.cold)}

        sets = {"seen": prompts_from(read_me, tok),
                "same-topic": prompts_from(paper("behavior"), tok),
                "unrelated": prompts_from(UNRELATED, tok, k=6)}

        print(f"\n{'regime':<12} {'accept':>8} {'tok/fwd':>9} "
              f"{'identical':>10}", flush=True)
        for label, prompts in sets.items():
            a = d = f = t = 0
            same = 0
            tf = tp = 0.0
            for p in prompts:
                t0 = time.time()
                text, st = complete_fast(s, p, n=N_GEN)
                tf += time.time() - t0
                a += st["accepted"]
                d += st["drafted"]
                f += st["forwards"]
                t += st["tokens"]
                t0 = time.time()
                plain = s.complete(p, n=N_GEN)
                tp += time.time() - t0
                same += text == plain
            row = {"prompts": len(prompts),
                   "accepted": a, "drafted": d,
                   "acceptance": round(a / max(1, d), 3),
                   "tokens_per_forward": round(t / max(1, f), 3),
                   "identical": same,
                   "seconds_fast": round(tf, 1),
                   "seconds_plain": round(tp, 1),
                   "wall_ratio": round(tf / max(1e-9, tp), 3)}
            res[label] = row
            print(f"{label:<12} {row['acceptance']:>7.0%} "
                  f"{row['tokens_per_forward']:>9.2f} "
                  f"{same:>7}/{len(prompts)}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    seen, warm, cold = res["seen"], res["same-topic"], res["unrelated"]
    res["verdict"] = {
        "L1_seen": seen["acceptance"], "L1_holds": seen["acceptance"] >= 0.45,
        "L2_same_topic": warm["acceptance"],
        "L2_holds": warm["acceptance"] >= 0.15,
        "L3_cold_wall_overhead": round(cold["wall_ratio"] - 1, 3),
        "L3_holds": cold["wall_ratio"] <= 1.25,
        "L4_identical": all(res[k]["identical"] == res[k]["prompts"]
                            for k in ("seen", "same-topic", "unrelated")),
        "note": "wall-clock on CPU understates a GPU: the papers measured "
                "the numpy readout, not the forward, as the CPU "
                "bottleneck. tokens_per_forward is the transferable number."}
    print("\n" + json.dumps(res["verdict"], indent=1, ensure_ascii=False))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "drafter_real.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
