"""Does the surprise gate earn its place where it should — under pressure?

`probe_which_surprise` found the uniform control matching Shannon on
second-pass perplexity (2.62 vs 2.65) and recall (100% both). That was
measured in the regime where the gate has the LEAST to do: the same
document re-read, no capacity pressure, no interference.

The gate's job, on this project's own account, is elsewhere. It sets
`slot[0]`, the surprise mass, and eviction keeps the highest-mass grams
(paper 3). It scales the amplitude write `sqrt(a^2 + g) - a`, so it
decides which traces survive the sqrt(N) noise of everything written
afterwards (paper 1). Both only bite under pressure.

So: plant facts, then bury them.

  target        a document with three invented facts, read twice
  interference  ~4x more unrelated text, read once
  pressure      a small cold store, so eviction has to choose

Recall is then measured twice per arm -- once with the whole memory, and
once with the cold store emptied, which isolates the MATRIX and is where
paper 1's claim lives alone.

Registered BEFORE the run:

  O1  Under capacity pressure, Shannon retains the planted facts better
      than uniform. FALSIFIED if uniform ties or wins -- and that would
      be a real finding about paper 1's central mechanism, in the one
      regime where it was supposed to matter.
  O2  With the cold store emptied, Shannon still beats uniform on
      matrix-only recall. This is paper 1's amplitude claim in
      isolation.
  O3  Bayes stays worst, for the reason already measured: its gate is
      zero on 98% of tokens.
  O4  Shannon's surviving grams carry more mass than uniform's -- the
      mechanism, not just the outcome.

Run:  python behav/probe_gate_pressure.py [--cold-max 1200]
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

from sillage.index import strip_latex                # noqa: E402
from sillage.runtime import Sillage                  # noqa: E402
from probe_which_surprise import (FACTS, PROBES,      # noqa: E402
                                  ROOT, read_gated)


def paper_body(name, chars):
    p = os.path.join(ROOT, "papers", name, name + ".tex")
    return strip_latex(io.open(p, encoding="utf-8",
                               errors="replace").read())[:chars]


def recall_of(s):
    hit = []
    for prompt, want in PROBES:
        out = s.complete(prompt, n=8)
        hit.append(bool(want.lower() in out.lower()))
    return sum(hit) / len(hit)


def fact_gram_mass(s):
    """Surprise mass carried by the grams that answer the probes."""
    import sillage.core as core
    tok = s.load_tokenizer()
    masses = []
    for prompt, _w in PROBES:
        ids = tok.encode(prompt)[-core.NGRAM:]
        g = np.array(ids, dtype=np.int32).tobytes()
        slot = s.mem.cold.get(g)
        masses.append(float(slot[0]) if slot else 0.0)
    return masses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--cold-max", type=int, default=1200)
    a = ap.parse_args()

    target = paper_body("sillage", 5000) + FACTS
    interference = (paper_body("behavior", 9000) + "\n\n"
                    + paper_body("benchmark", 9000))
    res = {"model": a.model, "cold_max": a.cold_max, "arms": {}}
    print(f"cold store capped at {a.cold_max} grams so eviction has to "
          f"choose\n", flush=True)

    for gate in ("shannon", "bayes", "uniform"):
        tmp = tempfile.mkdtemp(prefix="pressure_")
        try:
            s = Sillage(model=a.model, state=tmp, quiet=True,
                        fastweights=False, cold_max=a.cold_max)
            s.load_model()
            for _ in range(2):
                read_gated(s, target, gate)
            before = {"recall": recall_of(s),
                      "grams": len(s.mem.cold),
                      "fact_mass": fact_gram_mass(s)}
            inter = read_gated(s, interference, gate)
            s.mem.prune_cold()
            after_full = recall_of(s)
            after_mass = fact_gram_mass(s)
            grams_alive = sum(1 for m in after_mass if m > 0)
            cold_backup = s.mem.cold
            s.mem.cold = {}                      # paper 1 alone
            after_matrix = recall_of(s)
            s.mem.cold = cold_backup
            row = {"interference_tokens": inter["tokens"],
                   "recall_before": before["recall"],
                   "recall_after": after_full,
                   "recall_after_matrix_only": after_matrix,
                   "grams_after": len(s.mem.cold),
                   "fact_grams_alive": grams_alive,
                   "fact_mass_before": [round(m, 2)
                                        for m in before["fact_mass"]],
                   "fact_mass_after": [round(m, 2) for m in after_mass],
                   "mass_median_store": round(float(np.median(
                       [sl[0] for sl in s.mem.cold.values()])), 3)}
            res["arms"][gate] = row
            print(f"  {gate:<9} recall {row['recall_before']:.0%} -> "
                  f"{row['recall_after']:.0%}  (matrix only "
                  f"{row['recall_after_matrix_only']:.0%})  "
                  f"fact grams alive {grams_alive}/3  "
                  f"their mass {row['fact_mass_after']}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    sh, un, by = (res["arms"]["shannon"], res["arms"]["uniform"],
                  res["arms"]["bayes"])
    res["verdict"] = {
        "O1_shannon_vs_uniform_after":
            [sh["recall_after"], un["recall_after"]],
        "O1_holds": sh["recall_after"] > un["recall_after"],
        "O2_matrix_only":
            [sh["recall_after_matrix_only"], un["recall_after_matrix_only"]],
        "O2_holds": sh["recall_after_matrix_only"]
        > un["recall_after_matrix_only"],
        "O3_bayes_worst": by["recall_after"] <= min(sh["recall_after"],
                                                    un["recall_after"]),
        "O4_fact_grams_alive": {"shannon": sh["fact_grams_alive"],
                                "uniform": un["fact_grams_alive"],
                                "bayes": by["fact_grams_alive"]},
        "O4_store_median_mass": {"shannon": sh["mass_median_store"],
                                 "uniform": un["mass_median_store"],
                                 "bayes": by["mass_median_store"]}}
    print("\n" + json.dumps(res["verdict"], indent=1, ensure_ascii=False))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "gate_pressure.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
