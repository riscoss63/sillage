"""Does a bigger model bridge a rephrased question back to the document?

On the fresh corpus at 0.6B, 2 of 8 rephrased questions were answered,
both correctly -- and both answers began by RECONSTRUCTING the
document's own sentence before the memory took over and finished it
verbatim:

    "Qui a redige ce compte rendu ?"
      -> "Le compte rendu a ete redige par le technicien du groupement,
          monsieur Ovide Trenchard, carte apicole"     (16/30 moved)

The six failures instead wandered off into meta-questions ("Et quels
sont les principaux facteurs qui..."). That suggests the memory never
crosses the gap between a question and the document's surface: the
MODEL has to build that bridge, and a 0.6B does not know how to answer
in a document's register. If so, capacity -- not the readout, not the
keys -- is what buys rephrased recall.

Four arms, one document, same questions: 0.6B and 1.7B, each under the
published readout and paper 5's family settings. The 0.6B family arm is
the control that separates "bigger model" from "louder readout".

Registered BEFORE the run:

  P1  1.7B + family answers >= 5 of 8 rephrased questions (0.6B
      published answered 2).
      FALSIFIED below 5/8.
  P2  MECHANISM: among rephrased questions answered correctly, the
      completion's opening is text FROM THE DOCUMENT -- the model
      reconstructs the phrasing, the memory finishes it. At least 3 of
      4 correct rephrased answers show it.
      FALSIFIED if correct answers do not open on document text.
  P3  CONTROL: 0.6B + family does NOT reach 5/8 on rephrased questions.
      FALSIFIED if it does -- that would mean the readout was the whole
      story and capacity is irrelevant, and P1's gain would be
      misattributed.
  P4  COST: family raises intrusion on unanswerable questions -- more
      of them move >= 3 tokens than under published, at both
      capacities.
      Recorded either way; this is the price to publish next to any gain.
  P5  NO REGRESSION: verbatim recall stays >= 7/8 in all four arms.
      FALSIFIED if a louder readout or a bigger model costs what
      already works.

Run:  python behav/probe_bridge.py
"""
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sillage.runtime import Sillage                      # noqa: E402
from sillage.cli import FAMILY_READOUT, FAINT            # noqa: E402
from probe_abstain_gen import (DOC, VERBATIM, REWORDED,   # noqa: E402
                               UNANSWERABLE, N)

MODELS = [(None, "0.6B"), ("Qwen/Qwen3-1.7B", "1.7B")]
OPEN_CHARS = 22          # how much of an answer's opening must be verbatim


def bridges(text, out):
    """Does the answer OPEN on the document's own words?"""
    head = out.strip()[:OPEN_CHARS]
    return len(head) >= 12 and head in text


def ask_all(s, text, label):
    rows = []
    for prompt, want, kind in (
            [(p, w, "verbatim") for p, w in VERBATIM]
            + [(p, w, "reworded") for p, w in REWORDED]
            + [(p, None, "no-answer") for p in UNANSWERABLE]):
        out = s.complete(prompt, n=N, temp=0.0)
        at = s.attribution() or {}
        moved = at.get("moved") or 0
        rows.append({"kind": kind, "want": want, "moved": moved,
                     "spoke": moved >= FAINT,
                     "correct": bool(want and want.lower() in out.lower()),
                     "bridges": bridges(text, out),
                     "got": out.strip()[:56]})
    for k in ("verbatim", "reworded", "no-answer"):
        sel = [r for r in rows if r["kind"] == k]
        if k == "no-answer":
            print(f"    {k:<9} intrudes {sum(r['spoke'] for r in sel)}/8",
                  flush=True)
        else:
            print(f"    {k:<9} {sum(r['correct'] for r in sel)}/8 correct"
                  f"  (bridged: "
                  f"{sum(r['bridges'] for r in sel if r['correct'])})",
                  flush=True)
    return rows


def main():
    res = {"arms": {}}
    text = Sillage.reflow(DOC)
    for hub, cap in MODELS:
        tmp = tempfile.mkdtemp(prefix="bridge_")
        try:
            print(f"\n=== {cap}: reading ===", flush=True)
            s = Sillage(model="qwen", state=tmp, target=hub, quiet=True)
            for _ in range(2):
                rec = s.read_text(text)
            print(f"  {rec['tokens']} tokens, {len(s.mem.cold)} grams, "
                  f"ppl {rec['ppl_frozen']} -> {rec['ppl_with_memory']}",
                  flush=True)
            pub = (float(s.mem.beta_G), float(s.mem.lam_G),
                   float(s.mem.thr_qG))
            for triple, ro in ((pub, "published"), (FAMILY_READOUT, "family")):
                s.mem.beta_G, s.mem.lam_G, s.mem.thr_qG = triple
                key = f"{cap} {ro}"
                print(f"  -- {key} {triple}", flush=True)
                res["arms"][key] = ask_all(s, text, key)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def n_of(key, kind, field):
        return sum(r[field] for r in res["arms"][key] if r["kind"] == kind)

    rew17f = n_of("1.7B family", "reworded", "correct")
    rew06f = n_of("0.6B family", "reworded", "correct")
    ok_rew = [r for r in res["arms"]["1.7B family"]
              if r["kind"] == "reworded" and r["correct"]]
    res["verdict"] = {
        "P1": {"reworded_17B_family": rew17f, "holds": rew17f >= 5},
        "P2": {"bridged_of_correct": sum(r["bridges"] for r in ok_rew),
               "correct": len(ok_rew),
               "holds": len(ok_rew) >= 3
               and sum(r["bridges"] for r in ok_rew) >= 3},
        "P3": {"reworded_06B_family": rew06f, "holds": rew06f < 5},
        "P4": {k: n_of(k, "no-answer", "spoke") for k in res["arms"]},
        "P5": {k: n_of(k, "verbatim", "correct") for k in res["arms"]},
        "reworded_all": {k: n_of(k, "reworded", "correct")
                         for k in res["arms"]}}
    print("\n" + json.dumps(res["verdict"], indent=1, ensure_ascii=False))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "bridge.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
