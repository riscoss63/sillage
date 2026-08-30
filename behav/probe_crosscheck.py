"""Can the lexical channel catch what the generative one gets wrong?

The residual failure has a name: the TRANSPLANT. `La prochaine visite
du rucher aura lieu le` returns `11 avril 2026, par temps couvert...`
-- the visit that already happened, recited verbatim, with 16 of 30
tokens moved, where the best CORRECT answer of that corpus moves 12.
No threshold on contribution separates them.

The tool holds a second, independent channel: `ask`, a TF-IDF index
that knows which passage is lexically relevant. The idea under test is
that the two can check each other -- flag an answer when the passage
the memory recited is not the passage the index would have chosen.

Three candidate signals, measured on the same 24 questions:

  A  does `ask` abstain (no passage above its floor)?
  B  is the generated text actually PRESENT in the document, verbatim?
  C  do the two channels agree -- is the paragraph the completion came
     from the one `ask` ranks first?

Registered BEFORE the run. The honest expectation is written down
too, because it is not favourable:

  Z1  Signal C flags the transplant: the paragraph the completion
      recites is NOT the one `ask` ranks first.
      EXPECTED TO FAIL. `La prochaine visite du rucher de Peyrelonge
      aura lieu le` shares `visite`, `rucher`, `Peyrelonge` with the
      paragraph it wrongly recites; only `prochaine` distinguishes
      them, and that word is nowhere in the document. If the two
      channels agree on the wrong passage, C is worthless for this
      class and that has to be said.
  Z2  Signal A flags it: `ask` abstains on the unanswerable questions.
      FALSIFIED if `ask` returns a passage above its floor for the
      transplant.
  Z3  Signal B flags it.
      EXPECTED TO FAIL: the transplant IS verbatim from the document.
      Recorded to close the option explicitly.
  Z4  Whatever survives must not fire on correct answers: at most 1 of
      the 9 correct answers is flagged.

Run:  python behav/probe_crosscheck.py
"""
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sillage.runtime import Sillage                      # noqa: E402
from sillage.cli import FAINT                            # noqa: E402
from probe_abstain_gen import (DOC, VERBATIM, REWORDED,   # noqa: E402
                               UNANSWERABLE, N)


def para_of(text, needle):
    """Which paragraph of the document does this text come from, if any."""
    if len(needle) < 12:
        return None
    for i, p in enumerate(text.split("\n\n")):
        if needle in p:
            return i
    return None


def main():
    tmp = tempfile.mkdtemp(prefix="cross_")
    rows = []
    try:
        s = Sillage(model="qwen", state=tmp, quiet=True)
        text = Sillage.reflow(DOC)
        for _ in range(2):
            s.read_text(text)
        s.index.add(text, "rucher.md")

        items = ([(p, w, "verbatim") for p, w in VERBATIM]
                 + [(p, w, "reworded") for p, w in REWORDED]
                 + [(p, None, "no-answer") for p in UNANSWERABLE])
        for prompt, want, kind in items:
            out = s.complete(prompt, n=N, temp=0.0)
            at = s.attribution() or {}
            moved = at.get("moved") or 0
            hits = s.ask(prompt, k=3)
            top_score = hits[0][0] if hits else 0.0
            top_para = (para_of(text, hits[0][1]["text"][:60])
                        if hits else None)
            gen_para = para_of(text, out.strip()[:40])
            row = {"kind": kind, "prompt": prompt[-44:], "want": want,
                   "got": out.strip()[:44], "moved": moved,
                   "spoke": moved >= FAINT,
                   "correct": bool(want and want.lower() in out.lower()),
                   "ask_score": round(top_score, 4),
                   "ask_abstains": not hits,
                   "verbatim_in_doc": gen_para is not None,
                   "gen_para": gen_para, "ask_para": top_para,
                   "channels_agree": (gen_para is not None
                                      and gen_para == top_para)}
            rows.append(row)
            print(f"  [{kind:<9}] moved {moved:>2} "
                  f"{'SPOKE  ' if row['spoke'] else 'abstain'} "
                  f"{'OK' if row['correct'] else '  '} | ask "
                  f"{top_score:.3f} para {str(top_para):<4} | gen para "
                  f"{str(gen_para):<4} | agree "
                  f"{str(row['channels_agree']):<5} | {out.strip()[:32]!r}",
                  flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    spoke = [r for r in rows if r["spoke"]]
    bad = [r for r in spoke if r["kind"] == "no-answer"]
    good = [r for r in spoke if r["correct"]]
    verdict = {
        "spoke_total": len(spoke), "correct_when_spoke": len(good),
        "transplants": [{"prompt": r["prompt"], "got": r["got"],
                         "moved": r["moved"], "ask_score": r["ask_score"],
                         "agree": r["channels_agree"],
                         "verbatim": r["verbatim_in_doc"]} for r in bad],
        "Z1_C_flags_transplant": all(not r["channels_agree"] for r in bad)
        if bad else None,
        "Z2_A_flags_transplant": all(r["ask_abstains"] for r in bad)
        if bad else None,
        "Z3_B_flags_transplant": all(not r["verbatim_in_doc"] for r in bad)
        if bad else None,
        "Z4_false_alarms": {
            "C_on_correct": sum(1 for r in good if not r["channels_agree"]),
            "A_on_correct": sum(1 for r in good if r["ask_abstains"]),
            "B_on_correct": sum(1 for r in good
                                if not r["verbatim_in_doc"]),
            "of": len(good)},
        "ask_score_ranges": {
            "correct": [min((r["ask_score"] for r in good), default=None),
                        max((r["ask_score"] for r in good), default=None)],
            "transplant": [r["ask_score"] for r in bad]}}
    print("\n" + json.dumps(verdict, indent=1, ensure_ascii=False))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "crosscheck.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": verdict}, fh, indent=1,
                  ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
