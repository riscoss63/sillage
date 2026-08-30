"""If the memory reads reflowed text, does it answer questions people type?

`probe_tokenkey` isolated the cause of `complete`'s inventions: both
tiers key on the last four TOKENS, a line break is absorbed into a
token (`',\\n'`), and a question typed on one line therefore forms a key
the document never wrote. The memory then says NOTHING, and the frozen
model fills the silence with a plausible fabrication.

The fix this probe tests is the cheapest one available: join the lines
inside each paragraph BEFORE reading, so the stored keys have the shape
a person's question will have. Nothing else changes -- same document,
same readout, same questions.

Registered BEFORE the run:

  T1  Reflowed reading recalls strictly MORE facts than as-is reading,
      with the questions typed the way a person types them.
      FALSIFIED if reflow <= as-is.
  T2  The fact whose prefix the source wraps (`Brindas`) is recalled
      after reflow and missed before.
      FALSIFIED if it still misses after reflow.
  T3  No fact recalled in the as-is arm is lost in the reflow arm.
      FALSIFIED by any regression.
  T4  Locality is untouched: |witness(reflow) - witness(as-is)| < 0.2
      nat. Reflow changes WHICH keys exist, not how loudly they speak.
  T5  Invention on unanswerable questions does not get worse: the
      reflow arm moves no more tokens there than the as-is arm.
      Recorded either way.
  T6  Reading perplexity is NOT comparable across arms (the token
      stream differs). Recorded, not predicted -- it is the reason
      reflow cannot silently become the default.

Run:  python behav/probe_reflow.py [--target HUB]
"""
import argparse
import io
import json
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sillage.runtime import Sillage                        # noqa: E402
from probe_readout_dial import (DOC, WITNESS, ANSWERABLE,   # noqa: E402
                                UNANSWERABLE, nll_nowrite)


def reflow(text):
    """Join the lines inside each paragraph; keep the paragraph breaks."""
    paras = re.split(r"\n\s*\n", text)
    return "\n\n".join(" ".join(p.split()) for p in paras if p.strip())


def arm(label, text, target):
    tmp = tempfile.mkdtemp(prefix="reflow_")
    try:
        s = Sillage(model="qwen", state=tmp, target=target, quiet=True)
        for _ in range(2):
            rec = s.read_text(text)
        out = {"label": label,
               "state": {"tokens": rec["tokens"],
                         "cold_grams": len(s.mem.cold),
                         "ppl_frozen": rec["ppl_frozen"],
                         "ppl_with_memory": rec["ppl_with_memory"]},
               "answerable": [], "unanswerable": []}
        print(f"\n=== {label} === {out['state']}", flush=True)
        for prompt, want in ANSWERABLE:
            txt = s.complete(prompt, n=12, temp=0.0)
            at = s.attribution() or {}
            ok = want.lower() in txt.lower()
            out["answerable"].append(
                {"want": want, "ok": bool(ok), "got": txt.strip()[:48],
                 "moved": at.get("moved"), "tokens": at.get("tokens")})
            print(f"  {'OK ' if ok else 'MISS'} {want:<9} "
                  f"(moved {at.get('moved'):>2}/{at.get('tokens')}) "
                  f"{txt.strip()[:40]!r}", flush=True)
        out["recall"] = sum(f["ok"] for f in out["answerable"])
        for prompt in UNANSWERABLE:
            txt = s.complete(prompt, n=12, temp=0.0)
            at = s.attribution() or {}
            out["unanswerable"].append(
                {"prompt": prompt[-34:], "got": txt.strip()[:48],
                 "moved": at.get("moved"), "tokens": at.get("tokens")})
            print(f"  ???            (moved {at.get('moved'):>2}/"
                  f"{at.get('tokens')}) {txt.strip()[:40]!r}", flush=True)
        out["moved_unanswerable"] = sum(f["moved"] for f in
                                        out["unanswerable"])
        # the witness goes LAST: nll_nowrite folds states into the v1
        # tier's centre without the matching tier writes, and measured
        # before the questions it costs this document a fact
        b, m = nll_nowrite(s, WITNESS)
        out["witness"] = {"base": b, "mem": m, "delta": m - b}
        print(f"  recall {out['recall']}/{len(ANSWERABLE)}   "
              f"witness {m - b:+.4f} nat", flush=True)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = {"target": a.target,
           "reflow_example": reflow(DOC)[:160]}
    res["as_is"] = arm("as-is (the document as written)", DOC, a.target)
    res["reflow"] = arm("reflow (paragraph lines joined)", reflow(DOC),
                        a.target)

    A, B = res["as_is"], res["reflow"]
    lost = [x["want"] for x, y in zip(A["answerable"], B["answerable"])
            if x["ok"] and not y["ok"]]
    gained = [y["want"] for x, y in zip(A["answerable"], B["answerable"])
              if y["ok"] and not x["ok"]]
    res["verdict"] = {
        "T1": {"as_is": A["recall"], "reflow": B["recall"],
               "holds": B["recall"] > A["recall"]},
        "T2": {"brindas_before": next(f["ok"] for f in A["answerable"]
                                      if f["want"] == "Brindas"),
               "brindas_after": next(f["ok"] for f in B["answerable"]
                                     if f["want"] == "Brindas")},
        "T3": {"lost": lost, "holds": not lost},
        "T4": {"delta": abs(B["witness"]["delta"] - A["witness"]["delta"]),
               "holds": abs(B["witness"]["delta"]
                            - A["witness"]["delta"]) < 0.2},
        "T5": {"as_is": A["moved_unanswerable"],
               "reflow": B["moved_unanswerable"]},
        "T6": {"ppl_as_is": A["state"]["ppl_with_memory"],
               "ppl_reflow": B["state"]["ppl_with_memory"],
               "tokens_as_is": A["state"]["tokens"],
               "tokens_reflow": B["state"]["tokens"]},
        "gained": gained}
    print("\n" + json.dumps(res["verdict"], indent=1, ensure_ascii=False))

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results", "reflow.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
