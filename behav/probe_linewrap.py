"""Does an ordinary line wrap in the source silently break recall?

The readout probe found the memory completely silent -- zero tokens
moved, under both readouts -- on `madame Brindas Kolvec`, a proper noun
that IS in the document it read. The document wraps that line:

    ... par le technicien responsable,
    madame Brindas Kolvec, matricule 4471.

so the source holds `responsable,\\nmadame` while the question asks
`responsable, madame`. Both memory tiers key on the last four TOKENS,
and `\\nmadame` is not the token ` madame`. If that is the cause, then
every paragraph wrap in every real document is a place where recall
silently fails -- which matters far more than any readout constant.

Registered BEFORE the run:

  Q1  Verbatim prefixes (line breaks kept exactly as the document has
      them) recall strictly MORE facts than the same prefixes with the
      breaks rewrapped to spaces.
      FALSIFIED if rewrapped >= verbatim.
  Q2  The gap is concentrated on the facts whose prefix actually
      contains a break: facts with no break in their prefix score the
      same both ways.
      FALSIFIED if a no-break fact changes answer between the two arms.
  Q3  On a fact whose prefix contains a break, the rewrapped arm moves
      strictly fewer tokens than the verbatim arm -- the tiers go quiet
      rather than answering wrongly.
      FALSIFIED if the rewrapped arm moves as many or more.

Run:  python behav/probe_linewrap.py [--target HUB]
"""
import argparse
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sillage.runtime import Sillage            # noqa: E402
from probe_readout_dial import DOC, ANSWERABLE  # noqa: E402

CTX = 74            # characters of prefix handed to the model


def prefixes():
    """For each fact: (verbatim prefix, rewrapped prefix, has a break)."""
    out = []
    for _prompt, want in ANSWERABLE:
        i = DOC.find(want)
        if i < 0:
            continue
        raw = DOC[max(0, i - CTX):i]
        flat = raw.replace("\n", " ")
        out.append((want, raw, flat, "\n" in raw.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="linewrap_")
    res = {"target": a.target, "facts": []}
    try:
        s = Sillage(model="qwen", state=tmp, target=a.target, quiet=True)
        for _ in range(2):
            rec = s.read_text(DOC)
        res["state"] = {"tokens": rec["tokens"], "cold_grams": len(s.mem.cold)}
        print(f"state: {res['state']}", flush=True)

        for want, raw, flat, broken in prefixes():
            row = {"want": want, "wrapped_in_source": broken}
            for label, prompt in (("verbatim", raw), ("rewrapped", flat)):
                txt = s.complete(prompt, n=10, temp=0.0)
                at = s.attribution() or {}
                row[label] = {"ok": want.lower() in txt.lower(),
                              "got": txt.strip()[:44],
                              "moved": at.get("moved"),
                              "tokens": at.get("tokens")}
            res["facts"].append(row)
            v, r = row["verbatim"], row["rewrapped"]
            print(f"  {'BREAK' if broken else '     '} {want:<9} "
                  f"verbatim {'OK ' if v['ok'] else '.  '}"
                  f"(moved {v['moved']:>2}/{v['tokens']})   "
                  f"rewrapped {'OK ' if r['ok'] else '.  '}"
                  f"(moved {r['moved']:>2}/{r['tokens']})", flush=True)

        for lab in ("verbatim", "rewrapped"):
            hits = sum(f[lab]["ok"] for f in res["facts"])
            moved = sum(f[lab]["moved"] for f in res["facts"])
            tot = sum(f[lab]["tokens"] for f in res["facts"])
            res[lab] = {"recall": hits / len(res["facts"]),
                        "moved": moved, "tokens": tot}
            print(f"{lab:>10}: recall {hits}/{len(res['facts'])}  "
                  f"moved {moved}/{tot}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results", "linewrap.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
