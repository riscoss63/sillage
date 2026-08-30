"""Exactly how brittle is the memory's key to the shape of the question?

`probe_outvoted` showed the missed fact is not in the cold store at all:
a RETRIEVAL failure. Both tiers key on the last four TOKENS, so a
question that does not tokenize the way the document did at that point
finds nothing -- and says nothing, which reads as "the memory forgot".

Three variants of the same question, one fact:

  A  exactly as the document has it, line break included, no trailing space
  B  the same with the line break rewrapped to a space (how a person types)
  C  exactly as the document has it, but with a trailing space

Registered BEFORE the run:

  S1  Variant A hits the cold store and recalls the fact.
      FALSIFIED if A finds no gram.
  S2  At least one of B and C misses the store entirely -- naming the
      surface change that costs the recall.
  S3  The same three variants applied to a fact in the MIDDLE of a line
      (no break anywhere near) behave identically to each other, so the
      effect is attributable to the surface and not to the fact.

Run:  python behav/probe_tokenkey.py
"""
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sillage.runtime import Sillage            # noqa: E402
from probe_readout_dial import DOC             # noqa: E402
from probe_outvoted import inspect             # noqa: E402

CASES = [
    ("Brindas", "wrapped"),      # the document breaks the line before it
    ("62", "mid-line"),          # control: no break in the vicinity
]


def variants(want):
    i = DOC.find(want)
    exact = DOC[max(0, i - 90):i]
    a = exact.rstrip(" ")                       # doc shape, no trailing space
    b = a.replace("\n", " ")                    # rewrapped the way people type
    c = a + " "                                 # doc shape + trailing space
    return (("A doc-exact", a), ("B rewrapped", b), ("C trailing-space", c))


def main():
    tmp = tempfile.mkdtemp(prefix="tokenkey_")
    res = {"cases": []}
    try:
        s = Sillage(model="qwen", state=tmp, quiet=True)
        for _ in range(2):
            rec = s.read_text(DOC)
        print(f"state: {rec['tokens']} tokens, {len(s.mem.cold)} grams\n",
              flush=True)
        tokr = s.load_tokenizer()
        for want, kind in CASES:
            print(f"-- {want} ({kind})", flush=True)
            row = {"want": want, "kind": kind, "variants": []}
            for label, prompt in variants(want):
                d = inspect(s, prompt)
                txt = s.complete(prompt, n=8, temp=0.0)
                key = tokr.encode(prompt)[-4:]
                d.update({"label": label,
                          "recalled": want.lower() in txt.lower(),
                          "got": txt.strip()[:36],
                          "key": [tokr.decode([t]) for t in key]})
                row["variants"].append(d)
                print(f"   {label:<18} cold "
                      + ("HIT " if d["cold"] else "MISS")
                      + f"  {'OK ' if d['recalled'] else '.  '} "
                      f"key={d['key']}  -> {d['got']!r}", flush=True)
            res["cases"].append(row)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "tokenkey.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
