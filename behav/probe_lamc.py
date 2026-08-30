"""Reflow fixes retrieval. Does the cold store then get to finish the word?

The chain of refutations ends here. On the as-is document the question's
4-gram is absent (retrieval failure), so the earlier LAM_C test was
CONFOUNDED -- raising the cold weight cannot help a gram that is not
there. Reflowed, the gram is present with successor ' Br' at 2/2, and
`complete` emits exactly that... and then writes 'Brigitte Lefevre'.
The memory opened the word and the frozen model finished it with a
commoner French name.

So the question is arbitration after all, but only once retrieval works:
LAM_C = 0.3 gives the cold store 30% of the mixture, and a confident
continuation ('Br' -> 'ig') needs only p > 0.43 to outvote it.

Registered BEFORE the run:

  U1  The store holds the WHOLE name: following successors from the
      question's key spells `Brindas Kolvec` for at least 3 tokens.
      FALSIFIED if the chain breaks before then.
  U2  There is a LAM_C above which the reflowed state recalls 8/8.
      FALSIFIED if no value in {0.3,0.5,0.7,0.9} reaches 8/8.
  U3  That value costs something measurable: at the LAM_C that wins,
      the tokens moved on UNANSWERABLE questions is strictly greater
      than at 0.3 -- the memory speaks more where it should not.
      Recorded either way; this is the trade-off to publish.
  U4  No fact recalled at 0.3 is lost at the winning LAM_C.
      FALSIFIED by any regression.

Run:  python behav/probe_lamc.py
"""
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sillage.core as core                        # noqa: E402
from sillage.runtime import Sillage                # noqa: E402
from probe_readout_dial import (DOC, ANSWERABLE,   # noqa: E402
                                UNANSWERABLE)
from probe_reflow import reflow                    # noqa: E402
from probe_whymiss import Q                        # noqa: E402

GRID = [0.3, 0.5, 0.7, 0.9]


def chain(mem, tokr, prompt, steps=6):
    """Follow the cold store's own successors from a question's key."""
    ids = list(tokr.encode(prompt))
    out = []
    for _ in range(steps):
        gram = np.array(ids[-core.NGRAM:], dtype=np.int32).tobytes()
        slot = mem.cold.get(gram)
        if slot is None:
            break
        nxt = max(slot[1], key=slot[1].get)
        out.append(tokr.decode([nxt]))
        ids.append(nxt)
    return out


def main():
    tmp = tempfile.mkdtemp(prefix="lamc_")
    res = {"grid": GRID, "arms": {}}
    try:
        s = Sillage(model="qwen", state=tmp, quiet=True)
        text = reflow(DOC)
        for _ in range(2):
            rec = s.read_text(text)
        tokr = s.load_tokenizer()
        res["state"] = {"tokens": rec["tokens"],
                        "cold_grams": len(s.mem.cold)}
        res["chain"] = chain(s.mem, tokr, Q)
        print(f"state {res['state']}", flush=True)
        print(f"U1 cold-store chain from the question: "
              f"{''.join(res['chain'])!r}\n", flush=True)

        for lam in GRID:
            core.LAM_C = lam
            arm = {"answerable": [], "unanswerable": []}
            for prompt, want in ANSWERABLE:
                txt = s.complete(prompt, n=12, temp=0.0)
                at = s.attribution() or {}
                arm["answerable"].append(
                    {"want": want, "ok": want.lower() in txt.lower(),
                     "got": txt.strip()[:44], "moved": at.get("moved")})
            for prompt in UNANSWERABLE:
                txt = s.complete(prompt, n=12, temp=0.0)
                at = s.attribution() or {}
                arm["unanswerable"].append(
                    {"prompt": prompt[-30:], "got": txt.strip()[:44],
                     "moved": at.get("moved")})
            arm["recall"] = sum(f["ok"] for f in arm["answerable"])
            arm["moved_unanswerable"] = sum(f["moved"] for f in
                                            arm["unanswerable"])
            res["arms"][str(lam)] = arm
            miss = [f["want"] for f in arm["answerable"] if not f["ok"]]
            print(f"LAM_C {lam}: recall {arm['recall']}/8  "
                  f"moved-on-unanswerable {arm['moved_unanswerable']}/48  "
                  f"missing {miss}", flush=True)
            for f in arm["unanswerable"]:
                print(f"    ??? (moved {f['moved']:>2}) {f['got']!r}",
                      flush=True)
    finally:
        core.LAM_C = 0.3
        shutil.rmtree(tmp, ignore_errors=True)

    base = res["arms"]["0.3"]
    win = next((l for l in GRID if res["arms"][str(l)]["recall"] == 8), None)
    res["verdict"] = {
        "U1": {"chain": "".join(res["chain"]),
               "holds": "Brindas" in "".join(res["chain"])},
        "U2": {"winner": win, "holds": win is not None},
        "U3": ({"at_0.3": base["moved_unanswerable"],
                "at_winner": res["arms"][str(win)]["moved_unanswerable"]}
               if win else None),
        "U4": ({"lost": [a["want"] for a, b in
                         zip(base["answerable"],
                             res["arms"][str(win)]["answerable"])
                         if a["ok"] and not b["ok"]]} if win else None)}
    print("\n" + json.dumps(res["verdict"], indent=1, ensure_ascii=False))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "lamc.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
