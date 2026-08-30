"""Stop guessing: look inside the cold store for the gram that misses.

Three hypotheses for `Brindas` have now been refuted (line wrap alone,
arbitration, reflowed reading). This does no inference at all: it reads
the document, builds the exact 4-gram the question forms, and asks the
store whether it holds it, what it holds, and under what count.
"""
import os
import re
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sillage.core as core                  # noqa: E402
from sillage.runtime import Sillage          # noqa: E402
from probe_readout_dial import DOC           # noqa: E402
from probe_reflow import reflow              # noqa: E402

Q = ("Le rapport a ete signe le 14 juin 2026 par le technicien "
     "responsable, madame")


def look(label, text):
    tmp = tempfile.mkdtemp(prefix="whymiss_")
    try:
        s = Sillage(model="qwen", state=tmp, quiet=True)
        for _ in range(2):
            s.read_text(text)
        tokr = s.load_tokenizer()
        mem = s.mem
        print(f"\n=== {label}: {len(mem.cold)} grams", flush=True)

        qids = tokr.encode(Q)
        gram = np.array(qids[-core.NGRAM:], dtype=np.int32).tobytes()
        slot = mem.cold.get(gram)
        print("  question key :", [tokr.decode([t]) for t in
                                   qids[-core.NGRAM:]])
        if slot is None:
            print("  -> NOT in the store")
        else:
            tot = sum(slot[1].values())
            best = max(slot[1], key=slot[1].get)
            print(f"  -> in the store, count {tot}, best successor "
                  f"{tokr.decode([best])!r} ({slot[1][best]}/{tot})")

        # what the document itself forms at the same place
        i = text.find("Brindas")
        dids = tokr.encode(text[:i])
        dgram = np.array(dids[-core.NGRAM:], dtype=np.int32).tobytes()
        print("  document key :", [tokr.decode([t]) for t in
                                   dids[-core.NGRAM:]])
        dslot = mem.cold.get(dgram)
        if dslot is None:
            print("  -> the DOCUMENT's own key is not in the store either")
        else:
            tot = sum(dslot[1].values())
            best = max(dslot[1], key=dslot[1].get)
            print(f"  -> count {tot}, best successor "
                  f"{tokr.decode([best])!r} ({dslot[1][best]}/{tot})")
        print(f"  same key: {gram == dgram}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


look("as-is", DOC)
look("reflow", reflow(DOC))
