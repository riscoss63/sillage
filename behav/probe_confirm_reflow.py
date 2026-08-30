"""Reproduce the disagreement before believing either number.

`probe_reflow` scored 7/8 on the reflowed state (missing `Brindas`,
output 'Brigitte Lefevre'); `probe_lamc` scored 8/8 on what should be
the identical state at the identical LAM_C. One of them is wrong, or
something in the process differs. This builds the reflowed state twice
in one process and asks the one question both times.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sillage.core as core                  # noqa: E402
from sillage.runtime import Sillage          # noqa: E402
from probe_readout_dial import DOC, ANSWERABLE   # noqa: E402
from probe_reflow import reflow              # noqa: E402

PROMPT, WANT = ANSWERABLE[6]
assert WANT == "Brindas", WANT


def once(tag, first_read_asis):
    tmp = tempfile.mkdtemp(prefix="confirm_")
    try:
        s = Sillage(model="qwen", state=tmp, quiet=True)
        if first_read_asis:
            # the ONLY structural difference between the two probes:
            # probe_reflow ran a whole as-is arm first, in-process
            s2tmp = tempfile.mkdtemp(prefix="confirm_pre_")
            try:
                pre = Sillage(model="qwen", state=s2tmp, quiet=True)
                for _ in range(2):
                    pre.read_text(DOC)
                pre.complete(PROMPT, n=12, temp=0.0)
            finally:
                shutil.rmtree(s2tmp, ignore_errors=True)
        for _ in range(2):
            rec = s.read_text(reflow(DOC))
        txt = s.complete(PROMPT, n=12, temp=0.0)
        at = s.attribution() or {}
        ok = WANT.lower() in txt.lower()
        print(f"  {tag:<28} grams={len(s.mem.cold)} LAM_C={core.LAM_C} "
              f"{'OK ' if ok else 'MISS'} moved={at.get('moved')}/"
              f"{at.get('tokens')} -> {txt.strip()[:40]!r}", flush=True)
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


print("question:", PROMPT[-46:], flush=True)
a = once("clean process", False)
b = once("clean process (repeat)", False)
c = once("after an as-is arm", True)
print(f"\nreproducible: {a == b};  as-is arm first changes it: {a != c}")
