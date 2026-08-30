"""Which step before the question changes the answer? Bisect it.

Same reflowed state, same question, four orders of operations. One of
them turns `Brindas Kolvec` into `Brigitte Lefevre`, and no reading of
the source has found which -- `adapt` and `phi` are pure, `mix_true`
touches no reservoir, and `res_G` is appended only inside `read_text`.
So measure instead of arguing.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sillage.runtime import Sillage                       # noqa: E402
from probe_readout_dial import (DOC, WITNESS, ANSWERABLE,  # noqa: E402
                                nll_nowrite)
from probe_reflow import reflow                           # noqa: E402

PROMPT, WANT = ANSWERABLE[6]


def build():
    tmp = tempfile.mkdtemp(prefix="bisect_")
    s = Sillage(model="qwen", state=tmp, quiet=True)
    for _ in range(2):
        s.read_text(reflow(DOC))
    return s, tmp


def ask(s, tag):
    txt = s.complete(PROMPT, n=12, temp=0.0)
    at = s.attribution() or {}
    ok = WANT.lower() in txt.lower()
    print(f"  {tag:<34} {'OK  ' if ok else 'MISS'} "
          f"moved={at.get('moved'):>2}/{at.get('tokens')} "
          f"thrG={s.mem.thresholds()[0]:.4f} res_G={len(s.mem.res_G)} "
          f"-> {txt.strip()[:34]!r}", flush=True)
    return ok


CASES = ("question only",
         "witness pass, then question",
         "six earlier questions, then it",
         "witness + six questions (probe_reflow)")

for i, tag in enumerate(CASES):
    s, tmp = build()
    try:
        if i in (1, 3):
            nll_nowrite(s, WITNESS)
        if i in (2, 3):
            for p, _w in ANSWERABLE[:6]:
                s.complete(p, n=12, temp=0.0)
        ask(s, tag)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
