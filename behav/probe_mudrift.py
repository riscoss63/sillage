"""Is the drifting centring `mu` the channel that derails recall?

`probe_whatmutates` found that a teacher-forced pass with no writes
still moves `mu` (the running mean of hidden states, 826 -> 1008
observations) and `_graw`. `mu` is what the semantic tier centres its
keys on. If that is the channel, then switching the semantic tier OFF
should make the witness pass harmless.

Registered BEFORE the run:

  V1  With the semantic tier ON, the witness pass flips the answer
      (Brindas -> Brigitte), reproducing the bisect.
  V2  With the semantic tier OFF, the witness pass does NOT flip it.
      FALSIFIED if it flips anyway -- then `mu` is not the channel and
      `_graw` or something else is.
  V3  With the tier OFF, recall of the eight facts is at least as good
      as with it ON, on this document.
      Recorded: a tier that can derail a perfect cold-store recall for
      a 10% mixture weight is worth its own limit note.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sillage.runtime import Sillage                        # noqa: E402
from probe_readout_dial import (DOC, WITNESS, ANSWERABLE,   # noqa: E402
                                nll_nowrite)
from probe_reflow import reflow                            # noqa: E402

PROMPT, WANT = ANSWERABLE[6]


def trial(semantic, witness):
    tmp = tempfile.mkdtemp(prefix="mudrift_")
    try:
        s = Sillage(model="qwen", state=tmp, semantic=semantic, quiet=True)
        for _ in range(2):
            s.read_text(reflow(DOC))
        mu0 = float(s.mem.mu_n)
        if witness:
            nll_nowrite(s, WITNESS)
        txt = s.complete(PROMPT, n=12, temp=0.0)
        at = s.attribution() or {}
        hits = sum(w.lower() in s.complete(p, n=12, temp=0.0).lower()
                   for p, w in ANSWERABLE)
        print(f"  semantic={str(semantic):<5} witness={str(witness):<5} "
              f"mu_n {mu0:.0f}->{s.mem.mu_n} "
              f"{'OK  ' if WANT.lower() in txt.lower() else 'MISS'} "
              f"moved={at.get('moved'):>2}/{at.get('tokens')} "
              f"recall={hits}/8 -> {txt.strip()[:30]!r}", flush=True)
        return WANT.lower() in txt.lower(), hits
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


print("reflowed state, one question, four conditions:")
on_clean, r1 = trial(True, False)
on_wit, r2 = trial(True, True)
off_clean, r3 = trial(False, False)
off_wit, r4 = trial(False, True)

print(f"\nV1 semantic ON  flipped by the witness pass: {on_clean and not on_wit}")
print(f"V2 semantic OFF flipped by the witness pass: "
      f"{off_clean and not off_wit}  (prediction: False)")
print(f"V3 recall ON {r1}/8 (clean) {r2}/8 (witness) | "
      f"OFF {r3}/8 (clean) {r4}/8 (witness)")
