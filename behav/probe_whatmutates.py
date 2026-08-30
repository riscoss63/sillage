"""Snapshot every array on the memory, run the witness pass, diff.

The bisect says a teacher-forced scoring pass with NO writes changes a
later answer (9 tokens moved -> 1, `Brindas Kolvec` -> `Brigitte
Lefevre`) while thresholds and reservoir sizes stay identical. Reading
the source found nothing that mutates. So compare the object itself.
"""
import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sillage.runtime import Sillage                        # noqa: E402
from probe_readout_dial import DOC, WITNESS, nll_nowrite   # noqa: E402
from probe_reflow import reflow                            # noqa: E402


def snap(mem):
    out = {}
    for k in sorted(vars(mem)):
        v = getattr(mem, k)
        if isinstance(v, np.ndarray):
            out[k] = (v.shape, v.dtype.str,
                      hashlib.sha1(np.ascontiguousarray(v)).hexdigest()[:12])
        elif isinstance(v, (int, float, bool, str, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = f"{type(v).__name__}[{len(v)}]"
        elif isinstance(v, dict):
            out[k] = f"dict[{len(v)}]"
        else:
            out[k] = type(v).__name__
    return out


tmp = tempfile.mkdtemp(prefix="whatmut_")
try:
    s = Sillage(model="qwen", state=tmp, quiet=True)
    for _ in range(2):
        s.read_text(reflow(DOC))
    before = snap(s.mem)
    nll_nowrite(s, WITNESS)
    after = snap(s.mem)

    changed = [k for k in before if before[k] != after[k]]
    print("attributes that changed across a NO-WRITE scoring pass:")
    for k in changed:
        print(f"  {k}: {before[k]}  ->  {after[k]}")
    if not changed:
        print("  (none -- the difference is not on the memory object)")
    print(f"\nunchanged: {len(before) - len(changed)} attributes")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
