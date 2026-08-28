"""fast_ingest == read_text, state-wise: the guarantee, tested.

Builds the same synthetic document twice with GPT-2 on the CPU -- once
through `read_text` (the shipped path), once through `fast_ingest` --
and asserts the two states are IDENTICAL: every array of state.npz
bit-for-bit (M, MS, A, reservoirs, mu), the cold store key-for-key and
count-for-count, and the behavioral check: the same completions on
three invented facts.

    python test_fast_ingest.py
"""

import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break
sys.path.insert(0, HERE)

from sillage import Sillage                                    # noqa: E402
from fast_ingest import fast_ingest, fast_ingest_blocked      # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "behav"))
from behavioral import ENTS, VALS, A_PREFIX, build_doc         # noqa: E402

# big enough that the 1-in-8 subsampled reservoir passes the 500-sample
# floor under which the tiers stay silent (~5k tokens -> ~650 samples)
DOC = build_doc(list(zip(ENTS[:10], VALS[:10])), seed=3, reps=4, block=90)
ST_A = os.path.join(HERE, ".state_slow")
ST_B = os.path.join(HERE, ".state_fast")

for st in (ST_A, ST_B):
    shutil.rmtree(st, ignore_errors=True)

print("== A. read_text (chemin normal) ==", flush=True)
sA = Sillage(model="gpt2", state=ST_A, quiet=True)
t0 = time.time()
recA = sA.read_text(DOC, "doc")
sA.save()
tA = time.time() - t0
print(f"  {recA['tokens']} tokens en {tA:.1f}s "
      f"(PPL {recA['ppl_frozen']} -> {recA['ppl_with_memory']})",
      flush=True)

print("== B. fast_ingest ==", flush=True)
sB = Sillage(model="gpt2", state=ST_B, quiet=True)
t0 = time.time()
recB = fast_ingest(sB, DOC, "doc")
sB.save()
tB = time.time() - t0
print(f"  {recB['tokens']} tokens en {tB:.1f}s "
      f"(x{tA/max(tB,1e-6):.1f} vs read_text)", flush=True)

print("== B2. fast_ingest gate=torch (tolerance declaree) ==", flush=True)
ST_C = os.path.join(HERE, ".state_torchgate")
shutil.rmtree(ST_C, ignore_errors=True)
sC = Sillage(model="gpt2", state=ST_C, quiet=True)
t0 = time.time()
recC = fast_ingest(sC, DOC, "doc", gate="torch")
sC.save()
tC = time.time() - t0
print(f"  {recC['tokens']} tokens en {tC:.1f}s "
      f"(x{tA/max(tC,1e-6):.1f} vs read_text)", flush=True)

print("== B3. fast_ingest_blocked (GEMM par blocs de 64) ==", flush=True)
ST_D = os.path.join(HERE, ".state_blocked")
shutil.rmtree(ST_D, ignore_errors=True)
sD = Sillage(model="gpt2", state=ST_D, fastweights=False, quiet=True)
t0 = time.time()
recD = fast_ingest_blocked(sD, DOC, "doc")
sD.save()
tD = time.time() - t0
print(f"  {recD['tokens']} tokens en {tD:.1f}s "
      f"(x{tA/max(tD,1e-6):.1f} vs read_text)", flush=True)

print("== C. egalite des etats ==", flush=True)
zA = np.load(os.path.join(ST_A, "state.npz"), allow_pickle=True)
zB = np.load(os.path.join(ST_B, "state.npz"), allow_pickle=True)
assert set(zA.files) == set(zB.files), (set(zA.files) ^ set(zB.files))
diffs = []
for k in zA.files:
    a, b = zA[k], zB[k]
    if a.dtype.kind in "fc" and a.shape == b.shape:
        same = np.array_equal(a, b)
    else:
        same = (a.shape == b.shape) and bool(np.all(a == b))
    mark = "identique" if same else "DIFFERENT"
    if not same:
        diffs.append(k)
    print(f"  {k:16s} {str(a.shape):>14s}  {mark}", flush=True)

mA, mB = sA.mem, sB.mem
assert mA.cold.keys() == mB.cold.keys(), "cold: cles differentes"
for kk in mA.cold:
    a, b = mA.cold[kk], mB.cold[kk]
    assert a[1] == b[1], f"cold counts differ on {kk!r}: {a[1]} vs {b[1]}"
    assert abs(a[0] - b[0]) < 1e-9, f"cold mass differs on {kk!r}"
print(f"  cold store       {len(mA.cold)} grams   identique", flush=True)

print("== D. egalite comportementale ==", flush=True)
for e, v in list(zip(ENTS[:10], VALS[:10]))[:3]:
    outA = sA.complete(A_PREFIX.format(e=e), n=6)
    outB = sB.complete(A_PREFIX.format(e=e), n=6)
    assert outA == outB, f"completions divergent on {e}: {outA!r} {outB!r}"
    print(f"  {e:12s} -> {outA!r:40s} (les deux)", flush=True)

print("== C2. gate=torch : admissions identiques, amplitudes proches ==",
      flush=True)
mC = sC.mem
assert mA.cold.keys() == mC.cold.keys(), "torch-gate: cles cold differentes"
# the gate differs by float rounding (~1e-4 nats per write): on slots
# whose mass is a sum of near-zero surprises that is RELATIVELY large,
# absolutely negligible. Bound the absolute drift everywhere, and the
# relative drift only where consolidation ranking can care (mass >= .5).
worst_abs = worst_rel = 0.0
for kk in mA.cold:
    a, c = mA.cold[kk], mC.cold[kk]
    assert a[1] == c[1], f"torch-gate: counts differ on {kk!r}"
    worst_abs = max(worst_abs, abs(a[0] - c[0]))
    if a[0] >= 0.5:
        worst_rel = max(worst_rel, abs(a[0] - c[0]) / a[0])
assert worst_abs < 1e-2, f"masse cold derive (abs): {worst_abs:.2e}"
assert worst_rel < 1e-3, f"masse cold derive (rel>=0.5): {worst_rel:.2e}"
dM = float(np.abs(zA["M"] - np.load(os.path.join(
    ST_C, "state.npz"))["M"]).max())
assert dM < 1e-3, f"M derive: {dM:.2e}"
print(f"  cold: cles et comptes identiques ; drift masse abs "
      f"{worst_abs:.1e}, rel(masse>=0.5) {worst_rel:.1e} ; "
      f"M max drift {dM:.1e}", flush=True)
# reservoir subsampled 1-in-8 in torch mode: the QUANTILE the threshold
# reads must stay close. Compare the estimator directly (this doc is
# shorter than the 500-sample floor under which _thr stays silent --
# on the real 120k-token questions the floor is passed after ~2
# sessions, and silence is the conservative side anyway).
for nm, rA, rC, qq in (("res_G", mA.res_G, mC.res_G, mA.thr_qG),
                       ("res_S", mA.res_S, mC.res_S, mA.thr_qS)):
    if qq is None or len(rA) < 30 or len(rC) < 30:
        print(f"  {nm}: non compare (q={qq}, n={len(rA)}/{len(rC)})",
              flush=True)
        continue
    va = float(np.quantile(np.array(rA[-5000:]), qq))
    vc = float(np.quantile(np.array(rC[-5000:]), qq))
    rel = abs(va - vc) / max(1e-9, abs(va))
    assert rel < 0.10, f"{nm} quantile derive {rel:.1%} ({va:.4f} vs {vc:.4f})"
    print(f"  {nm} q{int(qq*100)}: {va:.4f} vs {vc:.4f} (drift {rel:.1%}, "
          f"reservoir 1-sur-8, n={len(rA)} vs {len(rC)})", flush=True)
for e, v in list(zip(ENTS[:10], VALS[:10]))[:3]:
    outA = sA.complete(A_PREFIX.format(e=e), n=6)
    outC = sC.complete(A_PREFIX.format(e=e), n=6)
    assert outA == outC, f"torch-gate: completions divergent on {e}"
print("  completions identiques a read_text", flush=True)

print("== C3. blocked : cold exact, comportement identique ==",
      flush=True)
mD = sD.mem
assert mA.cold.keys() == mD.cold.keys(), "blocked: cles cold differentes"
wa = wr = 0.0
for kk in mA.cold:
    a, dd = mA.cold[kk], mD.cold[kk]
    assert a[1] == dd[1], f"blocked: counts differ on {kk!r}"
    wa = max(wa, abs(a[0] - dd[0]))
    if a[0] >= 0.5:
        wr = max(wr, abs(a[0] - dd[0]) / a[0])
assert wa < 1e-2, f"blocked: masse cold (abs) {wa:.2e}"
assert wr < 1e-3, f"blocked: masse cold (rel>=0.5) {wr:.2e}"
dMD = float(np.abs(zA["M"] - np.load(os.path.join(
    ST_D, "state.npz"))["M"]).max())
assert dMD < 1.0, f"blocked: M derive {dMD:.2e}"
assert mA.tokens == mD.tokens and mA.g_cnt == mD.g_cnt, "compteurs"
for nm, rA, rD2, qq in (("res_G", mA.res_G, mD.res_G, mA.thr_qG),):
    if qq is not None and len(rA) >= 30 and len(rD2) >= 30:
        va = float(np.quantile(np.array(rA[-5000:]), qq))
        vd = float(np.quantile(np.array(rD2[-5000:]), qq))
        rel = abs(va - vd) / max(1e-9, abs(va))
        assert rel < 0.10, f"blocked: quantile {rel:.1%}"
        print(f"  res_G q{int(qq*100)}: {va:.4f} vs {vd:.4f} "
              f"(drift {rel:.1%})", flush=True)
print(f"  cold: cles/comptes identiques, masse abs {wa:.1e} ; "
      f"M max drift {dMD:.1e} (sur-escalade intra-bloc declaree) ; "
      f"compteurs identiques", flush=True)
adapt_note = 0
for e, v in list(zip(ENTS[:10], VALS[:10]))[:3]:
    outA = sA.complete(A_PREFIX.format(e=e), n=6)
    outD = sD.complete(A_PREFIX.format(e=e), n=6)
    assert outA == outD, f"blocked: completions divergent on {e}: "         f"{outA!r} vs {outD!r}"
print("  completions identiques a read_text", flush=True)

for st in (ST_A, ST_B, ST_C, ST_D):
    shutil.rmtree(st, ignore_errors=True)

if diffs:
    sys.exit(f"ETATS DIFFERENTS sur {diffs} -- fast_ingest n'est pas "
             f"un remplacement exact.")
print(f"\nFAST INGEST == READ_TEXT (bit-a-bit en mode exact x"
      f"{tA/max(tB,1e-6):.1f}; gate=torch dans les tolerances x"
      f"{tA/max(tC,1e-6):.1f})")
