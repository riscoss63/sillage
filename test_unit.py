"""Fast tests of the mechanisms themselves -- numpy only, no model, ~5 s.

These check the papers' rules where they can be checked exactly: the
Hebbian read/write, the square-root (amplitude) encoding, leaky forgetting,
the delta rule at the readout, consolidation by surprise mass, the state
round-trip, the multi-model paths, the readout tuner and its rolling window,
and the passage splitter. `test_sillage.py` is the slow end-to-end complement
that actually runs a frozen GPT-2.

    python test_unit.py
"""

import os
import shutil
import tempfile

import numpy as np

from sillage.core import (BETAS, CAP, D_K, D_V, ETA, R_FEAT,
                          SillageMemory, fit_readout, lse_grid,
                          peek, resolve)
from sillage.index import Index, paragraphs

VOCAB = 50257          # gpt2, so no model download is ever needed
passed = []


def check(name, cond, detail=""):
    assert cond, f"{name} FAILED {detail}"
    passed.append(f"{name} ok {detail}")


# --- 1. retrieval: writing a continuation makes it the top score -----------
mem = SillageMemory(None, "gpt2", semantic=False, fastweights=False)
mem.new_stream()
seq = [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]
scores_before = None
for j in range(len(seq) - 1):
    q = mem.step_key(seq[j])
    u, s = mem.scores(mem.M, q)
    if j == 8:            # key (10,20,30,40) again: 50 followed it at j=3
        scores_before = s.copy()
    mem.write_all(q, u, None, None, seq[j + 1], 2.0)
rank = int((scores_before > scores_before[50]).sum())
check("T1 n-gram retrieval", rank == 0,
      f"(the continuation seen after this exact 4-gram is rank {rank + 1})")

# --- 2. amplitude encoding is sublinear ------------------------------------
m2 = SillageMemory(None, "gpt2", semantic=False, fastweights=False)
q = np.zeros(D_K, dtype=np.float32)
q[0] = 1.0
tok = 7
for _ in range(4):
    u, _ = m2.scores(m2.M, q)
    m2.amp_write(m2.M, q, u, tok, 1.0)
coef = float(m2.M[0] @ m2.V[tok]) / float(m2.V[tok] @ m2.V[tok])
check("T2 amplitude encoding", abs(coef - 2.0) < 0.05,
      f"(4 unit writes -> coefficient {coef:.3f}, i.e. sqrt(4), not 4)")

# --- 3. forgetting halves the traces after one half-life -------------------
m3 = SillageMemory(None, "gpt2", semantic=False, fastweights=False,
                   half_life=640)
m3.M[:] = 1.0
before = float(np.abs(m3.M).sum())
for _ in range(640):
    m3.decay_step()
ratio = float(np.abs(m3.M).sum()) / before
check("T3 leaky forgetting", abs(ratio - 0.5) < 0.02,
      f"(traces at {ratio:.3f} of their mass after one half-life)")

# --- 4. the delta rule at the readout actually learns ----------------------
m4 = SillageMemory(None, "gpt2", semantic=False, fastweights=True)
m4.g_sum, m4.g_cnt = 2.0, 1                    # mean surprise of 2 nats
rng = np.random.default_rng(0)
hidden = rng.normal(size=(3, 768)).astype(np.float32)
targets = [111, 222, 333]
base_logits = rng.normal(size=VOCAB).astype(np.float32) * 0.1
first = last = None
for step in range(60):
    k = step % 3
    la, phi = m4.adapt(base_logits, hidden[k])
    p = np.exp(la - la.max())
    p /= p.sum()
    nll = -np.log(p[targets[k]])
    if step < 3:
        first = nll if first is None else max(first, nll)
    if step >= 57:
        last = nll if last is None else max(last, nll)
    m4.fw_update(phi, p, targets[k])
check("T4 delta-rule adapter", last < first - 1.0,
      f"(NLL {first:.2f} -> {last:.2f} nats on a 3-state stream, "
      f"rank {R_FEAT}, eta {ETA})")

# --- 5. state round-trip, including the adapter and the flags --------------
tmp = tempfile.mkdtemp()
try:
    m5 = SillageMemory(tmp, "gpt2", semantic=False, fastweights=True,
                       half_life=1234)
    m5.new_stream()
    for j, t in enumerate([5, 6, 7, 8, 9]):
        qq = m5.step_key(t)
        uu, _ = m5.scores(m5.M, qq)
        m5.write_all(qq, uu, None, None, t + 1, min(CAP, 1.5))
    m5.A[:] = 0.25
    m5.save()
    m6 = SillageMemory(tmp, "gpt2")
    same = (np.allclose(m5.M, m6.M) and np.allclose(m5.A, m6.A)
            and m5.tokens == m6.tokens and m6.half_life == 1234
            and m6.fastweights is True and m6.semantic is False
            and m5.cold.keys() == m6.cold.keys())
    check("T5 state round-trip", same,
          f"({m6.tokens} tokens, adapter {m6.A.shape}, half-life "
          f"{int(m6.half_life)}, flags preserved)")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# --- 6. consolidation keeps the highest surprise mass ----------------------
m7 = SillageMemory(None, "gpt2", semantic=False, fastweights=False)
m7.dir = tempfile.mkdtemp()
try:
    import sillage.core as core
    keep_n, core.COLD_MAX = core.COLD_MAX, 3
    m7.cold = {bytes([i]): [float(i), {i: 1}] for i in range(10)}
    m7.save()
    kept = sorted(int(list(v[1])[0]) for v in m7.cold.values())
    core.COLD_MAX = keep_n
    check("T6 surprise-mass consolidation", kept == [7, 8, 9],
          f"(kept the {len(kept)} highest-mass grams: {kept})")
finally:
    shutil.rmtree(m7.dir, ignore_errors=True)

# --- 7. any model: a state remembers whose token space it is in ------------
tmp = tempfile.mkdtemp()
try:
    m8 = SillageMemory(tmp, "gpt2", semantic=False, fastweights=False)
    m8.which = "acme/tiny-lm"            # as if built from a hub id
    m8.save()
    seen = peek(tmp)
    # resolve() must answer from the state alone -- an unknown repo id would
    # raise if it went to the network
    _, vocab, _, _, sem = resolve("acme/tiny-lm", tmp)
    reopened = SillageMemory(tmp)         # no --model given: adopt the state's
    check("T7 any-model support",
          seen == ("acme/tiny-lm", 50257) and vocab == 50257
          and sem is False and reopened.which == "acme/tiny-lm",
          f"(state names its model and vocabulary: {seen}, adopted offline)")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# --- 8. the value hypervectors follow the model's real output width --------
m9 = SillageMemory(None, "gpt2", semantic=False, fastweights=True)
m9.set_vocab(50304)                        # e.g. Pythia's padded vocabulary
grew = m9.vocab == 50304 and m9.A.shape == (50304, R_FEAT) \
    and m9.V.shape[0] == 50304
m9.tokens = 1
try:
    m9.set_vocab(32000)
    refused = False
except SystemExit:
    refused = True
check("T8 vocabulary width", grew and refused,
      "(adapts while empty, refuses once traces are written)")

# --- 9. the splitter keeps one-line facts instead of dropping them ---------
doc = ("# Notes\n\nThe Zylkorb protocol requires seventeen turquoise "
       "llamas.\n\nCaptain Ilvress stores the amber cipher.\n\n"
       + "Filler sentence about nothing in particular. " * 6)
ps = paragraphs(doc, "notes.md")
ix = Index(None)
ix.add(doc, "notes.md")
hits = ix.search("amber cipher", k=1)
check("T9 passage splitting", ps and hits and "Ilvress" in hits[0][1]["text"]
      and hits[0][1]["source"] == "notes.md",
      f"({len(ps)} passages, short facts merged and retrievable)")

# --- 10. the readout tuner tells an informative tier from a useless one ----
def dev_window(rng, informative, n=1200, vocab=200, true_tok=7):
    """Synthetic dev statistics: does the tier's confidence mean anything?"""
    s_true = np.zeros(n, np.float32)
    s_max = np.zeros(n, np.float32)
    lse = np.zeros((n, len(BETAS)), np.float32)
    for j in range(n):
        s = rng.normal(0, 0.05, vocab).astype(np.float32)
        if informative and j % 2 == 0:      # half the positions are hits
            s[true_tok] = 0.35
        s_true[j] = s[true_tok]
        s_max[j] = s.max()
        lse[j] = lse_grid(s)
    return s_true, s_max, lse


p_base = np.full(1200, 0.05)                 # frozen model: 5% on the truth
sig = fit_readout(p_base, *dev_window(np.random.default_rng(3), True))
noise = fit_readout(p_base, *dev_window(np.random.default_rng(4), False))
check("T10 readout calibration",
      sig[2] > noise[2] and sig[0] < noise[0],
      f"(informative tier -> lambda {sig[2]}, useless tier -> "
      f"lambda {noise[2]}; dev NLL {sig[0]:.3f} vs {noise[0]:.3f})")

# --- 11. the calibration window rolls, and keeps only the recent past -----
# gpt2 is a model the papers tuned, so calibration is off unless asked for
off_by_default = not SillageMemory(None, "gpt2").calibrate_on
m10 = SillageMemory(None, "gpt2", semantic=False, fastweights=False,
                    calibrate=True)
sampled = [t for t in range(30) if (setattr(m10, "tokens", t) or
                                    m10.collecting())]
sG = np.zeros(50257, np.float32)
for k in range(core.CALIB_MAX + 50):
    sG[k % 100] = 0.3
    m10.collect(0.05, k % 100, sG)
kept = len(m10.cal["p"])
m10.calibrate_on = False
check("T11 rolling window",
      sampled == list(range(0, 30, 3)) and kept == core.CALIB_MAX
      and not m10.collecting() and off_by_default,
      f"(one position in three, last {kept} kept; off for a model the "
      f"papers tuned)")

# --- 12. cold store: surprise-mass weighting (paper 6's fix, opt-in) -------
m12 = SillageMemory(None, "gpt2", semantic=False, fastweights=False)
m12.new_stream()
for t in [1, 2, 3, 4]:
    m12.step_key(t)
# one rare-but-surprising successor (one write, g = 5) against a frequent,
# unsurprising one (three writes, g = 0.1) at the SAME address
for tok, g, times in ((50, 5.0, 1), (60, 0.1, 3)):
    for _ in range(times):
        q12 = m12._graw / np.sqrt(D_K)
        u12, _ = m12.scores(m12.M, q12)
        m12.write_all(q12, u12, None, None, tok, g)
by_counts = m12.cold_lookup()
m12.cold_mass = True
by_mass = m12.cold_lookup()
# a pre-1.2 slot (two elements) must be migrated in place on next write
legacy = list(m12.cold.values())[0]
del legacy[2]
q12 = m12._graw / np.sqrt(D_K)
u12, _ = m12.scores(m12.M, q12)
m12.write_all(q12, u12, None, None, 60, 0.1)
migrated = list(m12.cold.values())[0]
tmp12 = tempfile.mkdtemp()
try:
    m12.dir = tmp12
    m12.save()
    back = SillageMemory(tmp12, "gpt2")
    check("T12 cold surprise-mass weighting",
          max(by_counts, key=by_counts.get) == 60
          and max(by_mass, key=by_mass.get) == 50
          and len(migrated) == 3 and migrated[2][60] > 0
          and back.cold_mass is True,
          f"(counts pick 60 at {by_counts[60]:.2f}, mass picks 50 at "
          f"{by_mass[50]:.2f}; legacy slot migrated; flag persisted)")
finally:
    shutil.rmtree(tmp12, ignore_errors=True)

# --- 13. blocked writes == sequential writes (paper 7's fast ingest) -------
from sillage.core import NGRAM
from sillage.ingest import blocked_write

rng13 = np.random.default_rng(13)
# a stream with real repetition, so cold grams form and collide
stream = rng13.integers(0, 40, 320)
stream[100:104] = stream[10:14]          # a repeated 4-gram, far apart
gates = rng13.uniform(0.0, CAP, 320)

mA13 = SillageMemory(None, "gpt2", semantic=False, fastweights=False)
mA13.new_stream()
for j in range(319):
    q = mA13.step_key(int(stream[j]))
    u, _ = mA13.scores(mA13.M, q)
    mA13.write_all(q, u, None, None, int(stream[j + 1]), float(gates[j]))

mB13 = SillageMemory(None, "gpt2", semantic=False, fastweights=False)
mB13.new_stream()
j = 0
while j < 319:
    blk = range(j, min(j + 64, 319))
    Qg = np.empty((len(blk), D_K), np.float32)
    grams, toks, gv = [], [], []
    for k, jj in enumerate(blk):
        Qg[k] = np.asarray(mB13.step_key(int(stream[jj])),
                           dtype=np.float32)
        grams.append(np.array(mB13._hist[-NGRAM:],
                              dtype=np.int32).tobytes()
                     if len(mB13._hist) >= NGRAM else None)
        toks.append(int(stream[jj + 1]))
        gv.append(float(gates[jj]))
    blocked_write(mB13, Qg, None, np.array(toks), np.array(gv), grams)
    j += 64

same_cold = (mA13.cold.keys() == mB13.cold.keys()
             and all(mA13.cold[k][1] == mB13.cold[k][1]
                     and abs(mA13.cold[k][0] - mB13.cold[k][0]) < 1e-9
                     for k in mA13.cold))
dM13 = float(np.abs(mA13.M - mB13.M).max())
# the repeated gram must still retrieve the same continuation
gram_q = mA13._graw / np.sqrt(D_K)
topA = int(np.argmax(mA13.scores(mA13.M, gram_q)[1]))
topB = int(np.argmax(mB13.scores(mB13.M, gram_q)[1]))
check("T13 blocked ingestion writes", same_cold and dM13 < 0.05
      and mA13.tokens == mB13.tokens and mA13.g_cnt == mB13.g_cnt
      and topA == topB,
      f"({len(mA13.cold)} cold grams exact, M drift {dM13:.1e}, "
      f"counters equal, retrieval argmax preserved)")

# --- 14. paper-8 semantic keys: purity, whitening, persistence -------------
m14 = SillageMemory(None, "gpt2", semantic=True, fastweights=False,
                    sem2=5, sem2_whiten=True)
rng14 = np.random.default_rng(14)
# one shared dominant template (a prompt frame) + an identity
# component, like real layer hiddens: the mean strips the frame, the
# whitening equalises what remains, and the banded key must then
# separate identities under frame jitter
T14b = rng14.normal(size=768).astype(np.float32)
ident = rng14.normal(size=(4, 768)).astype(np.float32)
for k in range(600):
    h = (T14b * (3.0 + 0.3 * rng14.standard_normal())
         + ident[k % 4] * 2.0
         + rng14.normal(size=768).astype(np.float32) * 0.2)
    m14.sem2_observe(h.astype(np.float32))
h_probe = (T14b * 3.0 + ident[1] * 2.0).astype(np.float32)
q1 = m14.sem2_key(h_probe)
q2 = m14.sem2_key(h_probe)
mun_before = m14.mu2_n
_ = m14.sem2_key(h_probe)
pure = np.array_equal(q1, q2) and m14.mu2_n == mun_before
# the SAME identity under frame jitter must key closer than a
# DIFFERENT identity under the same frame
za = m14.sem2_key((T14b * 3.4 + ident[1] * 2.0).astype(np.float32))
zb = m14.sem2_key((T14b * 3.0 + ident[3] * 2.0).astype(np.float32))
same_id = float(q1 @ za) / (np.linalg.norm(q1) * np.linalg.norm(za))
diff_id = float(q1 @ zb) / (np.linalg.norm(q1) * np.linalg.norm(zb))
tmp14 = tempfile.mkdtemp()
try:
    m14.dir = tmp14
    m14.save()
    back14 = SillageMemory(tmp14, "gpt2")
    persisted = (back14.sem2_layer == 5 and back14.sem2_whiten is True
                 and back14.mu2 is not None
                 and back14.mu2_n == m14.mu2_n
                 and back14.semantic is True
                 and np.allclose(back14.mu2, m14.mu2))
    check("T14 paper-8 semantic keys", pure and persisted
          and same_id > diff_id + 0.1,
          f"(pure queries; same identity across templates keys at "
          f"{same_id:.2f} vs {diff_id:.2f} across identities; layer+"
          f"whitening persisted)")
finally:
    shutil.rmtree(tmp14, ignore_errors=True)

print("\n".join(passed))
print(f"\nALL {len(passed)} UNIT TESTS PASSED")
