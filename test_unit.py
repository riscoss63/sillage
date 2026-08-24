"""Fast tests of the mechanisms themselves -- numpy only, no model, ~5 s.

These check the four papers' rules where they can be checked exactly: the
Hebbian read/write, the square-root (amplitude) encoding, leaky forgetting,
the delta rule at the readout, consolidation by surprise mass, the state
round-trip and the passage splitter. `test_sillage.py` is the slow
end-to-end complement that actually runs a frozen GPT-2.

    python test_unit.py
"""

import os
import shutil
import tempfile

import numpy as np

from sillage.core import CAP, D_K, D_V, ETA, R_FEAT, SillageMemory
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

# --- 7. the splitter keeps one-line facts instead of dropping them ---------
doc = ("# Notes\n\nThe Zylkorb protocol requires seventeen turquoise "
       "llamas.\n\nCaptain Ilvress stores the amber cipher.\n\n"
       + "Filler sentence about nothing in particular. " * 6)
ps = paragraphs(doc, "notes.md")
ix = Index(None)
ix.add(doc, "notes.md")
hits = ix.search("amber cipher", k=1)
check("T7 passage splitting", ps and hits and "Ilvress" in hits[0][1]["text"]
      and hits[0][1]["source"] == "notes.md",
      f"({len(ps)} passages, short facts merged and retrievable)")

print("\n".join(passed))
print(f"\nALL {len(passed)} UNIT TESTS PASSED")
