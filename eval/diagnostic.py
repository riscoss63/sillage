"""Adjudicate the surprisingly large memory gains before reporting anything.

Checks on the aba stream (first 20k positions, k=32, tau=0.5, lam=0.2):
  1. recall@32   : how often the true next token is among neighbor values
  2. repeat rate : fraction of positions whose preceding 8-gram occurred
                   earlier in the stream (verbatim repetition of the corpus)
  3. gain split  : mean dNLL on repeated-8-gram positions vs novel positions
                   (is the memory copying repeated passages, or something odd?)
  4. shuffle control : neighbor VALUES randomly permuted -> gain must vanish
                   (validates the evaluation math; a leak would survive this)
  5. unigram cache   : p_mem = past-token unigram -> how much of the gain is
                   mere domain unigram adaptation
"""


# --- repo bootstrap (added by reorganize.py) ---
import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "requirements.txt")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
for _sub in ("", "pipeline", "memory", "fastweights", "eval", "figures"):
    _p = _os.path.join(_d, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_d)
# --- end bootstrap ---

import numpy as np

from memories import knn_neighbors, knn_p_true, load_domain

N_USE = 20_000
K, TAU, LAM = 32, 0.5, 0.2

ids, H, LP, vals = load_domain("aba")
H, LP, vals = H[:N_USE], LP[:N_USE], vals[:N_USE]
ids = ids[: N_USE + 1]
p_base = np.exp(LP)

nd, nv = knn_neighbors(H, vals)

# 1. recall@32
hit = (nv == vals[:, None]).any(axis=1)
print(f"recall@32: {hit.mean():.3f}")

# 2. context-repeat rate (8-gram)
seen = set()
rep = np.zeros(len(vals), dtype=bool)
for j in range(len(vals)):
    if j >= 7:
        gram = ids[j - 7: j + 1].tobytes()
        rep[j] = gram in seen
        seen.add(gram)
print(f"repeated-8gram rate: {rep.mean():.3f}")

# 3. kNN gain, split by repetition
p_knn, _ = knn_p_true(nd, nv, vals, K, TAU)
p = LAM * p_knn + (1 - LAM) * p_base
dnll = (-np.log(np.maximum(p_base, 1e-30))) - (-np.log(np.maximum(p, 1e-30)))
print(f"kNN dNLL overall: {dnll.mean():+.4f} | on repeated: "
      f"{dnll[rep].mean():+.4f} ({rep.sum()} pos) | on novel: "
      f"{dnll[~rep].mean():+.4f} ({(~rep).sum()} pos)")

# 4. shuffled-values control
rng = np.random.default_rng(0)
nv_shuf = nv.copy()
flat = nv_shuf[nv_shuf >= 0]
nv_shuf[nv_shuf >= 0] = rng.permutation(flat)
p_shuf, _ = knn_p_true(nd, nv_shuf, vals, K, TAU)
ps = LAM * p_shuf + (1 - LAM) * p_base
dnll_s = (-np.log(np.maximum(p_base, 1e-30))) - (-np.log(np.maximum(ps, 1e-30)))
print(f"shuffle control dNLL: {dnll_s.mean():+.4f} (must be ~<= 0)")

# 5. unigram-cache baseline
counts = np.zeros(50257, dtype=np.float64)
p_uni = np.zeros(len(vals))
tot = 0
for j in range(len(vals)):
    p_uni[j] = counts[vals[j]] / tot if tot > 0 else 0.0
    counts[ids[j]] += 1        # token consumed at step j is x[j] ... then
    tot += 1                   # x[j+1] becomes visible only after scoring
best = None
for lam in [0.05, 0.1, 0.2, 0.3]:
    pu = lam * p_uni + (1 - lam) * p_base
    d = ((-np.log(np.maximum(p_base, 1e-30)))
         - (-np.log(np.maximum(pu, 1e-30)))).mean()
    if best is None or d > best[1]:
        best = (lam, d)
print(f"unigram-cache dNLL: {best[1]:+.4f} (lam={best[0]})")

# extra: where do kNN gains come from? top tokens by total gain
gain_by_tok = {}
for j in np.argsort(-dnll)[:2000]:
    gain_by_tok[int(vals[j])] = gain_by_tok.get(int(vals[j]), 0.0) + float(dnll[j])
top = sorted(gain_by_tok.items(), key=lambda kv: -kv[1])[:15]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
print("top gain tokens:", [(tok.decode([t]).replace("\n", "\\n"), round(g, 1))
                           for t, g in top])
