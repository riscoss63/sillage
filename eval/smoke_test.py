"""End-to-end smoke test of memories.py on synthetic data where memory must
win: a repeated 40-token phrase, hidden states tied to the current token,
and a base LM that assigns p = 0.01 to every true token. If kNN and BHD do
not both improve NLL substantially here, the pipeline is broken."""


# --- repo bootstrap: run this script from anywhere ---
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

import os

import numpy as np

os.makedirs("data", exist_ok=True)
os.makedirs("dumps", exist_ok=True)

rng = np.random.default_rng(0)
N, VOCAB_USED = 3000, 500
phrase = rng.integers(0, VOCAB_USED, size=40)
ids = np.concatenate([phrase for _ in range(N // 40 + 1)])[:N]
noise_pos = rng.random(N) < 0.05
ids[noise_pos] = rng.integers(0, VOCAB_USED, size=noise_pos.sum())
ids = ids.astype(np.int32)

E = rng.normal(size=(VOCAB_USED, 768)).astype(np.float32)
H = E[ids[:-1]] + 0.05 * rng.normal(size=(N - 1, 768)).astype(np.float32)
LP = np.full(N - 1, np.log(0.01), dtype=np.float32)

np.save("data/toy_ids.npy", ids)
np.save("dumps/toy_h.npy", H.astype(np.float16))
np.save("dumps/toy_lp.npy", LP)

from memories import eval_domain

out = eval_domain("toy")
ok_knn = out["knn"]["dnll_test"] > 1.0
ok_bhd = max(out["bhd_dense"]["dnll_test"], out["bhd_sparse"]["dnll_test"]) > 0.5
print("SMOKE:", "PASS" if (ok_knn and ok_bhd) else "FAIL",
      f"(knn dNLL {out['knn']['dnll_test']:.2f}, "
      f"bhd best dNLL "
      f"{max(out['bhd_dense']['dnll_test'], out['bhd_sparse']['dnll_test']):.2f})")
