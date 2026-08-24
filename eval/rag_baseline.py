"""RAG-lite baseline: retrieve-and-concatenate rescoring.

For each 256-token block (starting once >= 1024 tokens of past exist):
  - query   = mean of the last 128 hidden states (from the shared dump)
  - index   = mean-pooled hidden state of every past 128-token chunk that
              lies fully BEFORE the 512-token local context (causal, and not
              already visible to the model)
  - rescore = one GPT-2 forward over [retrieved chunk (128) | local context
              (512) | block (256)]; block tokens are scored conditioned on
              the retrieved passage.

Reported on the same dev/test split as the other methods, restricted to
RAG-covered positions (j >= 1024), with the base NLL recomputed on exactly
those positions. Two readouts: pure RAG, and lambda-interpolated with the
base (lambda tuned on dev).

Output: results/rag_<domain>.json
"""


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

import json
import os
import time

import numpy as np
import torch

DATA, DUMPS, RESULTS = "data", "dumps", "results"
CHUNK, CTX, BLOCK = 128, 512, 256
DEV_FRAC = 0.20
LAMS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@torch.no_grad()
def run_domain(model, d):
    ids = np.load(os.path.join(DATA, f"{d}_ids.npy"))
    H = np.load(os.path.join(DUMPS, f"{d}_h.npy")).astype(np.float32)
    LP = np.load(os.path.join(DUMPS, f"{d}_lp.npy"))
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
    N = len(ids)

    # chunk index (mean-pooled hidden states)
    n_chunks = (N - 1) // CHUNK
    C = np.stack([H[c * CHUNK:(c + 1) * CHUNK].mean(axis=0)
                  for c in range(n_chunks)])
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8

    x = torch.tensor(ids, dtype=torch.long)
    lp_rag = {}
    t0 = time.time()
    b = 1024
    while b + 1 < N:
        blk = min(BLOCK, N - 1 - b)
        qeme = H[b - CHUNK:b].mean(axis=0)
        qeme /= np.linalg.norm(qeme) + 1e-8
        last_chunk = (b - CTX) // CHUNK          # chunks fully before context
        sims = C[:last_chunk] @ qeme
        c_star = int(np.argmax(sims))
        retrieved = x[c_star * CHUNK:(c_star + 1) * CHUNK]
        # include x[b + blk] so the last block position has its target inside
        inp = torch.cat([retrieved, x[b - CTX:b + blk + 1]]).unsqueeze(0)
        out = model(inp)
        logprobs = torch.log_softmax(out.logits[0], dim=-1)
        off = CHUNK + CTX                    # inp position of x[b]
        for t in range(blk):
            j = b + t                        # logits[off + t] predict x[j + 1]
            lp_rag[j] = float(logprobs[off + t, ids[j + 1]])
        b += blk
    dt = (time.time() - t0) / 60

    pos = np.array(sorted(lp_rag.keys()))
    lr = np.array([lp_rag[j] for j in pos], dtype=np.float32)
    lb = LP[pos]
    n_pred = N - 1
    dev_mask = pos < int(DEV_FRAC * n_pred)
    test_mask = ~dev_mask

    p_base, p_rag = np.exp(lb), np.exp(lr)
    best = None
    for lam in LAMS:
        p = lam * p_rag + (1 - lam) * p_base
        nll_dev = float(-np.log(np.maximum(p[dev_mask], 1e-30)).mean())
        if best is None or nll_dev < best["lam_nll_dev"]:
            best = {"lam": lam, "lam_nll_dev": nll_dev,
                    "nll_test": float(-np.log(np.maximum(p[test_mask],
                                                         1e-30)).mean())}
    out = {
        "domain": d, "positions": int(len(pos)), "minutes": round(dt, 1),
        "base_nll_test_same_positions": float(-lb[test_mask].mean()),
        "rag_pure_nll_test": float(-lr[test_mask].mean()),
        "rag_interp": best,
    }
    with open(os.path.join(RESULTS, f"rag_{d}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"{d}: base {out['base_nll_test_same_positions']:.4f} | "
          f"RAG pure {out['rag_pure_nll_test']:.4f} | "
          f"RAG interp {best['nll_test']:.4f} (lam={best['lam']}) "
          f"[{dt:.1f} min]", flush=True)
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(os.cpu_count() or 4)
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.eval()
    for d in ["relativity", "alice", "bhd"]:
        run_domain(model, d)


if __name__ == "__main__":
    main()
