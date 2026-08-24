"""Generalized frozen-LM pass: any causal LM, any domain list.

Same sliding-window protocol as dump_base.py (window 1024, stride 512), plus
the base model's top-128 next-token log-probabilities at every position
(needed by the downstream cloze evaluation to compute mixed-argmax
predictions without re-running the LM).

Usage: python dump_any.py <gpt2|qwen> <domain> [<domain> ...]
Outputs (prefix "" for gpt2, "q_" for qwen):
  dumps/<prefix><domain>_h.npy    (N-1, hidden) fp16
  dumps/<prefix><domain>_lp.npy   (N-1,) fp32   log p(true next token)
  dumps/<prefix><domain>_topi.npy (N-1, 128) int32
  dumps/<prefix><domain>_topl.npy (N-1, 128) fp16
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
import sys
import time

import numpy as np
import torch

MODELS = {"gpt2": ("openai-community/gpt2", ""),
          "qwen": ("Qwen/Qwen3-0.6B", "q_")}
WINDOW, STRIDE, TOPK = 1024, 512, 128


@torch.no_grad()
def dump_domain(model, ids, hidden):
    N = len(ids)
    n = N - 1
    H = np.zeros((n, hidden), dtype=np.float16)
    LP = np.zeros(n, dtype=np.float32)
    TI = np.zeros((n, TOPK), dtype=np.int32)
    TL = np.zeros((n, TOPK), dtype=np.float16)
    done = np.zeros(n, dtype=bool)
    x = torch.tensor(ids, dtype=torch.long)
    a = 0
    while a < N - 1:
        w = min(WINDOW, N - a)
        out = model(x[a: a + w].unsqueeze(0), output_hidden_states=True)
        h = out.hidden_states[-1][0]
        logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)
        lo = 0 if a == 0 else WINDOW - STRIDE
        hi = min(w, n - a)
        if hi > lo:
            tv, tix = torch.topk(logprobs[lo:hi], TOPK, dim=-1)
            hh = h[lo:hi].to(torch.float16).numpy()
            for i in range(lo, hi):
                j = a + i
                if not done[j]:
                    H[j] = hh[i - lo]
                    LP[j] = float(logprobs[i, ids[j + 1]])
                    TI[j] = tix[i - lo].numpy().astype(np.int32)
                    TL[j] = tv[i - lo].numpy().astype(np.float16)
                    done[j] = True
        if a + w >= N:
            break
        a += STRIDE
    assert done.all(), f"{done.sum()}/{n} scored"
    return H, LP, TI, TL


def main():
    which = sys.argv[1]
    domains = sys.argv[2:]
    name, prefix = MODELS[which]
    torch.set_num_threads(os.cpu_count() or 4)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.eval()
    hidden = model.config.hidden_size
    os.makedirs("dumps", exist_ok=True)
    rp = os.path.join("dumps", "base_report.json")
    report = json.load(open(rp)) if os.path.exists(rp) else {}
    for d in domains:
        ids = np.load(os.path.join("data", f"{prefix}{d}_ids.npy"))
        t0 = time.time()
        H, LP, TI, TL = dump_domain(model, ids, hidden)
        for suffix, arr in [("h", H), ("lp", LP), ("topi", TI), ("topl", TL)]:
            np.save(os.path.join("dumps", f"{prefix}{d}_{suffix}.npy"), arr)
        report[f"{prefix}{d}"] = {
            "model": name, "tokens": int(len(ids)),
            "base_ppl": float(np.exp(-LP.mean())),
            "minutes": round((time.time() - t0) / 60, 1)}
        print(f"{prefix}{d}: N={len(ids)} PPL={report[f'{prefix}{d}']['base_ppl']:.2f} "
              f"({report[f'{prefix}{d}']['minutes']} min)", flush=True)
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
