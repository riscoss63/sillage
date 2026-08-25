"""One pass of frozen GPT-2 (124M) over each domain stream.

Sliding-window scoring (window 1024, stride 512): every scored position has
at least 512 tokens of context (except the very first window). For each
position j (predicting token x[j+1]) we dump:
  - h[j]        : final hidden state (fp16, 768) -> keys for all memories
  - lp_true[j]  : log p_LM(x[j+1] | window context) (fp32) -> base NLL and
                  interpolation
No memory method gets anything the others do not: they all consume exactly
these dumps.

Outputs per domain: dumps/<d>_h.npy (N-1 x 768 fp16), dumps/<d>_lp.npy,
plus base perplexity in dumps/base_report.json. NOTE: base_report's PPL is
over the WHOLE stream (dev+test); the papers' tables report base_ppl_test
(the last 80% only) -- do not mix the two numbers.
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

DATA, DUMPS = "data", "dumps"
WINDOW, STRIDE = 1024, 512
DOMAINS = ["relativity", "alice", "bhd", "aba"]


@torch.no_grad()
def dump_domain(model, ids, device="cpu"):
    N = len(ids)
    n_pred = N - 1
    H = np.zeros((n_pred, 768), dtype=np.float16)
    LP = np.zeros(n_pred, dtype=np.float32)
    done = np.zeros(n_pred, dtype=bool)
    x = torch.tensor(ids, dtype=torch.long, device=device)

    a = 0
    while a < N - 1:
        w = min(WINDOW, N - a)
        inp = x[a: a + w].unsqueeze(0)
        out = model(inp, output_hidden_states=True)
        h = out.hidden_states[-1][0]                     # (w, 768)
        logprobs = torch.log_softmax(out.logits[0], dim=-1)  # (w, V)
        lo = 0 if a == 0 else WINDOW - STRIDE
        for i in range(lo, w):
            j = a + i          # logits at in-window i predict x[j + 1]
            if j >= n_pred:
                break
            if not done[j]:
                H[j] = h[i].to(torch.float16).numpy()
                LP[j] = float(logprobs[i, ids[j + 1]])
                done[j] = True
        if a + w >= N:
            break
        a += STRIDE
    assert done.all(), f"{done.sum()}/{n_pred} positions scored"
    return H, LP


def main():
    os.makedirs(DUMPS, exist_ok=True)
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(os.cpu_count() or 4)
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.eval()

    report = {}
    for d in DOMAINS:
        ids = np.load(os.path.join(DATA, f"{d}_ids.npy"))
        t0 = time.time()
        H, LP = dump_domain(model, ids)
        np.save(os.path.join(DUMPS, f"{d}_h.npy"), H)
        np.save(os.path.join(DUMPS, f"{d}_lp.npy"), LP)
        ppl = float(np.exp(-LP.mean()))
        report[d] = {"tokens": int(len(ids)), "base_ppl": ppl,
                     "base_nll": float(-LP.mean()),
                     "minutes": round((time.time() - t0) / 60, 1)}
        print(f"{d}: N={len(ids)} base PPL={ppl:.2f} "
              f"({report[d]['minutes']} min)", flush=True)

    with open(os.path.join(DUMPS, "base_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
