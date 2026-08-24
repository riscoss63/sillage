"""Step 4 -- fast weights: a surprise-gated DELTA-RULE readout adapter.

Mechanism: adapted logits l' = l_base + A phi(h), with phi a FIXED random
projection of the frozen LM's hidden state (r = 256) and A (vocab x r)
updated online by the local delta rule

    A <- (1 - delta) A + eta * g * (onehot(x_{j+1}) - p') outer phi(h_j)

(the exact CE gradient AT THE READOUT -- no error transported through any
layer; g = clip(-ln p_base, 0, 5) is the usual surprise gate). Prequential
protocol: every position is scored BEFORE the update; dev = first 20% picks
(eta, gating, decay); test = last 80%.

One live pass evaluates all configurations simultaneously (they share the
forward). Usage: python fastweights.py <gpt2> <domain>
Output: results/fastweights_<domain>.json
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

import json
import sys
import time

import numpy as np
import torch

R_FEAT = 256
CAP = 5.0
WINDOW, STRIDE = 1024, 512
CONFIGS = [  # (name, eta, gated, half_life_tokens or None)
    ("eta0.1_gated", 0.1, True, None),
    ("eta0.3_gated", 0.3, True, None),
    ("eta1.0_gated", 1.0, True, None),
]
SEG = 5_000


@torch.no_grad()
def main(which, domain):
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.eval()
    vocab = model.config.vocab_size
    hidden = model.config.hidden_size
    ids = np.load(f"data/{domain}_ids.npy").astype(np.int64)
    n = len(ids) - 1
    rng = np.random.default_rng(7010)
    Rf = rng.normal(size=(hidden, R_FEAT)).astype(np.float32) / np.sqrt(hidden)

    A = {c[0]: np.zeros((vocab, R_FEAT), dtype=np.float32) for c in CONFIGS}
    nll_base = np.zeros(n, dtype=np.float32)
    nll_cfg = {c[0]: np.zeros(n, dtype=np.float32) for c in CONFIGS}
    x = torch.tensor(ids)
    a, done = 0, 0
    t0 = time.time()
    while a < n:
        w = min(WINDOW, len(ids) - a)
        out = model(x[a:a + w].unsqueeze(0), output_hidden_states=True)
        logits = out.logits[0].float().numpy()
        hs = out.hidden_states[-1][0].float().numpy()
        lo = 0 if a == 0 else WINDOW - STRIDE
        for i in range(lo, w):
            j = a + i
            if j >= n:
                break
            truth = int(ids[j + 1])
            lb = logits[i]
            lpb = lb - (lb.max() + np.log(np.exp(lb - lb.max()).sum()))
            nll_base[j] = -lpb[truth]
            g = min(CAP, max(0.0, float(-lpb[truth])))
            phi = hs[i] @ Rf
            phi /= np.linalg.norm(phi) + 1e-8
            for name, eta, gated, hl in CONFIGS:
                la = lb + A[name] @ phi
                m = la.max()
                p = np.exp(la - m)
                p /= p.sum()
                nll_cfg[name][j] = -np.log(max(p[truth], 1e-30))
                step = eta * (g if gated else 1.0)
                if hl:
                    A[name] *= 0.5 ** (1.0 / hl)
                if step > 0:
                    A[name] -= step * np.outer(p, phi)
                    A[name][truth] += step * phi
            done += 1
        if done and done % 8000 < STRIDE:
            print(f"  ... {done}/{n} ({(time.time()-t0)/60:.0f} min)",
                  flush=True)
        if a + w >= len(ids):
            break
        a += STRIDE

    n_dev = int(0.2 * n)
    dev = np.arange(n) < n_dev
    test = ~dev
    base_t = float(nll_base[test].mean())
    out_j = {"domain": domain, "n": int(n), "r": R_FEAT,
             "base_nll_test": base_t,
             "adapter_bytes": int(vocab * R_FEAT * 4)}
    best = None
    for name, *_ in CONFIGS:
        dv = float(nll_cfg[name][dev].mean())
        tv = float(nll_cfg[name][test].mean())
        segs = [float((nll_base[s:s + SEG] - nll_cfg[name][s:s + SEG]).mean())
                for s in range(0, n, SEG)]
        out_j[name] = {"nll_dev": dv, "nll_test": tv,
                       "dnll_test": float(base_t - tv), "segments": segs}
        print(f"  {name}: dev {dv:.4f} | test dNLL {base_t - tv:+.4f}",
              flush=True)
        if best is None or dv < best[1]:
            best = (name, dv)
    winner = best[0]
    d = nll_base[test] - nll_cfg[winner][test]
    nb = len(d) // 512
    blocks = [d[k * 512:(k + 1) * 512] for k in range(nb)]
    rb = np.random.default_rng(3)
    means = [np.concatenate([blocks[k] for k in rb.integers(0, nb, nb)]).mean()
             for _ in range(1000)]
    out_j["winner"] = winner
    out_j["winner_paired_vs_base"] = {
        "mean": float(d.mean()),
        "ci95": [float(np.quantile(means, 0.025)),
                 float(np.quantile(means, 0.975))],
        "p_better": float((np.array(means) > 0).mean())}
    print(f"{domain} WINNER {winner}: test dNLL {d.mean():+.4f} "
          f"CI {out_j['winner_paired_vs_base']['ci95']}", flush=True)
    with open(f"results/fastweights_{domain}.json", "w") as f:
        json.dump(out_j, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
