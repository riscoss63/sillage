"""Sillage demo: give a frozen GPT-2 a 4.2 MB Hebbian memory and watch it read.

    python demo.py [your_text_file.txt]

Streams the text once (left to right, online). At every position the frozen
LM predicts the next token; Sillage mixes in its memory of what it has already
read; then the observed token is written into the memory, gated by the LM's
own surprise. No gradients, no fine-tuning, constant 4.2 MB of storage.

With no argument, downloads "Alice's Adventures in Wonderland". For the full
effect, feed it a repetitive technical document the base model has never
seen (documentation, a codebase's docs, your own notes). Runtime: a few
minutes on CPU for ~30k tokens.

Reported: base vs Sillage perplexity on the last 80% of the stream (the first
20% calibrates the mixing weight's abstention threshold), plus example
completions the memory fixed.
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

import sys

import numpy as np
import torch

MAX_TOKENS = 30_000
WINDOW, STRIDE = 1024, 512
D_K, D_V, NGRAM = 4096, 256, 4
BETA, LAM, CAP, THR_Q = 40.0, 0.3, 5.0, 0.75
SEED = 7001


def load_text():
    if len(sys.argv) > 1:
        return open(sys.argv[1], encoding="utf-8", errors="replace").read()
    print("No file given -- downloading Alice in Wonderland as demo text.")
    from data_prep import fetch, normalize_newlines, strip_gutenberg, \
        unwrap_paragraphs
    return unwrap_paragraphs(normalize_newlines(strip_gutenberg(fetch(
        ["https://www.gutenberg.org/cache/epub/11/pg11.txt"]))))


def main():
    torch.set_num_threads(torch.get_num_threads())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.eval()
    vocab = model.config.vocab_size

    text = load_text()
    ids = np.array(tok.encode(text), dtype=np.int64)[:MAX_TOKENS]
    n = len(ids) - 1
    dev_end = int(0.2 * n)
    print(f"{n + 1} tokens; calibrating on the first {dev_end}, "
          f"scoring the rest.\n")

    rng = np.random.default_rng(SEED)
    V = ((rng.integers(0, 2, size=(vocab, D_V)) * 2.0 - 1.0)
         / np.sqrt(D_V)).astype(np.float32)
    T = (np.random.default_rng(SEED + 1).integers(
        0, 2, size=(vocab, D_K), dtype=np.int8) * 2 - 1)
    M = np.zeros((D_K, D_V), dtype=np.float32)
    g_raw = np.ones(D_K, dtype=np.float32)
    inv = 1.0 / np.sqrt(D_K)

    dev_smax, thr = [], None
    nll_base, nll_sillage, n_scored, n_active = 0.0, 0.0, 0, 0
    examples = []
    x = torch.tensor(ids)
    a, done = 0, 0
    with torch.no_grad():
        while a < n:
            w = min(WINDOW, len(ids) - a)
            logprobs = torch.log_softmax(
                model(x[a:a + w].unsqueeze(0)).logits[0].float(), -1).numpy()
            lo = 0 if a == 0 else WINDOW - STRIDE
            for i in range(lo, w):
                j = a + i
                if j >= n:
                    break
                truth = int(ids[j + 1])
                lp = float(logprobs[i, truth])
                # --- Sillage read
                g_raw = np.roll(g_raw, 1)
                g_raw *= T[ids[j]]
                if j >= NGRAM:
                    g_raw *= np.roll(T[ids[j - NGRAM]], NGRAM)
                q = g_raw * inv
                u = M.T @ q
                s = (V @ u) / (np.linalg.norm(u) + 1e-8)
                smax = float(s.max())
                if j < dev_end:
                    dev_smax.append(smax)
                else:
                    if thr is None:
                        thr = float(np.quantile(dev_smax, THR_Q))
                    m = BETA * s
                    mx = m.max()
                    lse = mx + np.log(np.exp(m - mx).sum())
                    p_mem = np.exp(BETA * s[truth] - lse)
                    active = smax >= thr
                    p = (LAM * p_mem + (1 - LAM) * np.exp(lp)) if active \
                        else np.exp(lp)
                    nll_base += -lp
                    nll_sillage += -np.log(max(p, 1e-30))
                    n_scored += 1
                    n_active += active
                    if active and len(examples) < 8:
                        base_pred = int(np.argmax(logprobs[i]))
                        cand = set(np.argpartition(-logprobs[i], 20)[:20])
                        cand |= set(np.argpartition(-s, 20)[:20])
                        pm_all = np.exp(BETA * s - lse)
                        pred = max(cand, key=lambda t: LAM * pm_all[t]
                                   + (1 - LAM) * np.exp(logprobs[i, t]))
                        if pred == truth != base_pred:
                            ctx = tok.decode(ids[max(0, j - 10):j + 1])
                            examples.append((ctx[-60:], tok.decode([truth]),
                                             tok.decode([base_pred])))
                # --- Sillage write (surprise-gated amplitude update)
                g = min(CAP, max(0.0, -lp))
                amp = max(0.0, float(u @ V[truth]))
                M += (np.sqrt(amp * amp + g) - amp) * q[:, None] * V[truth][None, :]
                done += 1
            if done % 5000 < STRIDE and done > 0:
                print(f"  ... {done}/{n} tokens read", flush=True)
            if a + w >= len(ids):
                break
            a += STRIDE

    b, s_ = nll_base / n_scored, nll_sillage / n_scored
    print(f"\nfrozen GPT-2 perplexity : {np.exp(b):8.2f}")
    print(f"GPT-2 + Sillage perplexity : {np.exp(s_):8.2f}   "
          f"(dNLL {b - s_:+.4f} nats; memory active on "
          f"{100 * n_active / n_scored:.0f}% of positions; 4.2 MB, no gradients)")
    if examples:
        print("\ncompletions the memory fixed (context -> truth | frozen said):")
        for ctx, tr, bp in examples:
            print(f"  ...{ctx!r} -> {tr!r}  | {bp!r}")
    print("\nDetails and the full evaluation suite: see README.md / the paper.")


if __name__ == "__main__":
    main()
