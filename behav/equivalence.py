"""Context-equivalence curve (axe 3 / paper 6): how many tokens of context
does the bare model need to match model+memory at a small context cap?

On the papers state (GPT-2, warm on the four preprints), a held-out tail
segment is teacher-forced under context caps C in {32..1024} using sliding
windows (stride C/2, scored positions have context in [C/2, C)), bare vs
memory-augmented -- same positions, same forwards, nothing written. The
headline is C*: the cap at which the bare model's NLL matches the
augmented model's NLL at C=64. "X MB of state is worth (C*-64) tokens of
context on this regime."

    python equivalence.py [--segment 6000]
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break
sys.path.insert(0, HERE)
ROOT = _p

from sillage import Sillage  # noqa: E402

CAPS = (32, 64, 128, 256, 512, 1024)
S0 = 1024              # first scored position: every cap has full headroom


def score_cap(s, ids, C):
    """(nll_bare, nll_mem, count) over scored positions q >= S0, cap C."""
    import torch
    tok, model = s.load_model()
    mem = s.mem
    thrG, thrS = mem.thresholds()
    need_h = mem.semantic or mem.fastweights
    stride = C // 2
    n = len(ids)
    nll_b = nll_m = 0.0
    cnt = 0
    mem.new_stream()
    for t in ids[:S0]:
        mem.step_key(int(t))
    q = S0
    w_start = S0 - stride
    logits = hs = None
    with torch.no_grad():
        while q < n - 1:
            if logits is None or q - w_start >= C:
                if q - w_start >= C:
                    w_start += stride * ((q - w_start - C) // stride + 1)
                out = model(torch.tensor([ids[w_start:w_start + C]],
                                         device=s.device),
                            output_hidden_states=need_h)
                logits = out.logits[0].float().cpu().numpy()
                hs = (out.hidden_states[-1][0].float().cpu().numpy()
                      if need_h else None)
            i = q - w_start - 1
            truth = int(ids[q])
            lb = logits[i]
            mx = lb.max()
            lp = float(lb[truth] - (mx + np.log(np.exp(lb - mx).sum())))
            la, _ = mem.adapt(lb, hs[i] if hs is not None else None)
            p_ad = np.exp(la - la.max())
            p_ad /= p_ad.sum()
            qk = mem._graw / np.sqrt(4096)
            _, sG = mem.scores(mem.M, qk)
            p = mem.mix_true(float(p_ad[truth]), sG, truth, None,
                             mem.cold_lookup(truth), thrG, thrS)
            nll_b += -lp
            nll_m += -np.log(max(p, 1e-30))
            cnt += 1
            mem.step_key(truth)
            q += 1
    return nll_b / cnt, nll_m / cnt, cnt


def _papers_state():
    p = os.path.join(ROOT, "papers_state", "memory")
    if not os.path.exists(os.path.join(p, "state.npz")):
        raise SystemExit("papers state not found (states are not "
                         "shipped) -- build it:\n"
                         "  sillage papers --with-memory")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", type=int, default=6000)
    a = ap.parse_args()

    s = Sillage(model="gpt2",
                state=_papers_state(),
                quiet=True)
    tok, _ = s.load_model()
    text = open(os.path.join(ROOT, "papers_state", "corpus.txt"),
                encoding="utf-8", errors="replace").read()
    ids = tok.encode(text)[-a.segment:]
    print(f"segment: {len(ids)} tokens (fin du corpus des papiers), "
          f"positions notees >= {S0}", flush=True)

    R = {"caps": {}}
    for C in CAPS:
        b, m, cnt = score_cap(s, ids, C)
        R["caps"][C] = {"nll_bare": b, "nll_mem": m,
                        "ppl_bare": float(np.exp(b)),
                        "ppl_mem": float(np.exp(m)), "count": cnt}
        print(f"  C={C:5d} : nu {np.exp(b):7.2f} PPL | +memoire "
              f"{np.exp(m):7.2f} PPL  ({cnt} positions)", flush=True)

    # headline: bare context needed to match memory@64
    target = R["caps"][64]["nll_mem"]
    xs = [np.log2(c) for c in CAPS]
    ys = [R["caps"][c]["nll_bare"] for c in CAPS]
    cstar = None
    for k in range(len(CAPS) - 1):
        y0, y1 = ys[k], ys[k + 1]
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            f = (y0 - target) / (y0 - y1)
            cstar = float(2 ** (xs[k] + f * (xs[k + 1] - xs[k])))
            break
    R["target_nll_mem_at_64"] = target
    R["c_star_bare"] = cstar
    if cstar:
        print(f"\nC* : le modele nu a besoin de ~{cstar:.0f} tokens de "
              f"contexte pour egaler memoire+64 tokens -> l'etat vaut "
              f"~{cstar - 64:.0f} tokens de contexte sur ce regime.")
    else:
        print("\nC* hors de la grille : le modele nu ne rejoint pas "
              "memoire+64 tokens meme a 1024 -- l'etat vaut plus que "
              "960 tokens de contexte sur ce regime.")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", "equivalence_gpt2.json")
    json.dump(R, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
