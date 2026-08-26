"""Calibrate the readout (beta, lambda, threshold) FOR the 1.7B target,
then re-run config C with old vs calibrated settings on identical prompts.

Protocol: the tool's own tuning loop, run read-only. Dev window = the first
20% of the lifetime stream (all inside document 1); at one position in three
the 1.7B is teacher-forced and we record (p_base_true, sG[truth], sG.max,
lse_grid) -- exactly what `SillageMemory.collect` stores -- then
`fit_readout` searches the published grids. Nothing is written to the
memory and nothing is saved to disk. The bench prompts sampled later in the
stream act as test; prompts overlapping the dev window are flagged.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bench_qwen import DOCS, STATE, TARGET, UNSEEN, run, seen_prompts  # noqa
from sillage_drafter import SpeculativeSillage                          # noqa

for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break
from sillage.core import BETAS, fit_readout, lse_grid                   # noqa

DEV_FRACTION = 0.20
STRIDE_COLLECT = 3            # CALIB_EVERY of the tool
CHUNK = 512


def nll_under(settings, p, gt, gm, gl):
    """Dev NLL of one (beta, lam, thr_q) triple on the collected window."""
    beta, lam, q = settings
    bi = BETAS.index(beta)
    p_mem = np.exp(beta * gt - gl[:, bi])
    mask = (np.ones(len(p), bool) if q is None
            else gm >= np.quantile(gm, q))
    mix = np.where(mask, lam * p_mem + (1 - lam) * p, p)
    return float(-np.log(np.maximum(mix, 1e-30)).mean())


def collect_dev(engine, n_dev):
    """Teacher-force the 1.7B over the dev window; return the dev arrays."""
    torch = engine.torch
    m = engine.mem
    text = open(DOCS[0], encoding="utf-8", errors="replace").read()
    ids = engine.tok.encode(text)[:n_dev + 1]
    m.new_stream()
    p_list, gt_list, gm_list, gl_list = [], [], [], []
    past = None
    pos = 0
    with torch.no_grad():
        for a in range(0, len(ids) - 1, CHUNK):
            chunk = ids[a:a + CHUNK]
            out = engine.model(
                torch.tensor([chunk], device=engine.device),
                past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0].float().cpu().numpy()
            for i, tok in enumerate(chunk):
                j = a + i
                if j + 1 >= len(ids):
                    break
                q = m.step_key(int(tok))
                if pos % STRIDE_COLLECT == 0:
                    truth = int(ids[j + 1])
                    lb = logits[i]
                    mx = lb.max()
                    p_true = float(np.exp(lb[truth] - mx)
                                   / np.exp(lb - mx).sum())
                    _, sG = m.scores(m.M, q)
                    p_list.append(p_true)
                    gt_list.append(float(sG[truth]))
                    gm_list.append(float(sG.max()))
                    gl_list.append(lse_grid(sG))
                pos += 1
    return (np.array(p_list, np.float64), np.array(gt_list, np.float32),
            np.array(gm_list, np.float32), np.array(gl_list, np.float32))


def main():
    print("=== chargement 1.7B + etat 0.6B (lecture seule) ===")
    eng = SpeculativeSillage(STATE, target_hub=TARGET,
                             memory_in_target=True, fastweights=False)
    m = eng.mem
    old = (m.beta_G, m.lam_G, m.thr_qG)
    n_dev = int(DEV_FRACTION * m.tokens)
    print(f"fenetre dev: {n_dev} premiers tokens (sur {m.tokens} a vie)")

    print("=== collecte teleforcee sur la fenetre dev ===")
    p, gt, gm, gl = collect_dev(eng, n_dev)
    print(f"{len(p)} observations (1 position sur {STRIDE_COLLECT})")

    nll_base = float(-np.log(np.maximum(p, 1e-30)).mean())
    nll_old = nll_under(old, p, gt, gm, gl)
    nll_fit, beta, lam, q, _ = fit_readout(p, gt, gm, gl)
    new = (beta, lam, q)
    print(f"NLL dev  | 1.7B seul: {nll_base:.4f} | reglages 0.6B "
          f"(beta {old[0]:g}, lam {old[1]:g}, q {old[2]}): {nll_old:.4f} "
          f"| calibres (beta {beta:g}, lam {lam:g}, q {q}): {nll_fit:.4f}")
    print(f"gain de calibration sur dev: {nll_old - nll_fit:+.4f} nats")

    seen = seen_prompts(10)
    unseen = [(x, None) for x in UNSEEN]
    # flag prompts overlapping the dev window (dev covers ~the first part
    # of doc 1; convert its token span to a word count for the flag)
    text1 = open(DOCS[0], encoding="utf-8", errors="replace").read()
    dev_words = len(engine_decode_words(eng, text1, n_dev))
    all_text = "\n\n".join(open(pth, encoding="utf-8",
                                errors="replace").read() for pth in DOCS)
    words = [w for w in all_text.split() if w]
    flags = []
    for k, (prompt, _t) in enumerate(seen):
        first = prompt.split()[:6]
        idx = find_words(words, first)
        flags.append(idx is not None and idx < dev_words)
    print(f"prompts chevauchant la fenetre dev: "
          f"{[i for i, f in enumerate(flags) if f]} "
          f"(dev ~ {dev_words} premiers mots)")

    report = {"dev_obs": len(p), "nll_base": nll_base, "nll_old": nll_old,
              "nll_new": nll_fit, "old": list(old),
              "new": [beta, lam, q], "dev_flags": flags}

    print("\n=== C avec reglages herites du 0.6B (re-mesure, meme process) ===")
    m.beta_G, m.lam_G, m.thr_qG = old
    print("  -- seen --")
    report["C_old_seen"] = run(eng, "Cold", seen, 28, 8)
    print("  -- unseen --")
    report["C_old_unseen"] = run(eng, "Cold", unseen, 28, 8)

    print("\n=== C avec reglages CALIBRES pour le 1.7B ===")
    m.beta_G, m.lam_G, m.thr_qG = new
    print("  -- seen --")
    report["C_new_seen"] = run(eng, "Cnew", seen, 28, 8)
    print("  -- unseen --")
    report["C_new_unseen"] = run(eng, "Cnew", unseen, 28, 8)

    out = os.path.join(HERE, "results", "calib_17b.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


def engine_decode_words(eng, text, n_tokens):
    """Words covered by the first n_tokens of the document."""
    ids = eng.tok.encode(text)[:n_tokens]
    return eng.tok.decode(ids).split()


def find_words(words, first):
    n = len(first)
    for i in range(len(words) - n):
        if words[i:i + n] == first:
            return i
    return None


if __name__ == "__main__":
    main()
