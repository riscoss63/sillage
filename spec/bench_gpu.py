"""GPU benches for the Sillage speculative drafter (Kaggle-ready, self-contained).

Configs:
  micro   forward-latency microbenchmark: cost of a 1/2/4/8/16-token forward
          with a warm KV cache -- on GPU they cost nearly the same, and that
          ratio is what converts acceptance into wall-clock speedup.
  gpt2    GPT-2 124M on its own state (the four papers), self-speculation.
  A       Qwen3-0.6B on its own state (two manuscripts), self-speculation.
  B       bigger target, VANILLA (pure-acceleration control -- expected weak).
  C       bigger target + the 0.6B's memory (the product configuration);
          --calibrate refits (beta, lambda, threshold) for THIS target,
          read-only, on the first 20% of the stream (the tool's protocol).

Everything is read-only: the memory state is never written, never saved.
Outputs are checked identical between plain and speculative decoding within
each config (the guarantee that matters); fp16 may differ marginally from
fp32-CPU runs, which is expected and irrelevant to that guarantee.

Examples:
  python bench_gpu.py --config micro --device cuda --dtype float16
  python bench_gpu.py --config A     --device cuda --dtype float16
  python bench_gpu.py --config C     --device cuda --dtype float16 \
         --target Qwen/Qwen3-1.7B --beta 40 --lam 0.85 --thrq 0.5
  python bench_gpu.py --config C     --device cuda --dtype float16 \
         --target Qwen/Qwen3-4B --calibrate
"""

import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break

from sillage_drafter import (PromptLookupDrafter, SillageDrafter,  # noqa
                             SpeculativeSillage, cache_len, crop_cache,
                             timed)
from torch_readout import TorchEngine                              # noqa
from sillage.core import BETAS, fit_readout, lse_grid              # noqa


def make_engine(state_dir, a, **kw):
    """numpy engine on CPU, torch-resident readout on CUDA (or --engine)."""
    use_torch = (a.engine == "torch" or
                 (a.engine == "auto" and a.device.startswith("cuda")))
    cls = TorchEngine if use_torch else SpeculativeSillage
    print(f"  moteur: {'torch-GPU' if use_torch else 'numpy'}")
    return cls(state_dir, device=a.device, dtype=a.dtype, **kw)

# Paths resolve for BOTH layouts: the repository (this file in spec/, states
# at the repo root) and the flat Kaggle kit (states/ and docs/ next to it).

def _first_existing(*cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


ROOT = os.path.dirname(HERE)
OUTSIDE = os.path.dirname(ROOT)
STATES = {
    "qwen": _first_existing(os.path.join(HERE, "states", "memory_state"),
                            os.path.join(ROOT, "memory_state")),
    "gpt2": _first_existing(
        os.path.join(HERE, "states", "papers_state_memory"),
        os.path.join(ROOT, "papers_state", "memory")),
}
DOCS_QWEN = [p for p in (
    _first_existing(
        os.path.join(HERE, "docs", "manuscrit_v1.txt"),
        os.path.join(OUTSIDE, "preprint_bhd_active_inference.txt")),
    _first_existing(
        os.path.join(HERE, "docs", "manuscrit_v2.md"),
        os.path.join(OUTSIDE, "PREPRINT_V2_1_BHD_ACTIVE_INFERENCE.md")),
) if p]
CORPUS_GPT2 = _first_existing(
    os.path.join(HERE, "docs", "corpus_papers.txt"),
    os.path.join(ROOT, "papers_state", "corpus.txt"))
MANUSCRIPT_NOTE = ("the papers' Manuscripts stream is not redistributed -- "
                   "pass --docs <doc1> <doc2> to run the qwen configs on "
                   "your own documents (the state was written from ours, so "
                   "acceptance on yours will differ)")
STATE_HINTS = {
    "qwen": ("build one by reading two documents:\n"
             "  sillage read doc1 doc2 --model qwen --state memory_state"),
    "gpt2": ("build it from the public papers:\n"
             "  sillage papers --with-memory   (writes papers_state/)"),
}


def need_state(key):
    """States are never shipped: a cold store would reveal what it read."""
    p = STATES.get(key)
    if p is None:
        raise SystemExit(f"state '{key}' not found -- {STATE_HINTS[key]}")
    return p

UNSEEN = [
    "The city council met on Thursday to discuss the new bicycle lanes",
    "In the morning she poured coffee, opened the window, and watched the",
    "The recipe calls for two cups of flour, a pinch of salt, and",
    "Financial markets rallied yesterday after the central bank announced",
    "The hikers followed the narrow trail along the ridge until the",
]


# ---------------------------------------------------------------- helpers ---

def sample_prompts(text, k, words_per=24, seed=1, truth_words=60):
    words = [w for w in text.split() if w]
    rng = random.Random(seed)
    lo = int(0.05 * len(words))
    hi = int(0.97 * len(words)) - words_per - truth_words
    out = []
    while len(out) < k:
        i = rng.randrange(lo, hi)
        seg = " ".join(words[i:i + words_per])
        if re.search(r"[^\x20-\x7e]", seg):
            continue
        truth = " ".join(words[i + words_per:i + words_per + truth_words])
        out.append((seg, truth))
    return out


def prefix_agreement(generated, truth):
    n = 0
    for a, b in zip(generated.split(), truth.split()):
        if a != b:
            break
        n += 1
    return n


def run(engine, prompts, n, gamma, with_pld=False):
    res = {}
    ref_out = None
    methods = ["plain", "spec:sillage"] + (["spec:pld"] if with_pld else [])
    for meth in methods:
        total = {"seconds": 0.0, "tokens": 0, "forwards": 0, "drafted": 0,
                 "accepted": 0, "rounds": 0, "drafted_cold": 0,
                 "acc_cold": 0, "drafted_mg": 0, "acc_mg": 0}
        outs = []
        for p, _t in prompts:
            if meth == "plain":
                o, st = timed(engine.generate_plain, p, n=n)
            elif meth == "spec:sillage":
                o, st = timed(engine.generate_spec, p, SillageDrafter,
                              n=n, gamma=gamma)
            else:
                o, st = timed(engine.generate_spec, p, PromptLookupDrafter,
                              n=n, gamma=gamma)
            outs.append(o)
            for key in total:
                total[key] += st.get(key, 0)
        total["tok_per_s"] = total["tokens"] / max(1e-9, total["seconds"])
        if meth == "plain":
            ref_out = outs
            agr = [prefix_agreement(engine.tok.decode(o), t)
                   for o, (_, t) in zip(outs, prompts) if t]
            if agr:
                total["verbatim_prefix_words"] = agr
                print(f"    (rappel verbatim moyen {sum(agr)/len(agr):.1f} "
                      f"mots ; {sorted(agr, reverse=True)})")
            print(f"    plain        : {total['tok_per_s']:7.2f} tok/s")
        else:
            total["identical_outputs"] = sum(
                a == b for a, b in zip(outs, ref_out))
            total["speedup"] = total["tok_per_s"] / res["plain"]["tok_per_s"]
            acc = total["accepted"] / max(1, total["drafted"])
            print(f"    {meth:13s}: {total['tok_per_s']:7.2f} tok/s   "
                  f"x{total['speedup']:.2f}   acc {acc:.0%} "
                  f"({total['accepted']}/{total['drafted']})   "
                  f"identical {total['identical_outputs']}/{len(prompts)}")
            if total["drafted_cold"] or total["drafted_mg"]:
                print(f"                   par source: cold "
                      f"{total['acc_cold']}/{total['drafted_cold']}   "
                      f"M_G {total['acc_mg']}/{total['drafted_mg']}")
        res[meth] = total
    return res


# ------------------------------------------------------------- microbench ---

def microbench(engine, warm_path, chunks=(1, 2, 4, 8, 16), reps=30,
               warm=256):
    """Latency of a k-token forward with a warm cache, per k."""
    torch = engine.torch
    text = open(warm_path, encoding="utf-8", errors="replace").read()
    ids = engine.tok.encode(text)[:warm + max(chunks)]
    sync = (torch.cuda.synchronize if engine.device.startswith("cuda")
            else (lambda: None))
    with torch.no_grad():
        out = engine.model(torch.tensor([ids[:warm]], device=engine.device),
                           use_cache=True)
        past = out.past_key_values
        base = cache_len(past)
        rows = {}
        for k in chunks:
            chunk = ids[warm:warm + k]
            t_best = float("inf")
            for _ in range(reps):
                sync()
                t0 = time.perf_counter()
                out = engine.model(
                    torch.tensor([chunk], device=engine.device),
                    past_key_values=past, use_cache=True)
                sync()
                t_best = min(t_best, time.perf_counter() - t0)
                past = crop_cache(out.past_key_values, base)
            rows[k] = t_best * 1000
    print("  latence d'un forward (cache chaud), meilleur de "
          f"{reps} essais :")
    for k, ms in rows.items():
        print(f"    {k:2d} tokens : {ms:7.2f} ms   "
              f"(ratio vs 1 token : {ms/rows[1]:.2f})")
    print(f"  facteur de conversion (16 tokens / 1 token) : "
          f"{rows[16]/rows[1]:.2f} -- plus il est proche de 1, plus "
          f"l'acceptation se convertit en speedup.")
    return rows


# ------------------------------------------------------------ calibration ---

def calibrate(engine, doc_paths, frac=0.20, stride=3, chunk=512):
    """Read-only refit of (beta, lambda, threshold) for THIS target."""
    torch = engine.torch
    m = engine.mem
    n_dev = int(frac * m.tokens)
    text = open(doc_paths[0], encoding="utf-8", errors="replace").read()
    ids = engine.tok.encode(text)[:n_dev + 1]
    m.new_stream()
    p_l, gt_l, gm_l, gl_l = [], [], [], []
    past, pos = None, 0
    with torch.no_grad():
        for a in range(0, len(ids) - 1, chunk):
            part = ids[a:a + chunk]
            out = engine.model(torch.tensor([part], device=engine.device),
                               past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0].float().cpu().numpy()
            for i, tok in enumerate(part):
                j = a + i
                if j + 1 >= len(ids):
                    break
                q = m.step_key(int(tok))
                if pos % stride == 0:
                    truth = int(ids[j + 1])
                    lb = logits[i]
                    mx = lb.max()
                    p_l.append(float(np.exp(lb[truth] - mx)
                                     / np.exp(lb - mx).sum()))
                    _, sG = m.scores(m.M, q)
                    gt_l.append(float(sG[truth]))
                    gm_l.append(float(sG.max()))
                    gl_l.append(lse_grid(sG))
                pos += 1
    p = np.array(p_l, np.float64)
    nll_base = float(-np.log(np.maximum(p, 1e-30)).mean())
    nll, beta, lam, q, _ = fit_readout(
        p, np.array(gt_l, np.float32), np.array(gm_l, np.float32),
        np.array(gl_l, np.float32))
    print(f"  calibration ({len(p)} obs, lecture seule) : cible seule "
          f"NLL {nll_base:.4f} -> calibree {nll:.4f}  "
          f"[beta {beta:g}, lam {lam:g}, thr q{q}]")
    return beta, lam, q


# ----------------------------------------------------------------- main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["micro", "gpt2", "A", "B", "C"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--target", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--prompts", type=int, default=10)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--thrq", type=float, default=None)
    ap.add_argument("--pld", action="store_true",
                    help="ajouter l'ablation prompt-lookup sur seen")
    ap.add_argument("--engine", default="auto",
                    choices=["auto", "numpy", "torch"])
    ap.add_argument("--docs", nargs=2, metavar=("DOC1", "DOC2"),
                    help="the two documents of the qwen state's stream "
                         "(defaults to the Manuscripts files when present)")
    a = ap.parse_args()

    docs = a.docs if a.docs else DOCS_QWEN
    if a.config in ("A", "B", "C") and len(docs) < 2:
        raise SystemExit(MANUSCRIPT_NOTE)
    warm_path = docs[0] if docs else CORPUS_GPT2

    cfg = a.config
    report = {"config": cfg, "device": a.device, "dtype": a.dtype,
              "n": a.n, "gamma": a.gamma}

    if cfg == "micro":
        eng = make_engine(need_state("qwen"), a, target_hub=a.target,
                          memory_in_target=False, fastweights=False)
        print(f"=== micro : {a.target} ({a.dtype}, {a.device}) ===")
        report["latency_ms"] = microbench(eng, warm_path)
    elif cfg == "gpt2":
        eng = make_engine(need_state("gpt2"), a)
        text = open(CORPUS_GPT2, encoding="utf-8", errors="replace").read()
        seen = sample_prompts(text, a.prompts, seed=0)
        print(f"=== gpt2 self ({a.dtype}, {a.device}) ===\n  -- seen --")
        report["seen"] = run(eng, seen, a.n, a.gamma, with_pld=a.pld)
        print("  -- unseen --")
        report["unseen"] = run(eng, [(p, None) for p in UNSEEN],
                               a.n, a.gamma)
    else:
        text = "\n\n".join(open(p, encoding="utf-8", errors="replace").read()
                           for p in docs)
        seen = sample_prompts(text, a.prompts, seed=1)
        if cfg == "A":
            eng = make_engine(need_state("qwen"), a)
            title = f"A. 0.6B self ({a.dtype})"
        elif cfg == "B":
            eng = make_engine(need_state("qwen"), a, target_hub=a.target,
                              memory_in_target=False, fastweights=False)
            title = f"B. {a.target} VANILLA"
        else:
            eng = make_engine(need_state("qwen"), a, target_hub=a.target,
                              memory_in_target=True, fastweights=False)
            title = f"C. {a.target} + memoire 0.6B"
            if a.calibrate:
                new = calibrate(eng, docs)
                (eng.mem.beta_G, eng.mem.lam_G, eng.mem.thr_qG) = new
                report["calibrated"] = list(new)
            elif a.beta is not None:
                eng.mem.beta_G = a.beta
                if a.lam is not None:
                    eng.mem.lam_G = a.lam
                if a.thrq is not None:
                    eng.mem.thr_qG = a.thrq
                report["settings"] = [eng.mem.beta_G, eng.mem.lam_G,
                                      eng.mem.thr_qG]
            if hasattr(eng, "refresh_settings"):
                eng.refresh_settings()
        print(f"=== {title} ===\n  -- seen --")
        report["seen"] = run(eng, seen, a.n, a.gamma, with_pld=a.pld)
        print("  -- unseen --")
        report["unseen"] = run(eng, [(p, None) for p in UNSEEN],
                               a.n, a.gamma)

    out = os.path.join(os.getcwd(), f"results_gpu_{cfg}.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
