"""Axe 4 : que vaut un etat SANS son cold store ni son index ?

Un etat qui se partage ne peut embarquer ni le cold store (une table
4-grammes -> successeurs en tokens clairs) ni l'index (les passages en
clair). Restent les matrices. Cette sonde mesure ce que l'on perd, sur
le meme etat, en coupant les tiers a la GENERATION -- rien n'est
reecrit, donc la comparaison est appariee.

    python probe_shareable_state.py [gpt2|qwen]
"""

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

from sillage import Sillage                                    # noqa: E402
from sillage.index import strip_latex                          # noqa: E402
from behavioral import (A_PREFIX, B_PREFIX, ENTS, VALS,        # noqa: E402
                        build_doc)

CFG = {"qwen": {"sem2": 1}, "gpt2": {"sem2": 5}}


def ppl_nowrite(s, text, use_cold=True):
    """Teacher-forced perplexity with the memory, writing nothing."""
    import torch
    tok, model = s.load_model()
    mem = s.mem
    ids = np.array(tok.encode(text), dtype=np.int64)
    n = min(len(ids) - 1, 900)
    mem.new_stream()
    thrG, thrS = mem.thresholds()
    need_h = mem.semantic or mem.fastweights
    nll_b = nll_m = 0.0
    with torch.no_grad():
        out = model(torch.tensor(ids[:n + 1], device=s.device
                                 ).unsqueeze(0),
                    output_hidden_states=need_h)
    logits = out.logits[0].float().cpu().numpy()
    mem.set_vocab(logits.shape[-1])
    hs = (out.hidden_states[-1][0].float().cpu().numpy()
          if need_h else None)
    for i in range(n):
        truth = int(ids[i + 1])
        lb = logits[i]
        mx = lb.max()
        lp = float(lb[truth] - (mx + np.log(np.exp(lb - mx).sum())))
        la, _ = mem.adapt(lb, hs[i] if need_h else None)
        p_ad = np.exp(la - la.max())
        p_ad /= p_ad.sum()
        qG = mem.step_key(int(ids[i]))
        _, sG = mem.scores(mem.M, qG)
        pc = mem.cold_lookup(truth) if use_cold else None
        p = mem.mix_true(float(p_ad[truth]), sG, truth, None, pc,
                         thrG, thrS)
        nll_b += -lp
        nll_m += -np.log(max(p, 1e-30))
    return float(np.exp(nll_b / n)), float(np.exp(nll_m / n))


def main():
    import shutil
    which = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    state = os.path.join(HERE, f".share_{which}")
    shutil.rmtree(state, ignore_errors=True)
    real = strip_latex(open(os.path.join(
        os.path.dirname(HERE), "papers", "behavior",
        "behavior.tex"), encoding="utf-8").read())[:12000]
    facts = list(zip(ENTS[:20], VALS[:20]))
    doc = build_doc(facts, seed=5, reps=4, block=60)
    f_real = os.path.join(HERE, "_share_real.txt")
    f_facts = os.path.join(HERE, "_share_facts.txt")
    open(f_real, "w", encoding="utf-8").write(real)
    open(f_facts, "w", encoding="utf-8").write(doc)

    s = Sillage(model=which, state=state, quiet=True,
                fastweights=False, **CFG[which])
    s.read(f_real, fast=True)
    s.read(f_facts, fast=True)
    test = facts[10:]
    print(f"{which}: etat construit ({s.mem.tokens} tokens, "
          f"{len(s.mem.cold)} grams, {len(s.index.passages)} passages)",
          flush=True)

    def recall(tmpl, use_cold, use_sem):
        keep_sem = s.mem.semantic
        s.mem.semantic = use_sem
        keep_cold = None
        if not use_cold:
            keep_cold, s.mem.cold = s.mem.cold, {}
        try:
            hits = sum(v.split()[0] in s.complete(tmpl.format(e=e), n=8)
                       for e, v in test)
        finally:
            s.mem.semantic = keep_sem
            if keep_cold is not None:
                s.mem.cold = keep_cold
        return hits

    known = doc[:4000]
    R = {"model": which, "configs": {}}
    for tag, use_cold, use_sem in (
            ("A complet", True, True),
            ("B sans cold", False, True),
            ("C sans semantique", True, False),
            ("D matrices seules", False, True)):
        keep = None
        if not use_cold:
            keep, s.mem.cold = s.mem.cold, {}
        base, mem_ppl = ppl_nowrite(s, known, use_cold=use_cold)
        if keep is not None:
            s.mem.cold = keep
        ca = recall(A_PREFIX, use_cold, use_sem)
        cb = recall(B_PREFIX, use_cold, use_sem)
        R["configs"][tag] = {"ppl_base": round(base, 2),
                             "ppl_mem": round(mem_ppl, 2),
                             "canonical": ca / len(test),
                             "paraphrase": cb / len(test)}
        print(f"  {tag:20s}: PPL {base:7.2f} -> {mem_ppl:6.2f} | "
              f"canonique {ca}/{len(test)} | paraphrase "
              f"{cb}/{len(test)}", flush=True)

    sizes = {f: os.path.getsize(os.path.join(state, f))
             for f in sorted(os.listdir(state))}
    keep_files = ("state.npz",)
    R["sizes_bytes"] = sizes
    R["shareable_bytes"] = sum(v for k, v in sizes.items()
                               if k in keep_files)
    R["full_bytes"] = sum(sizes.values())
    print(f"  taille : complet {R['full_bytes']/1e6:.1f} Mo | "
          f"partageable (matrices seules) "
          f"{R['shareable_bytes']/1e6:.1f} Mo", flush=True)
    out = os.path.join(HERE, "results", f"shareable_{which}.json")
    json.dump(R, open(out, "w"), indent=1)
    print(f"saved -> {out}")
    shutil.rmtree(state, ignore_errors=True)
    os.remove(f_real)
    os.remove(f_facts)


if __name__ == "__main__":
    main()
