"""Etape A/B : validation COMPORTEMENTALE du tier semantique v2.

Tier 2i (couche 1, ancres r1, SimHash sans ZCA, mu fige) sur le
dossier v1. A la generation : p' = (1-lam)*p_base +
lam*softmax(beta*sE_poole), declenche si max(sE) >= thr.
Anti-surapprentissage declare : (beta, lam) par grille sur les 10
faits DEV (entites "changed", valeurs v1) ; mesure finale sur les 20
STABLES ; thr = q95 des maxima pooles sur 20 prompts temoins.
Scorer = mot de tete de la valeur dans 8 tokens greedy.

    python probe_behavioral_v2.py
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
from sillage.core import SillageMemory                         # noqa: E402
from behavioral import (ALT, A_PREFIX, B_PREFIX, ENTS, VALS,   # noqa: E402
                        WITNESS, build_doc)
from probe_semantic_anchor import (anchors_from, fixed_mu)     # noqa: E402
from probe_semantic_layers import forwards_all_layers          # noqa: E402
from probe_semantic_l1 import gates_from                       # noqa: E402

LAYER = 1
STATE_TMP = os.path.join(HERE, ".bv2_tmp_state")
GRID_BETA = (5.0, 10.0, 20.0, 40.0)
GRID_LAM = (0.2, 0.5, 0.85)
N_GEN = 8


def pooled_scores(m, Hp):
    best = None
    for p in range(len(Hp)):
        qS = m.sem_key(Hp[p])
        _, sS = m.scores(m.MS, qS)
        best = sS.copy() if best is None else np.maximum(best, sS)
    return best


def generate(s, m, tok, model, prompt, sE, beta, lam, thr):
    import torch
    ids = list(tok.encode(prompt))
    fire = sE is not None and float(sE.max()) >= thr
    if fire:
        p_sem = np.exp(beta * (sE - sE.max()))
        p_sem = p_sem / p_sem.sum()
    for _ in range(N_GEN):
        with torch.no_grad():
            out = model(torch.tensor([ids], device=s.device))
        lg = out.logits[0, -1].float().cpu().numpy()
        mx = lg.max()
        p = np.exp(lg - mx)
        p = p / p.sum()
        if fire:
            p = (1.0 - lam) * p + lam * p_sem
        ids.append(int(np.argmax(p)))
    return tok.decode(ids[len(tok.encode(prompt)):])


def main():
    import shutil
    shutil.rmtree(STATE_TMP, ignore_errors=True)
    s = Sillage(model="qwen", state=STATE_TMP, quiet=True)
    tok, model = s.load_model()

    facts = list(zip(ENTS[:30], VALS[:30]))
    changed_e = {e for e, _ in
                 [(e, ALT[i]) for i, (e, _v) in enumerate(facts[:10])]}
    dev = [(e, v) for e, v in facts if e in changed_e]      # valeurs v1
    test = [(e, v) for e, v in facts if e not in changed_e]

    doc = build_doc(facts, seed=0)
    ids = tok.encode(doc)
    print(f"dossier v1 : {len(ids)} tokens ; forwards...", flush=True)
    H = forwards_all_layers(s, ids)[LAYER]
    G = gates_from(s, ids)
    mu = fixed_mu(H)

    pts = [t for t in range(len(ids)) if G[t] >= 2.5]
    anch = anchors_from(pts, len(ids))

    def build_tier(g_min):
        m = SillageMemory(None, "qwen", semantic=True,
                          fastweights=False)
        m.set_vocab(151936)
        m.mu = mu.copy()
        m.mu_n = 10 ** 9
        kept = 0
        for t in range(len(ids) - 1):
            a = int(anch[t])
            if a < 0 or float(G[t + 1]) < g_min:
                continue
            qS = m.sem_key(H[a])
            uS, _ = m.scores(m.MS, qS)
            m.amp_write(m.MS, qS, uS, int(ids[t + 1]),
                        float(G[t + 1]))
            kept += 1
        return m, kept

    m, kept = build_tier(0.0)
    print(f"tier v2 construit ({kept} ecritures)", flush=True)

    def pooled_for(prompt):
        Hp = forwards_all_layers(s, tok.encode(prompt))[LAYER]
        return pooled_scores(m, Hp)

    # thr = q95 of pooled maxima over witness prompts (null)
    wit_sents = [x.strip() for x in WITNESS.replace("\n", " ").split(".")
                 if len(x.strip()) > 20][:20]
    wit_prompts = [w[: max(20, len(w) // 2)] for w in wit_sents]
    null_max = [float(pooled_for(w).max()) for w in wit_prompts]
    thr = float(np.quantile(null_max, 0.95))
    print(f"thr (q95 des maxima temoins) = {thr:.4f} "
          f"(null median {np.median(null_max):.4f})", flush=True)

    # precompute pooled scores per probe prompt
    sE_cache = {}
    for e, _v in facts:
        for tag, tmpl in (("A", A_PREFIX), ("B", B_PREFIX)):
            sE_cache[(e, tag)] = pooled_for(tmpl.format(e=e))

    def recall(pairs, tag, beta, lam):
        tmpl = A_PREFIX if tag == "A" else B_PREFIX
        hits = 0
        for e, v in pairs:
            out = generate(s, m, tok, model, tmpl.format(e=e),
                           sE_cache[(e, tag)], beta, lam, thr)
            hits += v.split()[0] in out
        return hits / len(pairs)

    # --- dev grid: g_min x (beta, lam) on B -----------------------------
    best = None
    for g_min in (0.0, 0.5, 1.0, 2.0):
        m, kept = build_tier(g_min)
        sE_cache.clear()
        for e, _v in facts:
            for tag, tmpl in (("A", A_PREFIX), ("B", B_PREFIX)):
                sE_cache[(e, tag)] = pooled_for(tmpl.format(e=e))
        for beta in GRID_BETA:
            for lam in GRID_LAM:
                r = recall(dev, "B", beta, lam)
                if best is None or r > best[3]:
                    best = (g_min, beta, lam, r, kept)
        print(f"  g_min {g_min:.1f} ({kept:4d} ecritures) : meilleur "
              f"dev B jusqu'ici {best[3]:.0%} "
              f"(g_min {best[0]}, beta {best[1]}, lam {best[2]})",
              flush=True)
    g_min, beta, lam, rdev, kept = best
    m, kept = build_tier(g_min)
    sE_cache.clear()
    for e, _v in facts:
        for tag, tmpl in (("A", A_PREFIX), ("B", B_PREFIX)):
            sE_cache[(e, tag)] = pooled_for(tmpl.format(e=e))
    print(f"grille : g_min {g_min}, beta {beta}, lam {lam} "
          f"(dev B {rdev:.0%}, {kept} ecritures)", flush=True)

    # --- test, never seen by the grid -----------------------------------
    B_test = recall(test, "B", beta, lam)
    A_test = recall(test, "A", beta, lam)
    base_B = recall(test, "B", beta, 0.0)      # mixage coupe
    print(f"\nTEST (20 stables) : paraphrase B {B_test:.0%} "
          f"(base sans tier {base_B:.0%}) | canonique A {A_test:.0%}",
          flush=True)

    # --- locality: witness prompts, greedy continuation changed? --------
    changed_cnt = 0
    for w in wit_prompts:
        sE = pooled_for(w)
        g0 = generate(s, m, tok, model, w, None, beta, lam, thr)
        g1 = generate(s, m, tok, model, w, sE, beta, lam, thr)
        changed_cnt += int(g0 != g1)
    loc = changed_cnt / len(wit_prompts)
    print(f"localite : {changed_cnt}/{len(wit_prompts)} prompts temoins "
          f"changent ({loc:.0%})", flush=True)

    json.dump({"thr": thr, "g_min": g_min, "beta": beta, "lam": lam,
               "dev_B": rdev, "test_B": B_test, "test_A": A_test,
               "base_B": base_B, "locality_changed": loc},
              open(os.path.join(HERE, "results",
                                "semantic_behavioral_v2.json"), "w"),
              indent=1)
    print("saved -> results/semantic_behavioral_v2.json")
    shutil.rmtree(STATE_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
