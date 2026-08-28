"""Etape 3 : replication GPT-2 du protocole semantique complet.

1. balayage des couches (regle declaree : meilleure sur A, validee B) ;
2. recette v3 sur la couche choisie : ancres g>=2.5, integrite de mot,
   SimHash mu fige, pooling de requete, echo-suppression ;
3. rangs A/B, comportemental B (grille dev, test stables), localite.

    python probe_gpt2_replication.py
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
from probe_semantic_anchor import (anchors_from, entity_ends,  # noqa: E402
                                   fixed_mu)
from probe_semantic_layers import forwards_all_layers, wh      # noqa: E402
from probe_semantic_l1 import gates_from                       # noqa: E402

G_MIN = 0.5
GRID_BETA = (5.0, 10.0, 20.0)
GRID_LAM = (0.5, 0.85)
N_GEN = 8
STATE_TMP = os.path.join(HERE, ".gpt2rep_tmp_state")


def main():
    import shutil
    import torch
    shutil.rmtree(STATE_TMP, ignore_errors=True)
    s = Sillage(model="gpt2", state=STATE_TMP, quiet=True)
    tok, model = s.load_model()

    facts = list(zip(ENTS[:30], VALS[:30]))
    changed_e = {e for e, _ in
                 [(e, ALT[i]) for i, (e, _v) in enumerate(facts[:10])]}
    dev = [(e, v) for e, v in facts if e in changed_e]
    test = [(e, v) for e, v in facts if e not in changed_e]

    doc = build_doc(facts, seed=0)
    ids = tok.encode(doc)
    print(f"dossier v1 : {len(ids)} tokens ; forwards toutes couches...",
          flush=True)
    Hall = forwards_all_layers(s, ids)
    L = len(Hall)
    G = gates_from(s, ids)
    print(f"{L} couches, dim {Hall[-1].shape[-1]}", flush=True)

    ends_by = {e: entity_ends(tok, ids, [e]) for e, _ in test}

    # ---- phase 1 : balayage --------------------------------------------
    Pcache = {}
    for e, _v in test:
        for tag, tmpl in (("A", A_PREFIX), ("B", B_PREFIX)):
            pids = tok.encode(tmpl.format(e=e))
            Hp = forwards_all_layers(s, pids)
            pe = entity_ends(tok, pids, [e])[-1]
            Pcache[(e, tag)] = [Hp[li][pe] for li in range(L)]
    sweep = []
    for li in range(L):
        Z = wh(Hall[li])
        mean_ = (Hall[li] / (np.linalg.norm(Hall[li], axis=1,
                                            keepdims=True) + 1e-8)
                 ).mean(axis=0)
        deltas = {}
        for tag in ("A", "B"):
            same, null = [], []
            for e, _v in test:
                h = Pcache[(e, tag)][li]
                z = h / (np.linalg.norm(h) + 1e-8) - mean_
                z = z / (np.linalg.norm(z) + 1e-8)
                same.append(float(np.max(Z[ends_by[e]] @ z)))
                others = [i for e2, _ in test if e2 != e
                          for i in ends_by[e2]]
                null.append(float(np.max(Z[others] @ z)))
            deltas[tag] = float(np.median(same) - np.median(null))
        sweep.append({"layer": li, "dA": deltas["A"],
                      "dB": deltas["B"]})
        print(f"  couche {li:2d} : delta A {deltas['A']:+.3f} | "
              f"B {deltas['B']:+.3f}", flush=True)
    best = max(sweep[1:], key=lambda r: r["dA"])
    LAYER = best["layer"]
    print(f"couche choisie (sur A) : {LAYER} (dA {best['dA']:+.3f}, "
          f"validation B {best['dB']:+.3f})", flush=True)

    # ---- phase 2 : recette v3 ------------------------------------------
    H = Hall[LAYER]
    mu = fixed_mu(H)
    pts = [t for t in range(len(ids)) if G[t] >= 2.5]
    anch = anchors_from(pts, len(ids))

    def no_space(tid):
        d = tok.decode([int(tid)])
        return len(d) > 0 and not d[0].isspace()

    m = SillageMemory(None, "gpt2", semantic=True, fastweights=False)
    m.set_vocab(50257)
    m.mu = mu.copy()
    m.mu_n = 10 ** 9
    kept = np.zeros(len(ids), bool)
    n_write = 0
    for t in range(len(ids) - 1):
        a = int(anch[t])
        if a < 0:
            continue
        keep = (float(G[t + 1]) >= G_MIN
                or (kept[t] and no_space(ids[t + 1])))
        if not keep:
            continue
        kept[t + 1] = True
        qS = m.sem_key(H[a])
        uS, _ = m.scores(m.MS, qS)
        m.amp_write(m.MS, qS, uS, int(ids[t + 1]),
                    float(max(G[t + 1], 0.25)))
        n_write += 1
    print(f"tier v3 gpt2 ({n_write} ecritures, couche {LAYER})",
          flush=True)

    def pooled_for(pids):
        Hp = forwards_all_layers(s, pids)[LAYER]
        bestv = None
        for p in range(len(Hp)):
            qS = m.sem_key(Hp[p])
            _, sS = m.scores(m.MS, qS)
            bestv = sS.copy() if bestv is None else np.maximum(bestv, sS)
        return bestv

    ranks = {"A": [], "B": []}
    for e, v in test:
        vid = tok.encode(" " + v)[0]
        for tag, tmpl in (("A", A_PREFIX), ("B", B_PREFIX)):
            sE = pooled_for(tok.encode(tmpl.format(e=e)))
            ranks[tag].append(int((sE > sE[vid]).sum()) + 1)
    r_agg = {t: {"top10": sum(r <= 10 for r in ranks[t]) / len(ranks[t]),
                 "median": float(np.median(ranks[t]))}
             for t in ("A", "B")}
    print(f"rangs : A top10 {r_agg['A']['top10']:.0%} (med "
          f"{r_agg['A']['median']:.0f}) | B top10 "
          f"{r_agg['B']['top10']:.0%} (med {r_agg['B']['median']:.0f})",
          flush=True)

    # ---- phase 3 : comportemental --------------------------------------
    wit = [x.strip() for x in WITNESS.replace("\n", " ").split(".")
           if len(x.strip()) > 20][:20]
    wit_prompts = [w[: max(20, len(w) // 2)] for w in wit]
    null_max = [float(pooled_for(tok.encode(w)).max())
                for w in wit_prompts]
    thr = float(np.quantile(null_max, 0.95))

    def generate(prompt, beta, lam, use_tier=True):
        pids = list(tok.encode(prompt))
        sE = pooled_for(pids) if use_tier else None
        fire = sE is not None and float(sE.max()) >= thr
        if fire:
            sE = sE.copy()
            mask = np.zeros(len(sE), bool)
            mask[list(set(int(t) for t in pids))] = True
            sE[mask] = -1e9
            p_sem = np.exp(beta * (sE - sE.max()))
            p_sem = p_sem / p_sem.sum()
        gen_ids = list(pids)
        for _ in range(N_GEN):
            with torch.no_grad():
                out = model(torch.tensor([gen_ids], device=s.device))
            lg = out.logits[0, -1].float().cpu().numpy()
            p = np.exp(lg - lg.max())
            p = p / p.sum()
            if fire:
                p = (1 - lam) * p + lam * p_sem
            gen_ids.append(int(np.argmax(p)))
        return tok.decode(gen_ids[len(pids):])

    def recall(pairs, tag, beta, lam, use_tier=True):
        tmpl = A_PREFIX if tag == "A" else B_PREFIX
        return sum(v.split()[0] in generate(tmpl.format(e=e), beta,
                                            lam, use_tier)
                   for e, v in pairs) / len(pairs)

    bestg = None
    for beta in GRID_BETA:
        for lam in GRID_LAM:
            r = recall(dev, "B", beta, lam)
            if bestg is None or r > bestg[2]:
                bestg = (beta, lam, r)
    beta, lam, rdev = bestg
    B_test = recall(test, "B", beta, lam)
    base_B = recall(test, "B", beta, lam, use_tier=False)
    loc = sum(generate(w, beta, lam, False) != generate(w, beta, lam,
                                                        True)
              for w in wit_prompts[:10])
    print(f"\ncomportemental : dev B {rdev:.0%} (beta {beta}, lam "
          f"{lam}) | TEST B {B_test:.0%} (base {base_B:.0%}) | "
          f"localite {loc}/10", flush=True)

    json.dump({"layer": LAYER, "sweep": sweep, "ranks": r_agg,
               "thr": thr, "beta": beta, "lam": lam, "dev_B": rdev,
               "test_B": B_test, "base_B": base_B,
               "locality_changed": loc / 10, "writes": n_write},
              open(os.path.join(HERE, "results",
                                "semantic_gpt2_replication.json"),
                   "w"), indent=1)
    print("saved -> results/semantic_gpt2_replication.json")
    shutil.rmtree(STATE_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
