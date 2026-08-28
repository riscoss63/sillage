"""Etape 2 : echo-suppression + integrite de mot.

Config 2i + g_min 0.5, plus :
  R1 ECHO : a la requete, p_sem[token present dans le prompt] = 0 --
  ce qui est dans la fenetre est gratuit, le rappel ne paie que
  l'absent.
  R2 INTEGRITE DE MOT : a l'ecriture, garder t+1 si g >= 0.5 OU si
  t+1 est une piece sans-espace dont la tete (t) a ete gardee.
Meme protocole dev/test/null que l'etape A.

    python probe_behavioral_v3.py
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
G_MIN = 0.5
GRID_BETA = (5.0, 10.0, 20.0)
GRID_LAM = (0.5, 0.85)
N_GEN = 8
STATE_TMP = os.path.join(HERE, ".bv3_tmp_state")


def main():
    import shutil
    import torch
    shutil.rmtree(STATE_TMP, ignore_errors=True)
    s = Sillage(model="qwen", state=STATE_TMP, quiet=True)
    tok, model = s.load_model()

    facts = list(zip(ENTS[:30], VALS[:30]))
    changed_e = {e for e, _ in
                 [(e, ALT[i]) for i, (e, _v) in enumerate(facts[:10])]}
    dev = [(e, v) for e, v in facts if e in changed_e]
    test = [(e, v) for e, v in facts if e not in changed_e]

    doc = build_doc(facts, seed=0)
    ids = tok.encode(doc)
    print(f"dossier v1 : {len(ids)} tokens ; forwards...", flush=True)
    H = forwards_all_layers(s, ids)[LAYER]
    G = gates_from(s, ids)
    mu = fixed_mu(H)
    pts = [t for t in range(len(ids)) if G[t] >= 2.5]
    anch = anchors_from(pts, len(ids))

    def no_space(tid):
        d = tok.decode([int(tid)])
        return len(d) > 0 and not d[0].isspace()

    # R2: word integrity at write time
    m = SillageMemory(None, "qwen", semantic=True, fastweights=False)
    m.set_vocab(151936)
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
    print(f"tier v3 construit ({n_write} ecritures, integrite de mot)",
          flush=True)

    def pooled_for(prompt_ids):
        Hp = forwards_all_layers(s, prompt_ids)[LAYER]
        best = None
        for p in range(len(Hp)):
            qS = m.sem_key(Hp[p])
            _, sS = m.scores(m.MS, qS)
            best = sS.copy() if best is None else np.maximum(best, sS)
        return best

    wit = [x.strip() for x in WITNESS.replace("\n", " ").split(".")
           if len(x.strip()) > 20][:20]
    wit_prompts = [w[: max(20, len(w) // 2)] for w in wit]
    null_max = [float(pooled_for(tok.encode(w)).max())
                for w in wit_prompts]
    thr = float(np.quantile(null_max, 0.95))
    print(f"thr = {thr:.4f}", flush=True)

    def generate(prompt, beta, lam, use_tier=True):
        pids = list(tok.encode(prompt))
        sE = pooled_for(pids) if use_tier else None
        fire = sE is not None and float(sE.max()) >= thr
        if fire:
            sE = sE.copy()
            # R1: echo suppression -- kill tokens already in the window
            p_sem_mask = np.zeros(len(sE), bool)
            p_sem_mask[list(set(int(t) for t in pids))] = True
            sE[p_sem_mask] = -1e9
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
        hits = 0
        for e, v in pairs:
            out = generate(tmpl.format(e=e), beta, lam, use_tier)
            hits += v.split()[0] in out
        return hits / len(pairs)

    best = None
    for beta in GRID_BETA:
        for lam in GRID_LAM:
            r = recall(dev, "B", beta, lam)
            print(f"  dev B: beta {beta:5.1f} lam {lam:.2f} -> {r:.0%}",
                  flush=True)
            if best is None or r > best[2]:
                best = (beta, lam, r)
    beta, lam, rdev = best
    print(f"grille : beta {beta}, lam {lam} (dev B {rdev:.0%})",
          flush=True)

    B_test = recall(test, "B", beta, lam)
    A_test = recall(test, "A", beta, lam)
    base_B = recall(test, "B", beta, lam, use_tier=False)
    print(f"\nTEST (20 stables) : paraphrase B {B_test:.0%} "
          f"(base {base_B:.0%}) | canonique A {A_test:.0%}", flush=True)

    changed_cnt = 0
    for w in wit_prompts[:10]:
        g0 = generate(w, beta, lam, use_tier=False)
        g1 = generate(w, beta, lam, use_tier=True)
        changed_cnt += int(g0 != g1)
    print(f"localite : {changed_cnt}/10 temoins changent", flush=True)

    json.dump({"thr": thr, "beta": beta, "lam": lam, "dev_B": rdev,
               "test_B": B_test, "test_A": A_test, "base_B": base_B,
               "locality_changed": changed_cnt / 10,
               "writes": n_write},
              open(os.path.join(HERE, "results",
                                "semantic_behavioral_v3.json"), "w"),
              indent=1)
    print("saved -> results/semantic_behavioral_v3.json")
    shutil.rmtree(STATE_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
