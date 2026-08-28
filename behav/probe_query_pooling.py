"""Marche 2i : pooling de requete.

Ecriture : ancres r1 (dernier g>=2.5 -- 92% des valeurs sous leur
entite, mesure en 2h). Requete : cles de TOUTES les positions du
prompt, score final = max par token sur les positions -- aucun choix
d'ancre a la requete. SimHash sans ZCA (la recette minimale).

    python probe_query_pooling.py
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
                        build_doc)
from probe_semantic_anchor import (anchors_from, entity_ends,  # noqa: E402
                                   fixed_mu)
from probe_semantic_layers import forwards_all_layers          # noqa: E402
from probe_semantic_l1 import gates_from                       # noqa: E402

LAYER = 1
STATE_TMP = os.path.join(HERE, ".pool_tmp_state")


def main():
    import shutil
    shutil.rmtree(STATE_TMP, ignore_errors=True)
    s = Sillage(model="qwen", state=STATE_TMP, quiet=True)
    tok, _ = s.load_model()

    facts = list(zip(ENTS[:30], VALS[:30]))
    changed = {e for e, _ in
               [(e, ALT[i]) for i, (e, _v) in enumerate(facts[:10])]}
    stable = [(e, v) for e, v in facts if e not in changed]

    doc = build_doc(facts, seed=0)
    ids = tok.encode(doc)
    print(f"dossier v1 : {len(ids)} tokens ; forwards...", flush=True)
    H = forwards_all_layers(s, ids)[LAYER]
    G = gates_from(s, ids)
    mu = fixed_mu(H)

    # write side: r1 anchors, minimal recipe (SimHash, mu frozen)
    pts = [t for t in range(len(ids)) if G[t] >= 2.5]
    anch = anchors_from(pts, len(ids))
    m = SillageMemory(None, "qwen", semantic=True, fastweights=False)
    m.set_vocab(151936)
    m.mu = mu.copy()
    m.mu_n = 10 ** 9
    for t in range(len(ids) - 1):
        a = int(anch[t])
        if a < 0:
            continue
        qS = m.sem_key(H[a])
        uS, _ = m.scores(m.MS, qS)
        m.amp_write(m.MS, qS, uS, int(ids[t + 1]), float(G[t + 1]))
    print("tier construit (r1, SimHash, sans ZCA)", flush=True)

    rows = []
    for e, v in stable:
        vid = tok.encode(" " + v)[0]
        rr = {}
        for tag, tmpl in (("A", A_PREFIX), ("B", B_PREFIX)):
            pids = tok.encode(tmpl.format(e=e))
            Hp = forwards_all_layers(s, pids)[LAYER]
            best = None
            best_pos = -1
            for p in range(len(pids)):
                qS = m.sem_key(Hp[p])
                _, sS = m.scores(m.MS, qS)
                if best is None:
                    best = sS.copy()
                    pos_of_max = np.full(len(sS), p)
                else:
                    upd = sS > best
                    best[upd] = sS[upd]
                    pos_of_max[upd] = p
            rank = int((best > best[vid]).sum()) + 1
            # was the winning position an entity token of the prompt?
            ent_span = set()
            pat = tok.encode(" " + e)
            for i in range(len(pids) - len(pat) + 1):
                if list(pids[i:i + len(pat)]) == pat:
                    ent_span.update(range(i, i + len(pat)))
            on_ent = int(pos_of_max[vid]) in ent_span
            rr[tag] = {"rank": rank, "max_on_entity": bool(on_ent),
                       "argpos": int(pos_of_max[vid])}
        rows.append({"e": e, "A": rr["A"], "B": rr["B"]})
        print(f"  {e:12s} A rang {rr['A']['rank']:>6d} "
              f"(ent {int(rr['A']['max_on_entity'])}) | B rang "
              f"{rr['B']['rank']:>6d} (ent "
              f"{int(rr['B']['max_on_entity'])})", flush=True)

    n = len(rows)
    agg = {"A_top10": sum(r["A"]["rank"] <= 10 for r in rows) / n,
           "B_top10": sum(r["B"]["rank"] <= 10 for r in rows) / n,
           "A_top1": sum(r["A"]["rank"] == 1 for r in rows) / n,
           "B_top1": sum(r["B"]["rank"] == 1 for r in rows) / n,
           "A_max_on_entity": sum(r["A"]["max_on_entity"]
                                  for r in rows) / n,
           "B_max_on_entity": sum(r["B"]["max_on_entity"]
                                  for r in rows) / n,
           "median_A": float(np.median([r["A"]["rank"]
                                        for r in rows])),
           "median_B": float(np.median([r["B"]["rank"]
                                        for r in rows]))}
    print(f"\n== pooling (r1 + SimHash sans ZCA) == A top10 "
          f"{agg['A_top10']:.0%} (med {agg['median_A']:.0f}, max sur "
          f"entite {agg['A_max_on_entity']:.0%}) | B top10 "
          f"{agg['B_top10']:.0%} (med {agg['median_B']:.0f}, max sur "
          f"entite {agg['B_max_on_entity']:.0%})", flush=True)

    json.dump({"agg": agg, "rows": rows},
              open(os.path.join(HERE, "results",
                                "semantic_pooling_qwen.json"), "w"),
              indent=1)
    print("saved -> results/semantic_pooling_qwen.json")
    shutil.rmtree(STATE_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
