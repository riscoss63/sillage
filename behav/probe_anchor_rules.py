"""Marche 2h : la regle d'ancrage automatique.

Instrumente le CHOIX d'ancre (token decode, doc et prompts, vs oracle)
puis teste trois regles : r1 = dernier g>=2.5 ; r2 = max-g d'une
fenetre glissante de 16 ; r3 = dernier g>=4.0. Metrique intermediaire :
anchor-accuracy. Recuperation finale (SimHash sans ZCA, la recette
minimale) avec chaque regle.

    python probe_anchor_rules.py
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
STATE_TMP = os.path.join(HERE, ".rules_tmp_state")


def rule_points(name, G, n):
    if name == "r1_g2.5":
        return [t for t in range(n) if G[t] >= 2.5]
    if name == "r3_g4.0":
        return [t for t in range(n) if G[t] >= 4.0]
    if name == "r2_maxg16":
        # a position is an anchor iff it is the argmax of g over the
        # window of the 16 positions ending at it
        pts = []
        for t in range(n):
            lo = max(0, t - 15)
            if G[t] > 0 and t == lo + int(np.argmax(G[lo:t + 1])):
                pts.append(t)
        return pts
    raise ValueError(name)


def query_anchor(name, G, n):
    pts = rule_points(name, G, n)
    return pts[-1] if pts else n - 1


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

    ends = entity_ends(tok, ids, [e for e, _ in facts])
    end_set = set(ends)
    # doc positions of each fact's value first token, and its entity end
    val_pos = {}
    for e, v in facts:
        vt = tok.encode(" " + v)[0]
        for i in ends:
            # entity end i: the value token appears within the next 8
            for j in range(i + 1, min(i + 9, len(ids))):
                if int(ids[j]) == vt:
                    val_pos.setdefault(e, []).append((j, i))
                    break

    # prompt-side data
    PRideo = {}
    for e, _v in stable:
        for tag, tmpl in (("A", A_PREFIX), ("B", B_PREFIX)):
            pids = tok.encode(tmpl.format(e=e))
            Hp = forwards_all_layers(s, pids)[LAYER]
            Gp = gates_from(s, pids)
            pe = entity_ends(tok, pids, [e])[-1]
            PRideo[(e, tag)] = {"ids": pids, "H": Hp, "G": Gp,
                                "ent_end": pe}
    print("sondes prompts faites", flush=True)

    R = {"rules": {}}
    for rname in ("r1_g2.5", "r2_maxg16", "r3_g4.0"):
        pts = rule_points(rname, G, len(ids))
        anch = anchors_from(pts, len(ids))
        # write-side anchor accuracy: value tokens anchored on their
        # entity's end position
        ok_w = tot_w = 0
        for e, pairs in val_pos.items():
            for (j, i) in pairs:
                tot_w += 1
                ok_w += int(anch[j - 1] == i)
        # query-side anchor accuracy
        ok_q = tot_q = 0
        exq = []
        for e, _v in stable:
            for tag in ("A", "B"):
                d = PRideo[(e, tag)]
                qa = query_anchor(rname, d["G"], len(d["ids"]))
                tot_q += 1
                ok_q += int(qa == d["ent_end"])
                if len(exq) < 4:
                    exq.append(f"{e}/{tag}: ancre="
                               f"{tok.decode([d['ids'][qa]])!r}")
        acc_w, acc_q = ok_w / max(1, tot_w), ok_q / max(1, tot_q)
        print(f"[{rname}] {len(pts)} ancres doc | anchor-acc ecriture "
              f"{acc_w:.0%} ({ok_w}/{tot_w}) | requete {acc_q:.0%} "
              f"({ok_q}/{tot_q}) | ex: {'; '.join(exq)}", flush=True)

        # retrieval with the minimal recipe (SimHash, no ZCA)
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
        rows = []
        for e, v in stable:
            vid = tok.encode(" " + v)[0]
            rr = {}
            for tag in ("A", "B"):
                d = PRideo[(e, tag)]
                qa = query_anchor(rname, d["G"], len(d["ids"]))
                qS = m.sem_key(d["H"][qa])
                _, sS = m.scores(m.MS, qS)
                rr[tag] = int((sS > sS[vid]).sum()) + 1
            rows.append({"e": e, "rank_A": rr["A"], "rank_B": rr["B"]})
        n = len(rows)
        agg = {"A_top10": sum(r["rank_A"] <= 10 for r in rows) / n,
               "B_top10": sum(r["rank_B"] <= 10 for r in rows) / n,
               "median_A": float(np.median([r["rank_A"]
                                            for r in rows])),
               "median_B": float(np.median([r["rank_B"]
                                            for r in rows]))}
        R["rules"][rname] = {"anchor_acc_write": acc_w,
                             "anchor_acc_query": acc_q,
                             "agg": agg, "rows": rows}
        print(f"    recup : A top10 {agg['A_top10']:.0%} (med "
              f"{agg['median_A']:.0f}) | B top10 {agg['B_top10']:.0%} "
              f"(med {agg['median_B']:.0f})", flush=True)

    json.dump(R, open(os.path.join(HERE, "results",
                                   "semantic_rules_qwen.json"), "w"),
              indent=1)
    print("saved -> results/semantic_rules_qwen.json")
    shutil.rmtree(STATE_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
