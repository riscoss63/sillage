"""Marche 2f : blanchiment ZCA complet des cles couche-1.

La superposition exige des cles decorrelees ; la soustraction de
moyenne ne retire qu'un rang. Ici : ZCA = Cov^{-1/2} (eigh,
retrecissement 0.1) estimee sur les hiddens du dossier. Mesures :
cosinus meme-entite vs null APRES ZCA, puis tier dense-ZCA (ancres
oracle) et rangs A/B.

    python probe_semantic_zca.py
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
from probe_semantic_anchor import (anchors_from, entity_ends)  # noqa: E402
from probe_semantic_layers import forwards_all_layers          # noqa: E402
from probe_semantic_l1 import gates_from                       # noqa: E402

LAYER = 1
D_DENSE = 12288
SHRINK = 0.1
STATE_TMP = os.path.join(HERE, ".zca_tmp_state")


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

    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    mu = Hn.mean(axis=0)
    X = Hn - mu
    C = (X.T @ X) / len(X)
    C = (1 - SHRINK) * C + SHRINK * np.eye(1024,
                                           dtype=np.float32) * C.trace() / 1024
    w, V = np.linalg.eigh(C)
    W_zca = (V * (1.0 / np.sqrt(np.maximum(w, 1e-8)))) @ V.T
    W_zca = W_zca.astype(np.float32)

    rng = np.random.default_rng(2026)
    P = (rng.normal(size=(1024, D_DENSE))
         / np.sqrt(1024)).astype(np.float32)

    def zca(h):
        z = h / (np.linalg.norm(h) + 1e-8) - mu
        z = z @ W_zca
        return z / (np.linalg.norm(z) + 1e-8)

    def key(h):
        q = zca(h) @ P
        return (q / (np.linalg.norm(q) + 1e-8)).astype(np.float32)

    ends_by = {e: entity_ends(tok, ids, [e]) for e, _ in stable}
    Zdoc = np.stack([zca(H[i]) for i in range(len(ids))])

    # prompt side
    PR = {}
    for e, _v in stable:
        for tag, tmpl in (("A", A_PREFIX), ("B", B_PREFIX)):
            pids = tok.encode(tmpl.format(e=e))
            Hp = forwards_all_layers(s, pids)[LAYER]
            pe = entity_ends(tok, pids, [e])[-1]
            PR[(e, tag)] = Hp[pe]
    print("sondes prompts faites", flush=True)

    # (i) cosines after ZCA
    same_l, null_l = {"A": [], "B": []}, {"A": [], "B": []}
    for e, _v in stable:
        for tag in ("A", "B"):
            zp = zca(PR[(e, tag)])
            same_l[tag].append(float(np.max(Zdoc[ends_by[e]] @ zp)))
            others = [i for e2, _ in stable if e2 != e
                      for i in ends_by[e2]]
            null_l[tag].append(float(np.max(Zdoc[others] @ zp)))
    cos_agg = {t: {"same": float(np.median(same_l[t])),
                   "null": float(np.median(null_l[t]))}
               for t in ("A", "B")}
    print(f"cos ZCA : A same {cos_agg['A']['same']:.3f} null "
          f"{cos_agg['A']['null']:.3f} | B same {cos_agg['B']['same']:.3f}"
          f" null {cos_agg['B']['null']:.3f}", flush=True)

    # (ii) the 2x2 deliverability matrix: anchor rule x key function
    def simhash_zca_tier():
        m = SillageMemory(None, "qwen", semantic=True, fastweights=False)
        m.set_vocab(151936)
        m.mu = np.zeros(1024, np.float32)
        m.mu_n = 10 ** 9
        return m

    ends = entity_ends(tok, ids, [e for e, _ in facts])
    from probe_semantic_l1 import G_ANCHOR
    surp = [t for t in range(len(ids)) if G[t] >= G_ANCHOR]
    # prompt-side surprise anchors
    PRS = {}
    for e, _v in stable:
        for tag, tmpl in (("A", A_PREFIX), ("B", B_PREFIX)):
            pids = tok.encode(tmpl.format(e=e))
            Hp = forwards_all_layers(s, pids)[LAYER]
            Gp = gates_from(s, pids)
            cand = [t for t in range(len(pids)) if Gp[t] >= G_ANCHOR]
            PRS[(e, tag)] = Hp[cand[-1] if cand else len(pids) - 1]

    results = {"cos": cos_agg, "variants": {}}
    for arule, points in (("oracle", ends), ("surprise", surp)):
        anch = anchors_from(points, len(ids))
        for kfun in ("dense", "simhash"):
            if kfun == "dense":
                m = SillageMemory(None, "qwen", semantic=True,
                                  fastweights=False)
                m.set_vocab(151936)
                kf = key
            else:
                m = simhash_zca_tier()
                kf = lambda h: m.sem_key(zca(h))    # noqa: E731
            for t in range(len(ids) - 1):
                a = int(anch[t])
                if a < 0:
                    continue
                qS = kf(H[a])
                uS, _ = m.scores(m.MS, qS)
                m.amp_write(m.MS, qS, uS, int(ids[t + 1]),
                            float(G[t + 1]))
            rows = []
            for e, v in stable:
                vid = tok.encode(" " + v)[0]
                rr = {}
                for tag in ("A", "B"):
                    hq = (PR[(e, tag)] if arule == "oracle"
                          else PRS[(e, tag)])
                    qS = kf(hq)
                    _, sS = m.scores(m.MS, qS)
                    rr[tag] = int((sS > sS[vid]).sum()) + 1
                rows.append({"e": e, "rank_A": rr["A"],
                             "rank_B": rr["B"]})
            n = len(rows)
            agg = {"A_top1": sum(r["rank_A"] == 1 for r in rows) / n,
                   "A_top10": sum(r["rank_A"] <= 10
                                  for r in rows) / n,
                   "B_top1": sum(r["rank_B"] == 1 for r in rows) / n,
                   "B_top10": sum(r["rank_B"] <= 10
                                  for r in rows) / n,
                   "median_A": float(np.median([r["rank_A"]
                                                for r in rows])),
                   "median_B": float(np.median([r["rank_B"]
                                                for r in rows]))}
            results["variants"][f"{arule}+{kfun}"] = {"agg": agg,
                                                      "rows": rows}
            print(f"== {arule}+{kfun}: A top10 {agg['A_top10']:.0%} "
                  f"(med {agg['median_A']:.0f}) | B top10 "
                  f"{agg['B_top10']:.0%} (med {agg['median_B']:.0f})",
                  flush=True)

    json.dump(results,
              open(os.path.join(HERE, "results",
                                "semantic_zca_qwen.json"), "w"),
              indent=1)
    print("saved -> results/semantic_zca_qwen.json")
    shutil.rmtree(STATE_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
