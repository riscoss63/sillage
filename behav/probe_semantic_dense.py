"""Marche 2c : geometrie ou fonction de cle ?

(i)  Cosinus brut entre le hidden d'une entite dans le dossier et dans
     les prompts A/B (blanchi, mu fige), null inter-entites.
(ii) Tier DENSE : memes ancres oracle, meme amp_write, mais la cle est
     la projection aleatoire FIXE du hidden blanchi, sans quantisation
     SimHash. Si le dense adresse et le SimHash non, la quantisation
     est le tueur et le remede est livrable.

    python probe_semantic_dense.py
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
                                   fixed_mu, forwards)

STATE_TMP = os.path.join(HERE, ".dense_tmp_state")
D_DENSE = 12288                 # same width as M_S


def whiten(H, mu):
    Z = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8) - mu
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)


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
    H, G = forwards(s, ids)
    mu = fixed_mu(H)

    rng = np.random.default_rng(2026)
    P = (rng.normal(size=(1024, D_DENSE))
         / np.sqrt(1024)).astype(np.float32)

    def dense_key(h):
        z = h / (np.linalg.norm(h) + 1e-8) - mu
        z = z / (np.linalg.norm(z) + 1e-8)
        q = z @ P
        return (q / (np.linalg.norm(q) + 1e-8)).astype(np.float32)

    # ---- (ii) dense tier on oracle anchors ------------------------------
    ends = entity_ends(tok, ids, [e for e, _ in facts])
    anch = anchors_from(ends, len(ids))
    m = SillageMemory(None, "qwen", semantic=True, fastweights=False)
    m.set_vocab(151936)
    for t in range(len(ids) - 1):
        a = int(anch[t])
        if a < 0:
            continue
        qS = dense_key(H[a])
        uS, _ = m.scores(m.MS, qS)
        m.amp_write(m.MS, qS, uS, int(ids[t + 1]), float(G[t]))
    print("tier dense construit (ancres oracle)", flush=True)

    # per-entity doc hiddens at their occurrence ends
    ends_by = {e: entity_ends(tok, ids, [e]) for e, _ in stable}

    R = {"rows": [], "cos_null": None}
    Zdoc = whiten(H, mu)
    null = []
    for e, v in stable:
        vid = tok.encode(" " + v)[0]
        row = {"e": e}
        for tag, tmpl in (("A", A_PREFIX), ("B", B_PREFIX)):
            prompt = tmpl.format(e=e)
            pids = tok.encode(prompt)
            Hp, _ = forwards(s, pids)
            pe = entity_ends(tok, pids, [e])
            hp = Hp[pe[-1]]
            zp = whiten(hp[None, :], mu)[0]
            cos = float(np.max(Zdoc[ends_by[e]] @ zp))
            # null: this prompt hidden vs OTHER entities' doc hiddens
            others = [i for e2, _ in stable if e2 != e
                      for i in ends_by[e2]]
            null.append(float(np.max(Zdoc[others] @ zp)))
            qS = dense_key(hp)
            _, sS = m.scores(m.MS, qS)
            rank = int((sS > sS[vid]).sum()) + 1
            row[tag] = {"cos_same": cos, "rank_dense": rank}
        R["rows"].append(row)
        print(f"  {e:12s} A: cos {row['A']['cos_same']:.3f} rang "
              f"{row['A']['rank_dense']:>6d} | B: cos "
              f"{row['B']['cos_same']:.3f} rang "
              f"{row['B']['rank_dense']:>6d}", flush=True)

    n = len(R["rows"])
    agg = {
        "median_cos_A": float(np.median([r["A"]["cos_same"]
                                         for r in R["rows"]])),
        "median_cos_B": float(np.median([r["B"]["cos_same"]
                                         for r in R["rows"]])),
        "median_cos_null": float(np.median(null)),
        "A_top10_dense": sum(r["A"]["rank_dense"] <= 10
                             for r in R["rows"]) / n,
        "B_top10_dense": sum(r["B"]["rank_dense"] <= 10
                             for r in R["rows"]) / n,
        "A_top1_dense": sum(r["A"]["rank_dense"] == 1
                            for r in R["rows"]) / n,
        "median_rank_A": float(np.median([r["A"]["rank_dense"]
                                          for r in R["rows"]])),
        "median_rank_B": float(np.median([r["B"]["rank_dense"]
                                          for r in R["rows"]])),
    }
    R["agg"] = agg
    print("\n== agregats ==")
    for k, val in agg.items():
        print(f"  {k:16s}: {val:.3f}" if "cos" in k else
              f"  {k:16s}: {val}")
    out = os.path.join(HERE, "results", "semantic_dense_qwen.json")
    json.dump(R, open(out, "w"), indent=1)
    print(f"\nsaved -> {out}")
    shutil.rmtree(STATE_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
