"""Marche 2e : le tier M_E reconstruit sur la COUCHE 1 adresse-t-il ?

Quatre variantes : {ancre oracle, ancre surprise} x {SimHash bande (la
fonction du tier livre), cle dense}. Memes 20 faits stables, rangs de
la vraie valeur aux prefixes A et B.

    python probe_semantic_l1.py
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
from sillage.core import CAP, SillageMemory                    # noqa: E402
from behavioral import (ALT, A_PREFIX, B_PREFIX, ENTS, VALS,   # noqa: E402
                        build_doc)
from probe_semantic_anchor import (anchors_from, entity_ends)  # noqa: E402
from probe_semantic_layers import forwards_all_layers          # noqa: E402

LAYER = 1
G_ANCHOR = 2.5
D_DENSE = 12288
STATE_TMP = os.path.join(HERE, ".l1_tmp_state")


def gates_from(s, ids):
    import torch
    tok, model = s.load_model()
    n = len(ids)
    G = np.zeros(n, np.float32)
    x = torch.tensor(ids, device=s.device)
    a, W, S = 0, 1024, 512
    with torch.no_grad():
        while a < n:
            w = min(W, n - a)
            out = model(x[a:a + w].unsqueeze(0))
            lg = out.logits[0].float()
            lo = 0 if a == 0 else W - S
            hi = min(w, n - a - 1)
            if hi > lo:
                tr = x[a + lo + 1:a + hi + 1]
                lp = (torch.log_softmax(lg[lo:hi], dim=-1)
                      .gather(1, tr.unsqueeze(1))[:, 0].cpu().numpy())
                G[a + lo + 1:a + hi + 1] = np.clip(-lp, 0.0, CAP)
            if a + w >= n:
                break
            a += S
    return G


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
    Hall = forwards_all_layers(s, ids)
    H = Hall[LAYER]
    G = gates_from(s, ids)

    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    mu = Hn.mean(axis=0)

    rng = np.random.default_rng(2026)
    P = (rng.normal(size=(1024, D_DENSE))
         / np.sqrt(1024)).astype(np.float32)

    def dense_key(h):
        z = h / (np.linalg.norm(h) + 1e-8) - mu
        z = z / (np.linalg.norm(z) + 1e-8)
        q = z @ P
        return (q / (np.linalg.norm(q) + 1e-8)).astype(np.float32)

    def make_sim_tier():
        m = SillageMemory(None, "qwen", semantic=True, fastweights=False)
        m.set_vocab(151936)
        m.mu = mu.copy()
        m.mu_n = 10 ** 9
        return m

    ends = entity_ends(tok, ids, [e for e, _ in facts])
    surp = [t for t in range(len(ids)) if G[t] >= G_ANCHOR]

    # prompt-side: layer-1 hiddens + gates per prompt
    PR = {}
    for e, _v in stable:
        for tag, tmpl in (("A", A_PREFIX), ("B", B_PREFIX)):
            pids = tok.encode(tmpl.format(e=e))
            Hp = forwards_all_layers(s, pids)[LAYER]
            Gp = gates_from(s, pids)
            pe = entity_ends(tok, pids, [e])
            cand = [t for t in range(len(pids)) if Gp[t] >= G_ANCHOR]
            PR[(e, tag)] = {"oracle": Hp[pe[-1]],
                            "surprise": Hp[cand[-1] if cand
                                           else len(pids) - 1]}
    print("sondes prompts faites", flush=True)

    R = {"layer": LAYER, "variants": {}}
    for arule, points in (("oracle", ends), ("surprise", surp)):
        anch = anchors_from(points, len(ids))
        for kfun in ("simhash", "dense"):
            m = make_sim_tier()
            key = (m.sem_key if kfun == "simhash" else dense_key)
            for t in range(len(ids) - 1):
                a = int(anch[t])
                if a < 0:
                    continue
                qS = key(H[a])
                uS, _ = m.scores(m.MS, qS)
                m.amp_write(m.MS, qS, uS, int(ids[t + 1]),
                            float(G[t + 1]))
            rows = []
            for e, v in stable:
                vid = tok.encode(" " + v)[0]
                rr = {}
                for tag in ("A", "B"):
                    qS = key(PR[(e, tag)][arule])
                    _, sS = m.scores(m.MS, qS)
                    rr[tag] = int((sS > sS[vid]).sum()) + 1
                rows.append({"e": e, "rank_A": rr["A"],
                             "rank_B": rr["B"]})
            n = len(rows)
            agg = {"A_top1": sum(r["rank_A"] == 1 for r in rows) / n,
                   "A_top10": sum(r["rank_A"] <= 10 for r in rows) / n,
                   "B_top1": sum(r["rank_B"] == 1 for r in rows) / n,
                   "B_top10": sum(r["rank_B"] <= 10 for r in rows) / n,
                   "median_A": float(np.median([r["rank_A"]
                                                for r in rows])),
                   "median_B": float(np.median([r["rank_B"]
                                                for r in rows]))}
            R["variants"][f"{arule}+{kfun}"] = {"agg": agg,
                                                "rows": rows}
            print(f"== {arule}+{kfun}: A top1 {agg['A_top1']:.0%} "
                  f"top10 {agg['A_top10']:.0%} (med {agg['median_A']:.0f})"
                  f" | B top1 {agg['B_top1']:.0%} top10 "
                  f"{agg['B_top10']:.0%} (med {agg['median_B']:.0f})",
                  flush=True)

    out = os.path.join(HERE, "results", "semantic_l1_qwen.json")
    json.dump(R, open(out, "w"), indent=1)
    print(f"\nsaved -> {out}")
    shutil.rmtree(STATE_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
