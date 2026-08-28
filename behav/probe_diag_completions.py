"""Etape 1 : diagnostic instrumente des completions du tier v2.

Config gelee (g_min 0.5, beta 10, lam 0.85, thr q95). Pour chaque
prompt B (20 test + 10 dev) : la completion greedy complete, et au pas
0 les rangs de la valeur dans p_base / p_sem / p_mix, le top-3 de
p_sem decode, le tir de sE. Categorise chaque rate.

    python probe_diag_completions.py
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
G_MIN, BETA, LAM = 0.5, 10.0, 0.85
N_GEN = 8
STATE_TMP = os.path.join(HERE, ".diag_tmp_state")


def main():
    import shutil
    import torch
    shutil.rmtree(STATE_TMP, ignore_errors=True)
    s = Sillage(model="qwen", state=STATE_TMP, quiet=True)
    tok, model = s.load_model()

    facts = list(zip(ENTS[:30], VALS[:30]))
    changed_e = {e for e, _ in
                 [(e, ALT[i]) for i, (e, _v) in enumerate(facts[:10])]}

    doc = build_doc(facts, seed=0)
    ids = tok.encode(doc)
    print(f"dossier v1 : {len(ids)} tokens ; forwards...", flush=True)
    H = forwards_all_layers(s, ids)[LAYER]
    G = gates_from(s, ids)
    mu = fixed_mu(H)
    pts = [t for t in range(len(ids)) if G[t] >= 2.5]
    anch = anchors_from(pts, len(ids))
    m = SillageMemory(None, "qwen", semantic=True, fastweights=False)
    m.set_vocab(151936)
    m.mu = mu.copy()
    m.mu_n = 10 ** 9
    for t in range(len(ids) - 1):
        a = int(anch[t])
        if a < 0 or float(G[t + 1]) < G_MIN:
            continue
        qS = m.sem_key(H[a])
        uS, _ = m.scores(m.MS, qS)
        m.amp_write(m.MS, qS, uS, int(ids[t + 1]), float(G[t + 1]))
    print("tier construit (g_min 0.5)", flush=True)

    def pooled_for(prompt):
        Hp = forwards_all_layers(s, tok.encode(prompt))[LAYER]
        best = None
        for p in range(len(Hp)):
            qS = m.sem_key(Hp[p])
            _, sS = m.scores(m.MS, qS)
            best = sS.copy() if best is None else np.maximum(best, sS)
        return best

    wit = [x.strip() for x in WITNESS.replace("\n", " ").split(".")
           if len(x.strip()) > 20][:20]
    null_max = [float(pooled_for(w[: max(20, len(w) // 2)]).max())
                for w in wit]
    thr = float(np.quantile(null_max, 0.95))
    print(f"thr = {thr:.4f}", flush=True)

    def rank_in(vec, tid):
        return int((vec > vec[tid]).sum()) + 1

    rows = []
    cats = {"hit": 0, "a_bucket": 0, "b_mix": 0, "c_derail": 0,
            "d_nofire": 0, "e_cross": 0}
    all_val_heads = {tok.encode(" " + v)[0]: e for e, v in facts}
    for e, v in facts:
        split = "dev" if e in changed_e else "test"
        vid = tok.encode(" " + v)[0]
        prompt = B_PREFIX.format(e=e)
        sE = pooled_for(prompt)
        fire = float(sE.max()) >= thr
        p_sem = np.exp(BETA * (sE - sE.max()))
        p_sem = p_sem / p_sem.sum()
        pids = list(tok.encode(prompt))
        gen_ids = list(pids)
        step0 = None
        for step in range(N_GEN):
            with torch.no_grad():
                out = model(torch.tensor([gen_ids], device=s.device))
            lg = out.logits[0, -1].float().cpu().numpy()
            p_base = np.exp(lg - lg.max())
            p_base = p_base / p_base.sum()
            p_mix = ((1 - LAM) * p_base + LAM * p_sem
                     if fire else p_base)
            if step == 0:
                top3 = np.argsort(-p_sem)[:3]
                step0 = {"fire": bool(fire),
                         "rank_base": rank_in(p_base, vid),
                         "rank_sem": rank_in(p_sem, vid),
                         "rank_mix": rank_in(p_mix, vid),
                         "sem_top3": [tok.decode([int(t3)])
                                      for t3 in top3],
                         "sem_top3_ids": [int(t3) for t3 in top3]}
            gen_ids.append(int(np.argmax(p_mix)))
        out_txt = tok.decode(gen_ids[len(pids):])
        hit = v.split()[0] in out_txt
        if hit:
            cat = "hit"
        elif not fire:
            cat = "d_nofire"
        elif step0["rank_sem"] > 1:
            top_id = step0["sem_top3_ids"][0]
            cat = ("e_cross" if top_id in all_val_heads
                   and all_val_heads[top_id] != e else "a_bucket")
        elif step0["rank_mix"] > 1:
            cat = "b_mix"
        else:
            cat = "c_derail"
        cats[cat] += 1
        rows.append({"e": e, "split": split, "hit": bool(hit),
                     "cat": cat, "out": out_txt, **step0})
        print(f"  [{split}] {e:12s} {'HIT ' if hit else cat:9s} "
              f"sem_top3={step0['sem_top3']} rangs "
              f"base/sem/mix={step0['rank_base']}/"
              f"{step0['rank_sem']}/{step0['rank_mix']} "
              f"out={out_txt[:45]!r}", flush=True)

    print(f"\n== categories (30 faits) == {cats}", flush=True)
    json.dump({"thr": thr, "cats": cats, "rows": rows},
              open(os.path.join(HERE, "results",
                                "semantic_diag_completions.json"), "w"),
              indent=1)
    print("saved -> results/semantic_diag_completions.json")
    shutil.rmtree(STATE_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
