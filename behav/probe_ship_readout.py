"""Integration 1.4.0 : le readout du tier v2 dans le chemin LIVRE.

Les scores d'un tier SimHash sont plus tasses que ceux du prototype a
cles denses, donc softmax(beta*s) n'a pas le meme pic : beta et lambda
doivent etre re-mesures ici. Grille sur les 10 faits DEV, rapport sur
les 10 faits TEST jamais vus par la grille, localite sur 10 temoins.

    python probe_ship_readout.py [gpt2|qwen]
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
from behavioral import (A_PREFIX, B_PREFIX, ENTS, VALS,        # noqa: E402
                        WITNESS, build_doc)

CFG = {"qwen": {"layer": 1, "whiten": False},
       "gpt2": {"layer": 5, "whiten": True}}
BETAS = (10.0, 20.0, 40.0, 80.0, 160.0)
LAMS = (0.5, 0.85)


def main():
    import shutil
    import torch
    which = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    cfg = CFG[which]
    state = os.path.join(HERE, f".ro_{which}_state")
    docf = os.path.join(HERE, f"_ro_{which}.txt")
    shutil.rmtree(state, ignore_errors=True)
    doc = build_doc(list(zip(ENTS[:20], VALS[:20])), seed=5, reps=4,
                    block=60)
    open(docf, "w", encoding="utf-8").write(doc)
    s = Sillage(model=which, state=state, quiet=True,
                sem2=cfg["layer"], sem2_whiten=cfg["whiten"],
                fastweights=False)
    r = s.read(docf)[0]
    mem, (tok, model) = s.mem, s.load_model()
    res = np.array(mem.res_S)
    thr = float(np.quantile(res, 0.90))
    print(f"{which}: {r['tokens']} tokens en {r['minutes']:.1f} min | "
          f"couche {cfg['layer']}, blanchiment {cfg['whiten']} | "
          f"thr(q90) {thr:.3f}", flush=True)

    def pooled(prompt):
        ids = tok.encode(prompt)
        with torch.no_grad():
            out = model(torch.tensor([ids], device=s.device),
                        output_hidden_states=True)
        H2 = out.hidden_states[mem.sem2_layer][0].float().cpu().numpy()
        pl = None
        for k in range(0, len(H2), 64):
            Q = np.stack([mem.sem2_key(H2[p]) for p in
                          range(k, min(k + 64, len(H2)))])
            U = Q @ mem.MS
            S = (U / (np.linalg.norm(U, axis=1, keepdims=True)
                      + 1e-8)) @ mem.V.T
            m = S.max(axis=0)
            pl = m if pl is None else np.maximum(pl, m)
        pl[list(set(int(t) for t in ids))] = -1e9
        return ids, pl

    def gen(ids, pl, beta, lam, n=8, mode="sustained"):
        fire = pl is not None and float(pl.max()) >= thr
        if fire:
            ps = np.exp(beta * (pl - pl.max()))
            ps /= ps.sum()
        g = list(ids)
        for step in range(n):
            with torch.no_grad():
                o = model(torch.tensor([g], device=s.device))
            lg = o.logits[0, -1].float().cpu().numpy()
            p = np.exp(lg - lg.max())
            p /= p.sum()
            if fire:
                # the tier gives the IMPULSE; the frozen model finishes
                # the word (its pieces are predictable once the head is
                # out) -- the generation-side twin of write-time word
                # integrity
                if mode == "sustained":
                    w = lam
                elif mode == "impulse":
                    w = lam if step == 0 else 0.0
                else:
                    w = lam * (0.5 ** step)
                if w > 0:
                    p = (1 - w) * p + w * ps
            g.append(int(np.argmax(p)))
        return tok.decode(g[len(ids):])

    facts = list(zip(ENTS[:20], VALS[:20]))
    dev, test = facts[:10], facts[10:]
    PB = {e: pooled(B_PREFIX.format(e=e)) for e, _v in facts}
    PA = {e: pooled(A_PREFIX.format(e=e)) for e, _v in facts}
    wit = [x.strip() for x in WITNESS.replace("\n", " ").split(".")
           if len(x.strip()) > 20][:10]
    WP = [pooled(w[: max(20, len(w) // 2)]) for w in wit]
    base_w = [gen(i, None, 10.0, 0.85) for i, _ in WP]

    # how peaked is the pooled distribution? (why beta must move)
    e0 = facts[0][0]
    pl0 = PB[e0][1]
    srt = np.sort(pl0)[::-1]
    print(f"  forme des scores poules ({e0}) : max {srt[0]:.3f}, "
          f"q99 {srt[len(srt)//100]:.3f}, med {np.median(pl0):.3f}",
          flush=True)

    best = None
    for mode in ("sustained", "impulse", "decay"):
        for beta in BETAS:
            for lam in LAMS:
                hd = sum(v.split()[0] in gen(*PB[e], beta, lam,
                                             mode=mode)
                         for e, v in dev)
                if best is None or hd > best[3]:
                    best = (mode, beta, lam, hd)
        print(f"  dev {mode:9s} -> meilleur {best[3]}/10 "
              f"({best[0]}, beta {best[1]}, lam {best[2]})", flush=True)
    mode, beta, lam, hd = best
    hb = sum(v.split()[0] in gen(*PB[e], beta, lam, mode=mode)
             for e, v in test)
    ha = sum(v.split()[0] in gen(*PA[e], beta, lam, mode=mode)
             for e, v in test)
    chg = sum(gen(i, pl, beta, lam, mode=mode) != b
              for (i, pl), b in zip(WP, base_w))
    print(f"\nretenu {mode}, beta {beta}, lam {lam} (dev B {hd}/10)",
          flush=True)
    print(f"TEST (10 faits jamais vus) : B {hb}/10 ({hb/10:.0%}) | "
          f"A {ha}/10 | localite {chg}/10", flush=True)
    out = os.path.join(HERE, "results", f"ship_readout_{which}.json")
    json.dump({"model": which, "mode": mode, "layer": cfg["layer"],
               "whiten": cfg["whiten"], "thr_q": 0.90, "thr": thr,
               "beta": beta, "lam": lam, "dev_B": hd / 10,
               "test_B": hb / 10, "test_A": ha / 10,
               "locality_changed": chg / 10},
              open(out, "w"), indent=1)
    print(f"saved -> {out}")
    shutil.rmtree(state, ignore_errors=True)
    os.remove(docf)


if __name__ == "__main__":
    main()
