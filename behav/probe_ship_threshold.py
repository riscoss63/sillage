"""Integration 1.4.0 : ou placer le seuil d'abstention du tier v2 ?

Balaye le quantile du null in-document et mesure, pour chacun, le
rappel paraphrase ET la localite (temoins dont la completion greedy
change) -- la metrique du papier 6. Critere DECLARE : le plus petit
quantile dont la localite reste <= 1/10.

    python probe_ship_threshold.py [qwen|gpt2]
"""

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


def main():
    import shutil
    import torch
    which = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    cfg = CFG[which]
    state = os.path.join(HERE, f".ship_{which}_state")
    docf = os.path.join(HERE, f"_ship_{which}.txt")
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
    print(f"{which}: {r['tokens']} tokens en {r['minutes']:.1f} min | "
          f"couche {cfg['layer']}, blanchiment {cfg['whiten']} | null "
          f"n={len(res)}", flush=True)

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

    def gen(ids, pl, thr, beta=10.0, lam=0.85, n=8):
        fire = pl is not None and float(pl.max()) >= thr
        if fire:
            ps = np.exp(beta * (pl - pl.max()))
            ps /= ps.sum()
        g = list(ids)
        for _ in range(n):
            with torch.no_grad():
                o = model(torch.tensor([g], device=s.device))
            lg = o.logits[0, -1].float().cpu().numpy()
            p = np.exp(lg - lg.max())
            p /= p.sum()
            if fire:
                p = (1 - lam) * p + lam * ps
            g.append(int(np.argmax(p)))
        return tok.decode(g[len(ids):])

    facts = list(zip(ENTS[:20], VALS[:20]))
    FB = [(e, v) + pooled(B_PREFIX.format(e=e)) for e, v in facts]
    FA = [(e, v) + pooled(A_PREFIX.format(e=e)) for e, v in facts]
    wit = [x.strip() for x in WITNESS.replace("\n", " ").split(".")
           if len(x.strip()) > 20][:10]
    WP = [pooled(w[: max(20, len(w) // 2)]) for w in wit]
    base_w = [gen(i, None, np.inf) for i, _ in WP]
    print("critere declare : plus petit quantile avec localite <= 1/10",
          flush=True)
    rows = []
    for q in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        thr = float(np.quantile(res, q))
        hb = sum(v.split()[0] in gen(i, pl, thr) for _e, v, i, pl in FB)
        ha = sum(v.split()[0] in gen(i, pl, thr) for _e, v, i, pl in FA)
        chg = sum(gen(i, pl, thr) != b
                  for (i, pl), b in zip(WP, base_w))
        rows.append((q, thr, hb, ha, chg))
        print(f"  q{int(q*100)} (thr {thr:.3f}) : B {hb}/20 "
              f"({hb/20:.0%}) | A {ha}/20 | temoins changes {chg}/10",
              flush=True)
    ok = [r for r in rows if r[4] <= 1]
    if ok:
        best = min(ok, key=lambda r: r[0])
        print(f"\nretenu : q{int(best[0]*100)} -> B {best[2]}/20, "
              f"localite {best[4]}/10", flush=True)
    else:
        print("\naucun quantile ne tient la localite -- le max poole "
              "seul ne suffit pas comme detecteur", flush=True)
    shutil.rmtree(state, ignore_errors=True)
    os.remove(docf)


if __name__ == "__main__":
    main()
