"""Axe 4 : la porte de surprise survit-elle a la quantification ?

Lire avec un modele quantifie fait bouger les log-probabilites, donc la
porte g = clip(-ln p, 0, 5), donc les ecritures. Cette sonde lit le meme
document dans deux precisions et compare ce qui compte : les portes
elles-memes, les admissions du cold store (la regle des deux
occurrences), et le rappel.

Seuils declares AVANT le run : correlation des portes > 0.98 et
admissions identiques a 1 % pres, sinon le mode quantifie est annonce
comme approximatif -- mesure, pas suppose.

    python probe_quantised_gate.py [qwen|gpt2] [int8|bfloat16]
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
from sillage.core import CAP                                   # noqa: E402
from behavioral import (A_PREFIX, ENTS, VALS, build_doc)       # noqa: E402


def gates_of(s, ids):
    """The surprise gate at every position, as the reader sees it."""
    import torch
    tok, model = s.load_model()
    x = torch.tensor(ids, device=s.device)
    n = len(ids) - 1
    G = np.zeros(n, np.float32)
    with torch.no_grad():
        out = model(x[:n + 1].unsqueeze(0))
    lg = out.logits[0].float()
    lp = torch.log_softmax(lg[:n], dim=-1).gather(
        1, x[1:n + 1].unsqueeze(1))[:, 0].cpu().numpy()
    G[:] = np.clip(-lp, 0.0, CAP)
    return G


def main():
    import shutil
    import time
    which = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    mode = sys.argv[2] if len(sys.argv) > 2 else "int8"
    facts = list(zip(ENTS[:14], VALS[:14]))
    doc = build_doc(facts, seed=5, reps=3, block=40)
    path = os.path.join(HERE, f"_q_{which}.txt")
    open(path, "w", encoding="utf-8").write(doc)
    test = facts[7:]

    R = {"model": which, "mode": mode, "arms": {}}
    gates, colds = {}, {}
    for tag, dtype in (("float32", None), (mode, mode)):
        st = os.path.join(HERE, f".q_{which}_{tag}")
        shutil.rmtree(st, ignore_errors=True)
        s = Sillage(model=which, state=st, quiet=False,
                    fastweights=False, dtype=dtype)
        tok, _ = s.load_model()
        ids = tok.encode(doc)[:800]
        t0 = time.time()
        gates[tag] = gates_of(s, ids)
        fwd = time.time() - t0
        t0 = time.time()
        s.read(path, fast=True)
        read_s = time.time() - t0
        colds[tag] = {g: sum(slot[1].values())
                      for g, slot in s.mem.cold.items()}
        hits = sum(v.split()[0] in s.complete(A_PREFIX.format(e=e), n=8)
                   for e, v in test)
        R["arms"][tag] = {"forward_s": round(fwd, 2),
                          "read_s": round(read_s, 1),
                          "cold_grams": len(s.mem.cold),
                          "recall": hits / len(test)}
        print(f"  {tag:9s}: forward {fwd:5.2f}s | lecture {read_s:5.1f}s "
              f"| {len(s.mem.cold)} grams | rappel {hits}/{len(test)}",
              flush=True)
        shutil.rmtree(st, ignore_errors=True)

    a, b = gates["float32"], gates[mode]
    n = min(len(a), len(b))
    corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
    mad = float(np.mean(np.abs(a[:n] - b[:n])))
    ca, cb = colds["float32"], colds[mode]
    admitted_a = {g for g, c in ca.items() if c >= 2}
    admitted_b = {g for g, c in cb.items() if c >= 2}
    inter = len(admitted_a & admitted_b)
    union = max(1, len(admitted_a | admitted_b))
    R["gate_corr"] = round(corr, 4)
    R["gate_mad_nats"] = round(mad, 4)
    R["admissions_jaccard"] = round(inter / union, 4)
    R["verdict"] = ("faithful" if corr > 0.98
                    and abs(len(admitted_a) - len(admitted_b))
                    <= 0.01 * max(1, len(admitted_a)) else "approximate")
    print(f"\n  portes : correlation {corr:.4f}, ecart moyen {mad:.4f} "
          f"nats")
    print(f"  admissions (>=2) : {len(admitted_a)} vs {len(admitted_b)} "
          f"| Jaccard {inter/union:.3f}")
    print(f"  VERDICT (seuils declares) : {R['verdict']}")
    out = os.path.join(HERE, "results", f"quantised_{which}_{mode}.json")
    json.dump(R, open(out, "w"), indent=1)
    print(f"saved -> {out}")
    os.remove(path)


if __name__ == "__main__":
    main()
