"""Marche 2d : l'identite d'entite vit-elle dans une couche
intermediaire ? Cosinus meme-entite doc<->prompt vs null
inter-entites, balaye sur toutes les couches. Regle declaree : la
meilleure couche est choisie sur A, validee sur B.

    python probe_semantic_layers.py
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
from behavioral import (ALT, A_PREFIX, B_PREFIX, ENTS, VALS,   # noqa: E402
                        build_doc)
from probe_semantic_anchor import entity_ends                  # noqa: E402

STATE_TMP = os.path.join(HERE, ".layers_tmp_state")


def forwards_all_layers(s, ids):
    """(L+1, n, d) hiddens for every layer, windowed."""
    import torch
    tok, model = s.load_model()
    n = len(ids)
    x = torch.tensor(ids, device=s.device)
    out_layers = None
    a, W, S = 0, 1024, 512
    with torch.no_grad():
        while a < n:
            w = min(W, n - a)
            out = model(x[a:a + w].unsqueeze(0),
                        output_hidden_states=True)
            hs = [h[0].float().cpu().numpy() for h in out.hidden_states]
            if out_layers is None:
                out_layers = [np.empty((n, h.shape[-1]), np.float32)
                              for h in hs]
            lo = 0 if a == 0 else W - S
            for li, h in enumerate(hs):
                out_layers[li][a + lo:a + w] = h[lo:w]
            if a + w >= n:
                break
            a += S
    return out_layers


def wh(H):
    Z = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    Z = Z - Z.mean(axis=0, keepdims=True)
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
    print(f"dossier v1 : {len(ids)} tokens ; forwards toutes couches...",
          flush=True)
    Hdoc = forwards_all_layers(s, ids)
    L = len(Hdoc)
    print(f"{L} couches (embedding incluse), dim {Hdoc[-1].shape[-1]}",
          flush=True)

    ends_by = {e: entity_ends(tok, ids, [e]) for e, _ in stable}

    # prompt hiddens, all layers, per entity and prefix
    P = {}
    for e, _v in stable:
        for tag, tmpl in (("A", A_PREFIX), ("B", B_PREFIX)):
            pids = tok.encode(tmpl.format(e=e))
            Hp = forwards_all_layers(s, pids)
            pe = entity_ends(tok, pids, [e])[-1]
            P[(e, tag)] = [Hp[li][pe] for li in range(L)]
    print("prompts sondes faits", flush=True)

    # whitening stats per layer from the doc
    R = {"layers": []}
    for li in range(L):
        Z = wh(Hdoc[li])
        mean_ = (Hdoc[li] / (np.linalg.norm(Hdoc[li], axis=1,
                                            keepdims=True) + 1e-8)
                 ).mean(axis=0)
        sep = {}
        for tag in ("A", "B"):
            same, null = [], []
            for e, _v in stable:
                h = P[(e, tag)][li]
                z = h / (np.linalg.norm(h) + 1e-8) - mean_
                z = z / (np.linalg.norm(z) + 1e-8)
                same.append(float(np.max(Z[ends_by[e]] @ z)))
                others = [i for e2, _ in stable if e2 != e
                          for i in ends_by[e2]]
                null.append(float(np.max(Z[others] @ z)))
            sep[tag] = {"same": float(np.median(same)),
                        "null": float(np.median(null)),
                        "delta": float(np.median(same)
                                       - np.median(null))}
        R["layers"].append({"layer": li, "A": sep["A"], "B": sep["B"]})
        print(f"  couche {li:2d} : A same {sep['A']['same']:.3f} null "
              f"{sep['A']['null']:.3f} delta {sep['A']['delta']:+.3f} | "
              f"B delta {sep['B']['delta']:+.3f}", flush=True)

    best = max(R["layers"][1:], key=lambda r: r["A"]["delta"])
    R["best_by_A"] = best
    print(f"\nmeilleure couche (choisie sur A) : {best['layer']} "
          f"(delta A {best['A']['delta']:+.3f}) ; validation B : "
          f"delta {best['B']['delta']:+.3f}", flush=True)
    out = os.path.join(HERE, "results", "semantic_layers_qwen.json")
    json.dump(R, open(out, "w"), indent=1)
    print(f"saved -> {out}")
    shutil.rmtree(STATE_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
