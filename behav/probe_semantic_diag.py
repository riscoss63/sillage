"""Marche 1 de l'attaque du mur semantique : ou meurt le chemin
paraphrase ? Sonde READ-ONLY sur l'etat comportemental qwen.

Pour chaque fait stable, aux positions de valeur des prefixes A
(canonique) et B (paraphrase) : ce que dit le tier semantique (score de
la vraie valeur, max, rang, seuil d'abstention) et le tier n-gram en
controle. mu est snapshote/restaure (sem_key mute la moyenne courante).

    python probe_semantic_diag.py
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
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break
sys.path.insert(0, HERE)

from sillage import Sillage                                    # noqa: E402
from behavioral import ALT, A_PREFIX, B_PREFIX, ENTS, VALS     # noqa: E402

STATE = os.path.join(HERE, ".behav_state_qwen")


def main():
    import torch
    s = Sillage(model="qwen", state=STATE, quiet=True)
    tok, model = s.load_model()
    mem = s.mem
    thrG, thrS = mem.thresholds()
    print(f"seuils : n-gram {thrG:.3f} | semantique {thrS:.3f}",
          flush=True)

    facts = list(zip(ENTS[:30], VALS[:30]))
    changed = {e for e, _ in
               [(e, ALT[i]) for i, (e, _v) in enumerate(facts[:10])]}
    stable = [(e, v) for e, v in facts if e not in changed]

    mu0 = None if mem.mu is None else mem.mu.copy()
    mun0 = mem.mu_n

    R = {"thrG": float(thrG), "thrS": float(thrS), "rows": []}
    for e, v in stable:
        row = {"e": e, "v": v}
        vid = tok.encode(" " + v)[0]      # first token of the value
        for tag, tmpl in (("A", A_PREFIX), ("B", B_PREFIX)):
            prompt = tmpl.format(e=e)
            ids = tok.encode(prompt)
            with torch.no_grad():
                out = model(torch.tensor([ids], device=s.device),
                            output_hidden_states=True)
            h = out.hidden_states[-1][0, -1].float().cpu().numpy()
            # --- semantic tier, read-only (snapshot/restore mu) ------
            mem.mu = None if mu0 is None else mu0.copy()
            mem.mu_n = mun0
            qS = mem.sem_key(h)
            _, sS = mem.scores(mem.MS, qS)
            # --- n-gram tier control (key from the prompt tokens) ----
            mem.new_stream()
            for t in ids:
                qG = mem.step_key(int(t))
            _, sG = mem.scores(mem.M, qG)
            rankS = int((sS > sS[vid]).sum()) + 1
            rankG = int((sG > sG[vid]).sum()) + 1
            row[tag] = {
                "sS_val": float(sS[vid]), "sS_max": float(sS.max()),
                "rankS": rankS, "clearS": bool(sS.max() >= thrS),
                "topS_is_val": bool(rankS == 1),
                "sG_val": float(sG[vid]), "sG_max": float(sG.max()),
                "rankG": rankG, "clearG": bool(sG.max() >= thrG),
                "topG_is_val": bool(rankG == 1),
            }
        loss = (0.0 if row["A"]["sS_val"] <= 0 else
                1.0 - row["B"]["sS_val"] / row["A"]["sS_val"])
        row["sS_val_loss_A_to_B"] = float(loss)
        R["rows"].append(row)
        print(f"  {e:12s} A: semQ rang {row['A']['rankS']:>6d} "
              f"(clear {int(row['A']['clearS'])}) ngram rang "
              f"{row['A']['rankG']:>6d} | B: semQ rang "
              f"{row['B']['rankS']:>6d} (clear {int(row['B']['clearS'])})"
              f" ngram rang {row['B']['rankG']:>6d}", flush=True)

    mem.mu = mu0
    mem.mu_n = mun0

    n = len(R["rows"])
    agg = {
        "n": n,
        "A_top1_sem": sum(r["A"]["topS_is_val"] for r in R["rows"]) / n,
        "B_top1_sem": sum(r["B"]["topS_is_val"] for r in R["rows"]) / n,
        "A_clear_sem": sum(r["A"]["clearS"] for r in R["rows"]) / n,
        "B_clear_sem": sum(r["B"]["clearS"] for r in R["rows"]) / n,
        "A_top1_ngram": sum(r["A"]["topG_is_val"] for r in R["rows"]) / n,
        "B_top1_ngram": sum(r["B"]["topG_is_val"] for r in R["rows"]) / n,
        "median_sS_val_loss": float(np.median(
            [r["sS_val_loss_A_to_B"] for r in R["rows"]])),
        "median_rankS_A": float(np.median(
            [r["A"]["rankS"] for r in R["rows"]])),
        "median_rankS_B": float(np.median(
            [r["B"]["rankS"] for r in R["rows"]])),
    }
    R["agg"] = agg
    print("\n== agregats (20 faits stables) ==")
    for k, val in agg.items():
        print(f"  {k:22s}: "
              + (f"{val:.0%}" if isinstance(val, float) and val <= 1
                 and "median" not in k and k != "n" else f"{val}"))
    out = os.path.join(HERE, "results", "semantic_diag_qwen.json")
    json.dump(R, open(out, "w"), indent=1)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
