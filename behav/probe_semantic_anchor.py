"""Marche 2 : des cles semantiques ANCREES adressent-elles les faits ?

Prototype hors-outil : un tier M_E jetable est construit sur le dossier
v1 (qwen, mu FIGE precalcule en deux passes). La cle de la position t
est le SimHash du hidden de l'ANCRE a_t ; la valeur ecrite est le token
t+1, la porte g_t inchangee. Deux regles d'ancrage :

  oracle    a_t = derniere occurrence d'un token d'entite (bornes
            hautes : les positions sont connues de l'instrument)
  surprise  a_t = dernier token dont la porte g >= 2.5 nats (les
            entites inventees SONT les tokens surprenants -- le signal
            gratuit choisit les ancres)

Sondes A (canonique) et B (paraphrase) : la requete est la cle du
hidden d'ancrage du prompt (meme regle), le score est le rang de la
vraie valeur dans M_E. Baseline mesuree en marche 1 : hasard.

    python probe_semantic_anchor.py
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

G_ANCHOR = 2.5
STATE_TMP = os.path.join(HERE, ".anchor_tmp_state")


def forwards(s, ids, need_lp=True):
    """Windowed teacher-forced pass: hiddens (n,1024) and gates (n,)."""
    import torch
    tok, model = s.load_model()
    n = len(ids)
    H = np.empty((n, 1024), np.float32)
    G = np.zeros(n, np.float32)
    x = torch.tensor(ids, device=s.device)
    a, W, S = 0, 1024, 512
    with torch.no_grad():
        while a < n:
            w = min(W, n - a)
            out = model(x[a:a + w].unsqueeze(0),
                        output_hidden_states=True)
            hs = out.hidden_states[-1][0].float().cpu().numpy()
            lo = 0 if a == 0 else W - S
            H[a + lo:a + w] = hs[lo:w]
            if need_lp:
                lg = out.logits[0].float()
                hi = min(w, n - a - 1)
                if hi > lo:
                    import torch as _t
                    tr = x[a + lo + 1:a + hi + 1]
                    lp = (_t.log_softmax(lg[lo:hi], dim=-1)
                          .gather(1, tr.unsqueeze(1))[:, 0]
                          .cpu().numpy())
                    G[a + lo + 1:a + hi + 1] = np.clip(-lp, 0.0, CAP)
            if a + w >= n:
                break
            a += S
    return H, G


def entity_ends(tok, ids, ents):
    """Position of the LAST token of every entity occurrence."""
    ends = []
    arr = list(ids)
    for e in ents:
        pat = tok.encode(" " + e)
        L = len(pat)
        for i in range(len(arr) - L + 1):
            if arr[i:i + L] == pat:
                ends.append(i + L - 1)
    return sorted(set(ends))


def anchors_from(points, n):
    """a_t = most recent point <= t (or -1 before the first)."""
    a = np.full(n, -1, np.int64)
    cur = -1
    pts = set(points)
    for t in range(n):
        if t in pts:
            cur = t
        a[t] = cur
    return a


def fixed_mu(H):
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    return Hn.mean(axis=0)


def make_tier(mu):
    m = SillageMemory(None, "qwen", semantic=True, fastweights=False)
    m.set_vocab(151936)
    m.mu = mu.copy()
    m.mu_n = 10 ** 9            # freeze the running mean
    return m


def build_ME(m, H, G, ids, anch):
    for t in range(len(ids) - 1):
        a = int(anch[t])
        if a < 0:
            continue
        qS = m.sem_key(H[a])
        uS, _ = m.scores(m.MS, qS)
        m.amp_write(m.MS, qS, uS, int(ids[t + 1]), float(G[t + 1]))


def probe(m, s, tok, prompt, ent, vid, rule):
    import torch
    ids = tok.encode(prompt)
    H, G = forwards(s, ids)
    if rule == "oracle":
        ends = entity_ends(tok, ids, [ent])
        if not ends:
            return None
        a = ends[-1]
    else:
        cand = [t for t in range(len(ids)) if G[t] >= G_ANCHOR]
        a = cand[-1] if cand else len(ids) - 1
    qS = m.sem_key(H[a])
    _, sS = m.scores(m.MS, qS)
    rank = int((sS > sS[vid]).sum()) + 1
    return rank


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
    print(f"dossier v1 : {len(ids)} tokens ; passe 1 (hiddens+gates)...",
          flush=True)
    H, G = forwards(s, ids)
    mu = fixed_mu(H)

    ends = entity_ends(tok, ids, [e for e, _ in facts])
    surp = [t for t in range(len(ids)) if G[t] >= G_ANCHOR]
    print(f"ancres : {len(ends)} fins d'entite (oracle) ; {len(surp)} "
          f"tokens g>={G_ANCHOR} (regle) ; recouvrement "
          f"{len(set(ends) & set(surp))}", flush=True)

    R = {"doc_tokens": len(ids), "anchors_oracle": len(ends),
         "anchors_surprise": len(surp),
         "overlap": len(set(ends) & set(surp)), "rules": {}}
    for rule, points in (("oracle", ends), ("surprise", surp)):
        m = make_tier(mu)
        build_ME(m, H, G, ids, anchors_from(points, len(ids)))
        rows = []
        for e, v in stable:
            vid = tok.encode(" " + v)[0]
            rA = probe(m, s, tok, A_PREFIX.format(e=e), e, vid, rule)
            rB = probe(m, s, tok, B_PREFIX.format(e=e), e, vid, rule)
            rows.append({"e": e, "rank_A": rA, "rank_B": rB})
            print(f"  [{rule:8s}] {e:12s} A rang {rA:>6} | B rang "
                  f"{rB:>6}", flush=True)
        n = len(rows)
        agg = {
            "A_top1": sum(r["rank_A"] == 1 for r in rows) / n,
            "A_top10": sum(r["rank_A"] <= 10 for r in rows) / n,
            "B_top1": sum(r["rank_B"] == 1 for r in rows) / n,
            "B_top10": sum(r["rank_B"] <= 10 for r in rows) / n,
            "median_rank_A": float(np.median([r["rank_A"]
                                              for r in rows])),
            "median_rank_B": float(np.median([r["rank_B"]
                                              for r in rows])),
        }
        R["rules"][rule] = {"agg": agg, "rows": rows}
        print(f"== {rule} == A top10 {agg['A_top10']:.0%} (median "
              f"{agg['median_rank_A']:.0f}) | B top10 "
              f"{agg['B_top10']:.0%} (median {agg['median_rank_B']:.0f})",
              flush=True)

    out = os.path.join(HERE, "results", "semantic_anchor_qwen.json")
    json.dump(R, open(out, "w"), indent=1)
    print(f"\nsaved -> {out}")
    shutil.rmtree(STATE_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
