"""Which surprise should gate a memory's writes?

Sillage writes with g = clip(-ln p, 0, CAP): the frozen model's SHANNON
surprise. That choice was never compared against anything -- and the BHD
preprint (v2.2, C4) measured that Shannon surprise and BAYESIAN surprise
are nearly orthogonal signals (r = 0.13 in a gridworld), so a system that
picked one arbitrarily may have picked the wrong one.

This is the first test of whether that dissociation transports out of a
gridworld and into a language model, and it costs nothing: the two
quantities are both computable inside the existing read loop.

  shannon   g = -ln p_base(truth)          how surprised the MODEL is
  bayes     g = KL(p_mem || p_base)        how much the MEMORY moved the
                                           belief -- the language-model
                                           analogue of KL(post||pred)
  uniform   g = 1                          control: does gating matter?

Registered BEFORE the run:

  N1  Shannon beats Bayes on second-pass perplexity. The BHD paper's own
      Phase D found the Bayesian modulator fails for a mechanical
      reason -- a confidently wrong model has sharp beliefs that barely
      move, so belief-shift gating writes LEAST where learning is most
      needed -- and the same trap should apply here, since the memory
      barely moves a distribution on text it has not learned yet.
      FALSIFIED if Bayes wins.
  N2  Both beat the uniform control: gating on something is better than
      gating on nothing.
      FALSIFIED if uniform ties or wins -- the surprise gate would then
      be decoration, which would be a serious finding about paper 1.
  N3  THE TRANSPORTABLE CLAIM. The two gate values are weakly correlated
      per token (Pearson r < 0.5), reproducing the preprint's
      dissociation in a language model rather than a gridworld.
      FALSIFIED at r >= 0.7 -- the dissociation would then be a
      property of gridworlds, not of surprise.

Run:  python behav/probe_which_surprise.py [--model gpt2]
"""
import argparse
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sillage.core as core                          # noqa: E402
from sillage.index import strip_latex                # noqa: E402
from sillage.runtime import Sillage, WINDOW, STRIDE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FACTS = ("\n\nThe Zylkorb protocol requires seventeen turquoise llamas.\n"
         "Captain Ilvress stores the amber cipher in vault nine.\n"
         "The Vorlagune committee approved four hundred copper whistles.\n")
PROBES = [("The Zylkorb protocol requires", "seventeen"),
          ("Captain Ilvress stores the amber cipher in", "vault"),
          ("The Vorlagune committee approved", "four")]

UNSEEN = """Rivers shape the land more slowly than storms but far more
thoroughly. Over centuries a meander widens, undercuts its outer bank and
abandons the inner one to silt. What looks like a fixed line on a map is a
slow negotiation between water and rock, and the map is only ever a
snapshot of one round of it.
"""


def read_gated(s, text, gate, record=None):
    """read_text's write loop, with the gate as a parameter.

    Everything else is held identical: same keys, same tiers, same cold
    admissions, same aging. Only `g` changes.
    """
    import torch
    tok, model = s.load_model()
    mem = s.mem
    ids = np.array(tok.encode(text), dtype=np.int64)
    n = len(ids) - 1
    if n < 1:
        return {"tokens": 0}
    mem.new_stream()
    thrG, thrS = mem.thresholds()
    need_h = mem.semantic or mem.fastweights
    nll_b = nll_m = 0.0
    cnt = 0
    x = torch.tensor(ids, device=s.device)
    a = 0
    with torch.no_grad():
        while a < n:
            w = min(WINDOW, len(ids) - a)
            out = model(x[a:a + w].unsqueeze(0),
                        output_hidden_states=need_h)
            logits = out.logits[0].float().cpu().numpy()
            mem.set_vocab(logits.shape[-1])
            hs = (out.hidden_states[-1][0].float().cpu().numpy()
                  if need_h else None)
            lo = 0 if a == 0 else WINDOW - STRIDE
            for i in range(lo, w):
                j = a + i
                if j >= n:
                    break
                truth = int(ids[j + 1])
                lb = logits[i]
                mx = lb.max()
                lpb = lb - (mx + np.log(np.exp(lb - mx).sum()))
                lp = float(lpb[truth])
                la, phi = mem.adapt(lb, hs[i] if need_h else None)
                p_base = np.exp(la - la.max())
                p_base /= p_base.sum()
                qG = mem.step_key(int(ids[j]))
                uG, sG = mem.scores(mem.M, qG)
                mem.res_G.append(float(sG.max()))
                qS = uS = sS = None
                if mem.semantic:
                    qS = mem.sem_key(hs[i])
                    uS, sS = mem.scores(mem.MS, qS)
                    mem.res_S.append(float(sS.max()))

                # both gates, every token, whichever one is in force.
                # The Bayesian one needs the FULL mixed distribution,
                # which costs a vocabulary-sized pass, so it is computed
                # only when it is actually the gate or is being recorded.
                g_shannon = min(core.CAP, max(0.0, -lp))
                g_bayes = 0.0
                if gate == "bayes" or record is not None:
                    p_mem = mem.mix_full(p_base, sG, sS, mem.cold_lookup(),
                                         thrG, thrS)
                    p_mem = np.maximum(p_mem, 1e-30)
                    p_mem /= p_mem.sum()
                    kl = float(np.sum(p_mem * (np.log(p_mem)
                                               - np.log(np.maximum(
                                                   p_base, 1e-30)))))
                    g_bayes = min(core.CAP, max(0.0, kl))
                if record is not None:
                    record.append((g_shannon, g_bayes))

                pc = mem.cold_lookup(truth)
                p = mem.mix_true(float(p_base[truth]), sG, truth, sS, pc,
                                 thrG, thrS)
                nll_b += -lp
                nll_m += -np.log(max(p, 1e-30))
                cnt += 1
                g = {"shannon": g_shannon, "bayes": g_bayes,
                     "uniform": 1.0}[gate]
                mem.write_all(qG, uG, qS, uS, truth, g, phi, p_base)
            if a + w >= len(ids):
                break
            a += STRIDE
    return {"tokens": cnt,
            "ppl_frozen": float(np.exp(nll_b / cnt)),
            "ppl_with_memory": float(np.exp(nll_m / cnt))}


def recall(s):
    hit = []
    for prompt, want in PROBES:
        out = s.complete(prompt, n=8)
        hit.append(want.lower() in out.lower())
    return sum(hit) / len(hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    a = ap.parse_args()

    body = strip_latex(io.open(os.path.join(ROOT, "papers", "sillage",
                                            "sillage.tex"),
                               encoding="utf-8",
                               errors="replace").read())[:9000] + FACTS
    res = {"model": a.model, "gates": {}}
    rec_pairs = None

    for gate in ("shannon", "bayes", "uniform"):
        tmp = tempfile.mkdtemp(prefix="gate_")
        try:
            s = Sillage(model=a.model, state=tmp, quiet=True,
                        fastweights=False)
            s.load_model()
            record = [] if gate == "shannon" else None
            first = read_gated(s, body, gate, record)
            second = read_gated(s, body, gate)
            from probe_readout_dial import nll_nowrite
            b, m = nll_nowrite(s, UNSEEN)
            row = {"tokens": first["tokens"],
                   "ppl_first_pass": round(first["ppl_with_memory"], 2),
                   "ppl_frozen": round(first["ppl_frozen"], 2),
                   "ppl_second_pass": round(second["ppl_with_memory"], 2),
                   "gain": round(first["ppl_frozen"]
                                 / second["ppl_with_memory"], 2),
                   "recall": recall(s),
                   "cold_grams": len(s.mem.cold),
                   "locality_nats": round(float(np.log(m) - np.log(b)), 4)}
            res["gates"][gate] = row
            if record:
                rec_pairs = record
            print(f"  {gate:<9} 2nd-pass ppl {row['ppl_second_pass']:>6.2f} "
                  f"(frozen {row['ppl_frozen']:.2f}, x{row['gain']:.2f})  "
                  f"recall {row['recall']:.0%}  grams {row['cold_grams']}  "
                  f"locality {row['locality_nats']:+.4f}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    sh = np.array([p[0] for p in rec_pairs])
    by = np.array([p[1] for p in rec_pairs])
    r = float(np.corrcoef(sh, by)[0, 1])
    from scipy.stats import spearmanr
    rho = float(spearmanr(sh, by).correlation)
    res["dissociation"] = {"n": len(sh), "pearson": round(r, 3),
                           "spearman": round(rho, 3),
                           "shannon_mean": round(float(sh.mean()), 3),
                           "bayes_mean": round(float(by.mean()), 3),
                           "bayes_zero_share": round(float((by < 1e-6).mean()),
                                                     3)}
    g = res["gates"]
    res["verdict"] = {
        "N1_shannon_beats_bayes": g["shannon"]["ppl_second_pass"]
        < g["bayes"]["ppl_second_pass"],
        "N2_gating_beats_uniform":
            max(g["shannon"]["ppl_second_pass"],
                g["bayes"]["ppl_second_pass"])
            < g["uniform"]["ppl_second_pass"],
        "N3_dissociation_r": round(r, 3),
        "N3_holds": abs(r) < 0.7}
    print("\ndissociation of the two gates, per token:")
    print(json.dumps(res["dissociation"], indent=1))
    print(json.dumps(res["verdict"], indent=1))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "which_surprise.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
