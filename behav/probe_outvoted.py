"""When recall fails, does the memory not HAVE the fact, or is it outvoted?

`complete` misses `madame Brindas Kolvec` with the memory moving zero
tokens, at both capacities and under both readouts. Two very different
causes look identical from outside:

  (a) the tiers never stored that continuation -- a retrieval failure,
  (b) they stored it, and the frozen model's own confidence outvotes
      them in the mixture -- an ARBITRATION failure.

They call for opposite fixes, so this probe opens the box. For each
fact it reports whether the exact 4-gram is in the cold store, what the
store says the next token is, what the frozen model wants instead, and
how the three mixing weights resolve it.

Registered BEFORE the run:

  R1  At least one MISSED fact is present in the cold store with the
      correct successor -- i.e. some failures are arbitration, not
      retrieval.
      FALSIFIED if every missed fact is absent from the store.
  R2  Where the memory is outvoted, the frozen model's probability on
      its own (wrong) token exceeds LAM_C / (1 - LAM_C) times the
      store's mass on the right one -- the loss is arithmetic, not
      mysterious.
  R3  Raising LAM_C alone (0.3 -> 0.6) converts at least one missed
      fact, WITHOUT changing any fact already recalled.
      FALSIFIED if a recalled fact regresses.

Run:  python behav/probe_outvoted.py [--target HUB]
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

import sillage.core as core                    # noqa: E402
from sillage.runtime import Sillage            # noqa: E402
from probe_readout_dial import DOC, ANSWERABLE  # noqa: E402


def inspect(s, prompt):
    """One decoding step, opened up: what each source wants, and who wins."""
    import torch
    tok, model = s.load_model()
    mem = s.mem
    ids = tok.encode(prompt)
    mem.new_stream()
    for t in ids[:-1]:
        mem.step_key(int(t))
    thrG, thrS = mem.thresholds()
    need_h = mem.semantic or mem.fastweights
    with torch.no_grad():
        out = model(torch.tensor(ids, device=s.device).unsqueeze(0),
                    output_hidden_states=need_h)
    lb = out.logits[0, -1].float().cpu().numpy()
    mem.set_vocab(lb.shape[-1])
    h = (out.hidden_states[-1][0, -1].float().cpu().numpy()
         if need_h else None)
    la, _ = mem.adapt(lb, h)
    p_base = np.exp(la - la.max())
    p_base /= p_base.sum()
    qG = mem.step_key(int(ids[-1]))
    _, sG = mem.scores(mem.M, qG)
    sS = None
    if mem.semantic:
        _, sS = mem.scores(mem.MS, mem.sem_key(h))
    pc = mem.cold_lookup()
    p = mem.mix_full(p_base, sG, sS, pc, thrG, thrS)
    base_top = int(np.argmax(p_base))
    final_top = int(np.argmax(p))
    cold = None
    if pc:
        ct = max(pc, key=pc.get)
        cold = {"token": tok.decode([ct]), "id": ct, "mass": float(pc[ct]),
                "n": len(pc)}
    return {"base_top": tok.decode([base_top]),
            "base_p": float(p_base[base_top]),
            "final_top": tok.decode([final_top]),
            "final_p": float(p[final_top]),
            "moved": base_top != final_top,
            "cold": cold, "tiers": list(mem.last_src)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("--lam-c", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="outvoted_")
    res = {"target": a.target, "lam_c_default": core.LAM_C,
           "lam_c_raised": a.lam_c, "facts": []}
    try:
        s = Sillage(model="qwen", state=tmp, target=a.target, quiet=True)
        for _ in range(2):
            rec = s.read_text(DOC)
        print(f"state: {rec['tokens']} tokens, {len(s.mem.cold)} grams\n",
              flush=True)

        for prompt, want in ANSWERABLE:
            d = inspect(s, prompt)
            txt = s.complete(prompt, n=10, temp=0.0)
            d["want"] = want
            d["recalled"] = want.lower() in txt.lower()
            d["got"] = txt.strip()[:40]
            res["facts"].append(d)
            c = d["cold"]
            print(f"  {'OK ' if d['recalled'] else 'MISS'} {want:<9} "
                  f"model wants {d['base_top']!r:<12} p={d['base_p']:.3f}"
                  f" | cold "
                  + (f"{c['token']!r:<12} mass={c['mass']:.3f}" if c
                     else "ABSENT           ")
                  + f" | wins {d['final_top']!r}", flush=True)

        missed_with_cold = [f for f in res["facts"]
                            if not f["recalled"] and f["cold"]]
        res["R1"] = {"missed": sum(1 for f in res["facts"]
                                   if not f["recalled"]),
                     "missed_but_in_cold": len(missed_with_cold)}
        print(f"\nR1: {res['R1']}", flush=True)

        # R3: raise the cold weight alone and rerun every fact
        core.LAM_C = a.lam_c
        after = []
        for prompt, want in ANSWERABLE:
            txt = s.complete(prompt, n=10, temp=0.0)
            after.append({"want": want,
                          "recalled": want.lower() in txt.lower(),
                          "got": txt.strip()[:40]})
        core.LAM_C = res["lam_c_default"]
        res["raised"] = after
        before = [f["recalled"] for f in res["facts"]]
        now = [f["recalled"] for f in after]
        gained = [ANSWERABLE[i][1] for i in range(len(now))
                  if now[i] and not before[i]]
        lost = [ANSWERABLE[i][1] for i in range(len(now))
                if before[i] and not now[i]]
        res["R3"] = {"before": sum(before), "after": sum(now),
                     "gained": gained, "lost": lost}
        print(f"R3 (LAM_C {res['lam_c_default']} -> {a.lam_c}): "
              f"{sum(before)}/8 -> {sum(now)}/8  gained {gained}  "
              f"lost {lost}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results", "outvoted.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
