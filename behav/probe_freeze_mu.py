"""Freezing the centre during generation: what it fixes, what it costs.

`sem_key` folded every state it saw into the running centre `mu`,
including the ones produced while ANSWERING -- although `complete`'s
docstring promises it writes nothing. Measured consequence: 182 tokens
of unrelated prose shift the centre enough to turn a recall the cold
store holds in full (`Brindas Kolvec`) into a fabrication (`Brigitte
Lefevre`), and the semantic tier is what carried that recall in the
first place (with the tier off it is 7/8 either way).

The fix is `learn=False` at generation. This probe measures both sides.

Registered BEFORE the run:

  W1  OLD behaviour degrades with use: after ~240 generated tokens on
      unrelated prompts, recall of the eight facts is strictly lower
      than on the same state before generating.
      FALSIFIED if it does not drop.
  W2  NEW behaviour does not: recall is identical before and after the
      same 240 tokens, and mu_n does not move.
      FALSIFIED if either changes.
  W3  On a CLEAN state the two behaviours may still differ, because the
      old code folded a state into the centre BEFORE keying on it. This
      is the cost side and it is recorded, not predicted: any answer
      that changes has to be justified before shipping.
"""
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sillage.core as core                          # noqa: E402
from sillage.runtime import Sillage                  # noqa: E402
from probe_readout_dial import DOC, ANSWERABLE       # noqa: E402
from probe_reflow import reflow                      # noqa: E402

FIXED = core.SillageMemory.sem_key


def always_learn(self, h, learn=True):
    """The pre-fix behaviour: every state seen moves the centre."""
    return FIXED(self, h, learn=True)


DRIFT = [
    "La cuisson du pain au levain demande",
    "Le petrissage se fait en deux temps, avec",
    "La farine complete boit davantage d'eau que",
    "Un levain jeune double de volume en",
    "La vapeur pendant les vingt premieres minutes",
    "Le gluten se detend pendant le repos de",
    "A vingt-quatre degres la fermentation est",
    "Il faut augmenter l'hydratation d'environ",
    "La croute se colore quand le four est",
    "Une cuisine fraiche d'hiver ralentit",
    "Le levain se rafraichit tous les",
    "La mie devient plus dense si l'on",
    "Le four doit etre prechauffe pendant",
    "On enfourne la pate quand elle a",
    "La temperature de la piece gouverne",
    "Le sel se met apres l'autolyse pour",
    "Une pate trop hydratee devient",
    "Le faconnage demande une surface",
    "La pousse en banneton dure environ",
    "On scarifie la pate juste avant",
]


def recall(s):
    hits, detail = 0, []
    for prompt, want in ANSWERABLE:
        txt = s.complete(prompt, n=12, temp=0.0)
        ok = want.lower() in txt.lower()
        hits += ok
        detail.append({"want": want, "ok": bool(ok), "got": txt.strip()[:40]})
    return hits, detail


def trial(label, patched):
    core.SillageMemory.sem_key = always_learn if patched else FIXED
    tmp = tempfile.mkdtemp(prefix="freeze_")
    try:
        s = Sillage(model="qwen", state=tmp, quiet=True)
        for _ in range(2):
            s.read_text(reflow(DOC))
        mu0 = s.mem.mu_n
        before, d_before = recall(s)
        mu1 = s.mem.mu_n
        gen = 0
        for p in DRIFT:
            s.complete(p, n=12, temp=0.0)
            gen += 12
        mu2 = s.mem.mu_n
        after, d_after = recall(s)
        print(f"  {label:<26} mu_n {mu0} -> {mu1} -> {mu2} (after "
              f"{gen} generated tokens)   recall {before}/8 -> {after}/8",
              flush=True)
        for a, b in zip(d_before, d_after):
            if a["ok"] != b["ok"]:
                print(f"      {a['want']}: {a['ok']} -> {b['ok']}  "
                      f"{b['got']!r}", flush=True)
        return {"mu_n": [mu0, mu1, mu2], "before": before, "after": after,
                "detail_before": d_before, "detail_after": d_after}
    finally:
        core.SillageMemory.sem_key = FIXED
        shutil.rmtree(tmp, ignore_errors=True)


print("read the reflowed report, answer, generate on unrelated prompts, "
      "answer again:")
res = {"old": trial("OLD (centre learns)", True),
       "new": trial("NEW (centre frozen)", False)}

o, n = res["old"], res["new"]
same_clean = [a["got"] == b["got"] for a, b in
              zip(o["detail_before"], n["detail_before"])]
res["verdict"] = {
    "W1": {"old_before": o["before"], "old_after": o["after"],
           "holds": o["after"] < o["before"]},
    "W2": {"new_before": n["before"], "new_after": n["after"],
           "mu_frozen": n["mu_n"][1] == n["mu_n"][2],
           "holds": n["before"] == n["after"] and n["mu_n"][1] == n["mu_n"][2]},
    "W3": {"identical_answers_on_clean_state": sum(same_clean),
           "of": len(same_clean),
           "changed": [a["want"] for a, b, s_ in
                       zip(o["detail_before"], n["detail_before"], same_clean)
                       if not s_]}}
print("\n" + json.dumps(res["verdict"], indent=1, ensure_ascii=False))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "freeze_mu.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with io.open(out, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1, ensure_ascii=False)
print(f"written {out}")
