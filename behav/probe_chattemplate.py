"""Rephrased questions were tested through the wrong door.

`probe_bridge` fed bare questions to `complete`, which encodes the
prompt RAW. A base language model handed a question in completion mode
continues the genre: it writes more questions. The 1.7B did it eight
times out of eight ("Qui a redige ce compte rendu ? Qui a redige ce
compte..."), which is why it scored 0/8 -- nothing to do with capacity,
and nothing to do with the memory.

`chat` and `serve` apply the model's chat template. This runs the same
eight rephrased questions through it, at both capacities, to find out
what the memory can do when the model is actually being asked.

There is a structural reason to expect trouble, registered here before
the run: the template ends every prompt with the SAME assistant header,
so the n-gram key at the first generated token is identical for every
question. That tier cannot discriminate at position one; only the
semantic tier (keyed on a hidden state that does encode the question)
and later positions can.

Registered BEFORE the run:

  Q1  The template stops the question-echo: fewer outputs contain a
      question mark than without it, at both capacities.
      FALSIFIED if the echo persists.
  Q2  Rephrased recall improves at 1.7B, which scored 0/8 raw.
      FALSIFIED if it stays at 0/8.
  Q3  The memory contributes LESS at the first token under the
      template, for the structural reason above. Recorded as the mean
      `moved` with and without.
  Q4  Intrusion on unanswerable questions does not get worse under the
      template. Recorded.
"""
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sillage.runtime import Sillage                      # noqa: E402
from sillage.cli import FAMILY_READOUT, FAINT            # noqa: E402
from probe_abstain_gen import (DOC, REWORDED,            # noqa: E402
                               UNANSWERABLE, N)

MODELS = [(None, "0.6B"), ("Qwen/Qwen3-1.7B", "1.7B")]
# the unanswerable prompts are sentence openings, not questions -- turn
# them into questions so the template has something to answer
UNANSWERABLE_Q = [
    "Combien de ruches compte le rucher de Peyrelonge au total ?",
    "Quel est le prix du kilo de miel de colza a la cooperative ?",
    "Quand aura lieu la prochaine visite du rucher de Peyrelonge ?",
    "Quel modele de ruches est utilise au rucher de Peyrelonge ?",
    "Quelle a ete la production totale du rucher l'an dernier ?",
    "Dans quelle commune se trouve le rucher de Peyrelonge ?",
    "Quel age a la reine de la ruche numero 7 ?",
    "Quel traitement anti-varroa est retenu pour l'automne ?",
]


def templated(tok, q):
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
    except TypeError:                       # older tokenizers
        return tok.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True, tokenize=False)


def run(s, tok, items, use_template, label):
    rows = []
    for q, want in items:
        prompt = templated(tok, q) if use_template else q
        out = s.complete(prompt, n=N, temp=0.0)
        at = s.attribution() or {}
        rows.append({"q": q[:44], "want": want,
                     "correct": bool(want and want.lower() in out.lower()),
                     "asks_back": "?" in out,
                     "moved": at.get("moved") or 0,
                     "got": out.strip()[:60]})
    ok = sum(r["correct"] for r in rows if r["want"])
    echo = sum(r["asks_back"] for r in rows)
    mv = sum(r["moved"] for r in rows) / max(1, len(rows))
    intrude = sum(1 for r in rows if not r["want"]
                  and r["moved"] >= FAINT)
    print(f"    {label:<26} correct {ok}/8  echoes-a-question {echo}/8  "
          f"mean moved {mv:.1f}" + (f"  intrudes {intrude}/8"
                                    if not items[0][1] else ""), flush=True)
    return {"rows": rows, "correct": ok, "echo": echo, "mean_moved": mv,
            "intrudes": intrude}


def main():
    res = {}
    text = Sillage.reflow(DOC)
    rew = list(REWORDED)
    una = [(q, None) for q in UNANSWERABLE_Q]
    for hub, cap in MODELS:
        tmp = tempfile.mkdtemp(prefix="chattpl_")
        try:
            print(f"\n=== {cap} ===", flush=True)
            s = Sillage(model="qwen", state=tmp, target=hub, quiet=True)
            for _ in range(2):
                s.read_text(text)
            tok = s.load_tokenizer()
            s.mem.beta_G, s.mem.lam_G, s.mem.thr_qG = FAMILY_READOUT
            for tpl in (False, True):
                tag = "chat template" if tpl else "raw prompt"
                res[f"{cap} {tag} reworded"] = run(
                    s, tok, rew, tpl, f"{tag}, rephrased")
                res[f"{cap} {tag} unanswerable"] = run(
                    s, tok, una, tpl, f"{tag}, unanswerable")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    v = {"Q1_echo": {k: res[k]["echo"] for k in res if "reworded" in k},
         "Q2_correct": {k: res[k]["correct"] for k in res if "reworded" in k},
         "Q3_mean_moved": {k: round(res[k]["mean_moved"], 1) for k in res},
         "Q4_intrusion": {k: res[k]["intrudes"] for k in res
                          if "unanswerable" in k}}
    print("\n" + json.dumps(v, indent=1, ensure_ascii=False))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "chattemplate.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump({"arms": res, "verdict": v}, fh, indent=1,
                  ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
