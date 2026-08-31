"""Sillage's memory against the drafters that already ship.

The direction only survives if the memory beats what people already
have. Two incumbents:

  prompt lookup   vLLM's n-gram speculative decoding: match the last
                  NGRAM tokens against THE CURRENT PROMPT and propose
                  what followed. Free, stateless, no corpus.
  static cache    llama.cpp's persistent lookup cache: an n-gram map
                  built from a corpus and saved to disk. Raw counts, no
                  admission rule, no bound, no eviction.

What Sillage adds to the second is exactly three things -- admission by
the two-occurrence rule, a bounded store, and eviction by surprise mass
-- so this measures whether those three buy anything.

THE TARGET IS THE PLAIN MODEL. To sell speed and nothing else, the
memory must draft only: the frozen model verifies every token, so the
output is identical to what the user would have got with no memory at
all. Zero behaviour change, zero risk, and all four drafters are then
aiming at the same sequence, which is the only way the comparison is
fair.

Accounting is the standard one: a round drafts K tokens, spends ONE
verifying forward, and yields A+1 tokens where A is the number of
leading drafts that match. tokens-per-forward is therefore mean(A+1),
and a drafter that never guesses right scores exactly 1.00 -- the
no-drafter baseline.

Registered BEFORE the run:

  M1  Sillage beats prompt lookup on same-topic text (never read), on
      tokens-per-forward. FALSIFIED if prompt lookup ties or wins --
      a persistent memory would then be buying nothing over a
      stateless trick.
  M2  Sillage beats the static cache built from the SAME corpus.
      FALSIFIED if it ties or loses: the admission rule, the bound and
      the surprise eviction would then be decoration.
  M3  On unrelated text every drafter degrades to ~1.00 and none of
      them is harmful. Recorded.
  M4  The margin over the better incumbent is at least 10% of
      tokens-per-forward on same-topic text -- below that it is not
      worth anyone changing their stack for.
"""
import io
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sillage.core as core                          # noqa: E402
from sillage.index import strip_latex                # noqa: E402
from sillage.runtime import Sillage                  # noqa: E402
from probe_drafter_real import UNRELATED, paper, prompts_from  # noqa: E402

K_DRAFT = 8
N_GEN = 32


def greedy(s, prompt, n):
    """The plain frozen model's own continuation -- the target."""
    import torch
    tok, model = s.load_model()
    ids = tok.encode(prompt)
    out = []
    inp = torch.tensor(ids, device=s.device).unsqueeze(0)
    past = None
    with torch.no_grad():
        for _ in range(n):
            o = model(inp, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            out.append(nxt)
            inp = torch.tensor([[nxt]], device=s.device)
    return ids, out


def build_static(tok, corpus):
    """llama.cpp's lookup cache, in essence: raw counts, no rules."""
    ids = tok.encode(corpus)
    m = defaultdict(Counter)
    for i in range(len(ids) - core.NGRAM):
        m[tuple(ids[i:i + core.NGRAM])][ids[i + core.NGRAM]] += 1
    return {g: c.most_common(1)[0][0] for g, c in m.items()}


def draft_pld(hist, _mem, _static, k=K_DRAFT):
    """Prompt lookup: the last NGRAM tokens, found EARLIER in this same
    sequence, and whatever followed there."""
    if len(hist) < core.NGRAM + 1:
        return []
    key = hist[-core.NGRAM:]
    for a in range(len(hist) - core.NGRAM - 1, -1, -1):
        if hist[a:a + core.NGRAM] == key:
            return hist[a + core.NGRAM:a + core.NGRAM + k]
    return []


def draft_static(hist, _mem, static, k=K_DRAFT):
    out, h = [], list(hist)
    for _ in range(k):
        nxt = static.get(tuple(h[-core.NGRAM:]))
        if nxt is None:
            break
        out.append(nxt)
        h.append(nxt)
    return out


def draft_sillage(hist, mem, _static, k=K_DRAFT):
    out, h = [], list(hist)
    for _ in range(k):
        gram = np.array(h[-core.NGRAM:], dtype=np.int32).tobytes()
        slot = mem.cold.get(gram)
        if slot is None or sum(slot[1].values()) < core.COLD_MIN_COUNT:
            break
        nxt = max(slot[1], key=slot[1].get)
        out.append(nxt)
        h.append(nxt)
    return out


DRAFTERS = [("no drafter", lambda *a, **kw: []),
            ("prompt lookup", draft_pld),
            ("static cache", draft_static),
            ("sillage", draft_sillage)]


def score(fn, prompt_ids, target, mem, static):
    """Rounds of block-verified speculative decoding over a fixed target."""
    hist = list(prompt_ids)
    i = rounds = 0
    accepted = proposed = 0
    while i < len(target):
        d = fn(hist, mem, static)
        a = 0
        for j, t in enumerate(d):
            if i + j < len(target) and t == target[i + j]:
                a += 1
            else:
                break
        proposed += len(d)
        accepted += a
        take = target[i:i + a + 1]           # a drafts, plus the verified one
        hist.extend(take)
        i += len(take)
        rounds += 1
    return {"tokens": len(target), "rounds": rounds,
            "tok_per_fwd": len(target) / max(1, rounds),
            "accepted": accepted, "proposed": proposed}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="vslookup_")
    res = {"k_draft": K_DRAFT, "n_gen": N_GEN, "model": a.model}
    try:
        s = Sillage(model=a.model, state=tmp, quiet=True)
        s.load_model()
        tok = s.load_tokenizer()
        corpus = paper("sillage") + "\n\n" + paper("fastweights")
        print("reading two papers ...", flush=True)
        from sillage.ingest import ingest_text
        # exact by construction for the cold store (paper 7), and the
        # only way a 13k-token corpus fits in a probe's budget on CPU
        rec = ingest_text(s, corpus, "read", quiet=True)
        static = build_static(tok, corpus)
        print(f"  sillage: {len(s.mem.cold)} grams | static cache: "
              f"{len(static)} grams (no admission rule, no bound)",
              flush=True)
        res["state"] = {"tokens": rec["tokens"],
                        "sillage_grams": len(s.mem.cold),
                        "static_grams": len(static)}

        sets = {"seen": prompts_from(corpus, tok, k=6),
                "same-topic": prompts_from(paper("behavior"), tok, k=6),
                "unrelated": prompts_from(UNRELATED, tok, k=4)}

        for regime, prompts in sets.items():
            print(f"\n{regime}", flush=True)
            targets = [greedy(s, p, N_GEN) for p in prompts]
            res[regime] = {}
            for name, fn in DRAFTERS:
                tot = Counter()
                for pids, tgt in targets:
                    r = score(fn, pids, tgt, s.mem, static)
                    for key in ("tokens", "rounds", "accepted", "proposed"):
                        tot[key] += r[key]
                row = {"tok_per_fwd": round(tot["tokens"] / tot["rounds"], 3),
                       "acceptance": round(tot["accepted"]
                                           / max(1, tot["proposed"]), 3),
                       "proposed": tot["proposed"]}
                res[regime][name] = row
                print(f"  {name:<16} {row['tok_per_fwd']:>6.2f} tok/fwd   "
                      f"acceptance {row['acceptance']:>5.0%}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    warm = res["same-topic"]
    best_rival = max(warm["prompt lookup"]["tok_per_fwd"],
                     warm["static cache"]["tok_per_fwd"])
    mine = warm["sillage"]["tok_per_fwd"]
    res["verdict"] = {
        "M1_vs_prompt_lookup": [mine, warm["prompt lookup"]["tok_per_fwd"]],
        "M1_holds": mine > warm["prompt lookup"]["tok_per_fwd"],
        "M2_vs_static_cache": [mine, warm["static cache"]["tok_per_fwd"]],
        "M2_holds": mine > warm["static cache"]["tok_per_fwd"],
        "M3_unrelated": {k: res["unrelated"][k]["tok_per_fwd"]
                         for k, _ in DRAFTERS},
        "M4_margin": round(mine / max(1e-9, best_rival) - 1, 3),
        "M4_holds": mine / max(1e-9, best_rival) >= 1.10}
    print("\n" + json.dumps(res["verdict"], indent=1, ensure_ascii=False))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", f"vs_lookup_{a.model}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
