"""How much can a fixed-size memory hold before it stops recalling?

This is the question every user asks first and the one the project
cannot answer: "7 MB forever" is the selling point, and there is no
number saying forever *of what*. The pieces exist -- the matrix erodes
even without decay (90 -> 57 -> 17% under interference), the cold store
is capped at COLD_MAX grams and evicts by surprise mass, long retention
is carried by consolidation rather than by the matrix -- but nobody has
drawn the curve.

The tiers have different capacity laws and the behavioural ceiling is
their composition:

  M_G     4096x256, superposed. Signal per pair is fixed, noise from N
          others grows as sqrt(N), so recall should decay smoothly.
  cold    exact 4-gram -> successor, hard cap COLD_MAX, evicted by
          SURPRISE MASS -- so it should forget the banal before the
          remarkable.
  M_S     banded SimHash, D_S wide.
  adapter rank 16, measured to hold nothing durable.

Facts are planted exactly TWICE (paper 6's two-occurrence rule: the
minimum that enters the cold store at all), in cohorts at known depths,
and probed as the corpus grows.

Registered BEFORE the run:

  C1  The cold store grows roughly linearly with novel tokens and
      saturates at COLD_MAX. FALSIFIED if it plateaus below 60% of
      COLD_MAX, or exceeds it.
  C2  The OLDEST cohort holds (>= 80% of its recall at the first
      checkpoint) until the cold store saturates, and declines after.
      FALSIFIED if it declines materially BEFORE saturation -- that
      would mean the matrix, not the cap, sets the ceiling.
  C3  The NEWEST cohort stays >= 80% at every scale: writing more never
      stops the memory taking new facts.
      FALSIFIED if recent recall falls with corpus size.
  C4  Eviction is surprise-ranked, so planted facts SURVIVE IN THE
      STORE while ordinary grams are dropped: once saturated, at least
      80% of every planted fact's gram is still present, and the store's
      median surprise mass RISES.
      FALSIFIED if planted grams are evicted at the same rate as the
      rest -- "it forgets the boring parts first" would be a story, not
      a mechanism.
      (Measured in the store, not behaviourally: on an EMPTY memory the
      filler continuations score 100% because GPT-2 predicts its own
      grammar, so no generative probe can separate a recalled banality
      from a guessed one.)
  C5  Locality does not rot with scale: the witness perturbation at the
      largest checkpoint is within 3x of the first one.
      FALSIFIED if it grows without bound -- a memory that bleeds into
      unrelated text as it fills cannot ship at any size.

Run:  python behav/probe_capacity.py [--max 400000] [--model gpt2]
"""
import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sillage.core as core                              # noqa: E402
from sillage.runtime import Sillage                      # noqa: E402
from behavioral import (A_PREFIX, A_SENT, VALS, WITNESS,   # noqa: E402
                        filler, SUBJ, VERB, OBJ)

SYL1 = ["Vor", "Kres", "Zyl", "Mar", "Ilv", "Quan", "Bel", "Tarn", "Ozm",
        "Drev", "Palv", "Skog", "Yurm", "Cind", "Halb", "Nert", "Wisp",
        "From", "Ulve", "Brak", "Selp", "Grim", "Tovar", "Exqu", "Vand"]
SYL2 = ["la", "to", "kor", "me", "res", "dri", "fos", "wic", "ire", "kan",
        "eri", "fel", "ale", "rov", "rix", "val", "gar", "dah", "trem"]
SYL3 = ["gune", "mil", "b", "lune", "s", "x", "s", "k", "l", "t", "n",
        "d", "c", "el", "m", "th", "r", "v", "p"]


def entities(n, tok):
    """n invented names whose PROBE KEYS are distinct, by construction.

    The first version of this varied the first syllable and kept the
    last, so every name ended in the same two tokens -- and both fast
    tiers key on the last FOUR tokens, which for `The {e} protocol
    requires` are [..2 of the name.., ' protocol', ' requires']. Every
    entity therefore collided on ONE cold-store gram holding six rival
    successors, and recall read 0/6 while the mechanism was fine. The
    names are now filtered against the tokenizer itself: a candidate is
    kept only if its probe forms a 4-gram no other kept name forms.
    """
    out, seen_gram, i = [], set(), 0
    while len(out) < n and i < n * 400:
        e = (SYL1[(i // (len(SYL2) * len(SYL3))) % len(SYL1)]
             + SYL2[(i // len(SYL3)) % len(SYL2)]
             + SYL3[i % len(SYL3)]
             + ("" if i < len(SYL1) * len(SYL2) * len(SYL3) else str(i)))
        i += 1
        gram = tuple(tok.encode(A_PREFIX.format(e=e))[-core.NGRAM:])
        if gram in seen_gram:
            continue
        seen_gram.add(gram)
        out.append(e)
    if len(out) < n:
        raise SystemExit(f"only {len(out)} collision-free entities of {n}")
    return out


def value_for(i):
    return VALS[i % len(VALS)]


def block(seed, facts, sentences=60):
    """One block: filler prose, with each fact stated exactly twice."""
    parts = [filler(seed, sentences // 2)]
    for e, v in facts:
        parts.append(A_SENT.format(e=e, v=v))
    parts.append(filler(seed + 1000, sentences // 2))
    for e, v in facts:
        parts.append(A_SENT.format(e=e, v=v))
    return "\n\n".join(parts)


def filler_probe(seed):
    """A generic sentence opening from the filler grammar, and the word
    that should follow it. This is the BANAL half of C4."""
    s, v, o = SUBJ[seed % 10], VERB[(seed * 3) % 10], OBJ[(seed * 7) % 10]
    return f"The {s} {v}", o.split()[0]


def recall(s, pairs, n=8):
    hit = 0
    for e, v in pairs:
        out = s.complete(A_PREFIX.format(e=e), n=n)
        hit += v.split()[0].lower() in out.lower()
    return hit / max(1, len(pairs))


def grams_present(s, tok, pairs):
    """How many of these facts still have their probe gram in the store."""
    live = 0
    for e, _v in pairs:
        ids = tok.encode(A_PREFIX.format(e=e))[-core.NGRAM:]
        g = np.array(ids, dtype=np.int32).tobytes()
        slot = s.mem.cold.get(g)
        live += slot is not None and sum(slot[1].values()) >= core.COLD_MIN_COUNT
    return live / max(1, len(pairs))


def mass_quantiles(mem):
    """What surprise mass the store is holding, low to high."""
    if not mem.cold:
        return None
    tot = []
    for slot in mem.cold.values():
        src = slot[2] if mem.cold_mass and len(slot) > 2 and slot[2] else slot[1]
        tot.append(float(sum(src.values())))
    q = np.quantile(tot, [0.1, 0.5, 0.9])
    return [round(float(x), 2) for x in q]


def recall_filler(s, seeds, n=8):
    hit = 0
    for sd in seeds:
        prompt, want = filler_probe(sd)
        out = s.complete(prompt, n=n)
        hit += want.lower() in out.lower()
    return hit / max(1, len(seeds))


def witness_nats(s, text):
    """Locality, without letting the measurement move the v1 centre."""
    mu = None if s.mem.mu is None else s.mem.mu.copy()
    mu_n = s.mem.mu_n
    try:
        from probe_readout_dial import nll_nowrite
        b, m = nll_nowrite(s, text)
        return float(np.log(m) - np.log(b))
    finally:
        s.mem.mu, s.mem.mu_n = mu, mu_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--max", type=int, default=400_000)
    ap.add_argument("--per-block", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    checkpoints = [c for c in (5_000, 25_000, 100_000, 200_000, 400_000,
                               600_000, 800_000, 1_000_000) if c <= a.max]
    tmp = tempfile.mkdtemp(prefix="capacity_")
    res = {"model": a.model, "cold_max": core.COLD_MAX,
           "cold_min_count": core.COLD_MIN_COUNT,
           "checkpoints": [], "predictions": __doc__.split(
               "Registered BEFORE the run:")[1].split("Run:")[0].strip()}
    from sillage.ingest import ingest_text
    try:
        s = Sillage(model=a.model, state=tmp, quiet=True)
        s.load_model()
        # 6 facts per ~980-token block: 1M tokens needs ~6100 names,
        # and the first run died at 400k for asking 4000
        ents = entities(20000, s.load_tokenizer())
        base_facts = [(ents[j], value_for(j)) for j in range(10)]
        res["baseline"] = {"facts": recall(s, base_facts),
                           "filler": recall_filler(s, range(10))}
        print(f"  empty memory: facts {res['baseline']['facts']:.0%}, "
              f"filler {res['baseline']['filler']:.0%}  <- anything at or "
              f"below this is the frozen model, not the memory",
              flush=True)
        planted, seen_tokens, seed, fi = [], 0, 0, 0
        t0 = time.time()
        for target in checkpoints:
            while seen_tokens < target:
                facts = [(ents[fi + j], value_for(fi + j))
                         for j in range(a.per_block)]
                fi += a.per_block
                rec = ingest_text(s, block(seed, facts), f"b{seed}",
                                  quiet=True)
                seen_tokens += rec["tokens"]
                planted.append((seen_tokens, facts))
                seed += 1
            # cohorts: oldest planted, middle, newest
            old = planted[0][1]
            mid = planted[len(planted) // 2][1]
            new = planted[-1][1]
            row = {"tokens": seen_tokens,
                   "blocks": len(planted),
                   "facts_planted": fi,
                   "cold_grams": len(s.mem.cold),
                   "cold_full": len(s.mem.cold) >= core.COLD_MAX,
                   "M_norm": float(np.linalg.norm(s.mem.M)),
                   "MS_norm": float(np.linalg.norm(s.mem.MS)),
                   "recall_oldest": recall(s, old),
                   "recall_middle": recall(s, mid),
                   "recall_newest": recall(s, new),
                   "recall_filler": recall_filler(s, range(10)),
                   "grams_oldest": grams_present(s, s.load_tokenizer(), old),
                   "grams_newest": grams_present(s, s.load_tokenizer(), new),
                   "mass_q": mass_quantiles(s.mem),
                   "witness_log_ratio": witness_nats(s, WITNESS),
                   "minutes": round((time.time() - t0) / 60, 1)}
            res["checkpoints"].append(row)
            print(f"  {seen_tokens:>7} tok | cold {row['cold_grams']:>6}"
                  f"{' FULL' if row['cold_full'] else '    '} | "
                  f"oldest {row['recall_oldest']:.0%} middle "
                  f"{row['recall_middle']:.0%} newest "
                  f"{row['recall_newest']:.0%} | grams "
                  f"{row['grams_oldest']:.0%} mass-med "
                  f"{row['mass_q'][1] if row['mass_q'] else 0:.1f} | witness "
                  f"{row['witness_log_ratio']:+.4f} | "
                  f"{row['minutes']:.0f} min", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    cp = res["checkpoints"]
    if cp:
        first, last = cp[0], cp[-1]
        sat = next((r for r in cp if r["cold_full"]), None)
        res["verdict"] = {
            "C1": {"cold_at_end": last["cold_grams"],
                   "cap": core.COLD_MAX,
                   "saturated_at_tokens": sat["tokens"] if sat else None},
            "C2": {"oldest_first": first["recall_oldest"],
                   "oldest_last": last["recall_oldest"],
                   "held": last["recall_oldest"]
                   >= 0.8 * max(first["recall_oldest"], 1e-9)},
            "C3": {"newest_by_scale": [r["recall_newest"] for r in cp],
                   "holds": all(r["recall_newest"] >= 0.8 for r in cp)},
            "C4": {"grams_oldest_by_scale":
                   [r["grams_oldest"] for r in cp],
                   "mass_median_by_scale":
                   [r["mass_q"][1] if r["mass_q"] else None for r in cp],
                   "filler_is_base_model": res["baseline"]["filler"]},
            "C5": {"witness_first": first["witness_log_ratio"],
                   "witness_last": last["witness_log_ratio"]}}
        print("\n" + json.dumps(res["verdict"], indent=1, ensure_ascii=False))

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results", "capacity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
