"""How fast does REAL prose fill the cold store?

The capacity study runs on synthetic filler, which repeats its own
grammar and so produces few distinct 4-grams per token. That makes the
saturation point corpus-dependent, and a law stated in TOKENS would be
wrong for anyone else's documents.

The cold store admits a gram only when it has been seen at least
COLD_MIN_COUNT times, so what fills it is not tokens and not even
distinct grams -- it is REPEATED grams. This measures that rate on real
text, in three languages and three registers, so the capacity law can be
stated in the unit a user actually has: their own documents.

No model is loaded; only the tokenizer, which is what the tiers key on.
"""
import glob
import io
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sillage.core as core                        # noqa: E402
from sillage.index import strip_latex              # noqa: E402
from sillage.runtime import Sillage                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rate(tok, text, label):
    ids = tok.encode(text)
    n = len(ids)
    if n < 2000:
        return None
    grams = Counter(tuple(ids[i:i + core.NGRAM])
                    for i in range(n - core.NGRAM))
    distinct = len(grams)
    admitted = sum(1 for c in grams.values() if c >= core.COLD_MIN_COUNT)
    row = {"label": label, "tokens": n,
           "distinct_per_1k": round(1000 * distinct / n, 1),
           # read ONCE: only the grams a document repeats on its own get in
           "admitted_per_1k": round(1000 * admitted / n, 2),
           "tokens_to_fill_read_once": (int(core.COLD_MAX * n / admitted)
                                        if admitted else None),
           # read TWICE (paper 6's rule, and what the tool recommends):
           # every distinct gram now has two occurrences, so nearly all of
           # them are admitted -- and the store fills an order of magnitude
           # sooner. This is the number a user actually meets.
           "tokens_to_fill_read_twice": int(core.COLD_MAX * n / distinct)}
    print("  %-30s %6d tok | distinct %6.1f/1k | admitted %5.2f/1k"
          " | fills at %10s (1 read) %9s (2 reads)"
          % (label, n, row["distinct_per_1k"], row["admitted_per_1k"],
             f"{row['tokens_to_fill_read_once']:,}"
             if row["tokens_to_fill_read_once"] else "never",
             f"{row['tokens_to_fill_read_twice']:,}"), flush=True)
    return row


def main():
    # a temp state, never the default one: `Sillage(state=None)` opens the
    # user's real memory and would migrate it on the way past
    import tempfile
    tmp = tempfile.mkdtemp(prefix="gramrate_")
    s = Sillage(model="qwen", state=tmp, quiet=True)
    tok = s.load_tokenizer()
    rows = []
    print("cold store: admits a gram at >= %d occurrences, cap %d\n"
          % (core.COLD_MIN_COUNT, core.COLD_MAX), flush=True)

    for tex in sorted(glob.glob(os.path.join(ROOT, "papers", "*", "*.tex"))):
        body = strip_latex(io.open(tex, encoding="utf-8",
                                   errors="replace").read())
        r = rate(tok, body, "paper: " + os.path.basename(tex))
        if r:
            rows.append(r)

    # the two French documents the recent probes were built on, and the
    # repository's own prose, which is what a reader of the repo has
    sys.path.insert(0, os.path.join(ROOT, "behav"))
    try:
        from probe_readout_dial import DOC as FR1
        from probe_abstain_gen import DOC as FR2
        rows.append(rate(tok, (FR1 + "\n\n" + FR2) * 6,
                         "French reports x6 (short docs, reread)"))
    except Exception as exc:
        print("  (French docs unavailable: %s)" % exc)
    for name in ("README.md", "REPRODUCE.md"):
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            rows.append(rate(tok, io.open(p, encoding="utf-8",
                                          errors="replace").read(), name))

    rows = [r for r in rows if r]
    def med(key):
        v = sorted(r[key] for r in rows if r.get(key))
        return v[len(v) // 2]

    once, twice = (med("tokens_to_fill_read_once"),
                   med("tokens_to_fill_read_twice"))
    out = {"cold_max": core.COLD_MAX, "min_count": core.COLD_MIN_COUNT,
           "rows": rows,
           "median_admitted_per_1k": med("admitted_per_1k"),
           "median_tokens_to_fill_read_once": once,
           "median_tokens_to_fill_read_twice": twice,
           "ratio": round(once / twice, 1)}
    print("\nreal prose repeats only %.0f%% of its own 4-grams, so a SINGLE "
          "read\nadmits almost nothing: %.1f grams per 1k tokens."
          % (100 * med("admitted_per_1k") / 1000 * 1000 / 1000,
             med("admitted_per_1k")))
    print("  read once : the store fills after ~{:,} tokens".format(once))
    print("  read twice: after ~{:,} tokens -- {}x sooner, and reading twice"
          .format(twice, out["ratio"]))
    print("              is exactly what paper 6 tells you to do.")
    print("  => durability and capacity are the SAME budget.")
    dst = os.path.join(ROOT, "behav", "results", "gramrate.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with io.open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("written", dst)


if __name__ == "__main__":
    main()
