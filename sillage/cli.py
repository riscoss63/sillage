"""One command-line assistant for the whole project.

    sillage read notes.md            read + memorize (all four mechanisms)
    sillage index notes.md           searchable at once, no model needed
    sillage ask "what did it say?"   grounded excerpts, nothing generated
    sillage complete "the report"    generate with memory + fast weights
    sillage chat                     both of the above, interactively
    sillage status                   what it knows, tier by tier
    sillage papers                   index the eight preprints and ask them
    sillage serve                    OpenAI-compatible endpoint, any client
    sillage watch ~/notes            read a folder as it changes
    sillage review                   what is about to be forgotten
    sillage export out/              a state you can share, without your text
    sillage pull user/cartridge      open somebody else's, from the Hub
    sillage demo notes.md            watch the memory work in one sitting
    sillage forget --all

State lives in ./.sillage (override with --state or $SILLAGE_STATE) and
survives restarts. Reading is CPU-only: about 2 minutes per 10k tokens with
--model gpt2, about 8 with Qwen3-0.6B (the default, and the one that handles
languages other than English). `index` and `ask` are instant and need no
model at all.

--model takes any causal language model -- the two shortcuts (qwen, gpt2)
carry the settings tuned in the papers, and anything else on the Hugging
Face hub or on disk works with the defaults:

    sillage read notes.md --model HuggingFaceTB/SmolLM2-135M
    sillage read notes.md --model ./my-finetuned-llama

A memory is written in one model's token space, so each model needs its own
--state directory; after that, `sillage status` and the rest pick the right
model back up on their own.

Meet a model the papers did not tune and the readout CALIBRATES itself on a
rolling window of what you read (published grids, refitted at the end of each
read, governing the next one). For qwen and gpt2 the published settings are
kept instead: a window read by a cold memory measurably loses to them.
--calibrate forces fitting, --no-calibrate forbids it, `read --recalibrate`
starts over.
"""

import argparse
import glob
import os
import shutil
import sys

from . import __version__
from .core import CALIB_MIN, ETA, R_FEAT
from .index import show
from .runtime import Sillage, default_state

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def sem2_layer(v):
    """--sem2 takes a layer number, or `auto` to choose one."""
    return "auto" if str(v).strip().lower() == "auto" else int(v)


def expand(paths):
    """Expand globs and ~ ourselves: PowerShell and cmd.exe do neither."""
    out = []
    for p in paths:
        p = os.path.expanduser(p)
        hits = sorted(glob.glob(p))
        out += hits if hits else [p]
    return out


def find_papers():
    """The eight bundled preprints, if the repository is around."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for root in (os.getcwd(), here):
        d = os.path.join(root, "papers")
        if os.path.isdir(d):
            tex = sorted(glob.glob(os.path.join(d, "*", "*.tex")))
            if tex:
                return tex
    return []


def make(a, **over):
    """Build the assistant from the parsed flags (one place, every command)."""
    kw = {"model": a.model, "state": a.state, "semantic": a.semantic,
          "fastweights": False if a.no_fastweights else None,
          "half_life": a.half_life,
          "calibrate": getattr(a, "calibrate", None),
          "device": a.device,
          "target": getattr(a, "target", None),
          "cold_mass": getattr(a, "cold_mass", None),
          "sem2": getattr(a, "sem2", None),
          "sem2_whiten": getattr(a, "sem2_whiten", None),
          "dtype": getattr(a, "dtype", None)}
    kw.update(over)
    s = Sillage(**kw)
    apply_readout(s, getattr(a, "readout", None))
    return s


#: paper 5's family settings, measured at 0.6B, 1.7B and 4B (behav_4b.json).
#: On its conflict protocol these turn 10% conversion into 100% at all three
#: capacities, and stable serving 0.9 into 0.95-1.0 -- with the SAME state.
#: The shipped qwen readout is the quiet end of the same dial.
FAMILY_READOUT = (40.0, 0.85, 0.5)

#: below this many moved tokens, treat a completion as the frozen model's
#: own invention (results/abstain_gen.json, results/reflow.json). It is a
#: floor on *contribution*, and contribution is not relevance: a question
#: whose wording matches stored text pulls the memory hard and can still
#: be answered with the wrong passage -- measured, and NOT fixable by
#: moving this number, since that case moved 16 tokens where a correct
#: answer moved 12.
FAINT = 3


def apply_readout(s, spec):
    """Override the n-gram tier's readout: `family`, `published`, or B,L,Q.

    The three numbers decide whether the memory speaks at all, and until
    now they were reachable only from Python. Default is unchanged, so
    every published number still reproduces with no flag.
    """
    if not spec:
        return
    if spec == "published":
        return
    if spec == "family":
        b, l, q = FAMILY_READOUT
    else:
        try:
            b, l, q = (float(x) for x in spec.split(","))
        except ValueError:
            raise SystemExit(
                f"--readout wants 'family', 'published', or three numbers "
                f"beta,lambda,quantile (got {spec!r})")
    s.mem.beta_G, s.mem.lam_G, s.mem.thr_qG = b, l, q


def fmt_readout(params):
    """Render one tier's readout settings the way `status` shows them."""
    beta, lam, q = params
    thr = "always on" if q is None else f"abstain below q{int(q * 100)}"
    return f"beta {beta:g}, lambda {lam:g}, {thr}"


def show_calibration(rep):
    """What the window decided, and what that number does and does not mean."""
    if not rep:
        return
    verb = "refitted" if rep.get("refit") else "fitted"
    print(f"{verb} the readout on {rep['n']} observations from what was just "
          f"read (the papers' grids):")
    for tier, key in (("n-gram tier ", "ngram"),
                      ("semantic tier", "semantic")):
        if key in rep:
            r = rep[key]
            print(f"  {tier}: "
                  + fmt_readout((r["beta"], r["lam"], r["thr_q"])))
    gain = rep["nll_before"] - rep["nll_after"]
    print(f"  {gain:+.4f} nats on that window. It governs the NEXT read, "
          f"never this one,")
    print(f"  so no perplexity printed here was tuned on itself.")


# ------------------------------------------------------------- commands ----

def cmd_read(a):
    """read: stream documents through the model and every mechanism."""
    paths = expand(a.files)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit("no such file: " + ", ".join(missing))
    s = make(a)
    if getattr(a, "recalibrate", False):
        s.mem.recalibrate()
        print("readout calibration reset -- it will be fitted again on the "
              "next few thousand tokens.")
    for path in paths:
        r = s.read(path, fast=getattr(a, "fast", False),
                   reflow=getattr(a, "reflow", False))[0]
        if r.get("ppl_frozen") is None:
            print(f"read {r['file']}: {r['tokens']} tokens in "
                  f"{r['minutes']:.1f} min | {r['tok_per_s']:.0f} tok/s "
                  f"(fast ingest -- writes only, no perplexity)",
                  flush=True)
            continue
        line = (f"read {r['file']}: {r['tokens']} tokens in "
                f"{r['minutes']:.1f} min | PPL {r['ppl_frozen']}")
        if s.mem.fastweights:
            line += f" -> {r['ppl_fastweights']} (adapter)"
        line += f" -> {r['ppl_with_memory']} (+memory)"
        print(line, flush=True)
        show_calibration(r.get("calibration"))
    print(f"memory consolidated and saved ({s.mem.tokens} tokens lifetime, "
          f"{len(s.mem.cold)} cold grams, {len(s.index.passages)} passages "
          f"indexed).")
    if s.mem.sem2_layer is not None and len(s.mem.res_S) < 500:
        # paper 8's tier keys on surprising positions only and abstains
        # until its own null has 500 of them. On a short document that
        # leaves it on and contributing nothing -- say it here, where the
        # decision to use --sem2 was just made, not only in `status`.
        print(f"note: the paper-8 tier is on but still SILENT -- "
              f"{len(s.mem.res_S)} scored positions of the 500 it needs "
              f"before it stops\n  abstaining. It anchors on surprising "
              f"tokens, so short documents give it few. Read more into "
              f"this state, or read the same again.")


def cmd_index(a):
    """index: make documents searchable without running the model."""
    paths = expand(a.files)
    s = make(a)
    # two `index.md` in two folders must not evict each other: when more
    # than one file is given, key each by its path relative to their
    # common parent, which is what `watch` already does for a vault
    root = (os.path.dirname(os.path.commonpath([os.path.abspath(p)
                                                for p in paths]))
            if len(paths) > 1 else None)
    total = 0
    for path in paths:
        if not os.path.exists(path):
            print(f"  skipped (not found): {path}")
            continue
        key = (os.path.relpath(os.path.abspath(path), root).replace(os.sep,
                                                                   "/")
               if root else None)
        n = s.add_to_index(path, name=key)
        total += n
        print(f"  {(key or os.path.basename(path))[:40]:40s} "
              f"{n:4d} passages")
    print(f"indexed {total} passages -- `sillage ask` works now. "
          f"`sillage read` also memorizes them.")


def cmd_ask(a):
    """ask: exact passages from what has been read, with their source."""
    s = make(a)
    if not s.index.passages:
        if s.mem.tokens:
            sys.exit(
                f"this state has read {s.mem.tokens} tokens but carries no "
                f"passages -- which is what a pulled cartridge is: it ships "
                f"the matrices, not the text.\n`sillage complete` works on "
                f"it; `ask` quotes documents, so it needs ones you have read "
                f"yourself (`sillage index <file>` is instant).")
        sys.exit("nothing has been read yet: sillage read <file> "
                 "(or sillage index <file> for the instant version)")
    show(s.ask(" ".join(a.query), k=a.k, numeric_only=a.numbers))


def cmd_complete(a):
    """complete: continue a prompt with the memory and the adapter."""
    s = make(a)
    prompt = " ".join(a.prompt)
    print(prompt + s.complete(prompt, n=a.n, temp=a.temp,
                              fast=getattr(a, "fast", False)))
    at = s.attribution()
    if at is None:
        return
    # to stderr, not stdout: the completion IS the output of this command
    # (paper 5's --fast guarantees it byte for byte, and a pipe should get
    # the text and nothing else). The attribution is diagnostics, and a
    # terminal shows it just the same.
    if at["moved"] < FAINT:
        # FAINT, not zero: measured on two French documents and 36
        # questions, a memory that moved fewer than three tokens had
        # nothing for the question 11 times out of 12, while every
        # correct answer moved at least 12. Firing only at zero misses
        # a third of the cases where the answer is pure invention.
        print(f"\nnote: the memory barely spoke here ({at['moved']} of "
              f"{at['tokens']} tokens).\n  Treat this as the frozen model "
              f"answering alone -- what it just said is\n  most likely not "
              f"something it read in your documents.", file=sys.stderr)
    else:
        tiers = ", ".join(f"{k} {v}" for k, v in sorted(at["tiers"].items()))
        print(f"\n[memory moved {at['moved']}/{at['tokens']} tokens; "
              f"tiers spoke: {tiers}]", file=sys.stderr)


def cmd_watch(a):
    """watch: read a folder as it changes, and say what was new."""
    from .watch import watch
    s = make(a)
    exts = ([e if e.startswith(".") else "." + e
             for e in a.ext.split(",")] if a.ext else None)
    watch(s, a.folder, interval=a.interval, once=a.once, exts=exts,
          fast=not a.full, quiet=False, since=a.since)
    if s.mem.sem2_layer is not None and len(s.mem.res_S) < 500:
        # `watch --sem2 auto` is the command a vault user actually types,
        # and it was the one path that never said the tier was still
        # silent. Measured: the tier's null fills at roughly 1-2% of the
        # tokens read, so a folder of notes needs tens of thousands of
        # them -- or several passes -- before it speaks at all.
        print(f"\nnote: the paper-8 tier is on but still SILENT -- "
              f"{len(s.mem.res_S)} scored positions of the 500 it needs. It "
              f"anchors only on\n  surprising tokens (about 1-2% of what is "
              f"read), so a small vault takes several\n  passes to get "
              f"there. Until then, rephrased recall behaves as if the flag "
              f"were off.")


def cmd_export(a):
    """export: a state someone else can open, without your documents."""
    s = make(a)
    if not s.mem.tokens:
        sys.exit("nothing has been read yet: sillage read <file>")
    info = s.export_shareable(a.out)
    m = info["manifest"]
    print(f"cartridge written to {info['dir']} "
          f"({info['bytes']/1e6:.1f} MB)")
    print(f"  {m['model']} ({m['hub']}), {m['tokens_read']} tokens read "
          f"from {len(m['documents'])} document(s)")
    print(f"  files: {', '.join(info['files'])}")
    print(f"  left out: {'; '.join(m['left_out'])}")
    if info["thin"]:
        print(f"\n  WARNING: this memory has only {info['scored']} "
              f"scored positions, under the 500 a tier needs before it\n"
              f"  stops abstaining -- so this cartridge will stay "
              f"SILENT: everything it recalled\n  came from the cold "
              f"store, which cannot be shared. Read more into it first "
              f"(a few\n  thousand tokens), then export again.")
    print(f"\n  This carries no plain text -- which is NOT the same as "
          f"anonymous.\n  Inverting the matrices is hard, not proven "
          f"impossible, and no\n  inversion attack has been run against "
          f"this format yet. Share it\n  the way you would share a "
          f"model you fine-tuned on your own data.")


def cmd_pull(a):
    """pull: somebody else's matrices, opened as this state."""
    s = make(a)
    try:
        info = s.pull_cartridge(a.source, force=a.force)
    except RuntimeError as e:
        sys.exit(str(e))
    m = info["manifest"]
    print(f"pulled {info['from']} ({info['origin']}) into {info['dir']} "
          f"({info['bytes']/1e6:.1f} MB)")
    print(f"  written by sillage {m.get('sillage') or '?'} for "
          f"{m.get('model')} ({m.get('hub')})")
    docs = m.get("documents") or []
    print(f"  {m.get('tokens_read', 0)} tokens read from "
          f"{len(docs)} document(s)"
          + (": " + ", ".join(docs[:3]) + ("..." if len(docs) > 3 else "")
             if docs else ""))
    print(f"  left out by design: {'; '.join(m.get('left_out', []))}")
    print(f"\n  A cartridge speaks through its matrices alone. Measured "
          f"on the state\n  paper 8 was written from, that costs about "
          f"two canonical recalls in ten\n  and no perplexity -- and "
          f"paraphrased recall is untouched. Nothing here\n  is durable "
          f"the way your own reading is: the cold store, which carries\n"
          f"  long retention, cannot be shared.")
    print(f"\n  Read your own documents on top with `sillage read` -- "
          f"they will be\n  consolidated normally, in this state, next to "
          f"what you pulled.")


def cmd_review(a):
    """review: what is about to be forgotten, and what fixes it."""
    s = make(a)
    rows = s.review()
    if not rows:
        sys.exit("nothing has been read yet: sillage read <file>")
    print("consolidation, document by document (paper 6: this memory "
          "durably keeps what it has seen twice)\n")
    print(f"  {'document':34s} {'kept':>6s} {'fragile':>8s} "
          f"{'gone':>6s}   share")
    for r in rows:
        bar = "#" * int(round(r["share"] * 20))
        print(f"  {r['source'][:34]:34s} {r['consolidated']:6d} "
              f"{r['fragile']:8d} {r['gone']:6d}   "
              f"{r['share']:5.0%} {bar}")
    weakest = rows[0]
    print(f"\n  'fragile' means stored once: present, but under the "
          f"admission threshold, so the memory never speaks it. "
          f"Reading a document again is what consolidates it.")
    if a.read:
        n = max(1, a.read)
        # the log keeps the path each document was read from, so a reread
        # does not depend on standing in the same directory as that read
        where = {f["file"]: f.get("path") for f in s.mem.log["files"]}
        found, missing = [], []
        for r in rows[:n]:
            src = r["source"]
            cands = [c for c in (where.get(src), src) if c]
            hit = next((c for c in expand(cands) if os.path.exists(c)),
                       None)
            found.append((src, hit)) if hit else missing.append(src)
        if missing:
            print(f"\n  cannot reread (no longer at the path it was read "
                  f"from): {', '.join(missing)}")
        for src, path in found:
            r = s.read(path, fast=True, name=src)[0]
            print(f"  reread {r['file']}: {r['tokens']} tokens")
        if found:
            after = {x["source"]: x for x in s.review()}
            for src, _ in found:
                if src in after:
                    print(f"  {src}: {after[src]['share']:.0%} "
                          f"consolidated now")
    else:
        print(f"  least consolidated: {weakest['source']} "
              f"({weakest['share']:.0%}) -- `sillage review --read 1` "
              f"rereads it.")


def cmd_serve(a):
    """serve: the memory behind an OpenAI-compatible endpoint."""
    from .serve import serve
    s = make(a)
    s.load_model()          # pay the load now, not on the first request
    serve(s, host=a.host, port=a.port, context=not a.no_context,
          k=a.k, token=a.token, quiet=a.quiet)


def cmd_status(a):
    """status: what the assistant knows, tier by tier."""
    s = make(a)
    st = s.status()
    print(f"sillage {__version__} - {st['model']} (frozen)")
    print(f"  read so far        : {st['tokens']} tokens, "
          f"{st['documents']} document(s)")
    for tier, nbytes in st["sizes"].items():
        if not nbytes:
            continue
        extra = ""
        if tier.startswith("n-gram"):
            extra = f"   {st['writes_per_parameter']:.3f} writes/parameter"
        if tier.startswith("fast"):
            extra = f"   rank {R_FEAT}, eta {ETA}, uniform step"
            if not st.get("adapter_written"):
                # `watch` reads fast by default and a fast read does not
                # train the adapter, so a whole week of watching leaves
                # these megabytes at exactly zero. Say so.
                extra += ("   EMPTY: every read so far was a fast one, "
                          "which does not train it")
        if tier.startswith("cold"):
            extra = f"   {st['cold_grams']} grams"
        print(f"  {tier:19s}: {nbytes/1e6:6.1f} MB{extra}")
    print(f"  lexical index      : {st['passages']} passages"
          + ("" if st["passages"] else
             "   (a pulled cartridge ships none: `ask` needs your own "
             "documents)"))
    sem = ("on" if st["semantic"] else
           "off  (raw hidden states would need whitening on this model)")
    print(f"  semantic tier      : {sem}")
    if st.get("sem2_layer") is not None:
        wtxt = " + ZCA whitening" if st.get("sem2_whiten") else ""
        print(f"  semantic keys      : layer {st['sem2_layer']}, "
              f"surprise-anchored, query pooling (paper 8){wtxt}")
        if st.get("scored_S", 0) < 500:
            # the tier abstains until its own null has 500 observations,
            # and it only observes SURPRISING positions -- so a short
            # document can leave it on and silent with nothing said
            print(f"                       SILENT so far: {st['scored_S']} "
                  f"scored positions of the 500 it needs before it stops "
                  f"abstaining. Read more, or read the same again.")
    elif st["semantic"]:
        # two real-world trials measured "rephrased recall 0/10" against
        # paper 8's published 80% on a state where this tier had simply
        # never been switched on. Saying nothing here is what let that
        # happen twice.
        print("  semantic keys      : paper 2's, not paper 8's -- rephrased "
              "recall stays near 0 here.")
        print("                       `--sem2 auto` AT READ TIME is what "
              "buys it (measured 0% -> 80%);")
        print("                       the keys are written while reading, "
              "so a state read without")
        print("                       it has to be read again.")
    hl = (f"half-life {int(st['half_life'])} tokens" if st["half_life"]
          else "off")
    print(f"  forgetting         : {hl}")
    print(f"  cold weighting     : "
          f"{'surprise mass (paper 6)' if st.get('cold_mass') else 'counts'}")
    if st["calibrated"]:
        origin = "fitted on what you read"
    elif st["calibrating"]:
        need = max(0, CALIB_MIN - st["calib_seen"])
        origin = f"defaults, fitting in {need} more observations"
    else:
        origin = "as published for this model"
    print(f"  readout            : {fmt_readout(st['readout']['ngram'])}")
    if st["semantic"]:
        print(f"  readout, semantic  : "
              f"{fmt_readout(st['readout']['semantic'])}")
    print(f"                       ({origin})")
    print(f"  state on disk      : {st['disk']/1e6:.1f} MB  "
          f"({st['state_dir']})")
    if st["writes_per_parameter"] > 0.4 and not st["half_life"]:
        print("  note: past ~0.5 writes/parameter the matrix saturates; "
              "add --half-life 100000 to keep learning.")
    for f in st["files"][-10:]:
        if f.get("ppl_frozen") is None:      # read --fast, or watch
            print(f"  {f['date']}  {f['file']}: {f['tokens']} tok, "
                  f"fast ingest (writes only, no perplexity)")
            continue
        fw = f.get("ppl_fastweights")
        mid = f" -> {fw}" if fw else ""
        print(f"  {f['date']}  {f['file']}: {f['tokens']} tok, "
              f"PPL {f['ppl_frozen']}{mid} -> {f['ppl_with_memory']}")


def cmd_forget(a):
    """forget: wipe the state, or drop one document from the index."""
    state = a.state or default_state()
    if a.all:
        shutil.rmtree(state, ignore_errors=True)
        print(f"memory wiped ({state}).")
        return
    if not a.what:
        sys.exit("use: sillage forget --all   (or: sillage forget <file>)")
    s = make(a)
    name = os.path.basename(a.what[0])
    before = len(s.index.passages)
    s.index.passages = [p for p in s.index.passages if p["source"] != name]
    if len(s.index.passages) == before:
        sys.exit(f"{name} is not in the index.")
    s.index._rebuild()
    s.index.save()
    s.mem.log["files"] = [f for f in s.mem.log["files"] if f["file"] != name]
    s.mem.save()
    print(f"{name} removed from the index and the log "
          f"({before - len(s.index.passages)} passages).")
    print("Its Hebbian traces stay superposed in the matrices: only "
          "`forget --all`, or forgetting (--half-life), removes those.")


def cmd_papers(a):
    """papers: index the eight preprints that ship with the repository."""
    tex = find_papers()
    if not tex:
        sys.exit("papers/ not found -- run this from the repository, or "
                 "point `sillage index` at your own documents.")
    a.files = tex
    if a.with_memory:
        print("reading the eight preprints into the memory "
              f"({make(a).mem.hub}) -- a few minutes ...")
        cmd_read(a)
    else:
        cmd_index(a)
    print('try: sillage ask "does rank 16 suffice?"')


def cmd_chat(a):
    """chat: ask and generate in one interactive session."""
    s = make(a)
    print(f"sillage {__version__} - {len(s.index.passages)} passages, "
          f"{s.mem.tokens} tokens in memory.")
    print("  <question>      grounded excerpts (nothing is generated)")
    print("  /say <prompt>   generate with the memory and the adapter")
    print("  /read <file>    read a document now")
    print("  /status  /k N  /quit\n")
    k = a.k
    while True:
        try:
            line = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/exit", "q"):
            return
        if line.startswith("/k "):
            k = max(1, int(line.split()[1]))
            print(f"  excerpts per answer: {k}")
        elif line.startswith("/say "):
            prompt = line[5:]
            print(prompt + s.complete(prompt, n=a.n, temp=a.temp,
                                      fast=getattr(a, "fast", False)))
        elif line.startswith("/read "):
            path = line[6:].strip()
            if os.path.exists(path):
                r = s.read(path)[0]
                print(f"  {r['file']}: {r['tokens']} tokens, PPL "
                      f"{r['ppl_frozen']} -> {r['ppl_with_memory']}")
            else:
                print("  no such file")
        elif line.startswith("/status"):
            cmd_status(a)
        else:
            show(s.index.search(line, k=k))
        print()


def adapter(s, rec):
    """The adapter leg of a perplexity line, when there is an adapter.

    `--target` turns it off (paper 5: the adapter belongs to the reading
    model's geometry), and then it would print the frozen number twice.
    """
    return (f"  ->  adapter {rec['ppl_fastweights']}"
            if s.mem.fastweights else "")


def cmd_demo(a):
    """Two sessions on one document: read it, then read it again."""
    path = a.file[0] if a.file else None
    if path is None:
        tex = find_papers()
        if not tex:
            sys.exit("usage: sillage demo <a text file you own>")
        path = tex[0]
    if a.state:
        sys.exit("demo starts from an empty memory, so it wipes the state "
                 "directory it uses -- it will not do that to one you "
                 "named. Leave --state out and it uses ./.sillage-demo.")
    state = ".sillage-demo"
    shutil.rmtree(state, ignore_errors=True)
    from .index import read_text
    text = read_text(path)
    s = make(a, state=state)
    tok, _ = s.load_model()
    ids = tok.encode(text)[:a.max_tokens]
    text = tok.decode(ids)
    print(f"\n--- session 1: {os.path.basename(path)} "
          f"({len(ids)} tokens), memory empty ---")
    r1 = s.read_text(text, os.path.basename(path))
    s.index.add(text, os.path.basename(path))
    s.save()
    print(f"  frozen {r1['ppl_frozen']}{adapter(s, r1)}  ->  "
          f"+ memory {r1['ppl_with_memory']}")
    print("\n--- session 2: same document, everything remembered ---")
    s2 = make(a, state=state)
    r2 = s2.read_text(text, os.path.basename(path))
    s2.save()
    print(f"  frozen {r2['ppl_frozen']}{adapter(s2, r2)}  ->  "
          f"+ memory {r2['ppl_with_memory']}")
    gain = r1["ppl_frozen"] / max(1e-9, r2["ppl_with_memory"])
    print(f"\nperplexity divided by {gain:.1f}x on the second pass, with "
          f"{s2.status()['disk']/1e6:.1f} MB of state and no gradient.")
    words = text.split()
    if len(words) > 60:
        cut = " ".join(words[40:48])
        out = s2.complete(cut, n=a.n, temp=a.temp,
                          fast=getattr(a, "fast", False))
        print(f"\ncompletion of a phrase from the document:\n  {cut!r}"
              f" ->{out!r}")
    print(f"\n(demo state in {state}/ -- delete it or "
          f"`sillage forget --all --state {state}`)")


# ----------------------------------------------------------------- parser ---

def build_parser():
    """The whole command line: shared flags, then one parser per command."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=None, metavar="NAME",
                        help="frozen model to augment: a shortcut (qwen, "
                             "gpt2), any causal LM on the Hugging Face hub, "
                             "or a local path. Defaults to whatever the "
                             "state was built with, else qwen.")
    common.add_argument("--state", default=None,
                        help="state directory (default: ./.sillage)")
    common.add_argument("--semantic", dest="semantic", action="store_true",
                        default=None, help="force the semantic tier on")
    common.add_argument("--no-semantic", dest="semantic",
                        action="store_false", help="force it off")
    common.add_argument("--no-fastweights", action="store_true",
                        help="disable the rank-16 readout adapter")
    common.add_argument("--half-life", type=float, default=None,
                        metavar="N", help="forgetting half-life in tokens "
                                          "(off by default; try 100000)")
    common.add_argument("--sem2", type=sem2_layer, default=None,
                        metavar="LAYER|auto",
                        help="early-layer anchored semantic keys (paper "
                             "8): key the tier on hidden layer LAYER "
                             "(measured: 1 for qwen, 5 for gpt2), "
                             "anchor writes on surprising tokens, pool "
                             "the query over the prompt at generation. "
                             "`auto` picks the layer from the first "
                             "document you read -- and decides on "
                             "whitening too; the state remembers both")
    common.add_argument("--sem2-whiten", dest="sem2_whiten",
                        action="store_true", default=None,
                        help="add ZCA whitening estimated from what you "
                             "read (paper 8: the model-adaptive piece "
                             "-- unnecessary on qwen, needed on gpt2; "
                             "on by default for a model nobody has "
                             "measured)")
    common.add_argument("--no-sem2-whiten", dest="sem2_whiten",
                        action="store_false",
                        help="key the tier on raw hidden states instead")
    common.add_argument("--cold-mass", dest="cold_mass",
                        action="store_true", default=None,
                        help="weight the cold store's successors by "
                             "surprise mass instead of raw counts (paper "
                             "6's adversarial fix; off by default -- "
                             "counts reproduce the papers' numbers)")
    common.add_argument("--dtype", default=None,
                        choices=["float32", "bfloat16", "float16",
                                 "int8"],
                        help="how to load the frozen model: float32 "
                             "(default), bfloat16/float16 (half the "
                             "memory), or int8 (torch's dynamic "
                             "quantisation, CPU, no extra dependency). "
                             "A bigger model on modest hardware, "
                             "without losing the hidden states the "
                             "memory keys on")
    common.add_argument("--device", default=None, metavar="DEV",
                        help="where the frozen forward passes run: cpu, "
                             "cuda, mps (default: cuda when there is one)")
    common.add_argument("--calibrate", dest="calibrate",
                        action="store_true", default=None,
                        help="fit the readout on what you read (the default "
                             "for any model the papers did not tune)")
    common.add_argument("--no-calibrate", dest="calibrate",
                        action="store_false",
                        help="keep the readout settings as they are")
    common.add_argument("--readout", default=None, metavar="SPEC",
                        help="how loudly the memory speaks: 'published' "
                             "(default, the papers' numbers), 'family' "
                             "(paper 5's 40,0.85,0.5 -- louder recall, "
                             "measured at 0.6B/1.7B/4B), or beta,lambda,"
                             "quantile")

    gen = argparse.ArgumentParser(add_help=False)
    gen.add_argument("-n", type=int, default=40,
                     help="tokens to generate (default: 40)")
    gen.add_argument("--temp", type=float, default=0.0,
                     help="sampling temperature (0 = greedy)")
    gen.add_argument("--fast", action="store_true",
                     help="speculative decoding from the memory (paper 5): "
                          "identical output, greedy only, faster where the "
                          "memory is confident")
    gen.add_argument("--target", default=None, metavar="NAME",
                     help="generate with a bigger SAME-TOKENIZER sibling "
                          "reading this state (e.g. Qwen/Qwen3-1.7B on a "
                          "qwen state); the adapter stays off")

    ret = argparse.ArgumentParser(add_help=False)
    ret.add_argument("-k", type=int, default=3,
                     help="excerpts per answer (default: 3)")

    ap = argparse.ArgumentParser(
        prog="sillage", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version",
                    version=f"sillage {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("read", parents=[common],
                       help="read documents: memorize and index them")
    p.add_argument("files", nargs="+")
    p.add_argument("--reflow", action="store_true",
                   help="join the lines inside each paragraph before "
                        "reading, so a question typed on one line forms "
                        "the same 4-token key the document stored "
                        "(measured 7/8 -> 8/8 facts recalled; the token "
                        "stream changes, so the perplexity of this read "
                        "is not comparable to the published numbers)")
    p.add_argument("--recalibrate", action="store_true",
                   help="fit the readout again, on what you read next")
    p.add_argument("--fast", action="store_true",
                   help="blocked write-only ingestion (paper 7): ~40x "
                        "on long documents; exact cold store, declared "
                        "amplitude tolerances, no perplexity report; "
                        "the adapter does not learn during a fast read")
    p.set_defaults(fn=cmd_read)

    p = sub.add_parser("index", parents=[common],
                       help="index documents for `ask` (instant, no model)")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("ask", parents=[common, ret],
                       help="grounded excerpts from what has been read")
    p.add_argument("query", nargs="+")
    p.add_argument("--numbers", action="store_true",
                   help="only passages containing numeric results")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("complete", parents=[common, gen],
                       help="generate with the memory and the adapter")
    p.add_argument("prompt", nargs="+")
    p.set_defaults(fn=cmd_complete)

    p = sub.add_parser("chat", parents=[common, gen, ret],
                       help="interactive session")
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("status", parents=[common],
                       help="what it knows, tier by tier")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("watch", parents=[common],
                       help="read a folder as it changes, with a "
                            "salience journal of what was new")
    p.add_argument("folder")
    p.add_argument("--interval", type=int, default=60,
                   help="seconds between passes (default 60)")
    p.add_argument("--once", action="store_true",
                   help="one pass and exit -- for a cron job")
    p.add_argument("--ext", default=None, metavar="LIST",
                   help="extensions to read (default: md,txt,markdown)")
    p.add_argument("--full", action="store_true",
                   help="normal reads instead of fast ones (slower, "
                        "reports perplexity, and the only mode that "
                        "trains the rank-16 adapter)")
    p.add_argument("--since", type=float, default=None, metavar="DAYS",
                   help="rank only what was read in the last DAYS in the "
                        "salience journal (e.g. --since 7 for the week)")
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("export", parents=[common],
                       help="write a shareable state: the matrices, "
                            "without the cold store or the index")
    p.add_argument("out", help="directory to write the cartridge to")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("pull", parents=[common],
                       help="open somebody else's cartridge (a directory "
                            "or a Hugging Face repo id)")
    p.add_argument("source", help="user/name on the Hub, or a local "
                                  "directory written by `sillage export`")
    p.add_argument("--force", action="store_true",
                   help="replace the memory already in --state")
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser("review", parents=[common],
                       help="what is about to be forgotten (paper 6's "
                            "two-occurrence rule, as a command)")
    p.add_argument("--read", type=int, default=0, metavar="N",
                   help="reread the N least consolidated documents")
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("serve", parents=[common],
                       help="OpenAI-compatible endpoint for any client")
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 by default: this memory holds the "
                        "text you fed it, so it stays on this machine "
                        "unless you say otherwise")
    p.add_argument("--port", type=int, default=8000,
                   help="port to bind (default: 8000)")
    p.add_argument("-k", type=int, default=3,
                   help="passages injected into each prompt (paper 7: "
                        "formulation happens in the window)")
    p.add_argument("--no-context", action="store_true",
                   help="do not inject retrieved passages; the memory "
                        "then only speaks through the readout")
    p.add_argument("--token", default=None,
                   help="require `Authorization: Bearer TOKEN`")
    p.add_argument("--quiet", action="store_true",
                   help="do not log one line per request")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("forget", parents=[common],
                       help="wipe the memory (--all) or drop one document")
    p.add_argument("what", nargs="*")
    p.add_argument("--all", action="store_true",
                   help="wipe the whole state directory: matrices, cold "
                        "store, index and log")
    p.set_defaults(fn=cmd_forget)

    p = sub.add_parser("papers", parents=[common],
                       help="index the eight preprints shipped with the repo")
    p.add_argument("--with-memory", action="store_true",
                   help="also read them into the memory (slow)")
    p.set_defaults(fn=cmd_papers)

    p = sub.add_parser("demo", parents=[common, gen],
                       help="two sessions on one document, start to finish")
    p.add_argument("file", nargs="*")
    p.add_argument("--max-tokens", type=int, default=4000,
                   help="tokens of the document to read in each of the "
                        "two sessions (default: 4000)")
    p.set_defaults(fn=cmd_demo)
    return ap


def main(argv=None):
    """Entry point of the `sillage` command."""
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
