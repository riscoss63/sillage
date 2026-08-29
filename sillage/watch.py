"""`sillage watch`: read a folder as it changes, and say what was new.

Point it at a notes folder (an Obsidian vault, a directory of Markdown)
and it reads what changed since last time -- nothing else. Two things
make this worth having, and both come out of the papers rather than
from a product wish:

  * The SALIENCE JOURNAL. Every write is scaled by the frozen model's
    own surprise, a scalar it computes anyway at inference. Averaged per
    document it is a free measure of how unexpected a note's prose is to
    the model -- the kind of number a note-taking app pays an LLM call
    per decision for. Here it costs nothing extra: it was already on its
    way through.

    Read what it is, not what one would like it to be. It is the
    FROZEN MODEL's surprise, measured before the memory speaks, so:
    re-reading a file the memory already holds returns the same number
    (measured: byte-identical), it cannot say "new to me", only "unusual
    prose"; and it is a per-token MEAN, so a short dense note outranks a
    long one and appending new material to a note LOWERS its score by
    dilution (measured: 2.56 -> 2.42 nats after appending two genuinely
    new decisions, the appended block alone scoring 2.67). It ranks
    jargon and density, which correlates with novelty on technical notes
    and does not on prose.
  * REREADING CONSOLIDATES (paper 6). A file you edit and save twice
    crosses the cold store's two-occurrence threshold on its own, which
    is exactly what makes it durable. Watching a folder is therefore not
    just convenience -- it is the mechanism the law describes, running
    on its own schedule.

    sillage watch ~/notes                 # every 60s, until ctrl-c
    sillage watch ~/notes --once          # one pass, for a cron job
    sillage watch ~/notes --interval 600

State of the walk (which file, which size, which mtime, and what it
scored) lives in `watch.json` inside the memory's own state directory,
so the two never drift apart.
"""

import json
import os
import time

SKIP_DIRS = {".git", ".obsidian", "node_modules", "__pycache__",
             ".venv", "venv", ".sillage"}
# What a folder of notes is made of. Everything else is skipped, because
# index.read_text decodes with errors="replace": a PDF or a PNG would go
# into the matrices as replacement characters, irreversibly.
DEFAULT_EXTS = (".md", ".txt", ".markdown")


def scan(root, exts):
    """Files worth reading, with the fingerprint that says 'changed'."""
    out = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                   and not d.startswith(".")]
        for name in files:
            if exts and os.path.splitext(name)[1].lower() not in exts:
                continue
            path = os.path.join(base, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if st.st_size == 0:
                continue
            out[os.path.abspath(path)] = {"size": st.st_size,
                                          "mtime": int(st.st_mtime)}
    return out


class Watcher:
    def __init__(self, assistant, root, exts=None,
                 fast=True, quiet=False):
        self.s = assistant
        self.root = os.path.abspath(os.path.expanduser(root))
        # None means "the default set", never "read anything": the CLI
        # passes None whenever --ext is absent, and a shadowed default
        # would have this walk swallow binaries.
        self.exts = {e.lower() for e in (exts or DEFAULT_EXTS)}
        self.fast = fast
        self.quiet = quiet
        self.path = (None if assistant.state_dir is None else
                     os.path.join(assistant.state_dir, "watch.json"))
        self.seen = {}
        self.journal = []
        if self.path and os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                saved = json.load(f)
            self.seen = saved.get("seen", {})
            self.journal = saved.get("journal", [])

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"root": self.root, "seen": self.seen,
                       "journal": self.journal[-500:]}, f, indent=1)

    def changed(self):
        now = scan(self.root, self.exts)
        todo = [p for p, meta in now.items()
                if self.seen.get(p, {}).get("mtime") != meta["mtime"]
                or self.seen.get(p, {}).get("size") != meta["size"]]
        return sorted(todo), now

    def pass_once(self):
        """Read what changed. Returns the salience entries just made."""
        todo, now = self.changed()
        made = []
        for path in todo:
            g_before = (self.s.mem.g_sum, self.s.mem.g_cnt)
            # keyed by the path relative to the walk, not the basename:
            # two notes.md in two subfolders must not evict each other
            rec = self.s.read(
                path, fast=self.fast,
                name=os.path.relpath(path, self.root).replace(os.sep, "/"))[0]
            dg = self.s.mem.g_sum - g_before[0]
            dn = self.s.mem.g_cnt - g_before[1]
            entry = {"file": rec["file"],
                     "when": time.strftime("%Y-%m-%d %H:%M"),
                     "tokens": rec["tokens"],
                     # the free signal: mean surprise over what was
                     # written, in nats. This is the FROZEN model's
                     # surprise only -- it is computed before the memory
                     # speaks, so it says "unusual prose", never "new to
                     # this memory".
                     "salience": round(dg / max(1, dn), 3),
                     "reread": path in self.seen}
            self.journal.append(entry)
            made.append(entry)
            self.seen[path] = now[path]
            if not self.quiet:
                verb = "reread" if entry["reread"] else "read"
                print(f"  {verb} {entry['file']} -- {entry['tokens']} "
                      f"tokens, salience {entry['salience']:.2f} nats",
                      flush=True)
        for p, meta in now.items():         # remember files seen but
            self.seen.setdefault(p, meta)   # unchanged
        if made:
            self.save()
        return made

    def digest(self, n=10):
        """What was new lately, most surprising first."""
        return sorted(self.journal[-200:],
                      key=lambda e: -e["salience"])[:n]


def watch(assistant, root, interval=60, once=False, exts=None,
          fast=True, quiet=False):
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        raise SystemExit(f"not a folder: {root}")
    w = Watcher(assistant, root, exts=exts, fast=fast, quiet=quiet)
    cadence = "one pass" if once else f"every {interval}s"
    print(f"watching {w.root} ({cadence})")
    first = True
    try:
        while True:
            todo, _ = w.changed()
            if todo and not quiet:
                print(f"{len(todo)} file(s) changed", flush=True)
            made = w.pass_once()
            if first and not made and not quiet:
                print("  nothing new since last time", flush=True)
            first = False
            if made:
                print("\n  salience journal -- what was new, most "
                      "surprising first:")
                for e in w.digest(5):
                    print(f"    {e['salience']:5.2f} nats  "
                          f"{e['file']}  ({e['when']})")
                print()
            if once:
                break
            time.sleep(max(5, interval))
    except KeyboardInterrupt:
        print("\nstopping; state saved.", flush=True)
    finally:
        w.save()
        assistant.save()
    return w
