"""`sillage watch`: read a folder as it changes, and say what was new.

Point it at a notes folder (an Obsidian vault, a directory of Markdown)
and it reads what changed since last time -- nothing else. Two things
make this worth having, and both come out of the papers rather than
from a product wish:

  * The SALIENCE JOURNAL. Every write is scaled by the frozen model's
    own surprise, a scalar it computes anyway at inference. Averaged per
    document, that is a free answer to "what did I actually write that
    was new this week?" -- the question note-taking apps answer by
    paying an LLM call per decision. Here it costs nothing extra: the
    number was already on its way through.
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
    def __init__(self, assistant, root, exts=(".md", ".txt", ".markdown"),
                 fast=True, quiet=False):
        self.s = assistant
        self.root = os.path.abspath(root)
        self.exts = {e.lower() for e in exts} if exts else None
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
            rec = self.s.read(path, fast=self.fast)[0]
            dg = self.s.mem.g_sum - g_before[0]
            dn = self.s.mem.g_cnt - g_before[1]
            entry = {"file": os.path.relpath(path, self.root),
                     "when": time.strftime("%Y-%m-%d %H:%M"),
                     "tokens": rec["tokens"],
                     # the free signal: mean surprise over what was
                     # written, in nats. High = this was new to the model
                     # AND to the memory.
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
    w = Watcher(assistant, root, exts=exts, fast=fast, quiet=quiet)
    if not os.path.isdir(root):
        raise SystemExit(f"not a folder: {root}")
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
