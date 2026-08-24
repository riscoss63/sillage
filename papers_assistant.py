"""Papers assistant — the four preprints, queryable offline.

Loads the four papers of this project (Sillage, Router, Hierarchy, Fast Weights)
and answers with GROUNDED excerpts: every answer is a passage from a paper,
with its paper, section and a relevance score. No hallucination surface,
because nothing is generated in `ask`.

Two layers, and the difference matters:
  ask     lexical retrieval over the papers' paragraphs (TF-IDF).
          Instant, exact, always cites where the text came from. This is
          the layer you should trust.
  say     continuation by a frozen LM augmented with the Sillage memory of the
          papers (`build --with-memory` first). A 0.1-0.6B model writes
          plausible-looking prose: useful for phrasing and for seeing the
          memory work, NOT a source of truth.

Commands
  python papers_assistant.py build [--with-memory] [--model gpt2|qwen]
  python papers_assistant.py ask "how big is the adapter?" [-k 3]
  python papers_assistant.py numbers "additivity"        # numeric claims only
  python papers_assistant.py sections [paper]
  python papers_assistant.py say "The delta rule" [-n 40]
  python papers_assistant.py chat                        # interactive
  python papers_assistant.py stats
"""


# --- repo bootstrap (added by reorganize.py) ---
import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "requirements.txt")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
for _sub in ("", "pipeline", "memory", "fastweights", "eval", "figures"):
    _p = _os.path.join(_d, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_d)
# --- end bootstrap ---

import argparse
import json
import math
import os
import pickle
import re
import sys
from collections import Counter

PAPERS = [
    ("Sillage", "Surprise-Gated Amplitude Memory",
     os.path.join("papers", "sillage", "sillage.tex")),
    ("Router", "Route the Scores, Not the Keys",
     os.path.join("papers", "router", "router.tex")),
    ("Hierarchy", "One Signal, Three Tiers",
     os.path.join("papers", "hierarchy", "hierarchy.tex")),
    ("FastWeights", "Memory Remembers, Fast Weights Adapt",
     os.path.join("papers", "fastweights", "fastweights.tex")),
]
STATE_DIR = "papers_state"
INDEX = os.path.join(STATE_DIR, "index.pkl")
CORPUS = os.path.join(STATE_DIR, "corpus.txt")
MIN_CHARS = 140
STOP = set("""a an the of to in and or is are was were be been being for on
with as by at from that this these those it its we our us they their he she
which who whom what when where how why not no nor but if then than so such
can could may might will would shall should must do does did done have has
had having i you your yours also very more most much many few less least each
every all any some other another same own into over under between within
without across per via out up down off about above below only just even still
here there both either neither because while during before after again once
""".split())

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------- LaTeX ----

def strip_latex(tex):
    """Turn a .tex source into readable text, keeping section structure."""
    body = tex.split(r"\begin{document}", 1)[-1]
    body = body.split(r"\end{document}", 1)[0]
    body = re.sub(r"(?<!\\)%.*", "", body)
    body = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
                  "", body, flags=re.S)
    body = re.sub(r"\\(section|subsection|paragraph)\*?\{([^}]*)\}",
                  r"\n\n@@SEC@@ \2\n\n", body)
    body = re.sub(r"\\caption\{", "Figure/Table: ", body)
    body = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?", " ", body)
    body = re.sub(r"\\cite\w*\s*(\[[^\]]*\])?\{[^}]*\}", "[ref]", body)
    body = re.sub(r"\\(ref|label|includegraphics)\w*\s*(\[[^\]]*\])?"
                  r"\{[^}]*\}", " ", body)
    body = re.sub(r"\\(textbf|emph|textit|texttt|sillage|mathbf|mathrm)\{",
                  "", body)
    body = re.sub(r"\\ci\{([^}]*)\}\{([^}]*)\}", r"[\1, \2]", body)
    body = re.sub(r"\\dnll\b", "dNLL", body)
    for cmd, sym in (("sillage", "Sillage"), ("ln", "ln"), ("log", "log"),
                     ("exp", "exp"), ("sqrt", "sqrt"), ("max", "max"),
                     ("min", "min"), ("alpha", "alpha"), ("beta", "beta"),
                     ("lambda", "lambda"), ("mu", "mu"), ("sigma", "sigma"),
                     ("theta", "theta"), ("gamma", "gamma"), ("eta", "eta"),
                     ("times", "x"), ("ge", ">="), ("le", "<="),
                     ("approx", "~"), ("rightarrow", "->"), ("to", "->"),
                     ("pm", "+/-"), ("varphi", "phi"), ("ell", "l"),
                     ("Vert", "||"), ("langle", "<"), ("rangle", ">"),
                     ("leftarrow", "<-"), ("cdot", "."), ("dots", "..."),
                     ("ldots", "..."), ("mathcal", ""), ("top", "T")):
        body = re.sub(r"\\" + cmd + r"(?![a-zA-Z])", sym, body)
    body = body.replace(r"\%", "%").replace(r"\,", " ").replace(r"\;", " ")
    body = body.replace(r"\_", "_").replace(r"\&", "&")
    body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
    body = body.replace("{", " ").replace("}", " ").replace("&", " | ")
    body = body.replace("\\\\", " ").replace("$", "").replace("~", " ")
    body = body.replace("--", "-").replace("``", '"').replace("''", '"')
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def paragraphs(text, paper, title):
    out, section = [], "Abstract"
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("@@SEC@@"):
            section = block.replace("@@SEC@@", "").strip()
            continue
        block = re.sub(r"\s+", " ", block)
        if len(block) < MIN_CHARS:
            continue
        out.append({"paper": paper, "title": title, "section": section,
                    "text": block})
    return out


# ------------------------------------------------------------- retrieval ----

def tokens(s):
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]+|\d+\.?\d*", s.lower())
            if w not in STOP and len(w) > 1]


def build_index(passages):
    df = Counter()
    docs = []
    for p in passages:
        tf = Counter(tokens(p["text"]))
        docs.append(tf)
        df.update(tf.keys())
    N = len(docs)
    idf = {w: math.log((N + 1) / (c + 0.5)) for w, c in df.items()}
    vecs = []
    for tf in docs:
        v = {w: (1 + math.log(c)) * idf.get(w, 0.0) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return {"passages": passages, "vecs": vecs, "idf": idf}


def search(ix, query, k=3, numeric_only=False):
    q = Counter(tokens(query))
    qv = {w: (1 + math.log(c)) * ix["idf"].get(w, 0.0) for w, c in q.items()}
    norm = math.sqrt(sum(x * x for x in qv.values())) or 1.0
    qv = {w: x / norm for w, x in qv.items()}
    scored = []
    for i, v in enumerate(ix["vecs"]):
        if numeric_only and not re.search(r"\d\.\d{2,}", ix["passages"][i]["text"]):
            continue
        s = sum(x * v.get(w, 0.0) for w, x in qv.items())
        if s > 0:
            scored.append((s, i))
    scored.sort(reverse=True)
    return [(s, ix["passages"][i]) for s, i in scored[:k]]


def show(hits, width=100):
    if not hits:
        print("  (nothing matched -- try other words, or `sections` to see "
              "what is covered)")
        return
    for rank, (score, p) in enumerate(hits, 1):
        print(f"\n[{rank}] {p['paper']} · {p['section']}   "
              f"(relevance {score:.3f})")
        text = p["text"]
        line = ""
        for word in text.split():
            if len(line) + len(word) + 1 > width:
                print("    " + line)
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            print("    " + line)


# ------------------------------------------------------------- commands ----

def cmd_build(args):
    os.makedirs(STATE_DIR, exist_ok=True)
    passages, missing, corpus = [], [], []
    for paper, title, path in PAPERS:
        if not os.path.exists(path):
            missing.append(path)
            continue
        text = strip_latex(open(path, encoding="utf-8").read())
        corpus.append(f"# {title}\n\n{text}")
        got = paragraphs(text, paper, title)
        passages += got
        print(f"  {paper:12s} {len(got):4d} passages   ({path})")
    if missing:
        print("  missing:", ", ".join(missing))
    if not passages:
        print("no papers found -- nothing built.")
        return
    ix = build_index(passages)
    with open(INDEX, "wb") as f:
        pickle.dump(ix, f)
    with open(CORPUS, "w", encoding="utf-8") as f:
        f.write("\n\n".join(corpus))
    print(f"index built: {len(passages)} passages, "
          f"{len(ix['idf'])} vocabulary terms -> {INDEX}")
    if args.with_memory:
        print("\nreading the papers into the Sillage memory "
              f"({args.model}) -- this takes a few minutes ...")
        import subprocess
        r = subprocess.run(
            [sys.executable, "assistant.py", "read", CORPUS,
             "--model", args.model, "--state",
             os.path.join(STATE_DIR, "memory")],
            text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0:
            print("memory built -- `say` is now memory-augmented.")


def load_index():
    if not os.path.exists(INDEX):
        print("no index yet -- run: python papers_assistant.py build")
        sys.exit(1)
    with open(INDEX, "rb") as f:
        return pickle.load(f)


def cmd_ask(args):
    ix = load_index()
    show(search(ix, " ".join(args.args), k=args.k))


def cmd_numbers(args):
    ix = load_index()
    show(search(ix, " ".join(args.args), k=args.k, numeric_only=True))


def cmd_sections(args):
    ix = load_index()
    want = args.args[0].lower() if args.args else None
    seen = {}
    for p in ix["passages"]:
        if want and want not in p["paper"].lower():
            continue
        seen.setdefault(p["paper"], []).append(p["section"])
    for paper, secs in seen.items():
        uniq = list(dict.fromkeys(secs))
        print(f"\n{paper} — {len(secs)} passages")
        for s in uniq:
            print(f"    {s}  ({secs.count(s)})")


def cmd_say(args):
    import subprocess
    state = os.path.join(STATE_DIR, "memory")
    if not os.path.exists(os.path.join(state, "state.npz")):
        print("(no paper memory yet: run `build --with-memory` for the "
              "memory-augmented version; using the frozen model alone)\n")
    cmd = [sys.executable, "assistant.py", "complete", " ".join(args.args),
           "-n", str(args.n), "--model", args.model, "--state", state]
    subprocess.run(cmd, text=True)


def cmd_stats(args):
    ix = load_index()
    per = Counter(p["paper"] for p in ix["passages"])
    chars = sum(len(p["text"]) for p in ix["passages"])
    print(f"papers indexed : {len(per)}")
    for paper, n in per.items():
        print(f"    {paper:12s} {n:4d} passages")
    print(f"passages       : {len(ix['passages'])}  ({chars/1000:.0f}k chars)")
    print(f"vocabulary     : {len(ix['idf'])} terms")
    mem = os.path.join(STATE_DIR, "memory", "state.npz")
    if os.path.exists(mem):
        size = os.path.getsize(mem) / 1e6
        print(f"paper memory   : present ({size:.1f} MB) -- `say` is "
              f"memory-augmented")
    else:
        print("paper memory   : absent (run build --with-memory)")


def cmd_chat(args):
    ix = load_index()
    print("Papers assistant. Type a question for grounded excerpts.")
    print("  /say <prompt>   generate with the memory-augmented model")
    print("  /k <n>          number of excerpts (default 3)")
    print("  /quit\n")
    k = args.k
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
            continue
        if line.startswith("/say "):
            args.args = [line[5:]]
            cmd_say(args)
            continue
        show(search(ix, line, k=k))
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "ask", "numbers", "sections",
                                    "say", "chat", "stats"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--model", default="gpt2", choices=["gpt2", "qwen"])
    ap.add_argument("--with-memory", action="store_true")
    a = ap.parse_args()
    {"build": cmd_build, "ask": cmd_ask, "numbers": cmd_numbers,
     "sections": cmd_sections, "say": cmd_say, "chat": cmd_chat,
     "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
