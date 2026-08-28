"""Grounded retrieval over everything the assistant has read.

Two layers answer questions in this tool, and the difference matters:

    sillage ask       lexical retrieval (TF-IDF) over the paragraphs of your
                      documents -- this module. Instant, exact, always says
                      which file and section a passage came from. Nothing is
                      generated, so there is no hallucination surface: this
                      is the layer you should trust.
    sillage complete  continuation by the frozen model plus the memory (see
                      `sillage.runtime`), also reachable as /say inside
                      `sillage chat`. Useful to watch the memory work; never
                      a source of truth at 0.1-0.6B parameters.

Text extraction handles .txt, .md and .tex (LaTeX is stripped to readable
prose, keeping section structure, so `sillage read paper.tex` does the right
thing).
"""

import json
import math
import os
import re
import sys
from collections import Counter

MIN_CHARS = 140
STOP = set("""a an the of to in and or is are was were be been being for on
with as by at from that this these those it its we our us they their he she
which who whom what when where how why not no nor but if then than so such
can could may might will would shall should must do does did done have has
had having i you your yours also very more most much many few less least each
every all any some other another same own into over under between within
without across per via out up down off about above below only just even still
here there both either neither because while during before after again once
le la les un une des du de au aux et ou ni mais donc car que qui quoi dont
ce cet cette ces son sa ses leur leurs mon ma mes ton ta tes notre nos votre
vos il elle ils elles nous vous je tu on se me te lui y en est sont etait
ete etre avoir fait plus moins tres peu tout tous toute toutes meme aussi
pour par dans sur sous avec sans entre vers chez comme si alors pas ne
""".split())


# --------------------------------------------------------------- LaTeX ------

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


def read_text(path):
    """Extract readable text from a document (LaTeX gets stripped)."""
    raw = open(path, encoding="utf-8", errors="replace").read()
    if path.lower().endswith(".tex"):
        return strip_latex(raw)
    return raw


def blocks(text):
    """(section, paragraph) pairs, from LaTeX marks or markdown headings."""
    latex = "@@SEC@@" in text
    section = "Abstract" if latex else "Document"
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if latex:
            if block.startswith("@@SEC@@"):
                section = block.replace("@@SEC@@", "").strip()
                continue
        else:
            head = re.match(r"^#{1,6}\s+(.+)$", block.split("\n")[0])
            if head:
                section = head.group(1).strip()
                block = "\n".join(block.split("\n")[1:]).strip()
                if not block:
                    continue
        yield section, re.sub(r"\s+", " ", block)


def paragraphs(text, source):
    """Indexable passages: short blocks are merged, never dropped.

    A one-line fact in a notes file is exactly the kind of thing you want
    back verbatim, so consecutive short paragraphs of the same section are
    grouped up to MIN_CHARS instead of being thrown away.
    """
    out, buf, buf_sec = [], [], None

    def flush():
        if buf:
            joined = " ".join(buf)
            if len(joined) >= 40:
                out.append({"source": source, "section": buf_sec,
                            "text": joined})
        del buf[:]

    for section, block in blocks(text):
        if buf and section != buf_sec:
            flush()
        if not buf:
            buf_sec = section
        buf.append(block)
        if sum(len(b) + 1 for b in buf) >= MIN_CHARS:
            flush()
    flush()
    return out


# ------------------------------------------------------------- retrieval ----

def tokens(s):
    """Words and numbers. Unicode-aware, so accented text indexes properly."""
    return [w for w in re.findall(r"[^\W\d_][\w\-]*|\d+\.?\d*", s.lower())
            if w not in STOP and len(w) > 1]


class Index:
    """A small TF-IDF index that grows as documents are read."""

    def __init__(self, path=None):
        # `path` names the JSON file (an index is data, not code: before
        # 1.5 it was a pickle, which executes code when opened). Only the
        # passages are stored -- the TF-IDF vectors are derived from them,
        # so rebuilding costs milliseconds and cannot drift.
        self.path = None if path is None else (
            path[:-4] + ".json" if path.endswith(".pkl") else path)
        self.passages = []
        self.vecs = []
        self.idf = {}
        old = (None if self.path is None
               else self.path[:-5] + ".pkl")
        if self.path and os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.passages = json.load(f)["passages"]
            self._rebuild()
        elif old and os.path.exists(old):
            print(f"note: migrating the index from the pre-1.5 pickle "
                  f"format ({os.path.basename(old)}). Unpickling "
                  f"executes code -- only migrate states you created "
                  f"yourself.", file=sys.stderr, flush=True)
            if os.environ.get("SILLAGE_NO_PICKLE"):
                raise SystemExit(f"{old} is a pre-1.5 pickle and "
                                 f"SILLAGE_NO_PICKLE is set.")
            import pickle
            with open(old, "rb") as f:
                self.passages = pickle.load(f)["passages"]
            self._rebuild()
            self.save()                      # rewrites JSON, drops .pkl

    def add(self, text, source):
        """Index one document, replacing any earlier version of it."""
        self.passages = [p for p in self.passages if p["source"] != source]
        got = paragraphs(text, source)
        self.passages += got
        self._rebuild()
        return len(got)

    def _rebuild(self):
        df = Counter()
        docs = []
        for p in self.passages:
            tf = Counter(tokens(p["text"]))
            docs.append(tf)
            df.update(tf.keys())
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
        self.vecs = []
        for tf in docs:
            v = {w: (1 + math.log(c)) * self.idf.get(w, 0.0)
                 for w, c in tf.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            self.vecs.append({w: x / norm for w, x in v.items()})

    def search(self, query, k=3, numeric_only=False):
        """Best k passages for a query, as (score, passage) pairs."""
        q = Counter(tokens(query))
        qv = {w: (1 + math.log(c)) * self.idf.get(w, 0.0)
              for w, c in q.items()}
        norm = math.sqrt(sum(x * x for x in qv.values())) or 1.0
        qv = {w: x / norm for w, x in qv.items()}
        scored = []
        for i, v in enumerate(self.vecs):
            if numeric_only and not re.search(r"\d\.\d{2,}",
                                              self.passages[i]["text"]):
                continue
            s = sum(x * v.get(w, 0.0) for w, x in qv.items())
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [(s, self.passages[i]) for s, i in scored[:k]]

    def sources(self):
        """How many passages came from each document."""
        c = Counter(p["source"] for p in self.passages)
        return c

    def save(self):
        """Persist the index next to the memory it belongs to."""
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"passages": self.passages}, f, ensure_ascii=False)
        old = self.path[:-5] + ".pkl"        # a migrated index keeps
        if os.path.exists(old):              # no pickle behind it
            os.remove(old)


def show(hits, width=100):
    """Print retrieved passages with their provenance."""
    if not hits:
        print("  (nothing matched -- try other words, or `sillage status` "
              "to see what has been read)")
        return
    for rank, (score, p) in enumerate(hits, 1):
        print(f"\n[{rank}] {p['source']} - {p['section']}   "
              f"(relevance {score:.3f})")
        line = ""
        for word in p["text"].split():
            if len(line) + len(word) + 1 > width:
                print("    " + line)
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            print("    " + line)
