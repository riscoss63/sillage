"""Extract clean titles and abstracts from the four papers, for Zenodo /
arXiv metadata forms. Writes papers/_metadata.txt (plain text, ready to
paste). Uses the same LaTeX cleaner as papers_assistant.py.

    python papers/extract_metadata.py
"""

import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from papers_assistant import strip_latex  # noqa: E402

PAPERS = [("Sillage", "papers/sillage/sillage.tex"),
          ("Router", "papers/router/router.tex"),
          ("Hierarchy", "papers/hierarchy/hierarchy.tex"),
          ("FastWeights", "papers/fastweights/fastweights.tex")]


def main():
    chunks = []
    for name, path in PAPERS:
        tex = io.open(path, encoding="utf-8").read().replace("\r\n", "\n")
        m = re.search(re.escape("\\title{") + r"(.*?)\}\s*\n" +
                      re.escape("\\author"), tex, re.S)
        title = m.group(1) if m else ""
        title = re.sub(re.escape("\\bfseries") + r"|" +
                       re.escape("\\textsc{") + r"|\}|" +
                       re.escape("\\\\"), " ", title)
        title = re.sub(r"\s+", " ", title).strip()
        title = title.replace(" :", ":")
        a = re.search(re.escape("\\begin{abstract}") + r"(.*?)" +
                      re.escape("\\end{abstract}"), tex, re.S)
        body = strip_latex("\\begin{document}" + a.group(1) +
                           "\\end{document}") if a else ""
        body = re.sub(r"\s+", " ", body).strip()
        chunks.append(f"### {name}\nTITLE:\n{title}\n\n"
                      f"ABSTRACT ({len(body.split())} words):\n{body}\n")
        print(f"{name:12s} title ok, abstract {len(body.split())} words")
    out = os.path.join("papers", "_metadata.txt")
    io.open(out, "w", encoding="utf-8").write("\n".join(chunks))
    print("written", out)


if __name__ == "__main__":
    main()
