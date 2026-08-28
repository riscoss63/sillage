"""Static checker for the seven papers -- catches what would break pdfLaTeX
before you compile them.

Checks per file:
  * balanced braces
  * every \\begin{env} matched by \\end{env}
  * display-math delimiters \\[ ... \\] balanced (a stray \\[ is the classic
    "runaway math" error)
  * $...$ inline math delimiters even in number
  * macros used but never defined (beyond a known LaTeX/package set)
  * \\cite{key} without a matching \\bibitem{key}
  * \\ref{label} without a matching \\label{label}
  * \\includegraphics files that do not exist on disk

Comments are stripped first (an unescaped % ends the line), so a stray $ or
brace inside a comment cannot fail the counts; \\{ and \\} literals are
recognized as escaped on both sides.

    python papers/check_tex.py
"""

import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(HERE))
BS = chr(92)

PAPERS = ["papers/sillage/sillage.tex", "papers/router/router.tex",
          "papers/hierarchy/hierarchy.tex",
          "papers/fastweights/fastweights.tex",
          "papers/drafter/drafter.tex",
          "papers/behavior/behavior.tex",
          "papers/benchmark/benchmark.tex"]

KNOWN = set("""documentclass usepackage newcommand newtheorem definecolor
title author date maketitle begin end section subsection paragraph item
textbf textit emph texttt textsc small footnotesize normalsize bfseries
label ref cite includegraphics caption centering appendix bibitem
href url footnote qquad quad ldots dots cdot times ge le approx to pm
sum max min ln log exp sqrt frac mathbb mathbf mathrm mathcal langle rangle
Vert vert left right top varphi phi eta lambda beta alpha mu sigma theta
gamma delta epsilon varepsilon leftarrow rightarrow mapsto in notin subset
Delta cdots eqref mid ne nu odot otimes propto rho sim tau tilde verb
pi bar Vert vert n r cup cap emptyset forall exists
toprule midrule bottomrule multicolumn hline linewidth textwidth
sillage dnll ci big Big bigl bigr Bigl Bigr text mathopen mathclose
leftmargin itemsep noindent par vspace hspace S nobreakspace and
proof qed square blacksquare colon prime approxeq neq geq leq ll gg
""".split())


def strip_comments(src):
    """Drop % comments, line by line. A % preceded by an even number of
    backslashes starts a comment; a literal \\% stays. Line structure is
    preserved so reported line numbers stay exact."""
    out = []
    for line in src.split(chr(10)):
        i = 0
        while True:
            k = line.find("%", i)
            if k < 0:
                out.append(line)
                break
            if not _escaped(line, k):
                out.append(line[:k])
                break
            i = k + 1
    return chr(10).join(out)


def _escaped(s, i):
    """True when the character at index i sits behind an ODD number of
    backslashes (i.e. it is escaped)."""
    nb = 0
    j = i - 1
    while j >= 0 and s[j] == BS:
        nb += 1
        j -= 1
    return nb % 2 == 1


def check(path):
    src = strip_comments(io.open(path, encoding="utf-8").read())
    name = os.path.basename(path)
    errors, warnings = [], []

    # braces -- \{ and \} literals are escaped on BOTH sides
    depth, line = 0, 1
    for i, ch in enumerate(src):
        if ch == chr(10):
            line += 1
        elif ch == "{" and not _escaped(src, i):
            depth += 1
        elif ch == "}" and not _escaped(src, i):
            depth -= 1
            if depth < 0:
                errors.append(f"unbalanced closing brace near line {line}")
                depth = 0
    if depth:
        errors.append(f"{depth} unclosed brace(s)")

    # environments
    begins = re.findall(BS + BS + r"begin\{([a-zA-Z*]+)\}", src)
    ends = re.findall(BS + BS + r"end\{([a-zA-Z*]+)\}", src)
    for env in set(begins) | set(ends):
        if begins.count(env) != ends.count(env):
            errors.append(f"environment '{env}': {begins.count(env)} begin, "
                          f"{ends.count(env)} end")

    # display math delimiters -- a doubled backslash before '[' is a line
    # break with optional spacing (\\[2pt]), not display math
    disp_open = "(?<!" + re.escape(BS) + ")" + re.escape(BS) + re.escape("[")
    disp_close = "(?<!" + re.escape(BS) + ")" + re.escape(BS) + re.escape("]")
    opens = len(re.findall(disp_open, src))
    closes = len(re.findall(disp_close, src))
    if opens != closes:
        lines = [str(src[:m.start()].count(chr(10)) + 1)
                 for m in re.finditer(disp_open, src)]
        errors.append(f"display math unbalanced: {opens} '{BS}[' vs "
                      f"{closes} '{BS}]' ('{BS}[' at line(s) "
                      f"{', '.join(lines) or 'none'})")

    # inline math
    dollars = len(re.findall(r"(?<!" + re.escape(BS) + r")\$", src))
    if dollars % 2:
        errors.append(f"odd number of '$' ({dollars}) -- unclosed inline math")

    # macros used but never defined
    defined = set(re.findall(BS + BS + r"newcommand\{" + BS + BS +
                             r"([a-zA-Z]+)\}", src))
    used = set(re.findall(BS + BS + r"([a-zA-Z]+)", src))
    unknown = sorted(used - KNOWN - defined)
    for u in unknown:
        warnings.append(f"macro '{BS}{u}' not in the known list "
                        f"(fine if a package provides it)")

    # citations and references
    keys = set(re.findall(BS + BS + r"bibitem\{([^}]*)\}", src))
    cited = set()
    for grp in re.findall(BS + BS + r"cite[a-z]*\{([^}]*)\}", src):
        cited |= {k.strip() for k in grp.split(",")}
    for k in sorted(cited - keys):
        errors.append(f"\\cite{{{k}}} has no \\bibitem")
    labels = set(re.findall(BS + BS + r"label\{([^}]*)\}", src))
    refs = set(re.findall(BS + BS + r"ref\{([^}]*)\}", src))
    for r in sorted(refs - labels):
        errors.append(f"\\ref{{{r}}} has no \\label")

    # figures
    folder = os.path.dirname(path)
    for g in re.findall(BS + BS + r"includegraphics(?:\[[^\]]*\])?\{([^}]*)\}",
                        src):
        cand = [os.path.join(folder, g), os.path.join(folder, g + ".pdf"),
                os.path.join(folder, g + ".png")]
        if not any(os.path.exists(c) for c in cand):
            errors.append(f"missing figure: {g}")

    return name, errors, warnings, len(keys), len(cited)


def main():
    bad = 0
    for p in PAPERS:
        name, errors, warnings, nkeys, ncited = check(p)
        status = "OK " if not errors else "FAIL"
        print(f"[{status}] {name}: {nkeys} bibitems, {ncited} keys cited, "
              f"{len(errors)} errors, {len(warnings)} warnings")
        for e in errors:
            print("    ERROR   " + e)
        for w in warnings:
            print("    warning " + w)
        bad += len(errors)
    print(("all seven papers pass" if not bad
           else f"{bad} error(s) to fix before compiling"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
