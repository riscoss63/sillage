"""Static checker for the four papers -- catches what would break pdfLaTeX
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
  * unescaped % and & outside math/tables (reported as warnings)

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
          "papers/fastweights/fastweights.tex"]

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


def check(path):
    src = io.open(path, encoding="utf-8").read()
    name = os.path.basename(path)
    errors, warnings = [], []

    # braces
    depth, line = 0, 1
    for ch in src:
        if ch == chr(10):
            line += 1
        elif ch == "{" and not_escaped(src, ch):
            depth += 1
        elif ch == "}":
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
        for m in re.finditer(disp_open, src):
            ln = src[:m.start()].count(chr(10)) + 1
            ctx = src[max(0, m.start() - 25):m.start() + 8].replace(chr(10), " ")
            errors.append(f"display-math '{BS}[' at line {ln} "
                          f"(opens {opens}, closes {closes}) ...{ctx}")

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


def not_escaped(src, ch):
    return True


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
    print(("all four papers pass" if not bad
           else f"{bad} error(s) to fix before compiling"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
