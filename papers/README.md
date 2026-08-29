# The eight preprints

LaTeX sources and figures for the eight papers this repository implements.
Papers 1-6 are archived on Zenodo with permanent DOIs (7 and 8 are in
submission); each is self-contained (no paper depends on an unpublished
companion for its own results).

| # | file | title | DOI |
|---|---|---|---|
| 1 | [`sillage/sillage.tex`](sillage/sillage.tex) | Sillage: Surprise-Gated Amplitude Memory for Frozen Language Models | [10.5281/zenodo.22079016](https://doi.org/10.5281/zenodo.22079016) |
| 2 | [`router/router.tex`](router/router.tex) | Route the Scores, Not the Keys | [10.5281/zenodo.22079444](https://doi.org/10.5281/zenodo.22079444) |
| 3 | [`hierarchy/hierarchy.tex`](hierarchy/hierarchy.tex) | One Signal, Three Tiers | [10.5281/zenodo.22079471](https://doi.org/10.5281/zenodo.22079471) |
| 4 | [`fastweights/fastweights.tex`](fastweights/fastweights.tex) | Memory Remembers, Fast Weights Adapt | [10.5281/zenodo.22079481](https://doi.org/10.5281/zenodo.22079481) |
| 5 | [`drafter/drafter.tex`](drafter/drafter.tex) | The Memory Pays for Itself | [10.5281/zenodo.22109220](https://doi.org/10.5281/zenodo.22109220) |
| 6 | [`behavior/behavior.tex`](behavior/behavior.tex) | Stored Is Not Recalled | [10.5281/zenodo.22125859](https://doi.org/10.5281/zenodo.22125859) |
| 7 | [`benchmark/benchmark.tex`](benchmark/benchmark.tex) | Found Is Not Formulated | DOI pending |
| 8 | [`paraphrase/paraphrase.tex`](paraphrase/paraphrase.tex) | The Key Was in the Wrong Layer | DOI pending |

## Building a PDF

Each paper is a single self-contained `.tex` with its own `figs/` folder and
an inline bibliography — no `.bib`, no custom class, no external package
beyond a standard TeX distribution.

```bash
cd sillage && pdflatex sillage.tex && pdflatex sillage.tex
```

Twice, so that the cross-references resolve. Overleaf works too: upload the
paper's folder as-is.

## Before you compile

```bash
python check_tex.py
```

A static check of all eight sources: balanced braces and environments, display
math that opens and closes, `\cite` keys that have a `\bibitem` (and, as a
warning, `\bibitem` entries that nothing cites — pdfLaTeX still typesets
them, so a reader sees a numbered reference nothing points at), `\ref`
labels that exist, and `\includegraphics` files that are actually on disk. It
runs in CI on every push, because a single stray `\[` costs more time to
diagnose in a LaTeX log than it does to catch here.

## Reading them offline

The assistant in this repository indexes the papers and answers with
exact passages:

```bash
sillage papers                       # instant, no model needed
sillage ask "does rank 16 suffice?"
```

## Regenerating the figures

Each paper's figures come from one script, and each writes into that paper's
own `figs/` folder (see [`../REPRODUCE.md`](../REPRODUCE.md)):

```bash
python ../figures/make_figures.py      # paper 1
python ../figures/make_figures_p2.py   # paper 2
python ../figures/make_figures_p3.py   # paper 3
python ../figures/make_figures_p4.py   # paper 4
python ../figures/make_figures_p5.py   # paper 5
python ../figures/make_figures_p6.py   # paper 6
python ../figures/make_figures_p7.py   # paper 7
python ../figures/make_figures_p8.py   # paper 8
```
