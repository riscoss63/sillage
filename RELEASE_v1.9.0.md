# v1.9.0 — the inventions were a silence, not a hallucination

`complete` recalled a fact perfectly from one phrasing and fabricated from
another. Four hypotheses were registered with falsification thresholds and
measured in order. **Three died**, and the one that survived changes what the
failure *is*.

## What it turned out to be

Both fast tiers key on the last four **tokens**, and a line break is absorbed
*into* one. A document that wraps

```
... par le technicien responsable,
madame Brindas Kolvec, matricule 4471.
```

stores the key `[' responsable', ',\n', 'mad', 'ame']`, while the same
sentence typed on one line asks for `[' responsable', ',', ' mad', 'ame']`.
The store misses, **both tiers stay silent**, and the frozen model fills the
silence with a plausible French name. `Brigitte Lefevre` was never the
memory hallucinating — the memory said nothing at all.

`read --reflow` rejoins the lines inside each paragraph before reading:
**7/8 → 8/8 facts recalled**, questions typed the way a person types them
([results/reflow.json](results/reflow.json)). It is opt-in, because the token
stream changes and a reflowed read's perplexity is not comparable to the
published numbers (frozen 22.79 → 24.66). The v1 semantic tier is what
carries that recall: with it off it is 7/8 either way — the cold store opens
the name, the tier finishes it.

## What was refuted

- **The readout constants are not the lock.** Paper 5's family settings
  (`40,0.85,0.5`) turn 10 % conflict conversion into 100 % on the paper's own
  synthetic protocol, so they looked like the answer. On an ordinary French
  report they buy **nothing** at 0.6B — 88 % both ways — while perturbation on
  a document the memory never read goes **+0.16 → +2.14 nats**. At 1.7B it
  reverses: the published readout is too quiet (75 %) and family recovers the
  13 points at +1.25. Either way they make the memory speak **3–13× more on
  questions it cannot answer**. `published` stays the default; `--readout`
  exposes the dial so the trade-off is yours.
- **Line wrapping alone does not explain it** — verbatim 6/8 = rewrapped 6/8.
- **It is not an arbitration failure.** The missing 4-gram is *absent* from the
  cold store, and raising `LAM_C` from 0.3 to 0.9 converts nothing and loses
  nothing.

## Two corrections to my own measurements

- **My locality-witness pass was contaminating the recall arms.** It folds
  hidden states into the v1 tier's running centre *without* the matching tier
  writes — an imbalance only probe code produces — and measured that way it
  cost this document a fact and falsified `--reflow` wrongly. Bisected at
  identical thresholds and reservoir, then diffed attribute by attribute.
- **Reading more documents does *not* degrade recall.** The legitimate worry
  after watching the centre drift. Three unrelated documents later
  (`mu_n` 826 → 1320, cold grams 395 → 879): **8/8 at every checkpoint**.

## New: the memory says what it contributed

`complete` now reports, on **stderr** (stdout stays the byte-identical text
paper 5 guarantees), what the memory actually did:

```
[memory moved 9/12 tokens; tiers spoke: cold 8, ngram 7]
```

and warns when it moved fewer than three. On two corpora and 36 questions
that caught **11 of the 12 questions the documents could not answer**, while
no correct answer ever moved fewer than 12. On a fresh corpus:
7/8 unanswerable correctly refused, 7/8 verbatim answered with **zero wrong
answers**, and both reworded questions it chose to answer were right.

`sem_key(learn=False)` at generation makes `complete`'s own docstring promise
— "Writes nothing" — true; it used to move the centre on every generated
token. Measured as hygiene, not gain: all eight answers byte-identical.

## Left open, and now bounded

The residual failure is the **transplant**, and three candidate guards were
registered and all three failed. The same completion is the correct answer to
one question and a fabrication for another:

| question | output | moved | TF-IDF | verbatim | right? |
|---|---|---|---|---|---|
| « La visite **de printemps**… s'est déroulée le » | `11 avril 2026, par temps couvert` | 16/30 | 0.623 | yes | **yes** |
| « La **prochaine** visite… aura lieu le » | `11 avril 2026, par temps couvert` | 16/30 | 0.554 | yes | **no** |

Correct answers span 0.245–0.623, so the lexical score sits inside the range.
The two channels cannot audit each other because **both are surface matchers
and they fail together**. Only the word *prochaine* separates the questions,
and it is nowhere in the document. That is a boundary, not a bug — and the
class only reaches questions whose wording nearly covers a stored passage.

**Tests**: 20 unit + 14 end-to-end + 16 over HTTP + 28 for the axis-4
commands = **78**, all green. Nine new probes under `behav/`, eleven result
files under `results/`, and every prediction with its verdict in
`behav/JOURNAL.md` — including the three refutations and the two corrections.

`pip install -U sillage`
