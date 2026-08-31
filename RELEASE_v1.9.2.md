# v1.9.2 — the capacity law, and the dial it hands you

*"7 MB forever"* has been the headline since 1.0. It never said **forever of
what**. This release measures the answer, and turns the constant that decides
it into a setting.

## The law

**≈ 40 pages of durable memory per megabyte of exact store.** From the save
format: 16 bytes of key, 4 of surprise mass, 8 of CSR offset, 12 per
successor — about 42 bytes a gram, so the shipped 50,000-gram cap costs
~2.1 MB.

**Recall does not degrade.** Facts planted at every depth of a growing corpus
and probed from 5,850 to 1,000,107 tokens — **171× of scale**:

| tokens | grams | oldest cohort | newest | locality |
|---|---|---|---|---|
| 5,850 | 1,180 | 100 % | 100 % | +0.0000 |
| 100,735 | 9,417 | 100 % | 100 % | +0.0000 |
| 400,275 | 31,508 | 100 % | 83 % | +0.0000 |
| **1,000,107** | **63,898** | **100 %** | **100 %** | **+0.0000** |

Flat. The scattered 83 % are one fact in six, with no trend. And the
perturbation on a document the memory never read is **exactly zero** — 1.74e-07,
identical to seven digits, at every scale. It does not bleed as it fills.
([capacity.json](results/capacity.json))

**What fills it is not tokens — it is *repeated* 4-grams.** Real prose repeats
only ~6 % of its own, and the store admits a gram at two occurrences. So the
same document costs wildly different amounts depending on how you read it:

| | store fills after | retains |
|---|---|---|
| read **once** | ~890,000 tokens | almost nothing |
| read **twice** | ~**56,000 tokens** | everything distinct |

The second figure is stable within 5 % across ten independent texts. Reading
twice is exactly what paper 6 tells you to do — and it fills the store sixteen
times sooner. **Durability and capacity are the same budget**, and that price
was written nowhere. ([gramrate.json](results/gramrate.json))

**When it overflows, it forgets the ordinary and keeps the remarkable.**
Measured head-on for the first time, with the cap lowered so saturation takes
minutes: 4,865 grams pruned to 3,000 — **38 % of the store dropped, and 100 %
of 294 planted facts survived, with recall unchanged**. Keeping the highest
surprise mass protects exactly what matters, and what it drops is mostly the
grams seen once, which `COLD_MIN_COUNT` already made unretrievable.
([eviction.json](results/eviction.json))

## Two fixes that follow from it

**`--cold-max N` — the capacity dial.** `COLD_MAX` was a module constant no
user could reach. It is now per-memory, and lowering it on an existing state
prunes *immediately*, saying how many low-surprise grams went, instead of
waiting for the next `save()`.

**The cap now holds during ingestion.** Eviction lived only in `save()`, and
the fast path does not even go through `write_all`, so a long ingest ran past
the cap — measured at **5,972 grams against a cap of 2,000, or 2.99×**. Both
paths prune on a margin of 1.25 now:

```
cap 2000 | peak in memory 2497 (1.25x) | after save 2000
recall: oldest 100%, newest 100%, across 372 planted facts
```

Continuous pruning costs nothing measurable — the store is held to a third of
what it would otherwise reach and recall does not move.

## What was falsified, and two probe errors

**C1 falsified** and it was the defect above: the store reached 63,898 grams
against a 50,000 cap. **E4 falsified** on a technicality — the median surprise
mass does not rise after pruning, because the distribution has a large atom at
low values, though q90 does.

Recorded rather than hidden: the first entity generator gave every invented
name the same final syllable, so all probes collided on **one** cold gram with
six rival successors and recall read 0/6 while the mechanism was fine — names
are now filtered against the tokenizer for a unique 4-gram. And the "filler"
control is unusable as designed: on an **empty** memory it already scores
100 %, because GPT-2 predicts its own grammar. C4 is therefore measured inside
the store rather than by generation.

One contract changed deliberately: T6 patched `core.COLD_MAX` after
construction and called `save()`, asserting the cap was read at write time. It
is not any more — that is what makes `--cold-max` possible — so the test checks
the same property under the new contract, and T21 covers the dial.

**Tests**: 22 unit + 14 end-to-end + 16 over HTTP + 28 for the axis-4
commands = **80**, all green.

`pip install -U sillage`
