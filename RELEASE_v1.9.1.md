# v1.9.1 — a bigger reader does not remember better, it lies less

Two small fixes were asked for. The second one found a crash that had been
shipping for eight minor versions.

## `complete --target` and `serve --target` crashed on the default state

Paper 5's headline is *read with the small model, remember with the bigger
sibling*. In the tool, it raised this from inside the decoding loop:

```
ValueError: operands could not be broadcast together with shapes (2048,) (1024,)
```

The v1 semantic tier centres its keys on `mu`, whose width belongs to the
model that **wrote** the state — 1024 for Qwen3-0.6B. A 1.7B reader hands it
2048. Present since **1.1.0**, on the **default** qwen state (the semantic
tier is on by default), and never caught, because every measurement of the
transfer built its state *with* the target model, where the widths agree.
The probes written for this very release did exactly that.

The fix silences what cannot be keyed and says so, instead of dying:

```
this reader's hidden states are 2048d, so paper 2's semantic tier
(built at 1024d) cannot be keyed and stay silent.
  The n-gram matrix and the cold store are keyed on tokens and
  transfer normally (paper 5).
```

That last sentence is the point: the tiers keyed on **tokens** — the n-gram
matrix and the cold store — transfer exactly as the paper claims, because
the family shares a tokenizer. The tiers keyed on **hidden states** cannot,
and now abstain rather than crash. Regression test T20 builds both widths
without loading any weights.

`serve` also gained `--target`, which it never accepted: the flag was
declared in the `gen` argument group that the serve subparser does not
inherit. Exposing it is what surfaced the crash.

## What that buys, measured on the real endpoint

Three arms over real sockets, `--no-context` as the ablation
([results](results/serve_rephrase.json)):

| | rephrased questions | refuses | fabricates |
|---|---|---|---|
| 0.6B, passages injected | **8/8** | 2/8 | 6/8 |
| 0.6B, `--no-context` | 0/8 | 1/8 | 7/8 |
| **1.7B** (`--target`), injected | 7/8 | **5/8** | **3/8** |

Two things worth reading twice. **Eight rephrased questions out of eight**,
in correct French, on a CPU, from a 0.6B — against **0/8 from the same model
with the passages withheld**. That is paper 7's split replaying exactly: the
memory alone does not formulate; the same evidence placed in the window does.

And the bigger reader **does not recall better** — 7/8 against 8/8 — it
**halves fabrication and more than doubles refusals**. It is the first
benefit of capacity measured anywhere in this project that is neither speed
nor fluency, and it is the one that decides whether this endpoint is usable
by someone who is not its owner.

## Grounding, and an honest failure

The system note introducing retrieved passages said only *"use them if they
are relevant"*. It now also says to answer only from the notes and to say
plainly when they do not contain the answer.

Registered before the change: fabrication ≤ 3/8, rephrased recall ≥ 7/8.
The second holds (8/8, and the answers come out tighter). **The first is
falsified**: refusals 0 → 2 of 8, fabrication ~8 → 6. One line of system
prompt is not enough for a 0.6B, which obeys it twice in eight. It ships
anyway — nothing to none is a real gain at no measured cost — with its
insufficiency written next to it in the code, and with the measurement that
does halve fabrication named right beside it: a bigger reader.

**Tests**: 21 unit + 14 end-to-end + 16 over HTTP + 28 for the axis-4
commands = **79**, all green.

`pip install -U sillage`
