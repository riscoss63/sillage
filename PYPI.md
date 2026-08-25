# Sillage

**Your language model forgets everything. Sillage gives it a memory — and a
way to keep learning — in a fixed handful of megabytes, with no gradients and
no index that grows.**

> *sillage* (n., French) — the trace left behind by something that has passed:
> a ship's wake, a scent in a room. What a model keeps of what it read.

A frozen language model reads your documents, remembers them, and predicts
better next time. No gradients, no fine-tuning, no vector database. One
Hebbian matrix written as the model reads, a semantic tier routed by
confidence, a cold store that consolidates by surprise, and a rank-16 adapter
on the readout — four mechanisms, four preprints, one command-line tool.
Everything runs on a laptop CPU.

![demo](https://raw.githubusercontent.com/riscoss63/sillage/main/figs/demo.gif)

## Install

```bash
pip install sillage
```

## Use

```bash
sillage index notes.md                # instant: no model, already queryable
sillage ask "what did the report say?"

sillage read notes.md                 # memorize it (CPU: ~8 min per 10k tokens)
sillage complete "The report said"    # generate WITH the memory
sillage status                        # what it knows, tier by tier
sillage forget --all
```

```console
$ sillage read preprint_v1.txt                    # Monday, memory empty
read preprint_v1.txt: 5859 tokens in 4.9 min | PPL 12.14 -> 11.83 (adapter) -> 11.71 (+memory)

$ sillage read preprint_v2.md                     # Tuesday, a new process
read preprint_v2.md: 11041 tokens in 9.5 min | PPL 10.68 -> 9.82 (adapter) -> 5.39 (+memory)
```

Three numbers per file: what the frozen model alone predicts, what the
rank-16 adapter adds, and what the memory of everything read so far adds on
top. The state lives in `./.sillage`, survives restarts, and never grows:
7.4 MB with GPT-2, 25 MB with Qwen3, the same after one document or ten
thousand. It never learns from its own generations — only from what you give
it to read.

From Python:

```python
from sillage import Sillage

s = Sillage(model="gpt2")          # any causal LM; omit it and the state
                                   # says which model it belongs to
s.read("notes.md")                 # read, memorize, index -- then save
s.ask("what did the report say?")  # exact passages, nothing generated
print(s.complete("The report said"))
```

## How much better, exactly?

On 36k tokens of technical text the model had never seen (frozen GPT-2 124M,
every system tuned identically on a held-out prefix, 95 % bootstrap CIs):

| system | perplexity | change | memory used |
|---|---|---|---|
| frozen GPT-2 | 31.2 | — | 0 |
| \+ RAG-style retrieve & rescore | 29.9 | −4 % | corpus + index |
| \+ kNN-LM, **unbounded** store | 23.6 | −24 % | 55 MB, grows forever |
| \+ **this memory** (fixed) | **19.2** | **−38 %** | **4.2 MB, constant** |
| \+ memory **and** fast weights | **16.6** | **−47 %** | 7.4 MB, constant |

The fixed 4.2 MB memory beats the unbounded datastore it was designed to
approximate — paired bootstrap **P = 1.000**, replicated over 5 random seeds
and on a second model (Qwen3-0.6B).

**Where it does not win:** on long, low-repetition narrative text, kNN-LM
still beats it. This memory captures verbatim recurrence; that is its regime,
and it is measured and published rather than hidden.

## Any causal language model

```bash
sillage read notes.md --model qwen                          # shortcut
sillage read notes.md --model HuggingFaceTB/SmolLM2-135M    # any hub id
sillage read notes.md --model ./my-finetuned-llama          # any local folder
```

Verified across three architectures (GPT-2, Qwen3, GPT-NeoX). For a model
nobody has tuned, the readout calibrates itself on a rolling window of what
you read, following the papers' protocol; for the two the papers did tune,
their published settings are kept. A memory is written in one model's token
space, so give each model its own `--state` directory.

## The four preprints

| # | title | DOI |
|---|---|---|
| 1 | Sillage: Surprise-Gated Amplitude Memory for Frozen Language Models | [10.5281/zenodo.22079016](https://doi.org/10.5281/zenodo.22079016) |
| 2 | Route the Scores, Not the Keys | [10.5281/zenodo.22079444](https://doi.org/10.5281/zenodo.22079444) |
| 3 | One Signal, Three Tiers | [10.5281/zenodo.22079471](https://doi.org/10.5281/zenodo.22079471) |
| 4 | Memory Remembers, Fast Weights Adapt | [10.5281/zenodo.22079481](https://doi.org/10.5281/zenodo.22079481) |

Sources, figures, every number as committed JSON, the reproduction pipeline
and the three negative results are on GitHub:
**<https://github.com/riscoss63/sillage>**

## Citation

```bibtex
@article{sghairi2026sillage,
  title  = {Sillage: Surprise-Gated Amplitude Memory
            for Frozen Language Models},
  author = {Sghairi, Abderrahmane},
  year   = {2026},
  doi    = {10.5281/zenodo.22079016},
  url    = {https://doi.org/10.5281/zenodo.22079016}
}
```

MIT licensed.
