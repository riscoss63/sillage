# Reproducing every number in the four papers

Everything runs on CPU with fixed seeds. Total: roughly 2 h for the corpora
and frozen-LM passes, then 3–6 h for the experiments (most of them can be run
selectively — each block below is independent once the dumps exist).

Any script can be launched from anywhere; a bootstrap header resolves the repo
root and `chdir`s to it, so `data/`, `dumps/`, `results/` and `papers/*/figs`
always resolve identically.

```bash
pip install -r requirements.txt
```

## 0. Corpora and frozen-LM passes (needed by everything)

```bash
python pipeline/data_prep.py            # 3 short streams (network, once)
python pipeline/data_prep_500k.py       # War and Peace, King James Bible
python pipeline/data_prep_qwen.py       # same texts, Qwen3 tokenizer
python pipeline/dump_any.py gpt2 bhd relativity alice tolstoy
python pipeline/dump_500k.py
python pipeline/dump_any.py qwen bhd relativity alice
```

One pass per stream produces hidden states, base log-probabilities and the
base model's top-128 candidates. **Every method downstream consumes exactly
these dumps**, so no baseline ever sees more than another.

The `bhd` ("Manuscripts") stream is built from the author's unpublished
drafts, which are not redistributed. Put your own novel `.txt`/`.md` documents
in `manuscripts/` to run that protocol with your own contamination-free
domain; the public streams work as-is.

## 1. Memory — baselines, the factorial, replications (paper 1)

```bash
python memory/memories.py                    # kNN-LM, capped kNN, nulls,
                                             # hidden-state keys (negative)
python memory/ngram_memory.py 0.0 1.0        # n-gram memory + exact dictionary
python eval/rag_baseline.py                  # RAG-style retrieve & rescore
python memory/sillage_factorial.py       # amplitudes x gates x key scales
python memory/multiseed.py bhd               # 5-seed replication
python memory/exp_500k.py tolstoy && python memory/exp_500k.py bible
python memory/exp_500k_bigD.py               # capacity sweep
python memory/model2_qwen.py bhd             # second model
python eval/cloze_eval.py gpt2 bhd           # downstream recall
python eval/paired_test_v3.py                # headline paired test
```

## 2. Semantic keys and routing (paper 2)

```bash
python eval/semantic_diag.py gpt2 alice          # geometry + LSH diagnostics
python memory/sillage_semantic.py tune gpt2 alice   # key-level hybrid (fails)
python memory/sillage_router.py gpt2 bhd 999999999 multi
python memory/multiseed_router.py gpt2 bhd
python memory/router_500k.py tolstoy decay
python eval/cloze_router.py gpt2 bhd
```

Run the diagnostics first: they are what justified the design, and they take
minutes.

## 3. Hierarchy and consolidation (paper 3)

```bash
python memory/hierarchy_500k.py bible     # instruments one pass, caches it
python memory/hierarchy_500k.py tolstoy
python eval/hier_diag.py tolstoy          # coherence check vs paper 2
```

The instrumented pass is cached in `results/hier_cache_*.npz`; re-running the
script re-evaluates every admission policy and capacity from the cache in
seconds instead of hours.

## 4. Fast weights (paper 4)

```bash
python fastweights/fastweights.py gpt2 bhd            # learning-rate sweep
python fastweights/fastweights_combo.py bhd           # rank x gating + memory
python fastweights/fastweights_scale.py gpt2 tolstoy 16      # 500k stability
python fastweights/fastweights_scale.py qwen bhd 16,64       # second model
```

## 5. Figures

```bash
python figures/make_figures.py      # paper 1  -> papers/sillage/figs
python figures/make_figures_p2.py   # paper 2  -> papers/router/figs
python figures/make_figures_p3.py   # paper 3  -> papers/hierarchy/figs
python figures/make_figures_p4.py   # paper 4  -> papers/fastweights/figs
```

## Integrity controls

```bash
python eval/diagnostic.py       # shuffled-retrieval null, unigram null,
                                # base-model sanity perplexity
python eval/smoke_test.py       # end-to-end pipeline check on synthetic data
```

Run these before trusting any new result, including your own: they caught a
corpus-formatting artifact worth +1.35 phantom nats and a one-position
misalignment in a baseline.

## Statistical conventions

Identical across all four papers: development split = first 20 % of each
stream (hyperparameters tuned there, then frozen), test = remaining 80 %;
95 % block-bootstrap confidence intervals over 512-token blocks; headline
comparisons use *paired* bootstraps on identical positions; multi-seed studies
re-tune per seed and report mean ± SEM.
