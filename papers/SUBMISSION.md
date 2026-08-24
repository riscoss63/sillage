# Compiler et soumettre les quatre preprints

Les quatre papiers vivent ici, dans le dépôt, avec leurs figures :

| dossier | fichier | titre |
|---|---|---|
| `sillage/` | `sillage.tex` | Sillage: Surprise-Gated Amplitude Memory for Frozen Language Models |
| `router/` | `router.tex` | Route the Scores, Not the Keys: Gradient-Free Semantic Memory for Frozen Language Models |
| `hierarchy/` | `hierarchy.tex` | One Signal, Three Tiers: A Routed Memory Hierarchy with Surprise-Gated Consolidation |
| `fastweights/` | `fastweights.tex` | Memory Remembers, Fast Weights Adapt: Two Gradient-Free Ways to Learn at Test Time |

Chaque dossier contient `figs/` (PDF pour LaTeX + PNG d'aperçu). Les figures se
régénèrent depuis les résultats avec `python figures/make_figures.py`,
`make_figures_p2.py`, `make_figures_p3.py`, `make_figures_p4.py`.

## Compiler

**Overleaf (recommandé, aucune installation)** : nouveau projet vide, importer
le `.tex` d'un papier **et son dossier `figs/`** (les PDF suffisent), compileur
pdfLaTeX → le PDF sort directement. Tous les packages utilisés sont standard
TeX Live (geometry, microtype, amsmath, booktabs, graphicx, hyperref, caption,
enumitem, xcolor) — aucune dépendance exotique, aucun `.bib` (les
bibliographies sont incluses dans chaque fichier).

**En local** : MiKTeX ou TeX Live, puis `pdflatex sillage.tex` (deux passes pour
les renvois).

## Ordre de soumission conseillé

Les papiers 2, 3 et 4 citent les précédents ; soumettez-les dans l'ordre et
reportez l'identifiant arXiv obtenu dans les bibliographies des suivants
(entrées `\bibitem{sillage}`, `{router}`, `{hier}`).

1. **Sillage** — le papier fondateur, autonome.
2. **Router** — cite Sillage.
3. **Hierarchy** — cite Sillage et Router.
4. **Fast weights** — cite les trois.

## Checklist arXiv

1. Relire le PDF (chiffres, tables, figures — tout vient de `results/`,
   régénérable).
2. Catégories : **cs.CL** (primaire), cs.LG et cs.NE (croisées).
3. Archive : le `.tex` + le dossier `figs/` (PDF), en `.zip` ou `.tar.gz`.
4. Champ « Comments » : mentionner l'URL du dépôt GitHub une fois publié.
5. Licence : CC BY 4.0 si vous voulez maximiser la reprise, sinon la licence
   arXiv par défaut.

## Pousser le dépôt

Tout est prêt dans `llm_memory/` (README vitrine, LICENSE MIT,
`requirements.txt`, `.gitignore`, `CITATION.cff`, les scripts, les résultats
JSON et désormais les quatre papiers) :

```bash
cd llm_memory
git init && git add . && git commit -m "Sillage: gradient-free test-time learning"
git remote add origin https://github.com/<vous>/sillage-memory.git
git push -u origin main
```

`data/` et `dumps/` sont exclus par `.gitignore` (~2 GB, régénérables). Le flux
*Manuscripts* (vos brouillons non publiés) n'est pas redistribué : le README
explique comment un lecteur utilise ses propres documents à la place.

Après publication, reportez l'identifiant arXiv dans `CITATION.cff` et dans le
tableau du `README.md`.
