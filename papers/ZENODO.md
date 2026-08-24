# Dépôt Zenodo — champ par champ

Stratégie : **un enregistrement par papier** (chacun son DOI, c'est ce qui se
cite) **+ un enregistrement pour le code**. Zenodo ne demande pas d'endosseur ;
arXiv restera possible ensuite (arXiv accepte un travail déjà déposé ailleurs,
il faudra seulement mentionner le DOI Zenodo dans le champ « Comments »).

**Avant de remplir quoi que ce soit : il faut le PDF.** Compilez le `.tex` sur
Overleaf (importer le fichier + son dossier `figs/`), téléchargez le PDF.
Zenodo exige au moins un fichier.

---

## Enregistrement 1 — le papier Sillage

### Files
- `sillage.pdf` (le PDF compilé)
- *optionnel mais recommandé* : `sillage.tex` + `figs/` en `.zip`, pour que la
  source reste ouverte.

### Basic information

| champ | valeur à saisir |
|---|---|
| **DOI** | garder le DOI réservé : `10.5281/zenodo.22079016` |
| **Resource type** | `Publication` → **`Preprint`** |
| **Title** | `Sillage: Surprise-Gated Amplitude Memory for Frozen Language Models` |
| **Publication date** | `2026-08-24` (date du dépôt ; le travail n'a pas été publié ailleurs avant) |
| **Authors/Creators** | `Sghairi, Abderrahmane` — Affiliation : `Independent Researcher, Issoire, France` — ORCID : voir la note plus bas |
| **Description** | l'abstract en texte propre : voir `papers/_metadata.txt`, section `### Sillage` |
| **License** | `Creative Commons Attribution 4.0 International` (CC BY 4.0) |
| **Copyright** | `Copyright (C) 2026 Abderrahmane Sghairi` |

### Recommended information

| champ | valeur |
|---|---|
| **Contributors** | *(laisser vide)* |
| **Keywords** | `language models` · `associative memory` · `Hebbian learning` · `test-time adaptation` · `hyperdimensional computing` · `vector symbolic architectures` · `kNN-LM` · `continual learning` · `gradient-free learning` · `three-factor learning rules` |
| **Languages** | `English (eng)` |
| **Dates** | *(laisser vide)* |
| **Version** | `1.0.0` |
| **Publisher** | `Zenodo` (valeur par défaut, à conserver) |
| **Funding / Awards** | *(laisser vide — recherche non financée)* |
| **Alternate identifiers** | *(laisser vide)* |
| **Related works** | `is supplemented by` → URL : `https://github.com/riscoss63/sillage` |
| **References** | coller la bibliographie du papier (une référence par ligne) |

### Software
Cette section ne concerne **pas** l'enregistrement d'un papier : laisser vide
(elle sera remplie dans l'enregistrement « code » ci-dessous).

### Publishing information
`Journal`, `Imprint`, `Thesis`, `Conference` : **tout laisser vide.** Un
preprint n'a pas de revue ; remplir ces champs fabriquerait une fausse
référence.

---

## Enregistrements 2, 3, 4 — les papiers suivants

Tout est identique, sauf :

| | Router | Hierarchy | FastWeights |
|---|---|---|---|
| **Title** | Route the Scores, Not the Keys: Gradient-Free Semantic Memory for Frozen Language Models | One Signal, Three Tiers: A Routed Memory Hierarchy with Surprise-Gated Consolidation for Frozen Language Models | Memory Remembers, Fast Weights Adapt: Two Gradient-Free Ways to Learn at Test Time, and Why They Add Up |
| **Description** | `### Router` de `_metadata.txt` | `### Hierarchy` | `### FastWeights` |
| **Keywords** *(en plus des communs)* | `locality-sensitive hashing`, `SimHash`, `semantic retrieval` | `memory consolidation`, `complementary learning systems`, `memory hierarchy` | `fast weights`, `delta rule`, `online learning`, `low-rank adaptation` |
| **Related works** | `references` → DOI du papier Sillage | `references` → DOI Sillage **et** DOI Router | `references` → les trois DOI précédents |

⚠️ **Ordre de publication** : publiez **Sillage en premier**, notez son DOI,
puis mettez-le dans les `\bibitem{sillage}` des trois autres `.tex` **avant**
de les compiler. Idem en cascade pour Router et Hierarchy. Un DOI n'existe
qu'une fois l'enregistrement publié.

---

## Enregistrement 5 — le code

Le plus simple est la **connexion GitHub de Zenodo** (Settings → GitHub →
activer le dépôt, puis créer une *release* `v1.0.0` sur GitHub : Zenodo crée
l'enregistrement et le DOI tout seul). Si vous préférez le formulaire :

| champ | valeur |
|---|---|
| **Files** | archive `.zip` du dépôt (sans `data/` ni `dumps/`) |
| **Resource type** | `Software` |
| **Title** | `Sillage: gradient-free test-time memory for frozen language models (code)` |
| **Description** | le README, ou : *"Reference implementation, experiments and results for the four Sillage preprints: a fixed-size Hebbian memory, semantic routing, a consolidating memory hierarchy, and a delta-rule readout adapter for frozen language models. Pure NumPy/PyTorch, CPU-only, fully reproducible with fixed seeds."* |
| **License** | `MIT License` (celle du dépôt, pas CC BY) |
| **Version** | `1.0.0` |
| **Related works** | `is supplement to` → DOI de chacun des quatre papiers |
| **Software → Repository URL** | `https://github.com/riscoss63/sillage` |
| **Software → Programming language** | `Python` |
| **Software → Development Status** | `Active` |

---

## Deux points à régler avant de publier

**1. Créez un ORCID** (5 minutes, gratuit, [orcid.org](https://orcid.org)).
C'est l'identifiant qui rattache durablement les quatre papiers à vous plutôt
qu'à un homonyme — et il compte pour votre crédibilité en tant que chercheur
indépendant, y compris auprès d'un futur endosseur arXiv.

**2. Le DOI est déjà dans le papier.** J'ai ajouté sous la date de
`sillage.tex` :

```latex
DOI: 10.5281/zenodo.22079016   Code: github.com/riscoss63/sillage
```

C'est exactement l'intérêt de la réservation de DOI : le PDF déposé contient
déjà son propre identifiant. **Vérifiez l'URL GitHub** si votre nom
d'utilisateur diffère de `riscoss63`, et recompilez avant dépôt.

---

## Après Zenodo : l'endossement arXiv

Le DOI Zenodo, le dépôt public et les quatre papiers cohérents constituent
précisément le dossier qui rend un endossement crédible. Les chemins possibles :

1. **Demander à un auteur que vous citez** (les auteurs de kNN-LM, de Titans,
   ou des travaux VSA). Un message court : qui vous êtes, le lien Zenodo, le
   lien GitHub, ce que fait le papier en deux phrases, la demande explicite
   d'endossement pour `cs.CL`. Le taux de réponse est meilleur qu'on ne le
   croit quand le travail est déjà public et reproductible.
2. **Passer par un contact académique local** (un laboratoire, un ancien
   enseignant) — l'endosseur n'a pas besoin d'être coauteur ni de relire.
3. **Publier d'abord ailleurs** : un workshop (les workshops NeurIPS/ICLR
   acceptent les preprints), ou HAL, qui n'exige pas d'endossement en France
   et est bien indexé.

Dans tous les cas, indiquez le DOI Zenodo dans le champ « Comments » d'arXiv
lors de la soumission : c'est propre et cela signale l'antériorité.
