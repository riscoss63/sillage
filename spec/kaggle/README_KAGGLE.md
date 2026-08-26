# Kit Kaggle — bancs GPU du drafter spéculatif Sillage

Le kit s'assemble depuis le dépôt :
`python spec/kaggle/make_kit.py --docs <doc1> <doc2>` produit
`kaggle_kit.zip` (code + états en lecture seule + vos documents).

Objectif : convertir les acceptations mesurées sur CPU (75–76 %) en speedup
réel sur GPU, et étendre à Qwen3-4B — les chiffres qui manquent au papier 5.
Tout est en lecture seule : l'état mémoire n'est jamais modifié.

## 1. Préparer le compte (une fois)

1. Compte sur kaggle.com, puis **vérification du téléphone**
   (Settings → Phone verification) — obligatoire pour avoir le GPU et
   Internet dans les notebooks.
2. Quota : 30 h de GPU par semaine. Tout ce kit consomme ~1–2 h au total.

## 2. Charger le kit (une fois)

1. kaggle.com → **Datasets → New Dataset** → glisser `kaggle_kit.zip`
   → visibilité **Private** → nom, par ex. `sillage-spec-kit` → Create.
   (Kaggle dézippe automatiquement.)

## 3. Créer le notebook

1. **Code → New Notebook**.
2. Panneau de droite : **Add Input** → votre dataset `sillage-spec-kit`.
3. **Settings** :
   - Accelerator : **GPU P100** (recommandé : décodage batch 1 = bande
     passante ; les T4 ×2 ne servent que pour tenter un 8B shardé)
   - Internet : **ON** (téléchargement des modèles depuis Hugging Face)
4. Cellules, dans l'ordre :

```bash
# cellule 1 — installation (sillage vient de PyPI ; torch/transformers
# sont déjà sur l'image Kaggle, d'où le --no-deps)
!pip -q install sillage --no-deps

# cellule 2 — copier le kit en zone inscriptible
!cp -r /kaggle/input/sillage-spec-kit/kaggle_kit /kaggle/working/kit
%cd /kaggle/working/kit

# cellule 3 — LE chiffre-clé d'abord : le facteur de conversion GPU
# (latence d'un forward de 1 vs 16 tokens, cache chaud)
!python bench_gpu.py --config micro --device cuda --dtype float16 --target Qwen/Qwen3-1.7B

# cellule 4 — A : 0.6B sur son propre état (référence intra-modèle)
!python bench_gpu.py --config A --device cuda --dtype float16 --pld

# cellule 5 — C : 1.7B + mémoire du 0.6B, réglages calibrés du 26/08
!python bench_gpu.py --config C --device cuda --dtype float16 \
    --target Qwen/Qwen3-1.7B --beta 40 --lam 0.85 --thrq 0.5

# cellule 6 — C sur 4B, avec recalibration lecture seule pour CETTE cible
# (~8 Go à télécharger la première fois : patience)
!python bench_gpu.py --config C --device cuda --dtype float16 \
    --target Qwen/Qwen3-4B --calibrate

# cellule 7 — B : le contrôle négatif (cible vanilla, attendu faible)
!python bench_gpu.py --config B --device cuda --dtype float16 --target Qwen/Qwen3-1.7B

# cellule 8 — gpt2 sur l'état des quatre papiers (comparaison CPU/GPU)
!python bench_gpu.py --config gpt2 --device cuda --dtype float16 --pld
```

5. Les résultats (`results_gpu_*.json`) apparaissent dans
   `/kaggle/working/kit` → onglet **Output** du notebook → téléchargez-les
   et déposez-les dans `spec_drafter/results/` du projet.

## 4. Lecture des résultats

- **micro** : si « 16 tokens / 1 token » ≈ 1,0–1,5, le GPU est bien en
  régime de latence — chaque token accepté est quasi gratuit, et
  l'acceptation de 75 % doit se traduire en ×2+ sur les configs A/C.
  (Sur CPU ce ratio est ~16 : c'est le plafond qu'on a mesuré.)
- **A** : attendu speedup ×2+ à acceptation ~76 % ; l'ablation PLD doit
  rester derrière (~50 %).
- **C 1.7B calibré** : attendu acc ~75 %, rappel verbatim ~5 mots ;
  le speedup dépend du micro.
- **C 4B --calibrate** : la nouveauté — si la calibration lecture seule
  (~2 min GPU) donne au 4B le rappel des documents ET un speedup, le récit
  « lire avec le petit, servir toute la famille » tient à trois tailles.
- **B** : doit rester faible (acc ~30 %, ×~1) — c'est le contrôle qui rend
  le reste crédible. S'il « gagne », quelque chose cloche : le signaler.
- Chaque run vérifie `identical n/n` : le spéculatif reproduit exactement
  le décodage normal de sa cible. En fp16, les textes peuvent différer des
  runs CPU fp32 — normal et sans incidence sur cette garantie interne.

## 5. Hygiène

- États mémoire fournis en lecture seule (le code n'appelle jamais save()).
- Les documents inclus sont vos propres manuscrits/papiers.
- Sessions Kaggle : 12 h max ; pensez à **Stop Session** en quittant pour
  ne pas brûler le quota.
