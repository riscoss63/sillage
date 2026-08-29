# Kit Kaggle — bancs GPU du drafter spéculatif Sillage

Le kit s'assemble depuis le dépôt :
`python spec/kaggle/make_kit.py --docs <doc1> <doc2>` produit
`kaggle_kit.zip` (code + états en lecture seule + vos documents).

Objectif : reproduire les bancs GPU du papier 5 — convertir les acceptations
mesurées sur CPU (75–76 %) en speedup réel, et l'étendre à Qwen3-4B. Ces
chiffres sont mesurés et commités (`results/drafter_gpu_*.json`, table
`tab:speed` du papier) ; ce kit sert à les refaire, pas à les produire pour
la première fois. Tout est en lecture seule : l'état mémoire n'est jamais
modifié.

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
   - Accelerator : **GPU T4 ×2** — c'est le matériel des chiffres commités
     du papier 5 (Kaggle T4, fp16 ; le microbenchmark y donne 45,7 → 43,1 ms,
     soit c(16)/c(1) = 0,94). Les scripts n'utilisent qu'une seule des deux
     cartes (`--device cuda`) ; la seconde ne sert qu'à tenter un 8B shardé.
     Le P100 tourne aussi, mais ses chiffres ne sont alors plus comparables
     aux fichiers commités : déposez le microbenchmark sous
     `drafter_micro_p100.json` et non `drafter_micro_t4.json`, dont le nom
     désigne le matériel.
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

# cellule 5b — OBLIGATOIRE : bench_gpu.py écrit results_gpu_C.json pour les
# deux runs C, donc la cellule 6 écrase la cellule 5 si on ne met pas le
# 1.7B à l'abri d'abord (ou téléchargez le fichier avant de lancer la 6)
!mv results_gpu_C.json results_gpu_C_17b.json

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
   et renommez-les dans `results/` du dépôt avec le préfixe `drafter_`.
   La règle générale — `results_gpu_*.json` → `results/drafter_gpu_*.json` —
   est énoncée dans `REPRODUCE.md` à la racine du dépôt (l'annexe de
   reproduction du papier 5, dans `papers/drafter/drafter.tex`, liste les
   fichiers commités mais pas cette convention de renommage) :
   `results_gpu_A.json` → `results/drafter_gpu_A.json`, de même pour B et
   gpt2. Le microbenchmark sort sous `results_gpu_micro.json` (`--config
   micro`). Les deux runs C se distinguent par leur cible, d'où le renommage de
   la cellule 5b : `results_gpu_C_17b.json` → `results/drafter_gpu_C_17b.json`
   et le `results_gpu_C.json` de la cellule 6 →
   `results/drafter_gpu_C_4b.json`. Le microbenchmark est la seule exception
   à la règle : il est commité sous `results/drafter_micro_t4.json`, qui
   nomme le matériel, et non `drafter_gpu_micro.json` — exception que
   `REPRODUCE.md` ne mentionne pas.

## 4. Lecture des résultats

- **micro** : si « 16 tokens / 1 token » ≈ 0,9–1,5 (0,94 sur le T4 commité),
  le GPU est bien en
  régime de latence — chaque token accepté est quasi gratuit, et
  l'acceptation de 75 % se traduit en ×1,6–2,0 sur les configs A/C.
  (Sur CPU ce ratio est ~16 : c'est le plafond qu'on a mesuré.)
- **A** : mesuré ×1,63 à acceptation 76 % ; l'ablation PLD doit rester
  derrière (~50 %).
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
