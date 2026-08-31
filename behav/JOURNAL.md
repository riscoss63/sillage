# Axe 3 — Preuve comportementale (papier 6) : journal

Objectif : appliquer l'échelle de preuve de « Beyond Perplexity »
(arXiv:2607.00368) à Sillage AVANT que les reviewers la demandent — six
épreuves comportementales sur faits inventés, puis les benchmarks
reconnus (LongMemEval, BEAM, MemoryAgentBench) et la montée 4B.

## La suite comportementale (`behavioral.py`)

Faits inventés (entité imprononçable → le modèle de base ne peut PAS les
connaître : le contrôle est intégré par construction), incrustés ×3 dans
du remplissage administratif, gabarit canonique A. Six épreuves :

1. **Rappel libre** — `complete(préfixe A)` : la valeur sort-elle ?
   (mesuré AVANT lecture = contrôle base, attendu ≈ 0.)
2. **Paraphrase** — préfixe B, jamais vu dans le document, même sens.
   Attendu FAIBLE pour la clé n-gram (la limite assumée des papiers) —
   on la mesure au lieu de l'esquiver ; le tier sémantique (qwen) est
   l'espoir de la relever, à mesurer ensuite.
3. **Rétention** — relire ~20k tokens de remplissage pur, re-sonder A.
   (Sans décroissance d'abord ; --half-life en variante.)
4. **Localité** — NLL téléforcée d'un texte témoin non lié, mesurée SANS
   écrire, avant/après lecture du dossier : la mémoire ne doit rien
   changer hors de son domaine (l'abstention en acte).
5. **Conflits** — dossier v2 où 10 faits changent de valeur ; après 1
   lecture de v2 puis 2 : ancienne ou nouvelle valeur au probe ?
   (La dynamique d'écrasement des amplitudes, vue comportement.)
6. **Usage hors contexte** — par construction : aucun probe ne contient
   le fait dans son contexte ; seule la mémoire peut répondre
   (l'argument central du papier, hérité du protocole des tests e2e).

Modèles : gpt2 d'abord (vitesse), qwen ensuite (tier sémantique →
paraphrase), 4B après (GPU). Sorties JSON par épreuve, graines fixées.

## Critères honnêtes attendus (à écrire AVANT de voir les chiffres)

- Rappel A : nettement > 0 (le cloze des papiers dit ~2× le base sur
  termes récurrents ; ici les faits sont répétés 3×, on attend mieux).
- Paraphrase B : proche de 0 sur gpt2/M_G — c'est le résultat honnête
  qui cadre le papier 6 ; toute surprise positive vient du cold store
  (si un n-gram du préfixe B chevauche A) et devra être décomposée.
- Localité : ΔPPL témoin ≈ 0 (< 1 %).
- Conflits : v2×1 partiel, v2×2 majoritairement la nouvelle valeur
  (les amplitudes √masse convergent) — la vitesse de bascule est LA
  courbe intéressante.

## Campagne 1 — gpt2 (26/08 soir, results/behav_gpt2.json)

| épreuve | attendu | mesuré |
|---|---|---|
| contrôles base | ≈ 0 | 0 % / 0 % ✓ |
| rappel (1 lecture, faits ×3) | > 0 net | **100 %** (30/30) |
| paraphrase (gabarit jamais vu) | ≈ 0 | **0 %** — l'honnêteté attendue |
| rétention (+20k interférence) | dégradation possible | 100 % → 100 % |
| conflit v2 ×1 | partiel | nouvelle 40 %, ancienne **0 %**, confusion 60 % |
| conflit v2 ×2 | bascule | nouvelle **100 %** |

**Trouvaille : la dynamique de conflit est asymétrique** — l'ancienne
valeur meurt AVANT que la nouvelle ne s'impose (zone de confusion
transitoire à 60 % après une lecture de v2 : les amplitudes superposées
se brouillent mutuellement, puis la 2e lecture fait tout basculer).
Comportement mesurable, reproductible, très « papier 6 ».

**Défaut de protocole corrigé** : le témoin de localité (+1,9 %)
partageait les gabarits du remplissage (pas vraiment « non lié », PPL
1,18 mal conditionnée). Remplacé par de la prose naturelle sans
recouvrement (constante WITNESS) ; conflits reportent désormais la zone
de confusion explicitement. Campagnes relancées : gpt2 (témoin propre)
puis qwen (LA question : le tier sémantique relève-t-il la paraphrase ?).

## Campagne 1bis — témoin corrigé + qwen (26/08 tard)

**gpt2** : réplique le tour 1 à l'identique (déterminisme ✓) ; localité
désormais propre avec la prose naturelle : **+0,6 %** (PPL 114, bien
conditionnée).

**qwen (semantic ON)** — les deux résultats de fond :
1. **Paraphrase 0 % malgré le tier sémantique** = réplication
   comportementale indépendante du verdict du papier 2 (« improves
   likelihood, not greedy recall ») — deux méthodes, même conclusion.
2. **Bascule de conflit modèle-dépendante** : gpt2 bascule en 2 lectures
   (confusion 60 %→0 %) ; qwen englué (×1 : confusion 100 % ; ×2 :
   nouvelle 10 %, confusion 90 %). Hypothèse : λ_G=0,2/β=160 (réglages
   publiés qwen) font moins confiance à la mémoire → aucune valeur ne
   s'impose. LA figure du papier 6 = courbe de bascule ×k. Courbe ×3/×4
   qwen en cours (conflict_curve_qwen.py, états réutilisés).

À creuser ensuite : rappel qwen 93 % (2 ratés — tokenisation des valeurs
à inspecter dans le JSON) ; localité qwen +1,9 % à attribuer par ablation
(--no-semantic / --no-fastweights).

## Sonde de readout + décompositions (26/08, nuit) — la moisson

**① Le plateau de conflit qwen était un artefact de readout, prouvé** :
courbe ×1..×4 = 0/10/20/20 % (plateau, confusion 80 %) sous réglages
publiés (β160/λ0,2/q75) ; MÊME ÉTAT, zéro relecture, réglages de
confiance calibrée (β40/λ0,85/q50) → nouvelle **100 %**, confusion 0 %,
et les 2 ratés du rappel stable disparaissent aussi (93→100 %). Le
stockage avait la réponse ; le readout parlait trop bas.
(`results/behav_qwen_curve.json`, `behav_qwen_readout_probe.json`.)
→ Thèse unificatrice du papier 6 : **« behavioral conversion is a trust
problem »** — le même mécanisme explique le kNN inerte du P1 (λ 0,05),
le sémantique sans recall du P2, et l'acceptation du P5. Caveat de
rigueur : 40/0,85/q50 vient de la calibration 1.7B (P5), utilisé ici
comme SONDE mécanistique, pas comme réglage validé pour le 0.6B — le
protocole propre restera la calibration sur fenêtre.

**② La paraphrase est une frontière STRUCTURELLE, pas de confiance** :
0 % même à λ0,85 (`behav_qwen_tradeoff.json`). Conflits = problème de
readout ; paraphrase = problème de clés. Deux natures, désormais
séparées expérimentalement.

**③ Localité décomposée** (`behav_qwen_locality_ablation.json`) :
tout actif +4,00 % ; sans adaptateur +0,11 % ; sans adaptateur ni
sémantique **±0,00 %** ; adaptateur seul +3,88 %. La mémoire hebbienne
est chirurgicale (zéro exact hors domaine — abstention + orthogonalité) ;
le coût est ~97 % l'adaptateur — le prix hors domaine de sa
recalibration globale, jamais mesuré dans le P4. Indice complémentaire :
la PPL témoin est identique au centime sous λ0,2 et λ0,85 → M_G ne
franchit JAMAIS son seuil sur le témoin.

## Rétention longue + équivalence-contexte (27/08, results/)

**Rétention** (`retention_gpt2.json`) : sans décroissance 100 % jusqu'à
+65k d'interférence puis 97 % (1 fait/30) à +110k ; half-life 30k quasi
identique (97 % dès +43k). À w/p 0,11 (loin de 0,5), l'oubli est quasi
gratuit en comportement. Hypothèse mécanistique À VÉRIFIER : le cold
store (sans décroissance) porte les faits que M oublie — la hiérarchie
P3 comme assurance-rétention. Caveat : interférence à gabarits partagés
= douce ; stress-test adversarial (entités proches) à faire.

**Équivalence-contexte, DEUX RÉGIMES** (`equivalence_gpt2.json` +
`_transfer.json`) — LA figure du papier 6 :
- *Récitation* (corpus papiers relu) : mémoire+64 tokens = PPL 4,80 ;
  le nu à 1024 = 65,2 — il ne rejoint JAMAIS (borne : l'état ≻ 960
  tokens). Raison conceptuelle : l'information est hors de la fenêtre
  locale — le contexte ne remplace pas une mémoire cross-document.
- *Transfert* (sibling édité jamais lu) : **C\* = 195 tokens** (le nu
  égale mémoire+64 avec ~3× plus de contexte) — ET à contexte égal la
  mémoire aide toujours (1,27 vs 1,69 à C=1024, −25 %). Nuance à dire :
  C\* dépend de l'auto-répétitivité du segment (doc très auto-répétitif
  → le contexte long rattrape par ICL interne ; doc moins répétitif →
  C\* grandit). La métrique discrimine les régimes : c'est sa valeur.

Phrase-résumé candidate : « Sur du connu, l'état ne s'échange contre
aucune quantité de contexte (>16×) ; sur du nouveau apparenté, il vaut
~3× le contexte à budget de nats égal — et il aide encore à contexte
plein. »

## Interférence adversariale (27/08, adversarial_gpt2.json)

Prédictions écrites avant : B inoffensif ✓ ; C dégrade avec bascule vers
d≥3, ralentie par les amplitudes ✓✓.

| attaque | vrai | distracteur | confusion |
|---|---|---|---|
| B : 60 entités jumelles (≈dose 9) | 100 % | — | — |
| C dose 1 (vs 3 occ. du fait) | 100 % | 0 % | 0 % |
| C dose 3 (parité de masses) | **93 %** | 3 % | 3 % |
| C dose 9 (3× la masse) | 50 % | 50 % | 0 % |
| dose 9, cold DÉBRANCHÉ | 33 % | **0 %** | 67 % |

Lectures : ① sélectivité parfaite au bord (B) ; ② à parité le fait
domine 93/3, équilibre seulement à 3× la masse — chute bien plus lente
que les comptes purs (cold seul prédirait 75/25 distracteur à dose 9) ;
③ **le distracteur passe entièrement par le cold store** (0 % sans lui) :
la matrice d'amplitudes brouille l'attaquant en confusion mais ne cède
pas — la Proposition √masse du P1, observée en régime adversarial ; le
vecteur d'attaque = les comptes bruts du cold. ④ Piste défensive née de
l'attaque : **√comptes au cold aussi** (symétrie P1) — micro-expérience
à faire, candidate à une ligne du papier 6. Mécanisme du 93/3 à parité
encore à décomposer (récence adaptateur ? scores s_G bruts ? — mesurer
s_G[VAL] vs s_G[distracteur] directement dans l'état final).

## Défense √-cold + décomposition (27/08, adversarial_gpt2_defense.json)

- Contrôle dose 9 reconstruit : 50/50 (réplication exacte ✓).
- **Défense √-cold RÉFUTÉE** : 50/50 inchangé (√{9,3}=63/37, insuffisant
  à λ_C fixe). La vraie piste serait une confiance du cold fonction de la
  concentration de sa distribution — design futur, hors scope P6.
- Cold décomposé : parts vrai 0,25 / distr 0,75 = les comptes ✓ (canal
  d'attaque confirmé).
- ~~ANOMALIE 13:1~~ **RÉSOLUE par instrumentation d'amp_write**
  (27/08) — et le vrai mécanisme est meilleur que l'hypothèse :
  ① le diagnostic tokenizer avait confirmé l'alignement des adresses
  (2/2, v1 3/3) ; ② la trace des écritures montre la règle d'amplitude
  **exactement conforme** (VAL 3,22 / FUR 3,92 = √masses effectives) ;
  ③ MAIS les gates des collisions répétées s'effondrent en fenêtre
  (5,0 → 0,05-0,09 dès la 3e) : **la copie in-context du modèle gelé
  rend l'attaque prévisible → le gate de surprise la tarife à ~zéro.**
  9 écritures n'ont injecté que ~15 de masse au lieu de 45.
  **LE GATE EST UNE DÉFENSE ADVERSARIALE INTRINSÈQUE** : la redondance
  intra-fenêtre est gratuite pour le défenseur ; l'attaquant doit étaler
  hors fenêtre (coût ×). Gradient d'ordre mesuré sur les blocs de 30 :
  sG_distr 0,744 (entité 1, fraîche) → 0,10 (n°3) → 0,026 (5 dernières).
  Le 50/50 comportemental à dose 9 est entièrement expliqué : matrice
  ~parité (masses effectives ~10 vs ~15) + cold à comptes bruts 25/75.
  ⇒ **Correction de design principielle (candidate 1.2.0 + ligne P6)** :
  pondérer la distribution de successeurs du cold par la masse de
  surprise PAR SUCCESSEUR (aujourd'hui : comptes aveugles à g — le seul
  canal g-blind du système, et précisément le canal vulnérable). Dans la
  droite ligne « un signal, trois métiers » du P3 — le 4e métier.
  Nécessite un changement d'état (stocker la masse par successeur).
  (`adversarial_gpt2_defense.json` rows = gradient ; trace dans la
  sortie du run instrumenté.)

## Vérification cold-store-rétention (27/08, cold_retention_gpt2.json)

Prédiction confirmée, en plus fort. Décomposition à 4 voix, 3 jalons,
2 bras :

| jalon | full | sans cold | sans matrice | base+adapt |
|---|---|---|---|---|
| +0k | 100 % | 90 % | 97 % | 0 % |
| +43k (sans décr.) | 100 % | 57 % | 97 % | 0 % |
| +110k (sans décr.) | 97 % | **17 %** | **97 %** | 0 % |
| +110k (hl 30k) | 97 % | **0 %** | **97 %** | 0 % |

① La matrice s'érode vite MÊME sans décroissance (bruit de superposition
√N vs amplitude fixe — la Prop. P1 côté négatif) : c'est une mémoire de
TRAVAIL. ② Le cold est un roc (97 % constant) : la rétention longue
comportementale = la consolidation par surprise du P3, pas un bonus.
③ La décroissance est gratuite pour le système car le cold ne décroît
pas — « le froid garde ce que le chaud oublie », chiffré. ④ L'adaptateur
n'a RIEN mémorisé (0 % partout) : les faits vivent dans la mémoire, pas
dans le readout (division du travail P4 ✓). Cohérence adversarial : le
cold = la force ET la surface d'attaque → la correction « masse par
successeur » (1.2.0) sert les deux faces.
**Micro-exp à faire** : faits ×1 vs ×2 vs ×3 → « la règle des deux
occurrences » (admission cold ≥2) comme seuil de mémoire durable —
prédiction : rétention longue ≈ 0 pour ×1, ≈ pleine dès ×2.

## La règle des deux occurrences (27/08, two_occurrences_gpt2.json)

Prédiction confirmée des deux côtés, au manuel :
| jalon | G1 (×1) | G2 (×2) | G3 (×3) |
|---|---|---|---|
| couverture cold (grams) | **0/10** | 10/10 | 10/10 |
| +0k | 100 % | 100 % | 100 % |
| +43k | **0 %** | 90 % | 100 % |
| +110k | **0 %** | 90 % | 100 % |
| final sans cold | 0 % | 20 % | 30 % |

Un fait vu UNE fois ne franchit jamais l'admission (COLD_MIN_COUNT=2),
vit sur la matrice seule, et meurt dès +43k (plus vite que prévu) ; dès
DEUX occurrences, le cold le porte à 90-100 % jusqu'à +110k. « Ce
système retient durablement ce qu'il a vu deux fois » — une constante de
design devenue loi comportementale ; gradient 90→100 % : la 3e
occurrence consolide encore. Implication outil : relire = consolider.

**AXE 3 LOCAL : CAMPAGNE COMPLÈTE (10 blocs de résultats).** Restent les
chantiers externes (benchmarks LongMemEval/BEAM/MemoryAgentBench, 4B
Kaggle) puis la rédaction du papier 6, dont la charpente est prête :
thèse « trust », rôles des tiers (travail/durable/style), règle des deux
occurrences, équivalence-contexte 2 régimes, adversarial + gate-défense,
localité décomposée, frontière paraphrase.

## Suite du chantier (après cette campagne)

1. qwen (semantic ON) sur la même batterie → la colonne paraphrase.
2. Rétention avec --half-life (la décroissance échange rétention contre
   plasticité — chiffrer le taux).
3. Courbe d'équivalence-contexte (Mo ↔ tokens) — la métrique nouvelle.
4. Benchmarks externes : LongMemEval-S (lecteur constant), BEAM
   sous-ensemble, MemoryAgentBench (test-time learning + selective
   forgetting) — toujours « score compétitif à coût ~nul », jamais SOTA.
5. Montée Qwen3-4B (Kaggle, kit existant à étendre).

---

## 27/08/2026 — Montée en capacité (Kaggle, Qwen3-4B) — PRÉDICTIONS

Kernel `sillage-behav-4b` (T4, autonome, pip sillage 1.2.0.post1, tout
synthétique — aucun état ni document embarqué). Deux bras :

- **N (natif)** : Qwen3-4B (fp16) lit lui-même v1 puis v2 x2 ; rappel,
  paraphrase, témoin (sans écrire), conflits, puis probe de confiance
  (readout auto-ajusté vs réglages famille 40/0.85/q50).
- **T (transfert)** : 0.6B (fp32) construit v1+v2x2 ; 0.6B, 1.7B (fp32)
  et 4B (fp16) servent LE MÊME état — l'axe capacité à stockage
  constant, publié vs famille.

Référence 0.6B locale : rappel v1 93 %, paraphrase 0 %, conflits
plateau 20 % (x3/x4 et probe publié), calibré 100 % nouvelle / 100 %
stables, témoin +1.9 %.

**Prédictions (avant le run) :**
- P0 (contrôle) : à mémoire vide, rappel = paraphrase = 0 % aux deux
  capacités — sinon les faits ne sont pas inventés et tout est invalide.
- P1 : 4B natif, rappel v1 ≥ 80 % sous readout auto-ajusté ; ≥ 93 %
  sous réglages famille.
- P2 (cœur, falsifiable) : sur l'état 0.6B servi, à readout PUBLIÉ, la
  résolution de conflit reste ≤ 20 % aux trois capacités — la capacité
  n'achète pas la conversion, c'est le readout (loi 1) ; sous réglages
  famille, nouvelle ≥ 90 % aux trois. **Falsification : 4B publié
  ≥ 50 %.**
- P3 : paraphrase ≈ 0 % partout (frontière de surface tokens, pas de
  capacité — le 4B n'y peut rien).
- P4 : témoin 4B natif ≤ +3 % (localité tient à l'échelle).
- P5 : stables servis ≥ 85 % publié aux trois capacités (ne CHUTE pas
  en montant), ≥ 90 % famille, croissance faible avec la capacité.

**Caps déclarées :** pas de bras interférence à 4B (mécanisme cold
indépendant du lecteur, budget session sur l'axe conversion) ; l'état
transfert est SANS interférence donc ≠ état local — comparaisons
intra-kernel uniquement ; 4B en float16 (16 Go fp32 ne tient pas sur
T4).

Complément (avant le run) : dans le bras T, sémantique ET adaptateur
sont coupés aux trois capacités — tous deux sont des fonctions de la
géométrie cachée du LECTEUR (le papier 5 coupe déjà l'adaptateur sous
--target ; un hidden 4B n'a pas de sens pour un blanchiment 0.6B).
L'axe capacité compare donc M_G + cold seuls, à pile identique. Le
point 0.6B servi diffère en cela du probe local (pile complète) — les
prédictions P2/P5 portent sur la pile réduite.

---

## 27/08/2026 — LongMemEval, bras E (extractif, local) — PRÉDICTIONS

Banc externe LongMemEval-S (500 questions, ~121k tokens / ~50 sessions
de chat par question ; `xiaowu0162/longmemeval`). Bras E = la voix
extractive seule (`Index` lexical, celle de `sillage ask`) : chaque
session indexée comme un document, la question comme requête — aucun
modèle, aucun juge LLM. Métriques déterministes :
- evidence@k : une passage du top-k provient d'une session-preuve
  (answer_session_ids), k ∈ {1,3,5} ;
- answer-in-top3 (strict) : la chaîne réponse (normalisée) contenue
  dans le texte du top-3 ;
- multi-session : couverture complète vs partielle des sessions-preuves;
- les 30 questions _abs (pas de preuve) rapportées à part.

**Prédictions (avant le run) :**
- P-E1 : evidence@3 global ≥ 60 % (recouvrement lexical
  question↔session-preuve).
- P-E2 : ordre par type : single-session-user > knowledge-update >
  multi-session > temporal-reasoning (l'arithmétique de dates est hors
  de portée lexicale). **Falsification : temporal en tête.**
- P-E3 : answer-in-top3 NETTEMENT sous evidence@3 (≥ 20 points d'écart)
  — la marche « présence → formulation », version banc externe de
  « stored is not recalled ».
- P-E4 : multi-session — couverture partielle fréquente, complète rare
  (< 30 %).

Le bras G (génératif, qwen via fast_ingest — validé bit-à-bit ce jour,
x2.4 déjà sur CPU local) suivra sur sous-échantillon Kaggle.

**VERDICT bras E (500/500 questions, 17 min sur le CPU du bureau, zéro
GPU, zéro juge — results/lme_arm_e.json) :**

| métrique (470 hors _abs) | valeur |
|---|---|
| evidence@1 / @3 / @5 | 85.5 % / **92.6 %** / 94.5 % |
| answer-in-top3 (strict) | **29.6 %** |

Par type (evidence@3 → answer_top3) : single-session-user 100 → 78.1 ;
single-session-assistant 100 → 42.9 ; knowledge-update 98.6 → 54.2 ;
multi-session 92.6 → 10.7 ; temporal 89.8 → 10.2 ;
single-session-preference 60.0 → 0.

- P-E1 **CONFIRMÉE**, largement (92.6 % ≥ 60 %).
- P-E2 **CONFIRMÉE** sur la chaîne prédite exacte (100 > 98.6 > 92.6 >
  89.8) ; hors prédiction : preference dernier (60 %) — requêtes sans
  recouvrement lexical (la question ne cite pas le contenu).
- P-E3 **CONFIRMÉE**, au-delà : écart de **63 points** (92.6 → 29.6).
  La marche « présence → formulation » à l'échelle du banc — stored is
  not recalled, version externe. Décomposition : là où la réponse est
  une chaîne de surface (user facts : 78 %), l'extractif la livre ; là
  où il faut AGRÉGER (multi-session 10.7 %) ou CALCULER (temporal
  10.2 %), la présence ne suffit jamais.
- P-E4 **RÉFUTÉE** (négatif honnête) : couverture multi-session
  complète@5 = 43.8 % (> 30 % prédit) ; partielle 49.6 %, nulle 6.6 %.
  L'index couvre mieux que prédit — c'est la FORMULATION qui manque,
  pas la couverture.

Limite de métrique déclarée : answer-in-top3 n'est un proxy valable que
pour les réponses factuelles courtes (médiane 11 caractères) ; les
réponses « preference » (phrases descriptives) sont par nature
hors-proxy → 0 % mécanique, à lire comme non-mesuré, pas comme échec.
Positionnement : evidence@3 est une métrique de RETRIEVAL, pas
l'accuracy QA jugée du papier LongMemEval — ne jamais comparer les deux
colonnes (nommer le régime).

**VERDICT montée 4B (kernel T4, 55 min, results/behav_4b.json) :**

Natif Qwen3-4B (fp16, readout gouvernant 40/0.85/q75) :
| probe | 0.6B (local) | 4B natif |
|---|---|---|
| contrôles à vide | 0 % | 0 % |
| rappel v1 | 93.3 % | **93.3 %** (identique) |
| paraphrase | 0 % | 0 % |
| témoin | +1.9 % | **+0.47 %** |
| conflit x1 / x2 | 0 / 10 % | 0 / 0 % |
| conflit, réglages famille | 100 % | **100 %** (ancienne 0 %) |

Transfert (MÊME état 0.6B — 8922 tokens, 735 grams — pile M_G+cold,
servi par 0.6B / 1.7B / 4B) :
- readout PUBLIÉ (160/0.2/q75) : conflit nouvelle **10 % / 10 % / 10 %**
  — PLAT sur ×6.7 de paramètres ; stables 90 / 80 / 90 %.
- readout FAMILLE (40/0.85/q50) : nouvelle **100 % / 100 % / 100 %** ;
  stables 100 / 95 / 95 %.

- P0 ✓, P1 ✓ (93.3 ≥ 80), P3 ✓ (paraphrase 0 % partout), P4 ✓ (+0.47 ≤ +3).
- **P2 CONFIRMÉE EXACTEMENT — le résultat central : la capacité
  n'achète PAS la conversion.** À confiance égale, 0.6B ≡ 1.7B ≡ 4B,
  aux deux niveaux de confiance. La falsification (4B publié ≥ 50 %)
  n'est pas approchée (10 %).
- P5 partielle : stables publiés 90/80/90 — le point 1.7B (80 %) passe
  sous la borne ≥ 85 % prédite ; pas de croissance avec la capacité
  (plat bruité). Négatif honnête consigné.
- Détail qui affûte la loi 1 : à 4B natif, beta/lam gouvernants = déjà
  famille (40/0.85) — seul le seuil d'abstention diffère (q75 vs q50),
  et ce SEUL cran fait 0 % → 100 % sur les conflits. **Le cadran de la
  confiance, à cette échelle, c'est le seuil d'abstention.**
- recall_family 66.7 % sur les 30 v1 = cohérence interne : les 10 faits
  mis à jour répondent la NOUVELLE valeur (l'update fonctionne), les 20
  stables 100 %.

Timing clé pour le bras G : lecture pleine = 7 tok/s (142 ms/token) sur
le CPU 2-cœurs → fast_ingest obligatoire ; variante « gate GPU » à
valider localement (tolérance 1e-6 déclarée) pour viser ~2-4 min/question.

---

## 27/08/2026 — LongMemEval, bras G (génératif, Kaggle) — PRÉDICTIONS

Kernel `sillage-lme-g` : 43 questions de S (40 stratifiées
proportionnellement par type + 3 _abs, graine 7, tirage déclaré),
qwen 0.6B, pile = M_G + cold + sémantique, **adaptateur coupé** (tier
de style, précédent du service en cible), ingestion `fast_ingest`
gate=torch (tolérance validée localement), réservoirs sur GPU. Trois
voix par question, greedy n=24, scoring = containment normalisé de la
réponse (proxy réponses courtes ; preference hors-proxy) :
  (a) mémoire seule : « Question: … Answer: » sur l'état des ~50
      sessions (~121k tokens) ;
  (b) contexte+mémoire : top-3 passages de l'index + question, même
      état ;
  (c) contexte seul : même prompt que (b), état vide — le contrôle qui
      isole la contribution de la mémoire.

**Prédictions (avant le run) :**
- P-G1 : voix (a) ≤ 10 % global — double mur confiance + formulation à
  121k tokens d'état ; le négatif PRÉDIT par les lois du papier 6.
- P-G2 : voix (b) ≈ voix (c) à ±5 points — la redondance en fenêtre est
  tarifée ~0 par le gate, la mémoire ne nuit pas au régime RAG.
  **Falsification : (b) < (c) − 10 points (la mémoire NUIT).**
- P-G3 : (b) et (c) ≥ 4× la voix (a) — la formulation se fait en
  fenêtre ; l'écart 63 points du bras E, version générative.
- P-G4 : ingestion ≥ 100 tok/s (contre 7 tok/s mesurés en lecture
  pleine au kernel 4B) — la revendication outil.

Validation gate=torch (avant le run G, test_fast_ingest.py) : mode
exact = read_text BIT-À-BIT (23 tableaux, cold, complétions) ; mode
torch = admissions et comptes cold IDENTIQUES, dérive de masse absolue
5.7e-04 nats (arrondi sur les g≈0), relative 1.3e-04 sur les slots à
masse ≥ 0.5, M à 5.5e-06, complétions identiques. Première borne
relative naïve (1e-4 partout) déclenchée à 3.5e-2 par les slots à
masse minuscule — métrique corrigée, incident consigné.

**INCIDENT run G v3 (27/08 soir) :** ingestion mesurée à **37 tok/s**
(54 min/question → ~39 h pour 43 — session tuée à 12 h sans sauvegarde
de /kaggle/working). Run arrêté à 6/43. Causes (les deux miennes, en
dessous de la prédiction P-G4 ≥ 100) : (1) DEUX allers-retours GPU par
token pour les réservoirs d'abstention — latence de synchronisation
réelle ~3-5 ms pièce, estimée 0.3 ; (2) promotion float64 héritée de la
clé 4-gram (`_graw` en float64 par défaut numpy) qui double le coût des
matvecs et écritures CPU. Correctif v4 : réservoirs échantillonnés 1
token sur 8 en mode torch (le quantile roulant sur 5000 échantillons ne
change pas d'estimateur sur flux stationnaire — tolérance sur les
seuils vérifiée localement, borne 10 %) + clés castées float32 en mode
torch (dérive classe 1e-7/écriture, bornes M inchangées). Mode exact
intact (bit-à-bit, K=1). Estimation corrigée : ~250-330 tok/s → 6-8
min/question. Les 6 lignes de v3 sont perdues avec la session (Kaggle
ne persiste la sortie qu'à la complétion) — anecdote de la ligne 6
conservée ici : knowledge-update, a=False b=True c=True ev3=True,
102 674 tokens — première photo des trois voix, cohérente avec
P-G1/P-G2/P-G3.

**VERDICT bras G (kernel v8, T4, 43/43 questions, 4.5 M tokens ingérés
en 4.1 h, results/lme_arm_g.json) — 4 prédictions sur 4 :**

| voix (40 hors _abs) | score |
|---|---|
| (a) mémoire seule | **5 %** |
| (b) contexte + mémoire | **25 %** |
| (c) contexte seul | **25 %** |
| evidence@3 (écho) | 85 % |

- P-G1 **CONFIRMÉE** (5 % ≤ 10 %) — le négatif prédit par les lois :
  121k tokens stockés ne se convertissent pas en réponse libre à 0.6B.
- P-G2 **CONFIRMÉE au-delà** : b ≡ c non seulement en moyenne mais
  **question par question (accord 40/40, zéro divergence)** — le gate
  tarife la redondance en fenêtre à zéro exactement ; la mémoire est
  parfaitement neutre en régime RAG. Falsification jamais approchée.
- P-G3 **CONFIRMÉE** : 25 % = 5× la voix (a) — la formulation se fait
  en fenêtre.
- P-G4 **CONFIRMÉE** : 305.8 tok/s médian (min 302 / max 310 —
  d'une stabilité remarquable) = **×43 vs la lecture pleine** (7
  tok/s). `fast_ingest_blocked` validé à l'échelle : 4.5 M tokens.
- Par type, le gradient de formulation reproduit le bras E :
  single-session-user 67 % > assistant 40 % > multi-session 20 % ≈
  knowledge-update 17 % > temporal 10 % > preference 0 %.
- Voix (a) : 2 réussites, toutes deux temporal-reasoning — à inspecter
  (chance de prior probable, 2/40 sous le bruit).
- Les 3 _abs : silence des trois voix (aucun match) ✓.

**AXE 3 BOUCLÉ** : six lois locales (papier 6, publié), montée en
capacité (loi 1 plate sur ×6.7 — behav_4b.json), banc externe deux voix
(extraction 92.6 %/écart 63 pts — lme_arm_e ; génération a/b/c 5/25/25 —
lme_arm_g). L'histoire complète en une phrase : le sillage retrouve
presque toujours, ne formule presque jamais seul, et ne coûte jamais
rien quand le contexte est là.

---

## 28/08/2026 — Attaque du mur sémantique, marche 1 : DIAGNOSTIC — PRÉDICTIONS

Question : où meurt exactement le chemin paraphrase ? Sonde read-only
(probe_semantic_diag.py) sur l'état comportemental qwen existant
(.behav_state_qwen, sémantique actif), mu snapshoté/restauré (sem_key
mute la moyenne courante). Pour chaque fait stable (20), aux positions
de valeur des préfixes A (canonique) et B (paraphrase) : scores du tier
sémantique (valeur, max, rang, seuil), et du tier n-gram en contrôle.

**Prédictions (avant le run) :**
- P-S1 : au préfixe A, le tier sémantique SEUL classe la vraie valeur
  top-1 pour < 30 % des faits — c'est un lisseur de vraisemblance
  (rôle papier 2), pas un porteur de faits ; c'est M_G qui rappelle.
- P-S2 : au préfixe B, le max sémantique passe le seuil pour < 50 %
  des faits ET la valeur est top-1 < 10 % — le mur est dans les CLÉS,
  pas dans le mixage.
- P-S3 : le score sémantique de la vraie valeur chute nettement de A
  à B (désalignement de clé quantifié, > 50 % de perte médiane).
- **FALSIFICATION (grave si vraie)** : valeur top-1 sémantique ≥ 50 %
  au préfixe B → le mur serait un problème de confiance/mixage, PAS
  structurel — la loi 2 du papier 6 devrait être amendée. On teste
  notre propre loi.

Remèdes selon verdict : clés désalignées → marche 2 (clés ancrées
entité, re-lecture) puis marche 3 (encodeur externe gelé, axe 5) ;
abstention seule → readout sémantique dédié ; falsification → retour
au papier 6.

**VERDICT marche 1 (results/semantic_diag_qwen.json) :** plus dur que
prédit — le tier sémantique n'adresse JAMAIS les faits, même au préfixe
canonique : valeur top-1 0 % en A ET en B, rang médian ~71 600/151 936
(niveau hasard) en A, 18 208 en B ; au-dessus du seuil 20 %/0 %.
Contrôle n-gram : 100 % top-1 en A, 0 % en B (l'instrument est sain).
P-S1 dépassée (0 % < 30 %), P-S2 confirmée, P-S3 rendue sans objet par
les rangs (la perte A→B ne se mesure pas sur un signal déjà nul),
falsification jamais approchée. RELECTURE : le mur n'est pas un
désalignement de clé à la paraphrase — la clé n'a jamais pointé la
valeur. Les gains de vraisemblance du papier 2 = lissage diffus, pas
adressage. Cause mécanique supplémentaire notée : DÉRIVE DE MU (le
centrage courant évolue pendant la lecture → la même géométrie cachée
reçoit des clés différentes selon le moment d'écriture ; les sondes
utilisent le mu final).

## Marche 2 — clés ancrées : oracle entité + règle de surprise — PRÉDICTIONS

Prototype hors-outil (probe_semantic_anchor.py) : un tier M_E jetable
construit sur le dossier v1 (qwen), mu FIGÉ (précalculé en 2 passes),
clé de la position t = SimHash du hidden de l'ANCRE a_t, valeur =
token t+1, porte g_t inchangée. Deux règles d'ancrage :
  (2a) ORACLE : a_t = dernière occurrence d'un token d'entité (bornes
       hautes — les positions sont connues de l'instrument) ;
  (2b) RÈGLE LIVRABLE : a_t = dernier token de surprise g ≥ 2.5 nats
       (les entités inventées SONT les tokens surprenants — le signal
       gratuit choisit les ancres : le style de la série).
Sondes A/B : requête = clé du hidden de l'entité du prompt (2a) / du
dernier token surprenant du prompt (2b) → rang de la valeur dans M_E.
Baseline mesurée marche 1 : hasard.

**Prédictions (avant le run) :**
- P-M2a : oracle — valeur top-10 ≥ 50 % en A et ≥ 30 % en B (les
  hiddens d'une même entité s'alignent à travers les formulations).
- P-M2b : la règle de surprise reste à ≤ 15 points de l'oracle.
- **FALSIFICATION : oracle ≈ hasard en A aussi → la géométrie cachée
  du 0.6B ne supporte pas l'adressage inter-formulations, le remède
  est l'encodeur externe gelé (marche 3 / axe 5) et la refonte du
  tier meurt ici.**

**VERDICT marche 2 (results/semantic_anchor_qwen.json) : FALSIFICATION
DÉCLENCHÉE.** Oracle : A top-10 0 % (rang médian 46 394), B 0 %
(60 480) ; règle de surprise : idem (~75 800). Détail instructif : les
90 fins d'entité sont TOUTES g≥2.5 (le signal gratuit détecte
parfaitement les ancres) — c'est la CLÉ qui ne retrouve rien, pas
l'ancrage. La refonte par ancrage meurt ici, comme pré-enregistré.

## Marche 2c — géométrie ou fonction de clé ? — PRÉDICTIONS

Deux mesures (probe_semantic_dense.py), même prototype :
  (i) COSINUS BRUT : hidden de l'entité dans le doc vs dans les
      prompts A/B (blanchi mu figé), avec null inter-entités ;
  (ii) TIER DENSE : mêmes ancres oracle, même amp_write, mais clé =
      projection aléatoire FIXE du hidden blanchi, SANS quantisation
      SimHash (la machinerie M_S accepte tout q) — le test direct de
      « la quantisation est le tueur ».
**Prédictions :**
- P-M2c1 : cosinus même-entité doc↔prompt médian ≥ 0.5, null < 0.2
  (la géométrie porte le signal ; sinon → marche 3 directe).
- P-M2c2 : SI P-M2c1 tient, le tier dense fait A top-10 ≥ 50 % et
  B top-10 ≥ 30 % — la quantisation SimHash est le tueur, et le
  remède (clés denses) est livrable.
- Falsification : cosinus fort MAIS dense au hasard → le problème est
  dans l'écriture amplitude/interférence, pas dans la clé — retour à
  l'analyse.

**VERDICT marche 2c (results/semantic_dense_qwen.json) : LA GÉOMÉTRIE.**
Cosinus même-entité médian 0.441 (A) / 0.440 (B) — contre **0.467 pour
le null inter-entités** : le hidden blanchi de DERNIÈRE couche ne porte
aucun signal d'identité d'entité (tout est contexte gabarit partagé).
Tier dense : hasard aussi (cohérent — rien à préserver). P-M2c1
réfutée → ni la quantisation, ni l'ancrage, ni le mixage : la
géométrie de sortie. Trois réfutations instrumentées en escalier =
l'explication mécanique la plus profonde de la loi 2 et des gains
vraisemblance-seulement du papier 2 (la dernière couche est déjà
tournée vers le token suivant, pas vers l'identité).

## Marche 2d — l'identité vit-elle dans une couche intermédiaire ? — PRÉDICTIONS

probe_semantic_layers.py : même protocole cosinus (même-entité
doc↔prompt vs null inter-entités), balayé sur TOUTES les couches du
0.6B (28), A et B. Règle de sélection DÉCLARÉE (multiple
comparaisons) : la meilleure couche est choisie sur la séparation en
A, et VALIDÉE sur B — jamais l'inverse.
**Prédictions :**
- P-M2d : au moins une couche intermédiaire (tiers central du réseau)
  sépare même-entité vs null d'au moins +0.15 de cosinus médian en A,
  ET la même couche maintient ≥ +0.10 en B.
- Si OUI → le remède est livrable : « clé sémantique sur la couche
  L » (changement d'un indice dans le tier + re-lecture) → marche 2e
  (tier M_E re-testé sur cette couche).
- Si NON (toutes les couches plates) → l'adressage inter-formulations
  n'existe nulle part dans ce modèle → marche 3, encodeur externe
  gelé (axe 5), seule voie restante — et le négatif en escalier est
  lui-même un résultat de papier 8.

**VERDICT marche 2d (results/semantic_layers_qwen.json) : P-M2d
CONFIRMÉE, au-delà.** Gradient d'identité MONOTONE à travers le réseau :
couche 0 (embeddings) delta +0.810, couche 1 +0.712, décroissance
régulière → couche 28 (celle du tier !) −0.002 en A, −0.054 en B. Le
réseau efface l'identité d'entité au fil des couches pour la tourner
vers le token suivant — le tier sémantique lisait la seule couche où
elle est morte. Sélection déclarée : couche 1 (meilleure sur A,
+0.712), VALIDÉE sur B (+0.714). Portée déclarée : l'ancre étant le
token d'entité, la clé précoce est robuste au CADRE paraphrasé (l'axe
B de l'instrument), pas aux synonymes d'entité.

## Marche 2e — M_E sur la couche 1 : le tier adresse-t-il enfin ? — PRÉDICTIONS

probe_semantic_l1.py : le prototype M_E de la marche 2, mais clés sur
les hiddens de COUCHE 1, quatre variantes : {ancre oracle, ancre
surprise g≥2.5} × {SimHash bandé (la fonction du tier livré), clé
dense}. Mêmes 20 faits stables, rangs A/B.
**Prédictions :**
- P-M2e1 : oracle+dense — A top-10 ≥ 60 %, B top-10 ≥ 50 % (le signal
  +0.71 doit enfin se convertir en adressage).
- P-M2e2 : la variante SimHash reste à ≤ 20 points du dense (la
  quantisation survit à un signal fort) — si elle décroche, le tier
  livrable passera en clés denses.
- P-M2e3 : l'ancre surprise reste à ≤ 15 points de l'oracle (90/90 de
  recouvrement mesuré en marche 2).
- Falsification résiduelle : tout ≈ hasard malgré +0.71 de cosinus →
  le problème serait dans l'écriture amplitude (interférence des ~145
  tokens écrits sous chaque ancre) — remède : n'écrire sous l'ancre
  que les tokens de FORTE porte (le signal gratuit filtre aussi les
  valeurs).

**VERDICT marche 2e (results/semantic_l1_qwen.json) : falsification
résiduelle DÉCLENCHÉE** — les 4 variantes au hasard (médianes 53-92k)
malgré le cosinus +0.71. Mécanisme identifié : DIAPHONIE. Une matrice
à superposition exige des clés quasi orthogonales (les hypervecteurs
VSA de M_G, cos≈0) ; les clés couche-1 ont un cos null de 0.35-0.47
entre entités → ~2 600 écritures corrélées écrasent les 3 écritures de
la valeur. La « whitening » utilisée (soustraction de moyenne = rang 1)
ne décorrèle pas — c'est LE sens profond du « raw hidden states need
whitening » du papier 2.

## Marche 2f — blanchiment ZCA complet — PRÉDICTIONS

probe_semantic_zca.py : clés couche 1 = P · ZCA(h) avec ZCA =
Cov^{-1/2} (eigh, rétrécissement 0.1) estimée sur les hiddens du
dossier ; re-mesure cosinus même/null, puis tier dense oracle.
**Prédictions :**
- P-M2f1 : le ZCA écrase le null (médiane ≤ 0.15) en préservant le
  même-entité (≥ 0.35) — séparation multiplicative, pas additive.
- P-M2f2 : SI P-M2f1, le tier dense-ZCA adresse : A top-10 ≥ 50 %,
  B top-10 ≥ 40 %.
- Falsification : le ZCA écrase les deux (même-entité ≤ 0.2) → le
  signal d'identité VIVAIT dans le sous-espace partagé (il n'est pas
  séparable linéairement) → marche 3, encodeur externe.

**VERDICT marche 2f + INCIDENT (results/semantic_zca_qwen.json) :**
P-M2f1 CONFIRMÉE au-delà — cos ZCA même-entité 0.914, null 0.018 (A et
B identiques) : le blanchiment complet donne des clés quasi parfaites,
invariantes au cadre. P-M2f2 : rangs ~95 000, PIRE que le hasard →
signature d'un signal absent, pas faible → traque → **BUG DANS LES
PROTOTYPES (pas l'outil)** : décalage d'indice des portes — le token
t+1 était écrit avec la surprise du token t ; après « requires » la
surprise est ~0 → les VALEURS étaient écrites à amplitude quasi nulle
dans les marches 2, 2e et 2f (les mesures de cosinus, sans porte, sont
saines ; fast_ingest indexe correctement). Correctif : G[t+1] pour
l'écriture de ids[t+1]. Les trois expériences de récupération sont à
re-courir ; les prédictions restent celles déjà enregistrées.

**VERDICT marche 2f corrigée (results/semantic_zca_qwen.json) : LA
CHAÎNE EST COMPLÈTE.** Clés ZCA : même-entité 0.914 / null 0.018 (A et
B identiques — P-M2f1 ✓). Récupération dense-ZCA oracle : **A top-10
95 % / B top-10 95 %, rang médian 3** sur 151 936 (P-M2f2 ✓, largement)
— LE MUR PARAPHRASE EST CASSÉ AU NIVEAU DE L'ADRESSAGE : le préfixe B
adresse comme le A. Top-1 15 % : les premiers rangs sont les autres
tokens de la phrase écrits sous la même ancre (« protocol »,
« requires ») — attendu, le top-10 est la bonne métrique au niveau
tier ; le readout/mixage fera le tri comportemental. Outlier à
inspecter : Dulcifern (rang ~1100). Recette établie : COUCHE 1 + ANCRE
ENTITÉ + ZCA + CLÉ DENSE. L'escalier complet (5 réfutations
instrumentées → 1 incident → 1 percée) = la colonne du papier 8.

## Marche 2g — matrice de livrabilité 2×2 — PRÉDICTIONS

zca étendu : {ancre oracle, ancre surprise g≥2.5} × {clé dense-ZCA,
SimHash-sur-ZCA}. Et re-run de 2e corrigée (couche 1 SANS ZCA) comme
contrôle de nécessité du blanchiment.
**Prédictions :**
- P-M2g1 : surprise ≈ oracle à ≤ 10 points (recouvrement d'ancres
  90/90 mesuré) → l'ancrage est automatisable par le signal gratuit.
- P-M2g2 : SimHash-sur-ZCA ≥ dense − 20 points (la quantisation
  bandée tient sur des clés décorrélées) → compatibilité maximale
  avec le tier livré.
- P-M2g3 (contrôle) : couche 1 SANS ZCA reste ≈ hasard → le
  blanchiment complet est NÉCESSAIRE, pas cosmétique (et le mu-only
  du tier actuel est insuffisant par construction).

**VERDICT marche 2g (semantic_zca_qwen.json + semantic_l1_qwen.json
corrigés) :** oracle+dense-ZCA 95/95 ; oracle+SimHash-ZCA 80/80 (P-M2g2
✓) ; **oracle+SimHash SANS ZCA : 95/95 top-10, médiane 3 (P-M2g3
RÉFUTÉE — le ZCA est inutile, l'échec de 2e était entièrement le bug de
porte, et ma théorie de la diaphonie était FAUSSE : le SimHash bandé se
décorrèle par quantisation)** ; oracle+dense sans ZCA 85/85 (top-1
30 %). **P-M2g1 RÉFUTÉE : ancre de surprise 0 % partout** malgré la
détection 90/90 — avec 609 points ≥2.5 (vs 90 entités), l'ancre glisse
(la valeur, surprenante, vole l'ancre côté doc ; « requires » côté
prompt). Recette minimale restante : COUCHE 1 + SimHash EXISTANT + une
RÈGLE D'ANCRAGE correcte. Théories corrigées honnêtement : diaphonie
réfutée par l'expérience, ZCA relégué au rang d'option (+0-15 pts pour
le dense).

## Marche 2h — la règle d'ancrage — PRÉDICTIONS

probe_anchor_rules.py : instrumenter le CHOIX d'ancre (token décodé,
doc et prompts, vs oracle) puis tester : r1 = dernier g≥2.5 (la règle
échouée) ; r2 = max-g d'une fenêtre glissante de 16 ; r3 = dernier
g≥4.0. Métrique intermédiaire : anchor-accuracy (% des écritures de
tokens de valeur ancrées sur LEUR entité ; % des requêtes ancrées sur
l'entité du prompt). Puis récupération (SimHash sans ZCA, la recette
minimale) avec la meilleure règle.
**Prédictions :**
- P-M2h1 : l'anchor-accuracy de r1 est < 50 % d'un des deux côtés —
  elle explique le 0 %.
- P-M2h2 : au moins une règle raffinée atteint ≥ 80 % d'anchor-accuracy
  des deux côtés ET ≥ 70 % top-10 en récupération A/B.
- Si toutes échouent : l'ancrage automatique passe par un détecteur
  dédié (fréquence de token, majuscules, NER-léger) — ingénierie
  supplémentaire, pas un mur théorique.

**VERDICT marche 2h (results/semantic_rules_qwen.json) : P-M2h1
confirmée, verrou localisé CÔTÉ REQUÊTE seulement.** Écriture : r1
(g≥2.5) ancre 92 % des valeurs sur leur entité (86/93) — dans le doc,
les répétitions rendent « protocol requires » prévisible, le signal
trie. Requête : 0/40 pour les trois règles — dans un prompt nu TOUT
est surprenant (« requires » vole l'ancre en A, « requirement » en B),
et le max de surprise d'une entité est son PREMIER sous-token. Le
seuil absolu de surprise est contexte-dépendant : inutilisable en
prompt court.

## Marche 2i — pooling de requête — PRÉDICTIONS

probe_query_pooling.py : écriture = r1 (92 % validée) ; requête = clés
de TOUTES les positions du prompt, score final = max par token sur les
positions. Zéro choix d'ancre à la requête. SimHash sans ZCA (recette
minimale).
**Prédictions :**
- P-M2i1 : A top-10 ≥ 80 % et B top-10 ≥ 70 % (la position qui matche
  se retrouve d'elle-même ; le pooling ajoute un plancher de bruit
  borné par le nombre de positions).
- P-M2i2 : le max est atteint à une position d'entité du prompt pour
  ≥ 80 % des faits (vérification du mécanisme, pas seulement du
  score).
- Si échec : le bruit de pooling écrase le signal → requête à deux
  étages (pré-filtre par score max de position) avant d'abandonner.

**VERDICT marche 2i (results/semantic_pooling_qwen.json) : RECETTE
COMPLÈTE, SANS ORACLE.** A top-10 100 % / B top-10 100 %, rang médian
2, top-1 ~45-50 %, et le max de pooling tombe sur un token d'ENTITÉ
dans 40/40 cas (P-M2i1 et P-M2i2 confirmées au-delà). LA RECETTE :
**couche 1 + ancres de surprise à l'écriture (g≥2.5, 92 % write-acc)
+ SimHash EXISTANT (mu seul, sans ZCA) + pooling de requête (max sur
les positions du prompt — zéro choix d'ancre à la requête).** Du
« le tier n'adresse jamais, paraphrase 0 % structurel » au « rang
médian 2, invariant à la formulation » en une campagne d'un jour :
9 marches, 5 réfutations, 1 bug instrumenté, 1 percée — sans gradient,
presque entièrement avec la machinerie existante.

RESTE pour la revendication complète (campagne papier 8) :
1. Validation COMPORTEMENTALE : readout/mixage → complétion (la
   métrique de la loi 2 : % de rappel paraphrase généré) ; calibrer
   β/λ/seuil du nouveau tier.
2. La batterie complète sur le nouveau tier : localité (témoin !),
   rétention, conflits, interférence, au-delà des 20 faits, texte non
   inventé, GPT-2 (couche ? re-sweep), coût d'ingestion (les hiddens
   de couche 1 sont déjà calculés — quasi gratuit ; le pooling coûte
   × positions du prompt à la requête).
3. Intégration outil (1.4.0) : sélection de couche, suivi d'ancre en
   lecture, pooling dans complete/ask.

---

## 28/08/2026 soir — Étape A/B : VALIDATION COMPORTEMENTALE du tier v2 — PRÉDICTIONS

Décision (utilisateur) : finir de franchir le mur avant l'axe 4. La
loi 1 s'applique à nous : rang médian 2 = stockage, pas comportement.

probe_behavioral_v2.py : tier 2i (couche 1, ancres r1, SimHash sans
ZCA, mu figé) construit sur le dossier v1 ; à la GÉNÉRATION, mixage
p' = (1−λ)·p_base + λ·softmax(β·sE_poolé) où sE_poolé = max des
scores sur les positions du prompt, DÉCLENCHÉ seulement si
max(sE) ≥ thr. Règles anti-surapprentissage DÉCLARÉES : (β, λ)
choisis par grille sur les 10 faits DEV (les entités « changed », en
valeurs v1) ; mesure finale sur les 20 faits STABLES jamais vus par la
grille ; thr = q95 des maxima poolés sur 20 prompts témoins (null,
hors faits). Scorer = mot de tête de la valeur dans 8 tokens greedy
(la métrique du papier 6).
**Prédictions :**
- P-A1 : rappel PARAPHRASE (préfixe B) ≥ 60 % sur les 20 stables
  (contre 0 % pour le tier actuel, toutes conditions) ; le préfixe A
  reste ≥ 90 % (le mixage n'abîme pas le canonique).
- P-A2 (localité) : ≤ 10 % des 20 prompts témoins changent leur
  continuation greedy sous le mixage (l'abstention par thr tient).
- **FALSIFICATION : B < 30 % → le stockage rang-2 ne se convertit
  pas — la loi 1 mord aussi le tier v2, et l'étude devient
  confiance/mixage avant toute intégration.**

**VERDICT étape A/B (results/semantic_behavioral_v2.json) :**
paraphrase B **0 % → 25 %** en génération (base 0 % mesurée même
passe) — la conversion existe — MAIS falsification déclenchée (25 <
30) : P-A1 manquée (≥60 prédit). Le plafond n'est PAS le mixage (grille
plate à 30 % de β10-40 × λ0.2-0.85). Localité ✓ 0/10. A=20 % : tier v2
SEUL, sans M_G/cold (non alarmant, hors périmètre). Mécanisme suspecté
du plafond : les rangs 2i montraient top-1 ~45 % — la valeur perd le
greedy contre les tokens de CADRE (« protocol », « requirement »)
écrits sous la même ancre à amplitude comparable.

## Étape A2 — filtrage des écritures par la porte — PRÉDICTIONS

probe_behavioral_v2 étendu : sous une ancre, n'écrire que les tokens
g ≥ g_min (les cadres s'effondrent dès la rép. 2, les valeurs restent
surprenantes — le signal gratuit, 4e emploi). g_min ∈ {0, 0.5, 1.0,
2.0} choisi sur les 10 faits DEV (préfixe B), mesure finale 20 stables
— même protocole anti-surapprentissage.
**Prédictions :**
- P-A2a : le meilleur g_min > 0 porte le dev B à ≥ 50 % ET le test B
  à ≥ 50 % (le cadre sort du bucket, la valeur gagne le greedy).
- P-A2b : la localité reste ≤ 10 % de témoins changés.
- Falsification : test B < 35 % même filtré → le plafond est ailleurs
  (multi-token ? collision de bucket ?) → instrumenter les
  complétions elles-mêmes avant toute nouvelle règle.

**VERDICT étape A2 (results/semantic_behavioral_v2.json) :** le filtre
d'écriture par la porte AIDE (dev B 30→40 % à g_min 0.5 ; 2893→713
écritures ; test B 25→30 %) mais **falsification déclenchée à nouveau
(30 < 35)**. Localité toujours 0/10. Deux falsifications de suite sur
la conversion : le plafond ~30-40 % n'est ni le mixage ni (seulement)
les tokens de cadre. PROCHAINE ÉTAPE OBLIGÉE (pré-enregistrée) :
instrumenter les complétions elles-mêmes — imprimer ce que le greedy
produit réellement fait par fait (qui gagne ? le cadre ? un
multi-token cassé ? une collision de bucket ?) avant toute nouvelle
règle. À faire en début de prochaine session, à tête reposée.

BILAN DU MUR au 28/08 soir : adressage FRANCHI (rang médian 2,
invariant, sans oracle) ; conversion comportementale 0 → 30 % (réelle
mais plafonnée — la loi 1 mord le tier v2 comme elle mordait M_G, et
c'est désormais MESURÉ) ; localité intacte partout. Le papier 8 a déjà
son arc complet quoi qu'il arrive : escalier de réfutations →
géométrie par couches → recette d'adressage → le plafond de conversion
comme frontière ouverte (ou franchie, selon la suite).

---

## 29/08/2026 — Étape 1 : DIAGNOSTIC des complétions — PRÉDICTIONS

Décision (utilisateur) : étapes 1→2→3 puis papier 8 puis 1.4.0.

probe_diag_completions.py : config gelée (g_min 0.5, β10, λ0.85, thr
q95), les 30 prompts B (20 test + 10 dev, étiquetés) ; pour CHAQUE
fait : la complétion greedy complète, et au pas 0 : rang de la valeur
dans p_base / p_sem(poolé) / p_mix, top-3 de p_sem décodé, sE a-t-il
tiré. Catégories de ratés : (a) p_sem vise un token du bucket ≠ valeur
(cadre ou co-phrase) ; (b) p_sem vise la valeur mais p_mix la perd
(arithmétique de mélange) ; (c) déraillement après un bon départ ;
(d) sE n'a pas tiré (abstention) ; (e) collision inter-buckets (valeur
d'une AUTRE entité).
**Attentes (diagnostic exploratoire, enregistrées pour l'honnêteté) :**
- P-D1 : la majorité des ratés est en (a) — le bucket contient encore
  des concurrents à amplitude comparable malgré le filtre.
- P-D2 : (d) est rare (≤ 2/30) — le pooling tire presque toujours.
- Le remède 2 sera choisi sur ces catégories ; candidat prêt si (a)
  domine : SUPPRESSION D'ÉCHO à la requête — atténuer dans p_sem les
  tokens déjà présents dans la fenêtre (principe maison : ce qui est
  dans la fenêtre est gratuit, le rappel ne doit payer que l'absent).

**VERDICT étape 1 (results/semantic_diag_completions.json) — le
diagnostic paie :** hits 10, « bucket » 13, « mixage » 7, déraillement
0, abstention 0, collision 0 (P-D2 ✓). MAIS la lecture des sorties
requalifie tout : ① les 13 « bucket » = LES SOUS-TOKENS DE L'ENTITÉ
ELLE-MÊME (surprenants → écrits fort ; 'une'/'lag', 'rix', 'wick',
'oval'…) volent le greedy, valeur aux rangs sem 2-5 juste derrière —
or ils sont DÉJÀ DANS LE PROMPT → l'écho-suppression les tue ; ② les
7 « mixage » = MOTS COUPÉS : la bonne tête émise (' salt', ' mint',
' wax', ' chalk'…) mais la pièce de continuation ('ed','y'), prévisible
après sa tête, avait g<0.5 → supprimée par MON filtre d'écriture. Deux
remèdes chirurgicaux, zéro mystère restant.

## Étape 2 — écho-suppression + intégrité de mot — PRÉDICTIONS

probe_behavioral_v3.py : config 2i + g_min 0.5, plus :
  R1 ÉCHO : à la requête, p_sem[token présent dans le prompt] = 0
  (principe maison : ce qui est dans la fenêtre est gratuit — le
  rappel ne paie que l'absent) ;
  R2 INTÉGRITÉ DE MOT : à l'écriture, garder t+1 si g≥0.5 OU si t+1
  est une pièce sans-espace dont la tête (t) a été gardée.
Même protocole dev/test/null. Grille (β,λ) sur dev seulement.
**Prédictions :**
- P-E2a : test B ≥ 60 % (13 échos + une partie des 7 mots coupés
  convertis) ; dev B ≥ 60 %.
- P-E2b : localité toujours ≤ 10 %.
- **FALSIFICATION : test B < 45 % → un 3e mécanisme se cache — retour
  au diagnostic, pas de nouvelle règle à l'aveugle.**

**VERDICT étape 2 (results/semantic_behavioral_v3.json) :** écho +
intégrité de mot → dev B 40→**80 %**, test B 30→**55 %** (base 0 %),
A-seul 45 %, localité 0/10 ✓. Falsification (<45) NON déclenchée ;
prédiction (≥60) manquée de 5 points — zone intermédiaire honnête.
Trajectoire complète : 0 → 25 → 30 → 55 %. Écart dev/test (80/55) à
élucider. Clôture d'étape : catégoriser les 9 ratés test restants
(même instrumentation, tier v3) — le papier 8 posera la question.

## Étape 3 — réplication GPT-2 — PRÉDICTIONS

probe_gpt2_replication.py : le protocole complet sur GPT-2 (12
couches) : ① balayage de couches (règle déclarée : meilleure sur A,
validée sur B) ; ② recette v3 (ancres g≥2.5, intégrité de mot, écho,
pooling) sur la couche choisie ; ③ rangs A/B + comportemental B +
localité.
**Prédictions :**
- P-E3a : le gradient d'identité par couche existe aussi sur GPT-2
  (delta ≥ +0.3 à une couche précoce, ≈0 à la dernière) — la
  découverte est architecturale, pas un accident Qwen.
- P-E3b : rangs top-10 ≥ 80 % en A et B sur la couche choisie ;
  comportemental B ≥ 40 % (GPT-2 124M est plus faible générativement).
- Falsification : gradient absent sur GPT-2 → la découverte est
  spécifique à Qwen3 — à déclarer comme telle au papier 8.

**VERDICT étape 3, phase 1 (results/semantic_gpt2_replication.json) :**
P-E3a **CONFIRMÉE — le gradient d'identité est ARCHITECTURAL** : sur
GPT-2 il culmine en couche 5 (+0.356) et la dernière couche est
ANTI-CORRÉLÉE (−0.52/−0.56, plus dramatique que Qwen). P-E3b RÉFUTÉE :
la recette v3 nue ne transfère pas (rangs A top-10 20 %, comportemental
0 %) — cause lisible : séparation +0.36 = moitié de Qwen (+0.71), trop
faible pour le SimHash nu.

## Étape 3, phase 2 — le ZCA comme composant modèle-dépendant — PRÉDICTIONS

probe_gpt2_zca.py : même protocole GPT-2 couche 5, clés dense-ZCA
(rétrécissement 0.1, stats du dossier).
**Prédictions :**
- P-E3c : cos ZCA même-entité ≥ 0.6, null ≤ 0.1 (le blanchiment
  remonte la séparation comme sur Qwen 0.44→0.91).
- P-E3d : rangs top-10 ≥ 60 % A et B ; comportemental test B ≥ 30 %
  (GPT-2 124M, génération faible).
- Si confirmé : LA RECETTE GÉNÉRALE = couche optimale (balayage) +
  ancres + intégrité de mot + pooling + écho + ZCA-si-nécessaire (le
  blanchiment est la pièce adaptative — cohérent avec le papier 2 :
  « raw hidden states need whitening except where the geometry is
  already well conditioned »).
- Si réfuté : GPT-2 = limite déclarée de la recette au papier 8.

**VERDICT étape 3 phase 2 (results/semantic_gpt2_zca.json) — LES TROIS
ÉTAPES SONT CLOSES.** ZCA sur GPT-2 c5 : séparation 0.36 → **0.729 /
0.064** (P-E3c ✓) ; rangs A 95 % / B 90 % top-10, médiane 3 (P-E3d ✓) ;
comportemental dev 80 %, **TEST B 60 %** (base 0 %) — dépasse Qwen
(55 %). Localité 1/10 (borne). LA RECETTE GÉNÉRALE : couche optimale
par balayage (Qwen c1 / GPT-2 c5) + ancres de surprise g≥2.5 +
intégrité de mot + SimHash + BLANCHIMENT-SI-NÉCESSAIRE (la pièce
adaptative, = la clause du papier 2 mot pour mot) + pooling de requête
+ écho-suppression. Paraphrase générative 0 → 55-60 % sur DEUX
modèles, localité intacte. La loi 2 du papier 6 est AMENDÉE : le mur
n'était pas structurel au système, il était structurel au CHOIX DE
COUCHE — l'identité vit tôt, la mémoire lisait tard. Le signal gratuit
compte désormais 5 emplois (porte, consolidation, défense, ancrage,
filtrage). → RÉDACTION PAPIER 8, puis intégration 1.4.0 (mandat
utilisateur).

---

## 29/08/2026 — Intégration 1.4.0 : trois verrous du chemin LIVRÉ

Le prototype marchait (B 55-60 %) mais `--sem2` livré donnait B 0-1/10.
Instrumentation, pas théorie — trois causes distinctes, dans l'ordre :

1. **Statistiques immatures à l'écriture (la vraie cause).** Le
   prototype calculait mu (et le ZCA) sur UNE PASSE COMPLÈTE avant de
   construire le tier ; le chemin livré les accumulait en ligne et
   écrivait chaque clé avec le mu de l'instant → les premières
   écritures utilisaient un mu quasi vide, la requête le mu final :
   clés incohérentes. Correctif : **écritures du tier différées et
   consolidées en fin de document** (« sleep ») avec les statistiques
   mûres — les mêmes que la requête dérivera. Effet de bord majeur :
   l'eigendécomposition du blanchiment est payée une fois par flush au
   lieu d'une fois par token (lecture 4 189 tokens : ~10 min → 71 s).
   Buffer borné (8 192 ancres, flush intermédiaire) ; null décimé.
2. **Réglages du readout S.** Ceux du papier 2 (β40/λ0.1) étaient
   calibrés pour un tier qui lissait la vraisemblance ; le tier v2
   adresse → constantes mesurées du papier 8 (β10/λ0.85), la
   calibration gardant la priorité (précédent `--target`).
3. **Seuil d'abstention** (en cours) : le diagnostic montre que
   l'adressage LIVRÉ fonctionne (valeur rang 1,1,2,4 sur 5 faits) mais
   que 3/5 s'abstiennent. **Mon hypothèse « il faut des clés denses
   sur GPT-2 » est RÉFUTÉE par cette mesure** — le SimHash+ZCA adresse
   très bien ; c'est le référentiel du seuil qui est mauvais (null
   échantillonné APRÈS écriture = « à quel point le document ressemble
   à lui-même », pas « à quel point une requête étrangère ressemble à
   ce que j'ai lu »). Mesure en cours : scores poolés des prompts de
   faits vs prompts témoins, pour placer le quantile.

**Verrou 3 REQUALIFIÉ (mesure du seuil) :** le rappel est PLAT à 2/20
de q70 à q95 et la localité 0/10 partout → **ce n'est pas le seuil non
plus**. Or l'adressage livré est excellent : valeur top-1 13/20, top-3
17/20, top-10 19/20 sur les prompts PARAPHRASÉS (contre ~0 pour le
tier historique). Le verrou est donc le MIXAGE.

**Prédiction chiffrée (écrite avant le run) :** dans un mélange convexe
à λ=0.85, le token de la mémoire ne gagne l'argmax que s'il porte
~16 % de la masse de p_sem = softmax(β·s) ; sur V=50 257 tokens cela
exige β·(s_max − s_typique) > ln V ≈ 10.8. Les scores poolés mesurés
donnent un écart ≈ 0.2 → **β > 54** ; à β=10 le produit vaut 2, la
distribution est plate et AUCUN seuil ne peut aider — ce qui explique
exactement le plateau. Cause racine : β=10/λ=0.85 vient du prototype à
clés DENSES (scores étalés) ; le tier livré utilise le SimHash (scores
tassés) — l'échelle de β suit la fonction de clé, comme les β publiés
de la série varient de 20 à 160 selon le tier. Test : grille β ∈
{10..160} × λ ∈ {0.5, 0.85} sur 10 faits DEV, rapport sur 10 faits
TEST + localité 10 témoins (probe_ship_readout.py).

**Verrou 3 CONFIRMÉ puis raffiné (results/ship_readout_gpt2.json) :**
β=10 → 1/10 dev ; β=20 → 4/10 ; plateau ensuite. Prédiction β>27
(écart mesuré max 0.500 / q99 0.194 / médiane 0.093) vérifiée. TEST
B 4/10 = **40 %** (contre 0 % au tier historique), localité 2/10 —
au-dessus de ma borne 1/10. Les sorties nomment le dernier mécanisme :
la mémoire pousse le PREMIER SOUS-TOKEN de la valeur à chaque pas
(« tur tur tur » pour « turquoise »), donc les valeurs multi-tokens
échouent au scorer et la perturbation dure toute la génération.

**Prédiction (avant le run) :** un mixage en IMPULSION (le tier n'agit
qu'au premier pas ; le modèle gelé finit le mot, ses pièces étant
prévisibles une fois la tête sortie — le jumeau génération de
l'intégrité de mot à l'écriture) fait ≥ le mixage soutenu sur le
rappel B ET ≤ sur la localité. Variante intermédiaire testée :
décroissance λ·0.5^pas.

**Verdict impulsion + constantes livrées (grille finale, GPT-2) :**

| seuil | mixage | rappel B (test) | localité |
|---|---|---|---|
| q90 | soutenu | 40 % | 2/10 |
| q90 | **impulsion** | **30 %** | **1/10** |
| q95 | soutenu | 20 % | 1/10 |
| q95 | impulsion | 30 % | 1/10 |

Prédiction impulsion : **réfutée sur le rappel brut** (30 < 40),
**confirmée sur la localité** (1 < 2) — et sous la contrainte déclarée
(localité ≤ 1/10) c'est elle qui gagne : 30 % contre 20 % au soutenu.
Formule retenue et livrée : **le tier n-gram continue, le tier
sémantique rappelle** — une seule impulsion au premier token généré,
le modèle gelé finit le mot. Constantes 1.4.0 mesurées DANS L'OUTIL
(grille sur 10 faits dev, rapport sur 10 faits test) : couche par
`--sem2`, β_S 20, λ_S 0.85, seuil q90 du null in-document.
**Chemin livré, pipeline complet : paraphrase 3/10 (30 %) contre 0 %
au tier historique, canonique 10/10.** L'écart avec les 55-60 % du
papier 8 est une différence de configuration à déclarer : le papier
mesure des prototypes (clés DENSES sur GPT-2, seuil sur témoins
externes), l'outil livre le SimHash bandé avec un null in-document —
même recette, réglages et échelle différents. Bug corrigé au passage :
un état sem2 forçait `semantic=True` et écrasait `--no-semantic`.

**VALIDATION CROISÉE QWEN (constantes choisies sur GPT-2, aucun réglage
sur qwen) :** couche 1, sans blanchiment — **paraphrase 8/10 (80 %)
avec le tier, 0/10 sans (contrôle apparié), canonique 9/10, localité
0/10**, lecture 4 715 tokens en 2,3 min. Le transfert est meilleur que
le modèle de calibration (GPT-2 : 30 %, localité 1/10) — cohérent avec
la séparation d'identité mesurée (qwen +0.71 en couche 1 contre +0.36
en couche 5 sur GPT-2). L'INTÉGRATION 1.4.0 EST VALIDÉE SUR DEUX
MODÈLES : la loi 2 du papier 6, mesurée dans l'outil livré, passe de
0 % à 30-80 % selon le modèle, localité intacte.

---

## 29/08/2026 — Dette technique avant l'axe 4 (l'utilisateur a tranché :
## la traiter d'abord) — PRÉDICTIONS

Trois chantiers, chacun avec son critère de succès écrit d'avance.

**C1 — dé-pickliser l'état.** cold/calib → npz CSR et tableaux plats,
index → JSON (les vecteurs TF-IDF se reconstruisent). Migration unique
des états pré-1.5 avec avertissement, puis suppression du pickle
(SILLAGE_NO_PICKLE=1 refuse). Critère : aucun .pkl écrit, round-trip
identique, un état ancien migre. **FAIT — T15/T16 verts.**

**C2 — `--sem2` dans le chemin rapide.** La consolidation différée du
tier v2 a exactement la forme du mode bloqué ; factorisation de
`sem2_flush` dans core pour que les deux chemins construisent LE MÊME
tier. Prédiction : rappel paraphrase identique à ±1/10 entre lecture
normale et lecture rapide, et accélération ≥ ×3 sur ce document.

**C3 — `--sem2 auto`.** Le balayage de couches du papier 8 sans
annotation : un TOKEN RARE RÉPÉTÉ est la même identité dans deux
contextes, deux tokens rares différents sont le null. Le tool choisit
la couche qui maximise la séparation, et active le blanchiment si
elle est < 0.5 (qwen mesuré 0.71 → sans ; GPT-2 0.36 → avec).
Prédictions : (a) sur un vrai document, `auto` choisit une couche
BASSE, pas la dernière ; (b) il retrouve l'ordre de grandeur du papier
8 (GPT-2 vers 4-6, qwen vers 1-2) ; (c) il refuse proprement quand
rien ne se répète. **(c) FAIT — T17 vert** ; (a) et (b) en mesure.

**VERDICTS DES TROIS CHANTIERS.**

**C1 (dé-picklisation) ✓** — cold/calib en npz CSR, index en JSON
(vecteurs TF-IDF reconstruits au chargement), migration unique avec
avertissement puis suppression du pickle. T15/T16 verts. Un état
n'exécute plus rien ; reste la confidentialité (un cold store révèle
le texte lu), qui est un autre sujet.

**C2 (`--sem2` dans le chemin rapide) ✓ au-delà du critère** — le
tier construit par la lecture rapide est IDENTIQUE à celui de la
lecture normale (paraphrase 3/10 des deux côtés, canonique 10/10,
null 805, seuil 0.418, cold 1081 — pas « à ±1 », identiques). La
consolidation différée est factorisée dans `core.sem2_flush`, les deux
chemins l'appellent. Prédiction d'accélération ≥×3 RÉFUTÉE : ×1.9
seulement, et le profil dit pourquoi — 270 tok/s sans tier, 143 avec,
109 avec blanchiment : le tier coûte ses clés par token et le
blanchiment son accumulation de covariance (d² par token). Reste
×15-20 sur la lecture normale (7 tok/s), ce qui était l'enjeu.

**C3 (`--sem2 auto`) ✓ avec une limite mesurée et déclarée** — la
supervision gratuite (tokens rares répétés, requêtes fabriquées depuis
le document = le protocole du papier 8 automatisé) trouve bien le
gradient d'identité : sur texte naturel, qwen donne 1.01 → 0.43 (couche
1 choisie ✓). Mais sur GPT-2 le profil est PLAT (0.47-0.52) et le
balayage ne distingue pas la couche 5 (qui marche, 3/10) de la 6 (qui
ne marche pas, 0/10). Et AUCUN proxy bon marché ne prédit le besoin de
blanchiment : la séparation cosinus dit « inutile » pour GPT-2, le rang
de récupération d'un tier jetable aussi — les deux comparent deux
endroits d'un même document, ce qui n'est pas la question posée.
DÉCISION : `auto` applique la règle déjà en vigueur pour β/λ dans cette
série — **ce que les papiers ont mesuré gagne pour les modèles
mesurés** (SEM2_LAYER/SEM2_WHITEN : qwen 1/off, gpt2 5/on), le
balayage ne sert qu'aux modèles que personne n'a mesurés (et le
blanchiment y est activé par défaut, règle du papier 2). Résultat :
auto → gpt2 3/10, qwen 8/10, sans qu'on donne un numéro de couche.
Deux heuristiques inventées puis RETIRÉES parce qu'elles ne prédisaient
pas — elles ne sont pas dans le code livré.

---

## 29/08/2026 — AXE 4, étape 2 : que vaut un état SANS cold store ?
## (la mesure qui décide si les cartridges existent) — PRÉDICTIONS

Un état partagé ne peut contenir ni le cold store (table 4-grammes →
successeurs en tokens clairs) ni l'index (passages en clair). Reste les
matrices. La question, jamais posée : **que perd-on ?**

Protocole (probe_shareable_state.py) : un état lu une fois sur un vrai
document (papier 6, texte naturel) + le dossier de faits inventés ;
quatre configurations à la GÉNÉRATION, même état, rien de réécrit :
  A. complet (M_G + cold + sémantique)
  B. sans cold store
  C. sans index (pas d'injection de contexte — le régime `complete`)
  D. sans cold ni index = **ce qu'un cartridge publiable contiendrait**
Mesures : rappel canonique, rappel paraphrasé, PPL sur un extrait connu
du document (la métrique historique de la série), et taille sur disque.

**Prédictions (avant le run) :**
- P-C1 : la PPL sur texte connu se dégrade peu sans cold (< +15 %) — la
  matrice M_G porte l'essentiel de la vraisemblance ; c'est la rétention
  LONGUE qui dépendait du cold (papier 6), pas la lecture immédiate.
- P-C2 : le rappel canonique chute nettement sans cold (≥ 30 points) —
  le rappel exact d'un fait est précisément ce que le cold porte.
- P-C3 : le rappel paraphrasé (tier v2) est INDÉPENDANT du cold —
  il vient de M_S, donc un cartridge garde la capacité que le papier 8
  a débloquée.
- **DÉCISION liée : si P-C1 tient et P-C3 tient, un cartridge
  « matrices seules » a un intérêt réel (gain de vraisemblance +
  paraphrase) et l'item continue. Si les trois chutent, l'item devient
  une note négative.**

**VERDICT étape 2 — état partageable (results/shareable_{gpt2,qwen}.json) :
L'ITEM CARTRIDGES EST VIABLE.** Même état, tiers coupés à la génération
(comparaison appariée) :

| config (qwen) | PPL | canonique | paraphrase |
|---|---|---|---|
| A complet | 1.20 | 9/10 | 8/10 |
| B sans cold | 1.19 | 7/10 | **8/10** |
| C sans sémantique | 1.20 | 9/10 | **0/10** |
| D matrices seules (= cartridge) | **1.19** | **7/10** | **8/10** |

- P-C1 **confirmée au-delà** : la vraisemblance ne perd RIEN sans cold
  (1.20 → 1.19) — la matrice M_G la porte entièrement.
- P-C2 **RÉFUTÉE** : le canonique ne chute que de 9 à 7/10 (je prédisais
  ≥ 30 points de perte). GPT-2 : 10/10 → 9/10.
- P-C3 **confirmée** : le rappel paraphrasé est INDÉPENDANT du cold
  (8/10 des deux côtés) et vient entièrement du tier v2 (ligne C : 0/10
  sans lui). **Le cartridge conserve donc la capacité que le papier 8 a
  débloquée.**

**Piège découvert en implémentant `export`** : sur un état trop peu lu
(1,9k tokens, 237 positions notées < le plancher de 500), les tiers
matriciels S'ABSTIENNENT et le cartridge est MUET (0/8 alors que l'état
complet fait 8/8 — tout venait du cold). Diagnostic par diff : M, MS, A,
réservoirs, réglages tous identiques ; seul le cold manquait, et le
seuil valait inf. `export` avertit désormais quand `res_G` < 500 au lieu
de livrer un fichier silencieux. Les mesures ci-dessus (7k tokens) sont
au-dessus du plancher, d'où l'écart apparent entre les deux tests.

**VERDICT étape 4 — quantification (results/quantised_qwen_int8.json) :
NÉGATIF SUR CPU, et livré comme tel.** int8 dynamique de torch (197
couches linéaires quantifiées sur qwen, zéro dépendance ajoutée) :
- admissions du cold store **identiques** (483 vs 483, Jaccard 1.000) —
  la règle des deux occurrences survit intacte ;
- MAIS corrélation des portes **0.9735** < le seuil 0.98 déclaré
  d'avance → **prédiction réfutée, verdict « approximatif »** ; écart
  moyen 0.127 nat ;
- rappel **5/7 → 1/7** ;
- et **aucun gain de vitesse en lecture** (55,7 s → 60,9 s) : le goulot
  n'est pas le forward mais les écritures numpy.
Donc `--dtype int8` sert à FAIRE TENIR un plus gros modèle en mémoire,
pas à aller plus vite ni à lire quelque chose d'important — et l'outil
le dit lui-même à l'écran quand on l'active. bfloat16 en cours de
mesure (le mode réellement utile : moitié mémoire, perte attendue
faible).

**bfloat16 (results/quantised_qwen_bfloat16.json) : FIDÈLE mais LENT
sur ce CPU.** Portes corrélées **1.0000** (écart 0.0045 nat),
admissions identiques (Jaccard 1.000), rappel identique (5/7) →
verdict « faithful » selon les seuils déclarés. MAIS forward 7,7 s →
49,2 s et lecture 55,7 s → 236,9 s (×4) : sans support natif, le CPU
émule. **Conclusion des deux mesures : float32 reste le bon défaut sur
CPU ; bfloat16 sert quand la MÉMOIRE est la contrainte (moitié des
poids, fidélité intacte), int8 quand il faut juste faire tenir un plus
gros modèle (fidélité approximative, rappel dégradé).** Les deux
messages de l'outil disent exactement ça à l'écran. Le remplacement de
llama.cpp est donc livré avec son mode d'emploi honnête, pas comme une
accélération.

---

## 2026-08-30 — Pourquoi `complete` invente : ce n'est ni la taille du modèle, ni le réglage du readout

Question de départ : un 0,6B est trop petit pour converser, donc mesurer
avec de plus gros modèles. Quatre hypothèses testées dans l'ordre, trois
réfutées, la quatrième isolée.

**Nouveau dans l'outil** : `mix_full` enregistre les tiers qui ont
franchi leur seuil, `complete` compare l'argmax avec et sans mémoire, et
`Sillage.attribution()` rend `moved/tokens`. La CLI l'affiche **sur
stderr** (stdout reste le texte pur : la garantie « `--fast` identique »
du papier 5 en dépend). Nouveau drapeau `--readout
published|family|b,l,q` : les trois constantes qui décident si la
mémoire parle n'étaient atteignables que depuis Python.

**H1 — le readout publié serait le verrou.** Vrai à 1,7B, faux à 0,6B.
Rapport français de 430 tokens, lu deux fois, 413 grammes.

| | 0,6B publié | 0,6B famille | 1,7B publié | 1,7B famille |
|---|---|---|---|---|
| rappel | 88 % | 88 % | **75 %** | **88 %** |
| témoin (doc non lu) | +0,16 nat | **+2,14** | **−0,00** | **+1,25** |
| tokens déplacés (répondables) | 55 % | 59 % | 45 % | 62 % |
| tokens déplacés (**sans réponse**) | 6 % | **19 %** | 2 % | **27 %** |

Lecture : à 0,6B la famille n'achète **rien** et coûte 13× en localité.
À 1,7B le readout publié est trop discret (75 %) et la famille récupère
les 13 points — mais dans les deux cas elle fait parler la mémoire **3 à
13 fois plus sur les questions sans réponse**, où elle transplante un
vrai passage (« le coût total s'élève à » → « débit de 118 mètres cubes
par heure », 8 tokens déplacés). P2 falsifiée (×1,08 et ×1,37 < 1,5),
P3/P4/P5 confirmées. **Le défaut reste `published`.**

**H2 — le retour à la ligne casserait la clé.** Réfutée : verbatim 6/8 =
rewrappé 6/8 (`results/linewrap.json`). Sonde mal conçue au passage :
aucun témoin sans saut de ligne.

**H3 — la mémoire aurait le fait mais serait outvotée.** Réfutée : le
4-gramme manquant est **absent** du cold store, et monter `LAM_C` de 0,3
à 0,6 ne convertit rien et ne perd rien (`results/outvoted.json`).
Échec de **récupération**, pas d'arbitrage.

**H4 — la clé est exacte au token près. CONFIRMÉE**
(`results/tokenkey.json`). Même fait, trois formulations :

| variante | clé (4 derniers tokens) | cold | sortie |
|---|---|---|---|
| A doc-exact | `[' responsable', ',\n', 'mad', 'ame']` | **HIT** | `Brindas Kolvec, mat` |
| B rewrappé | `[' responsable', ',', ' mad', 'ame']` | MISS | `M. A. D.  Le` |
| C espace final | `[',\n', 'mad', 'ame', ' ']` | MISS | `44444444` |

Le saut de ligne est **absorbé dans un token** (`',\n'`). Le fait est
donc bien en mémoire — A le récite verbatim — mais la même question
tapée sur une ligne ne l'atteint pas, et **la mémoire se tait au lieu de
le dire**. Le modèle gelé comble alors le vide : « M. A. D. » à 0,6B,
« Brigitte Lefebvre » à 1,7B. Témoin en milieu de ligne (`62`) : HIT aux
trois variantes → l'effet est bien la surface, pas le fait.

**Conséquence.** Les inventions observées lors des essais réels ne
viennent ni du modèle ni du mélange : elles viennent d'un **silence** de
la mémoire, causé par un écart de surface entre le document et la
question. C'est aussi pourquoi `ask` (TF-IDF, plis d'accents, tokens de
mots) n'a pas ce défaut : sa clé n'est pas exacte au token.

### Suite du 2026-08-30 — deux corrections à ce qui précède, et le correctif qui marche

**Correction 1 : `reflow` FONCTIONNE. Mes T1/T2 étaient mesurées sur un
état contaminé par ma propre sonde.** `probe_reflow` faisait passer le
témoin de localité **avant** les questions. Or `nll_nowrite` replie
chaque état caché dans le centre courant `mu` du tier sémantique v1
**sans** les écritures correspondantes dans les tiers — un déséquilibre
que seul du code de sonde produit. Bissection (`probe_bisect`) : à
seuils identiques (thrG 0,9310) et réservoir identique (826), la seule
passe témoin fait passer `moved` de 9/12 à 1/12 et « Brindas Kolvec »
devient « Brigitte Lefevre ». Diff complet de l'objet
(`probe_whatmutates`) : quatre attributs bougent, dont `mu` et `mu_n`
(826 → 1008).

Mesuré proprement, sur trois exécutions indépendantes :

| document lu | rappel |
|---|---|
| tel quel (retours à la ligne) | **7/8** |
| reflowé (lignes recollées) | **8/8** |
| reflowé, tier sémantique coupé | 7/8 |

Donc la chaîne complète est : recoller les lignes → la clé de la
question existe dans le store (`' Br'`, 2/2) → **le tier sémantique
porte le reste du nom** et `complete` sort `Brindas Kolvec, matricule
4`. Le cold store seul ne suffit pas : il ouvre le mot et le modèle le
finit en « Brigitte ». C'est le tier v1 qui gagne l'arbitrage, pas
`LAM_C` (0,3 → 0,9 : aucun gain, aucune perte).

**Correction 2 : lire plus de documents ne dégrade PAS le rappel.**
C'était la crainte légitime après avoir vu `mu` dériver.
`probe_moredocs` : après le rapport, lecture de trois documents sans
rapport (pain, jardin, vélo), `mu_n` 826 → 1320, grammes 395 → 879 →
rappel **8/8 à chaque étape**. X1 falsifiée. La lecture ordinaire
déplace le centre **et** les clés stockées ensemble ; seule ma sonde
les désolidarisait. Le passage à l'échelle n'est pas remis en cause.

**Correctif livré : `sem_key(h, learn=False)` en génération.** `mix_full`
mis à part, `complete` promettait « Writes nothing » et déplaçait `mu`
à chaque token généré (826 → 1162 sur 240 tokens). Le tier du papier 8
avait déjà cette discipline (`sem2_observe` : « read time only »), pas
celui du papier 2. `drafting.py` aligné (le snapshot/restore de `mu`
qui garantissait la sortie identique devient un invariant plutôt qu'un
replay). **Mesuré (`probe_freeze_mu`) : W1 falsifiée** — 240 tokens
générés ne dégradaient pas le rappel (8/8 → 8/8), donc ce n'est pas un
gain de rappel ; **W3 : les huit réponses sont identiques** avant/après
sur un état propre. C'est une correction d'hygiène à effet nul mesuré
sur la sortie, pas une amélioration. Elle est livrée comme telle.

**Le tableau readout 0,6B est inchangé** après remesure sans
contamination (88 %/88 %, témoin +0,16/+2,14) : sur un document non
reflowé « Brindas » est de toute façon hors d'atteinte, le tier
sémantique n'avait rien à perdre.

### 2026-08-30, soir — la mémoire peut-elle dire qu'elle ne sait pas ?

Hypothèse issue du rapport de Vernouil (12 points, un seul document) :
les questions auxquelles le document répond déplacent 4 à 11 tokens sur
12, celles auxquelles il ne répond pas 0 à 2. Règle figée **avant**
mesure : *répondre si la mémoire a déplacé ≥ 3 tokens, sinon dire
qu'elle n'a pas atteint la question.* Corpus neuf (compte rendu de
rucher, 376 tokens, `--reflow`), 24 questions en trois familles, dont
une famille **reformulée** pour contrôler le confondant évident : dans
le premier document, toute question répondable était un préfixe
verbatim, donc `moved` pouvait ne mesurer que le recouvrement de
surface.

Fenêtre corrigée en cours de route : à 12 tokens, « Qui a rédigé ce
compte rendu ? » était coupé à « …par le techn| » et compté comme
erreur confiante. À 30 tokens il donne « monsieur **Ovide Trenchard**,
carte apicole ». Une fenêtre qui tronque la réponse mesure la fenêtre.

| famille | résultat |
|---|---|
| verbatim (8) | 7 répondues, **7 justes, 0 fausse** ; la 8ᵉ s'abstient (2/30) et se serait trompée |
| **reformulées** (8) | 2 répondues, **2 justes** ; 6 abstentions honnêtes (0 à 2 tokens déplacés) |
| sans réponse (8) | **7 abstentions**, 1 échec |

**Y1, Y2, Y3 tiennent. Y4 est FALSIFIÉE, et c'est le résultat.** Le seul
échec est « la prochaine visite aura lieu le » → « **11 avril 2026**,
par temps couvert… », c'est-à-dire la visite **déjà passée**, récitée
verbatim avec **16 tokens déplacés sur 30** — alors que la meilleure
réponse *juste* du corpus en déplace 12. Les plages se chevauchent :
**aucun seuil ne sépare ces deux cas.** `moved` mesure si la mémoire a
contribué, pas si elle a répondu à la question posée.

C'est le même échec que « 14 juin » sur le premier document : une
question dont la surface appelle fortement un passage stocké tire la
mémoire à fond, et le passage est le mauvais. C'est donc une classe
nommée, reproductible sur deux corpus, et non corrigeable par le
réglage — comme la normalisation de longueur l'était pour le classement
des paragraphes.

**Livré** : le garde-fou de `complete` passe de « zéro token déplacé » à
`FAINT = 3`. Sur les deux documents réunis, 11 des 12 questions sans
réponse tombent sous ce seuil, contre 9 en ne déclenchant qu'à zéro. Et
aucune réponse juste des deux corpus ne descend sous 12. Le seuil n'est
pas ajusté sur le corpus qui l'a suggéré, et sa limite est écrite dans
le code à côté de sa valeur.

### 2026-08-30, fin — les deux canaux ne peuvent pas se vérifier l'un l'autre

Idée testée : `ask` (TF-IDF) sait quel passage est pertinent, `complete`
sait quel passage la mémoire a tiré ; les faire se contrôler devait
attraper la transplantation. Trois signaux candidats, mêmes 24
questions : **A** `ask` s'abstient-il ? **B** le texte généré est-il
verbatim dans le document ? **C** les deux canaux désignent-ils le même
paragraphe ?

**Les trois échouent, et la démonstration tient en deux lignes jumelles :**

| question | sortie | moved | score `ask` | verbatim | juste ? |
|---|---|---|---|---|---|
| « La visite **de printemps**… s'est déroulée le » | `11 avril 2026, par temps couvert` | 16/30 | 0,623 | oui | **oui** |
| « La **prochaine** visite… aura lieu le » | `11 avril 2026, par temps couvert` | 16/30 | 0,554 | oui | **non** |

Sortie identique, contribution identique, présence verbatim identique,
scores lexicaux dans la même plage (les bonnes réponses vont de 0,245 à
0,623 ; la transplantation est à 0,554, en plein dedans). Z2 et Z3
falsifiées frontalement. Z1 a d'abord semblé tenir : **artefact de ma
sonde** — le localisateur de paragraphe rendait `None` pour 8 passages
d'index sur 24, donc « désaccord » se déclenchait aussi sur 3 bonnes
réponses sur 9. Corrigé dans le fichier de résultats.

**Pourquoi c'était perdu d'avance, et pourquoi c'est utile de le
publier :** les deux canaux sont **de surface**. La seule chose qui
distingue les deux questions est le mot « prochaine », absent du
document, contre « de printemps … s'est déroulée ». Séparer les deux
demande de savoir qu'une visite prochaine n'est pas une visite passée —
pas de comparer des sacs de mots. Deux mécanismes de surface ne
s'auditent pas mutuellement : ils se trompent ensemble.

La transplantation reste donc **une limite ouverte, nommée et bornée** :
elle ne touche que les questions dont la formulation recouvre presque
entièrement un passage stocké, et le garde-fou `FAINT` attrape tout le
reste (11 des 12 questions sans réponse sur deux corpus).

### 2026-08-30, nuit — la reformulation n'est ni un problème de taille, ni d'interface

Hypothèse : le 0,6B ne sait pas répondre dans le registre d'un document,
donc c'est LE MODÈLE qui doit faire le pont entre la question et la
surface où la mémoire mord ; un plus gros devrait donc mieux reformuler.
Quatre bras, même document reflowé (`results/bridge.json`) :

| bras | verbatim | **reformulées** | intrusion |
|---|---|---|---|
| 0,6B publié | 7/8 | 2/8 | 1/8 |
| **0,6B famille** | **8/8** | **3/8** | 2/8 |
| 1,7B publié | 6/8 | 0/8 | 2/8 |
| 1,7B famille | 8/8 | **0/8** | 3/8 |

**P1 FALSIFIÉE** (≥5/8 prédit, 0/8 obtenu) — et à l'envers : le gros
modèle fait MOINS bien. Cause : `complete` encode le prompt BRUT, et un
modèle de base à qui on donne une question continue le genre. Le 1,7B
écrit d'autres questions 7 fois sur 8 (« Qui a rédigé ce compte rendu ?
Qui a rédigé ce compt… »). La maladresse du 0,6B — retomber dans la
phrase du document — est justement ce qui laissait la mémoire prendre le
relais. **Erreur de conception de ma sonde : j'ai testé la mauvaise
porte.** P3 (contrôle) tient : 0,6B famille = 3/8 < 5, donc un gain à
1,7B n'aurait pas pu être attribué au readout seul. P5 partiellement
falsifiée (1,7B publié : 6/8 verbatim, sous le plancher).

**Correction 2 au passage** : « famille n'achète rien à 0,6B » était vrai
sur Vernouil, FAUX ici (7/8→8/8 verbatim, 2→3 reformulées). La valeur du
réglage dépend du corpus — il reste un drapeau, pas un défaut.

**Le vrai test, par le gabarit de conversation** (`results/chattemplate.json`),
celui que `chat` et `serve` appliquent et que `complete` n'applique pas :

| | reformulées | écho | intrusion | moved moyen (sans réponse) |
|---|---|---|---|---|
| 0,6B brut | 3/8 | 4/8 | 3/8 | 2,4 |
| 0,6B **gabarit** | **1/8** | 1/8 | **5/8** | **5,8** |
| 1,7B brut | 0/8 | 7/8 | 4/8 | 5,6 |
| 1,7B **gabarit** | **0/8** | **0/8** | **5/8** | **7,0** |

**Q1 confirmée** (l'écho disparaît, 7/8→0/8), **Q2 FALSIFIÉE** (1,7B reste
0/8, et le gabarit fait TOMBER le 0,6B de 3/8 à 1/8), **Q3 confirmée**
(moved baisse sur les répondables), **Q4 : l'intrusion MONTE**.

Raison structurelle, enregistrée avant la mesure : **le gabarit finit
chaque prompt par le même en-tête assistant, donc la clé 4-grammes du
premier token est IDENTIQUE pour toutes les questions.** La mémoire se
déclenche sur le gabarit et non sur la demande : elle parle plus (2,4 →
5,8 tokens) et discrimine moins (3/8 → 1/8). C'est la pire combinaison,
et elle touche `serve`.

**Conclusion des deux sondes : la reformulation n'est ni un problème de
capacité ni un problème d'interface. C'est un problème de CLÉ** — les
deux tiers rapides indexent de la surface, et aucune taille de modèle ni
aucun gabarit ne change ça. C'est exactement ce que vise le premier point
de l'axe 5 (clés d'encodeur gelé externe, invariant à la paraphrase par
construction). Pour poser une question aujourd'hui, la bonne porte reste
`ask` : 12/12 à l'entrée, verbatim, sourcé, rien de généré.

### 2026-08-30, très tard — la reformulation était DÉJÀ résolue, par la composition

Les deux sondes précédentes isolaient le canal READOUT. `serve` en a
deux : il injecte aussi les passages que `ask` retrouve — et **ce
canal-là est robuste à la reformulation**, parce que c'est du TF-IDF sur
les mots du demandeur, pas une clé de quatre tokens. Mesure sur le vrai
endpoint, vraies sockets, avec `--no-context` en ablation
(`results/serve_rephrase.json`) :

| | reformulées justes | questions sans réponse |
|---|---|---|
| 0,6B, **passages injectés** | **8/8** | **7/8 fabriquées, sources nommées** |
| 0,6B, `--no-context` (readout seul) | **0/8** | refus fréquents |

**S1 et S2 confirmées.** Le 0,6B répond à **huit questions reformulées
sur huit**, en français correct, le fait exact en gras : « Le compte
rendu a été rédigé par **monsieur Ovide Trenchard** », « La reine a été
vue en ponte sur le **quatrième cadre** ». Sans injection : 0/8, et des
fabrications (« M. Jean-Luc », « Roi d'Angleterre », « 16 mai »). Le
papier 7 rejoue exactement : 5 % pour la mémoire seule contre 25 % pour
la preuve dans la fenêtre.

**Donc la reformulation n'a jamais été un problème de clé pour l'outil
complet — seulement pour `complete`.** Les clés d'encodeur externe de
l'axe 5 ne sont PAS nécessaires à ça. Mes trois sondes précédentes
mesuraient un canal, pas le produit. (Correction du scorer au passage :
« 38 kilogrammes » et « quatrième » étaient comptés faux parce que je
cherchais « trente-huit » et « quatrieme » sans accent — 6/8 affiché
était en réalité 8/8. Repli d'accents + alias numériques ajoutés.)

**S3 FALSIFIÉE, et c'est le prix.** Sur les 8 questions auxquelles le
document ne répond pas, l'injection fait fabriquer **7 fois sur 8**, avec
aplomb, en gras, et `X-Sillage-Sources` nomme la source : « compte
**huit ruches** au total » (le document dit huit CADRES DE COUVAIN),
« **2,50 €** le kilo », « la reine a **12 ans** », « production totale
**14 kilogrammes** » (c'est le rendement colza par ruche). Et le bras
`--no-context` REFUSE plus souvent (« Je ne peux pas fournir le prix
exact ») : **mettre des passages dans la fenêtre pousse le modèle à
répondre à tout**, y compris à ce que les passages ne soutiennent pas.

C'est le problème d'ancrage classique du RAG, et l'outil n'instruit
actuellement pas le modèle de s'abstenir. Piste bon marché non testée :
une ligne de consigne système « si les passages ne contiennent pas la
réponse, dis-le ».

**Défaut trouvé en passant** : `serve` n'accepte pas `--target` — le
drapeau est déclaré dans le groupe `gen`, dont le sous-parseur `serve`
n'hérite pas. L'endpoint ne peut donc pas être pointé sur un lecteur plus
gros, alors que l'état le permet (papier 5). S4 non mesurable.

### 2026-08-31 — deux correctifs, une prédiction falsifiée, et un bug livré depuis huit versions

**① Ligne d'ancrage dans `serve`.** Le message système ne disait que
« use them if they are relevant ». Ajouté : « Answer only from these
notes. If they do not contain the answer, say plainly that it is not in
the notes rather than guessing. » Prédictions enregistrées avant :
G1 fabrications ≤ 3/8, G2 reformulées ≥ 7/8.

**G2 tient** (8/8, et les réponses sont plus nettes : « Monsieur Ovide
Trenchard. », « 3 colonies. »). **G1 FALSIFIÉE** : refus 0 → 2 sur 8,
fabrications ~8 → 6. **Une ligne de consigne ne suffit pas à un 0,6B**,
il l'applique deux fois sur huit. Livrée quand même — 0 → 2 refus est un
gain réel sans coût mesuré — mais son insuffisance est écrite dans le
code à côté d'elle.

**② `serve --target`, et ce qu'il a révélé.** Le drapeau était déclaré
dans le groupe `gen` dont `serve` n'hérite pas. Une fois exposé, le
serveur meurt à la première requête :

    ValueError: operands could not be broadcast together
                with shapes (2048,) (1024,)

**`complete --target` plantait aussi.** Le tier sémantique v1 centre sur
`mu`, dont la largeur est celle du modèle QUI A ÉCRIT (1024 pour le
0,6B) ; un lecteur 1,7B en fournit 2048. **Livré depuis la 1.1.0, huit
versions mineures, sur l'état qwen par défaut, et c'est la fonctionnalité
phare du papier 5.** Jamais vu parce que toute mesure du transfert
construisait son état AVEC le modèle cible, où les largeurs coïncident —
mes propres sondes de cette nuit comprises. Correctif
(`_check_hidden_width`) : les tiers indexés sur les états cachés (papier
2 et papier 8) s'abstiennent en le disant, ceux indexés sur les TOKENS
(n-gram, cold store) continuent — ce que le papier 5 revendique
réellement. Test T20, qui construit deux largeurs sans charger de poids.

**Résultat final, trois bras sur le vrai endpoint**
(`results/serve_rephrase.json`) :

| bras | reformulées | refus | fabrications |
|---|---|---|---|
| 0,6B + passages | **8/8** | 2/8 | 6/8 |
| 0,6B sans passages | 0/8 | 1/8 | 7/8 |
| **1,7B + passages** | 7/8 | **5/8** | **3/8** |

**S4 falsifiée sur le rappel** (7/8 < 8/8) **et c'est la mauvaise
métrique.** Le gros modèle ne se souvient pas mieux : **il ment moins.**
Fabrications divisées par deux, refus plus que doublés. C'est ce qui
manquait pour que `serve` soit utilisable par un tiers, et c'est le
premier bénéfice mesuré de la capacité dans tout ce projet qui ne soit
ni de la vitesse ni de la fluidité.

Compteur de refus corrigé au passage : il ratait « ne mentionne pas » et
sous-évaluait le 1,7B de trois. Recompté sur les textes enregistrés.

### 2026-08-31 — LA LOI DE CAPACITÉ

La question que tout utilisateur pose en premier et à laquelle le projet
ne savait pas répondre : « 7 Mo pour toujours », mais pour combien de
documents ? Trois sondes, cinq prédictions enregistrées + quatre sur
l'éviction.

**① Ce qui remplit le store, ce ne sont pas les tokens**
(`results/gramrate.json`, 11 textes réels, tokenizer seul). Une prose
réelle ne répète que **~6 %** de ses propres 4-grammes, et le cold store
n'admet qu'à partir de **deux occurrences**. D'où deux régimes :

| | le store se remplit après | ce qu'il retient |
|---|---|---|
| lu **une fois** | ~**890 000 tokens** | presque rien (les 6 % qui se répètent seuls) |
| lu **deux fois** | ~**56 000 tokens** | tout ce qui est distinct |

Le second chiffre est stable à ±5 % sur dix textes indépendants (53k-59k
: les 8 preprints, README, REPRODUCE). **Durabilité et capacité sont le
MÊME budget** — la règle des deux occurrences du papier 6 remplit le
store 16× plus vite, et ce prix n'était écrit nulle part.

Coût de stockage exact d'après `_cold_save` : 16 o de clé + 4 o de masse
+ 8 o de décalage + 12 o par successeur ≈ **42 o/gramme**. Donc le cap de
50 000 coûte **~2,1 Mo** → **≈ 27 000 tokens durables par Mo, soit ~40
pages par mégaoctet**. **Ce n'est pas un plafond, c'est une molette** —
et `COLD_MAX` est une constante de module, pas un réglage utilisateur.

**② Le rappel ne se dégrade PAS** (`results/capacity.json`, gpt2, faits
plantés 2× par cohorte, 1 million de tokens, 47 min) :

| tokens | grammes | ancien | milieu | récent | grammes présents | témoin |
|---|---|---|---|---|---|---|
| 5 850 | 1 180 | 100 % | 83 % | 100 % | 100 % | +0.0000 |
| 100 735 | 9 417 | 100 % | 100 % | 100 % | 100 % | +0.0000 |
| 400 275 | 31 508 | 100 % | 100 % | 83 % | 100 % | +0.0000 |
| 800 908 | 50 781 | 100 % | 83 % | 100 % | 100 % | +0.0000 |
| **1 000 107** | **63 898** | **100 %** | **100 %** | **100 %** | **100 %** | **+0.0000** |

**Plat sur 171× d'échelle.** C2, C3, C5 tiennent ; les 83 % isolés sont
du bruit à un fait près (5/6), sans tendance. **La localité est
EXACTEMENT nulle** (1,74e-07, identique à 7 chiffres) à toutes les
échelles : la matrice ne bave pas en se remplissant.

**C1 FALSIFIÉE, et c'est un défaut** : le store est monté à **63 898
grammes pour un cap de 50 000**. L'éviction n'est appliquée que dans
`save()` — une boucle d'ingestion qui ne persiste pas dépasse le cap de
28 %. Donc C4 n'était pas testée non plus : la course n'a jamais élagué.

**③ L'éviction, mesurée pour la première fois**
(`results/eviction.json`, cap abaissé à 3 000 pour saturer en 3 min) :

    avant save : 4 865 grammes (1,62× le cap), 294 faits plantés
    après save : 3 000 grammes -- 1 865 jetés, 38 % du store
    faits plantés encore présents : 100 %   rappel : 100 % -> 100 %

**E1, E2, E3 tiennent.** 38 % du store disparaît et **pas un seul fait
planté n'est perdu** : la règle « garder les COLD_MAX grammes de plus
forte masse de surprise » protège exactement ce qui compte. **E4
falsifiée** — la médiane de masse ne bouge pas (2,0 → 2,0), mais c'est un
artefact de granularité : la distribution a un gros atome aux basses
valeurs (q10 1,0 / q50 2,0 / q90 29 → 30), et le q90 monte bien.

**Énoncé de la loi.** À 40 pages par mégaoctet de cold store : le rappel
est plat jusqu'à au moins 1 million de tokens, la localité reste nulle,
et quand le store déborde il **oublie l'ordinaire et garde le
remarquable**, sans rien coûter à ce qu'il avait retenu.

**Deux actions qui en découlent** : exposer `COLD_MAX` (la molette de
capacité est aujourd'hui hors de portée de l'utilisateur, comme l'était
le readout) et appliquer le cap ailleurs qu'au `save()`.

**Erreur de sonde consignée** : mon premier générateur d'entités faisait
finir tous les noms par la même syllabe, donc les 4 derniers tokens du
probe étaient IDENTIQUES pour toutes — une seule clé partagée à six
successeurs rivaux, rappel lu 0/6 alors que le mécanisme allait bien. Les
noms sont désormais filtrés contre le tokenizer : un candidat n'est gardé
que si son probe forme un 4-gramme qu'aucun autre ne forme. Et le
contrôle « filler » est inutilisable tel quel : sur mémoire VIDE il vaut
déjà 100 %, GPT-2 prédisant sa propre grammaire — d'où C4 mesurée DANS le
store et non par génération.

### 2026-08-31 — la molette de capacité, et un cap qui tient enfin

Deux correctifs issus de la loi de capacité.

**① `COLD_MAX` devient un réglage.** C'était une constante de module que
personne ne pouvait atteindre — exactement le cas du readout avant-hier,
sauf qu'ici la constante décide de la CAPACITÉ du produit. Elle devient
`mem.cold_max` (par mémoire) et `--cold-max N` en ligne de commande.
L'abaisser sur un état existant **élague immédiatement** et dit combien
de grammes de plus faible surprise ont été jetés, au lieu d'attendre le
prochain `save()`.

**② Le cap tient pendant l'ingestion, plus seulement au `save()`.**
Mesuré avant : un cap de 2 000 laissait le store monter à **5 972
(2,99×)** parce que l'éviction ne vivait que dans `save()` — et le chemin
rapide `ingest_text` ne passe même pas par `write_all`. Les DEUX chemins
élaguent maintenant sur une marge `PRUNE_MARGIN = 1.25` (trier 60k items
est bon marché, le faire à chaque token ne l'est pas).

    cap 2000 | pic en memoire 2497 (1.25x) | apres save 2000
    rappel : plus ancien 100 %, plus recent 100 %, sur 372 faits plantes

**L'élagage continu ne coûte rien** : le rappel reste à 100 % aux deux
extrémités du corpus alors que le store est maintenu à 1/3 de ce qu'il
aurait atteint. C'est la contrepartie mesurée de l'éviction par masse de
surprise.

**Correction d'un contrat au passage.** T6 patchait `core.COLD_MAX` APRÈS
construction puis appelait `save()` : il testait que le cap était lu à
l'écriture. Ce n'est plus vrai — et c'est le but, puisque c'est ce qui
rend `--cold-max` possible. Le test vérifie la même propriété (garder la
masse la plus forte) sous le nouveau contrat, et T21 ajoute la molette
elle-même. Tests 22 + 14 + 16 + 28 = **80**.
