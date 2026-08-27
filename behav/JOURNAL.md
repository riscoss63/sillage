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
