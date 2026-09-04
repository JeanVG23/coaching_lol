# TODO — Qualité du coach LLM

> Chantier SÉPARÉ de `todo_portfolio.md`, qui reste prioritaire : d'abord un produit
> fonctionnel et vérifiable, ensuite l'amélioration incrémentale du LLM.
>
> Source : `review.md` (lecture par Jean d'une review kimi-k2.6, 2026-09-04) + les
> 12 reviews annotées de la boucle d'éval.
>
> Diagnostic chiffré au 2026-09-04 : mistakes utiles **90 %**, focus 92 %, habits 80 %,
> mais **forces 68 %**. Presque toutes les annotations négatives portent sur les reviews
> **agrégées**, pas sur le par-partie. Verbatims : « aucune idée de pourquoi »,
> « compliqué à dire pourquoi », « est-ce que c'est parce que je greed moins ? ».
> Cause mécanique : le payload agrégé n'est fait que de médianes, aucun LLM ne peut
> en tirer une cause. **La racine commune des items ci-dessous est la pauvreté du
> payload, pas le modèle.**

## L1 — Dégâts subis avant la mort (« fatal damage »)

Intuition de Jean : « si j'avance, je me prends 2 auto-attaques et ensuite je me fais
dive, le dive est peut-être rendu possible à cause des autos précédentes ». On parle
beaucoup des morts, jamais des dégâts qui les ont rendues possibles.

**La donnée existe déjà, dans le raw collecté, 0 appel API.** Chaque `CHAMPION_KILL`
de la timeline porte :
- `victimDamageReceived` : décomposition **par source et par sort** du coup fatal
  (`{name, participantId, spellName, spellSlot, physicalDamage, magicDamage, trueDamage, basic, type}`)
- `victimTeamfightDamageReceived` : même chose sur le **combat entier**, pas seulement
  le burst final. C'est celui-là qui porte l'insight de Jean.

- [x] Enrichir `game_journal.deaths` d'un bloc `damage` : part des PV perdus avant
      l'engage vs pendant, top 2-3 sources, part `basic` (autos) vs sorts.
- [x] Restituer dans `SYSTEM_GAME` règle 2 (la `cause` devient « 62 % de tes PV partis
      sur 3 autos de Caitlyn avant l'engage, puis Skarner finit » au lieu de
      « mort par gank de Skarner »).
- [x] Étendre les unités de `grounding.py` (une part de dégâts est un `pct`, une valeur
      de dégâts une nouvelle unité `dmg` : sans cloisonnement, ces nombres ancreront
      n'importe quoi).

**Pourquoi en premier** : répond mot pour mot à l'annotation « je ne sais pas vraiment
pourquoi je suis mort » (tag `non-actionnable`), et c'est le seul item qui améliore la
cause **sans toucher au prompt de fond**, donc le taux d'utilité reste comparable
avant/après.

## L2 — Contexte de matchup dans le payload par-game

Annotation de Jean, verbatim : « il faudrait aussi rajouter, pour que je puisse faire un
bon feedback, plus de détails sur la game comme le match-up par exemple ».

Le bloc `context` existe (comp botlane + jungles + mid, `lane_pattern`, `gank_exposure`)
mais ne nomme pas explicitement l'adversaire direct et ignore le reste.

- [x] Adversaire de lane explicite, sorts d'invocateur des deux côtés, runes clés,
      build final des deux ADC. Tout est dans le match brut, 0 appel API.
- [x] Asymétrie : champ select + sorts + items visibles au scoreboard = info que le
      joueur AVAIT. Reste safe.

## L3 — Analyse globale = agrégation des reviews par-partie

Idée de Jean : « pour l'analyse globale, il faudrait prendre les 20 dernières parties du
joueur, récupérer les analyses LLM par game et regarder ce qu'il en ressort ».

C'est le fix direct des forces à 68 %.

- [x] Map-reduce : N reviews par-partie en entrée, une synthèse en sortie.
- [x] ⚠️ **Précaution non négociable** : un LLM sur des sorties de LLM casse
      `grounding.py` (les chiffres ne viennent plus d'un payload déterministe, donc plus
      rien n'est vérifiable) et amplifie les erreurs. Version qui tient : le payload
      agrégé déterministe reste la **source des chiffres** (seuls ceux-là citables), les
      reviews par-partie sont la **source des causes**. Le grounding reste mesurable.

## L4 — Découpe en agents spécialisés + agent chef

Idée de Jean, affinée : plusieurs petits agents spécialisés (build, timings de recall,
trades, positionnement…), chacun produisant l'analyse de SON segment, puis **un chef qui
reprend ces analyses, les croise et conclut**, avec une phrase de synthèse pour les gens
pressés et la possibilité de déplier chaque analyse séparément.

Ce que ça apporte réellement :
- **Spécialisation** : un prompt build peut porter des règles d'items qui n'ont pas leur
  place dans un prompt générique.
- **Mesurabilité par axe** : taux d'utilité build vs recalls vs trades, au lieu d'un taux
  global. Vu l'écart forces 68 % / mistakes 90 %, savoir OÙ ça casse vaut cher.
- **Priorisation par le chef** : c'est là qu'est le vrai croisement, quand deux axes
  pointent la même racine (« tu retiens du gold » + « tu meurs en poussant » = un seul
  problème de reset).

Nuance honnête : la découpe seule ne « croise » rien, chaque agent voit une tranche
différente du payload, il n'y a ni vote ni redondance. Le croisement n'existe QUE dans
l'étape chef.

- [x] Commencer à **2 axes** (morts/positionnement contre économie/build), pas 5.
- [x] Le chef ne cite QUE des insights produits par les sous-agents, jamais de
      reformulation libre (sinon même problème de grounding qu'en L3).
- [x] Coût : N+1 appels par game au lieu de 1. À mettre en regard de la lenteur actuelle
      d'Ollama Cloud (60-150 s par appel, cf. les 6 timeouts du run contrefactuel).
- [x] UI : synthèse dépliable par axe.

## L5 — Chat interactif sur sa game

Idée de Jean : pouvoir répondre au coach (« voilà pourquoi j'ai fait ça ») et qu'il
tranche (« ok, cas particulier, mais dans ce cas-là il fallait plutôt… »).
Son argument : « les fausses croyances sont les pires pour s'améliorer ». Bénéfice
secondaire : chaque échange produit de l'annotation gratuite pour la boucle d'éval.

- [x] Le SSE Ollama existe déjà côté Worker (bouton coaching), donc ce n'est pas parti
      de zéro.
- [x] ⚠️ **Piège d'asymétrie** : en question libre, le joueur PEUT demander « où était le
      jungler ennemi ». Le modèle doit refuser plutôt que lire la timeline complète. Cela
      veut dire durcir la règle 1 dans un contexte où l'utilisateur pousse activement
      dans l'autre sens. À tester explicitement (cas de test dédié).
- [x] Chantier le plus lourd de la liste : à garder pour la fin.

## L6 — A/B de modèles (glm 5.3 vs kimi-k2.6)

**glm 5.3 est le dernier modèle d'Ollama et le plus puissant actuellement** (info Jean).
kimi-k2.6 avait été retenu après A/B (cf. `src/04_coaching/README.md`), mais l'A/B datait
d'avant le harness.

- [ ] Rejouer l'A/B **sans annotation humaine** : `grounding.py` (ancrage + asymétrie) et
      `counterfactual.py` (sensibilité au payload) suffisent à départager, plus la latence
      et le nombre de retries de schéma déjà tracés dans le bloc `run`.
      Harness implémenté dans `model_ab.py` ; le run Ollama réel reste à lancer.
- [ ] À faire **après** L1-L2 : comparer deux modèles sur un payload pauvre ne mesure que
      le bruit.

## Bug attrapé par l'annotation (à corriger quoi qu'il arrive)

Note de Jean : « avec 1 200 g, il faut prendre en compte que souvent les ADC doivent
attendre 1,3k g pour avoir la BT, donc ça semble normal ». C'est exactement ce que la
**règle 4 de `SYSTEM_GAME` interdit déjà** : le gold retenu se juge relativement au
PROCHAIN ACHAT RÉEL.

- [x] Diagnostiquer : le payload ne donnait pas l'achat suivant, ou le modèle a ignoré
      la règle ? Les deux se vérifient sur la review concernée.
      **Diagnostic :** les `items` existaient dans la liste de recalls, mais aucun lien
      explicite ne reliait la mort de 11:06 au recall de 11:17 ; le modèle n'a pas fait
      ce rapprochement et a ignoré l'exemple de la règle. Le nouveau `next_purchase`
      porte directement B.F. Sword (1 300 g) sur la mort à 1 268 g.
- [x] Ajouter le cas de test correspondant.

## Ordre retenu

1. **L1** (fatal damage) et **L2** (matchup) : payload plus riche, prompt inchangé, donc
   taux d'utilité comparable avant/après.
2. **L3** (analyse globale en map-reduce) : c'est le fix des forces à 68 %.
3. **L4** (agents spécialisés + chef).
4. **L6** (A/B glm 5.3), une fois le payload digne d'une comparaison.
5. **L5** (chat), le plus lourd.

> Ce que Jean a validé dans la review actuelle et qu'il ne faut pas casser : l'appui sur
> des **actions** plutôt que sur le gold (« le gold peut avoir plusieurs causes »), la
> mention des **limites du champion** (Jinx sans mobilité), et le poids donné au
> **positionnement** (« c'est le positionnement qui fait tout dans LoL »).
