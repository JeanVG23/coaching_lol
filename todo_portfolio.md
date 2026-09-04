# TODO — Valorisation portfolio / CV

> Objectif : rendre le projet **vérifiable en 5 minutes par un recruteur** et cocher les
> signaux attendus sur des postes Data Scientist / MLOps / Data Engineer / GenAI Engineer.
> Contexte : page CV `https://jeanvg.fr/projets/coaching-lol`, démo `https://coaching-lol.jeanvg.fr`,
> repo `github.com/JeanVG23/coaching_lol`.
>
> Constat : le contenu technique est déjà au niveau. Ce qui manque, c'est la **vérifiabilité**
> et les signaux d'industrialisation. Ne pas ajouter de ML, consolider ce qui existe.

## 🔥 P1 — Démo exploitable par quelqu'un qui ne connaît pas le projet

État actuel : la home affiche titre + menu + grille de comptes. Zéro explication, zéro exemple
visible. La page compte propose un bloc « Mettre à jour mes données » qui demande au visiteur de
lancer `poetry run python src/collection/refresh_cloudflare.py` dans son terminal.

- [ ] **Deep link depuis le CV vers une review de game précise** (`/c/spadzze`, onglet parties,
      review ouverte), pas vers la home. Router `web/frontend/app.js` à adapter si besoin
      (support d'un paramètre `?review=<match_id>` ou `#`).
- [ ] **Encart explicatif en tête de review** (3 lignes) : partie réelle -> ce que le pipeline a
      extrait -> ce que le LLM en a fait -> pourquoi il ne dit jamais X. L'asymétrie
      d'information (`COACHING_SAFE` / `ML_ONLY`) est le meilleur argument du projet et elle est
      aujourd'hui **invisible** sur le site.
- [ ] **Masquer le bloc terminal aux visiteurs** (`details.sync-help` dans `web/frontend/index.html`,
      ~ligne 100). N'a de sens que pour le propriétaire du compte.
- [ ] **Meta OpenGraph statiques** dans `web/frontend/index.html` (`og:title`, `og:description`,
      `og:image`). La SPA n'a aucun SSR : un partage de lien (LinkedIn, mail RH) affiche
      actuellement une coquille vide.

## 🔥 P2 — CI + linter (le manque le plus criant)

47 fichiers de tests, un test de parité KV entre deux runtimes, du vitest côté Worker, et
**aucun `.github/workflows/`, aucun linter configuré**. C'est le premier signal regardé sur un
poste MLOps/DE.

- [ ] `.github/workflows/ci.yml` : `poetry run pytest -k "not pretrain"`, `npm test --prefix web/cf`,
      `npm run --prefix web/cf typecheck`, `ruff check`.
- [ ] Ajouter `ruff` au groupe dev de `pyproject.toml` + config minimale.
- [ ] Badge CI dans le README (à côté des badges existants).

## 🔥 P3 — Fermer la boucle d'éval LLM et publier le chiffre

Critère de succès défini dans CLAUDE.md et **non atteint** : ≥70 % de mistakes utiles sur ≥10
reviews par-game annotées. L'outillage existe (`feedback.py annotate --pending`), les annotations
non. Sur un poste GenAI, la question d'entretien est littéralement « comment tu évalues ton LLM ? ».

- [ ] Annoter 10 à 15 reviews par-game (`poetry run python src/04_coaching/feedback.py annotate --player spadzze --pending`).
- [ ] Publier le taux obtenu, **même mauvais**, sur la page CV et sur le site (`/readme`).
- [ ] Enrichir chaque record de review : modèle, version de prompt, latence, coût estimé.
      Transforme « j'ai branché un LLM » en « j'ai un harness d'éval ».

## ⚙️ P4 — Reformuler la page CV autour des bons arguments

La page met en avant « compression ×13 » et « 6 couches » : de la plomberie, non différenciante.
Les trois vrais atouts, dans cet ordre :

- [ ] **Le protocole d'éval** (à mettre en titre) : split canonique par joueur
      (`data/04_dataset/split.json`), purged CV avec fuite mesurée (≈ +0.005 d'AUC via les games
      partagées), headline sur test held-out (0.677) et non sur du OOF optimiste, script d'audit
      de fuite dédié (`audit_leakage.py`). Quasiment aucun projet de portfolio ne fait ça.
- [ ] **La contrainte produit encodée dans le code** : `COACHING_SAFE` vs `ML_ONLY`, avec `assert`
      au chargement de `compare.py` qui crashe si une feature interdite fuit vers le coaching.
      C'est de la gouvernance de features, pas du storytelling.
- [ ] **Les résultats négatifs assumés** : SSL `delta = -0.0195`, `high_elo` plafonné à 0.589,
      per-game déprécié. Un lead DS valorise ça plus que trois AUC flatteuses.
- [ ] Toujours donner le contexte du chiffre (n + protocole). « AUC 0,645 vs 0,633 » seul ne
      prouve rien ; « test held-out, 201 joueurs, CV purgée » devient une preuve de méthode.

## ⚙️ P5 — Rendre le repo exécutable

`data/` est intégralement gitignoré : après `git clone && poetry install`, un lecteur ne peut
**rien** lancer.

- [ ] Jeu de fixtures anonymisées (quelques matchs raw compressés, versionnés).
- [ ] Cible `make demo` : silver -> gold -> `compare` -> une review avec client LLM mocké.
- [ ] Documenter `make demo` en tête du README. C'est ce qui fait passer de « README impressionnant »
      à « code réellement exécuté par un lead tech ».

## ⚙️ P6 — Orchestration réelle

Le pipeline récurrent est `run_scraping.sh` : une boucle bash `for i in {1..5}` avec `sleep 10`.
Point faible le plus visible pour un poste data engineer.

- [ ] **Minimum** : un `Makefile` déclarant les dépendances réelles entre étapes
      (collecte -> `reextract_silver` -> `rebuild_gold` -> `build_*_dataset` -> train -> `sync_cloudflare`),
      plus un cron GitHub Actions sur la partie légère.
- [ ] **Si cible DE explicite** : Airflow local en docker-compose avec le DAG réel. Le pipeline
      s'y prête (dépendances claires, idempotence déjà pensée dans `pipeline_ops/`). Coche une
      case entière de job description.
- [ ] Chantier le plus lourd : à garder pour un troisième temps.

## 🧹 P7 — Hygiène qui fait tache au clone

- [ ] Le README annonce « Python 3.11+ », `pyproject.toml` exige `>=3.13,<3.15`. Quiconque suit
      le README échoue à l'install. Aligner.
- [ ] `fastapi` et `uvicorn` toujours en dépendances principales alors que le backend est archivé ;
      le `Dockerfile` à la racine builde le legacy. Déplacer `web/backend/` + `Dockerfile` dans
      `legacy/` ou les supprimer (git garde l'historique).
- [ ] Le caveat torch/xgboost (double load libomp -> SIGSEGV sur Mac) n'existe que dans CLAUDE.md.
      Le remonter dans le README, sinon un lecteur qui tente le training se prend un segfault.

## 🧹 P8 — Model card

- [ ] `docs/MODEL_CARD.md` : données, période, label, protocole d'éval, métriques test, limites
      assumées (transfert de rang aux deux ADC, drift temporel du label LP, GM à n≈12).
      Source déjà disponible : `data/05_model/player_metrics.json` (historique des runs).
- [ ] L'exposer sur `/readme` du site. Répond d'avance à la moitié des questions d'entretien DS,
      et documenter ses propres flaws est un signal fort.

## 💡 Nice to have (si le temps le permet)

- [ ] DVC ou MLflow sur `data/05_model/` (traçabilité modèle).
- [ ] Validation de schéma (pandera / Pydantic) au passage silver -> gold, en gate.
- [ ] Endpoint `/api/health` + page « métriques » sur le site.

## Ce qui est déjà bon et sous-vendu

Architecture médaillon nommée et respectée, séparation `core`/`collection`/`pipeline_ops`,
specs datées dans `docs/superpowers/specs/`, parité KV verrouillée par test entre deux runtimes,
sortie LLM contrainte par schéma Pydantic. Tout ça est invisible depuis la page CV.

## Ordre proposé sur deux semaines

**Semaine 1** : P2 (CI + linter), P7 (hygiène), P1 (démo : deep link, encart, meta OG).
**Semaine 2** : P3 (annotations + publication du taux), P8 (model card), P5 (`make demo`),
P4 (réécriture de la page CV).
**Plus tard** : P6 (orchestration Airflow), P9 (nice to have).

> Si un seul item : **P2 (la CI)**. Quelques heures, et c'est le seul point où l'absence est
> activement lue comme un défaut.
