# TODO — Valorisation portfolio / CV

> Objectif : rendre le projet **vérifiable en 5 minutes par un recruteur** et cocher les
> signaux attendus sur des postes Data Scientist / MLOps / Data Engineer / GenAI Engineer.
> Contexte : page CV `https://jeanvg.fr/projets/coaching-lol`, démo `https://coaching-lol.jeanvg.fr`,
> repo `github.com/JeanVG23/coaching_lol`.
>
> Constat : le contenu technique est déjà au niveau. Ce qui manque, c'est la **vérifiabilité**
> et les signaux d'industrialisation. Ne pas ajouter de ML, consolider ce qui existe.
>
> Amélioration de la qualité du coach LLM : chantier séparé, cf. `todo_llm.md`
> (à ouvrir APRÈS cette liste, produit fonctionnel d'abord).
>
> Avancement au 2026-09-04 : **P1 fait et déployé**, **P2 fait** (CI + ruff), **P7 fait**,
> **P3 CLÔTURÉ** : 12 analyses annotées, 96 % de mistakes utiles (cible ≥70 % sur ≥10),
> taux publié et vérifié en production sur `/api/c/spadzze/eval`.
> Reste : P5 (`make demo`), P6 (orchestration).

## 🔥 P1 — Démo exploitable par quelqu'un qui ne connaît pas le projet

État actuel : la home affiche titre + menu + grille de comptes. Zéro explication, zéro exemple
visible. La page compte propose un bloc « Mettre à jour mes données » qui demande au visiteur de
lancer `poetry run python src/collection/refresh_cloudflare.py` dans son terminal.

- [x] **Deep link depuis le CV vers une review de game précise** (`/c/spadzze`, onglet parties,
      review ouverte), pas vers la home. Router `web/frontend/app.js` à adapter si besoin
      (support d'un paramètre `?review=<match_id>` ou `#`).
- [x] **Encart explicatif en tête de review** (3 lignes) : partie réelle -> ce que le pipeline a
      extrait -> ce que le LLM en a fait -> pourquoi il ne dit jamais X. L'asymétrie
      d'information (`COACHING_SAFE` / `ML_ONLY`) est le meilleur argument du projet et elle est
      aujourd'hui **invisible** sur le site.
- [x] **Masquer le bloc terminal aux visiteurs** (`details.sync-help` dans `web/frontend/index.html`,
      ~ligne 100). N'a de sens que pour le propriétaire du compte.
- [x] **Meta OpenGraph statiques** dans `web/frontend/index.html` (`og:title`, `og:description`,
      `og:image`). La SPA n'a aucun SSR : un partage de lien (LinkedIn, mail RH) affiche
      actuellement une coquille vide.

## 🔥 P2 — CI + linter (le manque le plus criant)

47 fichiers de tests, un test de parité KV entre deux runtimes, du vitest côté Worker, et
**aucun `.github/workflows/`, aucun linter configuré**. C'est le premier signal regardé sur un
poste MLOps/DE.

- [x] `.github/workflows/ci.yml` : `poetry run pytest -k "not pretrain"`, `npm test --prefix web/cf`,
      `npm run --prefix web/cf typecheck`, `ruff check`.
- [x] Ajouter `ruff` au groupe dev de `pyproject.toml` + config minimale.
- [x] Badge CI dans le README (à côté des badges existants).

## ✅ P3 — Fermer la boucle d'éval LLM et publier le chiffre

Critère de succès défini dans CLAUDE.md, **atteint le 2026-09-04** : 96 % de mistakes utiles
sur 12 reviews par-game annotées (cible ≥70 % sur ≥10). Sur un poste GenAI, la question
d'entretien est littéralement « comment tu évalues ton LLM ? » : la réponse est désormais un
chiffre public, adossé à trois familles d'éval (annotation humaine, ancrage déterministe,
contrefactuel).

- [x] Annoter 10 à 15 reviews par-game (`poetry run python src/04_coaching/feedback.py annotate --player spadzze --pending`).
      **Fait au 2026-09-04 : 12/12 analyses par-partie annotées, mistakes utiles 96 %**
      (cible ≥70 % sur ≥10) ; 0 rejet « trop-vague » sur les 5 dernières. Utilité globale
      87 % sur 92 items / 17 reviews ; par section forces 73 %, erreurs 93 %, habitudes 80 %,
      focus 94 %. Publié dans KV via `sync_cloudflare.py --push-coaching` (12 reviews
      par-game lisibles sur `/api/c/spadzze/reviews?kind=game`).
- [x] **Déployer le Worker** : fait le 2026-09-04. `GET /api/c/spadzze/eval` répond 200 en
      production avec `target_met: true`, et le Worker (qui recalcule à la lecture) retrouve
      exactement les chiffres de la CLI (12 analyses, 0,9643) : la parité des deux runtimes
      est vérifiée en prod, pas seulement par `tests/test_eval_parity.py`.
- [x] Publier le taux obtenu, **même mauvais**, sur la page CV et sur le site (`/readme`).
      Fait côté site : `GET /api/c/<slug>/eval` (calculé à la lecture, votes web inclus) +
      bandeau en tête de l'onglet Coaching + carte « critère de succès » dans `/readme`.
      Page CV faite le 2026-09-04 : le taux 96 % / 12 analyses y figure, avec le détail
      par section (73 % sur les forces) plutôt que le seul chiffre flatteur.
- [x] Enrichir chaque record de review : modèle, version de prompt, latence, coût estimé.
      Transforme « j'ai branché un LLM » en « j'ai un harness d'éval ».
- [x] **Éval automatique** (au-delà de l'annotation) : `grounding.py` (les chiffres et
      horodatages cités existent-ils ? + violations d'asymétrie, détecteur calibré par
      contrôle négatif) et `counterfactual.py` (perturbation du payload → la sortie
      suit-elle ?). Les deux tournent sans humain et en CI.

## ✅ P4 — Reformuler la page CV autour des bons arguments

La page met en avant « compression ×13 » et « 6 couches » : de la plomberie, non différenciante.
Les trois vrais atouts, dans cet ordre :

- [x] **Le protocole d'éval** (à mettre en titre) : split canonique par joueur
      (`data/04_dataset/split.json`), purged CV avec fuite mesurée (≈ +0.005 d'AUC via les games
      partagées), headline sur test held-out (0.677) et non sur du OOF optimiste, script d'audit
      de fuite dédié (`audit_leakage.py`). Quasiment aucun projet de portfolio ne fait ça.
- [x] **La contrainte produit encodée dans le code** : `COACHING_SAFE` vs `ML_ONLY`, avec `assert`
      au chargement de `compare.py` qui crashe si une feature interdite fuit vers le coaching.
      C'est de la gouvernance de features, pas du storytelling.
- [x] **Les résultats négatifs assumés** : SSL `delta = -0.0195`, `high_elo` plafonné à 0.589,
      per-game déprécié. Un lead DS valorise ça plus que trois AUC flatteuses.
- [x] Toujours donner le contexte du chiffre (n + protocole). « AUC 0,645 vs 0,633 » seul ne
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

- [x] Le README annonce « Python 3.11+ », `pyproject.toml` exige `>=3.13,<3.15`. Quiconque suit
      le README échoue à l'install. Aligner.
- [x] `fastapi` et `uvicorn` toujours en dépendances principales alors que le backend est archivé ;
      le `Dockerfile` à la racine builde le legacy. Déplacer `web/backend/` + `Dockerfile` dans
      `legacy/` ou les supprimer (git garde l'historique).
- [x] Le caveat torch/xgboost (double load libomp -> SIGSEGV sur Mac) n'existe que dans CLAUDE.md.
      Le remonter dans le README, sinon un lecteur qui tente le training se prend un segfault.

## ✅ P8 — Model card

- [x] `docs/MODEL_CARD.md` : 12 sections, chiffres extraits des artefacts de run (pas de
      recopie manuelle). Données (patch 16.13, EUW, 47 701 parties, 14 919 joueurs), label,
      protocole (split canonique, purged CV, fuite mesurée à +0,005), métriques test held-out
      (rang AUC 0.677 n=147 ; LP Spearman 0.5373 n=170), dispersion = 65,3 % du signal SHAP,
      résultats négatifs (SSL −0,019 et v2 −0,025 ; calibration master/GM inversée ; N>30
      inutile), limites (transfert de rang aux deux ADC, GM à n=16, drift modèle/dataset,
      un seul patch, drift temporel du label LP, un seul annotateur).
- [x] L'exposer sur `/readme` du site : carte « Model card » en fin d'onglet Data Science & ML,
      avec le tableau sélection/test, la purged CV, la dispersion et les résultats négatifs.
      Verrouillé par `tests/web/test_frontend.py::test_readme_exposes_the_model_card` (le test
      exige que les résultats **négatifs** restent affichés, pas seulement les bons).
- [ ] Redéployer le Worker pour publier la carte (`cd web/cf && npx wrangler deploy`).

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

## Journal P4 — page CV (2026-09-04)

Faite dans `~/code/website/cv` (`data.yaml` = source du site, `projets/coaching-lol.md`
= fiche privée d'entretien), deux commits, site régénéré via `./build.sh --site`.
**Non déployé** : lancer `./build.sh --site --deploy` pour publier.

Erreurs factuelles corrigées, qui auraient mal tourné en entretien :

- **LightGBM** annoncé dans la stack et deux formulations de CV alors que
  `train_ensemble.py` l'a retiré explicitement (« jumeau GBDT de XGBoost »).
- **Docker** dans la stack alors que le `Dockerfile` vient d'être supprimé.
- Le backend annoncé « archivé » alors qu'il est supprimé.
- La limite « le schéma ne vérifie pas que l'evidence correspond au payload » :
  devenue fausse depuis `grounding.py`.
- La note « `?review=` en cours d'implémentation » : en production.
