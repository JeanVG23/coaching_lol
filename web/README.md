# web/ — interface web de coaching_lol

Le site de production est un **Cloudflare Worker TypeScript** qui sert dans le même
déploiement :

- l'API sous `/api/*` (`web/cf/src/`) ;
- le frontend statique (`web/frontend/`) via le binding `ASSETS` ;
- les données de consultation dans Cloudflare KV (`DATA`).

Production : <https://coaching-lol.jeanvg.fr>

Les clés restent côté serveur. La collecte Riot, les agrégations et l'entraînement ML
continuent de tourner localement en Python ; seul le résultat utile au site est synchronisé
vers KV.

## Développement local du Worker

```bash
cd web/cf
npm install
npx wrangler dev
```

Pour lire les vraies données KV pendant un test local :

```bash
npx wrangler dev --remote
```

Commandes de validation :

```bash
cd web/cf
npm test
npm run typecheck
```

Le backend FastAPI dans `web/backend/` reste la référence Python et peut encore être
lancé pour comparaison :

```bash
poetry run uvicorn main:app --app-dir web/backend --reload
```

## Synchroniser les données locales vers KV

Le script fusionne les données locales avec celles déjà présentes dans KV ; il ne supprime
pas l'historique distant.

```bash
poetry run python src/collection/sync_cloudflare.py --dry-run
poetry run python src/collection/sync_cloudflare.py
```

Variables requises dans `.env` ou dans l'environnement :

- `CF_API_TOKEN` — jeton limité à l'écriture KV ;
- `CF_ACCOUNT_ID` — identifiant du compte Cloudflare ;
- `CF_NAMESPACE_ID` — namespace lié au binding `DATA`.

Le secret `OLLAMA_API_KEY` de production se configure avec Wrangler et ne doit pas être
placé dans le dépôt :

```bash
cd web/cf
npx wrangler secret put OLLAMA_API_KEY
```

## Déployer

```bash
cd web/cf
npm test
npm run typecheck
npm run deploy
```

Le domaine personnalisé `coaching-lol.jeanvg.fr` est rattaché au Worker dans Cloudflare.
Après chaque déploiement, vérifier au minimum :

```bash
curl https://coaching-lol.jeanvg.fr/api/health
curl https://coaching-lol.jeanvg.fr/api/accounts
```

## Architecture

```text
web/
  cf/
    src/index.ts        # routeur Worker, CORS, erreurs et assets
    src/kv.ts           # accès typé à Cloudflare KV
    src/coach.ts        # coaching en Server-Sent Events
    src/llm.ts          # client Ollama Cloud avec retries
    src/schema.ts       # validation des entrées/sorties
    wrangler.toml       # Worker, assets et binding DATA
  frontend/
    index.html          # shell SPA
    style.css           # composants et thème
    app.js              # Alpine, API et consommation SSE
  backend/              # implémentation FastAPI de référence
src/collection/sync_cloudflare.py  # publication locale vers KV
```

Flux de données :

```text
Riot + pipeline Python local -> sync_cloudflare.py -> Cloudflare KV
                                                    -> Worker API -> navigateur
navigateur -> POST /api/coach -> Worker -> Ollama Cloud -> événements SSE
```

## Endpoints de production

- `GET /api/health` — état du Worker ;
- `GET /api/accounts` — comptes préconfigurés et indicateurs ;
- `GET /api/c/{slug}/games` — historique paginé ;
- `GET /api/c/{slug}/rank` — rang Riot mis en cache ;
- `GET /api/c/{slug}/predicted-rank` — estimation ML per-player, disponible à partir de
  15 parties ADC ;
- `GET /api/c/{slug}/reviews` — historique des coachings ;
- `GET /api/c/{slug}/shap` — profil SHAP local ;
- `POST /api/coach` — génération Ollama diffusée en SSE ;
- `POST /api/feedback` — annotation des conseils.

La mise à jour Riot n'est volontairement plus exposée dans l'interface publique : collecte,
calcul ML et synchronisation se font depuis la machine locale. Les anciens endpoints
`/api/fetch` et `/api/jobs/{id}` ne font pas partie du Worker.

## État de la migration Fly.io

Le trafic de production est entièrement basculé sur Cloudflare. L'ancien domaine Fly ne
sert plus le site et l'application Fly est inactive depuis la fin de la période d'essai.
Sa suppression physique est bloquée par Fly tant qu'aucune carte bancaire n'est ajoutée ;
elle est donc laissée en l'état pour éviter d'activer une facturation.

L'historique local disponible a été fusionné dans KV (17 reviews et 5 feedbacks lors de la
migration). Le volume Fly n'a pas été rapatrié davantage, par choix : son contenu n'était
pas jugé important.

## Périmètre fonctionnel

- `/` liste les comptes préconfigurés ;
- `/c/{slug}` affiche historique, rang, estimation ML, coaching, feedback et SHAP ;
- `/readme` explique les recommandations et leurs benchmarks challenger ;
- l'authentification reste reportée : le site est publiquement accessible ;
- le coaching reste agrégé sur N parties ; le compte-rendu par partie et la CV restent des
  évolutions séparées.

La conception fonctionnelle détaillée est conservée dans
`docs/superpowers/specs/2026-07-01-web-app-design.md`.
