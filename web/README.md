# web/ — interface web de coaching_lol

Un seul process FastAPI (sur Fly.io) sert à la fois :
- **l'API** sous `/api/*` (`web/backend/main.py`),
- **le frontend statique** (`index.html`, CSS, JS) sous `/` et `/static` (`web/frontend/`).

Les clés (Riot, Ollama) restent côté serveur — jamais dans le navigateur.

## Lancer en local

```bash
.venv/bin/pip install -r requirements.txt        # fastapi + uvicorn la première fois
.venv/bin/python -m uvicorn main:app --app-dir web/backend --reload
```

Ouvre http://127.0.0.1:8000 — la page doit afficher « ✅ Backend en ligne ».
Sonde API : http://127.0.0.1:8000/api/health

## Déployer sur Fly.io

Prérequis : `flyctl` installé et authentifié (`fly auth login`).

```bash
fly apps create coaching-lol       # nom libre mais unique ; ajuste fly.toml en conséquence
fly deploy                          # build le Dockerfile + pousse l'image
fly open                            # ouvre l'URL publique
```

> La donnée (`data/`) n'est PAS dans l'image (cf. `.dockerignore`). Elle vivra sur un
> volume persistant Fly, monté au moment où on câblera le coaching réel.

## Structure

```
web/
  backend/main.py     # FastAPI : /api/* + sert index.html et /static
  frontend/
    index.html        # shell SPA (top-bar + templates par page)
    style.css         # tokens + composants (CSS sur-mesure, pas de build)
    app.js            # composants Alpine (app/homePage/accountPage/readmePage) + helpers
    vendor/
      alpine.min.js   # Alpine 3.x vendored
      chart.umd.min.js# Chart.js 4.x vendored
fly.toml
Dockerfile
.dockerignore
```

## Périmètre V1

> Cadrage fonctionnel validé (2026-07-01). Spec détaillée dans
> `docs/superpowers/specs/2026-07-01-web-app-design.md`.
>
> **Esthétique : sombre raffinée, classe/épuré (côté Apple, pas Razer).** Patterns UX
> inspirés d'op.gg/u.gg (rangées games denses, switcher de compte, onglets), mais fond
> charbon neutre, accent or discret, zéro néon. Densité équilibrée : le data reste
> compact, le narratif respire. Palette + principes de détail dans la spec.

### Pages

1. **`/` — Accueil = liste des comptes.** Cartes par slug préconfiguré
   (`spadzze`, `aceofspadzze`…), avec un mini-indicateur (nb de games en cache,
   dernière review). **Pas de création de compte depuis l'UI** — la liste est
   fixée côté serveur (le propriétaire seul peut l'éditer).
2. **`/c/{slug}` — Page compte.** Le cœur du site. Sections :
   - **Fetch** : bouton « Mettre à jour les games » (job = pull Riot → silver →
     gold, déterministe). Champ N (défaut 20, configurable). Statut du job en
     temps réel (« en cours… 12/20 » → « ✅ 18 games »).
   - **Historique** : tableau succinct des dernières games (champion, rôle,
     durée, KDA, win/loss), pagination légère.
   - **Coaching** : sélecteurs (scope `all`/`adc`/`zeri`, issue `loss`/`win`/
     `overall`, target challenger) + bouton « Générer le coaching » (job LLM
     séparé). Affiche la review (3 forces / 3 erreurs / 2 habitudes / 1 focus,
     preuves chiffrées). Historique des reviews précédentes.
   - **Feedback** : annotation par-insight (y/n/skip + tag + note) sur la review
     affichée. Taux de satisfaction visible.
   - **Mon profil ML (SHAP local)** : graphique interactif — contributions par
     feature pour *ce* compte, comparé au global. Survol/tri, pas un PNG figé.
3. **`/readme` — README.** Page statique : comment fonctionnent les recos
   coaching (asymétrie d'info, benchmark challenger, pourquoi positionnement >
   stats brutes, que mesurent les features). Vulgarisation.

### Parcours type

Accueil → clic compte → « Mettre à jour » (attendre fin du job) → régler
sélecteurs → « Générer le coaching » → lire la review → annoter le feedback →
consulter son profil SHAP.

### Décisions clés

- **Compte de coaching = un slug** (`spadzze`, `aceofspadzze`, smurfs…).
- **Auth reportée** — URL publique acceptée pour l'instant (faible proba de
  découverte). À revisiter si le site gagne en visibilité.
- **Fetch = pull + silver + gold en un seul job** (déterministe, 1 clic) ;
  **coaching LLM = job séparé** (coûteux/lent, relançable sans re-fetcher, p.ex.
  pour A/B de modèles ou changer les sélecteurs).
- **Coaching V1 = agrégé N games** (miroir du CLI actuel), pas par-game.

### Embarqué en V1

Fetch agrégé (pull+silver+gold, 1 job) · coaching LLM agrégé (job séparé) ·
historique games · feedback par-insight · SHAP local interactif · README ·
multi-compte préconfiguré.

### Repoussé (V2+)

Compte-rendu par-game · draft coaching live · UI d'ajout de compte · auth ·
SHAP explorateur global (rang/scope filtrable).

### Hors scope

Computer vision (Phase 2 du projet, gated sur la démonstration de valeur du
coach timeline).

## Endpoints prévus

- `GET  /api/accounts` — liste des slugs préconfigurés (+ indicateur par compte)
- `POST /api/fetch` — mettre à jour les games d'un slug (pull Riot → silver →
  gold, job async)
- `GET  /api/jobs/{id}` — suivi des jobs longs (fetch + coaching)
- `POST /api/coach` — générer le coaching (LLM, job async)
- `GET  /api/reviews` — historique des reviews d'un slug
- `POST /api/feedback` — annoter les conseils (boucle d'éval)
- `GET  /api/shap` — SHAP local d'un slug (graphique interactif)