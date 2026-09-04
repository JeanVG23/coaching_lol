"""Tests de câblage du frontend statique, lus sur disque.

Pas de test d'interactivité Alpine (pas de Playwright) : on verrouille le câblage.
Le service HTTP des fichiers appartient au Worker Cloudflare (binding `assets` de
wrangler.toml, testé côté vitest) ; ces tests portent sur le contenu livré.
"""
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[2] / "web" / "frontend"
WRANGLER = Path(__file__).resolve().parents[2] / "web" / "cf" / "wrangler.toml"


def _read(name):
    return (FRONTEND / name).read_text()


def test_index_references_its_assets():
    body = _read("index.html")
    assert 'x-data="app()"' in body
    assert '/vendor/alpine.min.js' in body
    assert '/vendor/chart.umd.min.js' in body
    assert '/style.css' in body
    assert '/app.js' in body
    assert '/static/' not in body


def test_worker_serves_the_spa_fallback():
    """Les routes /c/<slug> et /readme n'existent que cote client : sans ce repli,
    un acces direct ou un rechargement rend un 404."""
    toml = WRANGLER.read_text()
    assert 'directory = "../frontend"' in toml
    assert 'not_found_handling = "single-page-application"' in toml


def test_assets_present_and_non_empty():
    for name in ("style.css", "app.js", "vendor/alpine.min.js",
                 "vendor/chart.umd.min.js", "og.png"):
        asset = FRONTEND / name
        assert asset.exists(), name
        assert asset.stat().st_size > 1000, name


def test_style_css_has_tokens():
    css = _read("style.css")
    for token in ("--bg:#0e1116", "--panel:#16181d", "--gold:#c8aa6e",
                  "--win:#3fb950", "--loss:#f85149", "tabular-nums"):
        assert token in css, token


def test_app_js_has_router_and_helpers():
    js = _read("app.js")
    assert "function app()" in js
    assert "function api(" in js
    assert "getReader()" in js
    assert 'event === "review"' in js
    assert "location.pathname" in js


def test_components_do_not_double_init():
    """Alpine appelle deja init() d'un objet x-data : le repeter double les requetes."""
    body = _read("index.html")
    assert 'x-init="init()"' not in body


def test_deep_link_to_a_game_review_wired():
    """?review=<match_id> doit ouvrir l'onglet coaching sur la vue par-partie.

    Le lien depuis le CV pointe une analyse precise : sans ce cablage il retombe sur
    la page du compte, qui n'explique rien a un visiteur exterieur.
    """
    js = _read("app.js")
    assert "function deepLinkOf(" in js
    assert 'q.get("review")' in js
    assert "resolvePendingReview" in js
    assert "gameMatchId(r) === wanted" in js


def test_home_page_wired():
    body = _read("index.html")
    js = _read("app.js")
    assert "function homePage()" in js
    assert "/api/accounts" in js
    # marqueurs propres au template home (absents de la nav switcher F1)
    assert 'class="accounts-grid"' in body
    assert 'class="account-card"' in body


def test_account_page_history_wired():
    body = _read("index.html")
    js = _read("app.js")
    assert "function accountPage(" in js
    assert "/api/c/" in js and "/games" in js
    assert "/api/fetch" not in js
    assert "/api/jobs/" not in js
    assert "refresh_cloudflare.py" in body
    assert 'class="game-row"' in body or "game-row" in body
    assert "job-banner" in body


def test_account_header_rank_wired():
    body = _read("index.html")
    js = _read("app.js")
    assert "/api/c/" in js and "/rank" in js
    assert "loadRank" in js
    assert "rank-badge" in body or "rankLabel" in body


def test_dashboard_uses_accessible_tabs_and_refresh_help():
    body = _read("index.html")
    assert 'role="tablist"' in body
    assert 'type="button" class="tab"' in body
    assert 'class="sync-help"' in body
    assert "Mettre à jour mes données" in body
    # La page est publique : l'instruction terminal ne concerne que le proprietaire.
    assert 'x-show="ownerView"' in body


def test_owner_only_sync_help_gated_in_js():
    js = _read("app.js")
    assert "function ownerViewFrom(" in js
    assert "ownerView:" in js


def test_share_preview_meta_present():
    """Sans rendu serveur, un lien partage n'affiche rien sans ces balises."""
    body = _read("index.html")
    for tag in ('property="og:title"', 'property="og:image"',
                'property="og:description"', 'name="twitter:card"'):
        assert tag in body, tag


def test_game_reviews_primer_explains_the_pipeline():
    body = _read("index.html")
    assert 'class="review-primer"' in body
    assert "API Riot" in body


def test_history_tab_predicted_rank_wired():
    body = _read("index.html")
    js = _read("app.js")
    assert "/api/c/" in js and "/predicted-rank" in js
    assert "loadPredictedRank" in js
    assert "predictedRank" in body


def test_coaching_tab_wired():
    body = _read("index.html")
    js = _read("app.js")
    assert "/api/coach" in js
    assert "/api/c/" in js and "/reviews" in js
    assert "/api/feedback" in js
    assert "NEG_TAGS" in js
    # La séparation agrégé/par-game est portée par le filtre `kind` de l'API,
    # pas par des type-guards côté client (retirés : rien ne les appelait).
    assert "kind=aggregate" in js
    assert "kind=game" in js
    assert "gameReviewsCount" in js
    assert "insight-card" in body or "evidence-chip" in body
    assert 'class="coach-builder"' in body
    assert 'class="segmented-choice"' in body
    assert "Joueurs Challenger" in body
    assert "Analyses de parties" in body
    assert "game-review-layout" in body
    assert "selectedGameReview" in js


def test_feedback_note_textarea_wired():
    """Le champ note (Pydantic FeedbackItem.note, déjà accepté par l'API) était
    modélisé de bout en bout (schema/API/CLI) mais jamais exposé côté web — seuls
    les boutons y/n/tag existaient. Verrouille l'ajout du textarea + saveNote."""
    body = _read("index.html")
    js = _read("app.js")
    assert "fb-note-input" in body
    assert "saveNote" in js
    assert "noteDraft" in js


def test_feedback_sends_full_map():
    """Régression F4 : submitFb doit envoyer la fbMap complète (le backend
    écrase la ligne par ts — un envoi par insight perdait les notations précédentes)."""
    js = _read("app.js")
    assert "Object.entries(newMap)" in js
    # l'ancien pattern mono-insight ne doit plus être présent
    assert "responses = { [key]" not in js


def test_shap_tab_wired():
    body = _read("index.html")
    js = _read("app.js")
    assert "/api/c/" in js and "/shap" in js
    assert "new Chart(" in js
    assert "shap-wrap" in body or "shap-empty" in body


def test_readme_page_wired():
    body = _read("index.html")
    js = _read("app.js")
    assert "function readmePage()" in js
    # contenu vulgarisé clé présent dans le HTML servi
    assert "asymétrie" in body.lower() or "asymetrie" in body.lower()
    assert "benchmark" in body.lower()
    assert "positionnement" in body.lower()


def test_eval_rate_published_in_the_coaching_tab():
    """Le critere de succes du projet (>=70 % d'erreurs utiles sur >=10 analyses)
    n'existait que dans la CLI : un coach dont personne ne voit l'evaluation reste
    une opinion. Le taux est lu en direct depuis KV, vote compris."""
    body = _read("index.html")
    js = _read("app.js")
    assert 'class="eval-strip"' in body
    assert "mistake_useful_rate" in body
    assert "loadEval" in js
    assert "/eval" in js


def test_readme_states_the_success_criterion():
    body = _read("index.html")
    assert "≥70 %" in body and "10 analyses de parties annotées" in body


def test_readme_exposes_the_model_card():
    """Les metriques ML n'existaient que dans data/05_model/*.json : un modele
    servi sans carte publique est une boite noire. On verrouille la presence du
    headline test held-out, du protocole et des resultats negatifs (le point qui
    distingue une model card d'une plaquette)."""
    body = _read("index.html")
    assert "Model card" in body
    assert "AUC 0.677" in body and "Spearman 0.537" in body
    assert "held-out" in body
    assert "Purged CV" in body or "purg" in body.lower()
    assert "docs/MODEL_CARD.md" in body
    # les resultats negatifs doivent rester affiches, pas seulement les bons
    assert "auto-supervisé" in body or "auto-supervise" in body
    assert "déprécié" in body
