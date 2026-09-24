"""Tests de câblage statique du frontend Vite/Vue.

Les interactions sont couvertes par Vitest dans web/cf/client ; ce fichier
verrouille surtout l'architecture livrée et les contrats entre composants.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web" / "cf"


def _read(name: str) -> str:
    return (WEB / name).read_text()


def test_index_mounts_the_vue_application():
    body = _read("index.html")
    assert '<div id="app"></div>' in body
    assert 'type="module" src="/client/main.ts"' in body
    assert "x-data" not in body and "Alpine" not in body
    assert "/static/" not in body


def test_worker_serves_the_spa_fallback():
    toml = _read("wrangler.toml")
    assert "directory =" not in toml
    assert 'not_found_handling = "single-page-application"' in toml
    assert "cloudflare()" in _read("vite.config.ts")


def test_assets_and_dependencies_are_current():
    for name in ("client/style.css", "client/main.ts", "client/App.vue", "public/og.png"):
        asset = WEB / name
        assert asset.exists() and asset.stat().st_size > 100, name
    package = _read("package.json")
    assert '"vue"' in package and '"vue-router"' in package
    assert '"alpinejs"' not in package


def test_style_css_has_tokens():
    css = _read("client/style.css")
    for token in ("--bg:", "--panel:", "--gold:", "--win:", "--loss:", "tabular-nums"):
        assert token in css


def test_router_and_canonical_are_wired():
    main = _read("client/main.ts")
    router = _read("client/router.ts")
    body = _read("index.html")
    assert "createApp(App).use(router)" in main
    assert "createRouter" in router and "createWebHistory" in router
    assert "router.afterEach(syncCanonical)" in main
    assert 'link[rel="canonical"]' in main and 'meta[property="og:url"]' in main
    assert '<link rel="canonical" href="https://riftsense.jeanvg.fr/">' in body


def test_shell_and_home_are_vue_components():
    shell = _read("client/App.vue")
    home = _read("client/pages/HomePage.vue")
    assert "/api/accounts" in shell
    assert "route.name === 'home'" in shell
    assert "accounts-grid" in home and "account-card" in home
    assert "<RegisterForm" in home


def test_account_dashboard_is_vue_owned():
    page = _read("client/pages/AccountPage.vue")
    for component in ("AccountProfile", "GameHistory", "CoachingControls", "GlobalCoaching", "GameReviews", "ShapProfile"):
        assert f"<{component}" in page
    assert 'role="tablist"' in page
    assert 'v-if="ownerView"' in page and "refresh_cloudflare.py" in page


def test_account_header_and_history_contracts():
    page = _read("client/pages/AccountPage.vue")
    profile = _read("client/components/AccountProfile.ce.vue")
    history = _read("client/components/GameHistory.ce.vue")
    assert "/predicted-rank" in profile and "/rank" in profile
    assert '@prediction-loaded="predictedRank = $event"' in page
    assert ':predicted-rank="predictedRank"' in page
    assert "/games?page=${page.value}&size=${size}" in history
    assert "review_status" in history


def test_coaching_context_and_generation_are_wired():
    page = _read("client/pages/AccountPage.vue")
    history = _read("client/components/GameHistory.ce.vue")
    global_coaching = _read("client/components/GlobalCoaching.ce.vue")
    assert "/coaching-context" in page
    assert 'fetch("/api/coach"' in page and 'fetch("/api/coach/game"' in page
    assert "getReader()" in page and 'event === "review"' in page
    assert "props.coachingContext?.matches?.[matchId]" in history
    # Le meta-strip (transparence statistique) a été retiré du bilan global
    # en 8c1eb17 ; le garder absent évite qu'il refasse fuiter des chiffres
    # bruts que le hero de synthèse ne montre plus.
    assert "n_game_reviews_available" not in global_coaching and "n_game_reviews_used" not in global_coaching


def test_aggregate_and_game_reviews_stay_separate():
    page = _read("client/pages/AccountPage.vue")
    assert "kind=aggregate" in page and "kind=game" in page
    assert "findMatchingReview" in page and "gameReviewsCount" in page


def test_deep_link_to_game_review_is_preserved():
    page = _read("client/pages/AccountPage.vue")
    games = _read("client/components/GameReviews.vue")
    assert "route.query.review" in page
    assert 'reviewQuery ? "coaching"' in page and 'reviewQuery ? "games"' in page
    assert "resolveTarget" in games and "gameMatchId(item) === matchId" in games


def test_game_reviews_own_filters_pagination_and_details():
    component = _read("client/components/GameReviews.vue")
    # Le pavé pédagogique ("review-primer") a été retiré en 8c1eb17 ; le
    # garder absent évite qu'il revienne encombrer la liste des games.
    assert 'class="review-primer"' not in component
    assert 'class="game-review-layout"' in component
    assert "filterResult" in component and "filterChampion" in component
    assert "async function loadMore" in component and "/reviews?kind=game&page=" in component
    assert "async function selectReview" in component


def test_interactive_game_chat_is_vue_owned():
    component = _read("client/components/GameReviews.vue")
    assert 'class="game-chat"' in component
    assert "async function sendChat" in component and 'fetch("/api/chat"' in component
    assert "hidden enemy information" in component


def test_game_feedback_sends_full_map_and_notes():
    component = _read("client/components/GameReviews.vue")
    assert 'class="fb-note-input"' in component and "async function saveNote" in component
    assert "const next = { ...feedback.value, [key]: entry }" in component
    assert "responses: next" in component and "responses = { [key]" not in component


def test_global_feedback_saving_is_wired():
    controls = _read("client/components/CoachingControls.ce.vue")
    global_coaching = _read("client/components/GlobalCoaching.ce.vue")
    assert 'fetch("/api/feedback"' in global_coaching
    # La mesure de qualité (eval-strip) et son rafraîchissement (evalRevision,
    # @feedback-saved) ont été retirés en 8c1eb17 ; CoachingControls.test.ts:19
    # vérifie déjà l'absence de .eval-strip côté Vitest.
    assert 'class="eval-strip"' not in controls and "mistake_useful_rate" not in controls


def test_game_review_leads_with_a_timeline_not_a_wall_of_text():
    # La vue par game se lit d'abord comme une frise : l'horodatage que le
    # schéma impose dans chaque evidence est la seule matière que l'onglet ML
    # ne peut pas produire, puisque sa décomposition n'a pas d'axe temporel.
    component = _read("client/components/GameReviews.vue")
    for marker in ("game-timeline-band", "timeline-mark", "game-verdict",
                   "markPosition(mark.at, durationMin)", "focusMark(mark)"):
        assert marker in component, marker
    # Niveau 1 = catégorie, titre, horodatages. Le reste attend le clic : sans
    # ce repli, six insights posent ~2 100 caractères d'un bloc.
    assert "insight-toggle" in component and "insight-expanded" in component
    assert "stamp-chip" in component and "insightOpen(" in component
    assert "game-chat-toggle" in component and 'v-if="chatOpen"' in component
    # Branches mortes retirées : ni `review.summary` ni `review.axes` n'existent
    # dans game_review.schema.json ni dans les données publiées.
    assert "game-chief-summary" not in component and "game-axis" not in component


def test_coaching_does_not_borrow_the_shap_bar_idiom():
    # Le coaching est un COMPLÉMENT de l'onglet ML, pas son doublon. Servir un
    # classement à barres des deux côtés ferait passer un comptage
    # d'occurrences pour un poids de modèle. La vue globale publie donc une
    # grille de présence à colonnes fixes (une par partie, dans l'ordre du
    # temps), et la vue par game une frise.
    for name in ("GlobalCoaching.ce.vue", "GameReviews.vue"):
        component = _read(f"client/components/{name}")
        for borrowed in ("bar-track", "barStyle", "shap-theme-row", "shap-highlight", "gauge-track"):
            assert borrowed not in component, f"{name} emprunte {borrowed} à l'onglet ML"
    global_coaching = _read("client/components/GlobalCoaching.ce.vue")
    for marker in ("recurrence-grid", "recurrence-cell", "data-match", "game-select"):
        assert marker in global_coaching, marker
    # La frise globale superpose les frises des parties sur une horloge de jeu.
    # Ses bandes ont la largeur de leur DURÉE : une largeur proportionnelle au
    # compte referait le diagramme en barres de l'onglet ML.
    for marker in ("global-timeline", "global-timeline-mark", "phase-band",
                   "bandWidth(band.from, band.to)", "global-timeline-trend"):
        assert marker in global_coaching, marker
    # Le focus n'est plus servi deux fois : le hero le porte, la quatrième
    # colonne qui le répétait a disparu.
    assert "Focus &amp; Confidence" not in global_coaching
    assert "global-focus-confidence" in global_coaching


def test_both_coaching_views_share_one_collapse_idiom_and_one_clock():
    # Deux gestes différents sur deux onglets de la même fonctionnalité
    # coûteraient plus cher que le texte économisé : le repli et l'axe de
    # temps vivent donc dans la feuille partagée, pas en scoped dans un seul
    # composant, et les deux vues portent la même classe de frise.
    sheet = _read("client/styles/coaching-insights.css")
    for rule in (".shared-insights .insight-toggle", ".shared-insights .insight-chevron",
                 ".shared-insights .insight-expanded", ".coach-timeline .timeline-axis",
                 ".coach-timeline .legend-dot"):
        assert rule in sheet, rule
    for name in ("GameReviews.vue", "GlobalCoaching.ce.vue"):
        component = _read(f"client/components/{name}")
        assert "coach-timeline" in component, name
        assert "insight-toggle" in component, name


def test_timestamp_reading_has_a_single_owner():
    # Le Worker publie les instants, le client les replace sur un axe : la
    # regex qui les lit dans une evidence ne doit exister qu'UNE fois, sinon
    # les deux runtimes divergent en silence sur ce que « 14:16 » veut dire.
    owner = _read("src/game_moments.ts")
    assert "[0-5]" in owner and "export function parseTimestamps" in owner
    for name in ("client/game-timeline.ts", "client/components/GameReviews.vue",
                 "client/components/GlobalCoaching.ce.vue", "src/index.ts"):
        assert "[0-5]\\d" not in _read(name), name


# Pictogrammes couleur : chaque OS les dessine à sa façon et ils ignorent la
# palette. Les icônes passent par <AppIcon> (trait Lucide, currentColor) ou par
# les masques d'objectifs. Restent permis les glyphes typographiques qu'Inter
# dessine lui-même (flèches, coches, ✦ ornement de la DA).
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]")
_TYPOGRAPHIC = set("✦✓✕✗✎")


def test_client_uses_icons_not_emojis():
    offenders = []
    for path in sorted((WEB / "client").rglob("*")):
        if path.suffix not in {".ts", ".vue", ".css"}:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if any(char not in _TYPOGRAPHIC for char in _EMOJI.findall(line)):
                offenders.append(f"{path.relative_to(WEB)}:{number}")
    assert not offenders, offenders


def test_shap_renders_the_breakdown_without_a_canvas():
    # L'onglet ML n'a plus de canvas : les barres sont du DOM natif, donc
    # lisibles, sélectionnables et correctes en mobile. Un retour de Chart.js
    # ici réintroduirait un canevas à taille fixe et une dépendance que le
    # bundle n'a plus.
    component = _read("client/components/ShapProfile.ce.vue")
    assert "/shap" in component and "shap-empty" in component
    assert "chart.js" not in component and "<canvas" not in component
    # Les quatre couches de lecture, et les deux replis qui les allègent.
    for marker in ("shap-gauge", "shap-verdict", "shap-theme-row", "shap-highlight",
                   "shap-help-toggle", "shap-detail-toggle"):
        assert marker in component, marker


def test_readme_is_a_vue_page_and_keeps_methodology():
    page = _read("client/pages/ReadmePage.vue")
    assert "const tab = ref" in page and "v-if=\"tab === 'overview'\"" in page
    for phrase in ("asymmetry", "benchmark", "positioning", "Model card", "AUC 0.677", "Spearman 0.537", "held-out", "docs/MODEL_CARD.md", "self-supervised", "deprecated"):
        assert phrase.lower() in page.lower(), phrase


def test_readme_states_success_criterion_and_real_schema_bounds():
    import re
    page = _read("client/pages/ReadmePage.vue")
    assert "at least 70%" in page and "at least 10 rated game analyses" in page
    schema = (ROOT / "src" / "04_coaching" / "schema.py").read_text()
    review = schema.split("class Review")[1].split("class ")[0]
    for field in ("strengths", "mistakes", "habits"):
        line = next(line for line in review.splitlines() if line.strip().startswith(field))
        lo = int(re.search(r"min_length=(\d+)", line).group(1))
        hi = int(re.search(r"max_length=(\d+)", line).group(1))
        expected = f"{field}[{lo}]" if lo == hi else f"{field}[{lo}..{hi}]"
        assert expected in page


def test_registration_routes_and_polling_are_wired():
    shell = _read("client/App.vue")
    component = _read("client/components/RegisterForm.ce.vue")
    assert "route.name === 'register'" in shell and '<RegisterForm v-else-if=' in shell
    reg_lib = _read("client/account-registration.ts")
    assert ('"/api/register"' in component or '"/api/register"' in reg_lib) and "/api/register/${" in component
    assert "location.pathname.match" in component
    assert "onBeforeUnmount(stopPolling)" in component and "clearTimeout(timer)" in component
    for code in ("riot_id_not_found", "no_ranked_games", "riot_unavailable", "internal"):
        assert code in component or code in reg_lib


def test_authentication_is_shared_by_vue_components():
    shell = _read("client/App.vue")
    auth = _read("client/auth.ts")
    modal = _read("client/components/AuthModal.ce.vue")
    control = _read("client/components/AuthControl.ce.vue")
    assert "<AuthControl" in shell and "<AuthModal" in shell
    assert 'AUTH_TOKEN_KEY = "coach_auth_token"' in auth
    assert "export let authToken" in auth and "withAuthHeaders" in auth
    assert 'fetch("/api/auth/login"' in modal
    assert 'fetch("/api/auth/status"' in control and 'fetch("/api/auth/logout"' in control


def test_social_preview_robots_and_sitemap_are_present():
    body = _read("index.html")
    for tag in ('property="og:title"', 'property="og:image"', 'property="og:description"', 'name="twitter:card"'):
        assert tag in body
    robots = _read("public/robots.txt")
    sitemap = _read("public/sitemap.xml")
    assert "User-agent: *" in robots and "Disallow: /api/" in robots
    assert "Sitemap: https://riftsense.jeanvg.fr/sitemap.xml" in robots
    assert sitemap.startswith("<?xml") and "<loc>https://riftsense.jeanvg.fr/</loc>" in sitemap


def test_no_alpine_directives_or_runtime_remain():
    sources = "\n".join(path.read_text() for path in (WEB / "client").rglob("*") if path.suffix in {".ts", ".vue", ".js"})
    assert "Alpine" not in sources
    assert "x-data=" not in sources and "x-show=" not in sources and "x-if=" not in sources
    assert '"alpinejs"' not in _read("package.json")
