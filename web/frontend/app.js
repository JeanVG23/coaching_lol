// coaching_lol — frontend SPA (Alpine). Aucune clé/secret ici : tout passe par /api/*.

const NEG_TAGS = ["asymetrie", "stat-inventee", "profondeur-en-faute",
  "trop-vague", "non-actionnable", "autre"];
// Étapes purement informatives du flux SSE (les 2 autres events changent le statut).
const SSE_PROGRESS = { payload: "payload construit", llm: "génération LLM…" };
const DDRAGON = (patch, champ) =>
  `https://ddragon.leagueofgraphs.com/cdn/${patch}.1/img/champion/${champ}.png`;

// Un seul casing de tier : `rank.tier` arrive en MAJUSCULES de l'API Riot et
// `predicted_rank` en minuscules du modèle ML — deux implémentations coexistaient.
function titleCase(value) {
  const s = String(value || "");
  return s ? s.charAt(0).toUpperCase() + s.slice(1).toLowerCase() : s;
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} on ${path}`);
  return r.json();
}

function fmtKDA(k, d, a) {
  const count = (value) => Array.isArray(value) ? value.length : Number(value || 0);
  const kills = count(k), deaths = count(d), assists = count(a);
  const ka = kills + assists;
  const ratio = deaths === 0 ? "Perfect" : (ka / deaths).toFixed(2);
  return `${kills}/${deaths}/${assists} · ${ratio}`;
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10);
  return new Intl.DateTimeFormat("fr-FR", {
    day: "numeric", month: "short", year: "numeric",
  }).format(date);
}

function routeOf(path) {
  if (path === "/" || path === "") return { name: "home" };
  const m = path.match(/^\/c\/([^/]+)$/);
  if (m) return { name: "account", slug: decodeURIComponent(m[1]) };
  if (path === "/readme") return { name: "readme" };
  return { name: "home" };
}

// Lien profond : /c/<slug>?tab=coaching&view=games&review=<match_id>.
// Un lien externe (page CV, partage) doit tomber sur une analyse de partie précise,
// pas sur la page d'accueil du compte qui n'explique rien. `review` implique l'onglet
// coaching et la vue par-partie : un seul paramètre suffit côté appelant.
// Le bloc « Mettre à jour mes données » demande de lancer une commande dans un
// terminal : il n'a de sens que pour le propriétaire du compte, alors que la page est
// publique et sert de démonstration. `?admin=1` l'active et le mémorise, `?admin=0` le
// coupe ; le stockage peut être indisponible (navigation privée), d'où le try/catch.
const OWNER_FLAG = "coachlol:owner";

function ownerViewFrom(search) {
  const asked = new URLSearchParams(search || "").get("admin");
  try {
    if (asked === "1") { localStorage.setItem(OWNER_FLAG, "1"); return true; }
    if (asked === "0") { localStorage.removeItem(OWNER_FLAG); return false; }
    return localStorage.getItem(OWNER_FLAG) === "1";
  } catch (e) {
    return asked === "1";
  }
}

const ACCOUNT_TABS = ["history", "coaching", "shap"];
const COACHING_VIEWS = ["overall", "games"];

function deepLinkOf(search) {
  const q = new URLSearchParams(search || "");
  const review = q.get("review") || null;
  let tab = q.get("tab");
  let view = q.get("view");
  if (review) { tab = "coaching"; view = "games"; }
  if (!ACCOUNT_TABS.includes(tab)) tab = null;
  if (!COACHING_VIEWS.includes(view)) view = null;
  return { tab, view, review };
}

// Une analyse d'une partie ne permet pas d'inférer des habitudes : son format est
// volontairement différent et elle ne doit pas remplacer le coaching global dans cet
// onglet. La séparation est portée par l'API (`?kind=aggregate` vs `?kind=game`,
// cf. loadReviews) et non plus par des type-guards qui redevinaient le type depuis
// la forme du payload alors que chaque record porte déjà son `kind`.

function coachErrorMessage(error) {
  const raw = typeof error === "string" ? error : String(error || "");
  if (/\b429\b/.test(raw)) {
    return "Le modèle est temporairement limité. Attends quelques minutes avant de relancer le coaching.";
  }
  if (/OLLAMA_API_KEY/.test(raw)) return "Le service de coaching n’est pas configuré correctement.";
  return raw || "Le coaching n’a pas pu être généré. Réessaie dans un instant.";
}

function app() {
  return {
    path: location.pathname,
    accounts: [],
    accountsLoading: true,
    get route() { return routeOf(this.path); },

    init() {
      window.addEventListener("popstate", () => { this.path = location.pathname; });
      api("/api/accounts").then(a => { this.accounts = a; this.accountsLoading = false; })
        .catch(() => { this.accountsLoading = false; });
    },

    go(p) {
      if (p === this.path) return;
      history.pushState({}, "", p);
      this.path = p;
    },

    switcherOpen: false,
    toggleSwitcher() { this.switcherOpen = !this.switcherOpen; },
    closeSwitcher() { this.switcherOpen = false; },
    formatDate,
  };
}

// Placeholder components — remplis par les tâches suivantes.
function homePage() {
  return {
    init() {
      // Les comptes sont chargés par le store app() parent (GET /api/accounts).
      // Pas de fetch ici — on consomme `accounts` via le scope hérité dans le HTML.
    },
  };
}
function accountPage(slug, search) {
  const deep = deepLinkOf(search === undefined ? location.search : search);
  return {
    slug,
    ownerView: ownerViewFrom(search === undefined ? location.search : search),
    tab: deep.tab || "history",
    pendingReviewId: deep.review,
    games: [], page: 1, size: 20, total: 0,
    gamesLoading: true, gamesError: null,
    job: null, // {type, status, progress, error}
    rank: null, rankLoading: true,
    predictedRank: null, predictedRankLoading: true,
    // coaching
    scope: "adc", outcome: "loss", target: "challenger",
    reviews: [], review: null, aggregateReviewsTotal: 0,
    gameReviews: [], selectedGameReview: null, gameReviewsPage: 1, gameReviewsLoading: false,
    gameReviewLoading: false, gameReviewError: null,
    chatMessages: [], chatDraft: "", chatBusy: false, chatError: null,
    coachingView: deep.view || "overall", gameReviewsCount: 0, reviewsLoading: true,
    reviewsInFlight: false,
    evalReport: null, evalLoading: false,
    fbMap: {}, fbBusy: {}, noteDraft: {}, feedbackError: null,
    coachOpen: false,
    // shap (F5)
    shap: null, shapLoading: false, shapSort: "abs", chart: null,

    init() {
      this.loadGames(); this.loadRank(); this.loadPredictedRank();
      // setTab porte déjà le chargement paresseux de chaque onglet : on le rejoue
      // pour l'onglet ouvert par le lien profond plutôt que de dupliquer la logique.
      if (this.tab !== "history") this.setTab(this.tab);
    },

    // Charge une URL dans un champ, avec drapeau de chargement et valeur de repli :
    // loadRank / loadPredictedRank / loadShap avaient le même corps recopié.
    async loadInto(field, flag, url, fallback = null, after = null) {
      this[flag] = true;
      try {
        this[field] = await api(url);
        if (after) after();
      } catch (e) { this[field] = fallback; }
      finally { this[flag] = false; }
    },

    async loadGames() {
      this.gamesLoading = true; this.gamesError = null;
      try {
        const d = await api(`/api/c/${this.slug}/games?page=${this.page}&size=${this.size}`);
        this.games = d.items; this.total = d.total;
      } catch (e) { this.gamesError = "Impossible de charger les parties. Réessaie dans un instant."; }
      finally { this.gamesLoading = false; }
    },

    loadRank() {
      return this.loadInto("rank", "rankLoading", `/api/c/${this.slug}/rank`);
    },

    rankLabel() {
      if (!this.rank || !this.rank.tier) return "Non renseigné";
      return `${titleCase(this.rank.tier)} ${this.rank.division} · ${this.rank.league_points} LP`;
    },

    loadPredictedRank() {
      return this.loadInto("predictedRank", "predictedRankLoading",
                           `/api/c/${this.slug}/predicted-rank`);
    },

    rankTierLabel: titleCase,

    async prevPage() { if (this.page > 1) { this.page--; this.loadGames(); } },
    async nextPage() {
      if (this.page * this.size < this.total) { this.page++; this.loadGames(); }
    },

    setTab(t) {
      this.tab = t;
      if (t === "coaching" && this.reviews.length === 0 && this.reviewsLoading) {
        this.loadReviews();
      }
      if (t === "coaching" && this.evalReport === null) { this.loadEval(); }
      if (t === "shap" && this.shap === null) { this.loadShap(); }
    },

    async setCoachingView(view) {
      this.coachingView = view;
      await this.loadFeedback(this.activeFeedbackReview());
    },

    // Taux d'utilite du coaching : la boucle d'evaluation du projet, affichee
    // meme quand elle est mauvaise. Un conseil LLM non evalue n'est qu'une opinion.
    loadEval() {
      return this.loadInto("evalReport", "evalLoading", `/api/c/${this.slug}/eval`);
    },

    pct(v) { return v === null || v === undefined ? "—" : `${Math.round(v * 100)} %`; },

    loadShap() {
      return this.loadInto("shap", "shapLoading", `/api/c/${this.slug}/shap`,
                           { available: false, drivers: [] },
                           () => {
                             if (this.shap.available) this.$nextTick(() => this.renderChart());
                           });
    },

    sortedDrivers() {
      const d = (this.shap?.drivers || []).slice();
      if (this.shapSort === "abs") d.sort((a, b) => Math.abs(b.mean_shap) - Math.abs(a.mean_shap));
      else d.sort((a, b) => b.mean_shap - a.mean_shap);
      return d.slice(0, 16); // top 16 pour la lisibilité
    },

    renderChart() {
      this.destroyChart();
      const cv = this.$root.querySelector("#shap-canvas");
      if (!cv || !window.Chart) return;
      const d = this.sortedDrivers();
      this.chart = new Chart(cv, {
        type: "bar",
        data: {
          labels: d.map(x => x.feature),
          datasets: [{
            data: d.map(x => x.mean_shap),
            backgroundColor: d.map(x => x.mean_shap >= 0 ? "#c8aa6e" : "#f85149"),
            borderRadius: 3, borderSkipped: false,
          }],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { callbacks: {
            label: c => `SHAP ${c.raw.toFixed(4)}`,
          } } },
          scales: {
            x: { grid: { color: "#2a2d34" }, ticks: { color: "#9a9da4", font: { size: 11 } } },
            y: { grid: { display: false }, ticks: { color: "#e8e9ec", font: { size: 11 } } },
          },
        },
      });
    },

    toggleSort() {
      this.shapSort = this.shapSort === "abs" ? "val" : "abs";
      this.renderChart();
    },

    destroyChart() {
      if (this.chart) { this.chart.destroy(); this.chart = null; }
    },

    async loadReviews() {
      // Alpine initialise le composant deux fois (x-data dans un template x-if
      // re-rendu) : sans ce garde, le second appel concurrent rejouait la selection
      // par defaut et ecrasait la review ciblee par le lien profond.
      if (this.reviewsInFlight) return;
      this.reviewsInFlight = true;
      this.reviewsLoading = true;
      try {
        const [aggregatePage, gamePage] = await Promise.all([
          api(`/api/c/${this.slug}/reviews?kind=aggregate&page=1&size=20`),
          api(`/api/c/${this.slug}/reviews?kind=game&page=1&size=20`),
        ]);
        this.reviews = aggregatePage.items || [];
        this.aggregateReviewsTotal = aggregatePage.total || 0;
        this.review = this.reviews.length ? this.reviews[0] : null;
        this.gameReviews = gamePage.items || [];
        this.gameReviewsPage = gamePage.page || 1;
        this.gameReviewsCount = gamePage.total || 0;
        const target = await this.resolvePendingReview();
        const selected = target || this.gameReviews[0];
        if (selected) await this.selectGameReview(selected, false);
        const active = this.activeFeedbackReview();
        if (active) await this.loadFeedback(active);
      } catch (e) { /* keep reviews empty */ }
      finally { this.reviewsLoading = false; this.reviewsInFlight = false; }
    },

    // Le lien profond porte un `match_id`, alors que l'API identifie une review par son
    // `ts` : on pagine la liste jusqu'à le trouver. Introuvable (review supprimée, lien
    // périmé) : repli silencieux sur la plus récente, un lien mort ne doit pas casser
    // la page pour un visiteur venu du CV.
    async resolvePendingReview() {
      const wanted = this.pendingReviewId;
      if (!wanted) return null;
      for (;;) {
        const hit = this.gameReviews.find(r => this.gameMatchId(r) === wanted);
        if (hit) return hit;
        const before = this.gameReviews.length;
        if (before >= this.gameReviewsCount) return null;
        await this.loadMoreGameReviews();
        if (this.gameReviews.length === before) return null;
      }
    },

    async loadMoreGameReviews() {
      if (this.gameReviewsLoading || this.gameReviews.length >= this.gameReviewsCount) return;
      this.gameReviewsLoading = true;
      try {
        const next = await api(`/api/c/${this.slug}/reviews?kind=game&page=${this.gameReviewsPage + 1}&size=20`);
        this.gameReviews = [...this.gameReviews, ...(next.items || [])];
        this.gameReviewsPage = next.page || this.gameReviewsPage;
      } finally { this.gameReviewsLoading = false; }
    },

    async genCoach() {
      if (!this.slug) return;
      this.job = { type: "coach", status: "running" };
      try {
        const response = await fetch("/api/coach", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            slug: this.slug,
            scope: this.scope,
            outcome: this.outcome,
            target: this.target,
          }),
        });
        if (!response.ok || !response.body) {
          throw new Error(`HTTP ${response.status} sur /api/coach`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let boundary;
          while ((boundary = buffer.indexOf("\n\n")) >= 0) {
            const frame = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);
            const event = /^event: (.+)$/m.exec(frame)?.[1];
            const raw = /^data: (.+)$/m.exec(frame)?.[1];
            if (!event || !raw) continue;
            const data = JSON.parse(raw);
            if (event in SSE_PROGRESS) {
              this.job = { type: "coach", status: "running", progress: SSE_PROGRESS[event] };
            } else if (event === "review") {
              this.job = { type: "coach", status: "done" };
              this.loadReviews();
            } else if (event === "error") {
              this.job = {
                type: "coach",
                status: "error",
                error: coachErrorMessage(data.error),
              };
            }
          }
        }
      } catch (e) {
        this.job = { type: "coach", status: "error", error: coachErrorMessage(e) };
      }
    },

    activeFeedbackReview() {
      return this.coachingView === "games" ? this.selectedGameReview : this.review;
    },

    async loadFeedback(review = this.activeFeedbackReview()) {
      this.fbMap = {}; this.noteDraft = {}; this.feedbackError = null; this.coachOpen = false;
      if (!review) return;
      try {
        const list = await api(`/api/c/${this.slug}/feedback`);
        const mine = list.find(f => f.ts === review.ts);
        const m = {}, notes = {};
        if (mine) for (const it of (mine.items || [])) {
          const key = `${it.kind},${it.index}`;
          m[key] = { useful: it.useful, tag: it.tag, note: it.note };
          notes[key] = it.note || "";
        }
        this.fbMap = m;
        this.noteDraft = notes;
      } catch (e) { /* fbMap stays empty */ }
    },

    fbKey(kind, index) { return `${kind},${index}`; },
    fbState(kind, index) { return this.fbMap[this.fbKey(kind, index)] || null; },

    async setFb(kind, index, useful) {
      if (useful) {
        await this.submitFb(kind, index, { useful: true });
      } else {
        // ouvre le menu tag ; le choix de tag appelle submitFb avec tag
        this.coachOpen = this.fbKey(kind, index);
      }
    },

    async pickTag(kind, index, tag) {
      await this.submitFb(kind, index, { useful: false, tag });
      this.coachOpen = false;
    },

    noteValue(kind, index) {
      const key = this.fbKey(kind, index);
      return key in this.noteDraft ? this.noteDraft[key] : (this.fbState(kind, index)?.note || "");
    },

    async saveNote(kind, index) {
      const state = this.fbState(kind, index);
      if (!state) return; // note rattachée au vote existant (y/n) — pas de vote seul
      const key = this.fbKey(kind, index);
      const note = (this.noteDraft[key] || "").trim() || null;
      await this.submitFb(kind, index, { useful: state.useful, tag: state.tag, note });
    },

    async submitFb(kind, index, { useful, tag = null, note = null }) {
      const review = this.activeFeedbackReview();
      if (!review) return;
      const key = this.fbKey(kind, index);
      const entry = tag ? { useful, tag, note } : { useful, note };
      // Le backend persist_feedback écrase toute la ligne pour ce ts
      // (1 ligne/review = l'ensemble annoté, comme le flow CLI annotate).
      // On envoie donc la map complète à chaque POST pour ne pas perdre
      // les insights déjà notés.
      const newMap = { ...this.fbMap, [key]: entry };
      this.fbBusy = { ...this.fbBusy, [key]: true };
      const responses = {};
      for (const [k, v] of Object.entries(newMap)) {
        responses[k] = v.tag ? { useful: v.useful, tag: v.tag, note: v.note }
                             : { useful: v.useful, note: v.note };
      }
      try {
        await api("/api/feedback", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slug: this.slug, ts: review.ts, responses }),
        });
        this.fbMap = newMap;
        this.loadEval();          // le taux publie inclut ce vote
      } catch (e) {
        this.feedbackError = /\b429\b/.test(String(e))
          ? "Trop de votes en peu de temps. Réessaie dans une heure."
          : "Le vote n’a pas été enregistré. Réessaie dans un instant.";
      }
      finally { this.fbBusy = { ...this.fbBusy, [key]: false }; }
    },

    meta() { return this.review?.payload?.meta || null; },

    async selectGameReview(review, loadFeedback = true) {
      if (!review?.ts) return;
      // Un clic prend le pas sur le lien profond : la cible n'est conservee que pour
      // les selections automatiques (loadFeedback = false), pas apres un choix humain.
      if (loadFeedback) this.pendingReviewId = null;
      this.gameReviewLoading = true; this.gameReviewError = null;
      this.chatMessages = []; this.chatDraft = ""; this.chatError = null;
      try {
        this.selectedGameReview = await api(`/api/c/${this.slug}/reviews/${encodeURIComponent(review.ts)}`);
        if (loadFeedback && this.coachingView === "games") await this.loadFeedback(this.selectedGameReview);
      } catch (e) {
        this.gameReviewError = "Impossible de charger cette analyse. Réessaie dans un instant.";
      } finally { this.gameReviewLoading = false; }
    },

    async sendGameChat() {
      const content = this.chatDraft.trim();
      if (!content || !this.selectedGameReview?.ts || this.chatBusy) return;
      const pending = [...this.chatMessages, { role: "user", content }];
      this.chatMessages = pending; this.chatDraft = ""; this.chatBusy = true; this.chatError = null;
      try {
        const response = await fetch("/api/chat", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            slug: this.slug, review_ts: this.selectedGameReview.ts,
            messages: pending.slice(-12),
          }),
        });
        if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let boundary;
          while ((boundary = buffer.indexOf("\n\n")) >= 0) {
            const frame = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
            const event = /^event: (.+)$/m.exec(frame)?.[1];
            const raw = /^data: (.+)$/m.exec(frame)?.[1];
            if (!event || !raw) continue;
            const data = JSON.parse(raw);
            if (event === "message") {
              this.chatMessages = [...this.chatMessages, {
                role: "assistant", content: data.answer,
                refused_hidden_info: data.refused_hidden_info,
              }];
            } else if (event === "error") throw new Error(data.error);
          }
        }
      } catch (e) {
        this.chatError = coachErrorMessage(e);
      } finally { this.chatBusy = false; }
    },
    gameMeta(review) { return review?.payload?.meta || review?.meta || {}; },
    gameChampion(review) { return this.gameMeta(review).champion || "Partie analysée"; },
    gameOpponent(review) { return this.gameMeta(review).opponent || null; },
    gameResult(review) {
      const win = this.gameMeta(review).win;
      return win === true ? "Victoire" : win === false ? "Défaite" : "Analyse";
    },
    gameDuration(review) {
      const minutes = Number(this.gameMeta(review).duration_min);
      return Number.isFinite(minutes) ? `${Math.round(minutes)} min` : null;
    },
    gameKda(review) {
      const kda = this.gameMeta(review).kda;
      return kda ? fmtKDA(kda.kills, kda.deaths, kda.assists) : null;
    },
    gamePatch(review) { return this.gameMeta(review).patch || null; },
    gameMatchId(review) { return review?.match_id || this.gameMeta(review).match_id || "—"; },
    gameIcon(review) {
      const meta = this.gameMeta(review);
      return meta.patch && meta.champion ? DDRAGON(meta.patch, meta.champion) : "";
    },

    kda(g) { return fmtKDA(g.kills, g.deaths, g.assists); },
    champIcon(g) { return DDRAGON(g.patch, g.champion); },
    iconFallback(e, champ) { e.target.style.display = "none"; e.target.nextElementSibling.style.display = ""; },
    roleLabel(r) {
      return ({ MIDDLE: "Mid", JUNGLE: "Jungle", BOTTOM: "Bot", TOP: "Top", UTILITY: "Support" })[r] || r;
    },
  };
}
function readmePage() {
  return {
    tab: "overview",
    init() { /* page statique, rien à fetcher */ },
    setTab(t) { this.tab = t; },
  };
}
