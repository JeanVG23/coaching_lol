// coaching_lol — frontend SPA (Alpine). Aucune clé/secret ici : tout passe par /api/*.

const NEG_TAGS = ["asymetrie", "stat-inventee", "profondeur-en-faute",
  "trop-vague", "non-actionnable", "autre"];
const DDRAGON = (patch, champ) =>
  `https://ddragon.leagueofgraphs.com/cdn/${patch}.1/img/champion/${champ}.png`;

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} on ${path}`);
  return r.json();
}

function pollJob(jobId, onUpdate, onDone) {
  const id = setInterval(async () => {
    try {
      const j = await api(`/api/jobs/${jobId}`);
      onUpdate(j);
      if (j.status === "done" || j.status === "error") {
        clearInterval(id);
        onDone(j);
      }
    } catch (e) {
      clearInterval(id);
      onDone({ status: "error", error: String(e) });
    }
  }, 1500);
  return id;
}

function fmtKDA(k, d, a) {
  const ka = k + a;
  const ratio = d === 0 ? "Perfect" : (ka / d).toFixed(2);
  return `${k}/${d}/${a} · ${ratio}`;
}

function routeOf(path) {
  if (path === "/" || path === "") return { name: "home" };
  const m = path.match(/^\/c\/([^/]+)$/);
  if (m) return { name: "account", slug: decodeURIComponent(m[1]) };
  if (path === "/readme") return { name: "readme" };
  return { name: "home" };
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
function accountPage(slug) {
  return {
    slug,
    tab: "history",
    games: [], page: 1, size: 20, total: 0,
    gamesLoading: true, gamesError: null,
    fetchN: 20,
    job: null, // {type, status, progress, error}
    // coaching
    scope: "adc", outcome: "loss", target: "challenger",
    reviews: [], review: null, reviewsLoading: true,
    fbMap: {}, fbBusy: {},
    coachOpen: false,
    // shap (F5)
    shap: null, shapLoading: false, shapSort: "abs", chart: null,

    init() { this.loadGames(); },

    async loadGames() {
      this.gamesLoading = true; this.gamesError = null;
      try {
        const d = await api(`/api/c/${this.slug}/games?page=${this.page}&size=${this.size}`);
        this.games = d.items; this.total = d.total;
      } catch (e) { this.gamesError = String(e); }
      finally { this.gamesLoading = false; }
    },

    async prevPage() { if (this.page > 1) { this.page--; this.loadGames(); } },
    async nextPage() {
      if (this.page * this.size < this.total) { this.page++; this.loadGames(); }
    },

    async fetchGames() {
      this.job = { type: "fetch", status: "running", progress: "0/" + this.fetchN };
      try {
        const { job_id } = await api("/api/fetch", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slug: this.slug, n: Number(this.fetchN) }),
        });
        this.startPoll(job_id);
      } catch (e) { this.job = { type: "fetch", status: "error", error: String(e) }; }
    },

    startPoll(jobId) {
      pollJob(jobId,
        j => { this.job = j; },
        j => {
          this.job = j;
          if (j.status === "done" && j.type === "fetch") this.loadGames();
        },
      );
    },

    setTab(t) {
      this.tab = t;
      if (t === "coaching" && this.reviews.length === 0 && this.reviewsLoading) {
        this.loadReviews();
      }
      if (t === "shap" && this.shap === null) { this.loadShap(); }
    },

    async loadShap() {
      this.shapLoading = true;
      try {
        this.shap = await api(`/api/c/${this.slug}/shap`);
        if (this.shap.available) this.$nextTick(() => this.renderChart());
      } catch (e) { this.shap = { available: false, drivers: [] }; }
      finally { this.shapLoading = false; }
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
      this.reviewsLoading = true;
      try {
        const list = await api(`/api/c/${this.slug}/reviews`);
        this.reviews = list;
        this.review = list.length ? list[list.length - 1] : null;
        if (this.review) this.loadFeedback();
      } catch (e) { /* keep reviews empty */ }
      finally { this.reviewsLoading = false; }
    },

    async genCoach() {
      this.job = { type: "coach", status: "running", progress: "coaching" };
      try {
        const { job_id } = await api("/api/coach", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slug: this.slug, scope: this.scope,
            outcome: this.outcome, target: this.target }),
        });
        // étend startPoll pour recharger les reviews au done coach
        pollJob(job_id,
          j => { this.job = j; },
          j => {
            this.job = j;
            if (j.status === "done" && j.type === "coach") this.loadReviews();
          },
        );
      } catch (e) { this.job = { type: "coach", status: "error", error: String(e) }; }
    },

    async loadFeedback() {
      if (!this.review) return;
      try {
        const list = await api(`/api/c/${this.slug}/feedback`);
        const mine = list.find(f => f.ts === this.review.ts);
        const m = {};
        if (mine) for (const it of (mine.items || []))
          m[`${it.kind},${it.index}`] = { useful: it.useful, tag: it.tag, note: it.note };
        this.fbMap = m;
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

    async submitFb(kind, index, { useful, tag = null, note = null }) {
      if (!this.review) return;
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
          body: JSON.stringify({ slug: this.slug, ts: this.review.ts, responses }),
        });
        this.fbMap = newMap;
      } catch (e) { /* 422 si tag manquant — déjà géré par UI (tag imposé) */ }
      finally { this.fbBusy = { ...this.fbBusy, [key]: false }; }
    },

    meta() { return this.review?.payload?.meta || null; },

    kda(g) { return fmtKDA(g.kills.length, g.deaths.length, g.assists.length); },
    champIcon(g) { return DDRAGON(g.patch, g.champion); },
    iconFallback(e, champ) { e.target.style.display = "none"; e.target.nextElementSibling.style.display = ""; },
    roleLabel(r) {
      return ({ MIDDLE: "Mid", JUNGLE: "Jungle", BOTTOM: "Bot", TOP: "Top", UTILITY: "Support" })[r] || r;
    },
  };
}
function readmePage() {
  return { init() { /* page statique, rien à fetcher */ } };
}