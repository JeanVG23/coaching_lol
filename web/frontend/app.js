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

    setTab(t) { this.tab = t; },

    kda(g) { return fmtKDA(g.kills.length, g.deaths.length, g.assists.length); },
    champIcon(g) { return DDRAGON(g.patch, g.champion); },
    iconFallback(e, champ) { e.target.style.display = "none"; e.target.nextElementSibling.style.display = ""; },
    roleLabel(r) {
      return ({ MIDDLE: "Mid", JUNGLE: "Jungle", BOTTOM: "Bot", TOP: "Top", UTILITY: "Support" })[r] || r;
    },
  };
}
function readmePage()  { return { init() {} }; }