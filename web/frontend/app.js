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
function accountPage() { return { init() {} }; }
function readmePage()  { return { init() {} }; }