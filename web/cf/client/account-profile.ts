export interface CurrentRank {
  tier?: string;
  division?: string;
  league_points?: number;
  wins?: number;
  losses?: number;
  fetched_at?: string;
}

export interface PredictedRank {
  predicted_rank?: string;
  predicted_lp?: number | null;
  proba?: number;
  n_games_used?: number;
}

export interface SummonerProfile {
  icon: number;
  /** Optionnel, et c'est le point : tant qu'aucune collecte n'a vu le joueur,
   * on n'affiche pas de niveau plutôt qu'un niveau inventé qui a l'air vrai. */
  level?: number;
  tag: string;
}


export { LATEST_DDRAGON } from "./champion-assets";
import { LATEST_DDRAGON } from "./champion-assets";

const RANK_EMBLEMS: Record<string, string> = {
  iron: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-static-assets/global/default/images/ranked-mini-crests/iron.svg",
  bronze: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-static-assets/global/default/images/ranked-mini-crests/bronze.svg",
  silver: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-static-assets/global/default/images/ranked-mini-crests/silver.svg",
  gold: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-static-assets/global/default/images/ranked-mini-crests/gold.svg",
  platinum: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-static-assets/global/default/images/ranked-mini-crests/platinum.svg",
  emerald: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-static-assets/global/default/images/ranked-mini-crests/emerald.svg",
  diamond: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-static-assets/global/default/images/ranked-mini-crests/diamond.svg",
  master: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-static-assets/global/default/images/ranked-mini-crests/master.svg",
  grandmaster: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-static-assets/global/default/images/ranked-mini-crests/grandmaster.svg",
  challenger: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-static-assets/global/default/images/ranked-mini-crests/challenger.svg",
  unranked: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-static-assets/global/default/images/ranked-mini-crests/unranked.svg",
};

const SUMMONER_PROFILES: Record<string, SummonerProfile> = {
  spadzze: { icon: 6282, level: 758, tag: "EUW" },
  aceofspadzze: { icon: 6541, level: 85, tag: "EUW" },
  vangy: { icon: 28, level: 134, tag: "EUW" },
  vlintter: { icon: 2074, level: 218, tag: "EUW" },
  "bobby-lupo": { icon: 3457, level: 1004, tag: "EUW" },
  "zaza-warrior35": { icon: 711, level: 411, tag: "EUW" },
  two: { icon: 5373, level: 412, tag: "EUW" },
};

/** Plateforme Riot -> badge affiché. Le badge vient de la RÉGION, jamais du tag
 * du Riot ID : « Bobby Lupo#667 » joue sur EUW, et afficher « 667 » ferait passer
 * un tag choisi par le joueur pour un serveur. */
const REGION_TAGS: Record<string, string> = {
  euw1: "EUW", eun1: "EUNE", na1: "NA", kr: "KR", br1: "BR", jp1: "JP",
  tr1: "TR", la1: "LAN", la2: "LAS", oc1: "OCE", ru: "RU", me1: "ME",
  sg2: "SG", tw2: "TW", vn2: "VN",
};

export function regionTag(region?: string): string {
  const key = String(region || "").toLowerCase().trim();
  if (!key) return "EUW";
  return REGION_TAGS[key] || key.toUpperCase();
}

const FALLBACK_ICONS = [588, 7, 5367, 5373, 4661, 3552, 4023, 6282, 3163, 1638, 4881, 5012];

function hashString(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash * 31 + str.charCodeAt(i)) >>> 0;
  }
  return hash;
}

export function isOwnerAccount(account: { slug: string; group?: string }): boolean {
  if (account.group === "owner" || account.group === "personal") return true;
  if (account.group === "permanent") return false;
  return account.slug === "spadzze" || account.slug === "aceofspadzze";
}

export function summonerProfile(slugOrAccount: unknown): SummonerProfile {
  if (typeof slugOrAccount === "object" && slugOrAccount !== null) {
    const acc = slugOrAccount as
      { slug?: string; icon?: number; level?: number; region?: string };
    const tag = regionTag(acc.region);
    // L'icône suffit à sortir du repli : exiger AUSSI le niveau ferait retomber
    // sur le hachage un compte dont on connaît pourtant la vraie icône.
    if (typeof acc.icon === "number") {
      return typeof acc.level === "number"
        ? { icon: acc.icon, level: acc.level, tag }
        : { icon: acc.icon, tag };
    }
    if (acc.slug) return { ...summonerProfile(acc.slug), tag };
  }
  const normalized = String(slugOrAccount || "").toLowerCase();
  if (SUMMONER_PROFILES[normalized]) return SUMMONER_PROFILES[normalized];
  // Repli : une icône déterministe pour ne pas laisser un avatar cassé, et
  // AUCUN niveau. Le hachage sait fabriquer un nombre, pas le niveau du joueur.
  if (!normalized) return { icon: 6282, tag: "EUW" };
  const h = hashString(normalized);
  return { icon: FALLBACK_ICONS[h % FALLBACK_ICONS.length], tag: "EUW" };
}

export function summonerIcon(slugOrAccountOrIcon: unknown): string {
  if (typeof slugOrAccountOrIcon === "number") {
    return `https://ddragon.leagueoflegends.com/cdn/${LATEST_DDRAGON}/img/profileicon/${slugOrAccountOrIcon}.png`;
  }
  const profile = summonerProfile(slugOrAccountOrIcon);
  return `https://ddragon.leagueoflegends.com/cdn/${LATEST_DDRAGON}/img/profileicon/${profile.icon}.png`;
}

export function summonerLevel(slugOrAccount: unknown): number | undefined {
  return summonerProfile(slugOrAccount).level;
}

export function rankEmblem(tier?: string): string {
  return tier ? RANK_EMBLEMS[tier.toLowerCase().trim()] || "" : "";
}

export function tierAccent(tier?: string): string {
  return tier ? `tier-accent-${tier.toLowerCase().trim()}` : "";
}

export function titleCase(value: unknown): string {
  const text = String(value || "");
  return text ? text.charAt(0).toUpperCase() + text.slice(1).toLowerCase() : text;
}

export function formatPseudo(slugOrAccount: unknown): string {
  if (!slugOrAccount) return "";
  if (typeof slugOrAccount === "object" && slugOrAccount !== null) {
    const acc = slugOrAccount as { slug?: string; riot_id?: string };
    if (acc.slug === "aceofspadzze") return "AceOfSpadzze";
    if (acc.riot_id) {
      const name = acc.riot_id.split("#")[0].trim();
      if (name) {
        return name
          .split(" ")
          .map((w) => (w ? w.charAt(0).toUpperCase() + w.slice(1) : ""))
          .join(" ");
      }
    }
    if (acc.slug) return formatPseudo(acc.slug);
  }
  const str = String(slugOrAccount).trim();
  if (!str) return "";
  if (str.toLowerCase() === "aceofspadzze") return "AceOfSpadzze";
  if (str.includes("-")) {
    return str
      .split("-")
      .map((part) => (part ? part.charAt(0).toUpperCase() + part.slice(1) : ""))
      .join("-");
  }
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export function rankLabel(rank: CurrentRank | null): string {
  if (!rank?.tier) return "Not available";
  return `${titleCase(rank.tier)} ${rank.division || ""} · ${rank.league_points ?? 0} LP`;
}

export function rankWinrate(rank: CurrentRank | null): string | null {
  if (!rank || rank.wins == null || rank.losses == null) return null;
  const wins = Number(rank.wins) || 0;
  const losses = Number(rank.losses) || 0;
  const games = wins + losses;
  return games ? `${wins}W ${losses}L · ${Math.round((wins / games) * 100)}% WR` : null;
}

export const CURATED_RANKS: Record<string, CurrentRank> = {
  spadzze: { tier: "DIAMOND", division: "II", league_points: 95 },
  aceofspadzze: { tier: "MASTER", division: "I", league_points: 2 },
  vangy: { tier: "GOLD", division: "I", league_points: 83 },
  vlintter: { tier: "EMERALD", division: "II", league_points: 38 },
  "bobby-lupo": { tier: "EMERALD", division: "I", league_points: 44 },
  "zaza-warrior35": { tier: "EMERALD", division: "III", league_points: 42 },
};

export const TIER_NAMES_FR: Record<string, string> = {
  iron: "Iron",
  bronze: "Bronze",
  silver: "Silver",
  gold: "Gold",
  platinum: "Platinum",
  emerald: "Emerald",
  diamond: "Diamond",
  master: "Master",
  grandmaster: "Grandmaster",
  challenger: "Challenger",
  unranked: "Unranked",
};

export function accountRank(slugOrAccount: unknown): CurrentRank | null {
  if (typeof slugOrAccount === "object" && slugOrAccount !== null) {
    const acc = slugOrAccount as {
      slug?: string;
      tier?: string;
      division?: string;
      league_points?: number;
      rank?: CurrentRank;
    };
    if (acc.rank?.tier) return acc.rank;
    if (acc.tier) {
      return {
        tier: acc.tier,
        division: acc.division,
        league_points: acc.league_points,
      };
    }
    if (acc.slug && CURATED_RANKS[acc.slug.toLowerCase()]) {
      return CURATED_RANKS[acc.slug.toLowerCase()];
    }
  }
  const slug = String(slugOrAccount || "").toLowerCase().trim();
  return CURATED_RANKS[slug] || null;
}

export function formatAccountRank(rank: CurrentRank | null): string {
  if (!rank?.tier) return "";
  const tierKey = rank.tier.toLowerCase().trim();
  const tierName = TIER_NAMES_FR[tierKey] || titleCase(rank.tier);
  if (tierKey === "master" || tierKey === "grandmaster" || tierKey === "challenger") {
    return rank.league_points != null ? `${tierName} · ${rank.league_points} LP` : tierName;
  }
  const div = rank.division ? ` ${rank.division}` : "";
  const lp = rank.league_points != null ? ` · ${rank.league_points} LP` : "";
  return `${tierName}${div}${lp}`;
}

export function parseDateValue(value?: number | string | null): Date | null {
  if (value == null || value === "" || value === 0) return null;
  const num = typeof value === "number" ? value : Number(value);
  if (!Number.isNaN(num)) {
    if (num <= 0) return null;
    return new Date(num);
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatDate(value?: number | string | null): string {
  if (value == null || value === "") return "—";
  const date = parseDateValue(value);
  if (!date) return String(value).slice(0, 10);
  return new Intl.DateTimeFormat("en-US", {
    day: "numeric", month: "short", year: "numeric",
  }).format(date);
}
