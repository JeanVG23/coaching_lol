export type CombatFilter = "all" | "kill" | "death" | "assist";

export interface TimelineEvent {
  minute: number;
  victim_champ?: string;
  victim_role?: string;
  killer_champ?: string;
  killer_role?: string;
  zone?: string;
  is_solo?: boolean;
  is_ganked_by_jungle?: boolean;
  is_2v2?: boolean;
  gold_state?: string;
}

export interface GameObjective {
  type?: string;
  sub_type?: string;
  minute: number;
  is_ally?: boolean;
  lane?: string;
  tower_type?: string;
}

export interface GameSummary {
  match_id: string;
  champion: string;
  win: boolean;
  patch?: string;
  queue?: number;
  role?: string;
  kills?: number | TimelineEvent[];
  deaths?: number | TimelineEvent[];
  assists?: number | TimelineEvent[];
  lane?: Record<string, any>;
  position?: Record<string, any>;
  comp?: Record<string, string>;
  sides?: Record<string, string>;
  objectives?: GameObjective[];
  plates_diff_early?: number;
  game_ts?: number | string | null;
}

export interface GamesPage {
  items: GameSummary[];
  total: number;
  page?: number;
  size?: number;
}

export interface CombatEvent extends TimelineEvent {
  type: Exclude<CombatFilter, "all">;
  champ: string;
}

export interface TeamPlayer {
  role: string;
  roleName: string;
  champ: string;
  isSelf?: boolean;
}

export {
  CHAMPION_SLUGS,
  COMMUNITY_DRAGON_IDS,
  LATEST_DDRAGON,
  championIcon,
  championKey,
  fallbackChampionIcon,
} from "./champion-assets";

import {
  championIcon,
  championKey,
  fallbackChampionIcon,
  LATEST_DDRAGON,
} from "./champion-assets";

export {
  diffClass,
  formatDiff,
  formatFullDate,
  formatGameDate,
  formatNumber,
  formatPercent,
} from "./formatters";

import {
  countKdaValue as count,
  diffClass,
  formatDiff,
  formatKda as formatKdaValue,
  formatNumber,
  formatPercent,
} from "./formatters";

const POSITION_ICONS: Record<string, string> = {
  top: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-clash/global/default/assets/images/position-selector/positions/icon-position-top-blue.png",
  jungle: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-clash/global/default/assets/images/position-selector/positions/icon-position-jungle-blue.png",
  middle: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-clash/global/default/assets/images/position-selector/positions/icon-position-middle-blue.png",
  bottom: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-clash/global/default/assets/images/position-selector/positions/icon-position-bottom-blue.png",
  utility: "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-clash/global/default/assets/images/position-selector/positions/icon-position-utility-blue.png",
};

export function roleIcon(role?: string): string {
  const key = ({ MIDDLE: "middle", JUNGLE: "jungle", BOTTOM: "bottom", TOP: "top", UTILITY: "utility" } as Record<string, string>)[role || ""];
  return key ? POSITION_ICONS[key] : "";
}

export function roleLabel(role?: string): string {
  return ({ MIDDLE: "Mid", JUNGLE: "Jungle", BOTTOM: "Bot", TOP: "Top", UTILITY: "Support" } as Record<string, string>)[role || ""] || role || "";
}

export function queueLabel(queue?: number): string {
  if (queue === 420) return "Solo/Duo";
  if (queue === 440) return "Flex";
  return "Game";
}

export function formatKda(game: GameSummary): string {
  return formatKdaValue(game) || "0/0/0 · Perfect";
}

export function formatGold(value: unknown): string {
  return value === null || value === undefined ? "—" : `${Math.round(Number(value))} g`;
}

export function zoneLabel(zone?: string): string {
  const labels: Record<string, string> = {
    MID: "Mid lane", BOT: "Bot lane", TOP: "Top lane",
    "JUNGLE/RIVER": "Jungle / River", BASE: "Allied base", ENEMY_BASE: "Enemy base",
  };
  return zone ? labels[zone] || zone : "Unknown area";
}

function teamPlayers(game: GameSummary, side: "self" | "enemy"): TeamPlayer[] {
  const comp = game.comp;
  if (!comp) return [];
  const roles = [
    ["top", "TOP", "TOP"], ["jungle", "JUNGLE", "JGL"], ["mid", "MIDDLE", "MID"],
    ["adc", "BOTTOM", "ADC"], ["support", "UTILITY", "SUP"],
  ];
  return roles.flatMap(([field, role, roleName]) => {
    const champ = comp[`${side}_${field}`];
    return champ ? [{ role, roleName, champ, ...(side === "self" ? { isSelf: game.champion === champ } : {}) }] : [];
  });
}

export const alliedTeam = (game: GameSummary): TeamPlayer[] => teamPlayers(game, "self");
export const enemyTeam = (game: GameSummary): TeamPlayer[] => teamPlayers(game, "enemy");

export function sideSummary(game: GameSummary): string | null {
  const sides = game.sides;
  if (!sides?.ally_start && !sides?.enemy_start) return null;
  const allyBotWeak = sides.ally_weakside === "BOT";
  const enemyBotWeak = sides.enemy_weakside === "BOT";
  if (allyBotWeak && enemyBotWeak) return "Double bot weakside: both junglers path toward top in the early game (0–4 min).";
  if (!allyBotWeak && !enemyBotWeak) return "Double bot strongside: both junglers path toward bot in the early game (0–4 min).";
  if (!allyBotWeak && enemyBotWeak) return "Bot advantage: your jungler paths bot (strongside), while the enemy jungler paths top (weakside).";
  return "Bot warning: your jungler paths top (weakside), while the enemy jungler paths bot (strongside).";
}

// Masques monochromes tirés des icônes de minimap du jeu (CommunityDragon, trait
// seul) : la couleur vient du CSS, jamais du fichier.
export function objectiveIcon(objective: GameObjective): string {
  let slug: string | undefined = ({ HORDE: "grubs", RIFTHERALD: "herald", BARON_NASHOR: "baron", ATAKHAN: "atakhan", TURRET: "turret" } as Record<string, string>)[objective.type || ""];
  if (objective.type === "DRAGON") {
    const sub = objective.sub_type || "";
    const variants: Array<[string, string]> = [
      ["FIRE", "infernal"], ["WATER", "ocean"], ["EARTH", "mountain"],
      ["AIR", "cloud"], ["HEXTECH", "hextech"], ["CHEMTECH", "chemtech"], ["ELDER", "elder"],
    ];
    slug = variants.find(([key]) => sub.includes(key))?.[1];
  }
  return `/images/objectives/${slug || "dragon"}.png`;
}

export function objectiveLabel(objective: GameObjective): string {
  if (objective.type === "DRAGON") {
    const sub = objective.sub_type || "";
    const variants: Array<[string, string]> = [
      ["FIRE", "Infernal Drake"], ["WATER", "Ocean Drake"], ["EARTH", "Mountain Drake"],
      ["AIR", "Cloud Drake"], ["HEXTECH", "Hextech Drake"], ["CHEMTECH", "Chemtech Drake"],
      ["ELDER", "Elder Dragon"],
    ];
    return variants.find(([key]) => sub.includes(key))?.[1] || "Dragon";
  }
  if (objective.type === "HORDE") return "Void Grubs";
  if (objective.type === "RIFTHERALD") return "Rift Herald";
  if (objective.type === "BARON_NASHOR") return "Baron Nashor";
  if (objective.type === "ATAKHAN") return "Atakhan";
  if (objective.type === "TURRET") {
    const lane = ({ BOT_LANE: "Bot", MID_LANE: "Mid", TOP_LANE: "Top" } as Record<string, string>)[objective.lane || ""] || "Turret";
    const tower = ({ OUTER_TURRET: "T1", INNER_TURRET: "T2", BASE_TURRET: "T3", NEXUS_TURRET: "T4" } as Record<string, string>)[objective.tower_type || ""] || "";
    return `${lane} Turret ${tower}`.trim();
  }
  return objective.type || "Objective";
}

export function combatEvents(game: GameSummary, filter: CombatFilter): CombatEvent[] {
  const source = (value: number | TimelineEvent[] | undefined): TimelineEvent[] => Array.isArray(value) ? value : [];
  const events: CombatEvent[] = [
    ...source(game.kills).map((event) => ({ ...event, type: "kill" as const, champ: event.victim_champ || "Enemy" })),
    ...source(game.deaths).map((event) => ({ ...event, type: "death" as const, champ: event.killer_champ || "Enemy" })),
    ...source(game.assists).map((event) => ({ ...event, type: "assist" as const, champ: event.victim_champ || "Enemy" })),
  ];
  const rank: Record<CombatEvent["type"], number> = { kill: 1, assist: 2, death: 3 };
  events.sort((left, right) => left.minute - right.minute || rank[left.type] - rank[right.type]);
  return filter === "all" ? events : events.filter((event) => event.type === filter);
}

export function combatCount(game: GameSummary, filter: CombatFilter): number {
  if (filter === "kill") return count(game.kills);
  if (filter === "death") return count(game.deaths);
  if (filter === "assist") return count(game.assists);
  return count(game.kills) + count(game.deaths) + count(game.assists);
}

