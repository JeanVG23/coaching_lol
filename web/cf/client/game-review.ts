import { LATEST_DDRAGON } from "./account-profile";
import { championKey } from "./game-history";

const COMMUNITY_DRAGON_IDS: Record<string, number> = {
  Annie: 1, Olaf: 2, Galio: 3, "Twisted Fate": 4, "Xin Zhao": 5, Urgot: 6,
  LeBlanc: 7, Vladimir: 8, Fiddlesticks: 9, Kayle: 10, "Master Yi": 11,
  Alistar: 12, Ryze: 13, Sion: 14, Sivir: 15, Soraka: 16, Teemo: 17,
  Tristana: 18, Warwick: 19, "Nunu & Willump": 20, "Miss Fortune": 21,
  Ashe: 22, Tryndamere: 23, Jax: 24, Morgana: 25, Zilean: 26, Singed: 27,
  Evelynn: 28, Twitch: 29, Karthus: 30, "Cho'Gath": 31, Amumu: 32,
  Rammus: 33, Anivia: 34, Shaco: 35, "Dr. Mundo": 36, Sona: 37,
  Kassadin: 38, Irelia: 39, Janna: 40, Gangplank: 41, Corki: 42, Karma: 43,
  Taric: 44, Veigar: 45, Trundle: 48, Swain: 50, Caitlyn: 51, Blitzcrank: 53,
  Malphite: 54, Katarina: 55, Nocturne: 56, Maokai: 57, Renekton: 58,
  "Jarvan IV": 59, Elise: 60, Orianna: 61, Wukong: 62, Brand: 63,
  "Lee Sin": 64, Vayne: 67, Rumble: 68, Cassiopeia: 69, Skarner: 72,
  Heimerdinger: 74, Nasus: 75, Nidalee: 76, Udyr: 77, Poppy: 78, Gragas: 79,
  Pantheon: 80, Ezreal: 81, Mordekaiser: 82, Yorick: 83, Akali: 84, Kennen: 85,
  Garen: 86, Leona: 89, Malzahar: 90, Talon: 91, Riven: 92, "Kog'Maw": 96,
  Shen: 98, Lux: 99, Xerath: 101, Shyvana: 102, Ahri: 103, Graves: 104,
  Fizz: 105, Volibear: 106, Rengar: 107, Varus: 110, Nautilus: 111,
  Viktor: 112, Sejuani: 113, Fiora: 114, Ziggs: 115, Lulu: 117, Draven: 119,
  Hecarim: 120, "Kha'Zix": 121, Darius: 122, Jayce: 126, Lissandra: 127,
  Diana: 131, Quinn: 133, Syndra: 134, "Aurelion Sol": 136, Kayn: 141,
  Zoe: 142, Zyra: 143, "Kai'Sa": 145, Seraphine: 147, Gnar: 150, Zac: 154,
  Yasuo: 157, "Vel'Koz": 161, Taliyah: 163, Camille: 164, Akshan: 166,
  "Bel'Veth": 200, Braum: 201, Jhin: 202, Kindred: 203, Zeri: 221, Jinx: 222,
  "Tahm Kench": 223, Briar: 233, Viego: 234, Senna: 235, Lucian: 236,
  Zed: 238, Kled: 240, Ekko: 245, Qiyana: 246, Vi: 254, Aatrox: 266,
  Nami: 267, Azir: 268, Yuumi: 350, Samira: 360, Thresh: 412, Illaoi: 420,
  "Rek'Sai": 421, Ivern: 427, Kalista: 429, Bard: 432, Rakan: 497, Xayah: 498,
  Ornn: 516, Sylas: 517, Neeko: 518, Aphelios: 523, Rell: 526, Pyke: 555,
  Vex: 711, Yone: 777, Ambessa: 799, Mel: 800, Yunara: 804, Locke: 805,
  Sett: 875, Lillia: 876, Gwen: 887, "Renata Glasc": 888, Aurora: 893,
  Nilah: 895, "K'Sante": 897, Smolder: 901, Milio: 902, Zaahen: 904,
  Hwei: 910, Naafiri: 950,
};

export interface GameReview {
  ts: string;
  match_id?: string;
  meta?: Record<string, any>;
  payload?: { meta?: Record<string, any> };
  review?: {
    confidence?: number;
    next_focus?: string;
    strengths?: any[];
    mistakes?: any[];
  };
}

export function gameMeta(review?: GameReview | null): Record<string, any> {
  return review?.payload?.meta || review?.meta || {};
}

export const gameChampion = (review?: GameReview | null): string => gameMeta(review).champion || "Analyzed game";
export const gameOpponent = (review?: GameReview | null): string | null => gameMeta(review).opponent || null;
export const gamePatch = (review?: GameReview | null): string | null => gameMeta(review).patch || null;
export const gameMatchId = (review?: GameReview | null): string => review?.match_id || gameMeta(review).match_id || "—";

export function gameResult(review?: GameReview | null): string {
  const win = gameMeta(review).win;
  return win === true ? "Victory" : win === false ? "Defeat" : "Analysis";
}

export function gameDuration(review?: GameReview | null): string | null {
  const minutes = Number(gameMeta(review).duration_min);
  return Number.isFinite(minutes) ? `${Math.round(minutes)} min` : null;
}

export function gameKda(review?: GameReview | null): string | null {
  const kda = gameMeta(review).kda;
  if (!kda) return null;
  const count = (value: unknown) => Array.isArray(value) ? value.length : Number(value || 0);
  const kills = count(kda.kills), deaths = count(kda.deaths), assists = count(kda.assists);
  return `${kills}/${deaths}/${assists} · ${deaths === 0 ? "Perfect" : ((kills + assists) / deaths).toFixed(2)}`;
}

export function gameIcon(review?: GameReview | null): string {
  const meta = gameMeta(review);
  if (!meta.champion) return "";
  if (["Locke", "Yunara", "Mel", "Ambessa", "Zaahen"].includes(meta.champion)) {
    return `https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/${COMMUNITY_DRAGON_IDS[meta.champion]}.png`;
  }
  const patch = String(meta.patch || "");
  const version = patch ? (patch.split(".").length === 2 ? `${patch}.1` : patch) : LATEST_DDRAGON;
  return `https://ddragon.leagueoflegends.com/cdn/${version}/img/champion/${championKey(meta.champion)}.png`;
}

export function iconFallback(event: Event, champion: string): void {
  const image = event.target as HTMLImageElement;
  const id = COMMUNITY_DRAGON_IDS[champion] || COMMUNITY_DRAGON_IDS[championKey(champion)];
  if (id && !image.dataset.triedCd) {
    image.dataset.triedCd = "1";
    image.src = `https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/${id}.png`;
  } else if (!image.dataset.triedLatest) {
    image.dataset.triedLatest = "1";
    image.src = `https://ddragon.leagueoflegends.com/cdn/${LATEST_DDRAGON}/img/champion/${championKey(champion)}.png`;
  } else {
    image.style.display = "none";
  }
}
