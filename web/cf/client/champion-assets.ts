/** Gestion centralisée des identifiants et icônes de champions LoL.
 *
 * Résout les versions Data Dragon et gère le basculement vers Community Dragon
 * pour les champions récents non présents dans le CDN Data Dragon standard.
 */

export const LATEST_DDRAGON = "16.17.1";

export const CHAMPION_SLUGS: Record<string, string> = {
  "Kai'Sa": "Kaisa",
  "Kha'Zix": "Khazix",
  "Cho'Gath": "Chogath",
  "Vel'Koz": "Velkoz",
  Wukong: "MonkeyKing",
  LeBlanc: "Leblanc",
  "Nunu & Willump": "Nunu",
  "Renata Glasc": "Renata",
  "Bel'Veth": "Belveth",
  "K'Sante": "KSante",
  // Match-V5 écrit « FiddleSticks » ; Data Dragon ne connaît que « Fiddlesticks ».
  FiddleSticks: "Fiddlesticks",
};

export const COMMUNITY_DRAGON_IDS: Record<string, number> = {
  Ambessa: 799,
  Mel: 800,
  Yunara: 804,
  Locke: 805,
  Zaahen: 904,
};

/** Nom de fichier Data Dragon d'un champion, tel que Match-V5 ou l'affichage le nomment. */
export function championKey(champion: string): string {
  return CHAMPION_SLUGS[champion] || champion.replace(/['\s.]/g, "");
}

/** URL de l'icône du champion (Data Dragon ou Community Dragon pour champions récents). */
export function championIcon(champion: string | undefined, patch?: string): string {
  if (!champion) return "";
  const communityId = COMMUNITY_DRAGON_IDS[champion];
  if (communityId) {
    return `https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/${communityId}.png`;
  }
  const key = championKey(champion);
  const rawPatch = String(patch || LATEST_DDRAGON);
  const version = rawPatch.split(".").length === 2 ? `${rawPatch}.1` : rawPatch;
  return `https://ddragon.leagueoflegends.com/cdn/${version}/img/champion/${key}.png`;
}

/** Fallback en cascade lors d'une erreur de chargement d'image. */
export function fallbackChampionIcon(event: Event, champion?: string): void {
  const image = (event.currentTarget || event.target) as HTMLImageElement | null;
  if (!image) return;

  if (champion) {
    const cdId = COMMUNITY_DRAGON_IDS[champion] || COMMUNITY_DRAGON_IDS[championKey(champion)];
    if (cdId && !image.dataset.triedCd) {
      image.dataset.triedCd = "1";
      image.src = `https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/${cdId}.png`;
      return;
    }
    if (!image.dataset.triedLatest) {
      image.dataset.triedLatest = "1";
      image.src = `https://ddragon.leagueoflegends.com/cdn/${LATEST_DDRAGON}/img/champion/${championKey(champion)}.png`;
      return;
    }
  }

  image.style.display = "none";
  if (image.nextElementSibling?.classList.contains("champ-fallback")) {
    (image.nextElementSibling as HTMLElement).style.display = "";
  }
}
