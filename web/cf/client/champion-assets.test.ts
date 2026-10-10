// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import {
  CHAMPION_SLUGS,
  COMMUNITY_DRAGON_IDS,
  LATEST_DDRAGON,
  championIcon,
  championKey,
  fallbackChampionIcon,
} from "./champion-assets";

describe("champion-assets", () => {
  it("championKey maps slugs or strips special characters", () => {
    expect(championKey("Kai'Sa")).toBe("Kaisa");
    expect(championKey("Kha'Zix")).toBe("Khazix");
    expect(championKey("Cho'Gath")).toBe("Chogath");
    expect(championKey("Vel'Koz")).toBe("Velkoz");
    expect(championKey("Wukong")).toBe("MonkeyKing");
    expect(championKey("LeBlanc")).toBe("Leblanc");
    expect(championKey("Nunu & Willump")).toBe("Nunu");
    expect(championKey("Renata Glasc")).toBe("Renata");
    expect(championKey("Bel'Veth")).toBe("Belveth");
    expect(championKey("K'Sante")).toBe("KSante");
    expect(championKey("FiddleSticks")).toBe("Fiddlesticks");
    expect(championKey("Dr. Mundo")).toBe("DrMundo");
    expect(championKey("Jinx")).toBe("Jinx");
  });

  it("championIcon routes recent champions to Community Dragon", () => {
    expect(championIcon("Ambessa")).toBe(
      "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/799.png",
    );
    expect(championIcon("Mel")).toBe(
      "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/800.png",
    );
    expect(championIcon("Yunara")).toBe(
      "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/804.png",
    );
    expect(championIcon("Locke")).toBe(
      "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/805.png",
    );
    expect(championIcon("Zaahen")).toBe(
      "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/904.png",
    );
  });

  it("championIcon returns Data Dragon URL for standard champions with patch formatting", () => {
    expect(championIcon("Jinx", "16.17")).toBe(
      "https://ddragon.leagueoflegends.com/cdn/16.17.1/img/champion/Jinx.png",
    );
    expect(championIcon("Jinx", "16.17.2")).toBe(
      "https://ddragon.leagueoflegends.com/cdn/16.17.2/img/champion/Jinx.png",
    );
    expect(championIcon("Jinx")).toBe(
      `https://ddragon.leagueoflegends.com/cdn/${LATEST_DDRAGON}/img/champion/Jinx.png`,
    );
    expect(championIcon("")).toBe("");
    expect(championIcon(undefined)).toBe("");
  });

  it("fallbackChampionIcon cascades to CD then latest then hides", () => {
    const img = document.createElement("img");
    img.src = "https://broken.example.com/icon.png";
    const evt = { currentTarget: img, target: img } as unknown as Event;

    // 1st failure with a recent champion with CD ID
    fallbackChampionIcon(evt, "Ambessa");
    expect(img.dataset.triedCd).toBe("1");
    expect(img.src).toContain("/champion-icons/799.png");

    // 2nd failure falls back to latest ddragon
    fallbackChampionIcon(evt, "Ambessa");
    expect(img.dataset.triedLatest).toBe("1");
    expect(img.src).toContain(`/${LATEST_DDRAGON}/img/champion/Ambessa.png`);

    // 3rd failure hides image
    fallbackChampionIcon(evt, "Ambessa");
    expect(img.style.display).toBe("none");
  });
});
