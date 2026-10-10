import { describe, expect, it } from "vitest";
import {
  accountRank,
  formatAccountRank,
  formatDate,
  formatPseudo,
  isOwnerAccount,
  parseDateValue,
  regionTag,
  summonerIcon,
  summonerLevel,
  summonerProfile,
  LATEST_DDRAGON,
} from "./account-profile";

describe("account-profile", () => {
  describe("formatPseudo", () => {
    it("capitalise la première lettre des slugs simples", () => {
      expect(formatPseudo("spadzze")).toBe("Spadzze");
      expect(formatPseudo("vangy")).toBe("Vangy");
      expect(formatPseudo("vlintter")).toBe("Vlintter");
    });

    it("respecte la casse camelCase de AceOfSpadzze", () => {
      expect(formatPseudo("aceofspadzze")).toBe("AceOfSpadzze");
      expect(formatPseudo({ slug: "aceofspadzze", riot_id: "AceOfSpadzze#EQ4" })).toBe("AceOfSpadzze");
    });

    it("capitalise les slugs avec tiret", () => {
      expect(formatPseudo("bobby-lupo")).toBe("Bobby-Lupo");
      expect(formatPseudo("zaza-warrior35")).toBe("Zaza-Warrior35");
    });

    it("utilise le riot_id avec majuscules quand fourni dans un compte", () => {
      expect(formatPseudo({ slug: "bobby-lupo", riot_id: "Bobby Lupo#667" })).toBe("Bobby Lupo");
      expect(formatPseudo({ slug: "zaza-warrior35", riot_id: "zaza warrior35#BBL" })).toBe("Zaza Warrior35");
    });
  });

  describe("summonerProfile, summonerIcon & summonerLevel", () => {
    it("retourne les vrais icônes et niveaux Riot API pour les comptes curés", () => {
      expect(summonerProfile("spadzze")).toEqual({ icon: 6282, level: 758, tag: "EUW" });
      expect(summonerProfile("aceofspadzze")).toEqual({ icon: 6541, level: 85, tag: "EUW" });
      expect(summonerProfile("vangy")).toEqual({ icon: 28, level: 134, tag: "EUW" });
      expect(summonerProfile("vlintter")).toEqual({ icon: 2074, level: 218, tag: "EUW" });
      expect(summonerProfile("bobby-lupo")).toEqual({ icon: 3457, level: 1004, tag: "EUW" });
      expect(summonerProfile("zaza-warrior35")).toEqual({ icon: 711, level: 411, tag: "EUW" });
    });

    it("accepte directement un objet compte avec icon et level dynamiques", () => {
      const custom = { slug: "inconnu", icon: 1234, level: 99 };
      expect(summonerProfile(custom)).toEqual({ icon: 1234, level: 99, tag: "EUW" });
      expect(summonerIcon(custom)).toBe(
        `https://ddragon.leagueoflegends.com/cdn/${LATEST_DDRAGON}/img/profileicon/1234.png`
      );
      expect(summonerLevel(custom)).toBe(99);
    });

    it("génère des URL ddragon valides", () => {
      expect(summonerIcon("spadzze")).toBe(
        `https://ddragon.leagueoflegends.com/cdn/${LATEST_DDRAGON}/img/profileicon/6282.png`
      );
      expect(summonerIcon("aceofspadzze")).toBe(
        `https://ddragon.leagueoflegends.com/cdn/${LATEST_DDRAGON}/img/profileicon/6541.png`
      );
      expect(summonerIcon(28)).toBe(
        `https://ddragon.leagueoflegends.com/cdn/${LATEST_DDRAGON}/img/profileicon/28.png`
      );
    });
  });

  describe("données manquantes", () => {
    it("n'invente jamais de niveau pour un slug inconnu", () => {
      // Le hachage sait fabriquer un nombre plausible ; il ne sait pas lire le
      // niveau du joueur. Un niveau faux qui a l'air vrai est pire qu'un vide.
      const inconnu = summonerProfile("joueur-jamais-vu");
      expect(inconnu.level).toBeUndefined();
      expect(summonerLevel("joueur-jamais-vu")).toBeUndefined();
      // L'icône, elle, reste déterministe : un avatar cassé serait visible.
      expect(inconnu.icon).toBe(summonerProfile("joueur-jamais-vu").icon);
      expect(summonerIcon("joueur-jamais-vu")).toContain("/profileicon/");
    });

    it("garde l'icône d'un compte qui n'a pas encore de niveau", () => {
      // Exiger les deux champs ferait retomber sur le hachage un compte dont on
      // connaît pourtant la vraie icône.
      expect(summonerProfile({ slug: "nouveau", icon: 4403 }))
        .toEqual({ icon: 4403, tag: "EUW" });
    });
  });

  describe("regionTag", () => {
    it("traduit la plateforme Riot en badge de serveur", () => {
      expect(regionTag("euw1")).toBe("EUW");
      expect(regionTag("eun1")).toBe("EUNE");
      expect(regionTag("kr")).toBe("KR");
      expect(regionTag("la2")).toBe("LAS");
    });

    it("affiche la plateforme telle quelle si elle est inconnue, EUW si absente", () => {
      expect(regionTag("xx9")).toBe("XX9");
      expect(regionTag(undefined)).toBe("EUW");
    });

    it("badge le serveur du compte, jamais le tag de son Riot ID", () => {
      // « Bobby Lupo#667 » joue sur EUW : afficher « 667 » ferait passer un tag
      // choisi par le joueur pour une plateforme.
      expect(summonerProfile({ slug: "faker", icon: 12, level: 500, region: "kr" }).tag)
        .toBe("KR");
    });
  });

  describe("isOwnerAccount", () => {
    it("identifie les comptes owner basés sur le group ou le slug", () => {
      expect(isOwnerAccount({ slug: "spadzze" })).toBe(true);
      expect(isOwnerAccount({ slug: "aceofspadzze" })).toBe(true);
      expect(isOwnerAccount({ slug: "autre", group: "owner" })).toBe(true);
      expect(isOwnerAccount({ slug: "vangy", group: "permanent" })).toBe(false);
      expect(isOwnerAccount({ slug: "bobby-lupo" })).toBe(false);
    });
  });

  describe("accountRank & formatAccountRank", () => {
    it("fournit le rang des comptes curés", () => {
      expect(accountRank("spadzze")).toEqual({ tier: "DIAMOND", division: "II", league_points: 95 });
      expect(accountRank("aceofspadzze")).toEqual({ tier: "MASTER", division: "I", league_points: 2 });
      expect(accountRank("vangy")).toEqual({ tier: "GOLD", division: "I", league_points: 83 });
    });

    it("formats the rank in English with division and LP", () => {
      expect(formatAccountRank({ tier: "DIAMOND", division: "II", league_points: 95 })).toBe("Diamond II · 95 LP");
      expect(formatAccountRank({ tier: "MASTER", division: "I", league_points: 2 })).toBe("Master · 2 LP");
      expect(formatAccountRank({ tier: "GOLD", division: "I", league_points: 83 })).toBe("Gold I · 83 LP");
      expect(formatAccountRank({ tier: "EMERALD", division: "III", league_points: 42 })).toBe("Emerald III · 42 LP");
    });
  });

  describe("parseDateValue & formatDate", () => {
    it("parse et formate des chaînes ISO, timestamps numériques et gère les vides", () => {
      expect(formatDate(undefined)).toBe("—");
      expect(formatDate(null)).toBe("—");
      expect(formatDate("")).toBe("—");
      expect(formatDate("2026-09-06T12:00:00Z")).toBe("Sep 6, 2026");
      expect(formatDate(1725624000000)).toBe("Sep 6, 2024");
      expect(parseDateValue(null)).toBeNull();
      expect(parseDateValue(0)).toBeNull();
      expect(parseDateValue("invalid")).toBeNull();
      expect(parseDateValue("2026-09-06")?.getUTCFullYear()).toBe(2026);
    });
  });
});
