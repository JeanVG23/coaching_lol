import { describe, expect, it } from "vitest";
import { NEG_TAGS, reviewJsonSchema, validateGameReview, validateReview } from "../src/schema";

const INSIGHT = (point: string, evidence: string) => ({ point, evidence });

function validReview(): Record<string, unknown> {
  return {
    strengths: [INSIGHT("tu recalls tard", "recall 1450g vs 1100 challenger")],
    mistakes: [INSIGHT("m1", "e1"), INSIGHT("m2", "e2"), INSIGHT("m3", "e3")],
    habits: ["h1", "h2"],
    next_focus: "trader plus tôt",
    confidence: 0.7,
  };
}

describe("validateReview", () => {
  it("accepte une review conforme et ignore les champs extra", () => {
    const raw = { ...validReview(), champ_inconnu: 1 };
    const review = validateReview(raw);
    expect(review).not.toBeNull();
    expect(review?.strengths).toHaveLength(1);
    expect(review?.mistakes).toHaveLength(3);
    expect(review).not.toHaveProperty("champ_inconnu");
  });

  it("force 1-3 forces, exactement 3 erreurs, exactement 2 habitudes", () => {
    expect(validateReview({ ...validReview(), strengths: [] })).toBeNull();
    expect(validateReview({
      ...validReview(), strengths: [1, 2, 3, 4].map(() => INSIGHT("a", "b")),
    })).toBeNull();
    expect(validateReview({ ...validReview(), mistakes: [INSIGHT("m", "e")] })).toBeNull();
    expect(validateReview({ ...validReview(), habits: ["h1"] })).toBeNull();
    expect(validateReview({ ...validReview(), habits: ["h1", 2] })).toBeNull();
  });

  it("point/evidence doivent être des chaînes (vide accepté)", () => {
    expect(validateReview({
      ...validReview(), strengths: [{ point: "", evidence: "" }],
    })).not.toBeNull();
    expect(validateReview({
      ...validReview(), strengths: [{ point: 1, evidence: "x" }],
    })).toBeNull();
    expect(validateReview({ ...validReview(), strengths: [{ point: "x" }] })).toBeNull();
  });

  it("confidence nombre fini dans [0,1] ; next_focus chaîne", () => {
    expect(validateReview({ ...validReview(), confidence: 1.2 })).toBeNull();
    expect(validateReview({ ...validReview(), confidence: Number.NaN })).toBeNull();
    expect(validateReview({ ...validReview(), confidence: "0.7" })).toBeNull();
    expect(validateReview({ ...validReview(), confidence: 0 })).not.toBeNull();
    expect(validateReview({ ...validReview(), confidence: 1 })).not.toBeNull();
    expect(validateReview({ ...validReview(), next_focus: 42 })).toBeNull();
  });

  it("non-objet / null / tableau -> null", () => {
    expect(validateReview(null)).toBeNull();
    expect(validateReview("review")).toBeNull();
    expect(validateReview([])).toBeNull();
  });
});

describe("validateGameReview", () => {
  const validGameReview = () => ({
    strengths: [{ point: "bon timing", evidence: "12:30 : objectif sécurisé", cause: "priorité de voie" }],
    mistakes: [{ point: "recall tardif", evidence: "15:10 : 1 400 or en poche", cause: "wave mal préparée" }],
    next_focus: "Prépare ton recall avant la prochaine wave.",
    confidence: 0.8,
  });

  it("accepte le format d'analyse individuelle sans habitudes", () => {
    const review = validateGameReview(validGameReview());
    expect(review?.strengths).toHaveLength(1);
    expect(review?.mistakes).toHaveLength(1);
  });

  it("exige une cause sur chaque insight", () => {
    expect(validateGameReview({
      ...validGameReview(),
      strengths: [{ point: "bon timing", evidence: "12:30" }],
    })).toBeNull();
  });
});

describe("reviewJsonSchema", () => {
  it("contraint la génération comme Pydantic", () => {
    const schema = reviewJsonSchema() as any;
    expect(schema.type).toBe("object");
    expect(schema.properties.strengths.minItems).toBe(1);
    expect(schema.properties.strengths.maxItems).toBe(3);
    expect(schema.properties.mistakes.minItems).toBe(3);
    expect(schema.properties.mistakes.maxItems).toBe(3);
    expect(schema.properties.habits.minItems).toBe(2);
    expect(schema.properties.habits.maxItems).toBe(2);
    expect(schema.properties.confidence.minimum).toBe(0);
    expect(schema.properties.confidence.maximum).toBe(1);
    expect(schema.required).toEqual([
      "strengths", "mistakes", "habits", "next_focus", "confidence",
    ]);
  });
});

describe("NEG_TAGS", () => {
  it("reflète TagKind de schema.py", () => {
    expect(NEG_TAGS).toEqual([
      "asymetrie", "stat-inventee", "profondeur-en-faute",
      "trop-vague", "non-actionnable", "autre",
    ]);
  });
});
