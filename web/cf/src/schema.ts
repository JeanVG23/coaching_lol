/** Validation Review — miroir strict de src/04_coaching/schema.py. */

export interface Insight {
  point: string;
  evidence: string;
}

export interface Review {
  strengths: Insight[];
  mistakes: Insight[];
  habits: string[];
  next_focus: string;
  confidence: number;
}

export interface GameInsight extends Insight {
  cause: string;
}

export interface GameReview {
  strengths: GameInsight[];
  mistakes: GameInsight[];
  next_focus: string;
  confidence: number;
}

export const NEG_TAGS = [
  "asymetrie",
  "stat-inventee",
  "profondeur-en-faute",
  "trop-vague",
  "non-actionnable",
  "autre",
] as const;
export type TagKind = (typeof NEG_TAGS)[number];

function isInsight(value: unknown): value is Insight {
  return typeof value === "object" && value !== null
    && typeof (value as Insight).point === "string"
    && typeof (value as Insight).evidence === "string";
}

function isStringArray(value: unknown, length: number): value is string[] {
  return Array.isArray(value)
    && value.length === length
    && value.every((item) => typeof item === "string");
}

function isConfidence(value: unknown): value is number {
  return typeof value === "number"
    && Number.isFinite(value)
    && value >= 0
    && value <= 1;
}

function isGameInsight(value: unknown): value is GameInsight {
  return isInsight(value)
    && typeof (value as GameInsight).cause === "string";
}

export function validateReview(raw: unknown): Review | null {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return null;
  const value = raw as Record<string, unknown>;
  if (!Array.isArray(value.strengths)
      || value.strengths.length < 1
      || value.strengths.length > 3) return null;
  if (!Array.isArray(value.mistakes) || value.mistakes.length !== 3) return null;
  if (!value.strengths.every(isInsight) || !value.mistakes.every(isInsight)) return null;
  if (!isStringArray(value.habits, 2)) return null;
  if (typeof value.next_focus !== "string") return null;
  if (!isConfidence(value.confidence)) return null;
  return {
    strengths: value.strengths,
    mistakes: value.mistakes,
    habits: value.habits,
    next_focus: value.next_focus,
    confidence: value.confidence,
  };
}

/** Miroir de GameReview : aucune habitude n'est inférée sur une seule partie. */
export function validateGameReview(raw: unknown): GameReview | null {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return null;
  const value = raw as Record<string, unknown>;
  if (!Array.isArray(value.strengths) || value.strengths.length > 2) return null;
  if (!Array.isArray(value.mistakes)
      || value.mistakes.length < 1
      || value.mistakes.length > 3) return null;
  if (!value.strengths.every(isGameInsight) || !value.mistakes.every(isGameInsight)) return null;
  if (typeof value.next_focus !== "string" || value.next_focus.trim() === "") return null;
  if (!isConfidence(value.confidence)) return null;
  return {
    strengths: value.strengths,
    mistakes: value.mistakes,
    next_focus: value.next_focus,
    confidence: value.confidence,
  };
}

export function reviewJsonSchema(): Record<string, unknown> {
  const insight = {
    type: "object",
    properties: {
      point: { type: "string" },
      evidence: { type: "string" },
    },
    required: ["point", "evidence"],
    additionalProperties: false,
  };
  return {
    type: "object",
    properties: {
      strengths: { type: "array", minItems: 1, maxItems: 3, items: insight },
      mistakes: { type: "array", minItems: 3, maxItems: 3, items: insight },
      habits: {
        type: "array",
        minItems: 2,
        maxItems: 2,
        items: { type: "string" },
      },
      next_focus: { type: "string" },
      confidence: { type: "number", minimum: 0, maximum: 1 },
    },
    required: ["strengths", "mistakes", "habits", "next_focus", "confidence"],
    additionalProperties: false,
  };
}
