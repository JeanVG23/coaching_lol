<script setup lang="ts">
import AppIcon from "./AppIcon.vue";
import { computed, ref, watch } from "vue";
import { formatDate } from "../account-profile";
import { categoryLabel, insightBody, insightTitle } from "../coaching";
import { recurrenceRows } from "../coaching-recurrence";
import { axisMax, bandCounts, denseSpan, type PublishedMoment } from "../game-timeline";
import { withAuthHeaders } from "../auth";
import FeedbackButtons from "./FeedbackButtons.vue";

interface Insight {
  point?: string;
  cause?: string;
  evidence?: string;
}

interface ReviewContent {
  strengths?: Insight[];
  mistakes?: Insight[];
  habits?: string[];
  next_focus?: string;
  confidence?: number;
}

interface AggregateReview {
  ts: string;
  model?: string;
  scope?: string;
  outcome_focus?: string;
  review: ReviewContent;
}

interface FeedbackValue {
  useful: boolean;
  tag?: string | null;
  note?: string | null;
}

const props = withDefaults(defineProps<{
  slug: string;
  review?: AggregateReview | null;
  reviews?: AggregateReview[];
  loading?: boolean;
  scope?: string;
  scopeName?: string;
  authenticated?: boolean;
  busy?: boolean;
  gameReviews?: Array<Record<string, any>>;
}>(), {
  review: null,
  reviews: () => [],
  loading: false,
  scope: "",
  scopeName: "",
  authenticated: false,
  busy: false,
  gameReviews: () => [],
});

const emit = defineEmits<{
  generate: [];
  "review-select": [review: AggregateReview];
  "game-select": [matchId: string];
  "feedback-saved": [];
}>();

/** Ce qui revient d'une partie à l'autre. Volontairement une grille de
 * présence et non un classement à barres : l'onglet ML publie déjà des barres
 * de contribution, et deux objets visuels identiques pour deux vérités
 * différentes (un poids de modèle ici, un comptage d'occurrences là) se
 * feraient passer l'un pour l'autre. */
const recurrence = computed(() => recurrenceRows(props.gameReviews));

/** Toutes les frises des parties, superposées sur un seul axe. L'agrégat
 * lui-même ne peut pas la produire : ses `evidence` sont statistiques
 * (« gold diff @10 ») et ne portent aucun horodatage. */
const moments = computed<PublishedMoment[]>(() => props.gameReviews.flatMap(
  (game) => Array.isArray(game?.summary?.moments) ? game.summary.moments as PublishedMoment[] : []));

const timelineMax = computed(() => axisMax(moments.value));
const bands = computed(() => bandCounts(moments.value));

/** Une concentration, pas un classement. Sur la fenêtre réelle de Spadzze,
 * laning (55 erreurs) et mid (58) sont à égalité de fait : « le mid game
 * domine » serait faux, « 113 sur 140 entre 5 et 25 minutes » est exact. */
const trend = computed<string | null>(() => {
  const span = denseSpan(bands.value);
  if (!span) return null;
  const window = span.toMin === null
    ? `after ${span.fromMin} minutes`
    : `between ${span.fromMin} and ${span.toMin} minutes`;
  return `${span.count} of ${span.total} flagged mistakes fall ${window}.`;
});

const percent = (value: number) => Math.max(0, Math.min(100, (value / timelineMax.value) * 100));
const momentLeft = (at: number) => `${percent(at)}%`;
const bandLeft = (from: number) => `${percent(from)}%`;
const bandWidth = (from: number, to: number) => `${percent(Math.min(to, timelineMax.value)) - percent(from)}%`;

const openInsights = ref<Record<string, boolean>>({});
const insightOpen = (kind: string, index: number) => openInsights.value[key(kind, index)] === true;

function toggleInsight(kind: string, index: number): void {
  const itemKey = key(kind, index);
  openInsights.value = { ...openInsights.value, [itemKey]: !openInsights.value[itemKey] };
}

const feedback = ref<Record<string, FeedbackValue>>({});
const feedbackBusy = ref<Record<string, boolean>>({});
const openFeedback = ref<string | null>(null);
const feedbackError = ref<string | null>(null);
let feedbackSequence = 0;

function key(kind: string, index: number): string {
  return `${kind},${index}`;
}

async function loadFeedback(): Promise<void> {
  const sequence = ++feedbackSequence;
  feedback.value = {};
  openInsights.value = {};
  openFeedback.value = null;
  feedbackError.value = null;
  if (!props.review?.ts) return;
  try {
    const response = await fetch(`/api/c/${encodeURIComponent(props.slug)}/feedback`, {
      headers: withAuthHeaders(),
    });
    if (!response.ok) return;
    const list = await response.json() as Array<{ ts: string; items?: Array<FeedbackValue & { kind: string; index: number }> }>;
    const current = list.find(item => item.ts === props.review?.ts);
    const next: Record<string, FeedbackValue> = {};
    for (const item of current?.items || []) {
      next[key(item.kind, item.index)] = {
        useful: item.useful,
        tag: item.tag,
        note: item.note,
      };
    }
    if (sequence === feedbackSequence) feedback.value = next;
  } catch {
    // Le coaching reste consultable si les annotations sont indisponibles.
  }
}

async function submitFeedback(kind: string, index: number, useful: boolean, tag: string | null = null): Promise<void> {
  if (!props.review) return;
  const itemKey = key(kind, index);
  if (!useful && !tag) {
    openFeedback.value = itemKey;
    return;
  }
  const next = { ...feedback.value, [itemKey]: { useful, ...(tag ? { tag } : {}) } };
  feedbackBusy.value = { ...feedbackBusy.value, [itemKey]: true };
  feedbackError.value = null;
  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: withAuthHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ slug: props.slug, ts: props.review.ts, responses: next }),
    });
    if (!response.ok) throw new Error(String(response.status));
    feedback.value = next;
    openFeedback.value = null;
    emit("feedback-saved");
  } catch (error) {
    feedbackError.value = /429/.test(String(error))
      ? "Too many votes in a short period. Try again in an hour."
      : "Your vote was not saved. Try again in a moment.";
  } finally {
    feedbackBusy.value = { ...feedbackBusy.value, [itemKey]: false };
  }
}

watch([() => props.slug, () => props.review?.ts], loadFeedback, { immediate: true });
</script>

<template>
  <div v-if="loading" class="state">Loading analyses…</div>
  <div v-else-if="!review" class="state empty-state coach-empty-card">
    <AppIcon name="target" class="coach-empty-icon" />
    <strong>{{ scope ? "No global coaching saved yet" : "Main role not detected yet" }}</strong>
    <p class="coach-empty-copy">{{ scope ? `Generate one synthesis for your main role (${scopeName}), combining wins and losses.` : "Refresh the account first so coaching and ML & SHAP use the same main role." }}</p>
    <button class="btn btn-primary" :disabled="busy || !scope" @click="emit('generate')">
      <AppIcon v-if="!busy && !authenticated" name="lock" />
      {{ busy ? "Generating…" : (authenticated ? "Generate global coaching →" : "Sign in to generate →") }}
    </button>
  </div>

  <div v-else>
    <div v-if="review.review.next_focus" class="global-focus-hero">
      <div class="global-focus-content">
        <span class="global-focus-eyebrow">YOUR #1 IMPROVEMENT PRIORITY</span>
        <h3>{{ review.review.next_focus }}</h3>
        <p class="global-focus-hint">
          The main adjustment to apply in your next game to start climbing.
          <span class="global-focus-confidence">Confidence {{ Math.round((review.review.confidence || 0) * 100) }}%</span>
        </p>
      </div>
      <FeedbackButtons kind="focus" :index="0" :state="feedback['focus,0']" :busy="feedbackBusy['focus,0']" :open-key="openFeedback" prompt @vote="submitFeedback" @tag="(kind, index, tag) => submitFeedback(kind, index, false, tag)" />
    </div>

      <!-- Les frises des parties, superposées. L'agrégat ne peut pas la
           produire lui-même : ses evidence sont statistiques (« gold diff @10 »)
           et ne portent aucun horodatage. Les bandes ont la largeur de leur
           DURÉE, jamais de leur compte : sinon on aurait refait un diagramme
           en barres, et l'onglet ML en publie déjà. -->
      <section v-if="moments.length" class="global-timeline coach-timeline" aria-labelledby="global-timeline-title">
        <div class="recurrence-head">
          <h3 id="global-timeline-title">When the coach flags you</h3>
          <span class="faint">every flagged moment of your reviewed games, on one game clock</span>
        </div>
        <p v-if="trend" class="global-timeline-trend">{{ trend }}</p>
        <div class="global-timeline-strip">
          <span class="timeline-axis" aria-hidden="true"></span>
          <span
            v-for="(moment, index) in moments"
            :key="`${moment.kind}-${moment.at}-${index}`"
            class="global-timeline-mark"
            :class="moment.kind === 'strength' ? 'is-strength' : 'is-mistake'"
            :style="{ left: momentLeft(moment.at) }"
            aria-hidden="true"
          ></span>
        </div>
        <div class="phase-bands">
          <span
            v-for="band in bands"
            :key="band.key"
            class="phase-band"
            :data-band="band.key"
            :style="{ left: bandLeft(band.from), width: bandWidth(band.from, band.to) }"
          >
            <span class="phase-band-label">{{ band.label }}</span>
            <span class="phase-band-count" :class="{ 'is-quiet': !band.mistakes }">{{ band.mistakes }}</span>
          </span>
        </div>
        <p class="timeline-legend">
          <span class="legend-dot is-mistake" aria-hidden="true"></span>a flagged mistake
          <span class="legend-dot is-strength" aria-hidden="true"></span>something that worked
          <em>counts under each band are mistakes</em>
        </p>
      </section>

      <!-- Une colonne par partie, dans l'ordre du temps : toutes les lignes ont
           la même largeur, seul le motif de présence change. C'est une grille,
           pas un classement, et chaque case ramène à la partie qu'elle désigne. -->
      <section v-if="recurrence.length" class="recurrence-grid" aria-labelledby="recurrence-title">
        <div class="recurrence-head">
          <h3 id="recurrence-title">What keeps coming back</h3>
          <span class="faint">across your {{ recurrence[0].total }} reviewed games, oldest first</span>
        </div>
        <div v-for="row in recurrence" :key="row.category" class="recurrence-row">
          <span class="recurrence-label">{{ categoryLabel(row.category) }}</span>
          <span class="recurrence-cells">
            <button
              v-for="(game, index) in row.games"
              :key="`${row.category}-${game.matchId}-${index}`"
              type="button"
              class="recurrence-cell"
              :class="[game.hit ? 'is-hit' : 'is-clear', game.win === true ? 'was-win' : game.win === false ? 'was-loss' : '']"
              :data-match="game.matchId"
              :disabled="!game.hit"
              :title="game.hit ? `Open this game (${game.win === true ? 'win' : 'loss'})` : 'Not flagged in this game'"
              :aria-label="game.hit ? `Open the analysis of game ${index + 1}` : `Game ${index + 1}, not flagged`"
              @click="emit('game-select', game.matchId)"
            ></button>
          </span>
          <span class="recurrence-count">{{ row.count }} of {{ row.total }}</span>
        </div>
      </section>

      <div class="insight-grid shared-insights">
      <div class="card insight-col">
        <div class="col-head"><AppIcon name="shield-check" class="col-icon" /><h3>Strengths</h3></div>
        <div
          v-for="(item, index) in review.review.strengths || []"
          :key="`strength-${index}`"
          class="insight-card card-accent-strength"
          :class="{ 'is-open': insightOpen('strength', index) }"
        >
          <button type="button" class="insight-toggle" :aria-expanded="insightOpen('strength', index)" @click="toggleInsight('strength', index)">
            <span class="insight-chevron" :class="{ 'is-open': insightOpen('strength', index) }" aria-hidden="true">›</span>
            <span class="insight-summary"><span class="insight-title">{{ insightTitle(item.point) }}</span></span>
          </button>
          <div v-if="insightOpen('strength', index)" class="insight-expanded">
            <p v-if="insightBody(item.point)" class="insight-body">{{ insightBody(item.point) }}</p>
            <p v-if="item.cause" class="insight-cause">Why: {{ item.cause }}</p>
            <div class="insight-card-footer">
              <span class="evidence-chip kind-strength">{{ item.evidence }}</span>
              <FeedbackButtons kind="strength" :index="index" :state="feedback[key('strength', index)]" :busy="feedbackBusy[key('strength', index)]" :open-key="openFeedback" @vote="submitFeedback" @tag="(kind, itemIndex, tag) => submitFeedback(kind, itemIndex, false, tag)" />
            </div>
          </div>
        </div>
      </div>
      <div class="card insight-col">
        <div class="col-head"><AppIcon name="triangle-alert" class="col-icon" /><h3>Recurring mistakes</h3></div>
        <div
          v-for="(item, index) in review.review.mistakes || []"
          :key="`mistake-${index}`"
          class="insight-card card-accent-mistake"
          :class="{ 'is-open': insightOpen('mistake', index) }"
        >
          <button type="button" class="insight-toggle" :aria-expanded="insightOpen('mistake', index)" @click="toggleInsight('mistake', index)">
            <span class="insight-chevron" :class="{ 'is-open': insightOpen('mistake', index) }" aria-hidden="true">›</span>
            <span class="insight-summary"><span class="insight-title">{{ insightTitle(item.point) }}</span></span>
          </button>
          <div v-if="insightOpen('mistake', index)" class="insight-expanded">
            <p v-if="insightBody(item.point)" class="insight-body">{{ insightBody(item.point) }}</p>
            <p v-if="item.cause" class="insight-cause">Why: {{ item.cause }}</p>
            <div class="insight-card-footer">
              <span class="evidence-chip kind-mistake">{{ item.evidence }}</span>
              <FeedbackButtons kind="mistake" :index="index" :state="feedback[key('mistake', index)]" :busy="feedbackBusy[key('mistake', index)]" :open-key="openFeedback" @vote="submitFeedback" @tag="(kind, itemIndex, tag) => submitFeedback(kind, itemIndex, false, tag)" />
            </div>
          </div>
        </div>
      </div>
      <div class="card insight-col">
        <div class="col-head"><AppIcon name="repeat" class="col-icon" /><h3>Habits</h3></div>
        <div
          v-for="(habit, index) in review.review.habits || []"
          :key="`habit-${index}`"
          class="insight-card card-accent-habit"
          :class="{ 'is-open': insightOpen('habit', index) }"
        >
          <button type="button" class="insight-toggle" :aria-expanded="insightOpen('habit', index)" @click="toggleInsight('habit', index)">
            <span class="insight-chevron" :class="{ 'is-open': insightOpen('habit', index) }" aria-hidden="true">›</span>
            <span class="insight-summary"><span class="insight-title">{{ insightTitle(habit) }}</span></span>
          </button>
          <div v-if="insightOpen('habit', index)" class="insight-expanded">
            <p v-if="insightBody(habit)" class="insight-body">{{ insightBody(habit) }}</p>
            <div class="insight-card-footer">
              <span></span>
              <FeedbackButtons kind="habit" :index="index" :state="feedback[key('habit', index)]" :busy="feedbackBusy[key('habit', index)]" :open-key="openFeedback" @vote="submitFeedback" @tag="(kind, itemIndex, tag) => submitFeedback(kind, itemIndex, false, tag)" />
            </div>
          </div>
        </div>
      </div>
    </div>

    <details v-if="reviews.length > 1" class="reviews-history">
      <summary class="muted">Previous analyses ({{ reviews.length - 1 }})</summary>
      <button v-for="item in reviews.slice(1)" :key="item.ts" type="button" class="hist-row faint" @click="emit('review-select', item)">
        {{ formatDate(item.ts) }} · {{ item.model }} · global coaching
        <span class="badge badge-scope">View →</span>
      </button>
    </details>
    <p v-if="feedbackError" class="feedback-error">{{ feedbackError }}</p>
  </div>
</template>

<style scoped>
.global-focus-hero {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 20px;
  padding: 18px 22px;
  margin-bottom: 20px;
  background:
    var(--surface-alt);
  border: 1px solid var(--primary-border);
  border-left: 4px solid var(--primary);
  border-radius: var(--radius);
  box-shadow: var(--card-shadow);
}

.global-focus-content {
  flex: 1;
  min-width: 0;
}

.global-focus-eyebrow {
  display: inline-block;
  font-size: var(--fs-label);
  font-weight: 800;
  letter-spacing: .08em;
  color: var(--primary);
  margin-bottom: 6px;
}

.global-focus-hero h3 {
  margin: 0 0 6px;
  color: var(--text);
  font-size: var(--fs-lead);
  font-weight: 700;
  line-height: 1.45;
  letter-spacing: -.02em;
}

.global-focus-hint {
  margin: 0;
  color: var(--text-dim);
  font-size: var(--fs-small);
}

.global-focus-confidence {
  margin-left: 6px;
  padding-left: 8px;
  border-left: 1px solid var(--border-soft);
  color: var(--text-faint);
}

.global-timeline {
  padding: 14px 16px 12px;
  margin-bottom: 16px;
  background: var(--card-marble-bg);
  border: 1px solid var(--card-marble-border);
  border-radius: var(--radius);
  box-shadow: var(--card-shadow);
}

.global-timeline-trend {
  margin: 0 0 4px;
  color: var(--text);
  font-size: var(--fs-body-sm);
  font-weight: 650;
  line-height: 1.45;
}

.global-timeline-strip {
  position: relative;
  height: 44px;
}

/* Les forces au-dessus de l'axe, les erreurs en dessous : sans cette
   séparation, 190 points se superposent en une bande opaque et la densité
   ne se lit plus. Les points ne sont pas cliquables, un instant agrégé ne
   désignant aucune partie en particulier ; c'est la grille de récurrence
   qui porte le lien vers une partie. */
.global-timeline-mark {
  position: absolute;
  width: 8px;
  height: 8px;
  margin-left: -4px;
  border-radius: 50%;
  opacity: .55;
}

.global-timeline-mark.is-strength {
  top: 22%;
  background: var(--win);
}

.global-timeline-mark.is-mistake {
  top: 62%;
  background: var(--loss);
}

.phase-bands {
  position: relative;
  height: 34px;
  margin-top: 2px;
}

/* La largeur d'une bande est celle de sa DURÉE (5, 10, 10 minutes, puis le
   reste), jamais celle de son compte : une largeur proportionnelle au compte
   referait le diagramme en barres de l'onglet ML. */
.phase-band {
  position: absolute;
  top: 0;
  display: flex;
  flex-direction: column;
  gap: 1px;
  align-items: center;
  justify-content: center;
  height: 100%;
  border-left: 1px solid var(--border-soft);
}

.phase-band:last-child { border-right: 1px solid var(--border-soft); }

.phase-band-label {
  color: var(--text-faint);
  font-size: var(--fs-micro);
  font-variant-numeric: tabular-nums;
}

.phase-band-count {
  color: var(--text);
  font-size: var(--fs-body-sm);
  font-weight: 750;
  font-variant-numeric: tabular-nums;
}

.phase-band-count.is-quiet {
  color: var(--text-faint);
  font-weight: 500;
}

.recurrence-grid {
  padding: 14px 16px;
  margin-bottom: 16px;
  background: var(--card-marble-bg);
  border: 1px solid var(--card-marble-border);
  border-radius: var(--radius);
  box-shadow: var(--card-shadow);
}

.recurrence-head {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 4px 10px;
  margin-bottom: 12px;
}

.recurrence-head h3 {
  margin: 0;
  color: var(--text);
  font-size: var(--fs-small);
  font-weight: 750;
  letter-spacing: .05em;
  text-transform: uppercase;
}

.recurrence-head span { font-size: var(--fs-micro); }

.recurrence-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 7px 0;
  border-top: 1px solid var(--border-soft);
}

.recurrence-row:first-of-type { border-top: 0; }

.recurrence-label {
  flex: 0 0 120px;
  color: var(--text-dim);
  font-size: var(--fs-label);
}

.recurrence-cells {
  display: flex;
  flex: 1;
  flex-wrap: wrap;
  gap: 4px;
}

/* Une case pleine = la catégorie est ressortie dans cette partie, et le clic
   y mène. Une case vide reste affichée : c'est elle qui donne au motif sa
   lecture, et elle n'est pas cliquable faute de destination utile. */
.recurrence-cell {
  width: 15px;
  height: 15px;
  padding: 0;
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 4px;
}

.recurrence-cell.is-hit {
  background: var(--loss);
  border-color: var(--loss);
  cursor: pointer;
}

.recurrence-cell.is-hit.was-win {
  background: var(--win);
  border-color: var(--win);
}

.recurrence-cell.is-hit:hover,
.recurrence-cell.is-hit:focus-visible { transform: scale(1.2); }

.recurrence-count {
  flex: 0 0 auto;
  color: var(--text-dim);
  font-size: var(--fs-label);
  font-variant-numeric: tabular-nums;
}

@media (max-width: 640px) {
  .recurrence-label { flex-basis: 100%; }
}

.insight-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
}

.insight-col {
  min-height: 100%;
  display: flex;
  flex-direction: column;
}

.insight-col h3 {
  margin: 0;
  font-size: var(--fs-small);
  font-weight: 750;
  letter-spacing: .05em;
  text-transform: uppercase;
}

.insight-col:nth-child(1) h3 { color: var(--win); }
.insight-col:nth-child(2) h3 { color: var(--loss); }
.insight-col:nth-child(3) h3 { color: var(--info); }

.insight-card {
  padding: 12px 0 12px 10px;
  border-top: 1px solid var(--border-soft);
}

.insight-col .insight-card:first-of-type {
  border-top: none;
  padding-top: 0;
}

.cause-line {
  margin: 0 0 7px;
  color: var(--text-dim);
  font-size: var(--fs-label);
  line-height: 1.45;
}

.confidence-meter {
  margin-top: 14px;
  padding-top: 10px;
  border-top: 1px solid var(--border-soft);
}

.reviews-history {
  margin-top: 18px;
  padding: 6px 16px;
  background: var(--card-marble-bg);
  border: 1px solid var(--card-marble-border);
  border-radius: var(--radius);
  box-shadow: var(--card-shadow);
}

.reviews-history summary {
  cursor: pointer;
  font-size: var(--fs-body-sm);
  padding: 8px 0;
  color: var(--text-dim);
  font-weight: 600;
}

.hist-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: var(--fs-small);
  padding: 8px 0;
  border-top: 1px solid var(--border-soft);
  transition: background 140ms ease;
  width: 100%;
  color: inherit;
  background: transparent;
  border-right: 0;
  border-bottom: 0;
  border-left: 0;
  cursor: pointer;
  font-family: inherit;
  text-align: left;
}

.hist-row:hover {
  color: var(--text);
}

.feedback-error {
  margin: 12px 0 0;
  color: var(--loss);
  font-size: var(--fs-small);
}

.coach-empty-icon {
  margin-bottom: 8px;
  font-size: var(--fs-display);
}

.coach-empty-copy {
  margin: 6px 0 14px;
  color: var(--text-dim);
}

.confidence-label { font-size: var(--fs-label); }

.confidence-value {
  color: var(--text);
  font-size: var(--fs-body-sm);
}

@media (max-width: 640px) {
  .insight-grid {
    grid-template-columns: 1fr;
  }
}
</style>
