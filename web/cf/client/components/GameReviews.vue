<script setup lang="ts">
import AppIcon from "./AppIcon.vue";
import { computed, nextTick, ref, watch } from "vue";
import { formatDate } from "../account-profile";
import { authToken, openCoachAuth, setStoredAuthToken, withAuthHeaders } from "../auth";
import { categoryLabel, insightDetail, insightHeading } from "../coaching";
import {
  buildMarks, dominantCategory, dominantPhase, markPosition, parseTimestamps,
  type GameMark, type GamePhase,
} from "../game-timeline";
import {
  gameChampion, gameDuration, gameIcon, gameKda, gameMatchId, gameMeta,
  gameOpponent, gamePatch, gameResult, iconFallback, type GameReview,
} from "../game-review";
import FeedbackButtons from "./FeedbackButtons.vue";

interface FeedbackValue { useful: boolean; tag?: string | null; note?: string | null }
interface ChatMessage { role: "user" | "assistant"; content: string }

const props = withDefaults(defineProps<{
  slug: string;
  reviews?: GameReview[];
  total?: number;
  page?: number;
  loading?: boolean;
  authenticated?: boolean;
  targetMatchId?: string | null;
}>(), {
  reviews: () => [], total: 0, page: 1, loading: false,
  authenticated: false, targetMatchId: null,
});

const emit = defineEmits<{
  "reviews-loaded": [payload: { items: GameReview[]; total: number; page: number }];
  "feedback-saved": [];
  "review-select": [matchId: string];
}>();

const items = ref<GameReview[]>([]);
const currentPage = ref(1);
const filterResult = ref<"all" | "win" | "loss">("all");
const filterChampion = ref("all");
const selected = ref<GameReview | null>(null);
const detailLoading = ref(false);
const detailError = ref<string | null>(null);
const loadingMore = ref(false);
const feedback = ref<Record<string, FeedbackValue>>({});
const feedbackBusy = ref<Record<string, boolean>>({});
const openFeedback = ref<string | null>(null);
const openNotes = ref<Record<string, boolean>>({});
const noteDraft = ref<Record<string, string>>({});
const feedbackError = ref<string | null>(null);
const openInsights = ref<Record<string, boolean>>({});
const chatOpen = ref(false);
const chatMessages = ref<ChatMessage[]>([]);
const chatDraft = ref("");
const chatBusy = ref(false);
const chatError = ref<string | null>(null);
let selectionSequence = 0;
let feedbackSequence = 0;
let resolvingTarget: string | null = null;

const filtered = computed(() => items.value.filter((item) => {
  const win = gameMeta(item).win;
  if (filterResult.value === "win" && win !== true) return false;
  if (filterResult.value === "loss" && win !== false) return false;
  return filterChampion.value === "all" || gameChampion(item).toLowerCase() === filterChampion.value.toLowerCase();
}));
const wins = computed(() => items.value.filter(item => gameMeta(item).win === true).length);
const losses = computed(() => items.value.filter(item => gameMeta(item).win === false).length);
const champions = computed(() => [...new Set(items.value.map(gameChampion).filter(item => item !== "Analyzed game"))].sort());

function feedbackKey(kind: string, index: number): string { return `${kind},${index}`; }

/** Les instants cités par UNE partie : la seule lecture que l'onglet ML ne
 * peut pas produire, puisque sa décomposition n'a pas d'axe temporel. */
const marks = computed<GameMark[]>(() => buildMarks(selected.value?.review));
const durationMin = computed<number>(() => Number(gameMeta(selected.value).duration_min) || 0);

const PHASE_LABELS: Record<GamePhase, string> = {
  early: "early game", mid: "mid game", late: "late game",
};

/** La phrase du premier coup d'oeil. Elle LIT la taxonomie fermée du schéma
 * et la répartition des instants ; elle ne prescrit rien et disparaît quand
 * rien ne se détache, plutôt que de servir un dominant arbitraire. */
const verdictLine = computed<string | null>(() => {
  const top = dominantCategory(selected.value?.review?.mistakes);
  const phase = dominantPhase(marks.value, durationMin.value);
  const clauses: string[] = [];
  if (top) clauses.push(`${categoryLabel(top.category)} dominates: ${top.count} of ${top.total} mistakes`);
  if (phase) {
    clauses.push(clauses.length
      ? `mostly in the ${PHASE_LABELS[phase]}`
      : `flagged moments concentrate in the ${PHASE_LABELS[phase]}`);
  }
  if (!clauses.length) return null;
  const sentence = clauses.join(", ");
  return `${sentence.charAt(0).toUpperCase()}${sentence.slice(1)}.`;
});

const stampsOf = (item: { evidence?: unknown }) => parseTimestamps(item?.evidence);

function insightOpen(kind: string, index: number): boolean {
  return openInsights.value[feedbackKey(kind, index)] === true;
}

function toggleInsight(kind: string, index: number): void {
  const key = feedbackKey(kind, index);
  openInsights.value = { ...openInsights.value, [key]: !openInsights.value[key] };
}

/** Un clic sur la frise ouvre l'insight de ce moment et l'amène sous les yeux :
 * c'est ce qui fait de la frise une table des matières, et pas une décoration. */
function focusMark(mark: GameMark): void {
  const key = feedbackKey(mark.kind, mark.index);
  openInsights.value = { ...openInsights.value, [key]: true };
  void nextTick(() => {
    const target = document.querySelector(`[data-insight="${key}"]`);
    target?.scrollIntoView?.({ behavior: "smooth", block: "center" });
  });
}
function feedbackState(kind: string, index: number): FeedbackValue | null { return feedback.value[feedbackKey(kind, index)] || null; }

async function fetchJson(path: string, options: RequestInit = {}): Promise<any> {
  const response = await fetch(path, { ...options, headers: withAuthHeaders(options.headers) });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function loadFeedback(review: GameReview): Promise<void> {
  const sequence = ++feedbackSequence;
  feedback.value = {}; noteDraft.value = {}; openFeedback.value = null; openNotes.value = {}; feedbackError.value = null;
  openInsights.value = {};
  try {
    const list = await fetchJson(`/api/c/${encodeURIComponent(props.slug)}/feedback`);
    const current = list.find((item: any) => item.ts === review.ts);
    const next: Record<string, FeedbackValue> = {};
    const notes: Record<string, string> = {};
    for (const item of current?.items || []) {
      const key = feedbackKey(item.kind, item.index);
      next[key] = { useful: item.useful, tag: item.tag, note: item.note };
      notes[key] = item.note || "";
    }
    if (sequence === feedbackSequence) { feedback.value = next; noteDraft.value = notes; }
  } catch { /* La review reste lisible sans annotations. */ }
}

async function selectReview(review: GameReview): Promise<void> {
  if (!review?.ts) return;
  const sequence = ++selectionSequence;
  detailLoading.value = true; detailError.value = null;
  chatMessages.value = []; chatDraft.value = ""; chatError.value = null; chatOpen.value = false;
  try {
    const detail = await fetchJson(`/api/c/${encodeURIComponent(props.slug)}/reviews/${encodeURIComponent(review.ts)}`);
    if (sequence !== selectionSequence) return;
    selected.value = detail;
    await loadFeedback(detail);
  } catch {
    if (sequence === selectionSequence) detailError.value = "Unable to load this analysis. Try again in a moment.";
  } finally {
    if (sequence === selectionSequence) detailLoading.value = false;
  }
}

async function selectUserReview(review: GameReview): Promise<void> {
  await selectReview(review);
  if (selected.value?.ts === review.ts) emit("review-select", gameMatchId(review));
}

async function loadMore(): Promise<void> {
  if (loadingMore.value || items.value.length >= props.total) return;
  loadingMore.value = true;
  try {
    const next = await fetchJson(`/api/c/${encodeURIComponent(props.slug)}/reviews?kind=game&page=${currentPage.value + 1}&size=20`);
    items.value = [...items.value, ...(next.items || [])];
    currentPage.value = next.page || currentPage.value;
    emit("reviews-loaded", { items: items.value, total: next.total || props.total, page: currentPage.value });
  } finally { loadingMore.value = false; }
}

async function resolveTarget(matchId: string): Promise<void> {
  if (resolvingTarget === matchId) return;
  resolvingTarget = matchId;
  let target = items.value.find(item => gameMatchId(item) === matchId);
  try {
    while (!target && items.value.length < props.total) {
      const before = items.value.length;
      await loadMore();
      if (items.value.length === before) break;
      target = items.value.find(item => gameMatchId(item) === matchId);
    }
    await selectReview(target || filtered.value[0] || items.value[0]);
    await nextTick();
    document.querySelector(".game-reviews")?.scrollIntoView({ behavior: "smooth", block: "start" });
  } finally {
    resolvingTarget = null;
  }
}

function applyFilter(): void {
  if (filtered.value.length && !filtered.value.some(item => item.ts === selected.value?.ts)) void selectReview(filtered.value[0]);
}

async function submitFeedback(kind: string, index: number, useful: boolean, tag: string | null = null, note?: string | null): Promise<void> {
  if (!selected.value) return;
  const key = feedbackKey(kind, index);
  if (!useful && !tag) { openFeedback.value = key; return; }
  const previous = feedback.value[key];
  const entry: FeedbackValue = { useful, ...(tag ? { tag } : {}), ...(note !== undefined ? { note } : previous?.note ? { note: previous.note } : {}) };
  const next = { ...feedback.value, [key]: entry };
  feedbackBusy.value = { ...feedbackBusy.value, [key]: true };
  feedbackError.value = null;
  try {
    await fetchJson("/api/feedback", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slug: props.slug, ts: selected.value.ts, responses: next }),
    });
    feedback.value = next; openFeedback.value = null; emit("feedback-saved");
  } catch (error) {
    feedbackError.value = /429/.test(String(error)) ? "Too many votes in a short period. Try again in an hour." : "Your vote was not saved. Try again in a moment.";
  } finally { feedbackBusy.value = { ...feedbackBusy.value, [key]: false }; }
}

function toggleNote(kind: string, index: number): void {
  const key = feedbackKey(kind, index);
  openNotes.value = { ...openNotes.value, [key]: !openNotes.value[key] };
}

async function saveNote(kind: string, index: number): Promise<void> {
  const key = feedbackKey(kind, index);
  const state = feedback.value[key];
  if (state) await submitFeedback(kind, index, state.useful, state.tag || null, noteDraft.value[key]?.trim() || null);
}

function coachError(error: unknown): string {
  const raw = String((error as Error)?.message || error || "");
  if (/401|authentification|authentication|non autoris|unauthoriz/i.test(raw)) return "Sign-in required: the coach password is needed.";
  if (/429/.test(raw)) return "The model is temporarily rate-limited. Try again in a few minutes.";
  return raw || "The coach could not respond.";
}

async function sendChat(): Promise<void> {
  const content = chatDraft.value.trim();
  if (!content || !selected.value?.ts || chatBusy.value) return;
  if (!authToken) { openCoachAuth(() => void sendChat()); return; }
  const pending: ChatMessage[] = [...chatMessages.value, { role: "user", content }];
  chatMessages.value = pending; chatDraft.value = ""; chatBusy.value = true; chatError.value = null;
  try {
    const response = await fetch("/api/chat", {
      method: "POST", headers: withAuthHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ slug: props.slug, review_ts: selected.value.ts, messages: pending.slice(-12) }),
    });
    if (response.status === 401) {
      setStoredAuthToken(null);
      window.dispatchEvent(new CustomEvent("coach-auth-change", { detail: { authenticated: false } }));
      openCoachAuth(() => void sendChat());
    }
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
        const event = /^event: (.+)$/m.exec(frame)?.[1], raw = /^data: (.+)$/m.exec(frame)?.[1];
        if (!event || !raw) continue;
        const data = JSON.parse(raw);
        if (event === "message") chatMessages.value = [...chatMessages.value, { role: "assistant", content: data.answer }];
        else if (event === "error") throw new Error(data.error);
      }
    }
  } catch (error) { chatError.value = coachError(error); }
  finally { chatBusy.value = false; }
}

watch(() => [props.slug, props.reviews] as const, () => {
  items.value = [...props.reviews]; currentPage.value = props.page;
  if (props.targetMatchId && gameMatchId(selected.value) !== props.targetMatchId) void resolveTarget(props.targetMatchId);
  else if (!selected.value && items.value.length) void selectReview(filtered.value[0] || items.value[0]);
}, { immediate: true, deep: true });
watch(() => props.targetMatchId, value => { if (value && gameMatchId(selected.value) !== value) void resolveTarget(value); });
watch([filterResult, filterChampion], applyFilter);
</script>

<template>
  <section class="game-reviews" aria-labelledby="game-reviews-title">
    <div class="game-reviews-heading"><div><p class="eyebrow">DETAILED ANALYSES</p><h2 id="game-reviews-title">Review a game moment by moment</h2><p>Each analysis breaks down one game: deaths, recalls, objective timing, and the focus for your next match.</p></div></div>
    <div v-if="loading" class="state">Loading game analyses…</div>
    <div v-else-if="!items.length" class="state empty-state"><strong>No game analyses available.</strong><span>Future individual analyses will appear here.</span></div>
    <div v-else class="game-review-layout">
      <div class="game-review-sidebar">
        <div class="game-filter-bar">
          <div class="segmented-choice filter-segmented" role="group" aria-label="Filter by outcome">
            <button type="button" :class="{ selected: filterResult === 'all' }" @click="filterResult = 'all'">All ({{ items.length }})</button>
            <button type="button" :class="{ selected: filterResult === 'win', 'win-choice': filterResult === 'win' }" @click="filterResult = 'win'">Wins ({{ wins }})</button>
            <button type="button" :class="{ selected: filterResult === 'loss', 'loss-choice': filterResult === 'loss' }" @click="filterResult = 'loss'">Losses ({{ losses }})</button>
          </div>
          <select v-if="champions.length > 1" v-model="filterChampion" class="champ-filter-select"><option value="all">All champions</option><option v-for="champion in champions" :key="champion" :value="champion">{{ champion }}</option></select>
        </div>
        <nav class="game-review-list" aria-label="Available game analyses">
          <button v-for="item in filtered" :key="item.ts" type="button" class="game-review-option" :class="[selected?.ts === item.ts ? 'selected' : '', gameResult(item) === 'Victory' ? 'is-win' : 'is-loss']" :aria-current="selected?.ts === item.ts" @click="selectUserReview(item)">
            <img class="game-option-icon" :src="gameIcon(item)" :alt="gameChampion(item)" loading="lazy" @error="iconFallback($event, gameChampion(item))">
            <span class="game-option-copy"><span class="game-option-top"><strong>{{ gameChampion(item) }}</strong><span class="game-mini-badge" :class="gameResult(item) === 'Victory' ? 'win' : 'loss'">{{ gameResult(item) }}</span></span><span class="game-option-sub"><span v-if="gameKda(item)">{{ gameKda(item) }} · </span><span v-if="gameOpponent(item)">vs {{ gameOpponent(item) }} · </span>{{ formatDate(item.ts) }}</span></span><span class="game-option-arrow">→</span>
          </button>
          <div v-if="!filtered.length" class="state empty-state-compact">No games match this filter.</div>
          <button v-if="items.length < total" type="button" class="game-load-more" :disabled="loadingMore" @click="loadMore">{{ loadingMore ? "Loading…" : "Show more games" }}</button>
        </nav>
      </div>
      <div v-if="detailLoading" class="state game-detail-state">Loading analysis…</div>
      <div v-else-if="detailError" class="state err game-detail-state">{{ detailError }}</div>
      <article v-else-if="selected?.review" class="game-review-detail">
        <header class="game-detail-header"><div class="game-detail-title"><img class="game-detail-icon" :src="gameIcon(selected)" :alt="gameChampion(selected)" @error="iconFallback($event, gameChampion(selected))"><div><p class="eyebrow">GAME ANALYSIS</p><h3>{{ gameChampion(selected) }}<span v-if="gameOpponent(selected)"> · vs {{ gameOpponent(selected) }}</span></h3><p class="game-match-id">{{ gameMatchId(selected) }}</p></div></div><span class="game-result" :class="gameResult(selected) === 'Victory' ? 'win' : 'loss'">{{ gameResult(selected) }}</span></header>
        <!-- Une ligne de méta au lieu de quatre tuiles de poids égal : aucune
             d'elles n'est une conclusion, elles ne méritent pas le premier regard. -->
        <p class="game-detail-meta">
          <span v-if="gameKda(selected)"><small>KDA</small> {{ gameKda(selected) }}</span>
          <span v-if="gameDuration(selected)"><small>Duration</small> {{ gameDuration(selected) }}</span>
          <span v-if="gamePatch(selected)"><small>Patch</small> {{ gamePatch(selected) }}</span>
          <span><small>Confidence</small> {{ Math.round((selected.review.confidence || 0) * 100) }}%</span>
        </p>

        <!-- La frise EST la vue par game. L'onglet ML explique un score par des
             poids agrégés ; ici on montre QUAND, sur une partie, en relisant les
             horodatages que le schéma impose déjà dans chaque evidence. -->
        <section v-if="marks.length" class="game-timeline-band coach-timeline" aria-labelledby="game-timeline-title">
          <h4 id="game-timeline-title" class="game-timeline-title">Flagged moments</h4>
          <p v-if="verdictLine" class="game-verdict">{{ verdictLine }}</p>
          <div class="game-timeline">
            <span class="timeline-axis" aria-hidden="true"></span>
            <button
              v-for="mark in marks"
              :key="`${mark.kind}-${mark.index}-${mark.label}`"
              type="button"
              class="timeline-mark"
              :class="mark.kind === 'strength' ? 'is-strength' : 'is-mistake'"
              :style="{ left: `${markPosition(mark.at, durationMin)}%` }"
              :data-label="mark.label"
              :title="`${mark.label} · ${mark.title}`"
              :aria-label="`${mark.label}, ${mark.title}`"
              @click="focusMark(mark)"
            ></button>
          </div>
          <div class="timeline-scale" aria-hidden="true">
            <span>0:00</span>
            <span v-if="gameDuration(selected)">{{ gameDuration(selected) }}</span>
          </div>
          <p class="timeline-legend">
            <span class="legend-dot is-strength" aria-hidden="true"></span>what worked
            <span class="legend-dot is-mistake" aria-hidden="true"></span>what cost you
            <em>click a dot to open its analysis</em>
          </p>
        </section>
        <section class="game-focus-card"><span class="game-focus-label">Focus for your next game</span><p class="game-focus-text">{{ selected.review.next_focus }}</p><FeedbackButtons kind="focus" :index="0" :state="feedbackState('focus', 0)" :busy="feedbackBusy['focus,0']" :open-key="openFeedback" prompt @vote="submitFeedback" @tag="(kind, index, tag) => submitFeedback(kind, index, false, tag)" /></section>
        <!-- Niveau 1 : la catégorie, le titre et les instants. Le titre porte
             déjà la réclamation ; la prescription, la cause et l'evidence
             intégrale (~550 caractères par insight) attendent le clic. -->
        <div class="game-insight-grid shared-insights">
          <section class="game-insight-section strengths">
            <div class="col-head"><AppIcon name="shield-check" class="col-icon" /><h3>What worked well</h3></div>
            <p v-if="!(selected.review.strengths || []).length" class="game-empty-copy">No sufficiently clear strength was identified for this game.</p>
            <article
              v-for="(item, index) in selected.review.strengths || []"
              :key="`strength-${index}`"
              class="game-insight-item card-accent-strength"
              :class="{ 'is-open': insightOpen('strength', index) }"
              :data-insight="feedbackKey('strength', index)"
            >
              <button type="button" class="insight-toggle" :aria-expanded="insightOpen('strength', index)" @click="toggleInsight('strength', index)">
                <span class="insight-chevron" :class="{ 'is-open': insightOpen('strength', index) }" aria-hidden="true">›</span>
                <span class="insight-summary">
                  <span class="insight-head">
                    <span v-if="item.category" class="insight-cat">{{ categoryLabel(item.category) }}</span>
                    <span class="insight-title">{{ insightHeading(item) }}</span>
                  </span>
                  <span v-if="stampsOf(item).length" class="insight-stamps">
                    <span v-for="stamp in stampsOf(item)" :key="stamp.label" class="stamp-chip">{{ stamp.label }}</span>
                  </span>
                </span>
              </button>
              <div v-if="insightOpen('strength', index)" class="insight-expanded">
                <p v-if="insightDetail(item)" class="insight-body">{{ insightDetail(item) }}</p>
                <p v-if="item.cause" class="insight-cause">{{ item.cause }}</p>
                <div class="insight-card-footer">
                  <span class="evidence-chip kind-strength">{{ item.evidence }}</span>
                  <div class="insight-feedback">
                    <FeedbackButtons kind="strength" :index="index" :state="feedbackState('strength', index)" :busy="feedbackBusy[feedbackKey('strength', index)]" :open-key="openFeedback" @vote="submitFeedback" @tag="(kind, itemIndex, tag) => submitFeedback(kind, itemIndex, false, tag)" />
                  </div>
                </div>
              </div>
            </article>
          </section>
          <section class="game-insight-section mistakes">
            <div class="col-head"><AppIcon name="triangle-alert" class="col-icon" /><h3>Mistakes to remember</h3></div>
            <article
              v-for="(item, index) in selected.review.mistakes || []"
              :key="`mistake-${index}`"
              class="game-insight-item card-accent-mistake"
              :class="{ 'is-open': insightOpen('mistake', index) }"
              :data-insight="feedbackKey('mistake', index)"
            >
              <button type="button" class="insight-toggle" :aria-expanded="insightOpen('mistake', index)" @click="toggleInsight('mistake', index)">
                <span class="insight-chevron" :class="{ 'is-open': insightOpen('mistake', index) }" aria-hidden="true">›</span>
                <span class="insight-summary">
                  <span class="insight-head">
                    <span v-if="item.category" class="insight-cat">{{ categoryLabel(item.category) }}</span>
                    <span class="insight-title">{{ insightHeading(item) }}</span>
                  </span>
                  <span v-if="stampsOf(item).length" class="insight-stamps">
                    <span v-for="stamp in stampsOf(item)" :key="stamp.label" class="stamp-chip">{{ stamp.label }}</span>
                  </span>
                </span>
              </button>
              <div v-if="insightOpen('mistake', index)" class="insight-expanded">
                <p v-if="insightDetail(item)" class="insight-body">{{ insightDetail(item) }}</p>
                <p v-if="item.cause" class="insight-cause">{{ item.cause }}</p>
                <div class="insight-card-footer">
                  <span class="evidence-chip kind-mistake">{{ item.evidence }}</span>
                  <div class="insight-feedback">
                    <FeedbackButtons kind="mistake" :index="index" :state="feedbackState('mistake', index)" :busy="feedbackBusy[feedbackKey('mistake', index)]" :open-key="openFeedback" @vote="submitFeedback" @tag="(kind, itemIndex, tag) => submitFeedback(kind, itemIndex, false, tag)" />
                <button v-if="feedbackState('mistake', index)" type="button" class="fb-btn-compact" title="Add a note" @click="toggleNote('mistake', index)">✎</button>
                <div v-if="openNotes[feedbackKey('mistake', index)]" class="fb-note-editor">
                  <textarea v-model="noteDraft[feedbackKey('mistake', index)]" class="fb-note-input" rows="3" maxlength="500" placeholder="Explain what helps or what is missing…"></textarea>
                  <button type="button" class="btn btn-small" :disabled="feedbackBusy[feedbackKey('mistake', index)]" @click="saveNote('mistake', index)">Save note</button>
                </div>
                  </div>
                </div>
              </div>
            </article>
          </section>
        </div>
        <!-- Le chat est un outil de second temps : déplié d'office, son
             formulaire s'ajoute au mur de texte que cette page cherche à
             réduire. Il s'ouvre quand on a une question, pas avant. -->
        <section class="game-chat" aria-labelledby="game-chat-title">
          <button type="button" class="game-chat-toggle" :aria-expanded="chatOpen" @click="chatOpen = !chatOpen">
            <span class="chat-chevron" :class="{ 'is-open': chatOpen }" aria-hidden="true">›</span>
            <span class="game-chat-toggle-copy">
              <span class="game-focus-label">INTERACTIVE COACH</span>
              <span id="game-chat-title" class="game-chat-title">Challenge, explain, or dig deeper.</span>
            </span>
            <small>The coach refuses to use hidden enemy information.</small>
          </button>
          <div v-if="chatOpen" class="game-chat-body">
            <div v-if="chatMessages.length" class="game-chat-log">
              <p v-for="(message, index) in chatMessages" :key="index" :class="message.role">{{ message.content }}</p>
            </div>
            <form class="game-chat-form" @submit.prevent="sendChat">
              <label class="sr-only" for="game-chat-input">Question for the coach</label>
              <textarea id="game-chat-input" v-model="chatDraft" rows="2" maxlength="2000" :placeholder="authenticated ? 'Example: was this choice really a mistake?' : 'Sign in to chat with the AI coach…'"></textarea>
              <button type="submit" :disabled="chatBusy || (authenticated && !chatDraft.trim())"><AppIcon v-if="!chatBusy && !authenticated" name="lock" /> {{ chatBusy ? "The coach is thinking…" : (authenticated ? "Send" : "Sign in") }}</button>
            </form>
            <p v-if="chatError" class="feedback-error">{{ chatError }}</p>
          </div>
        </section>
      </article>
      <p v-if="feedbackError" class="feedback-error">{{ feedbackError }}</p>
    </div>
  </section>
</template>

<style scoped>
.game-reviews {
  padding: 24px;
  background:
    linear-gradient(120deg, var(--win-soft), transparent 36%),
    var(--card-marble-bg);
  border: 1px solid var(--card-marble-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--card-shadow);
}

.game-reviews-heading {
  margin: 0 0 20px;
}

.game-reviews-heading .eyebrow {
  margin-bottom: 5px;
}

.game-reviews-heading h2 {
  color: var(--text);
  font-size: var(--fs-title);
  letter-spacing: -.03em;
}

.game-reviews-heading p:not(.eyebrow) {
  max-width: 680px;
  margin: 5px 0 0;
  color: var(--text-dim);
  font-size: var(--fs-body-sm);
}

.game-review-layout {
  display: grid;
  grid-template-columns: minmax(240px, 310px) minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}

.game-review-sidebar {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}

.game-filter-bar {
  display: flex;
  flex-direction: column;
  gap: 6px;
  background: var(--panel);
  padding: 8px;
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
}

.filter-segmented {
  width: 100%;
}

.filter-segmented button {
  font-size: var(--fs-label);
  padding: 4px 6px;
}

.champ-filter-select {
  width: 100%;
  padding: 6px 10px;
  font-size: var(--fs-small);
  font-family: inherit;
  color: var(--text);
  background: var(--panel-2);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-sm);
  outline: none;
  cursor: pointer;
}

.champ-filter-select:focus {
  border-color: var(--primary);
}

.game-review-list {
  display: flex;
  max-height: 600px;
  flex-direction: column;
  gap: 6px;
  padding: 6px;
  overflow-y: auto;
  background: var(--panel);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
}

.game-review-option {
  display: flex;
  width: 100%;
  align-items: center;
  gap: 10px;
  padding: 9px 10px;
  color: var(--text-dim);
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-family: inherit;
  text-align: left;
  transition: var(--transition-fast);
}

.game-review-option:hover {
  color: var(--text);
  background: var(--surface-alt);
}

.game-review-option.selected {
  color: var(--text);
  background: var(--primary-soft);
  border-color: var(--primary-border);
}

.game-review-option.is-win {
  border-left: 3px solid var(--win);
}

.game-review-option.is-loss {
  border-left: 3px solid var(--loss);
}

.game-option-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 6px;
}

.game-mini-badge {
  font-size: var(--fs-micro);
  font-weight: 800;
  padding: 1px 5px;
  border-radius: var(--radius-xs);
  text-transform: uppercase;
  letter-spacing: .04em;
  flex-shrink: 0;
}

.game-mini-badge.win {
  color: var(--win);
  background: var(--win-soft);
  border: 1px solid var(--win-border);
}

.game-mini-badge.loss {
  color: var(--loss);
  background: var(--loss-soft);
  border: 1px solid var(--loss-border);
}

.game-option-sub {
  overflow: hidden;
  margin-top: 2px;
  color: var(--text-faint);
  font-size: var(--fs-label);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.empty-state-compact {
  padding: 20px 10px;
  font-size: var(--fs-small);
  color: var(--text-faint);
  text-align: center;
}

.game-load-more {
  padding: 9px;
  color: var(--primary);
  background: transparent;
  border: 1px dashed var(--border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font: 600 var(--fs-small)/1.3 inherit;
}

.game-load-more:hover {
  color: var(--text);
  border-color: var(--primary);
}

.game-load-more:disabled {
  opacity: .55;
  cursor: default;
}

.game-option-icon,
.game-detail-icon {
  flex: 0 0 auto;
  object-fit: cover;
  background: var(--panel);
}

.game-option-icon {
  width: 36px;
  height: 36px;
  border-radius: var(--radius-sm);
}

.game-option-copy {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
}

.game-option-copy strong {
  overflow: hidden;
  color: inherit;
  font-size: var(--fs-body-sm);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.game-option-arrow {
  color: var(--primary);
  font-size: 15px;
  flex-shrink: 0;
}

.game-review-detail {
  min-width: 0;
  padding: 22px;
  background: var(--card-marble-bg);
  border: 1px solid var(--card-marble-border);
  border-radius: var(--radius);
  box-shadow: var(--card-shadow);
}

.game-detail-state {
  min-height: 220px;
  margin: 0;
  background: var(--card-marble-bg);
  border: 1px solid var(--card-marble-border);
  border-radius: var(--radius);
  box-shadow: var(--card-shadow);
}

.game-detail-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.game-detail-title {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 12px;
}

.game-detail-icon {
  width: 48px;
  height: 48px;
  border-radius: var(--radius-sm);
  box-shadow: var(--card-shadow);
}

.game-detail-title .eyebrow {
  margin-bottom: 2px;
  font-size: var(--fs-micro);
}

.game-detail-title h3 {
  color: var(--text);
  font-size: var(--fs-title);
  letter-spacing: -.025em;
}

.game-match-id {
  max-width: 250px;
  margin: 2px 0 0;
  overflow: hidden;
  color: var(--text-faint);
  font-family: ui-monospace, Menlo, Consolas, monospace;
  font-size: var(--fs-micro);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.game-result {
  flex: 0 0 auto;
  padding: 4px 10px;
  color: var(--loss);
  background: var(--loss-soft);
  border: 1px solid var(--loss-border);
  border-radius: 999px;
  font-size: var(--fs-label);
  font-weight: 750;
}

.game-result.win {
  color: var(--win);
  background: var(--win-soft);
  border: 1px solid var(--win-border);
}

.game-focus-card {
  padding: 16px;
  margin: 0 0 18px;
  background: linear-gradient(100deg, var(--primary-soft), transparent 85%);
  border: 1px solid var(--primary-border);
  border-radius: var(--radius);
}

.game-focus-label {
  color: var(--primary);
  font-size: var(--fs-micro);
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.game-focus-text {
  margin: 5px 0 0;
  color: var(--text);
  font-size: var(--fs-body);
  font-weight: 650;
  line-height: 1.55;
}

.game-insight-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.game-insight-section {
  padding: 15px;
  background: var(--panel);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
}

.game-insight-section h3 {
  margin-bottom: 11px;
  color: var(--text-dim);
  font-size: var(--fs-label);
  text-transform: uppercase;
  letter-spacing: .05em;
}

.game-insight-section.strengths h3 { color: var(--win); }
.game-insight-section.mistakes h3 { color: var(--loss); }

.game-insight-item {
  padding-left: 10px;
}

.game-insight-item + .game-insight-item {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid var(--border-soft);
}

.game-insight-item p {
  margin: 0 0 9px;
  color: var(--text);
  font-size: var(--fs-body-sm);
  font-weight: 600;
  line-height: 1.5;
}

.game-insight-item details {
  margin: 0 0 9px;
  color: var(--text-dim);
  font-size: var(--fs-small);
  line-height: 1.5;
}

.game-insight-item summary {
  color: var(--text-faint);
  cursor: pointer;
  font-size: var(--fs-label);
  font-weight: 700;
}

.game-insight-item details span {
  display: block;
  padding: 7px 0 0;
}

.game-empty-copy {
  margin: 0;
  color: var(--text-faint);
  font-size: var(--fs-small);
  line-height: 1.55;
}

.insight-head {
  display: flex;
  align-items: baseline;
  gap: 7px;
  flex-wrap: wrap;
  margin-bottom: 5px;
}

.insight-head .insight-title {
  margin-bottom: 0;
}

.insight-cat {
  flex: 0 0 auto;
  padding: 2px 7px;
  color: var(--primary);
  background: var(--primary-soft);
  border: 1px solid var(--primary-border);
  border-radius: 999px;
  font-size: var(--fs-micro);
  font-weight: 750;
  letter-spacing: .05em;
  line-height: 1.35;
  text-transform: uppercase;
  white-space: nowrap;
}

.fb-note-editor {
  position: absolute;
  top: 32px;
  right: 0;
  z-index: 30;
  display: grid;
  gap: 7px;
  width: min(300px, 78vw);
  padding: 9px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-overlay);
}

.fb-note-input {
  width: 100%;
  resize: vertical;
  padding: 8px;
  color: var(--text);
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}

.fb-note-input:focus {
  border-color: var(--primary);
  outline: none;
}

.game-chat {
  margin-top: 18px;
  padding: 16px;
  background: var(--panel);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
}

.game-chat-log {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 14px 0;
}

.game-chat-log p {
  max-width: 86%;
  margin: 0;
  padding: 9px 12px;
  border-radius: var(--radius-sm);
  font-size: var(--fs-small);
  line-height: 1.5;
}

.game-chat-log .user {
  align-self: flex-end;
  color: var(--text);
  background: var(--primary-soft);
  border: 1px solid var(--primary-border);
}

.game-chat-log .assistant {
  align-self: flex-start;
  color: var(--text-dim);
  background: var(--panel-2);
}

.game-chat-form {
  display: flex;
  align-items: flex-end;
  gap: 9px;
  margin-top: 14px;
}

.game-chat-form textarea {
  flex: 1;
  resize: vertical;
  padding: 10px;
  color: var(--text);
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font: var(--fs-small)/1.45 inherit;
}

.game-chat-form button {
  min-height: 39px;
  padding: 0 14px;
  color: var(--primary-text);
  background: var(--primary-gradient);
  border: 0;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font: 700 var(--fs-label)/1 inherit;
}

.game-chat-form button:disabled {
  opacity: .5;
  cursor: default;
}

.feedback-error {
  margin: 12px 0 0;
  color: var(--loss);
  font-size: var(--fs-small);
}

@media (max-width: 640px) {
  .game-chat-form {
    align-items: stretch;
    flex-direction: column;
  }
  .game-reviews {
    padding: 18px;
    border-radius: var(--radius);
  }
  .game-review-layout {
    grid-template-columns: 1fr;
  }
  .game-review-list {
    max-height: 230px;
  }
  .game-review-detail {
    padding: 16px;
  }
  .game-insight-grid {
    grid-template-columns: 1fr;
  }
}

/* Une ligne de méta, pas quatre tuiles : ces quatre chiffres sont du contexte,
   et leur donner le même poids visuel que la conclusion coûtait le premier
   regard de la page. */
.game-detail-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 18px;
  margin: 12px 0 16px;
  color: var(--text-dim);
  font-size: var(--fs-small);
}

.game-detail-meta small {
  margin-right: 4px;
  color: var(--text-faint);
  font-size: var(--fs-micro);
  font-weight: 700;
  letter-spacing: .07em;
  text-transform: uppercase;
}

.game-timeline-band {
  padding: 14px 16px 12px;
  margin: 0 0 18px;
  background: var(--panel);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
}

.game-timeline-title {
  margin: 0;
  color: var(--text-faint);
  font-size: var(--fs-micro);
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.game-verdict {
  margin: 6px 0 0;
  color: var(--text);
  font-size: var(--fs-body-sm);
  font-weight: 650;
  line-height: 1.45;
}

.game-timeline {
  position: relative;
  height: 34px;
  margin: 12px 0 0;
}

.timeline-mark {
  position: absolute;
  top: 50%;
  width: 12px;
  height: 12px;
  padding: 0;
  margin: -6px 0 0 -6px;
  background: var(--loss);
  border: 2px solid var(--panel);
  border-radius: 50%;
  cursor: pointer;
  transition: transform 120ms ease;
}

.timeline-mark.is-strength { background: var(--win); }
.timeline-mark.is-mistake { background: var(--loss); }

.timeline-mark:hover,
.timeline-mark:focus-visible {
  z-index: 2;
  transform: scale(1.5);
}

.timeline-scale {
  display: flex;
  justify-content: space-between;
  color: var(--text-faint);
  font-size: var(--fs-micro);
}

.insight-stamps {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-top: 5px;
}

/* L'horodatage est ce que le joueur peut retrouver dans son replay : c'est la
   seule partie de l'evidence qui mérite d'être lue sans ouvrir la carte. */
.stamp-chip {
  padding: 1px 6px;
  color: var(--text-dim);
  background: var(--panel-2);
  border: 1px solid var(--border-soft);
  border-radius: 999px;
  font-size: var(--fs-micro);
  font-variant-numeric: tabular-nums;
}

.chat-chevron {
  flex: 0 0 auto;
  color: var(--text-faint);
  font-size: 15px;
  line-height: 1.2;
  transition: transform 140ms ease;
}

.chat-chevron.is-open { transform: rotate(90deg); }

.game-chat-toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 12px 0;
  color: inherit;
  background: transparent;
  border: 0;
  cursor: pointer;
  font-family: inherit;
  text-align: left;
}

.game-chat-toggle-copy {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.game-chat-title {
  color: var(--text);
  font-size: var(--fs-body-sm);
  font-weight: 700;
}

.game-chat-toggle small {
  margin-left: auto;
  color: var(--text-faint);
  font-size: var(--fs-micro);
}

.game-chat-body { padding-bottom: 4px; }

@media (max-width: 640px) {
  .game-chat-toggle small { display: none; }
}
</style>
