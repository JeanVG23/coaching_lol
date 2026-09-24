<script setup lang="ts">
import AppIcon from "./AppIcon.vue";
import { computed, onMounted, ref, watch } from "vue";
import { rankEmblem, titleCase, type PredictedRank } from "../account-profile";
import { withAuthHeaders } from "../auth";
import {
  alliedTeam,
  championIcon,
  combatCount,
  combatEvents,
  diffClass,
  enemyTeam,
  fallbackChampionIcon,
  formatDiff,
  formatFullDate,
  formatGameDate,
  formatGold,
  formatKda,
  formatNumber,
  formatPercent,
  objectiveIcon,
  objectiveLabel,
  queueLabel,
  roleIcon,
  roleLabel,
  sideSummary,
  zoneLabel,
  type CombatFilter,
  type GameSummary,
  type GamesPage,
} from "../game-history";

interface ReviewRecord {
  match_id?: string;
  payload?: { meta?: { match_id?: string } };
  meta?: { match_id?: string };
}

interface MatchCoachInfo {
  review_status?: string;
  analyzable?: boolean;
  analysis_unavailable_reason?: string | null;
  pedagogic?: { kind?: string; label?: string; hint?: string };
}

interface CoachingContext {
  matches?: Record<string, MatchCoachInfo>;
}

interface CoachJob {
  type?: string;
  status?: string;
  matchId?: string;
  progress?: string;
}

const props = withDefaults(defineProps<{
  slug: string;
  gameReviews?: ReviewRecord[];
  coachingContext?: CoachingContext | null;
  job?: CoachJob | null;
  predictedRank?: PredictedRank | null;
  authenticated?: boolean;
}>(), {
  gameReviews: () => [],
  coachingContext: null,
  job: null,
  predictedRank: null,
  authenticated: false,
});

const emit = defineEmits<{
  gamesLoaded: [page: GamesPage];
  coachGame: [game: GameSummary];
  regenerateGame: [game: GameSummary];
}>();

const games = ref<GameSummary[]>([]);
const page = ref(1);
const size = 20;
const total = ref(0);
const loading = ref(true);
const error = ref<string | null>(null);
const expandedGameId = ref<string | null>(null);
const combatFilter = ref<CombatFilter>("all");
let requestSequence = 0;

const pageCount = computed(() => Math.max(1, Math.ceil(total.value / size)));
const coachBusy = computed(() => props.job?.status === "running");

async function loadGames(): Promise<void> {
  const sequence = ++requestSequence;
  loading.value = true;
  error.value = null;
  try {
    const response = await fetch(
      `/api/c/${encodeURIComponent(props.slug)}/games?page=${page.value}&size=${size}`,
      { headers: withAuthHeaders() },
    );
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const result = await response.json() as GamesPage;
    if (sequence !== requestSequence) return;
    games.value = Array.isArray(result.items) ? result.items : [];
    total.value = Number(result.total) || 0;
    expandedGameId.value = null;
    emit("gamesLoaded", { ...result, items: games.value, total: total.value, page: page.value, size });
  } catch {
    if (sequence === requestSequence) {
      error.value = "Unable to load games. Try again in a moment.";
    }
  } finally {
    if (sequence === requestSequence) loading.value = false;
  }
}

function previousPage(): void {
  if (page.value <= 1 || loading.value) return;
  page.value -= 1;
  void loadGames();
}

function nextPage(): void {
  if (page.value * size >= total.value || loading.value) return;
  page.value += 1;
  void loadGames();
}

function toggleGame(matchId: string): void {
  expandedGameId.value = expandedGameId.value === matchId ? null : matchId;
}

function reviewMatchId(review: ReviewRecord): string | undefined {
  return review.match_id || review.payload?.meta?.match_id || review.meta?.match_id;
}

function matchCoachInfo(matchId: string): MatchCoachInfo | null {
  return props.coachingContext?.matches?.[matchId] || null;
}

function hasReview(matchId: string): boolean {
  const status = matchCoachInfo(matchId)?.review_status;
  if (status) return status === "ready" || status === "stale";
  return props.gameReviews.some((review) => reviewMatchId(review) === matchId);
}

function pedagogicTarget(game: GameSummary): (NonNullable<MatchCoachInfo["pedagogic"]> & { badge: string }) | null {
  const target = matchCoachInfo(game.match_id)?.pedagogic;
  return target ? { ...target, badge: target.label || "Recommended game" } : null;
}

function isGameCoachRunning(matchId: string): boolean {
  return props.job?.type === "game-coach" && props.job.status === "running" && props.job.matchId === matchId;
}

function coachButtonText(game: GameSummary): string {
  const info = matchCoachInfo(game.match_id);
  if (isGameCoachRunning(game.match_id)) return `Ollama analysis in progress… ${props.job?.progress || ""}`.trim();
  if (hasReview(game.match_id)) return "View LLM coaching analysis →";
  if (info && !info.analyzable) {
    const labels: Record<string, string> = {
      outside_window: "Analysis is available for the latest 50 games",
      raw_missing: "Match data unavailable",
      benchmark_missing: "Coaching reference unavailable",
      not_eligible: "This game cannot be analyzed",
      bundle_missing: "Refresh to prepare the game journal",
    };
    return labels[info.analysis_unavailable_reason || ""] || "Game journal unavailable";
  }
  if (!props.authenticated) return "Sign in to analyze";
  return pedagogicTarget(game) ? "Analyze this recommended game →" : "Analyze this game with AI →";
}

watch(() => props.slug, () => {
  page.value = 1;
  void loadGames();
});

onMounted(loadGames);
</script>

<template>
  <div>
    <div class="section-heading">
      <div>
        <h2>Recent games</h2>
      </div>
    </div>

    <div v-if="loading" class="state">Loading…</div>
    <div v-else-if="error" class="state err">{{ error }}</div>
    <div v-else-if="games.length === 0" class="state empty-state">
      <strong>No games available.</strong>
      <span>Refresh the account data, then come back here.</span>
    </div>

    <div v-else>
      <div
        v-for="game in games"
        :key="game.match_id"
        class="game-card"
        :class="[game.win ? 'w' : 'l', expandedGameId === game.match_id ? 'is-expanded' : '']"
      >
        <div
          class="game-row"
          role="button"
          tabindex="0"
          :aria-expanded="expandedGameId === game.match_id"
          :title="expandedGameId === game.match_id ? 'Click to collapse details' : 'Click to view all game details and statistics'"
          @click="toggleGame(game.match_id)"
          @keydown.enter.prevent="toggleGame(game.match_id)"
          @keydown.space.prevent="toggleGame(game.match_id)"
        >
          <div class="gr-champ-wrap">
            <img class="champ-icon" :src="championIcon(game.champion, game.patch)" :alt="game.champion" loading="lazy" @error="fallbackChampionIcon($event, game.champion)">
            <span class="champ-fallback">{{ game.champion }}</span>
          </div>
          <div class="gr-main">
            <div class="gr-top-row">
              <span class="outcome-pill" :class="game.win ? 'pill-win' : 'pill-loss'">{{ game.win ? "VICTORY" : "DEFEAT" }}</span>
              <span class="gr-champ">{{ game.champion }}</span>
              <div v-if="game.role" class="gr-role-tag">
                <img v-if="roleIcon(game.role)" class="role-icon" :src="roleIcon(game.role)" alt="" loading="lazy">
                <span>{{ roleLabel(game.role) }}</span>
              </div>
              <div v-if="game.lane?.opponent" class="gr-matchup" title="Direct lane opponent">
                <span class="gr-vs">VS</span>
                <img class="gr-opp-icon" :src="championIcon(game.lane.opponent, game.patch)" :alt="game.lane.opponent" loading="lazy" @error="fallbackChampionIcon($event, game.lane?.opponent)">
                <span class="gr-opp-name">{{ game.lane.opponent }}</span>
              </div>
            </div>
            <div class="gr-sub-row faint">
              <span class="gr-queue-badge">{{ queueLabel(game.queue) }}</span>
              <template v-if="game.game_ts">
                <span class="gr-sub-sep" aria-hidden="true">·</span>
                <span class="gr-date" :title="formatFullDate(game.game_ts)">{{ formatGameDate(game.game_ts) }}</span>
              </template>
              <template v-if="game.lane?.gd14 != null">
                <span class="gr-sub-sep" aria-hidden="true">·</span>
                <span class="gr-diff" :class="game.lane.gd14 >= 0 ? 'diff-pos' : 'diff-neg'">
                  {{ game.lane.gd14 >= 0 ? "+" : "" }}{{ game.lane.gd14 }}g at 14m
                </span>
              </template>
              <span v-if="hasReview(game.match_id)" class="gr-badge-coach">
                {{ matchCoachInfo(game.match_id)?.review_status === "stale" ? "✦ LLM coaching needs refresh" : "✦ LLM coaching ready" }}
              </span>
              <span
                v-else-if="pedagogicTarget(game)"
                class="gr-badge-pedagogic"
                :class="`badge-${pedagogicTarget(game)?.kind}`"
                :title="pedagogicTarget(game)?.hint"
              >{{ pedagogicTarget(game)?.badge }}</span>
            </div>
          </div>
          <div class="gr-kda num"><span class="gr-label">KDA</span><span>{{ formatKda(game) }}</span></div>
          <div class="gr-patch faint"><span class="gr-label">Patch</span><span>{{ game.patch }}</span></div>
          <div class="gr-expand-action">
            <span class="gr-expand-label">{{ expandedGameId === game.match_id ? "Close" : "Details" }}</span>
            <span class="gr-expand-chevron" :class="{ open: expandedGameId === game.match_id }">▾</span>
          </div>
        </div>

        <div v-show="expandedGameId === game.match_id" class="game-details-panel">
          <div class="gd-action-bar">
            <div class="gd-meta-info">
              <span class="gd-meta-tag mono">{{ game.match_id }}</span>
              <span class="gd-meta-tag">{{ queueLabel(game.queue) }}</span>
              <span v-if="game.game_ts" class="gd-meta-tag gd-meta-date" :title="formatFullDate(game.game_ts)"><AppIcon name="calendar" /> {{ formatGameDate(game.game_ts) }}</span>
              <span v-if="hasReview(game.match_id)" class="gd-coach-status ready">
                <span class="pulse-dot"></span>
                <span>{{ matchCoachInfo(game.match_id)?.review_status === "stale" ? "Existing LLM analysis (older version)" : "LLM analysis available for this match" }}</span>
              </span>
              <div v-else class="gd-coach-unreviewed-group">
                <span class="gd-coach-status empty">◌ No individual analysis</span>
                <span
                  v-if="pedagogicTarget(game)"
                  class="pedagogic-target-pill"
                  :class="`pill-${pedagogicTarget(game)?.kind}`"
                  :title="pedagogicTarget(game)?.hint"
                >{{ pedagogicTarget(game)?.badge }}</span>
              </div>
            </div>
            <button
              type="button"
              class="btn-coach-shortcut"
              :disabled="coachBusy || Boolean(!hasReview(game.match_id) && matchCoachInfo(game.match_id) && !matchCoachInfo(game.match_id)?.analyzable)"
              @click.stop="emit('coachGame', game)"
            >
              <AppIcon class="coach-icon" :name="hasReview(game.match_id) || authenticated ? 'target' : 'lock'" />
              <span class="coach-txt">{{ coachButtonText(game) }}</span>
            </button>
            <button
              v-if="matchCoachInfo(game.match_id)?.review_status === 'stale'"
              type="button"
              class="btn btn-small"
              :disabled="coachBusy"
              title="This action makes a new Ollama request"
              @click.stop="emit('regenerateGame', game)"
            ><AppIcon v-if="!authenticated" name="lock" /> Regenerate</button>
          </div>

          <div class="gd-grid">
            <div class="gd-card">
              <div class="gd-card-header"><AppIcon name="swords" class="gd-icon" /><h4>Matchup &amp; Laning Phase</h4></div>
              <div class="gd-card-body">
                <div v-if="game.lane?.opponent" class="gd-matchup-box">
                  <div class="gd-matchup-side self">
                    <img class="gd-mini-icon" :src="championIcon(game.champion, game.patch)" :alt="game.champion" @error="fallbackChampionIcon($event, game.champion)">
                    <div><strong>{{ game.champion }}</strong><span class="faint">You</span></div>
                  </div>
                  <span class="gd-matchup-vs">VS</span>
                  <div class="gd-matchup-side opp">
                    <img class="gd-mini-icon" :src="championIcon(game.lane.opponent, game.patch)" :alt="game.lane.opponent" @error="fallbackChampionIcon($event, game.lane?.opponent)">
                    <div><strong>{{ game.lane.opponent }}</strong><span class="faint">Direct opponent</span></div>
                  </div>
                </div>
                <div class="gd-stats-list">
                  <div class="gd-stat-row">
                    <span class="gd-stat-label">Gold difference (GD)</span>
                    <div class="gd-stat-group">
                      <span class="gd-diff-pill" :class="diffClass(game.lane?.gd10)" title="Gold Diff @ 10 min">10m: <strong>{{ formatDiff(game.lane?.gd10, "g") }}</strong></span>
                      <span class="gd-diff-pill" :class="diffClass(game.lane?.gd14)" title="Gold Diff @ 14 min">14m: <strong>{{ formatDiff(game.lane?.gd14, "g") }}</strong></span>
                      <span v-if="game.lane?.gd20 != null" class="gd-diff-pill" :class="diffClass(game.lane?.gd20)" title="Gold Diff @ 20 min">20m: <strong>{{ formatDiff(game.lane?.gd20, "g") }}</strong></span>
                    </div>
                  </div>
                  <div class="gd-stat-row">
                    <span class="gd-stat-label">CS difference (CSD)</span>
                    <div class="gd-stat-group">
                      <span class="gd-diff-pill" :class="diffClass(game.lane?.csd10)" title="CS Diff @ 10 min">10m: <strong>{{ formatDiff(game.lane?.csd10, " cs") }}</strong></span>
                      <span class="gd-diff-pill" :class="diffClass(game.lane?.csd14)" title="CS Diff @ 14 min">14m: <strong>{{ formatDiff(game.lane?.csd14, " cs") }}</strong></span>
                    </div>
                  </div>
                  <div class="gd-stat-row">
                    <span class="gd-stat-label">Experience difference (XPD)</span>
                    <div class="gd-stat-group"><span class="gd-diff-pill" :class="diffClass(game.lane?.xpd10)" title="XP Diff @ 10 min">10m: <strong>{{ formatDiff(game.lane?.xpd10, " xp") }}</strong></span></div>
                  </div>
                  <div class="gd-stat-row"><span class="gd-stat-label">Pace at 14m</span><span class="num text-highlight">{{ formatNumber(game.lane?.csm14) }} cs/min · {{ formatNumber(game.lane?.gpm14, 0) }} g/min</span></div>
                  <div v-if="game.plates_diff_early != null" class="gd-stat-row"><span class="gd-stat-label">Early plate diff.</span><span class="gd-diff-pill" :class="diffClass(game.plates_diff_early)"><strong>{{ formatDiff(game.plates_diff_early) }}</strong> plates</span></div>
                </div>
              </div>
            </div>

            <div class="gd-card">
              <div class="gd-card-header"><AppIcon name="eye" class="gd-icon" /><h4>Vision &amp; Map Control</h4></div>
              <div class="gd-card-body"><div class="gd-stats-list">
                <div class="gd-stat-row"><span class="gd-stat-label">Wards placed</span><span class="num"><strong>{{ game.position?.wards_placed ?? "—" }}</strong><span v-if="game.position?.wards_placed_early != null" class="faint"> ({{ game.position.wards_placed_early }} early)</span></span></div>
                <div class="gd-stat-row"><span class="gd-stat-label">Control wards</span><span class="num"><strong class="text-pink">{{ game.position?.control_wards_placed ?? 0 }}</strong></span></div>
                <div class="gd-stat-row"><span class="gd-stat-label">Wards cleared</span><span class="num"><strong>{{ game.position?.wards_killed ?? 0 }}</strong></span></div>
                <div class="gd-stat-row"><span class="gd-stat-label">Time overextended</span><span class="num" :class="(game.position?.frac_overextended || 0) > 0.15 ? 'text-danger' : 'text-good'"><strong>{{ formatPercent(game.position?.frac_overextended) }}</strong></span></div>
                <div class="gd-stat-row"><span class="gd-stat-label">Deaths in fog of war</span><span class="num" :class="(game.position?.frac_deaths_in_fog || 0) > 0.5 ? 'text-warning' : ''"><strong>{{ formatPercent(game.position?.frac_deaths_in_fog) }}</strong></span></div>
                <div v-if="game.position?.gold_dead_time != null" class="gd-stat-row"><span class="gd-stat-label">Gold lost while dead</span><span class="num text-danger"><strong>{{ formatGold(game.position.gold_dead_time) }}</strong></span></div>
              </div></div>
            </div>

            <div class="gd-card gd-card-full">
              <div class="gd-card-header"><AppIcon name="users" class="gd-icon" /><h4>Team Compositions &amp; Early Game Plan</h4></div>
              <div class="gd-card-body">
                <div v-if="sideSummary(game)" class="gd-sides-banner">
                  <div class="gd-sides-row">
                    <div class="gd-side-item ally"><span class="gd-side-label">Allies:</span><span class="gd-side-val"><strong>{{ game.sides?.ally_jungler || "Jungle" }}</strong> start <span class="mono">{{ game.sides?.ally_start }}</span> → <span class="badge-side" :class="game.sides?.ally_strongside === 'BOT' ? 'badge-strongside' : 'badge-weakside'">Bot {{ game.sides?.ally_strongside === "BOT" ? "STRONGSIDE" : "WEAKSIDE" }}</span><span class="faint">· Top {{ game.sides?.ally_strongside === "TOP" ? "Strongside" : "Weakside" }}</span></span></div>
                    <div class="gd-side-item enemy"><span class="gd-side-label">Enemies:</span><span class="gd-side-val"><strong>{{ game.sides?.enemy_jungler || "Jungle" }}</strong> start <span class="mono">{{ game.sides?.enemy_start }}</span> → <span class="badge-side" :class="game.sides?.enemy_strongside === 'BOT' ? 'badge-strongside' : 'badge-weakside'">Bot {{ game.sides?.enemy_strongside === "BOT" ? "STRONGSIDE" : "WEAKSIDE" }}</span><span class="faint">· Top {{ game.sides?.enemy_strongside === "TOP" ? "Strongside" : "Weakside" }}</span></span></div>
                  </div>
                  <div class="gd-side-summary">{{ sideSummary(game) }}</div>
                </div>
                <div class="gd-teams-grid">
                  <div class="gd-team-col ally-team">
                    <div class="gd-team-title">Allies (Blue Team)</div>
                    <div class="gd-team-players">
                      <div v-for="player in alliedTeam(game)" :key="`${player.role}-${player.champ}`" class="gd-player-row" :class="{ 'is-self': player.isSelf }">
                        <img class="gd-mini-icon" :src="championIcon(player.champ, game.patch)" :alt="player.champ" loading="lazy" @error="fallbackChampionIcon($event, player.champ)"><span class="gd-role-badge">{{ player.roleName }}</span><span class="gd-pname">{{ player.champ }}</span><span v-if="player.isSelf" class="gd-self-badge">You</span>
                      </div>
                      <div v-if="alliedTeam(game).length === 0" class="faint gd-composition-empty">Composition unavailable</div>
                    </div>
                  </div>
                  <div class="gd-team-col enemy-team">
                    <div class="gd-team-title">Enemies (Red Team)</div>
                    <div class="gd-team-players">
                      <div v-for="player in enemyTeam(game)" :key="`${player.role}-${player.champ}`" class="gd-player-row">
                        <img class="gd-mini-icon" :src="championIcon(player.champ, game.patch)" :alt="player.champ" loading="lazy" @error="fallbackChampionIcon($event, player.champ)"><span class="gd-role-badge">{{ player.roleName }}</span><span class="gd-pname">{{ player.champ }}</span>
                      </div>
                      <div v-if="enemyTeam(game).length === 0" class="faint gd-composition-empty">Composition unavailable</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div class="gd-card">
              <div class="gd-card-header"><span class="obj-glyph gd-icon" style="--obj-mask: url(/images/objectives/dragon.png)"></span><h4>Objectives &amp; Structures ({{ game.objectives?.length || 0 }})</h4></div>
              <div class="gd-card-body">
                <div v-if="!game.objectives?.length" class="gd-empty-objectives faint">No epic monsters or structures recorded in this game.</div>
                <div v-else class="gd-objectives-timeline">
                  <div v-for="(objective, index) in game.objectives" :key="index" class="gd-obj-entry" :class="objective.is_ally ? 'ally' : 'enemy'">
                    <span class="gd-obj-time">{{ objective.minute }}m</span><span class="obj-glyph gd-obj-icon" :style="{ '--obj-mask': `url(${objectiveIcon(objective)})` }"></span><span class="gd-obj-name">{{ objectiveLabel(objective) }}</span><span class="gd-obj-team-badge" :class="objective.is_ally ? 'badge-ally' : 'badge-enemy'">{{ objective.is_ally ? "Allies" : "Enemies" }}</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="gd-card">
              <div class="gd-card-header gd-combat-header">
                <div class="gd-combat-title-wrap"><AppIcon name="swords" class="gd-icon" /><h4>Combat Log ({{ combatCount(game, "all") }})</h4></div>
                <div class="gd-combat-filters">
                  <button type="button" class="combat-filter-btn" :class="{ active: combatFilter === 'all' }" @click="combatFilter = 'all'">All ({{ combatCount(game, "all") }})</button>
                  <button type="button" class="combat-filter-btn filter-kill" :class="{ active: combatFilter === 'kill' }" @click="combatFilter = 'kill'">K ({{ combatCount(game, "kill") }})</button>
                  <button type="button" class="combat-filter-btn filter-death" :class="{ active: combatFilter === 'death' }" @click="combatFilter = 'death'">D ({{ combatCount(game, "death") }})</button>
                  <button type="button" class="combat-filter-btn filter-assist" :class="{ active: combatFilter === 'assist' }" @click="combatFilter = 'assist'">A ({{ combatCount(game, "assist") }})</button>
                </div>
              </div>
              <div class="gd-card-body">
                <div v-if="combatEvents(game, combatFilter).length === 0" class="gd-flawless"><AppIcon name="feather" class="flawless-icon" /><strong>No combat recorded</strong><span class="faint">No kills, deaths, or assists in this selection.</span></div>
                <div v-else class="gd-combat-timeline">
                  <div v-for="(event, index) in combatEvents(game, combatFilter)" :key="index" class="gd-combat-entry" :class="`entry-${event.type}`">
                    <span class="gd-combat-time">{{ event.minute }}m</span><span class="gd-combat-type-badge" :class="`badge-${event.type}`">{{ event.type === "kill" ? "KILL" : event.type === "death" ? "DEATH" : "ASSIST" }}</span>
                    <div class="gd-combat-target"><img class="gd-mini-icon" :src="championIcon(event.champ, game.patch)" :alt="event.champ || 'Unknown'" loading="lazy" @error="fallbackChampionIcon($event, event.champ)"><div class="gd-combat-names"><span class="gd-combat-champ">{{ event.champ }}</span><span v-if="event.type === 'assist' && event.killer_champ" class="gd-combat-sub faint">by {{ event.killer_champ }}</span></div></div>
                    <span class="gd-combat-zone">{{ zoneLabel(event.zone) }}</span>
                    <div class="gd-combat-tags"><span v-if="event.is_solo" class="gd-tag tag-solo" :title="event.type === 'kill' ? 'Unassisted 1v1 kill' : '1v1 death with no enemy assistance'">Solo</span><span v-if="event.is_ganked_by_jungle" class="gd-tag tag-gank" title="The enemy jungler joined the play">Jungle Gank</span><span v-if="event.is_2v2" class="gd-tag tag-2v2" title="2v2 fight involving only the two bot lanes">Bot 2v2</span><span v-if="event.gold_state === 'behind'" class="gd-tag tag-behind" title="More than 600 gold behind the direct lane opponent at the time of the play">Behind</span></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="row gd-pagination">
        <button class="btn" :disabled="page <= 1 || loading" @click="previousPage">← Previous</button>
        <span class="faint">page {{ page }} / {{ pageCount }}</span>
        <button class="btn" data-testid="next-page" :disabled="page * size >= total || loading" @click="nextPage">Next →</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ==========================================================================
   Section Headings & ML Callout
   ========================================================================== */

.section-heading {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 16px;
  margin: 0 0 14px;
}
.section-heading h2 {
  color: var(--text);
  font-size: 18px;
  letter-spacing: -.02em;
}
.section-heading p {
  margin: 3px 0 0;
  color: var(--text-faint);
  font-size: var(--fs-body-sm);
}

.game-history {
  position: relative;
  z-index: 2;
}

.game-card {
  position: relative;
  z-index: 2;
  background: var(--card-marble-bg);
  border: 1px solid var(--card-marble-border);
  border-left: 4px solid var(--border);
  border-radius: var(--radius);
  transition: background 150ms ease, transform 150ms ease, border-color 150ms ease, box-shadow 150ms ease;
  overflow: hidden;
}
.game-card + .game-card { margin-top: 8px; }

.game-card.w {
  border-left-color: var(--win);
  background:
    linear-gradient(90deg, color-mix(in srgb, var(--win) 10%, transparent) 0%, transparent 35%),
    var(--card-marble-bg);
}
.game-card.l {
  border-left-color: var(--loss);
  background:
    linear-gradient(90deg, color-mix(in srgb, var(--loss) 9%, transparent) 0%, transparent 35%),
    var(--card-marble-bg);
}
.game-card:hover {
  border-color: var(--border-active);
}
.game-card.is-expanded {
  border-color: var(--gold);
  box-shadow: var(--card-shadow-hover);
}

.game-row {
  display: grid;
  grid-template-columns: 44px minmax(180px, 1fr) 130px 65px 75px;
  gap: 12px;
  align-items: center;
  min-height: 68px;
  padding: 10px 16px;
  cursor: pointer;
  user-select: none;
  background: transparent;
  transition: background 150ms ease;
}
.game-row:hover {
  background: var(--surface-alt);
}

.gr-expand-action {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 6px;
  color: var(--text-dim);
  font-size: var(--fs-label);
  font-weight: 600;
  transition: color 150ms ease;
}
.game-row:hover .gr-expand-action {
  color: var(--gold);
}
.gr-expand-chevron {
  display: inline-block;
  font-size: var(--fs-body-sm);
  transition: transform 200ms ease;
}
.gr-expand-chevron.open {
  transform: rotate(180deg);
  color: var(--gold);
}
.gr-badge-coach {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 1px 6px;
  border-radius: var(--radius-xs);
  background: var(--info-soft);
  border: 1px solid var(--info-border);
  color: var(--info);
  font-size: var(--fs-micro);
  font-weight: 700;
  letter-spacing: .02em;
}
.gr-badge-pedagogic {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 1px 7px;
  border-radius: var(--radius-xs);
  font-size: var(--fs-micro);
  font-weight: 700;
  letter-spacing: .02em;
}
.gr-badge-pedagogic.badge-loss,
.pedagogic-target-pill.pill-loss {
  background: var(--loss-soft);
  border: 1px solid var(--loss-border);
  color: var(--danger);
}
.gr-badge-pedagogic.badge-win,
.pedagogic-target-pill.pill-win {
  background: var(--win-soft);
  border: 1px solid var(--win-border);
  color: var(--success);
}

.gd-coach-unreviewed-group {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.pedagogic-target-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: var(--fs-label);
  font-weight: 700;
  cursor: help;
}
/* Volet de détails de la partie (style OP.GG / U.GG) */
.game-details-panel {
  border-top: 1px solid var(--border-soft);
  background: var(--surface-alt);
  padding: 16px 20px 20px;
  animation: gdFadeIn 180ms cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes gdFadeIn {
  from { opacity: 0; transform: translateY(-4px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Action bar */
.gd-action-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  padding-bottom: 14px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--border-soft);
}
.gd-meta-info {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  font-size: var(--fs-small);
}
.gd-meta-tag {
  display: inline-flex;
  gap: 5px;
  align-items: center;
  padding: 3px 8px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text-dim);
  font-size: var(--fs-label);
}
.gd-coach-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: var(--fs-small);
  font-weight: 600;
}
.gd-coach-status.ready {
  color: var(--info);
}
.gd-coach-status.empty {
  color: var(--text-dim);
}
.pulse-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--info);
  box-shadow: none;
  animation: pulseDot 2s infinite ease-in-out;
}
@keyframes pulseDot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.8); }
}

/* Bouton Raccourci Coaching LLM */
.btn-coach-shortcut {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: var(--primary-soft);
  border: 1px solid var(--gold);
  border-radius: var(--radius-sm);
  color: var(--gold-deep);
  font-size: var(--fs-body-sm);
  font-weight: 700;
  cursor: pointer;
  transition: all 180ms ease;
  box-shadow: none;
}
.btn-coach-shortcut:hover {
  background: var(--primary-soft);
  border-color: var(--gold-deep);
  color: var(--ink);
  transform: translateY(-1px);
  box-shadow: 0 10px 22px -18px rgba(126, 97, 52, .72);
}
.btn-coach-shortcut:disabled {
  opacity: .55;
  cursor: not-allowed;
}

/* Grille 2x2 des 4 blocs */
.gd-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.gd-card {
  background: var(--surface);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-sm);
  padding: 14px;
  display: flex;
  flex-direction: column;
}
.gd-card-full {
  grid-column: 1 / -1;
}
.gd-card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border-soft);
}
.gd-card-header h4 {
  margin: 0;
  font-size: var(--fs-body-sm);
  font-weight: 700;
  color: var(--text);
  letter-spacing: .02em;
}
.gd-card-body {
  flex: 1;
}

/* Matchup box */
.gd-matchup-box {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  margin-bottom: 12px;
}
.gd-matchup-side {
  display: flex;
  align-items: center;
  gap: 8px;
}
.gd-matchup-side strong {
  font-size: var(--fs-body-sm);
  display: block;
}
.gd-matchup-side .faint {
  font-size: var(--fs-label);
}
.gd-matchup-vs {
  font-size: var(--fs-label);
  font-weight: 800;
  color: var(--gold);
  letter-spacing: .05em;
}

/* Mini champion icons in drawer */
.gd-mini-icon {
  width: 28px;
  height: 28px;
  border-radius: var(--radius-sm);
  object-fit: cover;
  border: 1px solid var(--border);
  background: var(--bg);
}

/* Stats list */
.gd-stats-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.gd-stat-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: var(--fs-small);
  padding: 4px 0;
  border-bottom: 1px dashed var(--surface-alt);
}
.gd-stat-row:last-child {
  border-bottom: none;
}
.gd-stat-label {
  color: var(--text-dim);
}
.gd-stat-group {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.gd-diff-pill {
  display: inline-flex;
  align-items: center;
  padding: 2px 7px;
  border-radius: 4px;
  font-size: var(--fs-label);
  background: var(--panel-2);
  border: 1px solid var(--border);
  color: var(--text-dim);
}
.gd-diff-pill.diff-pos {
  background: var(--info-soft);
  border-color: var(--info-border);
  color: var(--info);
}
.gd-diff-pill.diff-neg {
  background: var(--loss-soft);
  border-color: var(--loss-border);
  color: var(--danger);
}

.text-pink { color: var(--gold-deep); }
.text-danger { color: var(--danger); }
.text-warning { color: var(--gold-deep); }
.text-good { color: var(--success); }
/* Sides Banner (Strongside / Weakside) */
.gd-sides-banner {
  background: var(--surface-alt);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  margin-bottom: 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.gd-sides-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}
.gd-side-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: var(--fs-small);
}
.gd-side-label {
  font-size: var(--fs-label);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .04em;
  color: var(--text-dim);
}
.gd-side-item.ally .gd-side-label {
  color: var(--info);
}
.gd-side-item.enemy .gd-side-label {
  color: var(--danger);
}
.gd-side-val {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--text);
  flex-wrap: wrap;
}
.badge-side {
  display: inline-flex;
  align-items: center;
  font-size: var(--fs-micro);
  font-weight: 800;
  padding: 2px 7px;
  border-radius: var(--radius-xs);
  letter-spacing: .04em;
  text-transform: uppercase;
}
.badge-strongside {
  background: var(--win-soft);
  border: 1px solid var(--win-border);
  color: var(--success);
  box-shadow: none;
}
.badge-weakside {
  background: var(--primary-soft);
  border: 1px solid var(--gold);
  color: var(--gold-deep);
}
.gd-side-summary {
  font-size: var(--fs-small);
  color: var(--gold-deep);
  background: var(--surface-alt);
  padding: 6px 10px;
  border-radius: var(--radius-sm);
  border-left: 3px solid var(--gold);
  line-height: 1.4;
}

/* Teams composition grid */
.gd-teams-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.gd-team-col {
  background: var(--panel-2);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
  border: 1px solid var(--border-soft);
  min-width: 0;
}
.gd-team-col.ally-team {
  border-top: 3px solid var(--info);
}
.gd-team-col.enemy-team {
  border-top: 3px solid var(--danger);
}
.gd-team-title {
  font-size: var(--fs-label);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .05em;
  margin-bottom: 8px;
  color: var(--text-dim);
}
.gd-team-players {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.gd-player-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: var(--fs-small);
  padding: 4px 8px;
  border-radius: var(--radius-sm);
  background: var(--surface-alt);
  min-width: 0;
}
.gd-player-row.is-self {
  background: var(--primary-soft);
  border: 1px solid var(--primary-border);
}
.gd-role-badge {
  font-size: var(--fs-micro);
  font-weight: 700;
  color: var(--text-dim);
  text-transform: uppercase;
  padding: 2px 5px;
  background: var(--surface);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-xs);
  letter-spacing: .04em;
  flex-shrink: 0;
  text-align: center;
  min-width: 28px;
}
.gd-pname {
  font-weight: 600;
  color: var(--text);
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}
.gd-self-badge {
  font-size: var(--fs-micro);
  font-weight: 800;
  padding: 1px 4px;
  background: var(--gold);
  color: var(--paper);
  border-radius: var(--radius-xs);
  text-transform: uppercase;
  flex-shrink: 0;
}

/* Timeline des Objectifs & Structures */
.gd-empty-objectives {
  padding: 20px 12px;
  text-align: center;
  font-size: var(--fs-small);
}
.gd-objectives-timeline,
.gd-combat-timeline {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 220px;
  overflow-y: auto;
  padding-right: 4px;
}
.gd-obj-entry,
.gd-combat-entry {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: var(--fs-label);
  padding: 5px 8px;
  background: var(--panel-2);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-xs);
  transition: background .15s ease;
}
.gd-obj-entry:hover,
.gd-combat-entry:hover {
  background: var(--surface-alt);
}
.gd-obj-entry.ally {
  border-left: 3px solid var(--info);
}
.gd-obj-entry.enemy {
  border-left: 3px solid var(--danger);
}
.gd-obj-time,
.gd-combat-time {
  font-weight: 800;
  color: var(--text-dim);
  min-width: 28px;
  font-family: var(--font-mono, monospace);
}
.gd-obj-icon {
  font-size: var(--fs-body);
  line-height: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
}
.gd-obj-name {
  font-weight: 600;
  color: var(--text);
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.gd-obj-team-badge {
  font-size: var(--fs-micro);
  font-weight: 800;
  padding: 1px 6px;
  border-radius: var(--radius-xs);
  text-transform: uppercase;
  letter-spacing: .03em;
  flex-shrink: 0;
}
.gd-obj-team-badge.badge-ally {
  background: var(--info-soft);
  color: var(--info);
  border: 1px solid var(--info-border);
}
.gd-obj-team-badge.badge-enemy {
  background: var(--loss-soft);
  color: var(--danger);
  border: 1px solid var(--loss-border);
}

/* Journal des Combats (KDA) */
.gd-combat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
}
.gd-combat-title-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
}
.gd-combat-filters {
  display: flex;
  align-items: center;
  gap: 4px;
}
.combat-filter-btn {
  font-size: var(--fs-micro);
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 4px;
  background: var(--panel-2);
  border: 1px solid var(--border-soft);
  color: var(--text-dim);
  cursor: pointer;
  transition: all .15s ease;
}
.combat-filter-btn:hover {
  color: var(--text);
  border-color: var(--border);
}
.combat-filter-btn.active {
  background: var(--panel-hover);
  color: var(--ink);
  border-color: var(--gold);
}
.combat-filter-btn.filter-kill.active {
  background: var(--win-soft);
  border-color: var(--success);
  color: var(--success);
}
.combat-filter-btn.filter-death.active {
  background: var(--loss-soft);
  border-color: var(--danger);
  color: var(--danger);
}
.combat-filter-btn.filter-assist.active {
  background: var(--info-soft);
  border-color: var(--info);
  color: var(--info);
}

.gd-flawless {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 24px 12px;
  text-align: center;
  gap: 6px;
}
.flawless-icon {
  font-size: var(--fs-display);
  margin-bottom: 4px;
}
.gd-flawless strong {
  color: var(--gold-deep);
  font-size: var(--fs-body);
}

.gd-combat-entry.entry-kill {
  border-left: 3px solid var(--success);
}
.gd-combat-entry.entry-death {
  border-left: 3px solid var(--danger);
}
.gd-combat-entry.entry-assist {
  border-left: 3px solid var(--info);
}

.gd-combat-type-badge {
  font-size: var(--fs-micro);
  font-weight: 800;
  padding: 1px 5px;
  border-radius: var(--radius-xs);
  text-transform: uppercase;
  letter-spacing: .03em;
  flex-shrink: 0;
  min-width: 44px;
  text-align: center;
}
.gd-combat-type-badge.badge-kill {
  background: var(--win-soft);
  color: var(--success);
  border: 1px solid var(--win-border);
}
.gd-combat-type-badge.badge-death {
  background: var(--loss-soft);
  color: var(--danger);
  border: 1px solid var(--loss-border);
}
.gd-combat-type-badge.badge-assist {
  background: var(--info-soft);
  color: var(--info);
  border: 1px solid var(--info-border);
}

.gd-combat-target {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 90px;
}
.gd-combat-names {
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.gd-combat-champ {
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.gd-combat-sub {
  font-size: var(--fs-micro);
  line-height: 1;
}

.gd-combat-zone {
  color: var(--text-dim);
  font-size: var(--fs-micro);
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.gd-combat-tags {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.gd-tag {
  font-size: var(--fs-micro);
  font-weight: 700;
  padding: 1px 5px;
  border-radius: var(--radius-xs);
  text-transform: uppercase;
}
.tag-solo {
  background: var(--loss-soft);
  color: var(--danger);
  border: 1px solid var(--loss-border);
}
.tag-gank,
.tag-2v2 {
  background: var(--info-soft);
  color: var(--info);
  border: 1px solid var(--info-border);
}
.tag-behind {
  background: var(--primary-soft);
  color: var(--gold-deep);
  border: 1px solid var(--primary-border);
}

.gr-champ-wrap {
  position: relative;
  width: 44px;
  height: 44px;
}

.champ-icon {
  width: 44px;
  height: 44px;
  border-radius: var(--radius-sm);
  display: block;
  background: var(--bg);
  box-shadow: var(--card-shadow);
  object-fit: cover;
  border: 1px solid var(--border);
}
.champ-fallback {
  font-size: var(--fs-label);
  color: var(--text-dim);
  width: 44px;
  text-align: center;
}

.gr-top-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.gr-sub-row {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-top: 4px;
  font-size: var(--fs-label);
}

.outcome-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 68px;
  padding: 2px 7px;
  border-radius: 6px;
  font-size: var(--fs-micro);
  font-weight: 800;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.pill-win {
  color: var(--info);
  background: var(--info-soft);
  border: 1px solid var(--info-border);
  box-shadow: none;
}
.pill-loss {
  color: var(--danger);
  background: var(--loss-soft);
  border: 1px solid var(--loss-border);
  box-shadow: none;
}

.gr-champ { font-size: 15px; font-weight: 700; color: var(--text); }

.gr-role-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 7px;
  background: var(--panel-2);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-sm);
  font-size: var(--fs-label);
  font-weight: 650;
  color: var(--text-dim);
}
.role-icon {
  width: 14px;
  height: 14px;
  object-fit: contain;
  filter: none;
}

.gr-matchup {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 7px;
  background: var(--panel-2);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-sm);
}
.gr-vs {
  color: var(--text-faint);
  font-size: var(--fs-micro);
  font-weight: 800;
  letter-spacing: .06em;
}
.gr-opp-icon {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 1px solid var(--border);
  object-fit: cover;
}
.gr-opp-name {
  font-size: var(--fs-label);
  color: var(--text-dim);
}

.gr-queue-badge {
  color: var(--text-faint);
  font-size: var(--fs-label);
  font-weight: 600;
}
.gr-sub-sep {
  color: var(--text-faint);
  opacity: 0.5;
  user-select: none;
  font-size: var(--fs-micro);
}
.gr-date {
  color: var(--text-dim);
  font-size: var(--fs-label);
  font-weight: 550;
  cursor: default;
}
.gd-meta-date {
  color: var(--text);
  background: var(--info-soft);
  border-color: var(--info-border);
}
.gr-diff {
  font-size: var(--fs-label);
  font-weight: 700;
}
.diff-pos { color: var(--win); }
.diff-neg { color: var(--loss); }

.gr-kda, .gr-patch {
  display: flex;
  flex-direction: column;
  gap: 1px;
  text-align: right;
}
.gr-kda { color: var(--text-dim); font-size: var(--fs-body-sm); font-weight: 600; }
.gr-label {
  color: var(--text-faint);
  font-size: var(--fs-micro);
  font-weight: 700;
  letter-spacing: .05em;
  text-transform: uppercase;
}

.champ-fallback { display: none; }

.gd-composition-empty {
  padding: 8px 0;
  font-size: var(--fs-label);
}

.gd-pagination {
  justify-content: space-between;
  margin-top: 16px;
}


@media (max-width: 860px) {
  .gd-grid { grid-template-columns: 1fr; }
}

@media (max-width: 640px) {
  .game-row { grid-template-columns: 44px minmax(100px, 1fr) auto auto; gap: 8px; padding: 10px 12px; }
  .gr-kda { grid-column: 2 / 3; align-items: flex-start; text-align: left; }
  .gr-patch { grid-column: 3; align-items: flex-end; justify-content: center; }
  .gr-expand-action { grid-column: 4; }
  .gr-expand-label { display: none; }
  .gr-label { font-size: var(--fs-micro); }
  .game-details-panel { padding: 12px; }
  .gd-grid { grid-template-columns: 1fr; }
  .gd-teams-grid { grid-template-columns: 1fr; }
  .gd-sides-row { flex-direction: column; align-items: flex-start; gap: 8px; }
  .gd-action-bar { flex-direction: column; align-items: stretch; gap: 10px; }
  .btn-coach-shortcut { width: 100%; justify-content: center; }
}

</style>
