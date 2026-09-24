<script setup lang="ts">
import AppIcon from "./AppIcon.vue";
import { rankEmblem } from "../account-profile";

type CoachingView = "overall" | "games";

withDefaults(defineProps<{
  view?: CoachingView;
  gameReviewsCount?: number;
  mainRoleName?: string;
  roleReady?: boolean;
  authenticated?: boolean;
  busy?: boolean;
}>(), {
  view: "overall",
  gameReviewsCount: 0,
  mainRoleName: "",
  roleReady: false,
  authenticated: false,
  busy: false,
});

const emit = defineEmits<{
  "view-change": [view: CoachingView];
  generate: [];
}>();
</script>

<template>
  <div class="coach-view-tabs" role="tablist" aria-label="Coaching type">
    <button
      type="button"
      :class="{ active: view === 'overall' }"
      :aria-selected="view === 'overall'"
      @click="emit('view-change', 'overall')"
    >
      <span class="coach-tab-icon" aria-hidden="true">✦</span>
      <span class="coach-tab-copy">
        <strong>Global coaching</strong>
        <small>Overview and recurring habits</small>
      </span>
    </button>
    <button
      type="button"
      :class="{ active: view === 'games' }"
      :aria-selected="view === 'games'"
      @click="emit('view-change', 'games')"
    >
      <span class="coach-tab-icon" aria-hidden="true">◎</span>
      <span class="coach-tab-copy">
        <strong>Game analyses</strong>
        <small>Detailed reviews, game by game</small>
      </span>
      <span v-if="gameReviewsCount" class="coach-tab-count">{{ gameReviewsCount }}</span>
    </button>
  </div>

  <template v-if="view === 'overall'">
    <section class="coach-builder" aria-labelledby="coach-builder-title">
      <div class="coaching-intro">
        <span class="coaching-kicker">GLOBAL ANALYSIS &amp; HABITS</span>
        <h2 id="coach-builder-title">See the bigger picture</h2>
        <p>One clear synthesis of your reviewed games, automatically adapted to your main role.</p>
      </div>
      <div class="coach-options">
        <div class="choice-group main-role-card" aria-label="Detected main role">
          <span class="choice-label">Main role detected</span>
          <strong>{{ mainRoleName || "Refresh required" }}</strong>
          <small>{{ roleReady ? "Shared with your ML & SHAP profile" : "Refresh the account to detect your main role" }}</small>
        </div>
        <div class="coach-reference" aria-label="Reference: Challenger players">
          <img class="coach-ref-emblem" :src="rankEmblem('challenger')" alt="Challenger" loading="lazy">
          <div><span>Reference</span><strong>Challenger players</strong></div>
        </div>
        <button class="btn btn-primary coach-generate" :disabled="busy || !roleReady" @click="emit('generate')">
          <AppIcon v-if="!busy && !authenticated" name="lock" />
          <span>{{ busy ? "Analyzing…" : (authenticated ? "Generate coaching" : "Unlock coaching") }}</span>
          <span v-if="!busy" aria-hidden="true">→</span>
        </button>
      </div>
    </section>
  </template>
</template>

<style scoped>
.coach-view-tabs {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  width: 100%;
  gap: 12px;
  margin: 0 0 22px;
}

.coach-view-tabs button {
  position: relative;
  display: flex;
  align-items: center;
  gap: 14px;
  min-width: 0;
  min-height: 102px;
  padding: 22px;
  color: var(--text);
  text-align: left;
  background:
    linear-gradient(120deg, var(--primary-soft), transparent 45%),
    var(--card-marble-bg);
  border: 1px solid var(--card-marble-border);
  border-radius: 16px;
  box-shadow: var(--card-shadow);
  cursor: pointer;
  font-family: inherit;
  transition: var(--transition-base);
}

.coach-view-tabs button:hover {
  color: var(--text);
  border-color: var(--primary-border);
  box-shadow: var(--card-shadow-hover);
  transform: translateY(-1px);
}

.coach-view-tabs button.active {
  color: var(--text);
  background:
    linear-gradient(120deg, color-mix(in srgb, var(--primary) 18%, transparent), transparent 52%),
    var(--panel-gradient);
  border-color: var(--primary);
  box-shadow:
    var(--card-shadow),
    inset 0 0 0 1px var(--primary-border);
}

.coach-tab-icon {
  display: grid;
  flex: 0 0 36px;
  width: 36px;
  height: 36px;
  place-items: center;
  color: var(--primary);
  background: var(--primary-soft);
  border: 1px solid color-mix(in srgb, var(--primary) 35%, transparent);
  border-radius: var(--radius-sm);
  font-size: var(--fs-title);
  line-height: 1;
}

.coach-view-tabs button.active .coach-tab-icon {
  color: var(--primary);
  background: color-mix(in srgb, var(--primary) 16%, transparent);
  border-color: var(--primary-border);
}

.coach-tab-copy {
  display: block;
  min-width: 0;
}

.coach-tab-copy strong,
.coach-tab-copy small {
  display: block;
}

.coach-tab-copy strong {
  color: var(--text);
  font-size: var(--fs-title);
  font-weight: 750;
  line-height: 1.26;
  letter-spacing: -.025em;
}

.coach-tab-copy small {
  margin-top: 5px;
  color: var(--text-dim);
  font-size: var(--fs-body-sm);
  font-weight: 400;
  line-height: 1.45;
}

.coach-view-tabs button.active .coach-tab-copy small {
  color: var(--text-dim);
}

.coach-tab-count {
  display: grid;
  flex: 0 0 auto;
  min-width: 25px;
  height: 25px;
  margin-left: auto;
  padding: 0 6px;
  place-items: center;
  color: var(--primary);
  background: var(--primary-soft);
  border: 1px solid color-mix(in srgb, var(--primary) 30%, transparent);
  border-radius: 999px;
  font-size: var(--fs-label);
  font-weight: 800;
}

.coach-view-tabs button.active .coach-tab-count {
  color: var(--primary);
  background: var(--primary-soft);
  border-color: var(--primary-border);
}

.coach-builder {
  padding: 22px;
  margin-bottom: 22px;
  background:
    linear-gradient(120deg, var(--primary-soft), transparent 45%),
    var(--card-marble-bg);
  border: 1px solid var(--card-marble-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--card-shadow);
}

.coach-builder .coaching-intro {
  margin: 0 0 18px;
}

.coach-builder .coaching-kicker {
  display: block;
  margin-bottom: 5px;
  color: var(--primary);
  font-size: var(--fs-label);
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.coach-builder .coaching-intro h2 {
  color: var(--text);
  font-size: var(--fs-title);
  letter-spacing: -.025em;
}

.coach-builder .coaching-intro p {
  max-width: 630px;
  margin: 5px 0 0;
  color: var(--text-dim);
  font-size: var(--fs-body-sm);
}

.coach-options {
  display: grid;
  grid-template-columns: minmax(205px, 1fr) minmax(245px, 1.15fr) auto;
  gap: 10px;
  align-items: stretch;
}

.choice-group,
.coach-reference {
  min-width: 0;
  padding: 10px 12px;
  background: var(--panel);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-sm);
}

.choice-label {
  display: block;
  margin: 0 0 6px;
  color: var(--text-faint);
  font-size: var(--fs-micro);
  font-weight: 750;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.main-role-card strong,
.main-role-card small {
  display: block;
}

.main-role-card strong {
  color: var(--text);
  font-size: 15px;
}

.main-role-card small {
  margin-top: 3px;
  color: var(--text-faint);
  font-size: var(--fs-micro);
}

.coach-reference {
  display: flex;
  gap: 10px;
  align-items: center;
}

.coach-ref-emblem {
  width: 34px;
  height: 34px;
  object-fit: contain;
}

.coach-reference span {
  display: block;
  color: var(--text-faint);
  font-size: var(--fs-micro);
  font-weight: 700;
  letter-spacing: .06em;
  text-transform: uppercase;
}

.coach-reference strong {
  display: block;
  margin-top: 2px;
  color: var(--text);
  font-size: var(--fs-small);
  line-height: 1.2;
}

.coach-generate {
  align-self: stretch;
  min-height: 62px;
  padding: 10px 16px;
  white-space: nowrap;
}

.coach-generate span {
  margin-left: 5px;
  font-size: var(--fs-lead);
}

.coach-generate .icon + span {
  margin-left: 0;
}

@media (max-width: 640px) {
  .coach-builder {
    padding: 18px;
    border-radius: var(--radius);
  }
  .coach-options {
    grid-template-columns: 1fr;
  }
  .coach-reference {
    min-height: 54px;
  }
  .coach-generate {
    min-height: 44px;
  }
  .coach-view-tabs {
    grid-template-columns: 1fr;
  }
  .coach-view-tabs button {
    width: 100%;
    min-height: 94px;
    padding: 18px;
  }
}
</style>
