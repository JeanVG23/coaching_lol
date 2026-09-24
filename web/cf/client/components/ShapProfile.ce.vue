<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { formatDate, rankEmblem } from "../account-profile";
import { withAuthHeaders } from "../auth";
import { formatFeatureValue } from "../feature-catalog";
import type { IngestSync } from "../ingest-sync";
import {
  boundaryLabel,
  formatLogit,
  isGlobalClosure,
  parseRoleAnalysis,
  reasonMessage,
  roleLabel,
  type RoleAnalysis,
} from "../role-analysis";
import {
  GAUGE_MAX_LOGIT,
  directionHint,
  formatContribution,
  gaugeOffset,
  groupByBase,
  groupByTheme,
  pickHighlights,
  verdictLine,
  type BaseGroup,
} from "../shap-view";

const props = defineProps<{ slug: string; sync: IngestSync; reloadToken: number }>();

const loading = ref(true);
const analysis = ref<RoleAnalysis | null>(null);
const unavailableReason = ref<string | null>(null);
const sort = ref<"abs" | "val">("abs");
const categoryFilter = ref<"all" | "actionable">("all");
/** Le tableau complet et le mode d'emploi sont repliés à l'ouverture. Ce qui
 * reste peint est ce qui se lit d'un coup d'oeil : le score situé, sa phrase,
 * les six domaines, les trois forces et les trois leviers. Le reste se
 * demande, il ne s'impose plus. */
const showDetail = ref(false);
const showHelp = ref(false);
const expanded = ref<Record<string, boolean>>({});
let requestSequence = 0;

/** Info-bulle du bouton d'actualisation, partagée par les deux branches du
 * template (états disponible et indisponible) : le markup reste dupliqué,
 * la chaîne vit ici une seule fois. */
const refreshTitle = computed(() => props.sync.cooling
  ? "Analysis already refreshed: you can refresh again every 15 minutes"
  : "Collect the latest 20 games for this role and recalculate the breakdown");

/** Libellé du bouton d'actualisation, partagé par les deux branches du
 * template : feedback du composable d'abord, sinon l'état (collecte en
 * cours, cooldown avec décompte, repos). */
const refreshLabel = computed(() => props.sync.feedback
  || (props.sync.syncing
    ? "Refreshing…"
    : (props.sync.cooling ? props.sync.cooldownLabel : "Refresh analysis")));

/** Vrai quand le motif d'indisponibilité courant est l'un des trois motifs
 * structurels (`role_closed`, `model_missing`, `model_mismatch`) : décidés
 * par les artefacts déployés, donc identiques pour tout compte du site. */
const isGlobalUnavailable = computed(() =>
  unavailableReason.value !== null && isGlobalClosure(unavailableReason.value));

const drivers = computed(() => analysis.value?.drivers ?? []);

/** Les thèmes résument la décomposition COMPLÈTE du score : métriques
 * descriptives incluses, et sans subir le filtre « leviers d'action ». C'est
 * d'où vient le score, pas une liste de consignes. Les leviers, eux, ne
 * lisent que l'actionnable (`pickHighlights`). */
const themeRows = computed(() => groupByTheme(drivers.value));
const allGroups = computed(() => groupByBase(drivers.value));
const highlights = computed(() => pickHighlights(allGroups.value));

const filteredGroups = computed<BaseGroup[]>(() => {
  const kept = categoryFilter.value === "actionable"
    ? allGroups.value.filter((group) => group.actionable)
    : allGroups.value;
  // `groupByBase` rend déjà l'ordre par poids absolu ; le tri par valeur
  // signée travaille sur une copie pour ne pas réordonner la source.
  return sort.value === "abs" ? kept : [...kept].sort((left, right) => right.net - left.net);
});

const maxThemeWeight = computed(() =>
  Math.max(1e-9, ...themeRows.value.map((theme) => Math.abs(theme.net))));
/** Une phrase, dérivée des seuls domaines : le premier coup d'oeil. */
const verdict = computed(() => verdictLine(themeRows.value));

const maxMetricWeight = computed(() =>
  Math.max(1e-9, ...filteredGroups.value.map((group) => Math.abs(group.net))));

const gaugeLeft = computed(() => {
  const logit = analysis.value?.logit;
  return `${gaugeOffset(typeof logit === "number" ? logit : 0).toFixed(2)}%`;
});

interface ReadLine {
  aggregation: string;
  value: string;
  crossover: string | null;
  hint: string;
}

interface MetricView {
  group: BaseGroup;
  read: ReadLine | null;
}

/** La lecture d'une métrique tient dans l'agrégation qui explique son net
 * (`BaseGroup.lead`) : sa valeur, le point de bascule de la shape function et
 * le sens qui rapproche de l'apex. Les trois sont publiés ; ils vivaient dans
 * une info-bulle. */
function readOf(group: BaseGroup): ReadLine | null {
  const part = group.lead;
  if (!part || part.value === null) return null;
  const label = part.aggregationLabel ?? "over the window";
  return {
    aggregation: label.charAt(0).toUpperCase() + label.slice(1),
    value: formatFeatureValue(part.value, part.unit),
    crossover: part.crossover === null
      ? null
      : `tipping point ${formatFeatureValue(part.crossover, part.unit)}`,
    hint: directionHint(part.sense),
  };
}

function toView(group: BaseGroup): MetricView {
  return { group, read: readOf(group) };
}

const visibleRows = computed(() => filteredGroups.value.map(toView));
const strengthViews = computed(() => highlights.value.strengths.map(toView));
const leverViews = computed(() => highlights.value.levers.map(toView));

/** Barre divergente : le zéro est au milieu de la piste, donc la plus forte
 * contribution occupe une demi-piste. */
function barStyle(value: number, max: number): Record<string, string> {
  const width = `${Math.min(50, (Math.abs(value) / max) * 50).toFixed(1)}%`;
  return value >= 0 ? { left: "50%", width } : { right: "50%", width };
}

function toggleMetric(base: string): void {
  expanded.value = { ...expanded.value, [base]: !expanded.value[base] };
}

async function loadAnalysis(): Promise<void> {
  const sequence = ++requestSequence;
  loading.value = true;
  analysis.value = null;
  unavailableReason.value = null;
  expanded.value = {};
  showDetail.value = false;
  showHelp.value = false;
  try {
    const response = await fetch(`/api/c/${encodeURIComponent(props.slug)}/shap-role`, {
      headers: withAuthHeaders(),
    });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const payload = parseRoleAnalysis(await response.json());
    if (sequence !== requestSequence) return;
    if (payload.available) analysis.value = payload;
    else unavailableReason.value = payload.reason;
  } catch {
    // Échec réseau côté client : motif propre, jamais déguisé en attente de
    // première collecte (ce serait faux pour un simple échec de chargement).
    if (sequence === requestSequence) unavailableReason.value = "fetch_failed";
  } finally {
    if (sequence === requestSequence) loading.value = false;
  }
}

function toggleSort(): void {
  sort.value = sort.value === "abs" ? "val" : "abs";
}

function toggleCategory(): void {
  categoryFilter.value = categoryFilter.value === "all" ? "actionable" : "all";
}

function goToAccount(targetSlug: string): void {
  window.dispatchEvent(new CustomEvent("coach-go", { detail: { path: `/c/${targetSlug}?tab=shap` } }));
}

watch(() => props.slug, loadAnalysis);
watch(() => props.reloadToken, loadAnalysis);
onMounted(loadAnalysis);
</script>

<template>
  <div class="shap-container">
    <div v-if="loading" class="state">
      <span class="loading-spinner" aria-hidden="true"></span>
      <span>Loading ML profile…</span>
    </div>
    <div v-else-if="unavailableReason" class="shap-empty-card">
      <div class="shap-unavail-badge">
        <span class="badge badge-region">EBM &amp; SHAP MODEL</span>
      </div>
      <h3 class="shap-unavail-title">ML analysis is unavailable for this account.</h3>
      <p class="shap-unavail-sub faint">{{ reasonMessage(unavailableReason) }}</p>

      <!-- Le bouton vit AUSSI dans l'état indisponible : les motifs
           collection_incomplete et scoring_failed disent « relance
           l'actualisation », le CTA doit exister là où on l'appelle. Retiré
           pour les trois motifs structurels (role_closed, model_missing,
           model_mismatch) : décidés par les artefacts déployés, une
           re-collecte ne peut rien y changer, et le clic consommerait pour
           rien le cooldown partagé avec le bouton « Actualiser les données »
           du hero (une seule instance de sync pour toute la page). -->
      <div v-if="!isGlobalUnavailable" class="shap-unavail-actions">
        <button
          class="btn btn-sort-shap btn-refresh-shap"
          :class="{ 'is-syncing': sync.syncing, 'is-cooling': sync.cooling }"
          :disabled="sync.syncing || sync.cooling"
          type="button"
          :title="refreshTitle"
          @click="sync.trigger"
        >
          <svg class="sync-icon-shap" :class="{ spinning: sync.syncing }" viewBox="0 0 20 20" width="14" height="14" fill="currentColor" aria-hidden="true">
            <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"/>
          </svg>
          {{ refreshLabel }}
        </button>
      </div>

      <!-- Les comptes démo montrent une analyse EN DIRECT. Pour les trois
           motifs structurels, ces comptes rendraient la MÊME carte
           indisponible (motif global, identique pour tout compte) : le CTA
           promettrait un palier et une décomposition que la destination ne
           peut pas montrer, retiré pour ces trois motifs uniquement. -->
      <div v-if="!isGlobalUnavailable" class="shap-demo-guidance">
        <span class="demo-guidance-title">Explore the SHAP explainability model live on calibrated profiles:</span>
        <div class="demo-guidance-buttons">
          <a class="btn btn-demo-shap" href="/c/spadzze?tab=shap" @click.prevent="goToAccount('spadzze')">
            <img class="demo-shap-avatar" :src="rankEmblem('diamond')" alt="" loading="lazy">
            <div class="demo-shap-info">
              <span class="demo-shap-name">Spadzze#euw</span>
              <span class="demo-shap-desc">Estimated tier: Diamond · 24 SHAP factors explained</span>
            </div>
            <span class="demo-shap-arrow" aria-hidden="true">→</span>
          </a>
          <a class="btn btn-demo-shap" href="/c/aceofspadzze?tab=shap" @click.prevent="goToAccount('aceofspadzze')">
            <img class="demo-shap-avatar" :src="rankEmblem('grandmaster')" alt="" loading="lazy">
            <div class="demo-shap-info">
              <span class="demo-shap-name">AceOfSpadzze#EQ4</span>
              <span class="demo-shap-desc">Estimated tier: Grandmaster · EBM calibration</span>
            </div>
            <span class="demo-shap-arrow" aria-hidden="true">→</span>
          </a>
        </div>
      </div>
    </div>
    <div v-else-if="analysis" class="shap-active-card">
      <div class="shap-header-row">
        <div>
          <div class="eyebrow-shap">MODEL EXPLAINABILITY</div>
          <h2>What shapes your ML profile</h2>
        </div>
        <div class="spacer"></div>
        <button
          class="btn btn-sort-shap shap-help-toggle"
          type="button"
          :aria-expanded="showHelp === true"
          title="What the axis, the colours and the aggregations mean"
          @click="showHelp = !showHelp"
        >
          How to read this
        </button>
        <button
          class="btn btn-sort-shap btn-refresh-shap"
          :class="{ 'is-syncing': sync.syncing, 'is-cooling': sync.cooling }"
          :disabled="sync.syncing || sync.cooling"
          type="button"
          :title="refreshTitle"
          @click="sync.trigger"
        >
          <svg class="sync-icon-shap" :class="{ spinning: sync.syncing }" viewBox="0 0 20 20" width="14" height="14" fill="currentColor" aria-hidden="true">
            <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"/>
          </svg>
          {{ refreshLabel }}
        </button>
      </div>

      <div class="shap-context-band">
        <span>Analysis of the latest {{ analysis.sample?.role_games_used ?? "?" }} {{ roleLabel(analysis.role) }} games</span>
        <span v-if="analysis.generated_at">generated {{ formatDate(analysis.generated_at) }}</span>
      </div>

      <div v-if="showHelp" class="shap-help-panel">
        <p class="shap-gauge-note">
          Log-odds scale, shown from -{{ GAUGE_MAX_LOGIT }} to +{{ GAUGE_MAX_LOGIT }}; zero is the boundary the
          model learned. A relative position, not a predicted rank or a probability.
        </p>
        <p class="section-hint">
          Every published factor, summed by area. This is the breakdown of your score, not a to-do list.
        </p>
        <p class="section-hint">
          The model publishes each metric five times (average, typical game, both extremes, swing
          between games) and a row sums them. Open a row for its definition and its five values.
        </p>
        <div class="shap-legend-strip">
          <div class="shap-legend-item">
            <span class="legend-box legend-box--gold" aria-hidden="true"></span>
            <span><strong>To the right</strong>: moves you closer to the apex</span>
          </div>
          <div class="shap-legend-item">
            <span class="legend-box legend-box--loss" aria-hidden="true"></span>
            <span><strong>To the left</strong>: moves you away from it</span>
          </div>
          <div class="shap-legend-item">
            <span class="metric-tag">context</span>
            <span>describes your game window and is never presented as something to fix</span>
          </div>
        </div>
      </div>

      <!-- Le score reste un logit, jamais converti en probabilité ni en rang.
           Le seul repère porté par l'axe est le zéro, qui est la frontière
           apprise par le modèle : un point que le modèle connaît vraiment. -->
      <div class="shap-score-block">
        <div class="shap-score-head">
          <span class="shap-score-label">Apex proximity</span>
          <strong class="shap-score-value">{{ analysis.logit !== null && analysis.logit !== undefined ? formatLogit(analysis.logit) : "?" }}</strong>
          <span v-if="analysis.model?.boundary" class="shap-score-boundary">{{ boundaryLabel(analysis.model.boundary) }} boundary</span>
        </div>
        <div class="shap-gauge">
          <span class="gauge-end">Diamond</span>
          <div class="gauge-track">
            <span class="gauge-zero" aria-hidden="true"></span>
            <span class="shap-gauge-cursor" :style="{ left: gaugeLeft }" aria-hidden="true"></span>
          </div>
          <span class="gauge-end gauge-end--high">GM+</span>
        </div>
        <p v-if="verdict" class="shap-verdict">{{ verdict }}</p>
        <span v-if="analysis.model?.auc_heldout_median !== undefined" class="shap-score-meta">
          {{ roleLabel(analysis.role) }} EBM · median AUC {{ analysis.model.auc_heldout_median }} across {{ analysis.model.n_seeds ?? "?" }} runs
        </span>
      </div>

      <section v-if="themeRows.length" class="shap-section">
        <h3 class="section-title">Where your score comes from</h3>
        <div v-for="theme in themeRows" :key="theme.theme" class="shap-theme-row">
          <span class="theme-label">{{ theme.label }}</span>
          <div class="bar-track">
            <span
              class="bar"
              :class="theme.net >= 0 ? 'bar--pos' : 'bar--neg'"
              :style="barStyle(theme.net, maxThemeWeight)"
            ></span>
          </div>
          <span class="theme-net" :class="theme.net >= 0 ? 'is-pos' : 'is-neg'">{{ formatContribution(theme.net) }}</span>
        </div>
      </section>

      <section v-if="strengthViews.length || leverViews.length" class="shap-highlights">
        <div class="highlight-column">
          <h3 class="section-title">What the model credits you for</h3>
          <article
            v-for="view in strengthViews"
            :key="view.group.base"
            class="shap-highlight shap-highlight--strength"
          >
            <div class="highlight-head">
              <span class="highlight-label">{{ view.group.label }}</span>
              <span class="highlight-net is-pos">{{ formatContribution(view.group.net) }}</span>
            </div>
            <p v-if="view.read" class="highlight-read">
              <strong>{{ view.read.aggregation }}</strong>: {{ view.read.value
              }}<span v-if="view.read.crossover">, {{ view.read.crossover }}</span>. {{ view.read.hint }}.
            </p>
          </article>
        </div>
        <div class="highlight-column">
          <h3 class="section-title">What costs you the most</h3>
          <article
            v-for="view in leverViews"
            :key="view.group.base"
            class="shap-highlight shap-highlight--lever"
          >
            <div class="highlight-head">
              <span class="highlight-label">{{ view.group.label }}</span>
              <span class="highlight-net is-neg">{{ formatContribution(view.group.net) }}</span>
            </div>
            <p v-if="view.read" class="highlight-read">
              <strong>{{ view.read.aggregation }}</strong>: {{ view.read.value
              }}<span v-if="view.read.crossover">, {{ view.read.crossover }}</span>. {{ view.read.hint }}.
            </p>
          </article>
        </div>
      </section>

      <section class="shap-section">
        <button
          class="shap-detail-toggle"
          type="button"
          :aria-expanded="showDetail === true"
          @click="showDetail = !showDetail"
        >
          <span class="metric-chevron" :class="{ 'is-open': showDetail }" aria-hidden="true">›</span>
          <span class="section-title">Every published metric ({{ allGroups.length }})</span>
        </button>

        <div v-if="showDetail" class="shap-detail-body">
          <div class="section-head">
            <div class="spacer"></div>
            <button class="btn btn-sort-shap" type="button" @click="toggleCategory">
              {{ categoryFilter === "all" ? "All" : "Actionable factors only" }}
            </button>
            <button class="btn btn-sort-shap" type="button" @click="toggleSort">
              <svg class="shap-sort-icon" viewBox="0 0 20 20" width="14" height="14" fill="currentColor" aria-hidden="true">
                <path d="M3 3a1 1 0 000 2h11a1 1 0 100-2H3zM3 7a1 1 0 000 2h7a1 1 0 100-2H3zM3 11a1 1 0 100 2h4a1 1 0 100-2H3zM15 8a1 1 0 10-2 0v5.586l-1.293-1.293a1 1 0 00-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L15 13.586V8z"/>
              </svg>
              {{ sort === "abs" ? "Sort by impact" : "Sort by value" }}
            </button>
          </div>

          <div v-if="visibleRows.length" class="shap-metric-list">
            <div v-for="view in visibleRows" :key="view.group.base" class="shap-metric-row">
              <button
                class="metric-head"
                type="button"
                :aria-expanded="expanded[view.group.base] === true"
                @click="toggleMetric(view.group.base)"
              >
                <span class="metric-title">
                  <span class="metric-label">{{ view.group.label }}</span>
                  <span v-if="!view.group.actionable" class="metric-tag">context</span>
                </span>
                <div class="bar-track">
                  <span
                    class="bar"
                    :class="view.group.net >= 0 ? 'bar--pos' : 'bar--neg'"
                    :style="barStyle(view.group.net, maxMetricWeight)"
                  ></span>
                </div>
                <span class="metric-net" :class="view.group.net >= 0 ? 'is-pos' : 'is-neg'">
                  {{ formatContribution(view.group.net) }}
                </span>
                <span class="metric-chevron" :class="{ 'is-open': expanded[view.group.base] }" aria-hidden="true">›</span>
              </button>

              <!-- Chaque fragment est un élément : le conteneur flex pose
                   l'espacement, sans dépendre des noeuds de texte que Vue
                   condense entre deux balises. -->
              <p v-if="view.read && expanded[view.group.base]" class="metric-read">
                <span><strong>{{ view.read.aggregation }}</strong>: {{ view.read.value }}</span>
                <span v-if="view.read.crossover" class="metric-cross">· {{ view.read.crossover }}</span>
                <span class="metric-sense">· {{ view.read.hint }}</span>
              </p>

              <div v-if="expanded[view.group.base]" class="metric-parts">
                <p class="metric-definition">{{ view.group.description }}</p>
                <div class="metric-parts-header">
                  <span>aggregation</span>
                  <span>your value</span>
                  <span>tipping point</span>
                  <span class="part-contribution">contribution</span>
                  <span>feature</span>
                </div>
                <div v-for="part in view.group.parts" :key="part.feature" class="metric-part">
                  <span class="part-agg">{{ part.aggregationLabel ?? "over the window" }}</span>
                  <span class="part-value">
                    {{ part.value === null ? "no value" : formatFeatureValue(part.value, part.unit) }}
                  </span>
                  <span class="part-crossover">
                    {{ part.crossover === null ? "none" : formatFeatureValue(part.crossover, part.unit) }}
                  </span>
                  <span class="part-contribution" :class="part.contribution >= 0 ? 'is-pos' : 'is-neg'">
                    {{ formatContribution(part.contribution) }}
                  </span>
                  <code class="part-technical">{{ part.feature }}</code>
                </div>
              </div>
            </div>
          </div>
          <p v-else class="shap-empty">
            {{ drivers.length === 0
              ? "No factors were published for this game window."
              : "No actionable factors among the published data. Switch back to “All” to view context factors." }}
          </p>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.shap-container {
  margin-top: 10px;
}

.shap-empty-card {
  padding: 32px 28px;
  background: var(--card-marble-bg);
  border: 1px solid var(--card-marble-border);
  border-radius: var(--radius);
  box-shadow: var(--card-shadow);
  text-align: center;
  max-width: 680px;
  margin: 20px auto;
}

.shap-unavail-title {
  font-size: 18px;
  color: var(--text);
  margin: 12px 0 8px;
}

.shap-unavail-sub {
  font-size: var(--fs-body-sm);
  line-height: 1.5;
  max-width: 500px;
  margin: 0 auto 24px;
}

.shap-unavail-actions {
  display: flex;
  justify-content: center;
  margin: 0 auto 24px;
}

.shap-demo-guidance {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 18px;
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  text-align: left;
}

.demo-guidance-title {
  font-size: var(--fs-small);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .03em;
  color: var(--primary);
}

.demo-guidance-buttons {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.btn-demo-shap {
  flex: 1 1 240px;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: var(--panel);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-sm);
  color: var(--text);
  text-decoration: none;
  transition: var(--transition-base);
}
.btn-demo-shap:hover {
  background: var(--panel-hover);
  border-color: var(--primary);
  transform: translateY(-1px);
  text-decoration: none;
}

.demo-shap-avatar {
  width: 30px;
  height: 30px;
  flex: none;
  object-fit: contain;
}

.demo-shap-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
}

.demo-shap-name {
  font-size: var(--fs-body-sm);
  font-weight: 750;
  color: var(--text);
}

.demo-shap-desc {
  font-size: var(--fs-label);
  color: var(--text-faint);
}

.demo-shap-arrow {
  color: var(--primary);
  font-weight: 700;
}

.shap-active-card {
  padding: 24px;
  background: var(--card-marble-bg);
  border: 1px solid var(--card-marble-border);
  border-radius: var(--radius);
  box-shadow: var(--card-shadow);
}

.shap-header-row {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.shap-header-row h2 {
  color: var(--text);
  font-size: var(--fs-title);
  letter-spacing: -.03em;
}

.eyebrow-shap {
  font-size: var(--fs-label);
  font-weight: 800;
  letter-spacing: .08em;
  color: var(--primary);
  text-transform: uppercase;
  margin-bottom: 2px;
}

.btn-sort-shap {
  display: inline-flex;
  align-items: center;
  padding: 6px 12px;
  font-size: var(--fs-small);
  font-weight: 650;
}

.shap-sort-icon { margin-right: 6px; }

.btn-refresh-shap {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.btn-refresh-shap:disabled {
  opacity: .65;
  cursor: not-allowed;
}

.sync-icon-shap.spinning {
  animation: spin-shap .9s linear infinite;
}

@keyframes spin-shap {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.shap-context-band {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  margin: 0 0 16px;
  padding: 8px 12px;
  font-size: var(--fs-small);
  color: var(--text-dim);
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}

/* ---- Couche 1 : le score situé ---- */

.shap-score-block {
  margin: 0 0 22px;
  padding: 18px 20px;
  background: var(--panel);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
}

.shap-score-head {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 4px 16px;
}

.shap-score-label {
  font-size: var(--fs-label);
  font-weight: 700;
  letter-spacing: .05em;
  text-transform: uppercase;
  color: var(--text-faint);
}

.shap-score-value {
  font-size: 30px;
  font-weight: 800;
  color: var(--ink);
  line-height: 1.1;
}

.shap-score-boundary {
  font-size: var(--fs-small);
  color: var(--text-dim);
}

.shap-gauge {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 14px 0 8px;
}

.gauge-end {
  flex-shrink: 0;
  font-size: var(--fs-label);
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: var(--text-faint);
}

.gauge-track {
  position: relative;
  flex: 1;
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--loss-soft) 0 50%, var(--gold-soft) 50% 100%);
  border: 1px solid var(--border);
}

.gauge-zero {
  position: absolute;
  top: -4px;
  bottom: -4px;
  left: 50%;
  width: 2px;
  transform: translateX(-1px);
  background: var(--border-strong);
}

.shap-gauge-cursor {
  position: absolute;
  top: 50%;
  width: 14px;
  height: 14px;
  margin-left: -7px;
  border-radius: 50%;
  transform: translateY(-50%);
  background: var(--ink);
  border: 2px solid var(--paper);
  box-shadow: var(--card-shadow);
}

.shap-gauge-note {
  margin: 0;
  font-size: var(--fs-label);
  color: var(--text-faint);
}

/* La phrase du premier coup d'oeil : une seule, sous la jauge, en taille de
   lecture et non en taille de note de bas de page. */
.shap-verdict {
  margin: 10px 0 0;
  font-size: var(--fs-body-sm);
  line-height: 1.5;
  color: var(--text);
}

.shap-help-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin: 0 0 16px;
  padding: 14px 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

.shap-help-panel .section-hint,
.shap-help-panel .shap-legend-strip { margin: 0; }

.shap-detail-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 10px 12px;
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  text-align: left;
}

.shap-detail-toggle:hover { border-color: var(--border-strong, var(--border)); }

.shap-detail-toggle .section-title { margin: 0; }

.shap-detail-body { margin-top: 12px; }

.shap-score-meta {
  display: block;
  margin-top: 6px;
  font-size: var(--fs-label);
  color: var(--text-faint);
}

/* ---- Couches 2 et 4 : sections, barres divergentes ---- */

.shap-section {
  margin: 0 0 22px;
}

.section-head {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}

.section-head .spacer { flex: 1; }

.section-title {
  margin: 0 0 4px;
  font-size: var(--fs-body);
  font-weight: 750;
  color: var(--text);
}

.section-hint {
  margin: 0 0 12px;
  font-size: var(--fs-small);
  line-height: 1.5;
  color: var(--text-dim);
  max-width: 720px;
}

.bar-track {
  position: relative;
  height: 12px;
  border-radius: var(--radius-xs);
  background: var(--surface-alt);
  overflow: hidden;
}

.bar-track::before {
  content: "";
  position: absolute;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 1px;
  background: var(--border-strong);
}

.bar {
  position: absolute;
  top: 2px;
  bottom: 2px;
  border-radius: var(--radius-xs);
}

.bar--pos { background: var(--gold); }
.bar--neg { background: var(--danger); }

.is-pos { color: var(--gold-deep, var(--gold)); }
.is-neg { color: var(--danger); }

.shap-theme-row {
  display: grid;
  grid-template-columns: 170px 1fr 64px;
  align-items: center;
  gap: 12px;
  padding: 6px 0;
}

.theme-label {
  font-size: var(--fs-body-sm);
  font-weight: 650;
  color: var(--text);
}

.theme-net,
.metric-net {
  font-size: var(--fs-small);
  font-weight: 750;
  font-variant-numeric: tabular-nums;
  text-align: right;
}

/* ---- Couche 3 : forces et leviers ---- */

.shap-highlights {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin: 0 0 22px;
}

.highlight-column {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.shap-highlight {
  padding: 12px 14px;
  border-radius: var(--radius);
  border: 1px solid var(--border-soft);
  background: var(--panel);
  border-left-width: 3px;
}

.shap-highlight--strength { border-left-color: var(--gold); }
.shap-highlight--lever { border-left-color: var(--danger); }

.highlight-head {
  display: flex;
  align-items: baseline;
  gap: 10px;
}

.highlight-label {
  flex: 1;
  font-size: var(--fs-body-sm);
  font-weight: 700;
  color: var(--text);
}

.highlight-net {
  font-size: var(--fs-small);
  font-weight: 750;
  font-variant-numeric: tabular-nums;
}

.highlight-read {
  margin: 6px 0 0;
  font-size: var(--fs-small);
  line-height: 1.55;
  color: var(--text-dim);
}

/* ---- Couche 4 : le détail par métrique ---- */

.shap-legend-strip {
  display: flex;
  gap: 18px;
  margin: 0 0 12px;
  padding: 10px 14px;
  background: var(--surface-alt);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  flex-wrap: wrap;
}

.shap-legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: var(--fs-small);
  color: var(--text-dim);
}

.legend-box {
  width: 14px;
  height: 14px;
  border-radius: var(--radius-xs);
  flex-shrink: 0;
}
.legend-box--gold { background: var(--gold); }
.legend-box--loss { background: var(--danger); }

.shap-metric-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.shap-metric-row {
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  border: 1px solid transparent;
}

.shap-metric-row:hover {
  background: var(--surface-alt);
  border-color: var(--border);
}

.metric-head {
  display: grid;
  grid-template-columns: 1fr 180px 64px 16px;
  align-items: center;
  gap: 12px;
  width: 100%;
  padding: 0;
  background: none;
  border: 0;
  cursor: pointer;
  text-align: left;
  color: inherit;
  font: inherit;
}

.metric-title {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.metric-label {
  font-size: var(--fs-body-sm);
  font-weight: 650;
  color: var(--text);
}

.metric-tag {
  flex-shrink: 0;
  padding: 1px 7px;
  font-size: var(--fs-micro);
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: var(--text-faint);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 999px;
}

.metric-chevron {
  color: var(--text-faint);
  transition: var(--transition-transform-fast, transform .15s ease);
}

.metric-chevron.is-open { transform: rotate(90deg); }

.metric-read {
  display: flex;
  flex-wrap: wrap;
  gap: 0 6px;
  margin: 4px 0 0;
  font-size: var(--fs-small);
  line-height: 1.5;
  color: var(--text-dim);
}

.metric-cross { color: var(--text); }

.metric-parts {
  margin-top: 8px;
  padding: 10px 12px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}

.metric-definition {
  margin: 0 0 8px;
  font-size: var(--fs-small);
  line-height: 1.5;
  color: var(--text-dim);
}

.metric-part,
.metric-parts-header {
  display: grid;
  grid-template-columns: 150px 1fr 1fr 92px minmax(0, 200px);
  align-items: center;
  gap: 10px;
  padding: 3px 0;
  font-size: var(--fs-label);
  color: var(--text-dim);
  font-variant-numeric: tabular-nums;
}

.part-agg {
  font-weight: 650;
  color: var(--text);
}

.metric-parts-header {
  padding-bottom: 5px;
  margin-bottom: 3px;
  border-bottom: 1px solid var(--border);
  font-size: var(--fs-micro);
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: var(--text-faint);
}

.part-contribution {
  font-weight: 750;
  text-align: right;
}

.part-technical {
  font-size: var(--fs-micro);
  color: var(--text-faint);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.shap-empty {
  padding: 40px 20px;
  line-height: 1.7;
  text-align: center;
  color: var(--text-dim);
  background: var(--panel);
  border: 1px dashed var(--border);
  border-radius: var(--radius);
}

@media (max-width: 860px) {
  .shap-active-card { padding: 16px; }

  .shap-highlights { grid-template-columns: 1fr; }

  /* Libellé et valeur sur la même ligne, barre en dessous : sans placement
     explicite, la valeur tombe sur une troisième ligne à elle seule. */
  .shap-theme-row { grid-template-columns: 1fr auto; }
  .shap-theme-row .theme-label { grid-area: 1 / 1; }
  .shap-theme-row .theme-net { grid-area: 1 / 2; }
  .shap-theme-row .bar-track { grid-area: 2 / 1 / 3 / -1; }

  .metric-head { grid-template-columns: 1fr 64px 16px; }
  .metric-head .bar-track { grid-column: 1 / -1; grid-row: 2; }

    .metric-part,
  .metric-parts-header { grid-template-columns: 1fr 1fr; }
  .part-technical { display: none; }
}
</style>
