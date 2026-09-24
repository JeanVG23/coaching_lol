<script setup lang="ts">
import AppIcon from "./AppIcon.vue";
import { NEGATIVE_FEEDBACK_TAGS } from "../coaching";

interface FeedbackState {
  useful: boolean;
  tag?: string | null;
}

const props = withDefaults(defineProps<{
  kind: string;
  index: number;
  state?: FeedbackState | null;
  busy?: boolean;
  openKey?: string | null;
  prompt?: boolean;
}>(), {
  state: null,
  busy: false,
  openKey: null,
  prompt: false,
});

const emit = defineEmits<{
  vote: [kind: string, index: number, useful: boolean];
  tag: [kind: string, index: number, tag: string];
}>();

const key = `${props.kind},${props.index}`;
const tagLabels: Record<string, string> = {
  asymetrie: "Hidden information",
  "stat-inventee": "Invented statistic",
  "profondeur-en-faute": "Misread map depth",
  "trop-vague": "Too vague",
  "non-actionnable": "Not actionable",
  autre: "Other",
};
</script>

<template>
  <div :class="prompt ? 'global-focus-actions' : 'fb-compact-row'">
    <span v-if="prompt" class="fb-compact-prompt">Useful?</span>
    <button
      type="button"
      class="fb-btn-compact"
      title="This recommendation is useful"
      aria-label="Useful"
      :class="{ 'active-win': state?.useful === true }"
      :disabled="busy"
      @click="emit('vote', kind, index, true)"
    ><AppIcon name="thumbs-up" /></button>
    <button
      type="button"
      class="fb-btn-compact"
      title="This recommendation is not useful"
      aria-label="Not useful"
      :class="{ 'active-loss': state?.useful === false }"
      :disabled="busy"
      @click="emit('vote', kind, index, false)"
    ><AppIcon name="thumbs-down" /></button>
    <div v-if="openKey === key" class="tag-menu">
      <span class="tag-menu-title">Why is it not useful?</span>
      <button
        v-for="item in NEGATIVE_FEEDBACK_TAGS"
        :key="item"
        type="button"
        class="tag-opt"
        @click="emit('tag', kind, index, item)"
      >{{ tagLabels[item] || item }}</button>
    </div>
  </div>
</template>

<style scoped>
.global-focus-actions {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  position: relative;
}

.fb-compact-row {
  display: flex;
  align-items: center;
  gap: 5px;
  position: relative;
}

.fb-compact-prompt {
  font-size: var(--fs-label);
  color: var(--text-faint);
  margin-right: 2px;
}

.fb-btn-compact {
  padding: 3px 8px;
  font-size: var(--fs-label);
  font-weight: 600;
  color: var(--text-dim);
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: var(--transition-fast);
  display: inline-flex;
  align-items: center;
  gap: 3px;
  font-family: inherit;
}

.fb-btn-compact:hover {
  color: var(--text);
  border-color: var(--primary);
  background: var(--panel-hover);
}

.fb-btn-compact.active-win {
  color: var(--win);
  border-color: var(--win);
  background: var(--win-soft);
}

.fb-btn-compact.active-loss {
  color: var(--loss);
  border-color: var(--loss);
  background: var(--loss-soft);
}

.fb-btn-compact:disabled {
  opacity: .5;
  cursor: default;
}

.tag-menu {
  position: absolute;
  top: 32px;
  right: 0;
  z-index: 30;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 6px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  box-shadow: var(--shadow-overlay);
  min-width: 150px;
}

.tag-menu-title {
  font-size: var(--fs-micro);
  font-weight: 750;
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--text-faint);
  padding: 3px 7px;
}

.tag-opt {
  text-align: left;
  background: none;
  border: none;
  color: var(--text-dim);
  padding: 6px 9px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: var(--fs-label);
  font-family: inherit;
  transition: var(--transition-fast);
}

.tag-opt:hover {
  background: var(--panel-hover);
  color: var(--text);
}
</style>
