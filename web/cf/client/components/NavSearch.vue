<script setup lang="ts">
import AppIcon from "./AppIcon.vue";
import { onBeforeUnmount } from "vue";
import { RIOT_PLATFORMS, useAccountRegistration } from "../account-registration";

const { riotId, platform, submitting, error, submitRegistration } = useAccountRegistration();

let errorTimer: ReturnType<typeof setTimeout> | null = null;

function clearError(): void {
  if (errorTimer) clearTimeout(errorTimer);
  error.value = null;
}

async function submit(): Promise<void> {
  clearError();
  const success = await submitRegistration({
    requireTag: true,
    clearOnSuccess: true,
  });
  if (!success && error.value) {
    if (errorTimer) clearTimeout(errorTimer);
    errorTimer = setTimeout(() => {
      error.value = null;
    }, 6000);
  }
}

onBeforeUnmount(() => {
  if (errorTimer) clearTimeout(errorTimer);
});
</script>

<template>
  <div class="nav-search-container">
    <form class="nav-search-form" @submit.prevent="submit" role="search" aria-label="Find and analyze a player">
      <div class="nav-search-fields">
        <!-- Champ Riot ID -->
        <label class="nav-search-group nav-group-riot" for="nav-riot-id">
          <div class="nav-search-input-wrap">
            <span v-if="submitting" class="loading-spinner-inline nav-search-spinner" aria-hidden="true"></span>
            <svg v-else class="nav-search-icon" viewBox="0 0 20 20" width="13" height="13" fill="currentColor" aria-hidden="true">
              <path fill-rule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clip-rule="evenodd"/>
            </svg>
            <input
              id="nav-riot-id"
              v-model="riotId"
              class="nav-search-input"
              type="text"
              placeholder="Summoner#TAG"
              aria-label="Riot ID in Summoner#TAG format"
              autocomplete="off"
              spellcheck="false"
              required
              @input="error ? clearError() : null"
            >
            <span v-if="riotId.trim() && !submitting" class="nav-search-enter-hint" title="Press Enter to analyze">↵</span>
          </div>
        </label>

        <span class="nav-search-sep" aria-hidden="true"></span>

        <!-- Sélecteur de Serveur -->
        <label class="nav-search-group nav-group-server" for="nav-platform">
          <div class="nav-search-select-wrap">
            <select
              id="nav-platform"
              v-model="platform"
              class="nav-search-select"
              aria-label="Game server"
            >
              <option v-for="p in RIOT_PLATFORMS" :key="p.value" :value="p.value">{{ p.shortLabel }}</option>
            </select>
          </div>
        </label>

        <!-- Bouton submit accessible pour validation au clavier Entrée -->
        <button
          type="submit"
          class="sr-only-submit"
          :disabled="submitting || !riotId.trim()"
          aria-hidden="true"
          tabindex="-1"
        ></button>
      </div>
    </form>

    <!-- Bulle d'erreur contextuelle -->
    <div v-if="error" class="nav-search-error-toast" role="alert">
      <AppIcon name="triangle-alert" class="nav-search-error-icon" />
      <span class="nav-search-error-msg">{{ error }}</span>
      <button type="button" class="nav-search-error-close" aria-label="Dismiss error" @click="clearError">×</button>
    </div>
  </div>
</template>

<style scoped>
.nav-search-container {
  position: relative;
  display: flex;
  align-items: center;
  width: 100%;
}

.nav-search-form {
  margin: 0;
  width: 100%;
}

.nav-search-fields {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 38px;
  width: 100%;
  padding: 3px 8px 3px 12px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  box-shadow: var(--card-shadow);
  transition: all 180ms ease;
}

.nav-search-fields:hover {
  border-color: var(--border-strong);
  background: var(--surface-alt);
  box-shadow: var(--card-shadow-hover);
}

.nav-search-fields:focus-within {
  border-color: var(--gold);
  background: var(--surface-alt);
  box-shadow: var(--focus-ring);
}

.nav-search-group {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0;
  cursor: pointer;
}

.nav-group-riot {
  flex: 1 1 auto;
  min-width: 0;
}

.nav-group-server {
  flex-shrink: 0;
}

.nav-search-input-wrap {
  display: flex;
  align-items: center;
  position: relative;
  width: 100%;
}

.nav-search-icon {
  position: absolute;
  left: 0;
  color: var(--primary);
  opacity: 0.8;
  pointer-events: none;
}

.nav-search-input {
  background: transparent;
  border: none;
  outline: none;
  box-shadow: none;
  color: var(--text);
  font-size: var(--fs-small);
  font-weight: 550;
  padding: 0 0 0 22px;
  height: 30px;
  width: 100%;
  min-width: 80px;
}

.nav-search-input::placeholder {
  color: var(--text-faint);
  font-size: var(--fs-small);
  font-weight: 450;
}

.nav-search-sep {
  display: inline-block;
  width: 1px;
  height: 18px;
  background: var(--border);
  margin: 0 4px;
  flex-shrink: 0;
}

.nav-search-select-wrap {
  display: flex;
  align-items: center;
  position: relative;
}

.nav-search-select {
  background: transparent;
  border: none;
  outline: none;
  box-shadow: none;
  color: var(--text);
  font-size: var(--fs-label);
  font-weight: 600;
  padding: 0 2px;
  height: 30px;
  cursor: pointer;
  max-width: 65px;
}

.nav-search-select option {
  background: var(--surface);
  color: var(--ink);
}

.nav-search-enter-hint {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: var(--fs-label);
  font-weight: 700;
  color: var(--primary);
  background: var(--primary-soft);
  border: 1px solid var(--primary-border);
  border-radius: var(--radius-xs);
  padding: 1px 5px;
  margin-left: 4px;
  line-height: 1;
  pointer-events: none;
  user-select: none;
  animation: navErrorSlideDown 0.15s ease-out;
  flex-shrink: 0;
}

.nav-search-spinner {
  margin-right: 2px;
}

.sr-only-submit {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.nav-search-error-toast {
  position: absolute;
  top: calc(100% + 8px);
  left: 0;
  right: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 7px 11px;
  background: var(--surface);
  border: 1px solid var(--loss-border);
  border-radius: var(--radius-sm);
  color: var(--danger);
  font-size: var(--fs-small);
  font-weight: 550;
  box-shadow: var(--shadow-overlay);
  backdrop-filter: blur(12px);
  animation: navErrorSlideDown 0.18s ease-out;
}

@keyframes navErrorSlideDown {
  from {
    opacity: 0;
    transform: translateY(-4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.nav-search-error-close {
  background: transparent;
  border: none;
  color: var(--danger);
  font-size: var(--fs-lead);
  line-height: 1;
  cursor: pointer;
  padding: 0 4px;
  opacity: 0.7;
  transition: opacity 120ms ease;
}

.nav-search-error-close:hover {
  opacity: 1;
}

@media (max-width: 1100px) {
  .nav-search-select { max-width: 65px; }
}

@media (max-width: 960px) {
  .nav-search-select { max-width: 65px; }
}

@media (max-width: 860px) {
  .nav-search-select { max-width: 60px; }
}

@media (max-width: 640px) {
  .nav-search-fields {
    height: 34px;
    padding: 2px 4px 2px 8px;
    gap: 5px;
  }
  .nav-search-input { font-size: var(--fs-label); }
  .nav-search-select { max-width: 60px; font-size: var(--fs-label); }
}
</style>
