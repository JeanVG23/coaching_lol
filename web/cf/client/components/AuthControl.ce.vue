<script setup lang="ts">
import AppIcon from "./AppIcon.vue";
import { onBeforeUnmount, onMounted, ref } from "vue";
import {
  authToken,
  notifyAuthChange,
  openCoachAuth,
  setStoredAuthToken,
  withAuthHeaders,
  type CoachAuthChangeDetail,
} from "../auth";

interface AuthStatusResponse {
  authenticated?: boolean;
}

const authenticated = ref(Boolean(authToken));

function onAuthChange(event: Event): void {
  authenticated.value = Boolean(
    (event as CustomEvent<CoachAuthChangeDetail>).detail?.authenticated,
  );
}

async function checkStatus(): Promise<void> {
  try {
    const response = await fetch("/api/auth/status", { headers: withAuthHeaders() });
    if (!response.ok) return;
    const status = await response.json() as AuthStatusResponse;
    authenticated.value = Boolean(status.authenticated);
    if (!authenticated.value && authToken) setStoredAuthToken(null);
    notifyAuthChange(authenticated.value);
  } catch {
    // Une indisponibilité réseau ne doit pas déconnecter une session locale valide.
  }
}

async function logout(): Promise<void> {
  try {
    await fetch("/api/auth/logout", { method: "POST" });
  } catch {
    // Le jeton local est supprimé même si le endpoint est indisponible.
  }
  setStoredAuthToken(null);
  authenticated.value = false;
  notifyAuthChange(false);
}

onMounted(() => {
  window.addEventListener("coach-auth-change", onAuthChange);
  void checkStatus();
});

onBeforeUnmount(() => {
  window.removeEventListener("coach-auth-change", onAuthChange);
});
</script>

<template>
  <button
    v-if="!authenticated"
    type="button"
    class="topbar-auth-btn"
    title="Sign in to unlock LLM features"
    @click="openCoachAuth()"
  >
    <AppIcon name="lock" class="auth-lock-icon" />
    <span>Coach sign-in</span>
  </button>
  <div v-else class="topbar-auth-logged">
    <span class="topbar-auth-pill" title="Coach mode unlocked">
      <AppIcon name="lock-open" class="auth-unlock-icon" />
      <span>Coach active</span>
    </span>
    <button type="button" class="topbar-logout-btn" title="Sign out" @click="logout">
      <span>Sign out</span>
    </button>
  </div>
</template>

<style scoped>
:host,
auth-control {
  display: contents;
}

.topbar-auth-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: var(--targon-veil-overlay);
  border: 1px solid var(--targon-veil-border);
  border-radius: var(--radius-sm);
  font-size: var(--fs-label);
  font-weight: 600;
  color: var(--targon-veil-text);
  cursor: pointer;
  transition: all 0.2s ease;
}

.topbar-auth-btn:hover {
  background: rgba(0, 0, 0, 0.55);
  border-color: var(--gold);
  color: var(--targon-veil-accent-strong);
}

.topbar-auth-logged {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.topbar-auth-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 10px;
  background: rgba(16, 185, 129, 0.22);
  border: 1px solid rgba(16, 185, 129, 0.45);
  border-radius: var(--radius-sm);
  font-size: var(--fs-label);
  font-weight: 650;
  color: #6ee7b7;
}

.topbar-logout-btn {
  display: inline-flex;
  align-items: center;
  padding: 5px 8px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  font-size: var(--fs-label);
  color: #e6d3cf;
  cursor: pointer;
  transition: all 0.18s ease;
}

.topbar-logout-btn:hover {
  color: #fecdd3;
  background: rgba(244, 63, 94, 0.2);
  border-color: rgba(244, 63, 94, 0.4);
}
</style>
