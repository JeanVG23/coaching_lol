<script setup lang="ts">
import AppIcon from "./AppIcon.vue";
import { nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import {
  notifyAuthChange,
  setStoredAuthToken,
  type CoachOpenAuthDetail,
  type PostAuthAction,
} from "../auth";

interface LoginResponse {
  ok?: boolean;
  token?: string;
  detail?: string;
}

const open = ref(false);
const password = ref("");
const error = ref<string | null>(null);
const loading = ref(false);
const passwordInput = ref<HTMLInputElement | null>(null);
let postAuthAction: PostAuthAction | null = null;

function show(event: Event): void {
  const detail = (event as CustomEvent<CoachOpenAuthDetail>).detail;
  postAuthAction = typeof detail?.action === "function" ? detail.action : null;
  password.value = "";
  error.value = null;
  open.value = true;
  void nextTick(() => passwordInput.value?.focus());
}

function close(): void {
  open.value = false;
  password.value = "";
  error.value = null;
  postAuthAction = null;
}

function onKeydown(event: KeyboardEvent): void {
  if (open.value && event.key === "Escape") close();
}

async function login(): Promise<void> {
  const trimmedPassword = password.value.trim();
  if (!trimmedPassword) {
    error.value = "Enter the coach password.";
    return;
  }

  loading.value = true;
  error.value = null;
  try {
    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: trimmedPassword }),
    });
    const data = await response.json().catch(() => ({})) as LoginResponse;
    if (!response.ok || !data.ok || !data.token) {
      throw new Error(data.detail || "Incorrect password.");
    }

    setStoredAuthToken(data.token);
    notifyAuthChange(true);
    const pendingAction = postAuthAction;
    close();
    if (pendingAction) window.setTimeout(pendingAction, 100);
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "Connection error.";
  } finally {
    loading.value = false;
  }
}

onMounted(() => {
  window.addEventListener("coach-open-auth", show);
  window.addEventListener("keydown", onKeydown);
});

onBeforeUnmount(() => {
  window.removeEventListener("coach-open-auth", show);
  window.removeEventListener("keydown", onKeydown);
});
</script>

<template>
  <div v-if="open" class="auth-modal-backdrop" @click.self="close">
    <div class="auth-modal-card" role="dialog" aria-modal="true" aria-labelledby="auth-modal-title">
      <div class="auth-modal-header">
        <div class="auth-modal-header-left">
          <div class="auth-modal-icon-wrap">
            <AppIcon name="key-round" class="auth-modal-icon" />
          </div>
          <div>
            <h3 id="auth-modal-title" class="auth-modal-title">AI Coach Access</h3>
            <p class="auth-modal-sub">A password is required to generate new analyses and use the LLM.</p>
          </div>
        </div>
        <button type="button" class="auth-modal-close" aria-label="Close dialog" @click="close">✕</button>
      </div>

      <form class="auth-modal-form" @submit.prevent="login">
        <div class="auth-field-group">
          <label for="coach-auth-password-input" class="auth-label">Coach password</label>
          <div class="auth-input-wrapper">
            <input
              id="coach-auth-password-input"
              ref="passwordInput"
              v-model="password"
              type="password"
              class="auth-input"
              placeholder="Enter the password…"
              autocomplete="current-password"
              required
            >
          </div>
        </div>

        <div v-if="error" class="auth-error-banner" role="alert">
          <AppIcon name="triangle-alert" class="auth-error-icon" />
          <span>{{ error }}</span>
        </div>

        <div class="auth-modal-actions">
          <button type="button" class="btn btn-secondary auth-cancel-btn" @click="close">Cancel</button>
          <button type="submit" class="btn btn-primary auth-submit-btn" :disabled="loading || !password.trim()">
            <span>{{ loading ? "Checking…" : "Unlock" }}</span>
            <AppIcon v-if="!loading" name="lock-open" />
          </button>
        </div>
      </form>
    </div>
  </div>
</template>

<style scoped>
.auth-modal-backdrop {
  position: fixed;
  inset: 0;
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  background: rgba(20, 23, 24, .34);
  animation: modalFadeIn 0.2s ease-out;
}

.auth-modal-card {
  width: 100%;
  max-width: 440px;
  background: var(--card-marble-bg);
  border: 1px solid var(--card-marble-border);
  border-radius: var(--radius-lg);
  padding: 24px;
  box-shadow:
    var(--shadow-overlay);
  animation: modalSlideUp 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  position: relative;
}

.auth-modal-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 20px;
}

.auth-modal-header-left {
  display: flex;
  align-items: center;
  gap: 14px;
}

.auth-modal-icon-wrap {
  width: 42px;
  height: 42px;
  border-radius: var(--radius);
  background: var(--primary-soft);
  border: 1px solid var(--primary-border);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: var(--fs-title);
  flex-shrink: 0;
}

.auth-modal-title {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.01em;
}

.auth-modal-sub {
  margin: 3px 0 0;
  font-size: var(--fs-body-sm);
  color: var(--text-dim);
  line-height: 1.4;
}

.auth-modal-close {
  background: transparent;
  border: none;
  color: var(--text-faint);
  font-size: 18px;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: var(--radius-sm);
  transition: all 0.15s ease;
  line-height: 1;
}

.auth-modal-close:hover {
  color: var(--text);
  background: var(--surface-alt);
}

.auth-modal-form {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.auth-field-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.auth-label {
  font-size: var(--fs-small);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-dim);
}

.auth-input-wrapper {
  position: relative;
}

.auth-input {
  width: 100%;
  padding: 12px 14px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text);
  font-size: var(--fs-body);
  font-family: inherit;
  transition: all 0.2s ease;
  outline: none;
  box-sizing: border-box;
}

.auth-input:focus {
  border-color: var(--primary);
  background: var(--panel-hover);
  box-shadow: var(--focus-ring);
}

.auth-error-banner {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: var(--loss-soft);
  border: 1px solid var(--loss-border);
  border-radius: var(--radius-sm);
  color: var(--danger);
  font-size: var(--fs-body-sm);
  font-weight: 500;
  animation: bannerShake 0.3s ease-in-out;
}

.auth-error-icon {
  flex-shrink: 0;
  font-size: var(--fs-body);
}

.auth-modal-actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 4px;
}

.auth-cancel-btn {
  padding: 10px 16px;
  font-size: var(--fs-body-sm);
  font-weight: 600;
}

.auth-submit-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  font-size: var(--fs-body-sm);
  font-weight: 650;
}

@keyframes modalFadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes modalSlideUp {
  from { opacity: 0; transform: translateY(12px) scale(0.98); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}

@keyframes bannerShake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-4px); }
  75% { transform: translateX(4px); }
}
</style>
