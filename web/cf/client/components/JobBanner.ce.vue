<script setup lang="ts">
import AppIcon from "./AppIcon.vue";
import { computed } from "vue";

interface CoachingJob {
  type: "coach" | "game-coach";
  status: "running" | "done" | "error";
  progress?: string;
  error?: string;
  matchId?: string;
  notice?: string;
}

const props = defineProps<{
  job?: CoachingJob | null;
}>();

const label = computed(() => {
  if (!props.job) return "";
  if (props.job.type === "game-coach") {
    if (props.job.status === "running") {
      return `Game analysis in progress… ${props.job.progress || ""}`.trim();
    }
    return props.job.status === "done" ? "Game analysis ready" : "Error";
  }
  if (props.job.status === "running") {
    return `Coaching in progress… ${props.job.progress || ""}`.trim();
  }
  return props.job.status === "done" ? "Coaching ready" : "Error";
});
</script>

<template>
  <div v-if="job" class="job-banner" :class="{ err: job.status === 'error' }">
    <AppIcon v-if="job.status !== 'running'" :name="job.status === 'done' ? 'circle-check' : 'circle-x'" /> <span aria-live="polite">{{ label }}</span>
    <span v-if="job.status === 'error' && job.error" class="faint">{{ job.error }}</span>
  </div>
</template>

<style scoped>
.job-banner {
  display: block;
}
</style>
