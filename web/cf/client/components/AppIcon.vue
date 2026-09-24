<script setup lang="ts">
// Icône au trait (Lucide), dimensionnée en em et teintée par currentColor : elle
// prend la taille et la couleur du texte qui l'entoure, comme un glyphe.
import { ICONS, type IconName } from "../icons";

defineProps<{ name: IconName }>();
</script>

<template>
  <svg
    class="icon"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.5"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
    focusable="false"
    v-html="ICONS[name]"
  />
</template>
