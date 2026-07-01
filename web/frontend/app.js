// Sonde le backend et affiche le statut. Prouve le lien front -> back.
async function loadHealth() {
  const el = document.getElementById("status");
  try {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    const t = new Date(data.server_time).toLocaleTimeString("fr-FR");
    el.className = "ok";
    el.textContent = `✅ Backend en ligne — ${data.service} (heure serveur : ${t})`;
  } catch (e) {
    el.className = "err";
    el.textContent = "❌ Backend injoignable : " + e.message;
  }
}

loadHealth();