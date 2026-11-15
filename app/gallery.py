"""Página HTML mínima para ver, descargar y borrar los vídeos desde el navegador."""

GALLERY_HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mis vídeos GPX</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font-family: system-ui, sans-serif; background: #11151c; color: #e8ebef; }
  header { padding: 20px 28px; border-bottom: 1px solid #252b36; display: flex; align-items: baseline; gap: 16px; }
  h1 { margin: 0; font-size: 20px; }
  #count { color: #8b94a3; font-size: 14px; }
  main { padding: 24px 28px; display: grid; gap: 20px; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); }
  article { background: #1a2029; border: 1px solid #252b36; border-radius: 12px; overflow: hidden; }
  video { width: 100%; aspect-ratio: 16 / 9; background: #000; display: block; }
  .body { padding: 14px 16px 16px; }
  h2 { margin: 0 0 6px; font-size: 16px; overflow-wrap: anywhere; }
  .meta { color: #8b94a3; font-size: 13px; line-height: 1.6; margin: 0 0 12px; }
  .actions { display: flex; gap: 10px; }
  a.btn, button { font: inherit; font-size: 14px; padding: 7px 14px; border-radius: 8px; cursor: pointer;
                  text-decoration: none; border: 1px solid #343c4a; background: #232a35; color: #e8ebef; }
  a.btn:hover, button:hover { background: #2c3542; }
  button.danger { border-color: #6b2b22; color: #ff8a70; }
  button.danger:hover { background: #3a1e1a; }
  #empty { padding: 60px 28px; color: #8b94a3; text-align: center; }
  #user { margin-left: auto; color: #8b94a3; font-size: 14px; display: flex; gap: 12px; align-items: center; }
  #login { max-width: 360px; margin: 80px auto; padding: 24px; background: #1a2029; border: 1px solid #252b36;
           border-radius: 12px; display: grid; gap: 12px; }
  #login label { font-size: 14px; color: #8b94a3; }
  #login input { font: inherit; padding: 9px 12px; border-radius: 8px; border: 1px solid #343c4a;
                 background: #11151c; color: #e8ebef; }
  #login-error { color: #ff8a70; font-size: 14px; margin: 0; min-height: 1.2em; }
</style>
</head>
<body>
<header><h1>Mis vídeos GPX</h1><span id="count"></span>
  <span id="user" hidden><span id="user-name"></span><button id="logout">Salir</button></span></header>
<form id="login" hidden>
  <label for="key">Clave de API</label>
  <input id="key" type="password" autocomplete="current-password" required>
  <button type="submit">Entrar</button>
  <p id="login-error" role="alert"></p>
</form>
<main id="grid"></main>
<p id="empty" hidden>Todavía no hay vídeos. Sube un GPX a <code>POST /api/v1/render-gpx</code>.</p>
<script>
const api = "api/v1/videos";
const fmtTime = s => { s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
                       return h ? `${h} h ${m} min` : `${m} min`; };
const $ = id => document.getElementById(id);
function showLogin(visible) {
  $("login").hidden = !visible;
  if (visible) { $("grid").replaceChildren(); $("empty").hidden = true; $("user").hidden = true;
                 $("count").textContent = ""; $("key").focus(); }
}
$("login").onsubmit = async e => {
  e.preventDefault();
  const r = await fetch("api/v1/auth/login", { method: "POST", headers: { "Content-Type": "application/json" },
                                               body: JSON.stringify({ api_key: $("key").value }) });
  $("key").value = "";
  if (r.ok) { $("login-error").textContent = ""; load(); } else { $("login-error").textContent = "Clave no válida."; }
};
$("logout").onclick = async () => { await fetch("api/v1/auth/logout", { method: "POST" }); showLogin(true); };
async function load() {
  const me = await fetch("api/v1/auth/me");
  if (me.status === 401) return showLogin(true);
  showLogin(false);
  const user = await me.json();
  $("user-name").textContent = user.name + (user.is_admin ? " (admin)" : "");
  $("user").hidden = user.name === "anonymous";
  const response = await fetch(api + "?limit=200");
  if (response.status === 401) return showLogin(true);
  const data = await response.json();
  const grid = document.getElementById("grid");
  grid.replaceChildren();
  document.getElementById("count").textContent = data.total + (data.total === 1 ? " vídeo" : " vídeos");
  document.getElementById("empty").hidden = data.total > 0;
  for (const v of data.items) {
    const card = document.createElement("article");
    const video = Object.assign(document.createElement("video"),
      { controls: true, preload: "metadata", src: `${api}/${v.video_id}/stream` });
    const body = document.createElement("div"); body.className = "body";
    const title = document.createElement("h2"); title.textContent = v.title;
    const meta = document.createElement("p"); meta.className = "meta";
    const m = v.metrics;
    meta.textContent = (user.is_admin && v.owner ? `${v.owner} · ` : "") +
      `${(m.total_distance_m / 1000).toFixed(1)} km · +${Math.round(m.elevation_gain_m)} m · ` +
      `${fmtTime(m.duration_s)} · ${v.height}p · ${(v.size_bytes / 1e6).toFixed(1)} MB · ` +
      new Date(v.created_at).toLocaleString();
    const actions = document.createElement("div"); actions.className = "actions";
    const dl = Object.assign(document.createElement("a"),
      { className: "btn", href: `${api}/${v.video_id}/download`, textContent: "Descargar" });
    const del = Object.assign(document.createElement("button"), { className: "danger", textContent: "Borrar" });
    del.onclick = async () => {
      if (!confirm(`¿Borrar "${v.title}"? No se puede deshacer.`)) return;
      const r = await fetch(`${api}/${v.video_id}`, { method: "DELETE" });
      if (r.ok) load(); else alert("No se pudo borrar el vídeo.");
    };
    actions.append(dl, del); body.append(title, meta, actions); card.append(video, body); grid.append(card);
  }
}
load();
</script>
</body>
</html>
"""
