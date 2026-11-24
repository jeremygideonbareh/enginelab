import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

// ------------------------------------------------------------------ constants
const NAMES = {
  engine_block: "Engine block", cylinder_head: "Cylinder head", cylinder_liner: "Cylinder liner",
  piston: "Piston", piston_ring: "Piston ring", piston_pin: "Piston pin", connecting_rod: "Connecting rod",
  crankshaft: "Crankshaft", camshaft: "Camshaft", valve: "Valve", valve_spring: "Valve spring",
  main_bearing: "Main bearing shell", flywheel: "Flywheel", timing_gear: "Timing gear", bolt: "Bolt",
  oil_pan: "Oil pan", spark_plug: "Spark plug", head_gasket: "Head gasket", valve_cover: "Valve cover",
  other: "Other / accessory", unknown: "Unknown",
  fastener: "Fastener", bearing: "Bearing", linear_motion: "Linear motion", transmission: "Transmission (gears, belts, shafts)",
  motor: "Motor", structure: "Structure (extrusions, plates)", electronics: "Electronics", spring: "Spring",
  seal: "Seal", fluid: "Fluid / pneumatic fitting", custom: "Custom part (made for this design)",
};
const PALETTE = {
  engine_block: "#aab3bd", cylinder_head: "#86a8c8", cylinder_liner: "#9bd18b", piston: "#e07b39",
  piston_ring: "#f2c14e", piston_pin: "#c9a227", connecting_rod: "#d64550", crankshaft: "#e8a87c",
  camshaft: "#8e7cc3", valve: "#4ecdc4", valve_spring: "#c3e88d", main_bearing: "#f78fb3",
  flywheel: "#b5651d", timing_gear: "#a1c4fd", bolt: "#5c6bc0", oil_pan: "#9ba6b1", spark_plug: "#ffd166",
  head_gasket: "#06d6a0", valve_cover: "#8e99a5", other: "#9e9e9e", unknown: "#ff00ff",
  fastener: "#5c6bc0", bearing: "#f78fb3", linear_motion: "#4ecdc4", transmission: "#e8a87c", motor: "#d64550",
  structure: "#9bd18b", electronics: "#ffd166", spring: "#c3e88d", seal: "#06d6a0", fluid: "#a1c4fd", custom: "#8e9aaf",
};
const HOUSINGS = new Set(["engine_block", "cylinder_head", "valve_cover", "oil_pan", "head_gasket", "cylinder_liner"]);
const CAT_COLORS = { steel: "#8fa3b8", stainless: "#cfd8dc", "valve steel": "#90a4ae", superalloy: "#ffb74d",
  titanium: "#b39ddb", "cast iron": "#616161", aluminium: "#e0e0e0", magnesium: "#a5d6a7", polymer: "#4db6ac",
  "bearing alloy": "#f48fb1", "copper alloy": "#d4a373", ceramic: "#fff59d" };

// ------------------------------------------------------------------ state
const S = { asm: null, info: null, materials: [], matByKey: {}, selected: null, meshes: new Map(),
  base: new Map(), center: new THREE.Vector3(), sim: null, fea: null, charts: {}, labelFilter: null };
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" :
  (Math.abs(v) >= 1e6 ? v.toExponential(2) : Number(v).toLocaleString(undefined, { maximumFractionDigits: d }));
const hours = (h) => h === null || h === undefined ? "∞" : (h > 1e6 ? "> 1,000,000 h" : `${fmt(h, 0)} h`);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const HL = "#5c1a36";                       // selection glow (beni pink)
const ICON = (n, cls = "ic") => `<svg class="${cls}"><use href="#i-${n}"/></svg>`;
const skel = (text) => `<div class="skel" aria-busy="true"><div class="skel-cap">${esc(text)}</div><i></i><i></i><i></i></div>`;

function toast(msg, err = false) {
  const life = err ? 7000 : 3500;
  const t = document.createElement("div");
  t.className = "toast" + (err ? " err" : "");
  t.setAttribute("role", err ? "alert" : "status");
  t.style.setProperty("--life", `${life}ms`);
  t.innerHTML = `${ICON(err ? "alert" : "check")}<span>${esc(msg)}</span><button class="x" aria-label="Dismiss">×</button>`;
  const box = $("toasts");
  box.appendChild(t);
  while (box.children.length > 4) box.firstElementChild.remove();
  const close = () => { t.classList.add("out"); setTimeout(() => t.remove(), 220); };
  t.querySelector(".x").onclick = close;
  setTimeout(close, life);
}

// Loading overlay: a title, the pipeline stages (ticked off from the server's progress messages) and elapsed time.
const STAGES = {
  upload: ["Uploading", "Reading the CAD geometry", "Splitting into parts", "Measuring shapes", "Identifying parts", "Specs and standard parts", "Loading the 3D view"],
  demo: ["Generating an engine", "Splitting into parts", "Identifying parts", "Measuring bore, stroke and journals", "Loading the 3D view"],
  sim: ["Solving the engine cycle", "Loading every part", "Running the Monte Carlo"],
};
const STAGE_KEYS = [[/upload/i, 0], [/read|unpack|tessellat/i, 1], [/split/i, 2], [/measuring (the )?shape/i, 3], [/identif/i, 4], [/recognis|specification/i, 5], [/loading the 3d/i, 6]];
const B = { t0: 0, timer: null, stages: null, at: 0 };
function busy(on, text = "Working…", stages = null, title = null) {
  const el = $("busy");
  if (!on) { el.classList.add("hidden"); clearInterval(B.timer); B.timer = null; B.stages = null; B.title = null; return; }
  if (el.classList.contains("hidden")) {
    B.t0 = performance.now(); B.at = 0; B.stages = stages; B.title = title || text;
    clearInterval(B.timer);
    B.timer = setInterval(() => { $("busy-time").textContent = `${Math.round((performance.now() - B.t0) / 1000)} s`; }, 500);
    $("busy-time").textContent = "0 s";
  }
  if (stages) B.stages = stages;
  if (title) B.title = title; else if (!B.stages && !B.keepTitle) B.title = text;
  $("busy-title").textContent = B.title || text;
  $("busy-text").textContent = B.stages ? text : "";
  if (B.stages) {
    if (B.stages === STAGES.upload) for (const [re, i] of STAGE_KEYS) if (re.test(text)) B.at = Math.max(B.at, i);
    $("stages").innerHTML = B.stages.map((s, i) => `<li class="${i < B.at ? "done" : i === B.at ? "now" : ""}">${esc(s)}</li>`).join("");
  } else $("stages").innerHTML = "";
  el.classList.remove("hidden");
}
function busyStage(i) { if (B.stages) { B.at = i; busy(true, $("busy-text").textContent); } }
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.arrayBuffer();
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// ------------------------------------------------------------------ 3D viewer
const viewer = $("viewer");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x000000, 0);          // transparent: the soft sakura gradient behind shows through
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewer.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 1, 100000);
camera.position.set(600, 400, 600);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08;
scene.add(new THREE.HemisphereLight(0xfff6f8, 0x8a7f86, 1.25));
const sun = new THREE.DirectionalLight(0xffffff, 1.55); sun.position.set(1, 2, 1.5); scene.add(sun);
const fill = new THREE.DirectionalLight(0xffe9ef, 0.55); fill.position.set(-1.5, -0.5, -1); scene.add(fill);
const rim = new THREE.DirectionalLight(0xffffff, 0.35); rim.position.set(-0.5, 1, -2); scene.add(rim);
const root = new THREE.Group(); scene.add(root);

function resize() {
  const w = viewer.clientWidth, h = viewer.clientHeight;
  renderer.setSize(w, h); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewer);
(function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();

function fit(onlySelected = false) {
  const box = new THREE.Box3();
  if (onlySelected && S.highlightSet && S.highlightSet.size) {
    for (const id of S.highlightSet) { const m = S.meshes.get(id); if (m) box.expandByObject(m); }
  } else {
    const target = onlySelected && S.selected !== null ? S.meshes.get(S.selected) : root;
    box.setFromObject(target || root);
  }
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3()).length();
  const c = box.getCenter(new THREE.Vector3());
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const dist = (size / 2) / Math.tan(fov / 2) / Math.min(1, camera.aspect) * 1.38;
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(0.75, 0.55, 0.9).normalize().multiplyScalar(dist));
  camera.near = size / 200; camera.far = size * 20; camera.updateProjectionMatrix();
}
$("fit-btn").onclick = () => fit($("isolate").checked);
$("shot-btn").onclick = () => {
  if (!S.asm) { toast("Load a model first"); return; }
  renderer.render(scene, camera);
  // composite the transparent render over a plain light background so the PNG isn't see-through
  const src = renderer.domElement, c = document.createElement("canvas");
  c.width = src.width; c.height = src.height;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(c.width / 2, c.height * 0.2, 0, c.width / 2, c.height * 0.2, Math.max(c.width, c.height));
  grad.addColorStop(0, "#ffffff"); grad.addColorStop(1, "#f3ecef");
  g.fillStyle = grad; g.fillRect(0, 0, c.width, c.height); g.drawImage(src, 0, 0);
  const a = document.createElement("a");
  a.href = c.toDataURL("image/png"); a.download = `${(S.asm.filename || "model").replace(/\.[^.]+$/, "")}_view.png`; a.click();
  toast("Picture saved");
};
$("full-btn").onclick = () => {
  const el = $("viewer-wrap");
  if (document.fullscreenElement) document.exitFullscreen(); else el.requestFullscreen?.().catch(() => toast("Full screen isn't available here", true));
};
function togglePanel(force) {
  const hide = force ?? !$("layout").classList.contains("panel-hidden");
  $("layout").classList.toggle("panel-hidden", hide);
  $("panel-btn").classList.toggle("on", hide);
  setTimeout(() => { resize(); moveInk(); }, 20);
}
$("panel-btn").onclick = () => togglePanel();

async function loadModel(asm) {
  root.clear(); S.meshes.clear(); S.base.clear();
  const buf = await api(`/api/assembly/${asm.id}/model.glb`);
  const gltf = await new GLTFLoader().parseAsync(buf, "");
  gltf.scene.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(gltf.scene);
  box.getCenter(S.center);
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    const m = /part_(\d+)/.exec(o.name) || /part_(\d+)/.exec(o.parent?.name || "");
    if (!m) return;
    const id = +m[1];
    const mesh = new THREE.Mesh(o.geometry, new THREE.MeshStandardMaterial({ metalness: 0.35, roughness: 0.55 }));
    o.getWorldPosition(mesh.position); o.getWorldQuaternion(mesh.quaternion);
    mesh.userData.id = id;
    mesh.geometry.computeVertexNormals();
    root.add(mesh);
    S.meshes.set(id, mesh);
    S.base.set(id, mesh.position.clone());
  });
  $("empty").classList.add("hidden"); Petals.stop(); $("viewer-tools").classList.remove("off");
  resize(); applyColors(); applyExplode(); fit();
}

function partColor(p) {
  const mode = $("color-mode").value;
  if (mode === "type") return PALETTE[p.label] || "#999";
  if (mode === "confidence") return new THREE.Color().setHSL(0.33 * Math.max(0, (p.confidence - 0.4) / 0.6), 0.75, 0.5);
  if (mode === "material") return CAT_COLORS[S.matByKey[p.material]?.category] || "#999";
  if (mode === "cad") return p.color ? new THREE.Color(p.color[0], p.color[1], p.color[2]) : "#a7b0ba";
  if (mode === "risk") {
    const r = S.sim?.parts?.find((q) => q.id === p.id);
    if (!r || !r.modelled) return "#c9c0c5";
    if (r.service_item) return "#5c7cfa";
    const horizon = S.sim.hours;
    const b10 = r.b10_hours ?? 1e12;
    const x = Math.max(0, Math.min(1, Math.log10(b10 / horizon + 1e-9) / 1.5 + 0.5));
    return new THREE.Color().setHSL(0.33 * x, 0.85, 0.5);
  }
  return "#c9c0c5";
}

function applyColors() {
  if (!S.asm) return;
  const mode = $("color-mode").value;
  const xray = $("xray").checked, iso = $("isolate").checked;
  for (const p of S.asm.parts) {
    const mesh = S.meshes.get(p.id); if (!mesh) continue;
    const mat = mesh.material;
    const isFea = mode === "fea" && S.fea && S.fea.part === p.id;
    mat.vertexColors = isFea;
    if (isFea) {
      const vs = S.fea.vertex_stress; const max = Math.max(...vs) || 1;
      const cols = new Float32Array(mesh.geometry.attributes.position.count * 3);
      for (let i = 0; i < cols.length / 3; i++) {
        const t = Math.min(1, (vs[i] ?? 0) / max);
        const c = new THREE.Color().setHSL(0.66 * (1 - t), 0.9, 0.5);
        cols.set([c.r, c.g, c.b], i * 3);
      }
      mesh.geometry.setAttribute("color", new THREE.BufferAttribute(cols, 3));
      mat.color.set("#ffffff");
    } else {
      mat.color.set(partColor(p));
    }
    const dim = (mode === "fea" && S.fea && !isFea);
    const lit = S.selected === p.id || (S.highlightSet && S.highlightSet.has(p.id));
    const housing = xray && (HOUSINGS.has(p.label) || (S.asm.mode === "general" && p.label === "structure")) && !lit;
    const focus = S.highlightSet ? S.highlightSet.has(p.id) : S.selected === p.id;
    const hide = iso && (S.selected !== null || S.highlightSet) && !focus &&
      !(S.labelFilter && p.label === S.labelFilter);
    mat.transparent = housing || dim;
    mat.opacity = housing ? 0.12 : dim ? 0.15 : 1;
    mat.depthWrite = !(housing || dim);
    mesh.visible = !hide;
    mat.emissive.set(lit ? HL : "#000000");
    mat.needsUpdate = true;
  }
  drawLegend();
}

function applyExplode() {
  const k = +$("explode").value;
  for (const [id, mesh] of S.meshes) {
    const p = S.asm.parts[id];
    const c = new THREE.Vector3(...p.center);
    const dir = c.clone().sub(S.center);
    mesh.position.copy(S.base.get(id)).addScaledVector(dir, k);
  }
}

function drawLegend() {
  const el = $("legend"); const mode = $("color-mode").value;
  if (!S.asm) { el.innerHTML = ""; return; }
  drawLegendBody(el, mode);
  const title = { type: "Part types", cad: "CAD colours", confidence: "Confidence", material: "Materials", risk: "Life risk", fea: "FEA stress" }[mode];
  if (S.legendClosed === undefined) S.legendClosed = $("viewer-wrap").clientWidth < 760;
  el.innerHTML = `<button class="legend-head" aria-expanded="${!S.legendClosed}">${title}${ICON("chevron")}</button><div class="legend-body">${el.innerHTML}</div>`;
  el.classList.toggle("closed", S.legendClosed);
  el.querySelector(".legend-head").onclick = () => { S.legendClosed = !S.legendClosed; el.classList.toggle("closed", S.legendClosed); };
  el.querySelectorAll(".row[data-label]").forEach((r) => r.onclick = () => selectLabel(r.dataset.label));
}
function drawLegendBody(el, mode) {
  if (mode === "type") {
    const counts = S.asm.counts;
    el.innerHTML = Object.keys(counts).sort().map((l) =>
      `<div class="row" data-label="${l}"><span class="sw" style="background:${PALETTE[l] || "#999"}"></span>${NAMES[l] || l} <span class="muted">×${counts[l]}</span></div>`).join("");
  } else if (mode === "cad") {
    el.innerHTML = `<div class="muted">grey = no colour set in the file</div>`;
  } else if (mode === "confidence") {
    el.innerHTML = `<div class="bar" style="background:linear-gradient(90deg,hsl(0,75%,50%),hsl(60,75%,50%),hsl(120,75%,50%))"></div><span class="muted">≤40%</span> <span style="float:right" class="muted">100%</span>`;
  } else if (mode === "material") {
    const cats = [...new Set(S.asm.parts.map((p) => S.matByKey[p.material]?.category))];
    el.innerHTML = cats.map((c) => `<div class="row"><span class="sw" style="background:${CAT_COLORS[c] || "#999"}"></span>${c}</div>`).join("");
  } else if (mode === "risk") {
    el.innerHTML = S.sim ? `B10 life vs ${fmt(S.sim.hours, 0)} h horizon<div class="bar" style="background:linear-gradient(90deg,hsl(0,85%,50%),hsl(60,85%,50%),hsl(120,85%,50%))"></div><span class="muted">fails early</span><span style="float:right" class="muted">long life</span><div class="muted">grey = no load model · blue = service item</div>`
      : `<span class="muted">Run a life simulation first.</span>`;
  } else if (mode === "fea") {
    el.innerHTML = S.fea ? `von Mises stress (MPa)<div class="bar" style="background:linear-gradient(90deg,hsl(240,90%,50%),hsl(120,90%,50%),hsl(0,90%,50%))"></div><span class="muted">0</span><span style="float:right" class="muted">${fmt(Math.max(...S.fea.vertex_stress), 0)}</span>`
      : `<span class="muted">Run FEA on a part first (Part tab).</span>`;
  }
}

// Selection by clicking.
const ray = new THREE.Raycaster(); const mouse = new THREE.Vector2(); let downAt = null;
renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
  const r = renderer.domElement.getBoundingClientRect();
  mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hits = ray.intersectObjects([...S.meshes.values()].filter((m) => m.visible && !(m.material.transparent && m.material.opacity < 0.5)));
  if (hits.length) selectPart(hits[0].object.userData.id); else selectPart(null);
});

["color-mode", "xray"].forEach((id) => $(id).addEventListener("change", applyColors));
$("isolate").addEventListener("change", () => { applyColors(); fit($("isolate").checked); });
$("explode").addEventListener("input", applyExplode);

// ------------------------------------------------------------------ tabs
const TABS = ["overview", "parts", "part", "sim", "opt"];
function moveInk() {
  const b = document.querySelector(".tabs button.active"), ink = $("tab-ink");
  if (!b || !ink) return;
  ink.style.width = `${b.offsetWidth - 16}px`;
  ink.style.transform = `translateX(${b.offsetLeft + 8}px)`;
}
function showTab(name) {
  const cur = document.querySelector(".tabs button.active")?.dataset.tab;
  document.querySelectorAll(".tabs button").forEach((b) => { b.classList.toggle("active", b.dataset.tab === name); b.setAttribute("aria-selected", b.dataset.tab === name); });
  document.querySelectorAll(".tab-body").forEach((t) => {
    const on = t.id === `tab-${name}`;
    t.classList.toggle("hidden", !on);
    if (on && cur !== name) { t.classList.remove("enter"); void t.offsetWidth; t.classList.add("enter"); }
  });
  moveInk();
  if (name === "opt" && S.charts.opt1) setTimeout(() => Object.values(S.charts).forEach((c) => c.resize()), 30);
}
document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => showTab(b.dataset.tab));
new ResizeObserver(moveInk).observe(document.querySelector(".tabs"));

// ------------------------------------------------------------------ loading
async function setAssembly(asm) {
  S.asm = asm; S.sim = null; S.fea = null; S.selected = null; S.labelFilter = null;
  S.highlightSet = null; S.partsView = null; S.partsQuery = ""; S.sourcing = null;
  $("selinfo").classList.add("hidden");
  if ($("color-mode").value === "risk" || $("color-mode").value === "fea") $("color-mode").value = "type";
  $("isolate").checked = false; $("explode").value = 0;
  await loadModel(asm);
  renderOverview(); renderParts(); renderSim(); renderOpt();
  $("tab-part").innerHTML = `<div class="empty-state">${ICON("part", "es-mark")}<p>Click a part in the 3D view or the parts list to inspect, test and improve it.</p></div>`;
  showTab("overview");
  $("crumb").hidden = false;
  $("crumb-file").textContent = asm.filename; $("crumb-file").title = asm.filename;
  $("crumb-mode").textContent = asm.mode === "engine" ? "Engine" : "Assembly";
  document.title = `${asm.filename} · EngineLab`;
  remember(asm);
  if (window.matchMedia("(max-width: 900px)").matches === false) togglePanel(false);
}

async function waitJob(job) {
  for (;;) {
    const s = await api(`/api/jobs/${job}`);
    if (s.status === "done") return s.result;
    if (s.status === "error") throw new Error(s.error);
    busy(true, s.message);
    await new Promise((r) => setTimeout(r, 900));
  }
}

async function finishLoad(id, title, doneMsg) {
  busy(true, "Loading the 3D view", null, title);
  busyStage(B.stages ? B.stages.length - 1 : 0);
  await setAssembly(await api(`/api/assembly/${id}`));
  toast(doneMsg);
}

let working = false;
async function guarded(fn) {
  if (working) { toast("Still working on the last request…"); return; }
  working = true; $("start-pop").hidePopover?.();
  try { await fn(); } catch (e) { toast(e.message, true); } finally { working = false; busy(false); }
}

function upload(file) {
  return guarded(async () => {
    const fd = new FormData();
    fd.append("file", file); fd.append("units", $("units").value);
    busy(true, `Uploading ${file.name} (${fmt(file.size / 1e6, 1)} MB)`, STAGES.upload, `Analysing ${file.name}`);
    const { job } = await api("/api/analyze", { method: "POST", body: fd });
    busyStage(1);
    const id = await waitJob(job);
    await finishLoad(id, `Analysing ${file.name}`, `${file.name} analysed`);
  });
}
$("file-input").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) upload(f); };
{ // drag and drop anywhere on the viewer
  const wrap = $("viewer-wrap"); let depth = 0;
  wrap.addEventListener("dragenter", (e) => { e.preventDefault(); depth++; $("dropzone").hidden = false; });
  wrap.addEventListener("dragover", (e) => e.preventDefault());
  wrap.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; $("dropzone").hidden = true; } });
  wrap.addEventListener("drop", (e) => { e.preventDefault(); depth = 0; $("dropzone").hidden = true; const f = e.dataTransfer.files[0]; if (f) upload(f); });
}

function openSample(key) {
  const s = S.samples?.find((x) => x.key === key);
  if (!s) { toast("Pick a sample first"); return; }
  return guarded(async () => {
    busy(true, "Starting", STAGES.upload, `Opening ${s.title.split("(")[0].trim()}`);
    busyStage(1);
    const { job } = await post(`/api/samples/${key}`, {});
    const id = await waitJob(job);
    await finishLoad(id, `Opening ${s.title.split("(")[0].trim()}`, "Sample analysed");
  });
}
$("sample-btn").onclick = () => openSample($("sample-select").value);

function runDemo() {
  return guarded(async () => {
    const body = { seed: +$("demo-seed").value || 0 };
    if ($("demo-layout").value) body.layout = $("demo-layout").value;
    busy(true, "Exporting it as one unnamed mesh, then analysing it like an upload", STAGES.demo, "Building a demo engine");
    const steps = setInterval(() => { if (B.stages === STAGES.demo && B.at < 3) busyStage(B.at + 1); }, 2200);
    try { await setAssembly(await post("/api/demo", body)); } finally { clearInterval(steps); }
    toast(`Demo engine #${body.seed} analysed`);
  });
}
$("demo-btn").onclick = runDemo;
$("w-demo").onclick = runDemo;
$("w-sample").onclick = () => openSample(S.samples?.find((s) => s.key === "koch")?.key || S.samples?.[0]?.key);

// ------------------------------------------------------------------ recent analyses (this browser only)
function recents() { try { return JSON.parse(localStorage.getItem("enginelab.recent") || "[]"); } catch (_) { return []; } }
function remember(a) {
  const list = recents().filter((r) => r.id !== a.id);
  list.unshift({ id: a.id, filename: a.filename, mode: a.mode, parts: a.parts.length, t: Date.now() });
  try { localStorage.setItem("enginelab.recent", JSON.stringify(list.slice(0, 6))); } catch (_) {}
  renderRecents();
}
function ago(t) {
  const m = Math.round((Date.now() - t) / 60000);
  return m < 1 ? "just now" : m < 60 ? `${m} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} d ago`;
}
function renderRecents() {
  const list = recents();
  const row = (r) => `<button class="menu-item" data-recent="${esc(r.id)}">${ICON(r.mode === "engine" ? "engine" : "part")}
    <span><b>${esc(r.filename)}</b><small>${r.parts} parts · ${r.mode === "engine" ? "engine" : "assembly"}</small></span><span class="meta">${ago(r.t)}</span></button>`;
  $("recent-sec").hidden = !list.length;
  $("recent-list").innerHTML = list.map(row).join("");
  $("w-recent").hidden = !list.length;
  $("w-recent").innerHTML = list.length ? `<h4>Recent</h4>${list.slice(0, 3).map(row).join("")}` : "";
  document.querySelectorAll("[data-recent]").forEach((b) => b.onclick = () => guarded(async () => {
    busy(true, "Reopening", null, "Reopening the analysis");
    try { await setAssembly(await api(`/api/assembly/${b.dataset.recent}`)); }
    catch (e) {
      try { localStorage.setItem("enginelab.recent", JSON.stringify(recents().filter((r) => r.id !== b.dataset.recent))); } catch (_) {}
      renderRecents();
      throw new Error("That analysis is no longer on the server (it keeps recent uploads only). Please upload the file again.");
    }
  }));
}

function renderSamples(samples) {
  S.samples = samples;
  $("sample-select").innerHTML = `<option value="">real CAD samples…</option>` + samples.map((s) => `<option value="${s.key}">${esc(s.title)}</option>`).join("");
  $("sample-list").innerHTML = samples.length ? samples.map((s) => {
    const m = /^(.*?)\s*\((.*)\)\s*$/.exec(s.title) || [null, s.title, ""];
    return `<button class="menu-item" data-sample="${s.key}">${ICON("robot")}<span><b>${esc(m[1])}</b><small>${esc(m[2])}</small></span></button>`;
  }).join("") : `<p class="pop-note">No samples on this server.</p>`;
  $("sample-list").querySelectorAll("[data-sample]").forEach((b) => b.onclick = () => openSample(b.dataset.sample));
  const k = samples.find((s) => s.key === "koch") || samples[0];
  if (k) $("w-sample-sub").textContent = (/^(.*?)\s*\(/.exec(k.title)?.[1] || k.title);
  else $("w-sample").hidden = true;
}

// ------------------------------------------------------------------ falling petals (idle welcome screen only)
const Petals = (() => {
  const cv = $("petals"), g = cv.getContext("2d");
  let raf = null, ps = [], w = 0, h = 0;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  function size() { const r = cv.getBoundingClientRect(); w = cv.width = r.width * devicePixelRatio; h = cv.height = r.height * devicePixelRatio; }
  function spawn(y) {
    return { x: Math.random() * w, y: y ?? -20, s: (6 + Math.random() * 7) * devicePixelRatio, vy: (0.25 + Math.random() * 0.45) * devicePixelRatio,
      vx: (0.15 + Math.random() * 0.35) * devicePixelRatio, a: Math.random() * 6.28, va: (Math.random() - 0.5) * 0.02,
      ph: Math.random() * 6.28, o: 0.35 + Math.random() * 0.4 };
  }
  function petal(p) {
    g.save(); g.translate(p.x, p.y); g.rotate(p.a); g.scale(1, 0.6 + 0.4 * Math.sin(p.ph));
    g.globalAlpha = p.o; g.fillStyle = "#f2b8c8";
    g.beginPath(); g.moveTo(0, p.s); g.bezierCurveTo(-p.s, p.s * 0.2, -p.s * 0.6, -p.s, -p.s * 0.18, -p.s * 0.9);
    g.lineTo(0, -p.s * 0.7); g.lineTo(p.s * 0.18, -p.s * 0.9); g.bezierCurveTo(p.s * 0.6, -p.s, p.s, p.s * 0.2, 0, p.s); g.fill(); g.restore();
  }
  function frame() {
    g.clearRect(0, 0, w, h);
    for (const p of ps) {
      p.ph += 0.03; p.y += p.vy; p.x += p.vx + Math.sin(p.ph) * 0.4; p.a += p.va;
      if (p.y > h + 20 || p.x > w + 20) Object.assign(p, spawn());
      petal(p);
    }
    raf = requestAnimationFrame(frame);
  }
  function start(force = false) {
    if ((reduce.matches && !force) || raf) return;
    size(); ps = Array.from({ length: Math.round(Math.min(26, w / (60 * devicePixelRatio))) }, () => spawn(Math.random() * h));
    raf = requestAnimationFrame(frame);
  }
  function stop() { cancelAnimationFrame(raf); raf = null; g.clearRect(0, 0, w, h); }
  new ResizeObserver(() => { if (raf) size(); }).observe(cv);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); else if (!S.asm) start(); });
  return { start, stop };
})();
window.EngineLabPetals = Petals;   // for testing with reduced motion on

// ------------------------------------------------------------------ overview
function specTable(obj, skip = []) {
  const rows = Object.entries(obj || {}).filter(([k, v]) => !skip.includes(k) && v !== null && typeof v !== "object");
  return `<table>${rows.map(([k, v]) => `<tr><td>${esc(k.replace(/_/g, " "))}</td><td class="num">${typeof v === "number" ? fmt(v, 3) : esc(v)}</td></tr>`).join("")}</table>`;
}

function renderOverview() {
  const a = S.asm; const e = a.engine || {};
  const t = a.timings || {}; const total = Object.values(t).reduce((x, y) => x + y, 0);
  const nParts = a.parts.length;
  const unknown = a.parts.filter((p) => p.label === "unknown").length;
  const avgConf = a.parts.reduce((s, p) => s + p.confidence, 0) / Math.max(nParts, 1);
  let html = `
    <div class="panel-title">${esc(a.filename)}</div>
    <p class="note">${a.mode === "engine" ? "Engine" : "Assembly"} · units ${esc(a.units)}</p>
    <div class="cards">
      <div class="card"><div class="k">Parts found</div><div class="v">${nParts}</div></div>
      <div class="card"><div class="k">Part types</div><div class="v">${Object.keys(a.counts).length}</div></div>
      <div class="card"><div class="k">Mean confidence</div><div class="v">${fmt(avgConf * 100, 0)}<small>%</small></div></div>
    </div>
    ${(() => { const unsure = a.parts.filter((p) => p.confidence < 0.75 && p.source !== "name" && p.source !== "user");
      return unsure.length ? `<div class="warn-box"><b>${unsure.length} part${unsure.length === 1 ? "" : "s"} to check:</b> the AI is unsure what ${unsure.length === 1 ? "it is" : "they are"} (confidence below 75%). Colour by "identification confidence" to see them, and correct the type in the Part tab — every correction teaches the AI. <button class="btn small" id="show-unsure">Highlight them</button></div>` : ""; })()}
    ${a.warnings.map((w) => `<div class="warn-box">${esc(w)}</div>`).join("")}
    ${a.mode === "general" ? generalOverview(a) : ""}
    ${a.mode === "general" ? "" : "<h3>Engine (measured from the geometry)</h3>"}
    ${e.cylinders ? `<div class="cards">
      <div class="card"><div class="k">Layout</div><div class="v">${esc(e.layout || "—")}</div></div>
      <div class="card"><div class="k">Displacement</div><div class="v">${fmt(e.displacement_cc, 0)}<small> cc</small></div></div>
      <div class="card"><div class="k">Bore × stroke</div><div class="v">${fmt(e.bore_mm, 1)}<small> × </small>${fmt(e.stroke_mm, 1)}<small> mm</small></div></div>
    </div>${specTable(e, ["layout", "displacement_cc"])}` : (a.mode === "general" ? "" : `<p class="muted">No pistons identified, so this is treated as a general assembly. Parts can still be tested individually.</p>`)}`;
  if (a.demo_truth) {
    const d = a.demo_truth;
    const found = a.counts;
    const rows = Object.keys({ ...d.label_counts, ...found }).sort().map((l) => {
      const tr = d.label_counts[l] || 0, fd = found[l] || 0;
      return `<tr><td><span class="swatch" style="background:${PALETTE[l]}"></span>${NAMES[l] || l}</td><td class="num">${tr}</td><td class="num">${fd}</td><td>${tr === fd ? '<span class="pill good">match</span>' : '<span class="pill bad">differs</span>'}</td></tr>`;
    }).join("");
    html += `<h3>Demo check: known answer vs analysis</h3>
      <table><tr><th></th><th class="num">true</th><th class="num">measured</th><th></th></tr>
      <tr><td>Layout</td><td class="num">${esc(d.layout)}${d.cylinders}</td><td class="num">${esc(e.layout || "—")}</td><td></td></tr>
      <tr><td>Bore (mm)</td><td class="num">${fmt(d.bore_mm, 2)}</td><td class="num">${fmt(e.bore_mm, 2)}</td><td>${errPill(e.bore_mm, d.bore_mm)}</td></tr>
      <tr><td>Stroke (mm)</td><td class="num">${fmt(d.stroke_mm, 2)}</td><td class="num">${fmt(e.stroke_mm, 2)}</td><td>${errPill(e.stroke_mm, d.stroke_mm)}</td></tr>
      <tr><td>Rod length (mm)</td><td class="num">${fmt(d.rod_length_mm, 2)}</td><td class="num">${fmt(e.rod_length_mm, 2)}</td><td>${errPill(e.rod_length_mm, d.rod_length_mm)}</td></tr>
      <tr><td>Displacement (cc)</td><td class="num">${fmt(d.displacement_cc, 0)}</td><td class="num">${fmt(e.displacement_cc, 0)}</td><td>${errPill(e.displacement_cc, d.displacement_cc)}</td></tr>
      <tr><td>Parts</td><td class="num">${d.part_count}</td><td class="num">${nParts}</td><td>${d.part_count === nParts ? '<span class="pill good">match</span>' : '<span class="pill bad">differs</span>'}</td></tr>
      </table>
      <details><summary>Part counts by type</summary><table><tr><th>type</th><th class="num">true</th><th class="num">found</th><th></th></tr>${rows}</table></details>`;
  }
  html += `<h3>Processing</h3><p class="muted">${Object.entries(t).map(([k, v]) => `${k.replace("_s", "")} ${fmt(v, 2)} s`).join(" · ")} — total ${fmt(total, 1)} s · units: ${esc(a.units)}</p>`;
  if (unknown) html += `<p class="muted">${unknown} part(s) are unknown — open them in the Part tab and set their type.</p>`;
  if (a.mode === "engine" && e.cylinders) html += `<h3>Balance and crankshaft vibration</h3><div id="balance-out">${skel("Analysing balance and torsional vibration")}</div>`;
  $("tab-overview").innerHTML = html;
  if (a.mode === "engine" && e.cylinders) loadBalance(a);
  if ($("show-unsure")) $("show-unsure").onclick = () => highlight(a.parts.filter((p) => p.confidence < 0.75 && p.source !== "name" && p.source !== "user").map((p) => p.id), "parts to check");
  if (a.mode === "general" && $("robot-torques")) { loadTorques(a); $("rt-btn").onclick = () => loadTorques(a); }
  $("tab-overview").querySelectorAll("tr.click[data-label]").forEach((r) => r.onclick = () => selectLabel(r.dataset.label));
  const bb = $("bulk-btn");
  if (bb) bb.onclick = async () => {
    busy(true, "Updating materials…");
    try {
      const r = await post(`/api/assembly/${a.id}/materials/bulk`, { label: $("bulk-label").value, key: $("bulk-mat").value });
      S.asm = { ...r, demo_truth: S.asm.demo_truth };
      renderOverview(); renderParts(); applyColors();
      toast(`${r.changed} parts updated`);
    } catch (e) { toast(e.message, true); } finally { busy(false); }
  };
  const ub = $("urdf-btn");
  if (ub) ub.onclick = async () => {
    busy(true, "Building URDF (links, exact inertias, meshes)…");
    try {
      const r = await fetch(`/api/assembly/${a.id}/robot/urdf`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url; link.download = a.filename.replace(/\.[^.]+$/, "") + "_urdf.zip"; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      toast("URDF downloaded");
    } catch (e) { toast(e.message, true); } finally { busy(false); }
  };
}
function treeStats(n) {
  let subs = 0, depth = 0;
  (function walk(x, d) { depth = Math.max(depth, d); for (const c of x.children) { subs++; walk(c, d + 1); } })(n, 1);
  return { subs, depth };
}

function generalOverview(a) {
  const bom = a.bom || [];
  const std = bom.filter((l) => l.standard_part);
  const mass = bom.reduce((s, l) => s + (l.total_mass_kg || 0), 0);
  const ts = a.tree ? treeStats(a.tree) : null;
  const byLabel = {};
  for (const l of bom) { const k = l.label; byLabel[k] ||= { lines: 0, qty: 0 }; byLabel[k].lines++; byLabel[k].qty += l.qty; }
  return `<div class="cards">
      <div class="card"><div class="k">Unique parts</div><div class="v">${bom.length}</div></div>
      <div class="card"><div class="k">Standard / bought</div><div class="v">${std.reduce((s, l) => s + l.qty, 0)}<small> pcs, ${std.length} types</small></div></div>
      <div class="card"><div class="k">Estimated mass</div><div class="v">${fmt(mass, 2)}<small> kg</small></div></div>
    </div>
    ${ts ? `<p class="muted">Assembly structure kept from the file: ${ts.subs} sub-assemblies, ${ts.depth} levels deep. Open the Parts tab for the tree and the parts list with supplier links.</p>` : ""}
    <div class="row-actions">
      <a class="btn" href="/api/assembly/${a.id}/bom.csv" download>Download parts list (CSV)</a>
      <button class="btn" id="urdf-btn" title="Robot description for ROS / Gazebo / MuJoCo / Isaac Sim">Export URDF (robot)</button>
    </div>
    <p class="note">URDF: one link per top-level sub-assembly, exact mass / centre of mass / inertia from the CAD, joints suggested only where bearings, shafts or motors connect links (check them before simulating).</p>
    ${(a.counts.motor || 0) > 0 ? `<h3>Robot joints: torque vs motors</h3>
      <div class="form"><label>Payload at the tip (kg)<input id="rt-payload" type="number" step="0.05" value="0.2"></label>
        <label>Joint acceleration (rad/s²)<input id="rt-accel" type="number" step="1" value="5"></label></div>
      <button class="btn small" id="rt-btn">Recalculate</button><div id="robot-torques"></div>` : ""}
    <h3>What it is made of</h3>
    <table><tr><th>type</th><th class="num">unique</th><th class="num">pieces</th></tr>
    ${Object.entries(byLabel).sort((x, y) => y[1].qty - x[1].qty).map(([k, v]) => `<tr class="click" data-label="${k}"><td><span class="swatch" style="background:${PALETTE[k] || "#999"}"></span>${NAMES[k] || k}</td><td class="num">${v.lines}</td><td class="num">${v.qty}</td></tr>`).join("")}
    </table><p class="note">Masses use the materials shown in the parts list (CAD files rarely carry materials); change them per part, or for a whole type here:</p>
    <div class="row-actions">
      <select id="bulk-label">${Object.keys(byLabel).map((k) => `<option value="${k}" ${k === "custom" ? "selected" : ""}>${NAMES[k] || k}</option>`).join("")}</select>
      <select id="bulk-mat" style="flex:1">${S.materials.map((m) => `<option value="${m.key}" ${m.key === "petg" ? "selected" : ""}>${esc(m.name)}</option>`).join("")}</select>
      <button class="btn small" id="bulk-btn">Apply to all</button>
    </div>`;
}

function errPill(m, t) {
  if (m === undefined || m === null || !t) return "";
  const e = Math.abs(m - t) / t * 100;
  return `<span class="pill ${e < 1 ? "good" : e < 5 ? "warn" : "bad"}">${fmt(e, 2)}%</span>`;
}

// ------------------------------------------------------------------ parts list
function renderParts() {
  const a = S.asm;
  S.partsView ||= a.tree ? "tree" : "bom";
  const views = [["bom", "Parts list (BOM)"], ...(a.tree ? [["tree", "Assembly tree"]] : []), ["type", "By type"], ["buy", "Buy & cost"]];
  const head = `<div class="row-actions" style="align-items:center">
      ${views.map(([k, t]) => `<button class="btn small ${S.partsView === k ? "primary" : ""}" data-view="${k}">${t}</button>`).join("")}
      <input id="part-search" type="text" placeholder="search parts…" value="${esc(S.partsQuery || "")}" style="flex:1;min-width:120px">
    </div>`;
  let body = "";
  if (S.partsView === "bom") body = bomView(a);
  else if (S.partsView === "buy") body = buyView(a);
  else if (S.partsView === "tree") body = treeView(a);
  else body = typeView(a);
  $("tab-parts").innerHTML = head + body;
  $("tab-parts").querySelectorAll("button[data-view]").forEach((b) => b.onclick = () => { S.partsView = b.dataset.view; renderParts(); });
  const sb = $("part-search");
  sb.oninput = () => { S.partsQuery = sb.value; clearTimeout(sb._t); sb._t = setTimeout(() => { renderParts(); const n = $("part-search"); n.focus(); n.setSelectionRange(n.value.length, n.value.length); }, 250); };
  $("tab-parts").querySelectorAll("tr.click[data-id]").forEach((r) => r.onclick = (ev) => { if (ev.target.closest("a")) return; selectPart(+r.dataset.id); });
  $("tab-parts").querySelectorAll("[data-ids]").forEach((el) => el.onclick = (ev) => {
    if (ev.target.closest("a")) return;
    const ids = el.dataset.ids.split(",").map(Number);
    highlight(ids, el.dataset.title || "");
  });
  $("tab-parts").querySelectorAll("summary[data-label]").forEach((s) => s.onclick = () => selectLabel(s.dataset.label, false));
}

// ------------------------------------------------------------------ robot joint torques
async function loadTorques(a) {
  const el = () => $("robot-torques");
  if (!el()) return;
  const payload = +($("rt-payload")?.value || 0.2), accel = +($("rt-accel")?.value || 5);
  el().innerHTML = skel("Working out joint torques");
  try {
    const r = await api(`/api/assembly/${a.id}/robot/torques?payload_kg=${payload}&accel=${accel}`);
    if (!el() || S.asm?.id !== a.id) return;
    if (!r.joints.length) { el().innerHTML = `<p class="muted">${esc(r.note)}</p>`; return; }
    const pill = (v) => !v ? "" : `<span class="pill ${v === "OK" ? "good" : v.startsWith("needs") || v === "marginal" ? "warn" : "bad"}">${esc(v)}</span>`;
    el().innerHTML = `
      ${!r.structure_ok ? `<div class="warn-box">The CAD's sub-assemblies don't map neatly onto arm links (e.g. one sub-assembly per servo), so link masses and joints are approximate.</div>` : ""}
      ${r.custom_parts && r.custom_printed < r.custom_parts / 2 ? `<div class="warn-box">${r.custom_parts} custom parts are assumed metal. If they are 3D printed, <button class="btn small" id="rt-printed">make them PETG</button> — torques fall with mass.</div>` : ""}
      <table><tr><th>Joint</th><th class="num">moving mass</th><th class="num">worst hold</th><th class="num">+ accel</th><th>Motor at the joint</th></tr>
      ${r.joints.map((j) => `<tr><td>${esc(j.joint)}<div class="note">axis ${j.axis.map((x) => fmt(x, 2)).join(", ")}${j.gravity_factor < 0.1 ? " · gravity doesn't load it" : ""}</div></td>
        <td class="num">${fmt(j.moving_mass_kg, 2)} kg</td><td class="num">${fmt(j.hold_worst_Nm, 2)} N·m</td><td class="num">${fmt(j.accel_Nm, 2)}</td>
        <td>${j.motor ? `${esc(j.motor.type)} (${fmt(j.motor.rated_Nm, 2)} N·m) ${pill(j.motor.verdict)}` : '<span class="muted">none recognised</span>'}</td></tr>`).join("")}</table>
      <p class="note">${esc(r.note)} Joint axes are suggested from the bearings and shafts at each joint — check them.</p>`;
    if ($("rt-printed")) $("rt-printed").onclick = async () => {
      busy(true, "Updating materials…");
      try { S.asm = await post(`/api/assembly/${a.id}/materials/bulk`, { label: "custom", key: "petg" }); renderParts(); renderOverview(); applyColors(); toast("Custom parts set to PETG"); }
      catch (e) { toast(e.message, true); } finally { busy(false); }
    };
  } catch (e) { if (el()) el().innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
}

// ------------------------------------------------------------------ engine balance + torsional vibration
async function loadBalance(a) {
  const out = () => $("balance-out");
  try {
    const r = await api(`/api/assembly/${a.id}/balance`);
    if (!out() || S.asm?.id !== a.id) return;
    const kN = (v) => `${fmt(v / 1000, 1)} kN`, kNm = (v) => `${fmt(v / 1000, 2)} kN·m`;
    const std = r.standard_phasing;
    const tv = r.torsional;
    const crit = tv.critical_speeds.filter((c) => c.major);
    out().innerHTML = `
      <p class="note">Crank throws measured from the crankshaft: ${r.phasing_deg.map((x) => `${x}°`).join(" · ")} (${r.throws.length} throws). Forces at ${fmt(r.rpm, 0)} rpm; reciprocating mass ${fmt(r.reciprocating_mass_kg, 3)} kg per cylinder.</p>
      <table><tr><th>Shaking (reciprocating masses)</th><th class="num">1st order</th><th class="num">2nd order</th>${std ? `<th class="num">standard ${std.phasing_deg.join("-")}</th>` : ""}</tr>
        <tr><td>Force</td><td class="num">${kN(r.primary_force_N)}</td><td class="num">${kN(r.secondary_force_N)}</td>${std ? `<td class="num">${kN(std.primary_force_N)} / ${kN(std.secondary_force_N)}</td>` : ""}</tr>
        <tr><td>Rocking couple</td><td class="num">${r.primary_couple_Nm > 500 ? `<span class="pill bad">${kNm(r.primary_couple_Nm)}</span>` : kNm(r.primary_couple_Nm)}</td><td class="num">${kNm(r.secondary_couple_Nm)}</td>${std ? `<td class="num">${kNm(std.primary_couple_Nm)} / ${kNm(std.secondary_couple_Nm)}</td>` : ""}</tr>
        <tr><td>Rotating unbalance (before counterweights)</td><td class="num">${kN(r.rotating_force_N)}</td><td class="num">${kNm(r.rotating_couple_Nm)} couple</td>${std ? "<td></td>" : ""}</tr></table>
      ${std ? `<div class="warn-box">The crank throws are not in the standard ${std.phasing_deg.join("-")} order. With the standard order the 1st-order rocking couple would be ${kNm(std.primary_couple_Nm)} instead of ${kNm(r.primary_couple_Nm)} — re-phase the throws, or add a balance shaft.</div>` : ""}
      <p class="note"><b>Torsional vibration:</b> first natural frequencies ${tv.natural_frequencies_hz.map((f) => `${fmt(f, 0)} Hz`).join(", ")} (${esc(tv.flywheel)}). ${crit.length ? `Resonances with the main firing orders inside the speed range: ${crit.map((c) => `order ${c.order} at ${fmt(c.rpm, 0)} rpm`).join(", ")} — check with a torsional damper study.` : "No main-order resonance inside the speed range."}</p>`;
  } catch (e) { if (out()) out().innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
}

// ------------------------------------------------------------------ sourcing (prices by country)
function money(v, sym) { return v === undefined || v === null ? "—" : `${sym}${Number(v).toLocaleString(undefined, { maximumFractionDigits: v < 10 ? 2 : 0 })}`; }

function buyView(a) {
  const key = `${a.id}|${S.region}|${a.parts.map((p) => p.material).join(",").length}`;
  if (!S.sourcing || S.sourcing.key !== key) {
    if (!S.sourcingLoading) {
      S.sourcingLoading = true;
      api(`/api/assembly/${a.id}/sourcing?region=${S.region}`).then((r) => { S.sourcing = { key, r }; })
        .catch((e) => toast(e.message, true)).finally(() => { S.sourcingLoading = false; if (S.partsView === "buy") renderParts(); });
    }
    return skel(`Pricing the parts list for ${$("region").selectedOptions[0]?.textContent || S.region}`);
  }
  const r = S.sourcing.r, t = r.totals, sym = r.symbol;
  const rows = r.lines.filter((l) => matches(`${l.name} ${l.basis} ${l.family}`));
  return `
    <div class="cards">
      <div class="card"><div class="k">Bought parts</div><div class="v" style="font-size:15px">${money(t.standard_low, sym)}–${money(t.standard_high, sym)}</div><div class="note">${t.priced_lines} lines priced</div></div>
      <div class="card"><div class="k">Made parts</div><div class="v" style="font-size:15px">${money(t.custom_low, sym)}–${money(t.custom_high, sym)}</div><div class="note">3D print / CNC estimate</div></div>
      <div class="card"><div class="k">Delivery + import</div><div class="v" style="font-size:15px">~${money(t.shipping_estimate + t.import_tax_estimate, sym)}</div><div class="note">${r.plan.length} shop${r.plan.length === 1 ? "" : "s"}</div></div>
    </div>
    <p class="note">${esc(r.note)} Currency ${r.currency} at ${fmt(r.fx_per_usd, 2)} per USD${r.fx_live ? " (today's rate)" : " (fallback rate)"}.${r.live_prices.length ? ` Live prices: ${r.live_prices.join(", ")}.` : ""}</p>
    <h3>Shopping plan — fewest shops for ${esc(r.region_name)}</h3>
    <table>${r.plan.map((p) => `<tr><td><b>${esc(p.shop)}</b>${p.import ? ' <span class="pill warn">ships from abroad</span>' : ' <span class="pill good">local</span>'}</td><td class="num">${p.items.length} item${p.items.length === 1 ? "" : "s"}</td></tr>`).join("")}</table>
    <h3>Every line</h3>
    <table><tr><th>Item</th><th class="num">qty</th><th class="num">each</th><th class="num">line</th></tr>
    ${rows.map((l) => `<tr class="click" data-ids="${(l.part_ids || []).join(",")}" data-title="${esc(l.name || "")}">
      <td><b>${esc(l.name || "")}</b>${l.includes ? ` <span class="pill">bought as one (${l.includes} modelled pieces)</span>` : ""}<div class="note">${esc(l.basis || "no typical price for this kind of part")}</div>
        <div class="note">${(l.shops || []).slice(0, 4).map((s) => `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.shop)}</a>${s.import && l.standard_part ? "*" : ""}`).join(" · ")}</div></td>
      <td class="num">${l.qty}</td><td class="num">${l.unit_low !== undefined ? `${money(l.unit_low, sym)}–${money(l.unit_high, sym)}` : "—"}</td>
      <td class="num">${l.total_low !== undefined ? `${money(l.total_low, sym)}–${money(l.total_high, sym)}` : "—"}</td></tr>`).join("")}</table>
    <p class="note">* ships from abroad (shipping and import tax extra). Click a line to highlight it in 3D. Prices are typical ranges; the links show today's real prices. Made parts are priced by their material: plastic as 3D printing, metal as CNC machining — set the right materials (Overview) for a realistic estimate.</p>`;
}

function initRegion(list) {
  const guess = (navigator.language || "").split("-")[1]?.toUpperCase();
  let saved = null;
  try { saved = localStorage.getItem("enginelab.region"); } catch (_) {}
  S.region = saved || (list.some((r) => r.code === guess) ? guess : "OTHER");
  $("region").innerHTML = list.map((r) => `<option value="${r.code}" ${r.code === S.region ? "selected" : ""}>${esc(r.name)} (${r.currency})</option>`).join("");
  $("region").onchange = () => {
    S.region = $("region").value;
    try { localStorage.setItem("enginelab.region", S.region); } catch (_) {}
    S.sourcing = null;
    if (S.asm && S.partsView === "buy") renderParts();
  };
}

function matches(text) {
  const q = (S.partsQuery || "").trim().toLowerCase();
  return !q || String(text).toLowerCase().includes(q);
}

function linkRow(links) {
  return Object.entries(links || {}).map(([k, u]) => `<a href="${esc(u)}" target="_blank" rel="noopener noreferrer">${esc(k)}</a>`).join(" · ");
}

function bomView(a) {
  const rows = (a.bom || []).filter((l) => matches(`${l.name} ${l.category} ${l.designation} ${l.path} ${l.material}`));
  const total = rows.reduce((s, l) => s + l.qty, 0);
  return `<p class="muted">${rows.length} unique parts, ${total} pieces. Click a row to highlight it in 3D.
      <a href="/api/assembly/${a.id}/bom.csv" download>Download parts list (CSV)</a></p>
    <table><tr><th class="num">qty</th><th>part</th><th class="num">mass</th></tr>
    ${rows.map((l) => `<tr class="click" data-ids="${l.part_ids.join(",")}" data-title="${esc(l.name)}">
      <td class="num">${l.qty}</td>
      <td><span class="swatch" style="background:${PALETTE[l.label] || "#999"}"></span><b>${esc(l.name)}</b>
        ${l.standard_part ? '<span class="pill good">standard</span>' : '<span class="pill">custom</span>'}
        <div class="note">${esc(l.designation || l.category)} · ${esc(l.material)} · ${l.dims_mm.map((d) => fmt(d, 1)).join(" × ")} mm</div>
        ${l.standard_part && Object.keys(l.links || {}).length ? `<div class="note">Find it: ${linkRow(l.links)}</div>` : ""}
      </td>
      <td class="num">${l.unit_mass_kg !== null && l.unit_mass_kg !== undefined ? fmt(l.unit_mass_kg * 1000, 1) + " g" : "—"}</td></tr>`).join("")}
    </table>`;
}

function subtreeIds(n) {
  const out = [...n.parts];
  for (const c of n.children) out.push(...subtreeIds(c));
  return out;
}

function treeView(a) {
  const q = (S.partsQuery || "").trim();
  const node = (n, depth) => {
    const ids = subtreeIds(n);
    const parts = n.parts.map((i) => a.parts[i]).filter((p) => matches(`${p.display_name} ${p.component?.designation || ""} ${n.name}`));
    const kids = n.children.map((c) => node(c, depth + 1)).filter(Boolean);
    if (q && !parts.length && !kids.length) return "";
    const open = depth < 2 || q ? "open" : "";
    return `<details ${open} style="margin-left:${depth ? 8 : 0}px"><summary><span data-ids="${ids.join(",")}" data-title="${esc(n.name)}" style="cursor:pointer"><b>${esc(n.name)}</b> <span class="muted">${ids.length} parts</span></span></summary>
      ${kids.join("")}
      ${parts.length ? `<table>${parts.map((p) => `<tr class="click ${S.selected === p.id ? "sel" : ""}" data-id="${p.id}"><td><span class="swatch" style="background:${PALETTE[p.label] || "#999"}"></span>${esc(p.display_name || p.name || "#" + p.id)}</td><td class="note">${esc(p.component?.designation || p.component?.category || p.specs.kind || "")}</td></tr>`).join("")}</table>` : ""}
    </details>`;
  };
  return `<p class="muted">The file's own assembly structure. Click a sub-assembly name to highlight all of its parts.</p>` + node(a.tree, 0);
}

function highlight(ids, title) {
  S.highlightSet = new Set(ids);
  S.selected = ids.length === 1 ? ids[0] : null;
  S.labelFilter = null;
  applyColors();
  if ($("isolate").checked) fit(true);
  showMassProps(ids, title);
}

async function showMassProps(ids, title) {
  const el = $("selinfo");
  if (!S.asm || !ids || !ids.length) { el.classList.add("hidden"); return; }
  try {
    const m = await api(`/api/assembly/${S.asm.id}/massprops?ids=${ids.join(",")}`);
    const I = m.inertia_kgmm2;
    el.innerHTML = `<b>${esc(title || "Selection")}</b> · ${m.parts} part${m.parts > 1 ? "s" : ""}<br>
      mass <b>${m.mass_kg < 1 ? fmt(m.mass_kg * 1000, 1) + " g" : fmt(m.mass_kg, 3) + " kg"}</b>
      ${m.com_mm ? `<br>centre of mass (mm) ${m.com_mm.map((x) => fmt(x, 1)).join(", ")}` : ""}
      ${I ? `<br>inertia about COM (kg·mm²): Ixx ${fmt(I[0][0], 1)} · Iyy ${fmt(I[1][1], 1)} · Izz ${fmt(I[2][2], 1)}
      <br><span class="muted">Ixy ${fmt(I[0][1], 1)} · Ixz ${fmt(I[0][2], 1)} · Iyz ${fmt(I[1][2], 1)}</span>` : ""}
      <div class="muted">uses the materials in the parts list</div>`;
    el.classList.remove("hidden");
  } catch (e) { el.classList.add("hidden"); }
}

function typeView(a) {
  const groups = {};
  for (const p of a.parts) if (matches(`${p.display_name} ${p.label} ${p.component?.designation || ""}`)) (groups[p.label] ||= []).push(p);
  const order = Object.keys(groups).sort((x, y) => (NAMES[x] || x).localeCompare(NAMES[y] || y));
  return `<p class="muted">${a.parts.length} parts in ${order.length} types. Click a type to highlight it, or a part to open it.</p>` +
    order.map((l) => {
      const ps = groups[l];
      const conf = ps.reduce((s, p) => s + p.confidence, 0) / ps.length;
      return `<details ${S.labelFilter === l ? "open" : ""}><summary data-label="${l}"><span class="swatch" style="background:${PALETTE[l]}"></span><b>${NAMES[l] || l}</b> ×${ps.length} <span class="pill ${conf > 0.8 ? "good" : conf > 0.5 ? "warn" : "bad"}">${fmt(conf * 100, 0)}%</span></summary>
        <table>${ps.map((p) => `<tr class="click ${S.selected === p.id ? "sel" : ""}" data-id="${p.id}"><td>#${p.id}${p.name ? ` <span class="muted">${esc(p.name)}</span>` : ""}</td><td>${esc(S.matByKey[p.material]?.name || p.material)}</td><td class="num">${fmt(p.specs.mass_kg, 3)} kg</td></tr>`).join("")}</table></details>`;
    }).join("");
}

function selectLabel(label, toggleIso = true) {
  S.labelFilter = S.labelFilter === label ? null : label;
  const first = S.asm.parts.find((p) => p.label === label);
  if (first) { S.selected = first.id; }
  for (const p of S.asm.parts) {
    const m = S.meshes.get(p.id); if (!m) continue;
    m.material.emissive.set(S.labelFilter && p.label === label ? HL : "#000000");
  }
  if (toggleIso) applyColors();
  else { for (const p of S.asm.parts) { const m = S.meshes.get(p.id); if (m) m.material.emissive.set(S.labelFilter && p.label === label ? HL : "#000000"); } }
}

// ------------------------------------------------------------------ part panel
function stepPart(d) {
  if (!S.asm) return;
  const n = S.asm.parts.length;
  const cur = S.selected ?? (d > 0 ? -1 : 0);
  selectPart(((cur + d) % n + n) % n);
  if ($("isolate").checked) fit(true);
}
function selectPart(id) {
  S.selected = id; S.labelFilter = null; S.highlightSet = null;
  applyColors();
  if (id === null) { $("selinfo").classList.add("hidden"); return; }
  showMassProps([id], S.asm.parts[id]?.display_name || `${NAMES[S.asm.parts[id]?.label] || "Part"} #${id}`);
  renderPart(); showTab("part");
  renderParts();
}

function renderPart() {
  const p = S.asm.parts[S.selected]; if (!p) return;
  const labels = S.asm.mode === "general" ? (S.info?.general_labels || []) : (S.info?.labels || Object.keys(NAMES));
  const copies = S.asm.parts.filter((q) => q.group === p.group && q.label === p.label).length;
  const c = p.component;
  const compCard = c ? `<div class="info-box">
      <b>${esc(c.designation || c.category)}</b> ${c.standard_part ? '<span class="pill good">standard / bought part</span>' : '<span class="pill">made for this design</span>'}
      <div class="note">${esc(c.category)} · recognised from its ${c.source === "name" ? "name" : "exact geometry"} (${fmt(c.confidence * 100, 0)}%)${c.size?.bought_assembly ? ` · part of bought assembly “${esc(c.size.bought_assembly)}”` : ""}</div>
      ${c.standard_part && Object.keys(c.links || {}).length ? `<div style="margin-top:6px">Find / buy / download CAD: ${linkRow(c.links)}</div>` : ""}
    </div>` : `<p class="muted">Made for this design${p.specs.kind ? ` (${esc(p.specs.kind)})` : ""}: not a standard catalogue part.</p>`;
  const crumbs = p.path && p.path.length ? `<p class="note">${p.path.map(esc).join(" › ")} › <b>${esc(p.display_name || "")}</b> · used ${p.quantity}× in the assembly</p>` : "";
  const mats = S.materials;
  const simPart = S.sim?.parts?.find((q) => q.id === p.id);
  $("tab-part").innerHTML = `
    <div class="part-head">
      <div><div class="panel-title"><span class="swatch" style="background:${PALETTE[p.label]}"></span>${p.display_name ? esc(p.display_name) : NAMES[p.label] || p.label}</div>
        <p class="note">#${p.id} · ${esc(NAMES[p.label] || p.label)}${p.subtype ? ` · <span class="pill">${esc(p.subtype)}</span>` : ""}</p></div>
      <div class="row-actions">
        <button class="icon-btn" id="part-prev" title="Previous part ([)" aria-label="Previous part">‹</button>
        <button class="icon-btn" id="part-next" title="Next part (])" aria-label="Next part">›</button>
        <button class="icon-btn" id="copy-specs" title="Copy the specifications" aria-label="Copy the specifications">${ICON("copy")}</button>
      </div>
    </div>
    ${crumbs}${compCard}
    <div class="cards">
      <div class="card"><div class="k">Confidence</div><div class="v">${fmt(p.confidence * 100, 0)}<small>%</small></div>${p.confidence < 0.75 && p.source !== "name" && p.source !== "user" ? '<div class="note" style="color:var(--warn)">unsure — please check</div>' : ""}</div>
      <div class="card"><div class="k">Identified by</div><div class="v" style="font-size:14px">${p.source === "name" ? "file name" : p.source === "user" ? "you" : p.source === "assembly" ? "assembly logic" : p.source === "learned model" ? "learned model (geometry)" : p.source === "geometry" ? "exact geometry rules" : "geometry model"}</div></div>
      <div class="card"><div class="k">Identical copies</div><div class="v">${copies}</div></div>
    </div>
    <p class="muted">Other candidates: ${p.alternatives.map(([l, c]) => `${NAMES[l] || l} ${fmt(c * 100, 0)}%`).join(", ")}</p>
    <div class="row-actions"><select id="relabel">${labels.map((l) => `<option value="${l}" ${l === p.label ? "selected" : ""}>${NAMES[l] || l}</option>`).join("")}</select>
      <button class="btn small" id="relabel-btn">Set type</button></div>

    <h3>Specifications</h3>${specTable(p.specs, ["watertight"])}

    <h3>Material</h3>
    <div class="row-actions"><select id="mat-select" style="flex:1">${materialOptions(p.material, p.label)}</select>
      <button class="btn small primary" id="mat-btn">Apply${copies > 1 ? ` to all ${copies}` : ""}</button></div>
    <div id="mat-props"></div>
    <div id="mat-preview"></div>
    <details><summary>Rank every suitable material for this part</summary>
      <div class="form"><label>Duty for ranking<select id="rec-duty">${dutyOptions()}</select></label>
        <label>Target life (h)<input id="rec-target" type="number" value="10000"></label></div>
      <button class="btn" id="rec-btn">Recommend materials (runs the physics for each)</button>
      <div id="rec-out"></div></details>

    <h3>How to make this part better</h3>
    ${(S.asm.design_changes || []).length ? `<div class="good-box"><b>Changes applied to this design</b>
      <ul class="reasons">${S.asm.design_changes.map((d) => `<li>${esc(d.part)}: ${esc(d.what.toLowerCase())} → ${esc(d.change)}</li>`).join("")}</ul>
      <button class="btn small" id="design-reset">Back to the uploaded design</button></div>` : ""}
    <div id="improve-out">${S.asm.mode === "engine" ? skel("Simulating every possible change to this part") : '<p class="muted">' + "Design changes are simulated for engines; for this part use the material menu and the FEA test below.</p>"}</div>

    ${simPart ? `<h3>Life simulation result</h3>${simPartTable(simPart)}` : ""}

    <h3>Test bench — engine loads at chosen conditions</h3>
    <div class="form">
      <label>Engine speed (rpm)<input id="tb-rpm" type="number" placeholder="75% of max"></label>
      <label>Load (0–1.2)<input id="tb-load" type="number" step="0.05" value="1"></label>
      <label>Peak cylinder pressure (bar)<input id="tb-p" type="number" placeholder="from engine settings"></label>
      <label>Metal temperature (°C)<input id="tb-t" type="number" placeholder="typical for this part"></label>
    </div>
    <button class="btn" id="tb-btn">Run test + pressure sweep</button>
    <div id="tb-out"></div>

    <h3>FEA — any force, torque or pressure</h3>
    <div class="form">
      <label>Load case<select id="fea-case"><option value="axial">axial (push/pull)</option><option value="bending">bending (cantilever)</option><option value="torsion">torsion</option><option value="pressure">pressure on end face</option></select></label>
      <label>Magnitude<input id="fea-mag" type="number" value="10000"></label>
      <label>Axis<select id="fea-axis"><option value="auto">auto (symmetry / longest)</option><option value="long">longest</option><option value="short">shortest</option><option value="x">X</option><option value="y">Y</option><option value="z">Z</option></select></label>
      <label>Temperature (°C)<input id="fea-t" type="number" value="20"></label>
    </div>
    <p class="note">Units: N for axial/bending, N·m for torsion, MPa for pressure. One end is fixed, the load goes on the other end.</p>
    <button class="btn" id="fea-btn">Run FEA</button>
    <div id="fea-out"></div>`;
  $("relabel-btn").onclick = async () => {
    busy(true, "Updating…");
    try { S.asm = await post(`/api/assembly/${S.asm.id}/part/${p.id}/label`, { label: $("relabel").value }); renderOverview(); renderParts(); renderPart(); applyColors();
      toast("Thanks — saved as a training example for the AI (shape numbers only, no geometry or names)"); }
    catch (e) { toast(e.message, true); } finally { busy(false); }
  };
  $("mat-btn").onclick = async () => {
    busy(true, "Updating…");
    try { S.asm = await post(`/api/assembly/${S.asm.id}/part/${p.id}/material`, { key: $("mat-select").value, all_copies: true }); S.sim = null; renderParts(); renderPart(); applyColors(); toast("Material applied — re-run the life simulation to see the effect"); }
    catch (e) { toast(e.message, true); } finally { busy(false); }
  };
  $("copy-specs").onclick = async () => {
    const lines = [`${p.display_name || NAMES[p.label] || p.label} (#${p.id}) — ${NAMES[p.label] || p.label}`,
      `Material: ${S.matByKey[p.material]?.name || p.material}`,
      ...Object.entries(p.specs || {}).filter(([k, v]) => v !== null && typeof v !== "object" && k !== "watertight")
        .map(([k, v]) => `${k.replace(/_/g, " ")}: ${typeof v === "number" ? fmt(v, 3) : v}`)];
    try { await navigator.clipboard.writeText(lines.join("\n")); toast("Specifications copied"); }
    catch (_) { toast("Copying isn't allowed in this browser", true); }
  };
  $("part-prev").onclick = () => stepPart(-1);
  $("part-next").onclick = () => stepPart(1);
  $("mat-select").onchange = () => materialPreview(p);
  materialPreview(p);
  if ($("design-reset")) $("design-reset").onclick = () => applyDesign({ reset: true }, "Back to the uploaded design");
  if (S.asm.mode === "engine") improvePart(p);
  $("rec-btn").onclick = () => recommend(p);
  $("tb-btn").onclick = () => testBench(p);
  $("fea-btn").onclick = () => runFea(p);
}

// ------------------------------------------------------------------ material menu + design improvement
function materialOptions(current) {
  const groups = {};
  for (const m of S.materials) (groups[m.category] = groups[m.category] || []).push(m);
  return Object.entries(groups).map(([cat, ms]) => `<optgroup label="${esc(cat)}">${ms.map((m) =>
    `<option value="${m.key}" ${m.key === current ? "selected" : ""}>${esc(m.name)}${m.key === current ? " (current)" : ""}</option>`).join("")}</optgroup>`).join("");
}

function matPropsRow(m, ref) {
  const d = (k, unit, dig = 0) => {
    const v = m[k], r = ref ? ref[k] : null;
    const ch = ref && r && v !== r ? ` <span class="${v > r ? "up" : "down"}">${v > r ? "▲" : "▼"}${fmt(Math.abs(v / r - 1) * 100, 0)}%</span>` : "";
    return `<td class="num">${fmt(v, dig)} ${unit}${ch}</td>`;
  };
  return `<tr><td>${esc(m.name)}</td>${d("density", "", 0)}${d("yield_strength", "")}${d("endurance", "")}${d("max_temp", "°C")}${d("cost", "", 1)}</tr>`;
}

async function materialPreview(p) {
  const key = $("mat-select").value, cur = S.matByKey[p.material], m = S.matByKey[key];
  if (!m) return;
  $("mat-props").innerHTML = `<table><tr><th></th><th class="num">kg/m³</th><th class="num">yield MPa</th><th class="num">fatigue MPa</th><th class="num">max</th><th class="num">cost</th></tr>
    ${key !== p.material ? matPropsRow(cur) : ""}${matPropsRow(m, key !== p.material ? cur : null)}</table>
    ${m.notes ? `<p class="note">${esc(m.notes)}</p>` : ""}`;
  const out = $("mat-preview");
  if (key === p.material) { out.innerHTML = ""; return; }
  const massNew = (p.specs.mass_kg || 0) * m.density / cur.density;
  if (S.asm.mode !== "engine") {
    out.innerHTML = `<p class="note">Mass per part ${fmt(p.specs.mass_kg, 3)} → <b>${fmt(massNew, 3)} kg</b>. Use the FEA test below to check strength with the new material.</p>`;
    return;
  }
  out.innerHTML = skel(`Simulating the engine with ${m.name}`);
  try {
    const r = await post(`/api/assembly/${S.asm.id}/whatif`, { design: { [`${p.label}.material`]: key } });
    if ($("mat-select").value !== key) return;            // the user picked another one meanwhile
    out.innerHTML = effectTable(r.before, r.after, p.label) +
      `<p class="note">Preview only — click Apply to use it. Lives are nominal (median) values from the engine physics; the Monte Carlo life simulation adds scatter.</p>`;
  } catch (e) { out.innerHTML = `<div class="warn-box">${esc(e.message)}</div>`; }
}

function effectTable(b, a, label) {
  const row = (name, x, y, f, better = (u, v) => v > u) => {
    const cls = x === y || x === null || y === null ? "" : better(x, y) ? "good" : "bad";
    return `<tr><td>${name}</td><td class="num">${f(x)}</td><td class="num"><span class="pill ${cls}">${f(y)}</span></td></tr>`;
  };
  const pb = b.parts[label], pa = a.parts[label];
  let html = `<table><tr><th>Effect</th><th class="num">now</th><th class="num">with change</th></tr>`;
  if (pb && pa) {
    html += row(`${esc(pb.name)} life`, pb.life_h, pa.life_h, hours);
    html += row(`${esc(pb.name)} safety (fatigue)`, pb.sf_fatigue, pa.sf_fatigue, (v) => fmt(v, 2));
    html += row(`${esc(pb.name)} mass (all)`, pb.mass_kg, pa.mass_kg, (v) => `${fmt(v, 3)} kg`, (u, v) => v < u);
  }
  html += row("Engine life (weakest part)", b.system_life_h, a.system_life_h, hours);
  for (const [lab, q] of Object.entries(a.parts)) {
    const o = b.parts[lab];
    if (lab === label || !o || !o.sf_fatigue || !q.sf_fatigue || Math.abs(q.sf_fatigue - o.sf_fatigue) < 0.05) continue;
    html += row(`${esc(q.name)} safety`, o.sf_fatigue, q.sf_fatigue, (v) => fmt(v, 2));
  }
  return html + `</table>`;
}

async function improvePart(p) {
  const out = $("improve-out");
  try {
    const r = await api(`/api/assembly/${S.asm.id}/part/${p.id}/improve`);
    if (S.selected !== p.id) return;
    if (!r.supported) { out.innerHTML = `<p class="muted">${esc(r.message)}</p>`; return; }
    const kindPill = { stronger: "good", lighter: "good", "trade-off": "bad" };
    out.innerHTML = `<p>${esc(r.status)}</p>` + (r.items.length ? r.items.map((it, i) => `
      <div class="rec" style="--i:${i}">
        <div class="rec-head"><span class="pill ${kindPill[it.kind] || ""}">${esc(it.kind)}</span> <b>${esc(it.what)}: ${esc(it.change)}</b></div>
        ${it.why ? `<div class="note">${esc(it.why)}</div>` : ""}
        <div class="rec-fx">
          ${it.part_life_ratio && Math.abs(it.part_life_ratio - 1) > 0.02 ? `<span>part life <b>×${fmt(it.part_life_ratio, 2)}</b></span>` : ""}
          ${it.part_sf_before && it.part_sf_after ? `<span>safety ${fmt(it.part_sf_before, 2)} → <b>${fmt(it.part_sf_after, 2)}</b></span>` : ""}
          ${Math.abs(it.system_life_after / it.system_life_before - 1) > 0.01 ? `<span>engine ${hours(it.system_life_before)} → <b>${hours(it.system_life_after)}</b></span>` : ""}
          ${Math.abs(it.mass_change_kg) > 0.0005 ? `<span>mass <b>${it.mass_change_kg > 0 ? "+" : ""}${fmt(it.mass_change_kg, 3)} kg</b></span>` : ""}
          ${Math.abs(it.cost_change_pct) >= 0.1 ? `<span>cost ${it.cost_change_pct > 0 ? "+" : ""}${fmt(it.cost_change_pct, 1)}%</span>` : ""}
        </div>
        ${it.also.length ? `<div class="note">Also: ${it.also.map(esc).join(" · ")}</div>` : ""}
        ${it.warnings.length ? `<div class="note" style="color:var(--bad)">Weakens: ${it.warnings.map(esc).join(" · ")}</div>` : ""}
        <button class="btn small" data-i="${i}">Apply</button>
      </div>`).join("") : `<p class="muted">No single change improves this part meaningfully${r.current?.sf_fatigue ? ` (it already has a fatigue safety factor of ${fmt(r.current.sf_fatigue, 2)})` : ""}. Try the Optimise tab to search combinations.</p>`) +
      `<p class="note">Every suggestion was simulated through the engine physics, including knock-on effects (a lighter pin unloads the rod and crankshaft). Nominal lives; treat as engineering estimates.</p>`;
    out.querySelectorAll("button[data-i]").forEach((b) => b.onclick = () => {
      const it = r.items[+b.dataset.i];
      applyDesign({ design: { [it.lever]: it.option } }, `${it.what}: ${it.change} applied`);
    });
  } catch (e) { out.innerHTML = `<div class="warn-box">${esc(e.message)}</div>`; }
}

async function applyDesign(body, msg) {
  busy(true, "Applying the design change…");
  try {
    S.asm = await post(`/api/assembly/${S.asm.id}/design`, body);
    S.sim = null;
    renderOverview(); renderParts(); renderSim(); if (S.selected !== null) renderPart(); applyColors();
    toast(`${msg} — run the life simulation to see the new life`);
  } catch (e) { toast(e.message, true); } finally { busy(false); }
}

// ------------------------------------------------------------------ continuous optimiser
const OBJECTIVES = { balanced: "Balanced: long life, safety ≥ 1.3, little extra mass or cost",
  life: "Longest life, whatever the mass or cost", light: "Lightest engine that still meets the target life" };

function renderOpt() {
  const el = $("tab-opt");
  if (!S.asm) { el.innerHTML = `<p class="muted">Load an engine first.</p>`; return; }
  if (S.asm.mode !== "engine") { el.innerHTML = `<p class="muted">The design optimiser simulates engines. This model is a general machine: use the material menu and FEA per part.</p>`; return; }
  el.innerHTML = `
    <h3>AI design optimiser</h3>
    <p class="muted">Runs engine simulations continuously. A learning model (random-forest surrogate) is trained on every result and picks the most promising designs to simulate next, so its predictions get better the longer it runs. Every simulation is saved, so opening this engine again continues from what it already learned.</p>
    <div class="form">
      <label>Goal<select id="opt-obj">${Object.entries(OBJECTIVES).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("")}</select></label>
      <label>Target life (h)<input id="opt-target" type="number" value="10000"></label>
      <label>Duty cycle<select id="opt-duty">${dutyOptions()}</select></label>
      <label>Simulations to run<input id="opt-budget" type="number" value="1500"></label>
    </div>
    <div class="row-actions"><button class="btn primary" id="opt-start">Start optimising</button><button class="btn" id="opt-stop">Stop</button></div>
    <div id="opt-out"></div>`;
  $("opt-start").onclick = startOpt;
  $("opt-stop").onclick = async () => { try { renderOptStatus(await post(`/api/assembly/${S.asm.id}/optimize/stop`, {})); } catch (e) { toast(e.message, true); } };
  pollOpt();
}

async function startOpt() {
  try {
    const st = await post(`/api/assembly/${S.asm.id}/optimize/start`, { objective: $("opt-obj").value,
      target_hours: +$("opt-target").value || 10000, duty: $("opt-duty").value, budget: +$("opt-budget").value || 1500 });
    renderOptStatus(st); pollOpt();
  } catch (e) { toast(e.message, true); }
}

async function pollOpt() {
  clearTimeout(S.optTimer);
  if (!S.asm || S.asm.mode !== "engine") return;
  const aid = S.asm.id;
  try {
    const st = await api(`/api/assembly/${aid}/optimize`);
    if (!S.asm || S.asm.id !== aid) return;
    renderOptStatus(st);
    if (st.running) S.optTimer = setTimeout(pollOpt, 1200);
  } catch (e) { /* server restarting: try again later */ S.optTimer = setTimeout(pollOpt, 4000); }
}

function renderOptStatus(st) {
  const out = $("opt-out"); if (!out) return;
  if (!st.evaluations_total) { out.innerHTML = `<p class="note">Not started yet.</p>`; return; }
  const base = st.baseline?.result, best = st.top?.[0];
  const h = st.history || [];
  out.innerHTML = `
    <div class="cards">
      <div class="card"><div class="k">Simulations</div><div class="v">${fmt(st.evaluations_total, 0)}</div><div class="note">${st.remembered_from_before ? `${fmt(st.remembered_from_before, 0)} remembered from before` : "this session"}${st.running ? " · running" : ""}</div></div>
      <div class="card"><div class="k">Engine life</div><div class="v">${best ? hours(best.result.system_life_h) : "—"}</div><div class="note">was ${base ? hours(base.system_life_h) : "—"}</div></div>
      <div class="card"><div class="k">Prediction error</div><div class="v">${h.length && h[h.length - 1].surrogate_error !== null ? fmt(h[h.length - 1].surrogate_error, 3) : "—"}</div><div class="note">falls as the model learns</div></div>
    </div>
    <p class="note">${esc(st.message || "")} · ${fmt(st.design_space, 0)} possible designs from ${st.levers} levers.</p>
    ${h.length > 1 ? `<div class="chart-box"><canvas id="opt-c1"></canvas></div><div class="chart-box"><canvas id="opt-c2"></canvas></div>` : ""}
    <h3>Best designs found</h3>
    ${(st.top || []).map((t, i) => `<details ${i === 0 ? "open" : ""}><summary><b>#${i + 1}</b> · life ${hours(t.result.system_life_h)} · min safety ${fmt(t.result.min_sf, 2)} · ${fmt(t.result.total_mass_kg, 2)} kg${base ? ` (${t.result.total_mass_kg >= base.total_mass_kg ? "+" : ""}${fmt(t.result.total_mass_kg - base.total_mass_kg, 2)})` : ""}</summary>
      ${t.changes.length ? `<ul class="reasons">${t.changes.map((c) => `<li><b>${esc(c.part)}</b> ${esc(c.what.toLowerCase())}: ${esc(c.change)}</li>`).join("")}</ul>` : `<p class="note">The design as uploaded.</p>`}
      <p class="note">Limited by: ${esc(t.result.parts[t.result.limiting_part]?.name || "—")} (${esc(t.result.parts[t.result.limiting_part]?.mode || "")})</p>
      ${t.changes.length ? `<button class="btn small" data-rank="${i}">Apply this design</button>` : ""}</details>`).join("")}`;
  out.querySelectorAll("button[data-rank]").forEach((b) => b.onclick = async () => {
    busy(true, "Applying the optimised design…");
    try { S.asm = await post(`/api/assembly/${S.asm.id}/optimize/apply`, { rank: +b.dataset.rank }); S.sim = null;
      renderOverview(); renderParts(); renderSim(); applyColors(); toast("Optimised design applied — run the life simulation to confirm it"); }
    catch (e) { toast(e.message, true); } finally { busy(false); }
  });
  if (h.length > 1) {
    const x = h.map((r) => r.evaluations);
    chart("opt1", "opt-c1", { type: "line", data: { labels: x, datasets: [
      { label: "best design score", data: h.map((r) => r.best_score), borderColor: "#c2185b", backgroundColor: "rgba(194,24,91,.08)", fill: true, pointRadius: 0, borderWidth: 2 }] },
      options: chartOpts("simulations run", "score (higher = better)") });
    chart("opt2", "opt-c2", { type: "line", data: { labels: x, datasets: [
      { label: "surrogate prediction error", data: h.map((r) => r.surrogate_error), borderColor: "#3b5ba5", pointRadius: 0, borderWidth: 2 }] },
      options: chartOpts("simulations run", "error (lower = smarter)", { type: "logarithmic" }) });
  }
}

function dutyOptions(sel = "automotive") {
  const d = S.info?.duty_cycles || { automotive: { label: "Passenger car" } };
  return Object.entries(d).map(([k, v]) => `<option value="${k}" ${k === sel ? "selected" : ""}>${esc(v.label)}</option>`).join("");
}

async function recommend(p) {
  $("rec-out").innerHTML = skel("Running every candidate material through this part's load model");
  try {
    const r = await api(`/api/assembly/${S.asm.id}/part/${p.id}/recommend?duty=${$("rec-duty").value}&target_hours=${+$("rec-target").value || 10000}`);
    $("rec-out").innerHTML = `<table><tr><th>#</th><th>Material</th><th class="num">life</th><th class="num">SF</th><th class="num">mass</th><th></th></tr>` +
      r.ranking.map((m, i) => `<tr><td>${i + 1}</td><td><b>${esc(m.name)}</b>${m.key === p.material ? ' <span class="pill">current</span>' : ""}<ul class="reasons">${m.reasons.slice(0, 4).map((x) => `<li>${esc(x)}</li>`).join("")}</ul></td>
        <td class="num">${hours(m.life_hours)}</td><td class="num">${fmt(m.min_safety, 2)}</td><td class="num">${fmt(m.mass_kg, 3)} kg</td>
        <td><button class="btn small" data-key="${m.key}">use</button></td></tr>`).join("") + `</table>
      <p class="note">Life = shortest predicted life over the duty cycle (fatigue, wear). SF = lowest safety factor. Scores weigh life, mass (more for moving parts), cost and margin.</p>`;
    $("rec-out").querySelectorAll("button[data-key]").forEach((b) => b.onclick = async () => {
      S.asm = await post(`/api/assembly/${S.asm.id}/part/${p.id}/material`, { key: b.dataset.key, all_copies: true });
      renderParts(); renderPart(); applyColors(); toast("Material applied");
    });
  } catch (e) { $("rec-out").innerHTML = `<div class="warn-box">${esc(e.message)}</div>`; }
}

function modesTable(modes) {
  return `<table><tr><th>Failure mode</th><th class="num">stress a / m</th><th class="num">SF fat</th><th class="num">SF static</th><th class="num">life</th></tr>` +
    modes.map((m) => {
      const life = m.hours_to_failure ?? m.nominal_hours;
      const sf = m.safety_fatigue ?? m.margin;
      return `<tr><td><b>${esc(m.name)}</b><div class="note">${esc(m.location)}${m.note ? " — " + esc(m.note) : ""}</div></td>
      <td class="num">${m.kind === "fatigue" ? `${fmt(m.sigma_a, 0)} / ${fmt(m.sigma_m, 0)}` : m.kind}</td>
      <td class="num">${sfPill(sf)}</td><td class="num">${sfPill(m.safety_static)}</td>
      <td class="num">${m.kind === "sealing" || m.kind === "check" || m.kind === "static" ? "—" : hours(life)}</td></tr>`;
    }).join("") + `</table>`;
}
function sfPill(v) {
  if (v === null || v === undefined) return "∞";
  return `<span class="pill ${v >= 1.5 ? "good" : v >= 1 ? "warn" : "bad"}">${fmt(v, 2)}</span>`;
}

async function testBench(p) {
  const body = { load: +$("tb-load").value || 1 };
  if ($("tb-rpm").value) body.rpm = +$("tb-rpm").value;
  if ($("tb-p").value) body.peak_pressure_bar = +$("tb-p").value;
  if ($("tb-t").value) body.temp_c = +$("tb-t").value;
  $("tb-out").innerHTML = skel("Solving the engine cycle and the part's load models");
  try {
    const r = await post(`/api/assembly/${S.asm.id}/part/${p.id}/test`, body);
    const c = r.cycle;
    let html = `<p class="muted">${fmt(r.conditions.rpm, 0)} rpm · load ${r.conditions.load} · peak ${fmt(c.peak_pressure_bar, 1)} bar at ${fmt(c.peak_pressure_angle_deg, 0)}° · ${fmt(c.indicated_power_kw, 0)} kW · rod ${fmt(c.max_rod_compression_kn, 1)} kN push / ${fmt(c.max_rod_tension_kn, 1)} kN pull</p>`;
    html += r.modes.length ? modesTable(r.modes) : `<div class="warn-box">${esc(r.note)}</div>`;
    if (r.pressure_sweep) {
      html += `<div class="chart-box"><canvas id="sweep-chart"></canvas></div>
        <p class="note">Fatigue (infinite-life) limit: ${r.infinite_life_limit_bar ? `<b>${fmt(r.infinite_life_limit_bar, 0)} bar</b>` : "above the sweep range"} · static limit: ${r.static_limit_bar ? `<b>${fmt(r.static_limit_bar, 0)} bar</b>` : "above the sweep range"}</p>`;
    }
    html += `<div class="chart-box"><canvas id="ptrace-chart"></canvas></div>`;
    $("tb-out").innerHTML = html;
    if (r.pressure_sweep) {
      chart("sweep", "sweep-chart", {
        type: "line", data: { labels: r.pressure_sweep.map((q) => q.peak_bar),
          datasets: [{ label: "fatigue SF", data: r.pressure_sweep.map((q) => q.safety_fatigue), borderColor: "#c2185b", tension: .2 },
                     { label: "static SF", data: r.pressure_sweep.map((q) => q.safety_static), borderColor: "#b7791f", tension: .2 },
                     { label: "SF = 1", data: r.pressure_sweep.map(() => 1), borderColor: "#d64545", borderDash: [5, 4], pointRadius: 0 }] },
        options: chartOpts("peak cylinder pressure (bar)", "safety factor", { min: 0 }) });
    }
    chart("ptrace", "ptrace-chart", { type: "line", data: { labels: r.pressure_trace.theta,
      datasets: [{ label: "cylinder pressure (bar)", data: r.pressure_trace.bar, borderColor: "#e07b39", pointRadius: 0 }] },
      options: chartOpts("crank angle (deg, 0 = firing TDC)", "bar") });
  } catch (e) { $("tb-out").innerHTML = `<div class="warn-box">${esc(e.message)}</div>`; }
}

async function runFea(p) {
  $("fea-out").innerHTML = skel("Voxelising the part and solving (a few seconds)");
  try {
    const r = await post(`/api/assembly/${S.asm.id}/part/${p.id}/fea`, { case: $("fea-case").value,
      magnitude: +$("fea-mag").value, axis: $("fea-axis").value, temp_c: +$("fea-t").value });
    S.fea = r;
    $("color-mode").value = "fea"; $("isolate").checked = true; applyColors(); fit(true);
    $("fea-out").innerHTML = `<div class="cards">
        <div class="card"><div class="k">Peak von Mises</div><div class="v">${fmt(r.peak_von_mises_mpa, 0)}<small> MPa</small></div></div>
        <div class="card"><div class="k">Yield at ${fmt(r.temp_c, 0)}°C</div><div class="v">${fmt(r.yield_at_temp_mpa, 0)}<small> MPa</small></div></div>
        <div class="card"><div class="k">Safety vs yield</div><div class="v">${sfPill(r.safety_factor_yield)}</div></div></div>
      <p class="muted">Max displacement ${fmt(r.max_displacement_mm, 4)} mm · ${r.elements} elements of ${fmt(r.voxel_mm, 2)} mm · ${fmt(r.elapsed_s, 1)} s · ${esc(r.material_name)}</p>
      <p class="note">${esc(r.note)} Red on the model = highest stress.</p>`;
  } catch (e) { $("fea-out").innerHTML = `<div class="warn-box">${esc(e.message)}</div>`; }
}

function simPartTable(r) {
  if (!r.modelled) return `<p class="muted">No built-in load model for this part type.</p>`;
  return `<div class="cards">
      <div class="card"><div class="k">${r.service_item ? "Replace every" : "B10 life"}</div><div class="v" style="font-size:15px">${r.service_item ? hours(r.service_interval_hours) : hours(r.b10_hours)}</div></div>
      <div class="card"><div class="k">B50 life</div><div class="v" style="font-size:15px">${hours(r.b50_hours)}</div></div>
      <div class="card"><div class="k">P(fail ≤ ${fmt(S.sim.hours, 0)} h)</div><div class="v">${fmt(r.prob_fail * 100, 1)}<small>%</small></div></div></div>
    <p class="muted">Most likely: <b>${esc(r.dominant_mode)}</b> at ${esc(r.location)}</p>${modesTable(r.modes)}`;
}

// ------------------------------------------------------------------ simulation
function renderSim() {
  const a = S.asm; const st = a.settings || {};
  $("tab-sim").innerHTML = `
    <p class="muted">Solves the engine's thermodynamic cycle at every operating point of the duty cycle, loads every part, then runs a Monte Carlo of virtual engines over the chosen hours, with wear and degradation feeding back into the loads.</p>
    <div class="form">
      <label>Duty cycle<select id="sim-duty">${dutyOptions()}</select></label>
      <label>Hours to simulate<input id="sim-hours" type="number" value="5000"></label>
      <label>Virtual engines (Monte Carlo)<input id="sim-samples" type="number" value="1000"></label>
      <label>Fuel<select id="sim-fuel"><option value="gasoline">petrol (spark ignition)</option><option value="diesel" ${st.fuel === "diesel" ? "selected" : ""}>diesel</option></select></label>
      <label>Compression ratio<input id="sim-cr" type="number" step="0.1" placeholder="10.5 petrol / 17 diesel"></label>
      <label>Boost (bar gauge)<input id="sim-boost" type="number" step="0.1" value="0"></label>
      <label>Max rpm<input id="sim-rpm" type="number" placeholder="from stroke"></label>
      <label>Peak pressure (bar)<input id="sim-p" type="number" placeholder="from fuel/boost"></label>
    </div>
    <button class="btn primary" id="sim-btn" ${a.engine?.cylinders ? "" : "disabled"}>Run life simulation</button>
    ${a.engine?.cylinders ? "" : `<p class="muted">Needs at least one identified piston.</p>`}
    <div id="sim-out"></div>`;
  $("sim-btn").onclick = runSim;
}

async function runSim() {
  const body = { duty: $("sim-duty").value, hours: +$("sim-hours").value || 5000, samples: +$("sim-samples").value || 1000,
    fuel: $("sim-fuel").value, boost_bar: +$("sim-boost").value || 0 };
  if ($("sim-cr").value) body.compression_ratio = +$("sim-cr").value;
  if ($("sim-rpm").value) body.max_rpm = +$("sim-rpm").value;
  if ($("sim-p").value) body.peak_pressure_bar = +$("sim-p").value;
  busy(true, `Simulating ${fmt(body.hours, 0)} hours × ${body.samples} engines…`);
  try {
    S.sim = await post(`/api/assembly/${S.asm.id}/simulate`, body);
    $("color-mode").value = "risk"; applyColors();
    renderSimResults();
    if (S.selected !== null) renderPart();
  } catch (e) { toast(e.message, true); }
  finally { busy(false); }
}

function renderSimResults() {
  const r = S.sim; const s = r.system; const em = r.engine_model;
  const service = r.parts.filter((p) => p.service_item && p.service_interval_hours);
  let html = `
    <h3>Result — ${esc(r.duty.label)}, ${fmt(r.hours, 0)} h, ${r.samples} engines</h3>
    <div class="cards">
      <div class="card"><div class="k">Engine B10 life</div><div class="v" style="font-size:15px">${hours(s.b10_hours)}</div>${s.b10_ci90 ? `<div class="note">90% CI ${fmt(s.b10_ci90[0], 0)}–${fmt(s.b10_ci90[1], 0)}</div>` : ""}</div>
      <div class="card"><div class="k">Engine B50 life</div><div class="v" style="font-size:15px">${hours(s.b50_hours)}</div>${s.b50_ci90 ? `<div class="note">90% CI ${fmt(s.b50_ci90[0], 0)}–${fmt(s.b50_ci90[1], 0)}</div>` : ""}</div>
      <div class="card"><div class="k">Fail within ${fmt(r.hours, 0)} h</div><div class="v">${fmt(s.prob_fail * 100, 1)}<small>%</small></div></div>
    </div>
    <p class="note">B10 = 10% of engines have had a failure by then; B50 = half. Lives past the horizon are projected from the final damage rates. Latin-hypercube sampling keeps run-to-run scatter of B10 below ~0.5%; the CI shows the remaining sampling uncertainty (not model uncertainty). Solved in ${fmt(r.elapsed_s, 1)} s.</p>
    <div class="chart-box"><canvas id="surv-chart"></canvas></div>
    <h3>What fails first</h3>
    <table>${Object.entries(s.first_failure_share).map(([l, v]) => `<tr><td><span class="swatch" style="background:${PALETTE[l]}"></span>${NAMES[l] || l}</td><td class="num">${fmt(v * 100, 1)}%</td></tr>`).join("")}</table>
    <h3>Highest risk parts (click to open)</h3>
    <table><tr><th>Part</th><th>Failure mode / location</th><th class="num">B10</th></tr>
    ${r.risk_ranking.map((q) => `<tr class="click" data-id="${q.id}"><td><span class="swatch" style="background:${PALETTE[q.label]}"></span>${NAMES[q.label] || q.label} #${q.id}</td><td>${esc(q.mode)}<div class="note">${esc(q.location)}</div></td><td class="num">${hours(q.b10_hours)}</td></tr>`).join("")}</table>
    ${service.length ? `<h3>Service items</h3><table>${[...new Map(service.map((p) => [p.label, p])).values()].map((p) => `<tr><td>${NAMES[p.label]}</td><td>${esc(p.dominant_mode)}</td><td class="num">replace every ${hours(p.service_interval_hours)}</td></tr>`).join("")}</table>` : ""}
    <h3>Damage build-up (worst copy of each part type)</h3>
    <div class="chart-box"><canvas id="dmg-chart"></canvas></div>
    <h3>Operating points solved</h3>
    <table><tr><th>mode</th><th class="num">time</th><th class="num">rpm</th><th class="num">p max</th><th class="num">power</th><th class="num">torque</th></tr>
    ${r.operating_points.map((o) => `<tr><td>${esc(o.name)}</td><td class="num">${fmt(o.time * 100, 0)}%</td><td class="num">${fmt(o.rpm, 0)}</td><td class="num">${fmt(o.peak_pressure_bar, 0)} bar</td><td class="num">${fmt(o.indicated_power_kw, 0)} kW</td><td class="num">${fmt(o.indicated_torque_nm, 0)} N·m</td></tr>`).join("")}</table>
    <p class="note">Engine model: ${fmt(em.displacement_l, 2)} L, CR ${em.compression_ratio}, ${em.fuel}, boost ${em.boost_bar} bar, max ${fmt(em.max_rpm, 0)} rpm, piston assembly ${fmt(em.m_piston_kg, 3)} kg, rod ${fmt(em.m_rod_kg, 3)} kg (masses from the geometry × material density). Power/torque are indicated (before friction).</p>`;
  $("sim-out").innerHTML = html;
  $("sim-out").querySelectorAll("tr.click").forEach((tr) => tr.onclick = () => selectPart(+tr.dataset.id));
  chart("surv", "surv-chart", { type: "line", data: { labels: s.survival_time,
    datasets: [{ label: "engines with no failure", data: s.survival.map((v) => v * 100), borderColor: "#c2185b", backgroundColor: "rgba(194,24,91,.08)", pointRadius: 0, fill: true, borderWidth: 2 }] },
    options: chartOpts("hours", "% surviving", { min: 0, max: 100 }) });
  const worst = {};
  for (const p of r.parts) {
    if (!p.modelled || p.service_item) continue;
    const last = p.history[p.history.length - 1] || 0;
    if (!worst[p.label] || last > worst[p.label].history.at(-1)) worst[p.label] = p;
  }
  const top = Object.values(worst).sort((a, b) => b.history.at(-1) - a.history.at(-1)).slice(0, 8);
  chart("dmg", "dmg-chart", { type: "line", data: { labels: r.history_time,
    datasets: top.map((p) => ({ label: NAMES[p.label], data: p.history.map((v) => v * 100), borderColor: PALETTE[p.label], pointRadius: 0 })) },
    options: chartOpts("hours", "% of life used (nominal)", { min: 0 }) });
}

function chartOpts(x, y, yExtra = {}) {
  return { responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { labels: { color: "#4a3a42", boxWidth: 10, font: { size: 11, family: "Zen Kaku Gothic New" } } } },
    scales: { x: { title: { display: true, text: x, color: "#7a6a72" }, ticks: { color: "#7a6a72", maxTicksLimit: 8 }, grid: { color: "#efe4e8" } },
              y: { title: { display: true, text: y, color: "#7a6a72" }, ticks: { color: "#7a6a72" }, grid: { color: "#efe4e8" }, ...yExtra } } };
}
function chart(key, canvasId, cfg) {
  if (S.charts[key]) S.charts[key].destroy();
  const el = $(canvasId); if (!el || !window.Chart) return;
  S.charts[key] = new Chart(el, cfg);
}

// ------------------------------------------------------------------ feedback
$("feedback-btn").onclick = () => { $("feedback-text").value = ""; $("feedback-dlg").showModal(); };
$("feedback-dlg").addEventListener("close", async () => {
  if ($("feedback-dlg").returnValue !== "send" || !$("feedback-text").value.trim()) return;
  const p = S.asm && S.selected !== null ? S.asm.parts[S.selected] : null;
  try {
    await post("/api/feedback", { text: $("feedback-text").value, mode: S.asm?.mode, label: p?.label,
      page: document.querySelector(".tabs button.active")?.dataset.tab });
    toast("Thanks — your feedback was saved");
  } catch (e) { toast(e.message, true); }
});

$("keys-btn").onclick = () => $("keys-dlg").showModal();

// ------------------------------------------------------------------ keyboard shortcuts
const COLOR_MODES = ["type", "cad", "confidence", "material", "risk", "fea"];
document.addEventListener("keydown", (e) => {
  const t = e.target;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (t.closest && t.closest("input, select, textarea, [contenteditable]")) {
    if (e.key === "Escape") t.blur();
    return;
  }
  if (document.querySelector("dialog[open]")) return;
  const k = e.key;
  const has = !!S.asm;
  const act = {
    "?": () => $("keys-dlg").showModal(),
    u: () => $("file-input").click(),
    n: () => $("start-pop").togglePopover?.(),
    f: () => has && fit($("isolate").checked),
    F: () => $("full-btn").click(),
    e: () => { if (!has) return; const x = $("explode"); x.value = +x.value > 0 ? 0 : 0.6; applyExplode(); },
    x: () => { if (!has) return; $("xray").checked = !$("xray").checked; applyColors(); },
    i: () => { if (!has) return; $("isolate").checked = !$("isolate").checked; applyColors(); fit($("isolate").checked); },
    c: () => { if (!has) return; const m = $("color-mode"); m.value = COLOR_MODES[(COLOR_MODES.indexOf(m.value) + 1) % COLOR_MODES.length]; applyColors(); toast(`Colour by: ${m.selectedOptions[0].textContent}`); },
    s: () => $("shot-btn").click(),
    p: () => togglePanel(),
    "/": () => { if (!has) return; showTab("parts"); setTimeout(() => $("part-search")?.focus(), 30); },
    "[": () => stepPart(-1),
    "]": () => stepPart(1),
    Escape: () => { if (has) { selectPart(null); S.labelFilter = null; applyColors(); } },
  };
  if (/^[1-5]$/.test(k)) { showTab(TABS[+k - 1]); e.preventDefault(); return; }
  const fn = act[k] || act[k.toLowerCase?.()];
  if (k === "F" && e.shiftKey) { act.F(); e.preventDefault(); return; }
  if (fn) { fn(); e.preventDefault(); }
});

// ------------------------------------------------------------------ resizable side panel
{
  const rz = $("resizer"); let x0 = 0, w0 = 0;
  try { const w = +localStorage.getItem("enginelab.panelw"); if (w >= 320) document.documentElement.style.setProperty("--panel-w", `${w}px`); } catch (_) {}
  const width = () => $("panel").getBoundingClientRect().width;
  const set = (w) => {
    w = Math.max(320, Math.min(w, window.innerWidth * 0.62));
    document.documentElement.style.setProperty("--panel-w", `${Math.round(w)}px`);
    try { localStorage.setItem("enginelab.panelw", Math.round(w)); } catch (_) {}
    resize(); moveInk();
  };
  rz.addEventListener("pointerdown", (e) => { x0 = e.clientX; w0 = width(); rz.classList.add("drag"); rz.setPointerCapture(e.pointerId); });
  rz.addEventListener("pointermove", (e) => { if (rz.classList.contains("drag")) set(w0 - (e.clientX - x0)); });
  rz.addEventListener("pointerup", () => rz.classList.remove("drag"));
  rz.addEventListener("dblclick", () => set(460));
  rz.addEventListener("keydown", (e) => { if (e.key === "ArrowLeft") set(width() + 24); if (e.key === "ArrowRight") set(width() - 24); });
}

// ------------------------------------------------------------------ boot
(async function boot() {
  const t0 = performance.now();
  try {
    const [info, mats, samples, regions] = await Promise.all([api("/api/info"), api("/api/materials"), api("/api/samples"), api("/api/regions")]);
    initRegion(regions);
    renderSamples(samples);
    S.info = info; S.materials = mats; S.matByKey = Object.fromEntries(mats.map((m) => [m.key, m]));
    let dismissed = false;
    try { dismissed = sessionStorage.getItem("enginelab.beta-ok") === "1"; } catch (_) {}
    if (info.beta && !dismissed) {
      const b = document.createElement("div");
      b.className = "beta-banner";
      b.innerHTML = `<span><b>Beta</b> · results are engineering estimates, not certification · uploads up to ${info.max_upload_mb} MB are processed in memory and not kept — please don't upload confidential CAD · part-type corrections are saved (shape numbers only) to improve the AI</span><button aria-label="Dismiss">×</button>`;
      b.querySelector("button").onclick = () => { b.remove(); try { sessionStorage.setItem("enginelab.beta-ok", "1"); } catch (_) {} resize(); };
      document.body.insertBefore(b, document.querySelector(".layout"));
    }
    if (info.classifier) {
      $("tab-overview").insertAdjacentHTML("beforeend", `<p class="note" style="text-align:center">Engine part classifier: trained on ${fmt(info.classifier.n_engines, 0)} generated engines · held-out accuracy ${fmt(info.classifier.holdout_accuracy * 100, 1)}%</p>`);
    }
  } catch (e) { toast("Server not reachable: " + e.message, true); }
  renderRecents();
  resize(); moveInk();
  // Shareable links: ?demo=1 (or ?demo=<seed>) opens a demo engine, ?sample=<key> opens a sample.
  const q = new URLSearchParams(location.search);
  if (q.has("demo")) { const n = parseInt(q.get("demo"), 10); if (n > 0) $("demo-seed").value = n; runDemo(); }
  else if (q.get("sample")) openSample(q.get("sample"));
  setTimeout(() => { $("splash").classList.add("gone"); Petals.start(); setTimeout(() => $("splash").remove(), 500); },
    Math.max(0, 650 - (performance.now() - t0)));
})();
