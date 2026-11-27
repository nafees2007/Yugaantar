const API_BASE = "http://localhost:8000";

let primeRollout = null;   // [10, 6, 32, 36]
let branchRollout = null;  // [10, 6, 32, 36]
let currentShock = "none";
let currentView = "prime"; // 'prime' | 'branch' | 'delta'
let currentVar = 1;        // Default: Precipitation (tp)
let currentLead = 1;       // 1..10
let customCenter = { r: 14, c: 16 };
let customJudgePreset = null;
let indiaClip = true;
let mapZoom = 1.0;
let playTimer = null;

const CHANNELS = [
  { name: "2m Temperature", key: "t2m", unit: "°C", isTemp: true, min: 10, max: 45, dMax: 4.5 },
  { name: "Precipitation", key: "tp", unit: "mm/day", isRain: true, min: 0, max: 45, dMax: 6.0 },
  { name: "10m Eastward Wind", key: "u10", unit: "m/s", min: -12, max: 12, dMax: 4.0 },
  { name: "10m Northward Wind", key: "v10", unit: "m/s", min: -12, max: 12, dMax: 4.0 },
  { name: "Sea Level Pressure", key: "msl", unit: "hPa", isPres: true, min: 995, max: 1020, dMax: 3.0 },
  { name: "Column Water Vapour", key: "tcwv", unit: "kg/m²", min: 10, max: 68, dMax: 8.0 }
];

const CITIES = [
  { name: "Delhi", lat: 28.61, lon: 77.21 },
  { name: "Mumbai", lat: 19.08, lon: 72.88 },
  { name: "Bengaluru", lat: 12.97, lon: 77.59 },
  { name: "Chennai", lat: 13.08, lon: 80.27 },
  { name: "Kolkata", lat: 22.57, lon: 88.36 },
  { name: "Hyderabad", lat: 17.39, lon: 78.49 },
  { name: "Ahmedabad", lat: 23.02, lon: 72.57 },
  { name: "Guwahati", lat: 26.14, lon: 91.74 },
  { name: "Kochi", lat: 9.93, lon: 76.27 }
];

const REGIONS = [
  { name: "Western Ghats", r0: 4, r1: 14, c0: 7, c1: 11 },
  { name: "Central India", r0: 12, r1: 19, c0: 10, c1: 19 },
  { name: "East Coast", r0: 9, r1: 17, c0: 17, c1: 23 },
  { name: "North / IGP", r0: 19, r1: 25, c0: 9, c1: 22 }
];

function lonToX(lon, W) { return ((lon - 65.5 + 0.5) / 36.0) * W; }
function latToY(lat, H) { return H - ((lat - 6.5 + 0.5) / 32.0) * H; }

// Coastline segments from LAND_MASK for ocean/neighbor context
const COAST_SEGS = (() => {
  const segs = [], thr = 0.45;
  for (let r = 0; r < 31; r++) {
    for (let c = 0; c < 35; c++) {
      const v00 = LAND_MASK[r][c] >= thr ? 1 : 0;
      const v10 = LAND_MASK[r][c + 1] >= thr ? 1 : 0;
      const v11 = LAND_MASK[r + 1][c + 1] >= thr ? 1 : 0;
      const v01 = LAND_MASK[r + 1][c] >= thr ? 1 : 0;
      const code = v00 | (v10 << 1) | (v11 << 2) | (v01 << 3);
      if (code === 0 || code === 15) continue;
      const interp = (a, b) => Math.abs(b - a) < 1e-6 ? 0.5 : (thr - a) / (b - a);
      const top = [c + interp(LAND_MASK[r][c], LAND_MASK[r][c + 1]), r];
      const right = [c + 1, r + interp(LAND_MASK[r][c + 1], LAND_MASK[r + 1][c + 1])];
      const bottom = [c + interp(LAND_MASK[r + 1][c], LAND_MASK[r + 1][c + 1]), r + 1];
      const left = [c, r + interp(LAND_MASK[r][c], LAND_MASK[r + 1][c])];
      const add = (p1, p2) => segs.push([p1, p2]);
      switch (code) {
        case 1: case 14: add(left, top); break;
        case 2: case 13: add(top, right); break;
        case 3: case 12: add(left, right); break;
        case 4: case 11: add(right, bottom); break;
        case 5: add(left, top); add(right, bottom); break;
        case 6: case 9: add(top, bottom); break;
        case 7: case 8: add(left, bottom); break;
        case 10: add(top, right); add(left, bottom); break;
      }
    }
  }
  return segs;
})();

function findIndianState(lat, lon) {
  if (typeof INDIA_STATES === "undefined") return null;
  for (const st of INDIA_STATES) {
    for (const ring of st.polys) {
      let inside = false;
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const xi = ring[i][0], yi = ring[i][1];
        const xj = ring[j][0], yj = ring[j][1];
        const intersect = ((yi > lat) !== (yj > lat)) &&
          (lon < ((xj - xi) * (lat - yi)) / (yj - yi + 1e-12) + xi);
        if (intersect) inside = !inside;
      }
      if (inside) return st.name;
    }
  }
  return null;
}

// Sidebar Navigation
function switchNav(section) {
  document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
  document.querySelector(`.nav-item[data-nav="${section}"]`).classList.add("active");

  document.querySelectorAll("#right-panel .panel-group").forEach(p => p.classList.add("hidden"));
  const target = document.getElementById(`panel-${section}`);
  if (target) target.classList.remove("hidden");
  else document.getElementById("panel-overview").classList.remove("hidden");
}

function runTimelinePrimeAction() {
  setShock("none");
  setViewMode("prime");
  if (!playTimer) togglePlay();
}

function zoomMap(factor) {
  mapZoom = Math.max(1.0, Math.min(2.4, mapZoom * factor));
  renderMap();
}
function resetZoom() {
  mapZoom = 1.0;
  renderMap();
}

function toggleIndiaClip() {
  indiaClip = !indiaClip;
  const btn = document.getElementById("mp-clip");
  btn.textContent = indiaClip ? "🇮🇳 India Clip: ON" : "🌏 Full Domain";
  btn.classList.toggle("active", indiaClip);
  renderMap();
}

async function checkHealth() {
  const sysDot = document.getElementById("sys-dot");
  const sysTxt = document.getElementById("sys-text");
  const apiDot = document.getElementById("api-dot");
  const apiTxt = document.getElementById("api-text");
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "POST" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    sysDot.style.backgroundColor = "#10b981";
    sysTxt.textContent = "System Ready";
    apiDot.style.backgroundColor = "#10b981";
    apiTxt.textContent = "API Connected";
  } catch (err) {
    sysDot.style.backgroundColor = "#ef4444";
    sysTxt.textContent = "Server Offline";
    apiDot.style.backgroundColor = "#ef4444";
    apiTxt.textContent = "Run serve.py";
  }
}

function syncPreset(val) {
  customJudgePreset = null;
  document.getElementById("preset-select").value = val;
  document.getElementById("preset-select-2").value = val;
  runSimulation();
}

function loadJudgeJson(event) {
  const file = event.target.files[0];
  if (!file) return;
  const statusEl = document.getElementById("judge-json-status");
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const obj = JSON.parse(e.target.result);
      if (!obj.init_states || obj.init_states.length !== 3 || obj.init_states[0].length !== 6) {
        throw new Error("init_states must be [3, 6, 32, 36]");
      }
      if (!obj.doy_sincos || obj.doy_sincos.length !== 2) {
        throw new Error("doy_sincos must be [sin, cos]");
      }
      customJudgePreset = {
        label: `Judge Input (${file.name})`,
        init_states: obj.init_states,
        doy_sincos: obj.doy_sincos
      };
      statusEl.textContent = `Loaded ${file.name} — predicting 10 days!`;
      statusEl.style.color = "#10b981";
      runSimulation();
    } catch (err) {
      statusEl.textContent = `Invalid JSON: ${err.message}`;
      statusEl.style.color = "#ef4444";
    }
  };
  reader.readAsText(file);
}

function downloadSampleJson() {
  const presetKey = document.getElementById("preset-select").value;
  const preset = WEATHER_PRESETS[presetKey];
  const payload = JSON.stringify({
    init_states: preset.init_states,
    doy_sincos: preset.doy_sincos,
    steps: 10,
    perturbation: null
  });
  const blob = new Blob([payload], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `sample_3day_${presetKey}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

async function runLiveEvaluation() {
  const btn = document.getElementById("eval-btn");
  const statusEl = document.getElementById("eval-status");
  btn.disabled = true;
  statusEl.textContent = "Running 5 holdout windows (Days 1, 4, 8) on serve.py...";
  try {
    const res = await fetch(`${API_BASE}/evaluate`, { method: "POST" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const d = await res.json();
    document.getElementById("eval-params").textContent = `${d.parameters.toLocaleString()} (${d.weights_mb} MB)`;
    document.getElementById("eval-rmse").textContent = `${d.rmse_pts.toFixed(2)} / 30.0`;
    document.getElementById("eval-csi").textContent = `${d.csi_pts.toFixed(2)} / 15.0`;
    document.getElementById("eval-blur").textContent = `${d.blur_pts.toFixed(2)} / 10.0`;
    document.getElementById("eval-causality").textContent = `${d.causality_pts.toFixed(2)} / 15.0`;
    document.getElementById("eval-total").textContent = `${d.total_pts.toFixed(2)} / 100.0`;
    document.getElementById("mod-lat").textContent = `${d.avg_latency_ms} ms`;
    statusEl.textContent = d.source === "live_holdout_evaluation"
      ? "Verified live on 2020–2021 holdout dataset!"
      : "Loaded verified holdout benchmark summary.";
    statusEl.style.color = "#10b981";
  } catch (err) {
    statusEl.textContent = `Evaluation error: ${err.message}`;
    statusEl.style.color = "#ef4444";
  } finally {
    btn.disabled = false;
  }
}

function updateCustomLabel() {
  const mag = parseFloat(document.getElementById("custom-mag").value);
  const v = document.getElementById("custom-var").value;
  const units = { t2m: "K", tcwv: "kg/m²", u10: "m/s", v10: "m/s", msl: "hPa" };
  document.getElementById("custom-mag-val").textContent = `${mag >= 0 ? "+" : ""}${mag} ${units[v]}`;
  if (currentShock === "custom") runSimulation();
}

function setShock(type) {
  currentShock = type;
  document.querySelectorAll(".shock-btn").forEach(b => b.classList.remove("active"));
  const mapBtn = { none: "sb-none", warm_ocean: "sb-bob", monsoon_stall: "sb-arabian", custom: "sb-custom" };
  if (mapBtn[type]) document.getElementById(mapBtn[type]).classList.add("active");

  document.getElementById("custom-shock-box").classList.toggle("hidden", type !== "custom");

  const shockPill = document.getElementById("active-shock-pill");
  if (type === "none") {
    shockPill.classList.add("hidden");
    setViewMode("prime", false);
    indiaClip = true;
    document.getElementById("mp-clip").textContent = "🇮🇳 India Clip: ON";
    document.getElementById("mp-clip").classList.add("active");
  } else {
    shockPill.classList.remove("hidden");
    shockPill.textContent = `Shock: ${type.replace("_", " ").toUpperCase()}`;
    setViewMode("delta", false);
    // In Delta mode, show Full Domain so ocean shock wave is immediately visible!
    indiaClip = false;
    document.getElementById("mp-clip").textContent = "🌏 Full Domain";
    document.getElementById("mp-clip").classList.remove("active");
  }
  runSimulation();
}

function setViewMode(mode, shouldRender = true) {
  currentView = mode;
  document.querySelectorAll(".map-mode-pills .m-pill").forEach(p => {
    if (p.id !== "mp-clip") p.classList.remove("active");
  });
  document.getElementById(`mp-${mode}`).classList.add("active");

  const isSplit = mode === "split";
  document.getElementById("split-slot").classList.toggle("hidden", !isSplit);
  document.getElementById("lbl-primary").classList.toggle("hidden", !isSplit);
  document.getElementById("windOverlay").classList.toggle("hidden", isSplit);

  const chipText = { prime: "TIMELINE PRIME", branch: "MULTIVERSE BRANCH", delta: "CAUSAL Δ RIPPLE", split: "PRIME vs BRANCH SPLIT" };
  document.getElementById("view-chip").textContent = chipText[mode];
  if (shouldRender) renderMap();
}

function setLeadDay(day) {
  currentLead = day;
  document.getElementById("lead-select").value = day;
  document.getElementById("lead-slider").value = day;
  document.getElementById("prime-day-lbl").textContent = `Day ${day}`;
  document.querySelectorAll(".day-btn").forEach((b, idx) => b.classList.toggle("active", idx + 1 === day));
  renderMap(); updateKPIsAndSummaries(); drawBottomTrends();
}

function buildPerturbation() {
  if (currentShock === "none") return null;
  const pert = Array.from({ length: 6 }, () =>
    Array.from({ length: 32 }, () => Array(36).fill(0.0))
  );
  if (currentShock === "warm_ocean") {
    for (let r = 2; r <= 14; r++) for (let c = 17; c <= 29; c++) pert[0][r][c] = 3.0;
  } else if (currentShock === "monsoon_stall") {
    for (let r = 2; r <= 14; r++) for (let c = 0; c <= 10; c++) { pert[2][r][c] = -3.0; pert[3][r][c] = -2.5; }
  } else if (currentShock === "custom") {
    const cVar = document.getElementById("custom-var").value;
    const mag = parseFloat(document.getElementById("custom-mag").value);
    const rad = parseInt(document.getElementById("custom-rad").value, 10);
    const chIdx = { t2m: 0, u10: 2, v10: 3, msl: 4, tcwv: 5 }[cVar];
    const scale = cVar === "msl" ? mag * 100.0 : mag;
    for (let dr = -rad; dr <= rad; dr++) {
      for (let dc = -rad; dc <= rad; dc++) {
        const rr = customCenter.r + dr, cc = customCenter.c + dc;
        if (rr >= 0 && rr < 32 && cc >= 0 && cc < 36) {
          const dist = Math.sqrt(dr * dr + dc * dc);
          pert[chIdx][rr][cc] = scale * Math.max(0.25, 1.0 - dist / (rad + 1.2));
        }
      }
    }
  }
  return pert;
}

async function callSimulate(preset, perturbation) {
  const res = await fetch(`${API_BASE}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      init_states: preset.init_states,
      doy_sincos: preset.doy_sincos,
      steps: 10,
      perturbation: perturbation
    })
  });
  if (!res.ok) throw new Error("Simulate failed: " + res.status);
  return await res.json();
}

async function runSimulation() {
  try {
    const presetKey = document.getElementById("preset-select").value;
    const preset = customJudgePreset || WEATHER_PRESETS[presetKey];
    document.getElementById("active-scenario-pill").textContent = `Input: ${preset.label}`;

    const resPrime = await callSimulate(preset, null);
    primeRollout = resPrime.rollout;

    let totalMs = resPrime.inference_ms;
    const pert = buildPerturbation();
    if (pert) {
      const resBranch = await callSimulate(preset, pert);
      branchRollout = resBranch.rollout;
      totalMs += resBranch.inference_ms;
    } else {
      branchRollout = primeRollout;
    }

    const now = new Date();
    const ts = now.toTimeString().slice(0, 5);
    document.getElementById("last-run-text").textContent = `Last Run: ${ts} (${totalMs.toFixed(0)} ms)`;
    document.getElementById("mod-lat").textContent = `${resPrime.inference_ms.toFixed(0)} ms`;

    renderMap();
    updateKPIsAndSummaries();
    drawBottomTrends();
  } catch (err) {
    document.getElementById("last-run-text").textContent = `API Error: ${err.message}`;
  }
}

function toDisplayUnit(val, chIdx) {
  if (chIdx === 0) return val - 273.15;
  if (chIdx === 4) return val / 100.0;
  return val;
}

function sampleBilinear(grid, gx, gy) {
  const x = Math.max(0, Math.min(35, gx)), y = Math.max(0, Math.min(31, gy));
  const c0 = Math.floor(x), c1 = Math.min(35, c0 + 1);
  const r0 = Math.floor(y), r1 = Math.min(31, r0 + 1);
  const fx = x - c0, fy = y - r0;
  return (grid[r0][c0] * (1 - fx) + grid[r0][c1] * fx) * (1 - fy) +
         (grid[r1][c0] * (1 - fx) + grid[r1][c1] * fx) * fy;
}

function lerpStops(val, stops) {
  if (val <= stops[0][0]) return stops[0][1];
  for (let i = 0; i < stops.length - 1; i++) {
    const [v0, c0] = stops[i], [v1, c1] = stops[i + 1];
    if (val <= v1) {
      const k = (val - v0) / (v1 - v0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * k),
        Math.round(c0[1] + (c1[1] - c0[1]) * k),
        Math.round(c0[2] + (c1[2] - c0[2]) * k)
      ];
    }
  }
  return stops[stops.length - 1][1];
}

// Vibrant IMD Radar Palette (matches reference image: Blue -> Cyan -> Green -> Yellow -> Orange -> Red)
const RAIN_STOPS = [
  [0.0,  [24, 48, 128]],
  [2.0,  [29, 98, 224]],
  [5.0,  [14, 185, 215]],
  [9.0,  [22, 214, 128]],
  [14.0, [186, 232, 28]],
  [20.0, [250, 204, 21]],
  [28.0, [249, 115, 22]],
  [40.0, [239, 68, 68]]
];

const TEMP_STOPS = [
  [10.0, [30, 64, 175]],
  [18.0, [6, 182, 212]],
  [25.0, [16, 185, 129]],
  [31.0, [250, 204, 21]],
  [37.0, [249, 115, 22]],
  [43.0, [220, 38, 38]]
];

const GENERAL_STOPS = [
  [0.0,  [20, 42, 115]],
  [0.25, [14, 145, 205]],
  [0.50, [20, 205, 145]],
  [0.75, [248, 200, 35]],
  [1.0,  [239, 68, 68]]
];

function getColorRGB(val, chIdx, isDelta) {
  const cfg = CHANNELS[chIdx];
  if (isDelta) {
    const t = Math.max(-1, Math.min(1, val / cfg.dMax));
    if (Math.abs(t) < 0.02) return [12, 22, 44];
    if (t > 0) return lerpStops(t, [[0,[18,28,50]], [0.3,[217,119,6]], [0.7,[239,68,68]], [1.0,[254,240,138]]]);
    return lerpStops(-t, [[0,[18,28,50]], [0.3,[2,132,199]], [0.7,[34,211,238]], [1.0,[207,250,254]]]);
  }
  if (cfg.isRain) return lerpStops(val, RAIN_STOPS);
  if (cfg.isTemp) return lerpStops(val, TEMP_STOPS);
  return lerpStops(Math.max(0, Math.min(1, (val - cfg.min) / (cfg.max - cfg.min))), GENERAL_STOPS);
}

function updateColorbar(chIdx, isDelta) {
  const cfg = CHANNELS[chIdx];
  const labels = document.getElementById("v-bar-labels");
  const grad = document.getElementById("v-bar-grad");
  if (isDelta) {
    labels.innerHTML = `<span>+${cfg.dMax}</span><span>+${(cfg.dMax/2).toFixed(1)}</span><span>0</span><span>-${(cfg.dMax/2).toFixed(1)}</span><span>-${cfg.dMax}</span>`;
    grad.style.background = "linear-gradient(180deg, #fef08a, #ef4444, #121c32, #0284c7, #cffafe)";
  } else if (cfg.isRain) {
    labels.innerHTML = `<span>40+</span><span>28</span><span>20</span><span>14</span><span>9</span><span>5</span><span>1</span>`;
    grad.style.background = "linear-gradient(180deg, #ef4444, #f97316, #facc15, #bae81c, #16d680, #0eb9d7, #1d62e0, #183080)";
  } else if (cfg.isTemp) {
    labels.innerHTML = `<span>43°C</span><span>37</span><span>31</span><span>25</span><span>18</span><span>10</span>`;
    grad.style.background = "linear-gradient(180deg, #dc2626, #f97316, #facc15, #10b981, #06b6d4, #1e40af)";
  } else {
    labels.innerHTML = `<span>${cfg.max}</span><span>${((cfg.max+cfg.min)/2).toFixed(0)}</span><span>${cfg.min}</span>`;
    grad.style.background = "linear-gradient(180deg, #ef4444, #f8c823, #14cd91, #0e91cd, #142a73)";
  }
}

function traceIndiaPath(ctx, W, H) {
  ctx.beginPath();
  for (const st of INDIA_STATES) {
    for (const ring of st.polys) {
      ring.forEach(([lon, lat], i) => {
        const x = lonToX(lon, W), y = latToY(lat, H);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
    }
  }
}

function renderMap() {
  if (!primeRollout || !branchRollout) return;
  const primary = document.getElementById("indiaCanvas");
  if (currentView === "split") {
    renderMapToCanvas(primary, "prime");
    renderMapToCanvas(document.getElementById("splitCanvas"), "branch");
  } else {
    renderMapToCanvas(primary, currentView);
  }
}

function renderMapToCanvas(canvas, fieldMode) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const tIdx = currentLead - 1;
  const isDelta = fieldMode === "delta";
  const smooth = document.getElementById("smooth-toggle") ? document.getElementById("smooth-toggle").checked : true;
  const showWinds = document.getElementById("wind-toggle") ? document.getElementById("wind-toggle").checked : true;
  const showStates = document.getElementById("state-toggle") ? document.getElementById("state-toggle").checked : true;
  const showCities = document.getElementById("city-toggle") ? document.getElementById("city-toggle").checked : true;

  updateColorbar(currentVar, isDelta);

  // Build 32x36 grid
  const dataGrid = Array.from({ length: 32 }, (_, r) =>
    Array.from({ length: 36 }, (_, c) => {
      const p = toDisplayUnit(primeRollout[tIdx][currentVar][r][c], currentVar);
      const b = toDisplayUnit(branchRollout[tIdx][currentVar][r][c], currentVar);
      if (fieldMode === "prime") return p;
      if (fieldMode === "branch") return b;
      return b - p;
    })
  );

  // Create smooth heatmap offscreen canvas
  const rw = 420, rh = 270;
  const off = document.createElement("canvas");
  off.width = rw; off.height = rh;
  const offCtx = off.getContext("2d");
  const img = offCtx.createImageData(rw, rh);
  for (let py = 0; py < rh; py++) {
    const gy = (1.0 - (py + 0.5) / rh) * 32 - 0.5;
    for (let pxIdx = 0; pxIdx < rw; pxIdx++) {
      const gx = ((pxIdx + 0.5) / rw) * 36 - 0.5;
      const val = smooth
        ? sampleBilinear(dataGrid, gx, gy)
        : dataGrid[Math.max(0, Math.min(31, Math.round(gy)))][Math.max(0, Math.min(35, Math.round(gx)))];
      const [r, g, b] = getColorRGB(val, currentVar, isDelta);
      const idx = (py * rw + pxIdx) * 4;
      img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
    }
  }
  offCtx.putImageData(img, 0, 0);

  // Clear & Apply Zoom Transform centered on India (~lon 80, lat 21)
  ctx.save();
  ctx.fillStyle = "#061022";
  ctx.fillRect(0, 0, W, H);

  if (mapZoom > 1.0) {
    const cx = lonToX(81.0, W), cy = latToY(21.5, H);
    ctx.translate(cx, cy);
    ctx.scale(mapZoom, mapZoom);
    ctx.translate(-cx, -cy);
  }

  // Dark Ocean & Neighboring Land Background
  ctx.fillStyle = "#07142b";
  ctx.fillRect(0, 0, W, H);

  // Subtle surrounding coastlines (Sri Lanka, Myanmar, Arabia, etc.)
  const cellW = W / 36, cellH = H / 32;
  ctx.strokeStyle = "rgba(56, 189, 248, 0.22)";
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  for (const [p1, p2] of COAST_SEGS) {
    ctx.moveTo((p1[0] + 0.5) * cellW, H - (p1[1] + 0.5) * cellH);
    ctx.lineTo((p2[0] + 0.5) * cellW, H - (p2[1] + 0.5) * cellH);
  }
  ctx.stroke();

  // Subtle Lat/Lon Grid
  ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
  ctx.lineWidth = 0.8;
  for (let lat = 10; lat <= 35; lat += 5) {
    const y = latToY(lat, H);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  for (let lon = 70; lon <= 95; lon += 5) {
    const x = lonToX(lon, W);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }

  // Draw Weather Field (either clipped strictly inside India OR full domain)
  ctx.save();
  if (indiaClip && typeof INDIA_STATES !== "undefined") {
    traceIndiaPath(ctx, W, H);
    ctx.clip();
  }
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, W, H);
  ctx.restore();

  // Crisp White Indian State & District Borders (Survey of India bounds up to 37.08N)
  if (showStates && typeof INDIA_STATES !== "undefined") {
    ctx.strokeStyle = "rgba(0, 0, 0, 0.35)";
    ctx.lineWidth = 1.1;
    traceIndiaPath(ctx, W, H);
    ctx.stroke();

    ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
    ctx.lineWidth = 0.55;
    traceIndiaPath(ctx, W, H);
    ctx.stroke();
  }

  // 10m Wind Vectors (flowing across Ocean & India!)
  if (showWinds) {
    const srcRoll = fieldMode === "prime" ? primeRollout : branchRollout;
    ctx.strokeStyle = "rgba(226, 232, 240, 0.52)";
    ctx.fillStyle = "rgba(248, 250, 252, 0.72)";
    ctx.lineWidth = 1.1;
    for (let r = 1; r < 32; r += 2) {
      for (let c = 1; c < 36; c += 2) {
        let u = srcRoll[tIdx][2][r][c], v = srcRoll[tIdx][3][r][c];
        if (isDelta) {
          u = branchRollout[tIdx][2][r][c] - primeRollout[tIdx][2][r][c];
          v = branchRollout[tIdx][3][r][c] - primeRollout[tIdx][3][r][c];
        }
        const mag = Math.sqrt(u * u + v * v);
        if (mag < 0.7) continue;
        const cx = (c + 0.5) * cellW, cy = H - (r + 0.5) * cellH;
        const len = Math.min(18, mag * 1.45);
        const ux = (u / mag) * len, vy = -(v / mag) * len;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + ux, cy + vy); ctx.stroke();
        const ang = Math.atan2(vy, ux);
        ctx.beginPath();
        ctx.moveTo(cx + ux, cy + vy);
        ctx.lineTo(cx + ux - 4 * Math.cos(ang - 0.5), cy + vy - 4 * Math.sin(ang - 0.5));
        ctx.lineTo(cx + ux - 4 * Math.cos(ang + 0.5), cy + vy - 4 * Math.sin(ang + 0.5));
        ctx.closePath(); ctx.fill();
      }
    }
  }

  // Cities
  if (showCities) {
    ctx.textAlign = "left";
    for (const city of CITIES) {
      const cx = lonToX(city.lon, W), cy = latToY(city.lat, H);
      ctx.fillStyle = "#040914";
      ctx.beginPath(); ctx.arc(cx, cy, 3.6, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#facc15";
      ctx.beginPath(); ctx.arc(cx, cy, 2.0, 0, Math.PI * 2); ctx.fill();
      ctx.font = "600 9.5px sans-serif";
      ctx.fillStyle = "rgba(0,0,0,0.85)";
      ctx.fillText(city.name, cx + 5, cy + 3);
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(city.name, cx + 4, cy + 2);
    }
  }

  // Shock Box Overlay
  if (currentShock !== "none" && fieldMode !== "prime") {
    ctx.strokeStyle = "#facc15";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    if (currentShock === "warm_ocean") ctx.strokeRect(17 * cellW, H - 15 * cellH, 13 * cellW, 13 * cellH);
    else if (currentShock === "monsoon_stall") ctx.strokeRect(0, H - 15 * cellH, 11 * cellW, 13 * cellH);
    else if (currentShock === "custom") {
      const rad = parseInt(document.getElementById("custom-rad").value, 10);
      const size = rad * 2 + 1;
      ctx.strokeRect(Math.max(0, customCenter.c - rad) * cellW, H - Math.min(32, customCenter.r + rad + 1) * cellH, size * cellW, size * cellH);
    }
    ctx.setLineDash([]);
  }

  ctx.restore();
}

function computeIndiaMeans(rollout, leadIdx) {
  let tSum = 0, rSum = 0, wSum = 0, mSum = 0, cnt = 0;
  let kharifRain = 0, kharifCnt = 0, heatCells = 0, floodCells = 0;
  for (let r = 0; r < 32; r++) {
    for (let c = 0; c < 36; c++) {
      if (LAND_MASK[r][c] > 0.5 && r >= 2 && r <= 29 && c >= 3 && c <= 31) {
        const tC = rollout[leadIdx][0][r][c] - 273.15;
        const tp = rollout[leadIdx][1][r][c];
        const wspd = Math.hypot(rollout[leadIdx][2][r][c], rollout[leadIdx][3][r][c]);
        const tcwv = rollout[leadIdx][5][r][c];
        tSum += tC; rSum += tp; wSum += wspd; mSum += tcwv; cnt++;
        if (tC > 33.5) heatCells++;
        if (tp > 20.0) floodCells++;
        if (r >= 10 && r <= 23 && c >= 9 && c <= 21) { kharifRain += tp; kharifCnt++; }
      }
    }
  }
  return {
    temp: tSum / cnt, rain: rSum / cnt, wind: wSum / cnt, moist: mSum / cnt,
    kharif: kharifRain / Math.max(1, kharifCnt),
    heatPct: (heatCells / Math.max(1, cnt)) * 100,
    floodCells
  };
}

function updateKPIsAndSummaries() {
  if (!primeRollout || !branchRollout) return;
  const tIdx = currentLead - 1;
  const mp = computeIndiaMeans(primeRollout, tIdx);
  const mb = computeIndiaMeans(branchRollout, tIdx);
  const m0 = computeIndiaMeans(primeRollout, 0);
  const active = currentView === "prime" ? mp : mb;

  const refTag = currentShock === "none" ? "vs D1" : "vs Prime";
  const fmtDelta = (val) => `${val >= 0 ? "↑ +" : "↓ "}${val.toFixed(1)} ${refTag}`;

  document.getElementById("kpi-temp").textContent = `${active.temp.toFixed(1)} °C`;
  document.getElementById("kpi-temp-d").textContent = currentShock === "none" ? fmtDelta(active.temp - m0.temp) : fmtDelta(mb.temp - mp.temp);

  document.getElementById("kpi-rain").textContent = `${active.rain.toFixed(1)} mm/day`;
  document.getElementById("kpi-rain-d").textContent = currentShock === "none" ? fmtDelta(active.rain - m0.rain) : fmtDelta(mb.rain - mp.rain);

  document.getElementById("kpi-wind").textContent = `${active.wind.toFixed(1)} m/s`;
  document.getElementById("kpi-wind-d").textContent = currentShock === "none" ? fmtDelta(active.wind - m0.wind) : fmtDelta(mb.wind - mp.wind);

  document.getElementById("kpi-moist").textContent = `${active.moist.toFixed(1)} kg/m²`;
  document.getElementById("kpi-moist-d").textContent = currentShock === "none" ? fmtDelta(active.moist - m0.moist) : fmtDelta(mb.moist - mp.moist);

  // Outlook & Impact Cards
  const outlookEl = document.getElementById("outlook-status");
  const descEl = document.getElementById("outlook-desc");
  if (active.rain > 8.0) {
    outlookEl.textContent = "Active"; outlookEl.className = "text-green";
    descEl.textContent = "Well distributed monsoon rainfall across central & coastal India.";
  } else if (active.heatPct > 15.0) {
    outlookEl.textContent = "Heat Stress"; outlookEl.className = "text-red";
    descEl.textContent = "Elevated pre-monsoon thermal anomaly across northern & central plains.";
  } else {
    outlookEl.textContent = "Moderate"; outlookEl.className = "text-amber";
    descEl.textContent = "Transitional convective activity with localized coastal showers.";
  }

  document.getElementById("ov-crop").textContent = `${active.kharif.toFixed(1)} mm/d`;
  document.getElementById("ov-heat").textContent = `${active.heatPct.toFixed(1)}% land`;
  document.getElementById("ov-flood").textContent = `${active.floodCells} cells`;

  document.getElementById("imp-crop").textContent = `${active.kharif.toFixed(1)} mm/d`;
  document.getElementById("imp-crop-d").textContent = currentShock === "none" ? "Kharif belt baseline" : `Shock Δ: ${(mb.kharif - mp.kharif).toFixed(2)} mm/d`;
  document.getElementById("imp-heat").textContent = `${active.heatPct.toFixed(1)}%`;
  document.getElementById("imp-heat-d").textContent = currentShock === "none" ? "Land area > 33.5°C daily mean" : `Shock Δ: ${(mb.heatPct - mp.heatPct).toFixed(1)}%`;
  document.getElementById("imp-flood").textContent = `${active.floodCells} cells`;
  document.getElementById("imp-flood-d").textContent = currentShock === "none" ? "Grid cells > 20 mm/d" : `Shock Δ: ${mb.floodCells - mp.floodCells} cells`;

  // Regional Comparison Table
  document.getElementById("cmp-day").textContent = currentLead;
  let rows = "";
  for (const reg of REGIONS) {
    let ps = 0, bs = 0, cnt = 0;
    for (let r = reg.r0; r <= reg.r1; r++) for (let c = reg.c0; c <= reg.c1; c++) {
      if (LAND_MASK[r][c] > 0.3) {
        ps += toDisplayUnit(primeRollout[tIdx][currentVar][r][c], currentVar);
        bs += toDisplayUnit(branchRollout[tIdx][currentVar][r][c], currentVar);
        cnt++;
      }
    }
    const pa = ps / cnt, ba = bs / cnt, d = ba - pa;
    rows += `<tr><td>${reg.name}</td><td>${pa.toFixed(1)}</td><td>${ba.toFixed(1)}</td><td style="color:${d>=0?'#f59e0b':'#38bdf8'}">${d>=0?'+':''}${d.toFixed(2)}</td></tr>`;
  }
  document.getElementById("cmp-tbody").innerHTML = rows;
}

function drawBottomTrends() {
  if (!primeRollout) return;
  const activeRoll = currentView === "prime" ? primeRollout : branchRollout;
  const temps = [], rains = [], winds = [];
  for (let t = 0; t < 10; t++) {
    const m = computeIndiaMeans(activeRoll, t);
    temps.push(m.temp); rains.push(m.rain); winds.push(m.wind);
  }

  document.getElementById("tr-temp-badge").textContent = `${temps[9] - temps[0] >= 0 ? "+" : ""}${(temps[9] - temps[0]).toFixed(1)}°C`;
  const rPct = temps[0] ? ((rains[9] - rains[0]) / Math.max(0.1, rains[0]) * 100) : 0;
  document.getElementById("tr-rain-badge").textContent = `${rPct >= 0 ? "+" : ""}${rPct.toFixed(0)}%`;
  document.getElementById("tr-wind-badge").textContent = `${winds[9] - winds[0] >= 0 ? "+" : ""}${(winds[9] - winds[0]).toFixed(1)} m/s`;

  drawLineChart("tempChart", temps, "#f59e0b", "rgba(245,158,11,0.18)");
  drawBarChart("rainChart", rains, "#38bdf8");
  drawLineChart("windChart", winds, "#10b981", "rgba(16,185,129,0.18)");
}

function drawLineChart(id, data, color, fillCol) {
  const c = document.getElementById(id), ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  let min = Math.min(...data), max = Math.max(...data);
  if (Math.abs(max - min) < 0.5) { min -= 0.5; max += 0.5; }
  const padL = 28, padR = 10, padT = 8, padB = 18;
  const mx = i => padL + (i / 9) * (W - padL - padR);
  const my = v => H - padB - ((v - min) / (max - min)) * (H - padT - padB);

  ctx.strokeStyle = "rgba(148,163,184,0.12)"; ctx.fillStyle = "#64748b"; ctx.font = "9px sans-serif";
  [min, (min + max) / 2, max].forEach(v => {
    const y = my(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(v.toFixed(0), 2, y + 3);
  });

  // Area fill
  ctx.fillStyle = fillCol;
  ctx.beginPath();
  data.forEach((v, i) => i === 0 ? ctx.moveTo(mx(i), my(v)) : ctx.lineTo(mx(i), my(v)));
  ctx.lineTo(mx(9), H - padB); ctx.lineTo(mx(0), H - padB); ctx.closePath(); ctx.fill();

  // Stroke
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((v, i) => i === 0 ? ctx.moveTo(mx(i), my(v)) : ctx.lineTo(mx(i), my(v)));
  ctx.stroke();

  // Dots + Day labels
  data.forEach((v, i) => {
    const x = mx(i), y = my(v);
    ctx.fillStyle = i === currentLead - 1 ? "#fff" : color;
    ctx.beginPath(); ctx.arc(x, y, i === currentLead - 1 ? 3.5 : 2, 0, Math.PI * 2); ctx.fill();
    if (i % 2 === 0 || i === 9) { ctx.fillStyle = "#64748b"; ctx.fillText(`D${i + 1}`, x - 6, H - 4); }
  });
}

function drawBarChart(id, data, color) {
  const c = document.getElementById(id), ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const max = Math.max(5, ...data) * 1.15;
  const padL = 26, padR = 10, padT = 8, padB = 18;
  const bw = ((W - padL - padR) / 10) * 0.58;
  const stepW = (W - padL - padR) / 10;

  ctx.fillStyle = "#64748b"; ctx.font = "9px sans-serif";
  ctx.fillText(max.toFixed(0), 2, padT + 6);
  ctx.fillText("0", 12, H - padB);

  data.forEach((v, i) => {
    const x = padL + i * stepW + (stepW - bw) / 2;
    const h = (v / max) * (H - padT - padB);
    ctx.fillStyle = i === currentLead - 1 ? "#38bdf8" : "rgba(56, 189, 248, 0.55)";
    ctx.fillRect(x, H - padB - h, bw, h);
    if (i % 2 === 0 || i === 9) { ctx.fillStyle = "#64748b"; ctx.fillText(`D${i + 1}`, x, H - 4); }
  });
}

function togglePlay() {
  const btn = document.getElementById("play-btn");
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    btn.textContent = "▶";
  } else {
    btn.textContent = "⏸";
    playTimer = setInterval(() => {
      setLeadDay((currentLead % 10) + 1);
    }, 650);
  }
}

document.getElementById("preset-select").addEventListener("change", (e) => syncPreset(e.target.value));
document.getElementById("var-select").addEventListener("change", (e) => {
  currentVar = parseInt(e.target.value, 10);
  renderMap(); updateKPIsAndSummaries();
});
document.getElementById("lead-select").addEventListener("change", (e) => setLeadDay(parseInt(e.target.value, 10)));
document.getElementById("lead-slider").addEventListener("input", (e) => setLeadDay(parseInt(e.target.value, 10)));

const canvasEl = document.getElementById("indiaCanvas");
canvasEl.addEventListener("mousemove", (e) => {
  if (!primeRollout) return;
  const rect = canvasEl.getBoundingClientRect();
  const gx = ((e.clientX - rect.left) / rect.width) * 36 - 0.5;
  const gy = (1.0 - (e.clientY - rect.top) / rect.height) * 32 - 0.5;
  const c = Math.max(0, Math.min(35, Math.round(gx)));
  const r = Math.max(0, Math.min(31, Math.round(gy)));
  const latNum = 6.5 + gy, lonNum = 65.5 + gx;
  const stName = findIndianState(latNum, lonNum);
  const label = stName ? `🇮🇳 ${stName}` : (LAND_MASK[r][c] > 0.5 ? "Land" : "Ocean");

  const tIdx = currentLead - 1;
  const p = toDisplayUnit(primeRollout[tIdx][currentVar][r][c], currentVar);
  const b = toDisplayUnit(branchRollout[tIdx][currentVar][r][c], currentVar);
  const cfg = CHANNELS[currentVar];

  const tt = document.getElementById("hover-tooltip");
  tt.classList.remove("hidden");
  tt.style.left = `${e.clientX + 14}px`;
  tt.style.top = `${e.clientY + 14}px`;
  tt.innerHTML = `
    <strong>${label}</strong> (${latNum.toFixed(1)}°N, ${lonNum.toFixed(1)}°E)<br>
    Prime: <b>${p.toFixed(2)} ${cfg.unit}</b><br>
    Branch: <b>${b.toFixed(2)} ${cfg.unit}</b><br>
    Δ Diff: <b style="color:${b - p >= 0 ? "#f59e0b" : "#38bdf8"}">${b - p >= 0 ? "+" : ""}${(b - p).toFixed(2)} ${cfg.unit}</b>
  `;
});
canvasEl.addEventListener("mouseleave", () => document.getElementById("hover-tooltip").classList.add("hidden"));
canvasEl.addEventListener("click", (e) => {
  const rect = canvasEl.getBoundingClientRect();
  const c = Math.floor(((e.clientX - rect.left) / rect.width) * 36);
  const r = 31 - Math.floor(((e.clientY - rect.top) / rect.height) * 32);
  if (r >= 0 && r < 32 && c >= 0 && c < 36) {
    customCenter = { r, c };
    document.getElementById("custom-coords").textContent = `${(6.5 + r).toFixed(1)}°N, ${(65.5 + c).toFixed(1)}°E`;
    switchNav("multiverse");
    if (currentShock !== "custom") setShock("custom");
    else runSimulation();
  }
});

// Animated Flowing Wind Streamlines + Shock Origin Pulse on #windOverlay
const WIND_PARTICLES = Array.from({ length: 160 }, () => ({
  gx: Math.random() * 35,
  gy: Math.random() * 31,
  age: Math.floor(Math.random() * 60)
}));

let pulseTick = 0;
function animateWindOverlay() {
  const overlay = document.getElementById("windOverlay");
  if (overlay && primeRollout && currentView !== "split") {
    const ctx = overlay.getContext("2d");
    const W = overlay.width, H = overlay.height;
    ctx.clearRect(0, 0, W, H);

    const showWinds = document.getElementById("wind-toggle") ? document.getElementById("wind-toggle").checked : true;
    const tIdx = currentLead - 1;
    const srcRoll = currentView === "prime" ? primeRollout : branchRollout;

    ctx.save();
    if (mapZoom > 1.0) {
      const cx = lonToX(81.0, W), cy = latToY(21.5, H);
      ctx.translate(cx, cy);
      ctx.scale(mapZoom, mapZoom);
      ctx.translate(-cx, -cy);
    }

    if (showWinds && srcRoll) {
      const uGrid = srcRoll[tIdx][2], vGrid = srcRoll[tIdx][3];
      ctx.lineWidth = 1.4;
      for (const p of WIND_PARTICLES) {
        const u = sampleBilinear(uGrid, p.gx, p.gy);
        const v = sampleBilinear(vGrid, p.gx, p.gy);
        const x0 = ((p.gx + 0.5) / 36.0) * W;
        const y0 = H - ((p.gy + 0.5) / 32.0) * H;
        p.gx += u * 0.032;
        p.gy += v * 0.032;
        p.age++;
        const x1 = ((p.gx + 0.5) / 36.0) * W;
        const y1 = H - ((p.gy + 0.5) / 32.0) * H;

        const spd = Math.hypot(u, v);
        ctx.strokeStyle = `rgba(224, 242, 254, ${Math.min(0.68, 0.15 + spd * 0.055)})`;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();

        if (p.age > 55 || p.gx < 0 || p.gx > 35 || p.gy < 0 || p.gy > 31) {
          p.gx = Math.random() * 35;
          p.gy = Math.random() * 31;
          p.age = 0;
        }
      }
    }

    // Pulsing Target Ring for Active Shock
    if (currentShock !== "none" && currentView !== "prime") {
      pulseTick = (pulseTick + 0.06) % (Math.PI * 2);
      const cellW = W / 36, cellH = H / 32;
      let px = 23 * cellW, py = H - 8 * cellH;
      if (currentShock === "monsoon_stall") { px = 5 * cellW; py = H - 8 * cellH; }
      else if (currentShock === "custom") { px = (customCenter.c + 0.5) * cellW; py = H - (customCenter.r + 0.5) * cellH; }
      const radius = 10 + 14 * (0.5 + 0.5 * Math.sin(pulseTick));
      ctx.strokeStyle = `rgba(250, 204, 21, ${0.75 - 0.45 * (0.5 + 0.5 * Math.sin(pulseTick))})`;
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }
  requestAnimationFrame(animateWindOverlay);
}

window.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  runSimulation();
  requestAnimationFrame(animateWindOverlay);
});
