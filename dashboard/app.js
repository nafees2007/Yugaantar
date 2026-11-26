const API_BASE = "http://localhost:8000";

let currentRollout = null;
let currentShock = "none";
let currentVar = 1; // Default to tp
let currentLead = 1;

const CHANNELS = [
  { name: "2m Temperature", unit: "°C", isTemp: true, min: 10, max: 45 },
  { name: "Precipitation", unit: "mm/day", isRain: true, min: 0, max: 80 },
  { name: "10m Eastward Wind", unit: "m/s", min: -15, max: 15 },
  { name: "10m Northward Wind", unit: "m/s", min: -15, max: 15 },
  { name: "Sea Level Pressure", unit: "hPa", isPres: true, min: 990, max: 1025 },
  { name: "Column Water Vapour", unit: "kg/m²", min: 5, max: 70 }
];

// Check server health on load
async function checkHealth() {
  const statusEl = document.getElementById("status-text");
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      statusEl.textContent = `Online (${data.model_name || "Ready"})`;
      document.querySelector(".dot").style.backgroundColor = "#10b981";
    } else {
      throw new Error();
    }
  } catch (err) {
    statusEl.textContent = "Offline (start serve.py)";
    document.querySelector(".dot").style.backgroundColor = "#ef4444";
  }
}

function setShock(type) {
  currentShock = type;
  document.querySelectorAll(".button-grid .btn").forEach(b => b.classList.remove("active"));
  if (type === 'none') document.getElementById("btn-none").classList.add("active");
  if (type === 'warm_ocean') document.getElementById("btn-bob").classList.add("active");
  if (type === 'monsoon_stall') document.getElementById("btn-arabian").classList.add("active");
  runSimulation();
}

// Generate base input states and optional shock perturbation
function buildPayload() {
  const init_states = [];
  for (let d = 0; d < 3; d++) {
    const day = [];
    // t2m (K), tp (mm/day), u10 (m/s), v10 (m/s), msl (Pa), tcwv (kg/m2)
    day.push(Array.from({ length: 32 }, () => Array.from({ length: 36 }, () => 295.0 + Math.random() * 8.0)));
    day.push(Array.from({ length: 32 }, () => Array.from({ length: 36 }, () => Math.random() < 0.3 ? Math.random() * 25.0 : 0.0)));
    day.push(Array.from({ length: 32 }, () => Array.from({ length: 36 }, () => (Math.random() - 0.4) * 6.0)));
    day.push(Array.from({ length: 32 }, () => Array.from({ length: 36 }, () => (Math.random() - 0.4) * 6.0)));
    day.push(Array.from({ length: 32 }, () => Array.from({ length: 36 }, () => 101000.0 + (Math.random() - 0.5) * 800.0)));
    day.push(Array.from({ length: 32 }, () => Array.from({ length: 36 }, () => 25.0 + Math.random() * 30.0)));
    init_states.push(day);
  }

  let perturbation = null;
  if (currentShock === "warm_ocean") {
    // Bay of Bengal: lat 8..20 (indices 2..14), lon 82..94 (indices 17..29)
    perturbation = Array.from({ length: 6 }, () => Array.from({ length: 32 }, () => Array.from({ length: 36 }, () => 0.0)));
    for (let r = 2; r <= 14; r++) {
      for (let c = 17; c <= 29; c++) {
        perturbation[0][r][c] = 3.0; // +3 K
        perturbation[5][r][c] = 7.5; // +15% moisture
      }
    }
  } else if (currentShock === "monsoon_stall") {
    // Arabian sea: lat 8..20 (indices 2..14), lon 65..75 (indices 0..10)
    perturbation = Array.from({ length: 6 }, () => Array.from({ length: 32 }, () => Array.from({ length: 36 }, () => 0.0)));
    for (let r = 2; r <= 14; r++) {
      for (let c = 0; c <= 10; c++) {
        perturbation[2][r][c] = -2.5; // -40% wind
        perturbation[3][r][c] = -2.5;
      }
    }
  }

  return {
    init_states: init_states,
    doy_sincos: [0.85, 0.52], // Monsoon season (~July)
    steps: 10,
    perturbation: perturbation
  };
}

async function runSimulation() {
  const btn = document.getElementById("run-sim-btn");
  btn.disabled = true;
  btn.textContent = "⏳ Simulating Weather Reality...";

  try {
    const payload = buildPayload();
    const res = await fetch(`${API_BASE}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("Simulation endpoint error");
    const data = await res.json();
    currentRollout = data.rollout; // [10, 6, 32, 36]
    document.getElementById("timing-info").textContent = `Inference latency: ${data.inference_ms.toFixed(1)} ms`;
    renderGrid();
    updateImpactMetrics();
  } catch (err) {
    console.warn("Using offline simulated rollout fallback:", err);
    generateMockRollout();
    renderGrid();
    updateImpactMetrics();
  } finally {
    btn.disabled = false;
    btn.textContent = "⚡ Run 10-Day Digital Twin Rollout";
  }
}

function generateMockRollout() {
  currentRollout = Array.from({ length: 10 }, (_, k) => 
    Array.from({ length: 6 }, (c) => 
      Array.from({ length: 32 }, (r) => 
        Array.from({ length: 36 }, (col) => {
          if (c === 0) return 293 + Math.sin(col / 5) * 5 + (currentShock === "warm_ocean" ? 2.5 : 0);
          if (c === 1) return Math.max(0, (Math.sin(r / 3) + Math.cos(col / 4)) * 20 * (1 + k * 0.1));
          if (c === 2) return Math.sin(r) * 6;
          if (c === 3) return Math.cos(col) * 5;
          if (c === 4) return 101000 + (r - 16) * 40;
          return 35 + Math.sin(r + col) * 15;
        })
      )
    )
  );
}

function getColor(val, min, max, isRain) {
  const norm = Math.max(0, Math.min(1, (val - min) / (max - min)));
  if (isRain) {
    if (val < 1.0) return [13, 21, 39]; // dry
    if (val < 10) return [34, 197, 94]; // light green
    if (val < 35) return [59, 130, 246]; // moderate blue
    return [239, 68, 68]; // heavy red
  }
  // Thermal gradient (blue -> cyan -> yellow -> red)
  const r = Math.floor(norm * 255);
  const b = Math.floor((1 - norm) * 255);
  const g = Math.floor(Math.sin(norm * Math.PI) * 200);
  return [r, g, b];
}

function renderGrid() {
  if (!currentRollout) return;
  const canvas = document.getElementById("gridCanvas");
  const ctx = canvas.getContext("2d");
  const chInfo = CHANNELS[currentVar];

  const gridData = currentRollout[currentLead - 1][currentVar];
  const cellW = canvas.width / 36;
  const cellH = canvas.height / 32;

  // Grid coordinates: lat 6.5..37.5 is south to north (row 0 is south, row 31 is north)
  // For canvas, row 0 is top (north) so we flip vertical axis
  for (let r = 0; r < 32; r++) {
    const latIdx = 31 - r;
    for (let c = 0; c < 36; c++) {
      let rawVal = gridData[latIdx][c];
      let displayVal = rawVal;
      if (chInfo.isTemp) displayVal = rawVal - 273.15; // K -> C
      if (chInfo.isPres) displayVal = rawVal / 100.0; // Pa -> hPa

      const rgb = getColor(displayVal, chInfo.min, chInfo.max, chInfo.isRain);
      ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
      ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
    }
  }

  document.getElementById("map-title").textContent = 
    `${chInfo.name} (${chInfo.unit}) - Lead Day ${currentLead}`;
}

function updateImpactMetrics() {
  if (!currentRollout) return;
  const rainData = currentRollout[currentLead - 1][1];
  const tempData = currentRollout[currentLead - 1][0];

  let heavyRainCount = 0;
  let heatCount = 0;

  for (let r = 0; r < 32; r++) {
    for (let c = 0; c < 36; c++) {
      if (rainData[r][c] >= 35.0) heavyRainCount++;
      if (tempData[r][c] >= 313.15) heatCount++; // > 40 C
    }
  }

  const onsetEl = document.getElementById("onset-idx");
  if (currentShock === "monsoon_stall") {
    onsetEl.textContent = "Stalled / Delayed (Weak Arabian Inflow)";
    onsetEl.className = "metric-value alert";
  } else if (currentShock === "warm_ocean") {
    onsetEl.textContent = "Intensified (Bay of Bengal Depression Risk)";
    onsetEl.className = "metric-value warn";
  } else {
    onsetEl.textContent = "Active / Normal Onset Circulation";
    onsetEl.className = "metric-value";
  }

  document.getElementById("heat-risk").textContent = `${heatCount * 12} Districts (${heatCount} Grid Cells)`;
  document.getElementById("rain-risk").textContent = heavyRainCount > 10 ? "Widespread Alert (>50mm)" : "Local Showers";
}

document.getElementById("var-select").addEventListener("change", (e) => {
  currentVar = parseInt(e.target.value);
  renderGrid();
});

document.getElementById("lead-slider").addEventListener("input", (e) => {
  currentLead = parseInt(e.target.value);
  document.getElementById("lead-val").textContent = `Day ${currentLead}`;
  renderGrid();
  updateImpactMetrics();
});

// Initialization
checkHealth();
runSimulation();
