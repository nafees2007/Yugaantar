# MONSOONVERSE: Physics-Guided Climate Digital Twin of India

**Track 2: Multiverse of Monsoon — Earth-System Modelling & Climate Simulation**  
**Team:** Prince Kumar & Nafees Hyder  
Built for the Yugaantar DevForge Hackathon

---

## 🌟 Overview

**MONSOONVERSE** is a 100% offline, CPU-optimized Spatiotemporal Digital Twin of India's weather (`6.5°N–37.5°N, 65.5°E–100.5°E`, `32×36` grid at `1°×1°` resolution). Given 3 consecutive days of atmospheric state across 6 variables (`t2m, tp, u10, v10, msl, tcwv`), it:
1. **Simulates (Timeline Prime):** Autoregressively rolls weather forward for 10+ days in **~186.5 ms on CPU** without exploding or collapsing into blurry climatological averages.
2. **Branches (Multiverse):** Evolves localized physical shocks using a **physical transport prior** (Clausius-Clapeyron thermodynamic coupling + 10m surface wind advection + Gaussian diffusion + damping), rather than learned counterfactuals, ensuring strict Day-1 localization and smooth propagation.
3. **Visualizes Impact (`dashboard/`):** Drives a live, 100% offline 35-state India Mission Control dashboard with Kharif crop moisture, heatwave proxy exposure, and heavy convective rainfall alerts.

---

## 🏗️ System Architecture

```text
  Input: 3 Days [3, 6, 32, 36] + Static Terrain (Mask/Orography) + DOY sin/cos
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
  [ Climatology Anomaly ]                         [ Physical Shock Prior ]
  Subtract 366-day Baseline                       If perturbation != null:
  Log1p Transform on Rain (tp)                    Clausius-Clapeyron coupling +
              │                                   10m wind advection + diffusion
              ▼                                               │
  [ Dilated Residual CNN ]                                    │
  197,670 params (d=1,2,3,1)                                  │
  Predicts Next-Day Delta                                     │
              │                                               │
              ▼                                               │
  [ Blend + Tail Calibration ]                                │
  w4=0.70, w8=0.40, tp × 1.15                                 │
              │                                               │
              ▼                                               │
  [ Inversion + Physical Clamps ]                             │
  Enforce tp >= 0, tcwv >= 0, bounded t2m                     │
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      ▼
                     10-Day Physical Rollout [10, 6, 32, 36]
```

---

## 📊 Local Holdout Metric Summary (2020–2021 Windows)

| Metric | Local holdout result |
|---|---|
| **RMSE skill (days 1, 4, 8)** | Mean skill score **+0.0883** (`22.09 / 30.0` on local scorer) |
| **Heavy-rain CSI (>10 mm/day)** | **10.00 / 15.0** on the local scorer |
| **Day-8 variance retained** | **94.9%** (`9.49 / 10.0`; gate: 40%) |
| **Shock checks** | Local, causal, stable, exact zero null (`15.00 / 15.0` on local scorer) |
| **Latency** | **186.5 ms** mean model time, **395 ms** mean HTTP RTT |

> *Caveat: Blend weights, rain scaling, and the loss were tuned on the 2020–2021 holdout, so hidden-window results may be lower.*

---

## ⚠️ Known Limitations & Failure Modes

1. **Heavy-Rain & Lead-8 Limits:** Heavy-rain skill is partial (`10.00 / 15.0` on our local holdout metric), and Day-8 rainfall is the weakest forecast lead.
2. **Surface Wind Steering vs. Upper-Level Depressions:** Because only 10m surface winds are provided, Bay of Bengal anomalies advect northeast along the surface SW monsoon flow toward Odisha/Assam rather than upper-level (500 hPa) northwest depression tracks.
3. **Coarse 1°×1° Grid (~110 km):** Sub-grid urban heat islands, point cloudbursts, and full cyclogenesis are not natively resolved.
4. **Physical Transport Prior:** Shocks use a deterministic transport prior rather than learned counterfactual ground truth.

---

## 🚀 Setup, Run & Train

### 1. Environment Setup
* **Python:** 3.12 (sandbox target; tested on Python 3.10+)
* **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

### 2. Run the Offline Inference Server (Round-1 Contract)
`serve.py` exposes **only** `POST /health` and `POST /simulate`, requires no internet, and loads all weights and climatology from `weights.pt`:
```bash
python serve.py --port 8000
```
* **Check it works** (with the official kit installed separately):
  ```bash
  python -m monsoon_r1.schema_validator http://localhost:8000
  ```

### 3. Open the Interactive Dashboard (100% Offline)
While `serve.py` is running, open [`dashboard/index.html`](dashboard/index.html) in Chrome or Edge.  
*(Note: `dashboard/samples.js` and `dashboard/india_geo.js` are small demo inputs and offline state borders for the dashboard only, not used by the model.)*

### 4. Retrain from Scratch
`train.py` reads `india_climate_train.nc` from either the repo root (`./india_climate_train.nc`) or parent directory (`../india_climate_train.nc`)—kept outside the repo. Per [`NOTICE`](NOTICE), the dataset contains modified Copernicus Climate Change Service ERA5 reanalysis data (Hersbach et al. 2020) accessed via Google WeatherBench 2 (Rasp et al. 2024).
```bash
python train.py
```
**Training output:** `weights.pt` (`16.91 MB`), which bundles the trained model weights, 366-day baseline climatologies, and calibrated `blend_weights`.

---

## 🔧 Optional Extras (Not Used by Round-1 Grading)

* **Product / Live Evaluation Server (`app_product.py`):** Runs on a separate entry point so `serve.py` stays strictly minimal:
  ```bash
  python app_product.py --port 8001
  ```
* **Local Benchmark Script:** `python evaluate.py` (uses `monsoon_r1.scoring` when the kit is installed).

---

## 📁 Repository Structure

```text
├── serve.py              # Main offline FastAPI server (/health and /simulate only)
├── app_product.py        # Optional evaluation server (/evaluate on port 8001)
├── model.py              # Dilated Residual CNN (197,670 params)
├── inference_utils.py    # Anomaly transforms & bounded physical shock evolution
├── train.py              # Composite Soft-CSI loss + targeted oversampling trainer
├── data.py               # NetCDF/HDF5 dataset loader & climatology builder
├── evaluate.py           # Local diagnostic & holdout benchmark script
├── requirements.txt      # Sandbox-matched Python dependencies
├── weights.pt            # Self-contained checkpoint + climatology + blend bundle (16.91 MB)
├── blend_weights.json    # Calibrated lead-time climatology blend weights
├── stats.json            # Channel normalization statistics
├── docs/
│   └── BRIEF.pdf         # 2-page technical brief (Architecture + Impact)
└── dashboard/            # Standalone 100% offline frontend UI
    ├── index.html
    ├── style.css
    ├── app.js
    ├── india_geo.js      # Offline 35-state India border GeoJSON (UI only)
    └── samples.js        # 3 small demo 3-day windows + land-sea mask (UI only)
```
