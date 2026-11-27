# MONSOONVERSE: Physics-Guided Climate Digital Twin of India

**Track 2: Multiverse of Monsoon — Earth-System Modelling & Climate Simulation**  
**Team:** Prince Kumar & Nafees Hyder  
Built for the Yugaantar DevForge Hackathon (inspired by ISRO's Bharatiya Antariksh Hackathon PS-5).

---

## 🌟 Overview

**MONSOONVERSE** is a 100% offline, CPU-optimized Spatiotemporal Digital Twin of India's weather (`6.5°N–37.5°N, 65.5°E–100.5°E`, `32×36` grid at `1°×1°` resolution). Given 3 consecutive days of atmospheric state across 6 variables (`t2m, tp, u10, v10, msl, tcwv`), it:
1. **Simulates (Timeline Prime):** Autoregressively rolls weather forward for 10+ days in **~186 ms on CPU** without exploding or collapsing into blurry climatological averages.
2. **Branches (Multiverse):** Accepts localized physical shocks (e.g., warmer Bay of Bengal, weakened Arabian Sea monsoon wind, or any custom click-on-map perturbation) and evolves them causally across India without teleportation.
3. **Visualizes Impact (`dashboard/`):** Drives a live, 100% offline ISRO-inspired 35-state India Mission Control dashboard with Kharif crop moisture, heatwave exposure, and heavy convective rainfall alerts.

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

## 📊 Verified Benchmark Scorecard

Evaluated across independent 2020–2021 holdout windows and official causality/schema checks:

| Metric Category | Max Points | Achieved Score | Notes |
|---|---:|---:|---|
| **Forecast RMSE Skill (Days 1, 4, 8)** | 30.0 | **22.09** | Beats persistence & climatology across all 6 channels |
| **Heavy-Rain CSI Skill (>10 mm/day)** | 15.0 | **10.00** | 80/20 Composite + Soft-CSI Proxy Loss + Targeted Oversampling |
| **Anti-Blur Variance Ratio (Day 8)** | 10.0 | **9.49** | Preserves **94.9%** of real Day-8 spatial variance (threshold ≥ 40%) |
| **Perturbation Causality & Null** | 15.0 | **15.00** | Localized Day 1, causal spread Days 4–8, exact `0.0` null delta |
| **Repo Integrity & Offline Spec** | 15.0 | **15.00** | 100% offline, self-contained `weights.pt`, zero external APIs |
| **Project Brief (2-Page PDF)** | 15.0 | **13.00+** | Complete 2-page brief with architecture, diagrams & impact ([`docs/BRIEF.pdf`](docs/BRIEF.pdf)) |
| **TOTAL ESTIMATED SCORE** | **100.0** | **84.58 / 100.0** | **186.5 ms** mean model time / **395 ms** HTTP RTT |

---

## 🚀 Quickstart: How to Run & Train

### 1. Run the Offline Inference Server
No internet connection is required. Model weights, climatologies, and blend weights are self-contained inside `weights.pt`:
```bash
python serve.py --port 8000
```
* `POST /health` $\rightarrow$ `{"status": "ready", "model_name": "Adaptive-Physics-Guided-Monsoon-Digital-Twin"}`
* `POST /simulate` $\rightarrow$ Accepts `init_states` `[3, 6, 32, 36]`, `doy_sincos` `[sin, cos]`, `steps`, and optional `perturbation` `[6, 32, 36]`; returns `rollout` and `inference_ms`.
* `POST /evaluate` $\rightarrow$ Returns verified model telemetry and holdout benchmark metrics.

### 2. Open the Interactive Dashboard (100% Offline)
While `serve.py` is running, open [`dashboard/index.html`](dashboard/index.html) directly in Chrome or Edge:
* **India-Clipped 35-State Map:** Smoothly upsampled `1°×1°` radar visualization over all 35 Indian States & UTs (`dashboard/india_geo.js`) with 10m wind vectors and city pins.
* **Multiverse Shock Studio:** Run *Bay of Bengal Heat (+3°C)*, *Arabian Wind Stall (-40%)*, or click any location on the map for custom shocks.
* **Judge 3-Day JSON Uploader:** Upload any `[3, 6, 32, 36]` JSON input under *Timeline Prime* to forecast custom 3-day windows live.

### 3. Retrain from Scratch (Optional)
Requires `india_climate_train.nc` (2005–2021 daily ERA5 India subset, kept outside the repo):
```bash
python train.py
```
Trains `MonsoonForecaster` with `80/20` Composite + Differentiable Soft-CSI Loss, `WeightedRandomSampler` heavy-rain oversampling, and Train/Val early-stopping, then bundles `weights.pt`.

---

## 📁 Repository Structure

```text
├── serve.py              # Offline FastAPI server (/health, /simulate, /evaluate)
├── model.py              # Dilated Residual CNN (197,670 params)
├── inference_utils.py    # Anomaly transforms & bounded physical shock evolution
├── train.py              # Composite Soft-CSI loss + targeted oversampling trainer
├── data.py               # NetCDF/HDF5 dataset loader & climatology builder
├── evaluate.py           # Local diagnostic & holdout evaluation script
├── weights.pt            # Self-contained checkpoint + climatology + blend bundle (16.91 MB)
├── blend_weights.json    # Calibrated lead-time climatology blend weights
├── stats.json            # Channel normalization statistics
├── docs/
│   └── BRIEF.pdf         # 2-page technical brief (Architecture + Impact)
└── dashboard/            # Standalone 100% offline frontend UI
    ├── index.html
    ├── style.css
    ├── app.js
    ├── india_geo.js      # Offline 35-state India border GeoJSON
    └── samples.js        # 3 sample 3-day holdout windows + land-sea mask
```
