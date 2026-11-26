"""Production inference server for Indian Climate Digital Twin.

Rules & Contract:
  - Listens on 0.0.0.0:8000
  - Fully offline, self-contained within the repo
  - POST /health -> {"status": "ready", "model_name": ...}
  - POST /simulate -> {"rollout": [steps, 6, 32, 36], "inference_ms": float}
  - Inference returns within 5.0 seconds on standard CPU
  - Null shock consistency guaranteed (zero perturbation = zero change)
"""
from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import sys
import time
from typing import Optional

# Ensure torch, scipy and modern fastapi from environments are accessible
venv_site = Path(__file__).parent / ".venv" / "Lib" / "site-packages"
if venv_site.is_dir() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))

py310_site = r"C:\Users\kumar\AppData\Local\Programs\Python\Python310\lib\site-packages"
if py310_site not in sys.path:
    sys.path.append(py310_site)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import uvicorn

from inference_utils import (
    CHANNELS,
    NC,
    NLAT,
    NLON,
    evolve_physical_shock,
    from_anomaly,
    load_stats,
    to_anomaly,
)
from model import MonsoonForecaster

MODEL_NAME = "Adaptive-Physics-Guided-Monsoon-Digital-Twin"
ROOT = Path(__file__).parent
stats = load_stats(ROOT / "stats.json")

# Load artifacts
clim_path = ROOT / "climatology.npy"
clim_logtp_path = ROOT / "clim_logtp.npy"
weights_path = ROOT / "weights.pt"
blend_path = ROOT / "blend_weights.json"

clim: Optional[np.ndarray] = None
clim_logtp: Optional[np.ndarray] = None
blend_weights: dict = {}
static_norm: Optional[np.ndarray] = None
model: Optional[MonsoonForecaster] = None


def init_engine():
    global clim, clim_logtp, blend_weights, static_norm, model
    if model is not None:
        return

    # Load climatology
    if clim_path.is_file():
        clim = np.load(clim_path)
    if clim_logtp_path.is_file():
        clim_logtp = np.load(clim_logtp_path)

    # Load blend weights
    if blend_path.is_file():
        with open(blend_path, "r", encoding="utf-8") as f:
            blend_weights = json.load(f)
    else:
        blend_weights = {}

    # Load static fields from nc file or fallback
    nc_candidates = [ROOT / "india_climate_train.nc", ROOT.parent / "india_climate_train.nc"]
    s_loaded = False
    for p in nc_candidates:
        if p.is_file():
            try:
                import h5py

                with h5py.File(p, "r") as f:
                    if "static" in f:
                        raw_static = f["static"][:].astype(np.float32)
                        static_norm = np.empty_like(raw_static)
                        static_norm[0] = raw_static[0]
                        static_norm[1] = raw_static[1] / 1500.0
                        s_loaded = True
                        break
            except Exception:
                pass

    if not s_loaded:
        static_norm = np.zeros((2, NLAT, NLON), dtype=np.float32)

    # Load model
    m = MonsoonForecaster()
    if weights_path.is_file():
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            m.load_state_dict(ckpt["model_state"])
            if clim is None and "climatology" in ckpt:
                clim = ckpt["climatology"]
            if clim_logtp is None and "clim_logtp" in ckpt:
                clim_logtp = ckpt["clim_logtp"]
            if not blend_weights and "blend_weights" in ckpt:
                blend_weights = ckpt["blend_weights"]
        else:
            m.load_state_dict(ckpt)
    m.eval()
    model = m


def _arr(x) -> np.ndarray:
    """Accepts nested lists (the contract) or {'b64': ..., 'shape': [...]}."""
    if isinstance(x, dict):
        return np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    return np.asarray(x, dtype="f4")


def simulate(
    init_states: np.ndarray,
    doy_sincos: list[float],
    steps: int,
    perturbation: Optional[np.ndarray],
) -> np.ndarray:
    """Computes a 10-day weather forecast rollout across all 6 channels.

    Args:
      init_states: [3, 6, 32, 36] physical units, oldest first
      doy_sincos: [sin, cos] for the last input day (init_states[-1])
      steps: forecast lead days (typically 10)
      perturbation: [6, 32, 36] physical delta, or None
    Returns:
      [steps, 6, 32, 36] physical units
    """
    init_engine()

    # Determine day-of-year of the last input day
    sin_val, cos_val = float(doy_sincos[0]), float(doy_sincos[1])
    angle = math.atan2(sin_val, cos_val)
    if angle < 0:
        angle += 2.0 * math.pi
    init_doy = int(round(angle / (2.0 * math.pi) * 365.0)) % 365

    # 1. Base Rollout: Convert 3 input days to normalized anomalies
    anom_history = [
        to_anomaly(init_states[k], (init_doy - 2 + k) % 366, clim, clim_logtp, stats)
        for k in range(3)
    ]

    base_rollout = []
    with torch.no_grad():
        for step in range(1, steps + 1):
            cur_doy = (init_doy + step) % 366
            sin_d = math.sin(2.0 * math.pi * cur_doy / 365.0)
            cos_d = math.cos(2.0 * math.pi * cur_doy / 365.0)

            doy_map = np.empty((2, NLAT, NLON), dtype=np.float32)
            doy_map[0].fill(sin_d)
            doy_map[1].fill(cos_d)

            in_18 = np.concatenate(anom_history[-3:], axis=0)
            feat = np.concatenate([in_18, static_norm, doy_map], axis=0)
            feat_t = torch.from_numpy(feat).unsqueeze(0)

            pred_anom_t = model(feat_t)
            pred_anom = pred_anom_t.squeeze(0).numpy()

            # Apply empirical lead-time blending per channel
            step_key = str(min(step, 10))
            blended_anom = pred_anom.copy()
            if step_key in blend_weights:
                for ch, ch_name in enumerate(CHANNELS):
                    w = blend_weights[step_key].get(ch_name, 0.70)
                    blended_anom[ch] = pred_anom[ch] * w

            anom_history.append(blended_anom)

            # Invert to physical units
            phys_step = from_anomaly(blended_anom, cur_doy, clim, clim_logtp, stats)
            base_rollout.append(phys_step)

    base_rollout = np.stack(base_rollout, axis=0)  # [steps, 6, 32, 36]

    # 2. Multiverse / Shock Branch
    if perturbation is None or np.all(perturbation == 0):
        # Null shock: output strictly equals unperturbed rollout (100% null consistency)
        return base_rollout

    # Perturbed branch: evolve physical shock
    u_wind = init_states[-1, 2]
    v_wind = init_states[-1, 3]

    final_rollout = base_rollout.copy()
    for step in range(1, steps + 1):
        delta = evolve_physical_shock(perturbation, u_wind, v_wind, step, stats["std"])
        final_rollout[step - 1] += delta

    # Enforce contract safety clamps
    final_rollout[:, 1] = np.maximum(final_rollout[:, 1], 0.0)  # rain non-negative
    final_rollout[:, 0] = np.clip(final_rollout[:, 0], 233.15, 333.15)  # -40 C to +60 C
    final_rollout[:, 5] = np.maximum(final_rollout[:, 5], 0.0)  # tcwv non-negative

    return np.nan_to_num(final_rollout, nan=292.0, posinf=330.0, neginf=240.0).astype(np.float32)


app = FastAPI(title="Indian Climate Digital Twin")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/health")
def health():
    init_engine()
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    t0 = time.perf_counter()
    body = await request.json()
    init = _arr(body["init_states"])
    pert = None if body.get("perturbation") is None else _arr(body["perturbation"])
    steps = int(body.get("steps", 10))
    doy = body["doy_sincos"]

    out = simulate(init, doy, steps, pert)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "rollout": np.asarray(out, dtype="f4").tolist(),
        "inference_ms": elapsed_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    init_engine()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
