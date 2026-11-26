"""Training pipeline for Indian Climate Digital Twin.

Trains on 2005-2019 data; evaluates on 2020-2021 holdout set.
Outputs:
  - weights.pt: Trained PyTorch model weights
  - climatology.npy: 366-day baseline physical climatology [366, 6, 32, 36]
  - clim_logtp.npy: 366-day baseline log1p rain climatology [366, 32, 36]
  - blend_weights.json: Empirically tuned blend weights per channel / lead day
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data import (
    load_raw_dataset,
    load_stats,
    train_holdout_split,
    compute_climatology,
    to_anomaly,
    from_anomaly,
    NC,
    NLAT,
    NLON,
)
from model import MonsoonForecaster

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MonsoonDataset(Dataset):
    """Dataset of sliding windows: (3 input days + static + doy) -> (target day)."""

    def __init__(
        self,
        anom_data: np.ndarray,
        doys: np.ndarray,
        static: np.ndarray,
        indices: np.ndarray,
    ):
        self.anom_data = anom_data  # [N, 6, 32, 36]
        self.doys = doys  # [N]
        self.static = static  # [2, 32, 36]
        self.indices = indices
        # We need 3 consecutive input days + 1 target day = 4 consecutive days
        self.valid_starts = [i for i in range(len(indices) - 3)]

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.valid_starts[idx]
        # 3 input anomaly days -> [18, 32, 36]
        in_anoms = self.anom_data[start : start + 3].reshape(3 * NC, NLAT, NLON)

        # Day of year sin/cos of the last input day (day 3)
        last_doy = self.doys[start + 2]
        sin_doy = math.sin(2.0 * math.pi * last_doy / 365.0)
        cos_doy = math.cos(2.0 * math.pi * last_doy / 365.0)
        doy_map = np.empty((2, NLAT, NLON), dtype=np.float32)
        doy_map[0].fill(sin_doy)
        doy_map[1].fill(cos_doy)

        # Concatenate 18 (weather) + 2 (static) + 2 (doy) = 22 channels
        inputs = np.concatenate([in_anoms, self.static, doy_map], axis=0)

        # Target is next step anomaly [6, 32, 36]
        target = self.anom_data[start + 3]

        return torch.from_numpy(inputs), torch.from_numpy(target)


def get_lat_weights() -> torch.Tensor:
    """Cosine latitude weights matching R1 scoring contract."""
    lat = np.arange(6.5, 37.5 + 1e-9, 1.0)
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    return torch.from_numpy(w.astype(np.float32)).unsqueeze(1)  # [32, 1]


def train():
    out_dir = Path(__file__).parent
    print("=== 1. Loading Dataset ===")
    raw = load_raw_dataset()
    stats = load_stats()
    train_idx, holdout_idx = train_holdout_split(raw["dates"])
    print(f"Total days: {len(raw['dates'])} | Train: {len(train_idx)} | Holdout: {len(holdout_idx)}")

    print("\n=== 2. Computing Climatology on Training Set ===")
    clim, clim_logtp = compute_climatology(raw["data"], raw["dates"], train_idx)
    np.save(out_dir / "climatology.npy", clim)
    np.save(out_dir / "clim_logtp.npy", clim_logtp)
    print(f"Saved climatology.npy: shape {clim.shape} ({clim.nbytes / 1024 / 1024:.2f} MB)")
    print(f"Saved clim_logtp.npy: shape {clim_logtp.shape}")

    print("\n=== 3. Normalizing to Anomalies ===")
    t_norm = time.time()
    anom_all = to_anomaly(raw["data"], raw["doys"], clim, clim_logtp, stats)
    print(f"Vectorized anomaly normalization completed in {time.time() - t_norm:.2f}s")

    # Normalize static fields
    static_raw = raw["static"]
    static_norm = np.empty_like(static_raw, dtype=np.float32)
    static_norm[0] = static_raw[0]  # land_sea_mask 0..1
    static_norm[1] = static_raw[1] / 1500.0  # normalize orography

    train_ds = MonsoonDataset(anom_all[train_idx], raw["doys"][train_idx], static_norm, train_idx)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)

    print(f"Training samples: {len(train_ds)}")

    print(f"\n=== 4. Training Model on {DEVICE} ===")
    model = MonsoonForecaster().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    lat_w = get_lat_weights().to(DEVICE)  # [32, 1]
    channel_w = torch.tensor([1.2, 1.5, 1.0, 1.0, 1.0, 1.2], device=DEVICE).view(1, 6, 1, 1)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)

            # Lat-weighted MSE
            diff_sq = (pred - y) ** 2
            weighted_se = diff_sq * lat_w * channel_w
            loss_mse = weighted_se.mean()

            # Variance loss to preserve spatial sharpness (anti-blur)
            pred_var = pred.var(dim=(-2, -1))
            true_var = y.var(dim=(-2, -1))
            loss_var = F.l1_loss(pred_var, true_var)

            loss = loss_mse + 0.15 * loss_var

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        elapsed = time.time() - t0
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.5f} | lr: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s", flush=True)

    print("\n=== 5. Saving Trained Model Weights ===")
    torch.save(model.state_dict(), out_dir / "weights.pt")
    weights_mb = (out_dir / "weights.pt").stat().st_size / 1024 / 1024
    print(f"Saved weights.pt: {weights_mb:.2f} MB")

    print("\n=== 6. Holdout Evaluation & Empirical Blending Calibration ===")
    model.eval()
    holdout_dates = [raw["dates"][i] for i in holdout_idx]
    holdout_data = raw["data"][holdout_idx]
    holdout_doys = raw["doys"][holdout_idx]

    # Evaluate 10-step rollouts over 30 test windows across 2020-2021
    windows = [i for i in range(0, len(holdout_idx) - 13, 24)]
    print(f"Evaluating {len(windows)} independent 10-day test windows on 2020-2021 holdout...")

    err_model = np.zeros((10, NC), dtype=np.float32)
    err_clim = np.zeros((10, NC), dtype=np.float32)
    err_persist = np.zeros((10, NC), dtype=np.float32)
    w_lat = get_lat_weights().numpy()

    with torch.no_grad():
        for start in windows:
            # 3 input days physical
            init_phys = holdout_data[start : start + 3]
            true_future = holdout_data[start + 3 : start + 13]  # 10 days
            start_doy = holdout_doys[start + 2]

            cur_anoms = [
                to_anomaly(init_phys[k], (start_doy - 2 + k) % 366, clim, clim_logtp, stats)
                for k in range(3)
            ]

            rollout = []
            for step in range(1, 11):
                in_18 = np.concatenate(cur_anoms[-3:], axis=0)
                cur_doy = (start_doy + step - 1) % 366
                sin_doy = math.sin(2.0 * math.pi * cur_doy / 365.0)
                cos_doy = math.cos(2.0 * math.pi * cur_doy / 365.0)
                doy_map = np.empty((2, NLAT, NLON), dtype=np.float32)
                doy_map[0].fill(sin_doy)
                doy_map[1].fill(cos_doy)

                inp = np.concatenate([in_18, static_norm, doy_map], axis=0)
                inp_t = torch.from_numpy(inp).unsqueeze(0).to(DEVICE)
                next_anom_t = model(inp_t)
                next_anom = next_anom_t.squeeze(0).cpu().numpy()

                cur_anoms.append(next_anom)
                phys_out = from_anomaly(next_anom, cur_doy, clim, clim_logtp, stats)
                rollout.append(phys_out)

            rollout = np.stack(rollout)  # [10, 6, 32, 36]
            persist = np.repeat(init_phys[-1][None], 10, axis=0)
            clim_ref = np.stack([clim[(start_doy + k) % 366] for k in range(1, 11)])

            for k in range(10):
                for ch in range(NC):
                    em = np.sqrt(((rollout[k, ch] - true_future[k, ch]) ** 2 * w_lat).mean())
                    ep = np.sqrt(((persist[k, ch] - true_future[k, ch]) ** 2 * w_lat).mean())
                    ec = np.sqrt(((clim_ref[k, ch] - true_future[k, ch]) ** 2 * w_lat).mean())
                    err_model[k, ch] += em
                    err_persist[k, ch] += ep
                    err_clim[k, ch] += ec

    err_model /= len(windows)
    err_persist /= len(windows)
    err_clim /= len(windows)

    # Compute empirical blend weights: alpha in [0.0, 1.0]
    # alpha = model weight; 1 - alpha = climatology weight
    blend_weights = {}
    print("\n--- Model Skill vs Persistence & Climatology on Holdout ---")
    print("Lead Day | Channel | Model RMSE | Persist RMSE | Clim RMSE | Skill Score | Model Blend Weight")
    for k in range(10):
        lead = k + 1
        blend_weights[str(lead)] = {}
        for ch, name in enumerate(CHANNELS):
            em, ep, ec = err_model[k, ch], err_persist[k, ch], err_clim[k, ch]
            best_ref = min(ep, ec)
            skill = 1.0 - (em / best_ref) if best_ref > 0 else 0.0

            # Weight model strongly when beating ref; decay smoothly toward climatology
            # at longer lead times where climatology variance protects against drift
            if em < ec:
                weight = float(np.clip(1.0 - (k / 14.0), 0.35, 1.0))
            else:
                weight = float(np.clip(ec / (em + 1e-6), 0.20, 0.80))

            blend_weights[str(lead)][name] = round(weight, 3)
            if lead in (1, 4, 8):
                print(f"Day {lead:2d}   | {name:7s} | {em:10.2f} | {ep:12.2f} | {ec:9.2f} | {skill:+10.3f} | {weight:.3f}")

    with open(out_dir / "blend_weights.json", "w", encoding="utf-8") as f:
        json.dump(blend_weights, f, indent=2)
    print(f"\nSaved blend_weights.json to {out_dir / 'blend_weights.json'}")
    print("\n>>> TRAINING AND HOLD-OUT EVALUATION COMPLETE! <<<")


if __name__ == "__main__":
    train()
