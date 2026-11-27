"""Training pipeline for Indian Climate Digital Twin.

Trains on 2005-2019 data; evaluates on 2020-2021 holdout set.
Includes:
  - Phase 1: Composite Loss (Lat-MSE + Weighted Heavy-Rain MSE + Differentiable Soft-CSI Proxy + Anti-Blur Variance)
  - Phase 2: Targeted Heavy-Rain Oversampling (WeightedRandomSampler) + Physics-Safe Jitter
  - Train vs. Validation Loss monitoring with early-stopping guardrail
Outputs:
  - weights.pt: Self-contained bundle (model_state, climatology, clim_logtp, blend_weights)
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from data import (
    CHANNELS,
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
LOG_TP_10MM = math.log1p(10.0)


class MonsoonDataset(Dataset):
    """Dataset of sliding windows: (3 input days + static + doy) -> (target day, target_clog)."""

    def __init__(
        self,
        anom_data: np.ndarray,
        doys: np.ndarray,
        static: np.ndarray,
        indices: np.ndarray,
        clim_logtp: np.ndarray,
        raw_tp: np.ndarray | None = None,
        augment_heavy: bool = False,
    ):
        self.anom_data = anom_data  # [N, 6, 32, 36]
        self.doys = doys  # [N]
        self.static = static  # [2, 32, 36]
        self.indices = indices
        self.clim_logtp = clim_logtp  # [366, 32, 36]
        self.augment_heavy = augment_heavy
        self.valid_starts = [i for i in range(len(indices) - 3)]

        # Identify true extreme heavy-rain windows (>10 mm/day over >=8% of grid or peak >50 mm/day)
        self.is_heavy = np.zeros(len(self.valid_starts), dtype=bool)
        if raw_tp is not None:
            for k, start in enumerate(self.valid_starts):
                tp_t = raw_tp[start + 3]
                if (tp_t > 10.0).mean() >= 0.08 or tp_t.max() > 50.0:
                    self.is_heavy[k] = True

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.valid_starts[idx]
        in_anoms = self.anom_data[start : start + 3].reshape(3 * NC, NLAT, NLON).copy()

        # Physics-safe jitter exclusively on oversampled heavy-rain weather channels
        # (never alter static geography or wind direction symmetry)
        if self.augment_heavy and self.is_heavy[idx]:
            in_anoms += np.random.normal(0.0, 0.015, size=in_anoms.shape).astype(np.float32)

        last_doy = self.doys[start + 2]
        sin_doy = math.sin(2.0 * math.pi * last_doy / 365.0)
        cos_doy = math.cos(2.0 * math.pi * last_doy / 365.0)
        doy_map = np.empty((2, NLAT, NLON), dtype=np.float32)
        doy_map[0].fill(sin_doy)
        doy_map[1].fill(cos_doy)

        inputs = np.concatenate([in_anoms, self.static, doy_map], axis=0)
        target = self.anom_data[start + 3]
        target_doy = int(self.doys[start + 3]) % 366
        target_clog = self.clim_logtp[target_doy]

        return (
            torch.from_numpy(inputs),
            torch.from_numpy(target),
            torch.from_numpy(target_clog.astype(np.float32)),
        )


def get_lat_weights() -> torch.Tensor:
    """Cosine latitude weights matching R1 scoring contract."""
    lat = np.arange(6.5, 37.5 + 1e-9, 1.0)
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    return torch.from_numpy(w.astype(np.float32)).unsqueeze(1)  # [32, 1]


def compute_composite_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    target_clog: torch.Tensor,
    lat_w: torch.Tensor,
    channel_w: torch.Tensor,
    w_mse: float = 0.70,
    w_focus: float = 0.30,
    lambda_csi: float = 0.25,
) -> tuple[torch.Tensor, float]:
    """Composite Loss: Standard Lat-MSE + Weighted Rain MSE + Differentiable Soft-CSI + Anti-Blur Var."""
    diff_sq = (pred - y) ** 2
    loss_mse = (diff_sq * lat_w * channel_w).mean()

    # Weighted/Focal MSE targeting heavy positive rain anomalies
    rain_focus_w = 1.0 + 3.0 * F.relu(y[:, 1:2])
    loss_focus = ((pred[:, 1:2] - y[:, 1:2]) ** 2 * lat_w * rain_focus_w).mean()

    # Differentiable Soft-CSI Proxy Loss at 10 mm/day physical threshold
    pred_logtp = pred[:, 1] + target_clog
    true_logtp = y[:, 1] + target_clog
    p_hit = torch.sigmoid(4.0 * (pred_logtp - LOG_TP_10MM))
    t_hit = (true_logtp > LOG_TP_10MM).float()
    lw = lat_w.unsqueeze(0)  # [1, 32, 1]
    inter = (p_hit * t_hit * lw).sum()
    union = ((p_hit + t_hit - p_hit * t_hit) * lw).sum()
    soft_csi = (inter + 1e-4) / (union + 1e-4)
    loss_csi = 1.0 - soft_csi

    # Spatial variance loss (anti-blur)
    loss_var = F.l1_loss(pred.var(dim=(-2, -1)), y.var(dim=(-2, -1)))

    total = w_mse * loss_mse + w_focus * loss_focus + lambda_csi * loss_csi + 0.15 * loss_var
    return total, float(soft_csi.item())


def build_oversampled_loader(train_ds: MonsoonDataset, batch_size: int = 64, heavy_target_frac: float = 0.20) -> DataLoader:
    """Creates a DataLoader with Targeted Oversampling so >=15-20% of each batch has heavy rain."""
    n_heavy = int(train_ds.is_heavy.sum())
    n_normal = len(train_ds) - n_heavy
    if n_heavy == 0 or n_normal == 0:
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    w_heavy = heavy_target_frac / n_heavy
    w_normal = (1.0 - heavy_target_frac) / n_normal
    weights = np.where(train_ds.is_heavy, w_heavy, w_normal).astype(np.float64)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    return DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)


def train(
    epochs: int = 12,
    w_mse: float = 0.70,
    w_focus: float = 0.30,
    lambda_csi: float = 0.25,
    warm_start: Path | None = None,
):
    out_dir = Path(__file__).parent

    print("=== 1. Loading Dataset ===")
    raw = load_raw_dataset()
    stats = load_stats()
    train_idx, holdout_idx = train_holdout_split(raw["dates"])
    print(f"Total days: {len(raw['dates'])} | Train: {len(train_idx)} | Holdout: {len(holdout_idx)}")

    print("\n=== 2. Computing Climatology on Training Set ===")
    clim, clim_logtp = compute_climatology(raw["data"], raw["dates"], train_idx)

    print("\n=== 3. Normalizing to Anomalies & Building Targeted Datasets ===")
    import gc
    data_arr = raw["data"]
    doys_arr = raw["doys"]
    anom_all = np.empty_like(data_arr, dtype=np.float32)
    day_idx = doys_arr % 366
    for ch in (0, 2, 3, 4, 5):
        anom_all[:, ch] = (data_arr[:, ch] - clim[day_idx, ch]) / stats["std"][ch]
    anom_all[:, 1] = np.log1p(np.maximum(data_arr[:, 1], 0.0)) - clim_logtp[day_idx]

    static_raw = raw["static"]
    static_norm = np.empty_like(static_raw, dtype=np.float32)
    static_norm[0] = static_raw[0]
    static_norm[1] = static_raw[1] / 1500.0

    train_ds = MonsoonDataset(
        anom_all[train_idx], doys_arr[train_idx], static_norm, train_idx,
        clim_logtp, raw_tp=data_arr[train_idx, 1], augment_heavy=True
    )
    val_ds = MonsoonDataset(
        anom_all[holdout_idx], doys_arr[holdout_idx], static_norm, holdout_idx,
        clim_logtp, raw_tp=None, augment_heavy=False
    )
    del raw, data_arr, anom_all
    gc.collect()
    train_loader = build_oversampled_loader(train_ds, batch_size=64, heavy_target_frac=0.22)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    print(f"Train samples: {len(train_ds)} (Heavy-rain windows: {int(train_ds.is_heavy.sum())}) | Val samples: {len(val_ds)}")

    print(f"\n=== 4. Training Model on {DEVICE} (w_mse={w_mse}, w_focus={w_focus}, lambda_csi={lambda_csi}) ===")
    model = MonsoonForecaster().to(DEVICE)
    lr = 1e-3
    if warm_start is not None and warm_start.is_file():
        ckpt = torch.load(warm_start, map_location="cpu", weights_only=False)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state)
        lr = 3.5e-4
        print(f"Warm-started from {warm_start.name} (lr={lr})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
    lat_w = get_lat_weights().to(DEVICE)
    channel_w = torch.tensor([1.2, 1.5, 1.0, 1.0, 1.0, 1.2], device=DEVICE).view(1, 6, 1, 1)

    best_val_loss = float("inf")
    best_state = None
    overfit_streak = 0
    prev_train_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_csi = 0.0, 0.0
        t0 = time.time()

        for x, y, clog in train_loader:
            x, y, clog = x.to(DEVICE), y.to(DEVICE), clog.to(DEVICE)
            pred = model(x)
            loss, scsi = compute_composite_loss(pred, y, clog, lat_w, channel_w, w_mse, w_focus, lambda_csi)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_csi += scsi

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        train_csi = total_csi / len(train_loader)

        # Validation pass
        model.eval()
        val_loss_sum, val_csi_sum = 0.0, 0.0
        with torch.no_grad():
            for x, y, clog in val_loader:
                x, y, clog = x.to(DEVICE), y.to(DEVICE), clog.to(DEVICE)
                pred = model(x)
                vloss, vcsi = compute_composite_loss(pred, y, clog, lat_w, channel_w, w_mse, w_focus, lambda_csi)
                val_loss_sum += vloss.item()
                val_csi_sum += vcsi
        val_loss = val_loss_sum / len(val_loader)
        val_csi = val_csi_sum / len(val_loader)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.5f} (CSI={train_csi:.3f}) | "
            f"Val Loss: {val_loss:.5f} (CSI={val_csi:.3f}) | {elapsed:.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            overfit_streak = 0
        else:
            if train_loss < prev_train_loss:
                overfit_streak += 1
            if overfit_streak >= 3:
                print(f">>> Early stopping at epoch {epoch}: Val Loss increased for 3 epochs while Train Loss fell.")
                break
        prev_train_loss = train_loss

    if best_state is not None:
        model.load_state_dict(best_state)

    # Bundle using existing optimal blend_weights.json
    blend_path = out_dir / "blend_weights.json"
    blend_weights = json.loads(blend_path.read_text(encoding="utf-8")) if blend_path.is_file() else {}
    bundle = {
        "model_state": model.state_dict(),
        "climatology": clim,
        "clim_logtp": clim_logtp,
        "blend_weights": blend_weights,
    }
    cand_path = out_dir / "weights_candidate.pt"
    torch.save(bundle, cand_path)
    print(f"\nSaved best validation bundle to {cand_path.name} ({cand_path.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    import sys
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    train(epochs=ep)
