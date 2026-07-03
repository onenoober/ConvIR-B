#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TVF


TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)


SEVERE = -0.20
STRONG_REG = -0.05


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def cvar(values: list[float], pct: float = 5.0) -> float:
    if not values:
        return float("nan")
    k = max(1, int(round(len(values) * pct / 100.0)))
    return mean(sorted(values)[:k])


def first_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"none of {names} under {root}")


def train_dirs(data_dir: Path) -> tuple[Path, Path]:
    train = data_dir / "train"
    return first_dir(train, ("IN", "haze", "hazy")), first_dir(train, ("GT", "gt"))


def label_path(gt_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    ext = Path(image_name).suffix
    candidates = [image_name]
    if "_" in stem:
        base = stem.split("_")[0]
        candidates.extend([f"{base}{ext}", f"{base}.png"])
    for candidate in candidates:
        p = gt_dir / candidate
        if p.is_file():
            return p
    raise FileNotFoundError(f"no GT for {image_name} under {gt_dir}")


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    return TVF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def pad_to(x: torch.Tensor, factor: int = 32) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(x, (0, pw, 0, ph), "reflect"), h, w


def load_state(path: Path, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def tensor_psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    mse = F.mse_loss(torch.clamp(pred, 0, 1), label).clamp_min(1e-12)
    return float((10 * torch.log10(1 / mse)).detach().cpu())


def haar_dwt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) / 2.0
    lh = (a - b + c - d) / 2.0
    hl = (a + b - c - d) / 2.0
    hh = (a - b - c + d) / 2.0
    return ll, lh, hl, hh, h, w


def haar_iwt(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor, h: int, w: int) -> torch.Tensor:
    a = (ll + lh + hl + hh) / 2.0
    b = (ll - lh + hl - hh) / 2.0
    c = (ll + lh - hl - hh) / 2.0
    d = (ll - lh - hl + hh) / 2.0
    out = torch.empty(
        (ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2),
        dtype=ll.dtype,
        device=ll.device,
    )
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out[:, :, :h, :w]


def forward_parts(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    x_2 = F.interpolate(x, scale_factor=0.5)
    x_4 = F.interpolate(x_2, scale_factor=0.5)
    z2 = model.SCM2(x_2)
    z4 = model.SCM1(x_4)
    x_ = model.feat_extract[0](x)
    res1 = model.Encoder[0](x_)
    z = model.feat_extract[1](res1)
    z = model.FAM2(z, z2)
    res2 = model.Encoder[1](z)
    z = model.feat_extract[2](res2)
    z = model.FAM1(z, z4)
    z = model.Encoder[2](z)
    z = model.Decoder[0](z)
    z = model.feat_extract[3](z)
    z = torch.cat([z, res2], dim=1)
    z = model.Convs[0](z)
    mid = model.Decoder[1](z)
    z = model.feat_extract[4](mid)
    z = torch.cat([z, res1], dim=1)
    z = model.Convs[1](z)
    final = model.Decoder[2](z)
    return {"res1": res1, "mid": mid, "final": final}


def head_from_final(model: torch.nn.Module, final: torch.Tensor, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return (model.feat_extract[5](final) + x)[:, :, :h, :w]


def final_from_mid(model: torch.nn.Module, mid: torch.Tensor, res1: torch.Tensor) -> torch.Tensor:
    z = model.feat_extract[4](mid)
    z = torch.cat([z, res1], dim=1)
    z = model.Convs[1](z)
    return model.Decoder[2](z)


def channel_scale(ll: torch.Tensor, delta_scale: float) -> torch.Tensor:
    flat = ll.detach().flatten(2)
    scale = flat.std(dim=2).view(ll.shape[0], ll.shape[1], 1, 1)
    return (scale.clamp_min(1e-4) * delta_scale).detach()


def grid_context(ll: torch.Tensor, grid: int) -> torch.Tensor:
    return F.adaptive_avg_pool2d(ll.detach(), (grid, grid)).squeeze(0).cpu()


def upsample_grid(delta_grid: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if delta_grid.dim() == 3:
        delta_grid = delta_grid.unsqueeze(0)
    if delta_grid.shape[-2:] == target_hw:
        return delta_grid
    return F.interpolate(delta_grid, size=target_hw, mode="bilinear", align_corners=False)


def spatial_delta(raw: torch.Tensor, target_hw: tuple[int, int], scale: torch.Tensor) -> torch.Tensor:
    delta = torch.tanh(raw) * scale
    if delta.shape[-2:] != target_hw:
        delta = F.interpolate(delta, size=target_hw, mode="bilinear", align_corners=False)
    return delta


def apply_final_delta(final: torch.Tensor, delta_grid: torch.Tensor) -> torch.Tensor:
    ll, lh, hl, hh, h, w = haar_dwt(final)
    delta = upsample_grid(delta_grid.to(ll.device, dtype=ll.dtype), ll.shape[-2:])
    return haar_iwt(ll + delta, lh, hl, hh, h, w)


def apply_mid_delta(mid: torch.Tensor, delta_grid: torch.Tensor) -> torch.Tensor:
    ll, lh, hl, hh, h, w = haar_dwt(mid)
    delta = upsample_grid(delta_grid.to(ll.device, dtype=ll.dtype), ll.shape[-2:])
    return haar_iwt(ll + delta, lh, hl, hh, h, w)


def optimize_final_grid(
    *,
    model: torch.nn.Module,
    final: torch.Tensor,
    x: torch.Tensor,
    gt: torch.Tensor,
    h: int,
    w: int,
    grid: int,
    steps: int,
    lr: float,
    delta_scale: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base_ll, lh, hl, hh, fh, fw = haar_dwt(final.detach())
    scale = channel_scale(base_ll, delta_scale)
    raw = torch.zeros((1, base_ll.shape[1], grid, grid), device=base_ll.device, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    best_psnr = -1.0
    best_loss = float("inf")
    best_grid: torch.Tensor | None = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        delta = spatial_delta(raw, base_ll.shape[-2:], scale)
        recon = haar_iwt(base_ll + delta, lh, hl, hh, fh, fw)
        pred = head_from_final(model, recon, x, h, w)
        loss = F.l1_loss(torch.clamp(pred, 0, 1), gt) + 1e-4 * delta.abs().mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            psnr = tensor_psnr(pred, gt)
            if psnr > best_psnr:
                best_psnr = psnr
                best_loss = float(loss.detach().cpu())
                best_grid = (torch.tanh(raw.detach()) * scale).squeeze(0).cpu()
    assert best_grid is not None
    return best_grid.float(), {
        "best_psnr": best_psnr,
        "best_loss": best_loss,
        "final_delta_abs_mean": float(best_grid.abs().mean()),
        "final_delta_rms": float(torch.sqrt(torch.mean(best_grid**2))),
        "final_delta_abs_max": float(best_grid.abs().max()),
        "final_delta_spatial_entropy": spatial_entropy(best_grid),
    }


def optimize_mid_final_grids(
    *,
    model: torch.nn.Module,
    mid: torch.Tensor,
    res1: torch.Tensor,
    x: torch.Tensor,
    gt: torch.Tensor,
    h: int,
    w: int,
    mid_grid: int,
    final_grid: int,
    steps: int,
    lr: float,
    delta_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    mid_ll, mid_lh, mid_hl, mid_hh, mh, mw = haar_dwt(mid.detach())
    mid_scale = channel_scale(mid_ll, delta_scale)
    raw_mid = torch.zeros((1, mid_ll.shape[1], mid_grid, mid_grid), device=mid.device, requires_grad=True)
    with torch.no_grad():
        base_final = final_from_mid(model, mid.detach(), res1.detach())
        final_ll, _, _, _, _, _ = haar_dwt(base_final)
        final_scale = channel_scale(final_ll, delta_scale)
    raw_final = torch.zeros((1, final_ll.shape[1], final_grid, final_grid), device=mid.device, requires_grad=True)
    opt = torch.optim.Adam([raw_mid, raw_final], lr=lr)
    best_psnr = -1.0
    best_loss = float("inf")
    best_mid: torch.Tensor | None = None
    best_final: torch.Tensor | None = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        mid_delta = spatial_delta(raw_mid, mid_ll.shape[-2:], mid_scale)
        mid_recon = haar_iwt(mid_ll + mid_delta, mid_lh, mid_hl, mid_hh, mh, mw)
        final = final_from_mid(model, mid_recon, res1.detach())
        ll, lh, hl, hh, fh, fw = haar_dwt(final)
        final_delta = spatial_delta(raw_final, ll.shape[-2:], final_scale)
        final_recon = haar_iwt(ll + final_delta, lh, hl, hh, fh, fw)
        pred = head_from_final(model, final_recon, x, h, w)
        reg = mid_delta.abs().mean() + final_delta.abs().mean()
        loss = F.l1_loss(torch.clamp(pred, 0, 1), gt) + 1e-4 * reg
        loss.backward()
        opt.step()
        with torch.no_grad():
            psnr = tensor_psnr(pred, gt)
            if psnr > best_psnr:
                best_psnr = psnr
                best_loss = float(loss.detach().cpu())
                best_mid = (torch.tanh(raw_mid.detach()) * mid_scale).squeeze(0).cpu()
                best_final = (torch.tanh(raw_final.detach()) * final_scale).squeeze(0).cpu()
    assert best_mid is not None and best_final is not None
    return best_mid.float(), best_final.float(), {
        "best_psnr": best_psnr,
        "best_loss": best_loss,
        "mid_delta_abs_mean": float(best_mid.abs().mean()),
        "mid_delta_rms": float(torch.sqrt(torch.mean(best_mid**2))),
        "mid_delta_abs_max": float(best_mid.abs().max()),
        "mid_delta_spatial_entropy": spatial_entropy(best_mid),
        "final_delta_abs_mean": float(best_final.abs().mean()),
        "final_delta_rms": float(torch.sqrt(torch.mean(best_final**2))),
        "final_delta_abs_max": float(best_final.abs().max()),
        "final_delta_spatial_entropy": spatial_entropy(best_final),
    }


def cosine_pair(
    mid_pred: torch.Tensor,
    final_pred: torch.Tensor,
    mid_target: torch.Tensor,
    final_target: torch.Tensor,
) -> tuple[float, float]:
    a = torch.cat([mid_pred.flatten().float(), final_pred.flatten().float()])
    b = torch.cat([mid_target.flatten().float(), final_target.flatten().float()])
    dot = float(torch.dot(a, b))
    denom = float(torch.linalg.norm(a) * torch.linalg.norm(b))
    return (float(dot / denom) if denom > 1e-12 else 0.0, dot)


def spatial_entropy(delta_grid: torch.Tensor) -> float:
    energy = delta_grid.detach().float().abs().mean(dim=0).flatten()
    total = float(energy.sum())
    if total <= 1e-12:
        return 0.0
    p = (energy / total).clamp_min(1e-12)
    return float(-(p * p.log()).sum() / math.log(float(p.numel())))


def local_peak_ratio(delta_grid: torch.Tensor) -> float:
    energy = delta_grid.detach().float().abs().mean(dim=0)
    mean_energy = float(energy.mean())
    if mean_energy <= 1e-12:
        return 0.0
    return float(energy.max() / mean_energy)


def standardize_spatial(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = train.mean(dim=(0, 2, 3), keepdim=True)
    sigma = train.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    return (train - mu) / sigma, (test - mu) / sigma, mu, sigma


def standardize_matrix(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = train.mean(dim=0, keepdim=True)
    sigma = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train - mu) / sigma, (test - mu) / sigma, mu, sigma


class SmallSpatialCNN(torch.nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=True),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiheadContextPredictor(torch.nn.Module):
    def __init__(
        self,
        *,
        final_channels: int,
        mid_channels: int,
        global_dim: int,
        hidden: int,
        use_final: bool,
        use_mid: bool,
        use_global: bool,
        output_mid: bool,
        output_final: bool,
    ):
        super().__init__()
        self.use_final = use_final
        self.use_mid = use_mid
        self.use_global = use_global
        self.output_mid = output_mid
        self.output_final = output_final
        self.final_encoder = torch.nn.Conv2d(final_channels, hidden, kernel_size=1, bias=True)
        self.mid_encoder = torch.nn.Conv2d(mid_channels, hidden, kernel_size=1, bias=True)
        self.global_encoder = torch.nn.Sequential(
            torch.nn.Linear(global_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
        )
        self.final_body = torch.nn.Sequential(
            torch.nn.Conv2d(hidden * 3, hidden, kernel_size=3, padding=1, bias=True),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            torch.nn.GELU(),
        )
        self.mid_body = torch.nn.Sequential(
            torch.nn.Conv2d(hidden * 3, hidden, kernel_size=3, padding=1, bias=True),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            torch.nn.GELU(),
        )
        self.final_head = torch.nn.Conv2d(hidden, final_channels, kernel_size=1, bias=True)
        self.mid_head = torch.nn.Conv2d(hidden, mid_channels, kernel_size=1, bias=True)

    def _tokens(self, x_final: torch.Tensor, x_mid: torch.Tensor, x_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x_final.shape[0]
        final_zero = torch.zeros((b, self.final_encoder.out_channels, x_final.shape[-2], x_final.shape[-1]), device=x_final.device)
        mid_zero = torch.zeros((b, self.mid_encoder.out_channels, x_mid.shape[-2], x_mid.shape[-1]), device=x_mid.device)
        final_tok = self.final_encoder(x_final) if self.use_final else final_zero
        mid_tok_small = self.mid_encoder(x_mid) if self.use_mid else mid_zero
        global_vec = self.global_encoder(x_global) if self.use_global else torch.zeros((b, self.final_encoder.out_channels), device=x_final.device)
        final_mid_tok = F.interpolate(mid_tok_small, size=x_final.shape[-2:], mode="bilinear", align_corners=False)
        final_global_tok = global_vec[:, :, None, None].expand(-1, -1, x_final.shape[-2], x_final.shape[-1])
        mid_final_tok = F.interpolate(final_tok, size=x_mid.shape[-2:], mode="bilinear", align_corners=False)
        mid_global_tok = global_vec[:, :, None, None].expand(-1, -1, x_mid.shape[-2], x_mid.shape[-1])
        final_features = torch.cat([final_tok, final_mid_tok, final_global_tok], dim=1)
        mid_features = torch.cat([mid_tok_small, mid_final_tok, mid_global_tok], dim=1)
        return final_features, mid_features

    def forward(
        self,
        x_final: torch.Tensor,
        x_mid: torch.Tensor,
        x_global: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        final_features, mid_features = self._tokens(x_final, x_mid, x_global)
        mid_out = self.mid_head(self.mid_body(mid_features)) if self.output_mid else None
        final_out = self.final_head(self.final_body(final_features)) if self.output_final else None
        return mid_out, final_out


def fit_final_cnn(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    hidden: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    model = SmallSpatialCNN(x_train.shape[1], hidden).to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n = x_train.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
    preds = []
    with torch.no_grad():
        for start in range(0, x_test.shape[0], batch_size):
            preds.append(model(x_test[start : start + batch_size]).detach().cpu())
    return torch.cat(preds, dim=0)


def fit_multihead(
    *,
    x_final_train: torch.Tensor,
    x_mid_train: torch.Tensor,
    x_global_train: torch.Tensor,
    y_mid_train: torch.Tensor,
    y_final_train: torch.Tensor,
    x_final_test: torch.Tensor,
    x_mid_test: torch.Tensor,
    x_global_test: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    hidden: int,
    seed: int,
    use_final: bool,
    use_mid: bool,
    use_global: bool,
    output_mid: bool,
    output_final: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = MultiheadContextPredictor(
        final_channels=x_final_train.shape[1],
        mid_channels=x_mid_train.shape[1],
        global_dim=x_global_train.shape[1],
        hidden=hidden,
        use_final=use_final,
        use_mid=use_mid,
        use_global=use_global,
        output_mid=output_mid,
        output_final=output_final,
    ).to(device)
    x_final_train = x_final_train.to(device)
    x_mid_train = x_mid_train.to(device)
    x_global_train = x_global_train.to(device)
    y_mid_train = y_mid_train.to(device)
    y_final_train = y_final_train.to(device)
    x_final_test = x_final_test.to(device)
    x_mid_test = x_mid_test.to(device)
    x_global_test = x_global_test.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n = x_final_train.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            pred_mid, pred_final = model(x_final_train[idx], x_mid_train[idx], x_global_train[idx])
            loss = torch.tensor(0.0, device=device)
            if output_mid and pred_mid is not None:
                loss = loss + F.mse_loss(pred_mid, y_mid_train[idx])
            if output_final and pred_final is not None:
                loss = loss + F.mse_loss(pred_final, y_final_train[idx])
            loss.backward()
            opt.step()
    mid_preds = []
    final_preds = []
    with torch.no_grad():
        for start in range(0, x_final_test.shape[0], batch_size):
            pred_mid, pred_final = model(x_final_test[start : start + batch_size], x_mid_test[start : start + batch_size], x_global_test[start : start + batch_size])
            if pred_mid is None:
                pred_mid = torch.zeros((x_final_test[start : start + batch_size].shape[0], y_mid_train.shape[1], y_mid_train.shape[2], y_mid_train.shape[3]), device=device)
            if pred_final is None:
                pred_final = torch.zeros((x_final_test[start : start + batch_size].shape[0], y_final_train.shape[1], y_final_train.shape[2], y_final_train.shape[3]), device=device)
            mid_preds.append(pred_mid.detach().cpu())
            final_preds.append(pred_final.detach().cpu())
    return torch.cat(mid_preds, dim=0), torch.cat(final_preds, dim=0)


def fit_ridge_broadcast(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    ridge_lambda: float,
) -> torch.Tensor:
    x_train_s, x_test_s, _, _ = standardize_matrix(x_train, x_test)
    y_train_s, _, y_mu, y_sigma = standardize_matrix(y_train, y_train[:1])
    ones_train = torch.ones((x_train_s.shape[0], 1), dtype=x_train_s.dtype)
    ones_test = torch.ones((x_test_s.shape[0], 1), dtype=x_test_s.dtype)
    xb = torch.cat([x_train_s, ones_train], dim=1)
    xt = torch.cat([x_test_s, ones_test], dim=1)
    eye = torch.eye(xb.shape[1], dtype=x_train_s.dtype)
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(xb.T @ xb + ridge_lambda * eye, xb.T @ y_train_s)
    return (xt @ weights) * y_sigma + y_mu


def summarize_rows(rows: list[dict[str, Any]], variant: str, scope: str = "all") -> dict[str, Any]:
    subset = [r for r in rows if r["variant"] == variant]
    vals = [float(r["dPSNR"]) for r in subset]
    a0 = [float(r["A0_PSNR"]) for r in subset]
    hard_cut = percentile(a0, 25)
    easy_cut = percentile(a0, 75)
    hard = [float(r["dPSNR"]) for r in subset if float(r["A0_PSNR"]) <= hard_cut]
    easy = [float(r["dPSNR"]) for r in subset if float(r["A0_PSNR"]) >= easy_cut]
    strong = [float(r["dPSNR"]) for r in subset if float(r["A0_PSNR"]) >= easy_cut]
    strong_easy = [float(r["dPSNR"]) for r in subset if float(r["A0_PSNR"]) >= easy_cut]
    strong_count = len(strong)
    strong_reg = sum(v <= STRONG_REG for v in strong)
    return {
        "variant": variant,
        "scope": scope,
        "count": len(vals),
        "mean_dPSNR": mean(vals),
        "median_dPSNR": median(vals),
        "p01_dPSNR": percentile(vals, 1),
        "p05_dPSNR": percentile(vals, 5),
        "CVaR5_dPSNR": cvar(vals, 5),
        "hard_bottom25_dPSNR": mean(hard),
        "easy_top25_dPSNR": mean(easy),
        "positive_ratio": sum(v > 0 for v in vals) / len(vals) if vals else float("nan"),
        "severe_count": sum(v <= SEVERE for v in vals),
        "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
        "strong_reference_count": strong_count,
        "strong_reference_regressions": strong_reg,
        "strong_reference_regression_rate": strong_reg / strong_count if strong_count else float("nan"),
        "strong_easy_mean_dPSNR": mean(strong_easy),
        "strong_easy_p05_dPSNR": percentile(strong_easy, 5),
    }


def direction_summary(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    subset = [r for r in rows if r["variant"] == variant]
    return {
        "variant": variant,
        "count": len(subset),
        "mean_cosine_to_o3_delta": mean([float(r["cosine_to_o3_delta"]) for r in subset]),
        "median_cosine_to_o3_delta": median([float(r["cosine_to_o3_delta"]) for r in subset]),
        "wrong_direction_rate": mean([float(r["wrong_direction"]) for r in subset]),
        "mean_mid_pred_delta_abs_mean": mean([float(r["mid_pred_delta_abs_mean"]) for r in subset]),
        "mean_final_pred_delta_abs_mean": mean([float(r["final_pred_delta_abs_mean"]) for r in subset]),
        "mean_mid_target_delta_abs_mean": mean([float(r["mid_target_delta_abs_mean"]) for r in subset]),
        "mean_final_target_delta_abs_mean": mean([float(r["final_target_delta_abs_mean"]) for r in subset]),
        "mean_mid_local_peak_ratio": mean([float(r["mid_pred_local_peak_ratio"]) for r in subset]),
        "mean_final_local_peak_ratio": mean([float(r["final_pred_local_peak_ratio"]) for r in subset]),
        "mean_mid_spatial_entropy": mean([float(r["mid_pred_spatial_entropy"]) for r in subset]),
        "mean_final_spatial_entropy": mean([float(r["final_pred_spatial_entropy"]) for r in subset]),
    }


def p1_gates(
    *,
    summary: dict[str, Any],
    shuffled_summary: dict[str, Any],
    direction: dict[str, Any],
    fold_summaries: list[dict[str, Any]],
    fold_report: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    fold_tail_ok = [
        row
        for row in fold_summaries
        if row["variant"] == variant
        and str(row["scope"]).startswith("replay_fold")
        and float(row["easy_top25_dPSNR"]) >= -0.02
        and float(row["p05_dPSNR"]) >= -0.15
        and float(row["severe_rate"]) <= 0.035
    ]
    mse_rows = [row for row in fold_report if row.get("variant") == variant]
    mse_beats = all(
        float(row["target_mse"]) < float(row["shuffled_target_mse"])
        and float(row["target_mse"]) <= float(row["final_only_target_mse"])
        for row in mse_rows
    )
    mechanism_checks = {
        "mean_dPSNR_ge_0p25": float(summary["mean_dPSNR"]) >= 0.25,
        "hard_bottom25_ge_0p50": float(summary["hard_bottom25_dPSNR"]) >= 0.50,
        "positive_ratio_ge_0p60": float(summary["positive_ratio"]) >= 0.60,
        "real_beats_shuffled_by_0p20": float(summary["mean_dPSNR"]) - float(shuffled_summary["mean_dPSNR"]) >= 0.20,
        "wrong_direction_rate_le_0p12": float(direction["wrong_direction_rate"]) <= 0.12,
        "target_mse_beats_shuffled_and_final_only": mse_beats,
    }
    safety_checks = {
        "easy_top25_ge_neg0p02": float(summary["easy_top25_dPSNR"]) >= -0.02,
        "p05_ge_neg0p15": float(summary["p05_dPSNR"]) >= -0.15,
        "CVaR5_ge_neg0p35": float(summary["CVaR5_dPSNR"]) >= -0.35,
        "severe_rate_le_0p035": float(summary["severe_rate"]) <= 0.035,
        "strong_reference_regression_rate_le_0p075": float(summary["strong_reference_regression_rate"]) <= 0.075,
        "fold_tail_pass_ge_4_of_5": len(fold_tail_ok) >= 4,
        "strong_easy_mean_ge_0": float(summary["strong_easy_mean_dPSNR"]) >= 0.0,
        "strong_easy_p05_ge_neg0p15": float(summary["strong_easy_p05_dPSNR"]) >= -0.15,
    }
    return {
        "mechanism_pass": all(mechanism_checks.values()),
        "training_authorization_pass": all(safety_checks.values()),
        "mechanism_checks": mechanism_checks,
        "training_authorization_checks": safety_checks,
        "control_gap_vs_shuffled": float(summary["mean_dPSNR"]) - float(shuffled_summary["mean_dPSNR"]),
        "fold_tail_ok_count": len(fold_tail_ok),
    }


def make_group_flags(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    a0_values = [float(r["A0_PSNR"]) for r in rows]
    hard_cut = percentile(a0_values, 25)
    easy_cut = percentile(a0_values, 75)
    return {
        str(r["name"]): {
            "hard_bottom25": int(float(r["A0_PSNR"]) <= hard_cut),
            "easy_top25": int(float(r["A0_PSNR"]) >= easy_cut),
            "strong_reference": int(float(r["A0_PSNR"]) >= easy_cut),
            "middle50": int(hard_cut < float(r["A0_PSNR"]) < easy_cut),
        }
        for r in rows
    }


def global_feature_row(final_ctx: torch.Tensor, mid_ctx: torch.Tensor, a0_psnr: float) -> torch.Tensor:
    final_mean = final_ctx.mean(dim=(1, 2))
    final_std = final_ctx.flatten(1).std(dim=1)
    mid_mean = mid_ctx.mean(dim=(1, 2))
    mid_std = mid_ctx.flatten(1).std(dim=1)
    scalars = torch.tensor([a0_psnr, float(final_ctx.abs().mean()), float(mid_ctx.abs().mean())], dtype=torch.float32)
    return torch.cat([final_mean, final_std, mid_mean, mid_std, scalars], dim=0)


def generate_targets(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    split_rows = read_csv(args.split_csv)
    names = [row["name"] for row in split_rows]
    if args.max_images:
        names = names[: args.max_images]
    split_by_name = {row["name"]: row for row in split_rows}
    input_dir, gt_dir = train_dirs(args.data_dir)
    final_context_rows: list[torch.Tensor] = []
    mid_context_rows: list[torch.Tensor] = []
    global_rows: list[torch.Tensor] = []
    o2_final_targets: list[torch.Tensor] = []
    o3_mid_targets: list[torch.Tensor] = []
    o3_final_targets: list[torch.Tensor] = []
    meta_rows: list[dict[str, Any]] = []

    for idx, name in enumerate(names, 1):
        hazy = image_tensor(input_dir / name, device)
        gt_full = image_tensor(label_path(gt_dir, name), device)
        x, h, w = pad_to(hazy, 32)
        gt = gt_full[:, :, :h, :w]
        with torch.no_grad():
            parts = forward_parts(model, x)
            a0_pred = head_from_final(model, parts["final"], x, h, w)
            a0_psnr = tensor_psnr(a0_pred, gt)
            final_ll, _, _, _, _, _ = haar_dwt(parts["final"])
            mid_ll, _, _, _, _, _ = haar_dwt(parts["mid"])
            final_ctx = grid_context(final_ll, args.final_grid)
            mid_ctx = grid_context(mid_ll, args.mid_grid)
        o2_final, o2_stats = optimize_final_grid(
            model=model,
            final=parts["final"],
            x=x,
            gt=gt,
            h=h,
            w=w,
            grid=args.final_grid,
            steps=args.steps_o2,
            lr=args.lr,
            delta_scale=args.delta_scale,
        )
        o3_mid, o3_final, o3_stats = optimize_mid_final_grids(
            model=model,
            mid=parts["mid"],
            res1=parts["res1"],
            x=x,
            gt=gt,
            h=h,
            w=w,
            mid_grid=args.mid_grid,
            final_grid=args.final_grid,
            steps=args.steps_o3,
            lr=args.lr,
            delta_scale=args.delta_scale,
        )
        fold = int(split_by_name[name].get("oof_fold", 0))
        final_context_rows.append(final_ctx.float())
        mid_context_rows.append(mid_ctx.float())
        global_rows.append(global_feature_row(final_ctx.float(), mid_ctx.float(), a0_psnr))
        o2_final_targets.append(o2_final.float())
        o3_mid_targets.append(o3_mid.float())
        o3_final_targets.append(o3_final.float())
        meta_rows.append(
            {
                "name": name,
                "oof_fold": fold,
                "A0_PSNR": a0_psnr,
                "o2_oracle_candidate_PSNR": o2_stats["best_psnr"],
                "o2_oracle_dPSNR": o2_stats["best_psnr"] - a0_psnr,
                "o3_oracle_candidate_PSNR": o3_stats["best_psnr"],
                "o3_oracle_dPSNR": o3_stats["best_psnr"] - a0_psnr,
                "o3_minus_o2_dPSNR": o3_stats["best_psnr"] - o2_stats["best_psnr"],
                "o2_final_delta_abs_mean": o2_stats["final_delta_abs_mean"],
                "o2_final_delta_rms": o2_stats["final_delta_rms"],
                "o3_mid_delta_abs_mean": o3_stats["mid_delta_abs_mean"],
                "o3_mid_delta_rms": o3_stats["mid_delta_rms"],
                "o3_final_delta_abs_mean": o3_stats["final_delta_abs_mean"],
                "o3_final_delta_rms": o3_stats["final_delta_rms"],
                "o3_mid_delta_spatial_entropy": o3_stats["mid_delta_spatial_entropy"],
                "o3_final_delta_spatial_entropy": o3_stats["final_delta_spatial_entropy"],
                "final_context_abs_mean": float(final_ctx.abs().mean()),
                "mid_context_abs_mean": float(mid_ctx.abs().mean()),
            }
        )
        if idx % args.print_freq == 0:
            print(f"V220_P1_TARGET {idx}/{len(names)}", flush=True)
    return (
        torch.stack(final_context_rows).float(),
        torch.stack(mid_context_rows).float(),
        torch.stack(global_rows).float(),
        torch.stack(o2_final_targets).float(),
        torch.stack(o3_mid_targets).float(),
        torch.stack(o3_final_targets).float(),
        meta_rows,
    )


def fit_fold_predictors(
    args: argparse.Namespace,
    x_final_all: torch.Tensor,
    x_mid_all: torch.Tensor,
    x_global_all: torch.Tensor,
    y_o2_final: torch.Tensor,
    y_o3_mid: torch.Tensor,
    y_o3_final: torch.Tensor,
    meta_rows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], list[dict[str, Any]]]:
    folds = sorted({int(row["oof_fold"]) for row in meta_rows})
    zero_mid = torch.zeros_like(y_o3_mid)
    zero_final = torch.zeros_like(y_o3_final)
    pred: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "P1_noop_control": (zero_mid.clone(), zero_final.clone()),
        "P1_exact_o2_final_only_oracle_upper_bound": (zero_mid.clone(), y_o2_final.clone()),
        "P1_exact_o3_midfinal_oracle_upper_bound": (y_o3_mid.clone(), y_o3_final.clone()),
        "P1_v219_final_only_spatial_replicate": (zero_mid.clone(), torch.zeros_like(y_o3_final)),
        "P1_mid_only_context_predictor": (torch.zeros_like(y_o3_mid), zero_final.clone()),
        "P1_final_mid_context_predictor": (torch.zeros_like(y_o3_mid), torch.zeros_like(y_o3_final)),
        "P1_final_mid_global_context_predictor": (torch.zeros_like(y_o3_mid), torch.zeros_like(y_o3_final)),
        "P1_shuffled_target_control": (torch.zeros_like(y_o3_mid), torch.zeros_like(y_o3_final)),
        "P1_global_broadcast_control": (torch.zeros_like(y_o3_mid), torch.zeros_like(y_o3_final)),
    }
    fold_report: list[dict[str, Any]] = []
    rng = random.Random(args.seed)

    for fold in folds:
        train_idx = [i for i, row in enumerate(meta_rows) if int(row["oof_fold"]) != fold]
        test_idx = [i for i, row in enumerate(meta_rows) if int(row["oof_fold"]) == fold]
        xf_train, xf_test, _, _ = standardize_spatial(x_final_all[train_idx], x_final_all[test_idx])
        xm_train, xm_test, _, _ = standardize_spatial(x_mid_all[train_idx], x_mid_all[test_idx])
        xg_train, xg_test, _, _ = standardize_matrix(x_global_all[train_idx], x_global_all[test_idx])
        yo2_train, _, yo2_mu, yo2_sigma = standardize_spatial(y_o2_final[train_idx], y_o2_final[test_idx])
        ym_train, _, ym_mu, ym_sigma = standardize_spatial(y_o3_mid[train_idx], y_o3_mid[test_idx])
        yf_train, _, yf_mu, yf_sigma = standardize_spatial(y_o3_final[train_idx], y_o3_final[test_idx])

        final_only_std = fit_final_cnn(
            xf_train,
            yo2_train,
            xf_test,
            device=device,
            epochs=args.cnn_epochs,
            batch_size=args.batch_size,
            hidden=args.cnn_hidden,
            seed=args.seed + fold,
        )
        pred["P1_v219_final_only_spatial_replicate"][1][test_idx] = final_only_std * yo2_sigma + yo2_mu

        mid_std, final_zero_std = fit_multihead(
            x_final_train=xf_train,
            x_mid_train=xm_train,
            x_global_train=xg_train,
            y_mid_train=ym_train,
            y_final_train=yf_train,
            x_final_test=xf_test,
            x_mid_test=xm_test,
            x_global_test=xg_test,
            device=device,
            epochs=args.cnn_epochs,
            batch_size=args.batch_size,
            hidden=args.cnn_hidden,
            seed=args.seed + 100 + fold,
            use_final=False,
            use_mid=True,
            use_global=False,
            output_mid=True,
            output_final=False,
        )
        pred["P1_mid_only_context_predictor"][0][test_idx] = mid_std * ym_sigma + ym_mu
        pred["P1_mid_only_context_predictor"][1][test_idx] = final_zero_std * yf_sigma + yf_mu

        mid_std, final_std = fit_multihead(
            x_final_train=xf_train,
            x_mid_train=xm_train,
            x_global_train=xg_train,
            y_mid_train=ym_train,
            y_final_train=yf_train,
            x_final_test=xf_test,
            x_mid_test=xm_test,
            x_global_test=xg_test,
            device=device,
            epochs=args.cnn_epochs,
            batch_size=args.batch_size,
            hidden=args.cnn_hidden,
            seed=args.seed + 200 + fold,
            use_final=True,
            use_mid=True,
            use_global=False,
            output_mid=True,
            output_final=True,
        )
        pred["P1_final_mid_context_predictor"][0][test_idx] = mid_std * ym_sigma + ym_mu
        pred["P1_final_mid_context_predictor"][1][test_idx] = final_std * yf_sigma + yf_mu

        mid_std, final_std = fit_multihead(
            x_final_train=xf_train,
            x_mid_train=xm_train,
            x_global_train=xg_train,
            y_mid_train=ym_train,
            y_final_train=yf_train,
            x_final_test=xf_test,
            x_mid_test=xm_test,
            x_global_test=xg_test,
            device=device,
            epochs=args.cnn_epochs,
            batch_size=args.batch_size,
            hidden=args.cnn_hidden,
            seed=args.seed + 300 + fold,
            use_final=True,
            use_mid=True,
            use_global=True,
            output_mid=True,
            output_final=True,
        )
        pred["P1_final_mid_global_context_predictor"][0][test_idx] = mid_std * ym_sigma + ym_mu
        pred["P1_final_mid_global_context_predictor"][1][test_idx] = final_std * yf_sigma + yf_mu

        shuffled_order = list(range(len(train_idx)))
        rng.shuffle(shuffled_order)
        mid_std, final_std = fit_multihead(
            x_final_train=xf_train,
            x_mid_train=xm_train,
            x_global_train=xg_train,
            y_mid_train=ym_train[shuffled_order],
            y_final_train=yf_train[shuffled_order],
            x_final_test=xf_test,
            x_mid_test=xm_test,
            x_global_test=xg_test,
            device=device,
            epochs=args.shuffle_epochs,
            batch_size=args.batch_size,
            hidden=args.cnn_hidden,
            seed=args.seed + 400 + fold,
            use_final=True,
            use_mid=True,
            use_global=True,
            output_mid=True,
            output_final=True,
        )
        pred["P1_shuffled_target_control"][0][test_idx] = mid_std * ym_sigma + ym_mu
        pred["P1_shuffled_target_control"][1][test_idx] = final_std * yf_sigma + yf_mu

        global_mid = fit_ridge_broadcast(xg_train, y_o3_mid[train_idx].mean(dim=(2, 3)), xg_test, ridge_lambda=args.ridge_lambda)
        global_final = fit_ridge_broadcast(xg_train, y_o3_final[train_idx].mean(dim=(2, 3)), xg_test, ridge_lambda=args.ridge_lambda)
        pred["P1_global_broadcast_control"][0][test_idx] = global_mid[:, :, None, None].expand(-1, -1, args.mid_grid, args.mid_grid)
        pred["P1_global_broadcast_control"][1][test_idx] = global_final[:, :, None, None].expand(-1, -1, args.final_grid, args.final_grid)

        final_only_mse = float(
            F.mse_loss(pred["P1_v219_final_only_spatial_replicate"][0][test_idx], y_o3_mid[test_idx])
            + F.mse_loss(pred["P1_v219_final_only_spatial_replicate"][1][test_idx], y_o3_final[test_idx])
        )
        shuffled_mse = float(
            F.mse_loss(pred["P1_shuffled_target_control"][0][test_idx], y_o3_mid[test_idx])
            + F.mse_loss(pred["P1_shuffled_target_control"][1][test_idx], y_o3_final[test_idx])
        )
        for variant in (
            "P1_mid_only_context_predictor",
            "P1_final_mid_context_predictor",
            "P1_final_mid_global_context_predictor",
        ):
            mid_pred, final_pred = pred[variant]
            fold_report.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "train_count": len(train_idx),
                    "test_count": len(test_idx),
                    "target_mse": float(F.mse_loss(mid_pred[test_idx], y_o3_mid[test_idx]) + F.mse_loss(final_pred[test_idx], y_o3_final[test_idx])),
                    "mid_target_mse": float(F.mse_loss(mid_pred[test_idx], y_o3_mid[test_idx])),
                    "final_target_mse": float(F.mse_loss(final_pred[test_idx], y_o3_final[test_idx])),
                    "shuffled_target_mse": shuffled_mse,
                    "final_only_target_mse": final_only_mse,
                }
            )
        print(f"V220_P1_FOLD_FIT fold={fold} train={len(train_idx)} test={len(test_idx)}", flush=True)
    return pred, fold_report


def replay_predictions(
    args: argparse.Namespace,
    model: torch.nn.Module,
    pred_by_variant: dict[str, tuple[torch.Tensor, torch.Tensor]],
    y_o3_mid: torch.Tensor,
    y_o3_final: torch.Tensor,
    meta_rows: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    input_dir, gt_dir = train_dirs(args.data_dir)
    replay_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(meta_rows):
        name = str(row["name"])
        hazy = image_tensor(input_dir / name, device)
        gt_full = image_tensor(label_path(gt_dir, name), device)
        x, h, w = pad_to(hazy, 32)
        gt = gt_full[:, :, :h, :w]
        with torch.no_grad():
            parts = forward_parts(model, x)
        target_mid = y_o3_mid[idx]
        target_final = y_o3_final[idx]
        for variant, (pred_mid_all, pred_final_all) in pred_by_variant.items():
            pred_mid = pred_mid_all[idx]
            pred_final = pred_final_all[idx]
            with torch.no_grad():
                if float(pred_mid.abs().max()) > 0:
                    mid_recon = apply_mid_delta(parts["mid"], pred_mid)
                    final_base = final_from_mid(model, mid_recon, parts["res1"])
                else:
                    final_base = parts["final"]
                if float(pred_final.abs().max()) > 0:
                    final_recon = apply_final_delta(final_base, pred_final)
                else:
                    final_recon = final_base
                psnr = tensor_psnr(head_from_final(model, final_recon, x, h, w), gt)
            dpsnr = psnr - float(row["A0_PSNR"])
            cos, dot = cosine_pair(pred_mid, pred_final, target_mid, target_final)
            replay_rows.append(
                {
                    "name": name,
                    "oof_fold": row["oof_fold"],
                    "variant": variant,
                    "A0_PSNR": row["A0_PSNR"],
                    "candidate_PSNR": psnr,
                    "dPSNR": dpsnr,
                    "o2_oracle_dPSNR": row["o2_oracle_dPSNR"],
                    "o3_oracle_dPSNR": row["o3_oracle_dPSNR"],
                    "o3_minus_o2_dPSNR": row["o3_minus_o2_dPSNR"],
                    "mid_pred_delta_abs_mean": float(pred_mid.abs().mean()),
                    "mid_pred_delta_rms": float(torch.sqrt(torch.mean(pred_mid**2))),
                    "mid_pred_delta_abs_max": float(pred_mid.abs().max()),
                    "mid_pred_spatial_entropy": spatial_entropy(pred_mid),
                    "mid_pred_local_peak_ratio": local_peak_ratio(pred_mid),
                    "final_pred_delta_abs_mean": float(pred_final.abs().mean()),
                    "final_pred_delta_rms": float(torch.sqrt(torch.mean(pred_final**2))),
                    "final_pred_delta_abs_max": float(pred_final.abs().max()),
                    "final_pred_spatial_entropy": spatial_entropy(pred_final),
                    "final_pred_local_peak_ratio": local_peak_ratio(pred_final),
                    "mid_target_delta_abs_mean": row["o3_mid_delta_abs_mean"],
                    "mid_target_delta_rms": row["o3_mid_delta_rms"],
                    "final_target_delta_abs_mean": row["o3_final_delta_abs_mean"],
                    "final_target_delta_rms": row["o3_final_delta_rms"],
                    "cosine_to_o3_delta": cos,
                    "dot_to_o3_delta": dot,
                    "wrong_direction": int(dot <= 0.0),
                }
            )
        if (idx + 1) % args.print_freq == 0:
            print(f"V220_P1_REPLAY {idx + 1}/{len(meta_rows)}", flush=True)
    return replay_rows


def run_p1(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    write_text(
        args.out_dir / "v220_p1_o3_context_protocol.md",
        "\n".join(
            [
                "# v2.20 P1 O3 Mid+Final/Context Learnability Protocol",
                "",
                "This audit tests whether deployable mid+final/context features can learn safe NoPost lowband actions.",
                "It is a train-derived learnability and replay audit only: no deployable model training and no locked Haze4K test.",
                "",
                "Controls:",
                "",
                "- exact O2 final-feature LL oracle upper bound",
                "- exact O3 mid+final LL oracle upper bound",
                "- v2.19-style final-only spatial CNN replicate",
                "- mid-only predictor",
                "- final+mid predictor",
                "- final+mid+global context predictor",
                "- shuffled-target control",
                "- global-broadcast control",
                "- no-op control",
                "",
                f"O2 steps: `{args.steps_o2}`; O3 steps: `{args.steps_o3}`; mid grid: `{args.mid_grid}`; final grid: `{args.final_grid}`.",
                f"CNN epochs: `{args.cnn_epochs}`; shuffled epochs: `{args.shuffle_epochs}`.",
            ]
        ),
    )
    (
        x_final_all,
        x_mid_all,
        x_global_all,
        y_o2_final,
        y_o3_mid,
        y_o3_final,
        meta_rows,
    ) = generate_targets(args, model, device)
    write_csv(args.out_dir / "v220_p1_o3_target_energy_summary.csv", meta_rows)
    write_csv(
        args.out_dir / "v220_p1_o3_minus_o2_residual_summary.csv",
        [
            {
                "name": row["name"],
                "oof_fold": row["oof_fold"],
                "A0_PSNR": row["A0_PSNR"],
                "o2_oracle_dPSNR": row["o2_oracle_dPSNR"],
                "o3_oracle_dPSNR": row["o3_oracle_dPSNR"],
                "o3_minus_o2_dPSNR": row["o3_minus_o2_dPSNR"],
            }
            for row in meta_rows
        ],
    )
    pred_by_variant, fold_report = fit_fold_predictors(
        args,
        x_final_all,
        x_mid_all,
        x_global_all,
        y_o2_final,
        y_o3_mid,
        y_o3_final,
        meta_rows,
        device,
    )
    replay_rows = replay_predictions(args, model, pred_by_variant, y_o3_mid, y_o3_final, meta_rows, device)
    variants = sorted(pred_by_variant)
    summary_rows = [summarize_rows(replay_rows, variant) for variant in variants]
    direction_rows = [direction_summary(replay_rows, variant) for variant in variants]
    folds = sorted({int(row["oof_fold"]) for row in meta_rows})
    fold_summary_rows = []
    for variant in variants:
        for fold in folds:
            subset = [r for r in replay_rows if r["variant"] == variant and int(r["oof_fold"]) == fold]
            fold_summary_rows.append(summarize_rows(subset, variant, scope=f"replay_fold{fold}"))

    candidate_variants = [
        "P1_mid_only_context_predictor",
        "P1_final_mid_context_predictor",
        "P1_final_mid_global_context_predictor",
    ]
    shuffled_summary = next(row for row in summary_rows if row["variant"] == "P1_shuffled_target_control")
    direction_by_variant = {row["variant"]: row for row in direction_rows}
    gate_by_variant: dict[str, Any] = {}
    for variant in candidate_variants:
        s = next(row for row in summary_rows if row["variant"] == variant)
        gate_by_variant[variant] = p1_gates(
            summary=s,
            shuffled_summary=shuffled_summary,
            direction=direction_by_variant[variant],
            fold_summaries=fold_summary_rows,
            fold_report=fold_report,
            variant=variant,
        )
        s["mechanism_gate_pass"] = gate_by_variant[variant]["mechanism_pass"]
        s["training_authorization_gate_pass"] = gate_by_variant[variant]["training_authorization_pass"]
        s["control_gap_vs_shuffled"] = gate_by_variant[variant]["control_gap_vs_shuffled"]
        s["fold_tail_ok_count"] = gate_by_variant[variant]["fold_tail_ok_count"]

    mechanism_passing = [variant for variant in candidate_variants if gate_by_variant[variant]["mechanism_pass"]]
    training_passing = [variant for variant in candidate_variants if gate_by_variant[variant]["training_authorization_pass"]]
    primary_variant = max(candidate_variants, key=lambda v: float(next(row for row in summary_rows if row["variant"] == v)["mean_dPSNR"]))
    if training_passing:
        primary_variant = max(training_passing, key=lambda v: float(next(row for row in summary_rows if row["variant"] == v)["mean_dPSNR"]))
        decision = "P1B_PASS_O3_CONTEXT_TRAINING_AUTHORIZATION_REVIEW"
    elif mechanism_passing:
        primary_variant = max(mechanism_passing, key=lambda v: float(next(row for row in summary_rows if row["variant"] == v)["mean_dPSNR"]))
        decision = "P1A_PASS_MECHANISM_P1B_FAIL_CONTINUE_DIAGNOSTICS_NO_TRAINING"
    else:
        decision = "P1A_FAIL_O3_CONTEXT_NOT_SAFELY_LEARNED_NO_TRAINING"
    primary_summary = next(row for row in summary_rows if row["variant"] == primary_variant)
    primary_gate = gate_by_variant.get(primary_variant, {})

    write_csv(args.out_dir / "v220_p1_context_predictor_fold_report.csv", fold_report + fold_summary_rows)
    write_csv(args.out_dir / "v220_p1_replay_metrics.csv", replay_rows)
    write_csv(args.out_dir / "v220_p1_replay_summary.csv", summary_rows)
    write_csv(args.out_dir / "v220_p1_direction_shape_stats.csv", direction_rows)
    write_csv(
        args.out_dir / "v220_p1_controls_summary.csv",
        [
            row
            for row in summary_rows
            if row["variant"]
            in {
                "P1_noop_control",
                "P1_shuffled_target_control",
                "P1_global_broadcast_control",
                "P1_v219_final_only_spatial_replicate",
                "P1_exact_o2_final_only_oracle_upper_bound",
                "P1_exact_o3_midfinal_oracle_upper_bound",
            }
        ],
    )
    v219_like = [r for r in replay_rows if r["variant"] == "P1_v219_final_only_spatial_replicate"]
    flags = make_group_flags(v219_like)
    primary_rows = {r["name"]: r for r in replay_rows if r["variant"] == primary_variant}
    rescue_rows = []
    for row in v219_like:
        name = str(row["name"])
        final_only_failed = float(row["dPSNR"]) <= SEVERE or (flags[name]["strong_reference"] and float(row["dPSNR"]) <= STRONG_REG)
        if final_only_failed:
            current = primary_rows[name]
            rescue_rows.append(
                {
                    "name": name,
                    "oof_fold": row["oof_fold"],
                    "source": "internal_v219_final_only_spatial_replicate",
                    "final_only_dPSNR": row["dPSNR"],
                    "o3_context_dPSNR": current["dPSNR"],
                    "rescued_to_nonsevere": int(float(row["dPSNR"]) <= SEVERE and float(current["dPSNR"]) > SEVERE),
                    "rescued_to_nonregression": int(float(current["dPSNR"]) > STRONG_REG),
                    **flags[name],
                }
            )
    write_csv(args.out_dir / "v220_p1_v219_tail_rescue_manifest.csv", sorted(rescue_rows, key=lambda r: float(r["final_only_dPSNR"])))

    write_text(
        args.out_dir / "v220_p1_decision.md",
        "\n".join(
            [
                "# v2.20 P1 O3 Context Learnability Decision",
                "",
                f"Decision: `{decision}`",
                "",
                f"- primary predictor: `{primary_variant}`",
                f"- mean dPSNR: `{primary_summary['mean_dPSNR']}`",
                f"- hard bottom25 dPSNR: `{primary_summary['hard_bottom25_dPSNR']}`",
                f"- easy top25 dPSNR: `{primary_summary['easy_top25_dPSNR']}`",
                f"- positive ratio: `{primary_summary['positive_ratio']}`",
                f"- p05 dPSNR: `{primary_summary['p05_dPSNR']}`",
                f"- CVaR5 dPSNR: `{primary_summary['CVaR5_dPSNR']}`",
                f"- severe rate: `{primary_summary['severe_rate']}`",
                f"- strong-reference regression rate: `{primary_summary['strong_reference_regression_rate']}`",
                f"- wrong-direction rate: `{direction_by_variant[primary_variant]['wrong_direction_rate']}`",
                f"- control gap vs shuffled: `{primary_gate.get('control_gap_vs_shuffled')}`",
                f"- fold tail pass count: `{primary_gate.get('fold_tail_ok_count')}` / 5",
                "",
                "Mechanism gate checks:",
                "",
                json.dumps(primary_gate.get("mechanism_checks", {}), indent=2, sort_keys=True),
                "",
                "Training-authorization safety gate checks:",
                "",
                json.dumps(primary_gate.get("training_authorization_checks", {}), indent=2, sort_keys=True),
                "",
                "P1 does not launch training. P1-B pass would only authorize a separate route-card review for N3 microfit.",
                "Locked Haze4K remains untouched.",
            ]
        ),
    )
    return {
        "decision": decision,
        "primary_variant": primary_variant,
        "primary_gate": primary_gate,
        "summary": summary_rows,
        "direction": direction_rows,
        "replay_rows": replay_rows,
        "target_rows": meta_rows,
        "fold_report": fold_report,
    }


def train_logistic_classifier(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    epochs: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    x_train_s, x_test_s, _, _ = standardize_matrix(x_train, x_test)
    y_train = y_train.float().view(-1, 1)
    model = torch.nn.Linear(x_train_s.shape[1], 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=1e-3)
    pos = float(y_train.sum())
    neg = float(y_train.numel() - y_train.sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = model(x_train_s)
        loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(x_test_s)).view(-1)


def binary_metrics(probs: list[float], labels: list[int], threshold: float = 0.5) -> dict[str, Any]:
    preds = [int(p >= threshold) for p in probs]
    tp = sum(p == 1 and y == 1 for p, y in zip(preds, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(preds, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(preds, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(preds, labels))
    return {
        "threshold": threshold,
        "count": len(labels),
        "positive_labels": sum(labels),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / len(labels) if labels else float("nan"),
    }


def run_p2(args: argparse.Namespace, p1: dict[str, Any]) -> dict[str, Any]:
    primary = p1["primary_variant"]
    rows = [r for r in p1["replay_rows"] if r["variant"] == primary]
    flags = make_group_flags(rows)
    meta_by_name = {str(r["name"]): r for r in p1["target_rows"]}
    names = [str(r["name"]) for r in rows]
    x_global = []
    unsafe_labels = []
    for row in rows:
        name = str(row["name"])
        m = meta_by_name[name]
        x_global.append(
            torch.tensor(
                [
                    float(row["A0_PSNR"]),
                    float(m["final_context_abs_mean"]),
                    float(m["mid_context_abs_mean"]),
                    float(m["o3_oracle_dPSNR"]),
                    float(m["o3_minus_o2_dPSNR"]),
                    float(row["mid_pred_delta_rms"]),
                    float(row["final_pred_delta_rms"]),
                    float(row["cosine_to_o3_delta"]),
                    float(flags[name]["easy_top25"]),
                    float(flags[name]["strong_reference"]),
                ],
                dtype=torch.float32,
            )
        )
        unsafe_labels.append(int(float(row["dPSNR"]) <= STRONG_REG))
    x_all = torch.stack(x_global)
    y_all = torch.tensor(unsafe_labels, dtype=torch.float32)
    folds = sorted({int(r["oof_fold"]) for r in rows})
    prob_by_name: dict[str, float] = {}
    fold_report = []
    for fold in folds:
        train_idx = [i for i, row in enumerate(rows) if int(row["oof_fold"]) != fold]
        test_idx = [i for i, row in enumerate(rows) if int(row["oof_fold"]) == fold]
        probs = train_logistic_classifier(x_all[train_idx], y_all[train_idx], x_all[test_idx], epochs=args.classifier_epochs, seed=args.seed + 700 + fold)
        labels = [int(y_all[i].item()) for i in test_idx]
        for local_i, global_i in enumerate(test_idx):
            prob_by_name[names[global_i]] = float(probs[local_i])
        report = binary_metrics([float(v) for v in probs], labels, threshold=0.5)
        report.update({"fold": fold, "variant": primary, "train_count": len(train_idx), "test_count": len(test_idx)})
        fold_report.append(report)
    enriched = []
    for row in rows:
        name = str(row["name"])
        enriched.append(
            {
                **row,
                **flags[name],
                "unsafe_action_label": int(float(row["dPSNR"]) <= STRONG_REG),
                "unsafe_action_probability": prob_by_name.get(name, float("nan")),
                "predicted_noop": int(prob_by_name.get(name, 0.0) >= 0.5),
            }
        )
    write_csv(args.out_dir / "v220_p2_noop_or_action_classifier_fold_report.csv", fold_report)
    easy_strong = [r for r in enriched if int(r["easy_top25"]) or int(r["strong_reference"])]
    sensitivity = [
        {
            "group": "easy_or_strong",
            **binary_metrics(
                [float(r["unsafe_action_probability"]) for r in easy_strong],
                [int(r["unsafe_action_label"]) for r in easy_strong],
                threshold=0.5,
            ),
            "mean_dPSNR": mean([float(r["dPSNR"]) for r in easy_strong]),
        },
        {
            "group": "all",
            **binary_metrics(
                [float(r["unsafe_action_probability"]) for r in enriched],
                [int(r["unsafe_action_label"]) for r in enriched],
                threshold=0.5,
            ),
            "mean_dPSNR": mean([float(r["dPSNR"]) for r in enriched]),
        },
    ]
    bins = []
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        subset = [r for r in enriched if lo <= float(r["unsafe_action_probability"]) < hi]
        bins.append(
            {
                "prob_bin": f"[{lo:.1f},{hi:.1f})",
                "count": len(subset),
                "unsafe_rate": mean([float(r["unsafe_action_label"]) for r in subset]),
                "mean_dPSNR": mean([float(r["dPSNR"]) for r in subset]),
                "severe_rate": mean([float(float(r["dPSNR"]) <= SEVERE) for r in subset]),
            }
        )
    ablation = []
    for group in ("all", "hard_bottom25", "easy_top25", "strong_reference"):
        subset = enriched if group == "all" else [r for r in enriched if int(r[group])]
        ablation.append(
            {
                "group": group,
                "count": len(subset),
                "mean_unsafe_probability": mean([float(r["unsafe_action_probability"]) for r in subset]),
                "unsafe_label_rate": mean([float(r["unsafe_action_label"]) for r in subset]),
                "mean_dPSNR": mean([float(r["dPSNR"]) for r in subset]),
            }
        )
    write_csv(args.out_dir / "v220_p2_easy_strong_sensitivity_report.csv", sensitivity)
    write_csv(args.out_dir / "v220_p2_safe_action_confidence_calibration.csv", bins)
    write_csv(args.out_dir / "v220_p2_context_feature_ablation.csv", ablation)
    unsafe_recall = next(r for r in sensitivity if r["group"] == "easy_or_strong")["recall"]
    false_noop_cost = mean([float(r["dPSNR"]) for r in enriched if int(r["predicted_noop"])])
    if unsafe_recall >= 0.85 and false_noop_cost <= 0.10:
        decision = "P2_NOOP_CLASSIFIER_DIAGNOSTIC_POSITIVE_CONTEXT_CAN_FLAG_UNSAFE_ACTIONS"
    elif unsafe_recall >= 0.70:
        decision = "P2_NOOP_CLASSIFIER_PARTIAL_CONTEXT_SIGNAL_NEEDS_CALIBRATION"
    else:
        decision = "P2_NOOP_CLASSIFIER_FAIL_UNSAFE_EASY_STRONG_ACTIONS_NOT_RELIABLY_FLAGGED"
    write_text(
        args.out_dir / "v220_p2_decision.md",
        "\n".join(
            [
                "# v2.20 P2 No-Op / Action Classifier Audit",
                "",
                f"Decision: `{decision}`",
                "",
                f"- primary predictor: `{primary}`",
                f"- easy/strong unsafe recall: `{unsafe_recall}`",
                f"- predicted no-op mean dPSNR: `{false_noop_cost}`",
                "",
                "P2 is diagnostic only and does not authorize training by itself.",
            ]
        ),
    )
    return {"decision": decision, "enriched_rows": enriched, "sensitivity": sensitivity}


def run_p3(args: argparse.Namespace, p1: dict[str, Any]) -> dict[str, Any]:
    primary = p1["primary_variant"]
    rows = [r for r in p1["replay_rows"] if r["variant"] == primary]
    flags = make_group_flags(rows)
    shape_rows = []
    group_rows = []
    for row in rows:
        name = str(row["name"])
        shape_rows.append(
            {
                **row,
                **flags[name],
                "combined_pred_delta_rms": math.sqrt(float(row["mid_pred_delta_rms"]) ** 2 + float(row["final_pred_delta_rms"]) ** 2),
                "combined_target_delta_rms": math.sqrt(float(row["mid_target_delta_rms"]) ** 2 + float(row["final_target_delta_rms"]) ** 2),
                "mid_shape_ratio": float(row["mid_pred_delta_rms"]) / max(float(row["mid_target_delta_rms"]), 1e-12),
                "final_shape_ratio": float(row["final_pred_delta_rms"]) / max(float(row["final_target_delta_rms"]), 1e-12),
            }
        )
    groups = {
        "all": shape_rows,
        "hard_bottom25": [r for r in shape_rows if int(r["hard_bottom25"])],
        "easy_top25": [r for r in shape_rows if int(r["easy_top25"])],
        "strong_reference": [r for r in shape_rows if int(r["strong_reference"])],
        "tail_severe": [r for r in shape_rows if float(r["dPSNR"]) <= SEVERE],
    }
    for group, subset in groups.items():
        group_rows.append(
            {
                "group": group,
                "count": len(subset),
                "mean_dPSNR": mean([float(r["dPSNR"]) for r in subset]),
                "wrong_direction_rate": mean([float(r["wrong_direction"]) for r in subset]),
                "mean_cosine_to_o3_delta": mean([float(r["cosine_to_o3_delta"]) for r in subset]),
                "mean_mid_local_peak_ratio": mean([float(r["mid_pred_local_peak_ratio"]) for r in subset]),
                "mean_final_local_peak_ratio": mean([float(r["final_pred_local_peak_ratio"]) for r in subset]),
                "mean_mid_spatial_entropy": mean([float(r["mid_pred_spatial_entropy"]) for r in subset]),
                "mean_final_spatial_entropy": mean([float(r["final_pred_spatial_entropy"]) for r in subset]),
                "mean_mid_shape_ratio": mean([float(r["mid_shape_ratio"]) for r in subset]),
                "mean_final_shape_ratio": mean([float(r["final_shape_ratio"]) for r in subset]),
            }
        )
    write_csv(args.out_dir / "v220_p3_action_shape_vs_damage.csv", shape_rows)
    write_csv(args.out_dir / "v220_p3_local_peak_entropy_report.csv", group_rows)
    write_csv(args.out_dir / "v220_p3_tail_case_shape_manifest.csv", sorted([r for r in shape_rows if float(r["dPSNR"]) <= SEVERE or int(r["wrong_direction"])], key=lambda r: float(r["dPSNR"]))[:240])
    tail = next(r for r in group_rows if r["group"] == "tail_severe")
    all_group = next(r for r in group_rows if r["group"] == "all")
    if tail["count"] and (
        float(tail["mean_final_local_peak_ratio"]) > float(all_group["mean_final_local_peak_ratio"]) * 1.25
        or float(tail["mean_mid_local_peak_ratio"]) > float(all_group["mean_mid_local_peak_ratio"]) * 1.25
    ):
        decision = "P3_SHAPE_FAIL_TAIL_ACTIONS_PEAKIER_THAN_GLOBAL_CONTEXT_REVIEW_MASK_OR_REGULARIZER"
    elif float(all_group["wrong_direction_rate"]) > 0.12:
        decision = "P3_SHAPE_FAIL_WRONG_DIRECTION_TOO_HIGH"
    elif tail["count"]:
        decision = "P3_SHAPE_DIAG_TAIL_DAMAGE_NOT_EXPLAINED_BY_DIRECTION_OR_PEAK_ONLY"
    else:
        decision = "P3_SHAPE_PASS_NO_SEVERE_TAIL_DAMAGE_FOUND"
    write_text(
        args.out_dir / "v220_p3_decision.md",
        "\n".join(
            [
                "# v2.20 P3 Action Shape Decomposition",
                "",
                f"Decision: `{decision}`",
                "",
                f"- primary predictor: `{primary}`",
                f"- all wrong-direction rate: `{all_group['wrong_direction_rate']}`",
                f"- tail severe count: `{tail['count']}`",
                "",
                "P3 is diagnostic only and does not authorize training by itself.",
            ]
        ),
    )
    return {"decision": decision, "group_rows": group_rows}


def run_p4(args: argparse.Namespace, p1: dict[str, Any]) -> dict[str, Any]:
    primary = p1["primary_variant"]
    rows = [r for r in p1["replay_rows"] if r["variant"] == primary]
    flags = make_group_flags(rows)
    oracle_norms = [math.sqrt(float(r["mid_target_delta_rms"]) ** 2 + float(r["final_target_delta_rms"]) ** 2) for r in rows]
    pred_norms = [math.sqrt(float(r["mid_pred_delta_rms"]) ** 2 + float(r["final_pred_delta_rms"]) ** 2) for r in rows]
    thresholds = {
        "oracle_p75": percentile(oracle_norms, 75),
        "oracle_p90": percentile(oracle_norms, 90),
        "predicted_p75": percentile(pred_norms, 75),
    }
    per_image = []
    for row, oracle_norm, pred_norm in zip(rows, oracle_norms, pred_norms):
        name = str(row["name"])
        dpsnr = float(row["dPSNR"])
        strong_or_easy = bool(flags[name]["strong_reference"] or flags[name]["easy_top25"])
        item = {
            **row,
            **flags[name],
            "combined_oracle_delta_rms": oracle_norm,
            "combined_pred_delta_rms": pred_norm,
            "tail_hinge_margin_neg0p15": max(0.0, -0.15 - dpsnr),
            "severe_hinge_margin_neg0p20": max(0.0, SEVERE - dpsnr),
            "preserve_mask_strong_or_easy": int(strong_or_easy),
            "preserve_hinge_margin_neg0p05": max(0.0, STRONG_REG - dpsnr) if strong_or_easy else 0.0,
            "tail_hinge_active": int(dpsnr < -0.15),
            "preserve_hinge_active": int(strong_or_easy and dpsnr <= STRONG_REG),
            "positive_sample": int(dpsnr > 0),
        }
        for label, threshold in thresholds.items():
            item[f"budget_{label}_active"] = int(pred_norm > threshold)
            item[f"safe_oracle_over_budget_{label}"] = int(oracle_norm > threshold and float(row["o3_oracle_dPSNR"]) > 0)
        per_image.append(item)
    severe_rows = [r for r in per_image if float(r["dPSNR"]) <= SEVERE]
    positive_rows = [r for r in per_image if int(r["positive_sample"])]
    preserve_fail = [r for r in per_image if int(r["preserve_mask_strong_or_easy"]) and float(r["dPSNR"]) <= STRONG_REG]
    tail_report = [
        {
            "primary_variant": primary,
            "count": len(per_image),
            "severe_count": len(severe_rows),
            "tail_hinge_active_count": sum(int(r["tail_hinge_active"]) for r in per_image),
            "tail_hinge_coverage_on_severe": sum(int(r["tail_hinge_active"]) for r in severe_rows) / len(severe_rows) if severe_rows else 1.0,
            "positive_tail_hinge_activation_rate": sum(int(r["tail_hinge_active"]) for r in positive_rows) / len(positive_rows) if positive_rows else 0.0,
            "mean_tail_hinge": mean([float(r["tail_hinge_margin_neg0p15"]) for r in per_image]),
        }
    ]
    preserve_report = [
        {
            "primary_variant": primary,
            "strong_or_easy_count": sum(int(r["preserve_mask_strong_or_easy"]) for r in per_image),
            "strong_or_easy_regression_count": len(preserve_fail),
            "preserve_hinge_active_count": sum(int(r["preserve_hinge_active"]) for r in per_image),
            "preserve_hinge_coverage_on_regressions": sum(int(r["preserve_hinge_active"]) for r in preserve_fail) / len(preserve_fail) if preserve_fail else 1.0,
            "mean_preserve_hinge": mean([float(r["preserve_hinge_margin_neg0p05"]) for r in per_image]),
        }
    ]
    budget_report = []
    safe_oracle_report = []
    for label, threshold in thresholds.items():
        budget_report.append(
            {
                "primary_variant": primary,
                "threshold_label": label,
                "threshold": threshold,
                "budget_activation_rate": mean([float(r[f"budget_{label}_active"]) for r in per_image]),
                "positive_budget_activation_rate": mean([float(r[f"budget_{label}_active"]) for r in positive_rows]),
                "severe_budget_activation_rate": mean([float(r[f"budget_{label}_active"]) for r in severe_rows]),
            }
        )
        safe_oracle_report.append(
            {
                "primary_variant": primary,
                "threshold_label": label,
                "threshold": threshold,
                "safe_oracle_overpenalty_rate": mean([float(r[f"safe_oracle_over_budget_{label}"]) for r in per_image]),
            }
        )
    write_csv(args.out_dir / "v220_p4_per_image_loss_terms.csv", per_image)
    write_csv(args.out_dir / "v220_p4_tail_hinge_activation_report.csv", tail_report)
    write_csv(args.out_dir / "v220_p4_preserve_hinge_activation_report.csv", preserve_report)
    write_csv(args.out_dir / "v220_p4_budget_activation_report.csv", budget_report)
    write_csv(args.out_dir / "v220_p4_safe_oracle_overpenalty_report.csv", safe_oracle_report)
    budget_ok = any(0.02 <= float(r["budget_activation_rate"]) <= 0.25 for r in budget_report)
    if tail_report[0]["tail_hinge_coverage_on_severe"] >= 0.95 and preserve_report[0]["preserve_hinge_coverage_on_regressions"] >= 0.95 and budget_ok:
        decision = "P4_OBJECTIVE_REPLAY_PASS_AS_GUARD_ONLY"
    else:
        decision = "P4_OBJECTIVE_REPLAY_FAIL_OR_OVERWEAK_GUARDS"
    write_text(
        args.out_dir / "v220_p4_decision.md",
        "\n".join(
            [
                "# v2.20 P4 Objective Replay",
                "",
                f"Decision: `{decision}`",
                "",
                f"- primary predictor: `{primary}`",
                f"- tail coverage on severe: `{tail_report[0]['tail_hinge_coverage_on_severe']}`",
                f"- preserve coverage on regressions: `{preserve_report[0]['preserve_hinge_coverage_on_regressions']}`",
                f"- budget nonzero/not-oracle-killing candidate exists: `{budget_ok}`",
                "",
                "P4 guard pass is not training authorization; P1-B must pass before any N3 route-card review.",
            ]
        ),
    )
    return {"decision": decision, "budget_report": budget_report}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--steps-o2", type=int, default=25)
    ap.add_argument("--steps-o3", type=int, default=18)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--delta-scale", type=float, default=0.50)
    ap.add_argument("--final-grid", type=int, default=16)
    ap.add_argument("--mid-grid", type=int, default=8)
    ap.add_argument("--cnn-hidden", type=int, default=64)
    ap.add_argument("--cnn-epochs", type=int, default=180)
    ap.add_argument("--shuffle-epochs", type=int, default=100)
    ap.add_argument("--classifier-epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=220)
    ap.add_argument("--print-freq", type=int, default=25)
    args = ap.parse_args()

    from models.ConvIR import build_net

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_net("base", "Haze4K", "original").to(device)
    model.load_state_dict(load_state(args.checkpoint, device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    p1 = run_p1(args, model, device)
    p2 = run_p2(args, p1)
    p3 = run_p3(args, p1)
    p4 = run_p4(args, p1)
    training_authorized = bool(p1["primary_gate"].get("training_authorization_pass", False))
    if training_authorized:
        decision = "V220_P1B_PASS_REVIEW_N3_MICROFIT_ROUTE_CARD_NO_TRAINING_LAUNCHED"
    elif bool(p1["primary_gate"].get("mechanism_pass", False)):
        decision = "V220_P1A_PASS_P1B_FAIL_NORMAL_GATE_PAUSE_NO_TRAINING"
    else:
        decision = "V220_P1A_FAIL_NORMAL_GATE_PAUSE_NO_TRAINING"
    closeout = {
        "decision": decision,
        "p1_decision": p1["decision"],
        "p2_decision": p2["decision"],
        "p3_decision": p3["decision"],
        "p4_decision": p4["decision"],
        "primary_variant": p1["primary_variant"],
        "training_authorized": training_authorized,
        "training_launched": False,
        "locked_test_touched": False,
    }
    write_json(args.out_dir / "v220_p1_p2_p3_p4_closeout.json", closeout)
    print("V220_P1_P2_P3_P4_OK", decision, flush=True)


if __name__ == "__main__":
    main()
