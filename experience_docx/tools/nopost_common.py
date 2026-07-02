from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim
import torchvision.transforms.functional as TVF


TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)


IMG_EXT = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SEVERE_DPSNR = -0.20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def first_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"none of {names} under {root}")


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


def train_dirs(data_dir: Path) -> tuple[Path, Path]:
    train = data_dir / "train"
    return first_dir(train, ("IN", "haze", "hazy")), first_dir(train, ("GT", "gt"))


def pad_to(x: torch.Tensor, factor: int = 32) -> tuple[torch.Tensor, int, int, int, int]:
    _, _, h, w = x.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(x, (0, pw, 0, ph), "reflect"), h, w, h + ph, w + pw


def tensor_psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    mse = F.mse_loss(pred, label).clamp_min(1e-12)
    return float((10 * torch.log10(1 / mse)).item())


def metric(pred: torch.Tensor, label: torch.Tensor, hp: int, wp: int) -> tuple[float, float]:
    pred = torch.clamp(pred, 0, 1)
    psnr = tensor_psnr(pred, label)
    down = max(1, round(min(hp, wp) / 256))
    ss = ssim(
        F.adaptive_avg_pool2d(pred, (int(hp / down), int(wp / down))),
        F.adaptive_avg_pool2d(label, (int(hp / down), int(wp / down))),
        data_range=1,
        size_average=False,
    ).mean().item()
    return psnr, float(ss)


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    return TVF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def load_state(path: Path, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def build_official(device: torch.device):
    from models.ConvIR import build_net

    return build_net("base", "Haze4K", "original").to(device)


def build_nopost(device: torch.device, use_detail: bool = False, gate_bias: float = -3.0):
    from models.NoPostFGAConvIR import build_net

    return build_net(
        "base",
        "Haze4K",
        "original",
        nopost_use_low=True,
        nopost_use_detail=use_detail,
        nopost_gate_bias=gate_bias,
    ).to(device)


def load_official_checkpoint(model, checkpoint: Path, device: torch.device) -> None:
    model.load_state_dict(load_state(checkpoint, device))


def partial_load_nopost(model, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    state = load_state(checkpoint, device)
    model_state = model.state_dict()
    loaded = {}
    unexpected = []
    shape_mismatch = []
    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
        elif tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatch.append([key, list(value.shape), list(model_state[key].shape)])
        else:
            loaded[key] = value
    missing = [key for key in model_state if key not in loaded]
    bad_missing = [key for key in missing if not key.startswith("nopost_adapter.")]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            "partial-load failed: "
            f"unexpected={unexpected} shape_mismatch={shape_mismatch} bad_missing={bad_missing[:20]}"
        )
    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    return {
        "loaded_count": len(loaded),
        "missing_new_modules": sorted(missing),
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
    }


def infer_final(model, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    out = model(x)
    pred = out[2] if isinstance(out, (list, tuple)) else out
    return torch.clamp(pred[:, :, :h, :w], 0, 1)


def read_split_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fold_names(train_core: list[str], fold: int, folds: int = 5) -> tuple[list[str], list[str]]:
    val = [name for idx, name in enumerate(train_core) if idx % folds == fold]
    train = [name for idx, name in enumerate(train_core) if idx % folds != fold]
    return train, val


def names_for_scope(split_manifest: Path, scope: str, fold: int = 0, max_images: int = 0) -> list[dict[str, str]]:
    payload = read_split_manifest(split_manifest)
    if scope == "train_core":
        rows = [{"name": name, "split": "train_core"} for name in payload["train_core"]]
    elif scope == "c8_val":
        rows = [{"name": row["name"], "split": row["split"]} for row in payload["val"]]
    elif scope == "fold_train":
        train, _ = fold_names(list(payload["train_core"]), fold)
        rows = [{"name": name, "split": f"fold{fold}_train"} for name in train]
    elif scope == "fold_val":
        _, val = fold_names(list(payload["train_core"]), fold)
        rows = [{"name": name, "split": f"fold{fold}_val"} for name in val]
    else:
        raise ValueError(scope)
    if max_images:
        rows = rows[:max_images]
    return rows


def summarize_delta_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        raise ValueError("no rows to summarize")
    ds = [float(r["dPSNR"]) for r in rows]
    ss = [float(r.get("dSSIM", 0.0)) for r in rows]
    ordered = sorted(rows, key=lambda r: float(r["A0_PSNR"]))
    k = max(1, n // 4)
    hard = [float(r["dPSNR"]) for r in ordered[:k]]
    easy = [float(r["dPSNR"]) for r in ordered[-k:]]
    severe = sum(d <= SEVERE_DPSNR for d in ds)
    return {
        "count": n,
        "mean_dPSNR": statistics.mean(ds),
        "median_dPSNR": statistics.median(ds),
        "hard_bottom25_dPSNR": statistics.mean(hard),
        "easy_top25_dPSNR": statistics.mean(easy),
        "dSSIM": statistics.mean(ss),
        "positive_ratio": sum(d > 0 for d in ds) / n,
        "severe_loss_count": severe,
        "severe_loss_per_600": severe / n * 600.0,
        "p05_dPSNR": sorted(ds)[max(0, math.floor(0.05 * n) - 1)],
        "p95_dPSNR": sorted(ds)[min(n - 1, math.ceil(0.95 * n) - 1)],
    }
