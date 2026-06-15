from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim
from torchvision.transforms import functional as TVF


IMG_EXT = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SEVERE = -0.20


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


def load_split_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_train_core_names(path: Path) -> list[str]:
    return list(load_split_manifest(path)["train_core"])


def load_val_rows(path: Path) -> list[dict[str, str]]:
    return list(load_split_manifest(path)["val"])


def load_teacher_metric_map(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def stable_subset(names: list[str], count: int, seed: int) -> list[str]:
    if count <= 0 or count >= len(names):
        return list(names)
    rng = random.Random(seed)
    selected = list(names)
    rng.shuffle(selected)
    return sorted(selected[:count])


def pad_to(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int, int, int]:
    _, _, h, w = x.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(x, (0, pw, 0, ph), "reflect"), h, w, h + ph, w + pw


def tensor_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).clamp_min(1e-12)
    return float((10 * torch.log10(1 / mse)).item())


def metric(pred: torch.Tensor, label: torch.Tensor, hp: int, wp: int) -> tuple[float, float]:
    psnr = tensor_psnr(pred, label)
    down = max(1, round(min(hp, wp) / 256))
    ss = ssim(
        F.adaptive_avg_pool2d(pred, (int(hp / down), int(wp / down))),
        F.adaptive_avg_pool2d(label, (int(hp / down), int(wp / down))),
        data_range=1,
        size_average=False,
    ).mean().item()
    return psnr, float(ss)


def load_convir(convir_dir: Path):
    convir_dir = convir_dir.resolve()
    if str(convir_dir) not in sys.path:
        sys.path.insert(0, str(convir_dir))
    from models.ConvIR import build_net  # type: ignore

    return build_net


def load_c13(convir_dir: Path):
    convir_dir = convir_dir.resolve()
    if str(convir_dir) not in sys.path:
        sys.path.insert(0, str(convir_dir))
    from models.C13Residual import build_net  # type: ignore

    return build_net


def load_model_state(path: Path, device: torch.device) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"expected state dict in {path}")
    return state


def load_a0_model(convir_dir: Path, checkpoint: Path, device: torch.device):
    build_net = load_convir(convir_dir)
    model = build_net("base", "Haze4K", "original").to(device)
    model.load_state_dict(load_model_state(checkpoint, device))
    model.eval()
    return model


def load_c13_model(
    convir_dir: Path,
    a0_checkpoint: Path,
    checkpoint: Path | None,
    device: torch.device,
    feature_mode: str,
    adapter_width: int,
    adapter_depth: int,
    bootstrap_scale: float,
):
    build_net = load_c13(convir_dir)
    model = build_net(
        "base",
        "Haze4K",
        a0_checkpoint=str(a0_checkpoint),
        feature_mode=feature_mode,
        adapter_width=adapter_width,
        adapter_depth=adapter_depth,
        bootstrap_scale=bootstrap_scale,
    ).to(device)
    if checkpoint is not None:
        model.load_state_dict(load_model_state(checkpoint, device), strict=True)
    model.eval()
    return model


def infer_final(model, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    out = model(x)
    pred = out[2] if isinstance(out, (list, tuple)) else out
    return torch.clamp(pred[:, :, :h, :w], 0, 1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[pos]


def summarize_rows(rows: list[dict[str, Any]], dkey: str = "dPSNR") -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"count": 0}
    ds = [float(r[dkey]) for r in rows]
    ss = [float(r.get("dSSIM", 0.0)) for r in rows]
    ordered = sorted(rows, key=lambda r: float(r.get("A0_PSNR", 0.0)))
    k = max(1, n // 4)
    hard = [float(r[dkey]) for r in ordered[:k]]
    easy = [float(r[dkey]) for r in ordered[-k:]]
    severe = sum(d <= SEVERE for d in ds)
    return {
        "count": n,
        "mean_dPSNR": sum(ds) / n,
        "median_dPSNR": sorted(ds)[n // 2],
        "hard_bottom25_dPSNR": sum(hard) / len(hard),
        "easy_top25_dPSNR": sum(easy) / len(easy),
        "dSSIM": sum(ss) / len(ss),
        "positive_ratio": sum(d > 0 for d in ds) / n,
        "nonnegative_ratio": sum(d >= 0 for d in ds) / n,
        "severe_loss_count": severe,
        "severe_loss_per_600": severe / n * 600.0,
        "p05_dPSNR": quantile(ds, 0.05),
        "p95_dPSNR": quantile(ds, 0.95),
    }


def image_tensor(path: Path) -> torch.Tensor:
    return TVF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)


def list_image_names(path: Path) -> list[str]:
    return sorted(p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def finite_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / max(1, len(vals))
