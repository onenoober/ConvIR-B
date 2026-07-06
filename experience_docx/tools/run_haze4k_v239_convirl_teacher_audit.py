#!/usr/bin/env python3
"""Haze4K v2.39 ConvIR-L same-family teacher contract audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.ConvIR import build_net  # noqa: E402

ROUTE_ID = "haze4k_v2_39_convirl_same_family_teacher_contract_audit_20260706"
ALPHAS = (0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)
SEVERE_THRESHOLD = -0.30
STRONG_REFERENCE_REGRESSION_THRESHOLD = -0.05


def fnum(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def cvar_low(values: list[float], frac: float = 0.05) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = max(1, math.ceil(len(ordered) * frac))
    return statistics.mean(ordered[:k])


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alpha_label(alpha: float) -> str:
    text = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return "a" + text.replace(".", "p")


def load_checkpoint_model(path: str | Path, map_location: Any) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def build_large(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_net("large", "Haze4K", "original").to(device)
    model.load_state_dict(load_checkpoint_model(checkpoint, device), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_tensor(path: str | Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(tensor, dict) and "tensor" in tensor:
        tensor = tensor["tensor"]
    if not torch.is_tensor(tensor):
        raise TypeError(f"expected tensor at {path}")
    tensor = tensor.detach().float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"expected CHW/NCHW tensor at {path}, got shape {tuple(tensor.shape)}")
    if tensor.shape[1] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return torch.clamp(tensor, 0.0, 1.0)


def load_image_tensor(path: str | Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().clamp(0.0, 1.0)


def save_tensor(path: str | Path, tensor: torch.Tensor, *, fp16: bool = True) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = tensor.detach().cpu()
    if fp16:
        payload = payload.to(torch.float16)
    torch.save(payload, path)
    return file_sha256(path)


def pad_to_factor(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    return F.pad(x, (0, pad_w, 0, pad_h), "reflect"), h, w


def infer_large(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    padded, h, w = pad_to_factor(x, 32)
    out = model(padded)[2]
    return torch.clamp(out[:, :, :h, :w], 0, 1)


def psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    if tuple(pred.shape) != tuple(label.shape):
        raise ValueError(f"shape mismatch pred={tuple(pred.shape)} label={tuple(label.shape)}")
    mse = (torch.clamp(pred, 0, 1) - torch.clamp(label, 0, 1)).pow(2).flatten(1).mean().clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse).item())


def blend(a0: torch.Tensor, expert: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.clamp(a0 + alpha * (expert - a0), 0.0, 1.0)


def hardness_bucket(row: dict[str, Any]) -> str:
    bucket = str(row.get("hardness_bucket") or row.get("selection_source_bucket") or "")
    if bucket.startswith("hard"):
        return "hard"
    if bucket.startswith("easy") or bucket.startswith("strong"):
        return "easy"
    return "mid"


def summarize_values(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    hard = [float(row[key]) for row in rows if row.get(key) not in ("", None) and hardness_bucket(row) == "hard"]
    easy = [float(row[key]) for row in rows if row.get(key) not in ("", None) and hardness_bucket(row) == "easy"]
    mid = [float(row[key]) for row in rows if row.get(key) not in ("", None) and hardness_bucket(row) == "mid"]
    strong = [
        float(row[key])
        for row in rows
        if row.get(key) not in ("", None) and row.get("strong_reference_bucket") == "strong_reference"
    ]
    severe = [value for value in values if value <= SEVERE_THRESHOLD]
    strong_regress = [value for value in strong if value < STRONG_REFERENCE_REGRESSION_THRESHOLD]
    return {
        "mean": mean(values),
        "hard": mean(hard),
        "easy": mean(easy),
        "mid": mean(mid),
        "p05": percentile(values, 5),
        "CVaR5": cvar_low(values),
        "worst_delta": min(values) if values else None,
        "negative_count": sum(1 for value in values if value < 0),
        "severe_count": len(severe),
        "severe_rate": len(severe) / len(values) if values else None,
        "strong_reference_count": len(strong),
        "strong_reference_regression_count": len(strong_regress),
        "strong_reference_regression_rate": len(strong_regress) / len(strong) if strong else None,
        "sample_count": len(values),
        "severe_threshold_dB": SEVERE_THRESHOLD,
        "strong_reference_regression_threshold_dB": STRONG_REFERENCE_REGRESSION_THRESHOLD,
    }


def gate_stats(stats: dict[str, Any]) -> bool:
    return bool(
        (stats.get("mean") is not None and stats["mean"] >= 0.30)
        and (stats.get("hard") is not None and stats["hard"] >= 0.50)
        and (stats.get("easy") is not None and stats["easy"] >= 0.05)
        and (stats.get("p05") is not None and stats["p05"] >= 0.00)
        and (stats.get("CVaR5") is not None and stats["CVaR5"] >= -0.01)
        and (stats.get("worst_delta") is not None and stats["worst_delta"] >= -0.05)
        and stats.get("severe_count") == 0
        and stats.get("strong_reference_regression_count") == 0
    )


def assign_buckets(rows: list[dict[str, Any]]) -> None:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_id.setdefault(str(row["image_id"]), row)
    unique = list(by_id.values())
    sorted_rows = sorted(unique, key=lambda row: float(row["A0_same_context_psnr"]))
    bucket_count = max(1, len(sorted_rows) // 4)
    hard_ids = {str(row["image_id"]) for row in sorted_rows[:bucket_count]}
    easy_ids = {str(row["image_id"]) for row in sorted_rows[-bucket_count:]}
    strong_cut = percentile([float(row["A0_same_context_psnr"]) for row in unique], 75) or 0.0
    for row in rows:
        image_id = str(row["image_id"])
        if image_id in hard_ids:
            row["hardness_bucket"] = "hard"
        elif image_id in easy_ids:
            row["hardness_bucket"] = "easy"
        else:
            row["hardness_bucket"] = "mid"
        row["easy_bucket"] = "easy_top25" if image_id in easy_ids else "not_easy_top25"
        row["strong_reference_cut_psnr"] = strong_cut
        row["strong_reference_bucket"] = (
            "strong_reference" if float(row["A0_same_context_psnr"]) >= strong_cut else "not_strong_reference"
        )


def unique_source_rows(source_p0_csv: Path) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for row in read_csv(source_p0_csv):
        image_id = row.get("image_id")
        if image_id and image_id not in by_id:
            by_id[image_id] = row
    rows = list(by_id.values())
    rows.sort(key=lambda item: item.get("image_id", ""))
    return rows


def fold_pass(rows: list[dict[str, Any]], key: str) -> tuple[int, list[dict[str, Any]]]:
    count = 0
    summaries = []
    for fold_id in range(5):
        subset = [row for row in rows if int(row["fold_id"]) == fold_id]
        stats = summarize_values(subset, key)
        passed = gate_stats(stats)
        count += int(passed)
        summaries.append({"fold_id": fold_id, "gate_pass": passed, **stats})
    return count, summaries


def run_p0(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    source_rows = unique_source_rows(args.source_p0_csv)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = build_large(args.checkpoint, device)
    rows: list[dict[str, Any]] = []
    cache_ready = 0
    for index, rec in enumerate(source_rows, start=1):
        image_id = rec["image_id"]
        a0_path = rec["A0_full_output_path"]
        gt_path = rec["gt_path"]
        teacher_path = args.cache_root / image_id / "ConvIRL_full.pt"
        input_tensor = load_image_tensor(rec["input_path"])
        a0 = load_tensor(a0_path)
        gt = load_image_tensor(gt_path)
        if teacher_path.exists() and not args.recompute:
            convirl = load_tensor(teacher_path)
            teacher_sha = file_sha256(teacher_path)
        else:
            with torch.no_grad():
                convirl = infer_large(model, input_tensor.to(device)).detach().cpu()
            teacher_sha = save_tensor(teacher_path, convirl, fp16=True)
        a0_psnr = psnr(a0, gt)
        convirl_psnr = psnr(convirl, gt)
        if teacher_path.exists() and a0_path and gt_path:
            cache_ready += 1
        for alpha in ALPHAS:
            teacher = blend(a0, convirl, alpha)
            teacher_psnr = psnr(teacher, gt)
            delta = teacher_psnr - a0_psnr
            rows.append({
                "image_id": image_id,
                "image_name": rec.get("image_name", ""),
                "sample_index": rec.get("sample_index", ""),
                "alpha": alpha,
                "alpha_label": alpha_label(alpha),
                "context_contract": "full_image_same_context",
                "A0_same_context_psnr": a0_psnr,
                "ConvIRL_full_psnr": convirl_psnr,
                "teacher_alpha_psnr": teacher_psnr,
                "delta_vs_A0": delta,
                "fold_id": int(float(rec.get("fold_id", (index - 1) % 5))),
                "negative": delta < 0,
                "severe": delta <= SEVERE_THRESHOLD,
                "input_path": rec.get("input_path", ""),
                "gt_path": gt_path,
                "A0_full_output_path": a0_path,
                "ConvIRL_full_output_path": str(teacher_path),
                "ConvIRL_full_output_sha256": teacher_sha,
                "locked_test_touched": False,
            })
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if index % 25 == 0 or index == len(source_rows):
            print(f"v239_p0_progress {index}/{len(source_rows)} {image_id}", flush=True)
    assign_buckets(rows)
    for row in rows:
        row["strong_reference_regression"] = (
            row["strong_reference_bucket"] == "strong_reference"
            and float(row["delta_vs_A0"]) < STRONG_REFERENCE_REGRESSION_THRESHOLD
        )
    write_csv(args.out_dir / "v239_p0_convirl_fullimage_teacher_sweep_per_image.csv", rows)
    cache_sha_coverage = cache_ready / len(source_rows) if source_rows else 0.0
    alpha_summaries = []
    passing = []
    for alpha in ALPHAS:
        subset = [row for row in rows if abs(float(row["alpha"]) - alpha) < 1e-12]
        stats = summarize_values(subset, "delta_vs_A0")
        fold_count, fold_summary = fold_pass(subset, "delta_vs_A0")
        gate_pass = bool(stats.get("sample_count") == 600 and cache_sha_coverage == 1.0 and gate_stats(stats) and fold_count == 5)
        if gate_pass:
            passing.append(alpha)
        alpha_summaries.append({
            "alpha": alpha,
            "alpha_label": alpha_label(alpha),
            "gate_pass": gate_pass,
            "fold_pass": f"{fold_count}/5",
            "fold_pass_count": fold_count,
            "summary": stats,
            "fold_summary": fold_summary,
        })
    selected_alpha = max(passing) if passing else None
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P0 ConvIR-L full-image same-context teacher alpha sweep",
        "locked_test_touched": False,
        "source_p0_csv": str(args.source_p0_csv),
        "checkpoint": str(args.checkpoint),
        "cache_root": str(args.cache_root),
        "image_count": len(source_rows),
        "row_count": len(rows),
        "cache_sha_coverage": cache_sha_coverage,
        "alpha_summaries": alpha_summaries,
        "passing_alphas": passing,
        "selected_alpha": selected_alpha,
        "gate_pass": bool(passing),
        "decision": "P0_PASS_CONVIRL_SAFE_TEACHER_ALPHA" if passing else "P0_FAIL_CONVIRL_NO_SAFE_TEACHER_ALPHA",
        "p1_free_tensor_projection_authorized": bool(passing),
        "gate": {
            "image_count": 600,
            "cache_sha_coverage": 1.0,
            "mean_delta_min_dB": 0.30,
            "hard_delta_min_dB": 0.50,
            "easy_delta_min_dB": 0.05,
            "p05_min_dB": 0.00,
            "CVaR5_min_dB": -0.01,
            "worst_delta_min_dB": -0.05,
            "severe_count": 0,
            "strong_reference_regression_count": 0,
            "fold_pass": "5/5",
        },
    }
    write_json(args.out_dir / "v239_p0_convirl_fullimage_teacher_sweep_summary.json", payload)
    write_json(args.out_dir / "v239_p0_closeout.json", payload)
    write_json(args.out_dir / "v239_closeout.json", {
        "route_id": ROUTE_ID,
        "inherited_reference": "v2.38 P0_FAIL_NO_MICROALPHA_SAFE_SUBSTRATE and v2.38B P0_FAIL_RICH_TARGET_ONLY_SEPARABILITY_DIAGNOSTIC",
        "locked_test_touched": False,
        "bridge_training_authorized": False,
        "generator_training_authorized": False,
        "canary80_authorized": False,
        "selected_alpha": selected_alpha,
        "p0_convirl_teacher_sweep_pass": bool(passing),
        "p1_free_tensor_projection_pass": None,
        "decision": payload["decision"],
    })
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V239_P0_CONVIRL_TEACHER_SWEEP_OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-p0-csv", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-large.pkl"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recompute", action="store_true")
    return parser


def main() -> None:
    run_p0(build_parser().parse_args())


if __name__ == "__main__":
    main()
