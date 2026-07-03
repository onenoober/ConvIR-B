#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from nopost_common import (
    build_nopost,
    image_tensor,
    pad_to,
    partial_load_nopost,
    sha256,
    train_dirs,
    write_csv,
    write_json,
)


RISK_THRESHOLDS = (-0.10, -0.20, -0.30)
SPATIAL_MAP_NAMES = (
    "hazy_brightness",
    "hazy_dark",
    "hazy_saturation",
    "hazy_gradient",
    "hazy_local_variance",
    "hazy_low_haze_proxy",
    "final_norm",
    "final_local_variance",
    "final_detail_energy",
    "res1_norm",
    "res2_norm",
    "scm2_norm",
    "scm4_norm",
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def same_size(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def local_mean(x: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def channel_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x * x, dim=1, keepdim=True).clamp_min(1e-12))


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def tensor_scalar(x: torch.Tensor) -> float:
    return float(x.detach().cpu())


def quantile_value(flat: torch.Tensor, q: float) -> float:
    return tensor_scalar(torch.quantile(flat, q))


def top_fraction_mean(flat: torch.Tensor, frac: float) -> float:
    n = flat.numel()
    k = max(1, int(math.ceil(n * frac)))
    return tensor_scalar(torch.topk(flat, k=k, largest=True).values.mean())


def gini_value(flat: torch.Tensor) -> float:
    values = torch.sort(flat.clamp_min(0).reshape(-1)).values
    n = values.numel()
    total = values.sum()
    if n == 0 or tensor_scalar(total) <= 1e-12:
        return 0.0
    index = torch.arange(1, n + 1, device=values.device, dtype=values.dtype)
    return tensor_scalar((2.0 * (index * values).sum()) / (n * total) - (n + 1.0) / n)


def entropy_value(flat: torch.Tensor) -> float:
    values = flat.clamp_min(0).reshape(-1)
    total = values.sum()
    n = values.numel()
    if n <= 1 or tensor_scalar(total) <= 1e-12:
        return 0.0
    probs = values / total
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum() / math.log(n)
    return tensor_scalar(entropy)


def connected_components(mask: torch.Tensor) -> tuple[int, float]:
    arr = mask.detach().cpu().numpy().astype("uint8")
    h, w = arr.shape
    seen = [[False for _ in range(w)] for _ in range(h)]
    total_active = int(arr.sum())
    if total_active == 0:
        return 0, 0.0
    count = 0
    largest = 0
    for y in range(h):
        for x in range(w):
            if arr[y, x] == 0 or seen[y][x]:
                continue
            count += 1
            size = 0
            q: deque[tuple[int, int]] = deque([(y, x)])
            seen[y][x] = True
            while q:
                cy, cx = q.popleft()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and arr[ny, nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        q.append((ny, nx))
            largest = max(largest, size)
    return count, largest / total_active


def component_stats(x: torch.Tensor) -> dict[str, float]:
    small = x
    h, w = x.shape[-2:]
    if max(h, w) > 64:
        scale = 64 / max(h, w)
        small = F.interpolate(x, size=(max(4, int(round(h * scale))), max(4, int(round(w * scale)))), mode="bilinear", align_corners=False)
    flat = small.reshape(-1)
    thresh = torch.quantile(flat, 0.90)
    count, largest = connected_components((small[0, 0] >= thresh))
    return {
        "component_count_p90": float(count),
        "largest_component_ratio_p90": float(largest),
    }


def patch_grid_stats(x: torch.Tensor, grid: int) -> dict[str, float]:
    mean = F.adaptive_avg_pool2d(x, (grid, grid))
    sq_mean = F.adaptive_avg_pool2d(x * x, (grid, grid))
    std = torch.sqrt((sq_mean - mean * mean).clamp_min(0))
    mx = F.adaptive_max_pool2d(x, (grid, grid))
    out: dict[str, float] = {}
    for stat_name, values in (("patch_mean", mean), ("patch_std", std), ("patch_max", mx)):
        flat = values.reshape(-1)
        out[f"g{grid}_{stat_name}_mean"] = tensor_scalar(flat.mean())
        out[f"g{grid}_{stat_name}_std"] = tensor_scalar(flat.std(unbiased=False))
        out[f"g{grid}_{stat_name}_max"] = tensor_scalar(flat.max())
        out[f"g{grid}_{stat_name}_p90"] = quantile_value(flat, 0.90)
        out[f"g{grid}_{stat_name}_p95"] = quantile_value(flat, 0.95)
        out[f"g{grid}_{stat_name}_top1"] = tensor_scalar(torch.topk(flat, k=1).values.mean())
        out[f"g{grid}_{stat_name}_top3"] = tensor_scalar(torch.topk(flat, k=min(3, flat.numel())).values.mean())
    return out


def map_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    flat = x.reshape(-1)
    mean = flat.mean()
    std = flat.std(unbiased=False)
    threshold = mean + std
    out = {
        f"{prefix}_mean": tensor_scalar(mean),
        f"{prefix}_std": tensor_scalar(std),
        f"{prefix}_min": tensor_scalar(flat.min()),
        f"{prefix}_max": tensor_scalar(flat.max()),
        f"{prefix}_p01": quantile_value(flat, 0.01),
        f"{prefix}_p05": quantile_value(flat, 0.05),
        f"{prefix}_p10": quantile_value(flat, 0.10),
        f"{prefix}_p25": quantile_value(flat, 0.25),
        f"{prefix}_p50": quantile_value(flat, 0.50),
        f"{prefix}_p75": quantile_value(flat, 0.75),
        f"{prefix}_p90": quantile_value(flat, 0.90),
        f"{prefix}_p95": quantile_value(flat, 0.95),
        f"{prefix}_p99": quantile_value(flat, 0.99),
        f"{prefix}_top_1pct_mean": top_fraction_mean(flat, 0.01),
        f"{prefix}_top_5pct_mean": top_fraction_mean(flat, 0.05),
        f"{prefix}_top_10pct_mean": top_fraction_mean(flat, 0.10),
        f"{prefix}_high_energy_area_ratio": tensor_scalar((x >= threshold).float().mean()),
        f"{prefix}_gini": gini_value(flat),
        f"{prefix}_entropy": entropy_value(flat),
    }
    for key, value in component_stats(x).items():
        out[f"{prefix}_{key}"] = value
    for grid in (4, 8, 16):
        for key, value in patch_grid_stats(x, grid).items():
            out[f"{prefix}_{key}"] = value
    return out


def vector_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    return map_stats(prefix, channel_norm(x))


def extract_audit_features(model, x: torch.Tensor) -> dict[str, torch.Tensor]:
    x_2 = F.interpolate(x, scale_factor=0.5)
    x_4 = F.interpolate(x_2, scale_factor=0.5)
    z2 = model.SCM2(x_2)
    z4 = model.SCM1(x_4)

    x0 = model.feat_extract[0](x)
    res1 = model.Encoder[0](x0)

    z_pre2 = model.feat_extract[1](res1)
    z_fam2 = model.FAM2(z_pre2, z2)
    fam2_response = z_fam2 - z_pre2
    res2 = model.Encoder[1](z_fam2)

    z_pre1 = model.feat_extract[2](res2)
    z_fam1 = model.FAM1(z_pre1, z4)
    fam1_response = z_fam1 - z_pre1
    z = model.Encoder[2](z_fam1)

    z = model.Decoder[0](z)
    up2 = model.feat_extract[3](z)
    skip2_disagreement = up2 - res2

    z = torch.cat([up2, res2], dim=1)
    z = model.Convs[0](z)
    z = model.Decoder[1](z)
    up1 = model.feat_extract[4](z)
    skip1_disagreement = up1 - res1

    z = torch.cat([up1, res1], dim=1)
    z = model.Convs[1](z)
    final_feature = model.Decoder[2](z)
    return {
        "res1": res1,
        "res2": res2,
        "scm2": z2,
        "scm4": z4,
        "final_feature": final_feature,
        "fam1_response": fam1_response,
        "fam2_response": fam2_response,
        "skip1_disagreement": skip1_disagreement,
        "skip2_disagreement": skip2_disagreement,
    }


def dense_evidence_maps(hazy: torch.Tensor, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    final_feature = features["final_feature"]
    size = final_feature.shape[-2:]
    hazy_small = same_size(hazy, size)
    brightness = hazy_small.mean(dim=1, keepdim=True)
    dark = hazy_small.min(dim=1, keepdim=True).values
    saturation = hazy_small.max(dim=1, keepdim=True).values - dark
    gradient = gradient_magnitude(brightness)
    local_brightness = local_mean(brightness)
    local_variance = local_mean((brightness - local_brightness) ** 2)
    low_haze_proxy = local_mean(dark, kernel_size=15)

    final_norm = channel_norm(final_feature)
    final_mean = final_feature.mean(dim=1, keepdim=True)
    final_low = local_mean(final_mean)
    final_local_variance = local_mean((final_mean - final_low) ** 2)
    final_detail_energy = channel_norm(final_feature - local_mean(final_feature))
    return {
        "hazy_brightness": brightness,
        "hazy_dark": dark,
        "hazy_saturation": saturation,
        "hazy_gradient": gradient,
        "hazy_local_variance": local_variance,
        "hazy_low_haze_proxy": low_haze_proxy,
        "final_norm": final_norm,
        "final_local_variance": final_local_variance,
        "final_detail_energy": final_detail_energy,
        "res1_norm": channel_norm(same_size(features["res1"], size)),
        "res2_norm": channel_norm(same_size(features["res2"], size)),
        "scm2_norm": channel_norm(same_size(features["scm2"], size)),
        "scm4_norm": channel_norm(same_size(features["scm4"], size)),
    }


def base_metadata(row: dict[str, Any]) -> dict[str, Any]:
    dpsnr = float(row["WD0375_dPSNR"])
    out = dict(row)
    out["v215_primary_severe_risk_label"] = int(dpsnr <= -0.20)
    for threshold in RISK_THRESHOLDS:
        key = str(threshold).replace("-", "m").replace(".", "p")
        out[f"v215_severe_risk_label_{key}"] = int(dpsnr <= threshold)
    return out


def validate_numeric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nan_columns: dict[str, int] = {}
    inf_columns: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if key in ("name", "source_split"):
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(v):
                nan_columns[key] = nan_columns.get(key, 0) + 1
            if math.isinf(v):
                inf_columns[key] = inf_columns.get(key, 0) + 1
    return {
        "nan_columns": nan_columns,
        "inf_columns": inf_columns,
        "nan_total": sum(nan_columns.values()),
        "inf_total": sum(inf_columns.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-table", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--print-freq", type=int, default=50)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.feature_table)
    if args.max_images:
        rows = rows[: args.max_images]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_nopost(device)
    load_report = partial_load_nopost(model, args.checkpoint, device)
    model.eval()
    input_dir, _ = train_dirs(args.data_dir)

    spatial_rows: list[dict[str, Any]] = []
    fam_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []
    jitter_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for idx, src in enumerate(rows, 1):
            name = src["name"]
            hazy = image_tensor(input_dir / name, device)
            x, _, _, _, _ = pad_to(hazy, 32)
            features = extract_audit_features(model, x)
            maps = dense_evidence_maps(x, features)

            spatial_row = base_metadata(src)
            for map_name in SPATIAL_MAP_NAMES:
                spatial_row.update(map_stats(f"sp_{map_name}", maps[map_name]))
            spatial_rows.append(spatial_row)

            fam_row = base_metadata(src)
            fam_row.update(vector_stats("sens_fam1_response", features["fam1_response"]))
            fam_row.update(vector_stats("sens_fam2_response", features["fam2_response"]))
            fam_rows.append(fam_row)

            skip_row = base_metadata(src)
            skip_row.update(vector_stats("sens_skip1_disagreement", features["skip1_disagreement"]))
            skip_row.update(vector_stats("sens_skip2_disagreement", features["skip2_disagreement"]))
            skip_rows.append(skip_row)

            jitter_row = base_metadata(src)
            flipped_features = extract_audit_features(model, torch.flip(x, dims=(-1,)))
            jitter = features["final_feature"] - torch.flip(flipped_features["final_feature"], dims=(-1,))
            jitter_row.update(vector_stats("sens_final_hflip_consistency", jitter))
            jitter_rows.append(jitter_row)

            if idx % args.print_freq == 0:
                print(f"V215_FEATURE_BUILD {idx}/{len(rows)}", flush=True)

    spatial_path = args.out_dir / "v215_s2_spatial_feature_rows.csv"
    fam_path = args.out_dir / "v215_s3_fam_response_features.csv"
    skip_path = args.out_dir / "v215_s3_skip_merge_disagreement.csv"
    jitter_path = args.out_dir / "v215_s3_feature_jitter_consistency.csv"
    write_csv(spatial_path, spatial_rows)
    write_csv(fam_path, fam_rows)
    write_csv(skip_path, skip_rows)
    write_csv(jitter_path, jitter_rows)

    spatial_columns = list(spatial_rows[0].keys()) if spatial_rows else []
    quality = {
        "rows": len(spatial_rows),
        "spatial_feature_columns": len([c for c in spatial_columns if c.startswith("sp_")]),
        "spatial_maps": list(SPATIAL_MAP_NAMES),
        "numeric_validation": validate_numeric(spatial_rows),
        "locked_test_touched": False,
        "training_launched": False,
    }
    manifest = {
        "route": "haze4k-v2-15-nopost-spatial-internal-risk-audit",
        "feature_table": str(args.feature_table),
        "feature_table_sha256": sha256(args.feature_table),
        "data_dir": str(args.data_dir),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "outputs": {
            "spatial": str(spatial_path),
            "fam_response": str(fam_path),
            "skip_disagreement": str(skip_path),
            "jitter_consistency": str(jitter_path),
        },
        "partial_load": load_report,
        "contracts": {
            "uses_rgb_output_diff": False,
            "uses_teacher_output_diff": False,
            "uses_a0_output_diff": False,
            "locked_test_touched": False,
            "training_launched": False,
        },
        "spatial_maps": list(SPATIAL_MAP_NAMES),
    }
    write_json(args.out_dir / "v215_s2_spatial_feature_manifest.json", manifest)
    write_json(args.out_dir / "v215_s3_internal_sensitivity_manifest.json", manifest)
    (args.out_dir / "v215_s2_spatial_feature_quality_report.md").write_text(
        "# v2.15 S2 Spatial Feature Quality Report\n\n"
        f"- rows: `{quality['rows']}`\n"
        f"- spatial feature columns: `{quality['spatial_feature_columns']}`\n"
        f"- NaN values: `{quality['numeric_validation']['nan_total']}`\n"
        f"- Inf values: `{quality['numeric_validation']['inf_total']}`\n"
        "- locked test touched: `false`\n"
        "- training launched: `false`\n",
        encoding="utf-8",
    )
    print("V215_SPATIAL_FEATURE_BUILD_OK", json.dumps(quality, sort_keys=True))


if __name__ == "__main__":
    main()
