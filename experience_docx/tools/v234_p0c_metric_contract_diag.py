#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
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


def load_v234_tool(repo: Path):
    tool = repo / "experience_docx/tools/run_haze4k_v234_nopost_projection_audit.py"
    spec = importlib.util.spec_from_file_location("v234_tool", tool)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {tool}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def crop_with_coords(tool, input_img: torch.Tensor, label_img: torch.Tensor, size: int, seed: int):
    input_img = tool.ensure_crop_size(input_img, size)
    label_img = tool.ensure_crop_size(label_img, size)
    _, h, w = input_img.shape
    rng = random.Random(seed)
    top = rng.randint(0, max(0, h - size))
    left = rng.randint(0, max(0, w - size))
    return (
        input_img[:, top : top + size, left : left + size],
        label_img[:, top : top + size, left : left + size],
        top,
        left,
    )


def infer_official_padded(tool, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    padded, h, w = tool.pad_to_factor(x, 32)
    out = model(padded)[2]
    return torch.clamp(out[:, :, :h, :w], 0, 1)


def psnr(tool, pred: torch.Tensor, label: torch.Tensor) -> float:
    return float(tool.psnr_per_sample(torch.clamp(pred, 0, 1), label).item())


def blend(a0: torch.Tensor, expert: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.clamp(a0 + alpha * (expert - a0), 0, 1)


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    return {
        f"{key}_mean": mean(values),
        f"{key}_p05": percentile(values, 5),
        f"{key}_cvar5": cvar_low(values),
        f"{key}_negative_count": sum(1 for value in values if value < 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--data-dir", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl"))
    parser.add_argument("--wdmamba-repo", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba"))
    parser.add_argument("--wdmamba-checkpoint", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth"))
    parser.add_argument("--wdmamba-table", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/v22_c8_1_wdmamba_full_per_image.csv"))
    args = parser.parse_args()

    tool = load_v234_tool(args.repo)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir
    manifest_path = out_dir / "v234_p0b_balanced_canary_manifest.csv"
    benefit_path = out_dir / "v234_p0b_balanced_canary_teacher_direct_benefit.csv"
    table = {row["name"]: row for row in read_csv(args.wdmamba_table)}
    recorded = {row["sample_name"]: row for row in read_csv(benefit_path)}
    manifest = read_csv(manifest_path)[: args.limit]

    official = tool.build_official(args.checkpoint, device)
    wdmamba = tool.load_wdmamba(args.wdmamba_repo, args.wdmamba_checkpoint, device)
    dataset = tool.make_dataset(args.data_dir, "train")

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for rec in manifest:
            name = rec["sample_name"]
            index = int(rec["sample_index"])
            seed = int(rec["crop_seed"])
            inp, label = dataset[index]
            crop_inp, crop_label, top, left = crop_with_coords(tool, inp, label, args.crop_size, seed)

            x_full = inp.unsqueeze(0).to(device)
            y_full = label.unsqueeze(0).to(device)
            x_crop = crop_inp.unsqueeze(0).to(device)
            y_crop = crop_label.unsqueeze(0).to(device)

            a0_full = infer_official_padded(tool, official, x_full)
            wd_full = tool.infer_wdmamba(wdmamba, x_full)
            a0_crop_direct = torch.clamp(official(x_crop)[2], 0, 1)
            wd_crop_direct = tool.infer_wdmamba(wdmamba, x_crop)

            a0_full_slice = a0_full[:, :, top : top + args.crop_size, left : left + args.crop_size]
            wd_full_slice = wd_full[:, :, top : top + args.crop_size, left : left + args.crop_size]

            row: dict[str, Any] = {
                "sample_name": name,
                "sample_index": index,
                "crop_seed": seed,
                "crop_top": top,
                "crop_left": left,
                "selection_source_bucket": rec.get("selection_source_bucket", ""),
                "table_A0_PSNR": fnum(table.get(name, {}).get("A0_PSNR")),
                "table_a0p375_dPSNR": fnum(table.get(name, {}).get("expert_a0p375_dPSNR")),
                "table_a0p5_dPSNR": fnum(table.get(name, {}).get("expert_a0p5_dPSNR")),
                "v234_recorded_crop_direct_a0p5_dPSNR": fnum(recorded.get(name, {}).get("teacher_delta_vs_A0")),
                "full_recompute_A0_PSNR": psnr(tool, a0_full, y_full),
                "crop_direct_A0_PSNR": psnr(tool, a0_crop_direct, y_crop),
                "crop_fullslice_A0_PSNR": psnr(tool, a0_full_slice, y_crop),
                "crop_A0_context_gap_direct_minus_fullslice": psnr(tool, a0_crop_direct, y_crop) - psnr(tool, a0_full_slice, y_crop),
            }
            for alpha, label_key in [(0.375, "a0p375"), (0.5, "a0p5")]:
                full_blend = blend(a0_full, wd_full, alpha)
                crop_direct_blend = blend(a0_crop_direct, wd_crop_direct, alpha)
                crop_fullslice_blend = blend(a0_full_slice, wd_full_slice, alpha)

                full_delta = psnr(tool, full_blend, y_full) - psnr(tool, a0_full, y_full)
                crop_direct_delta = psnr(tool, crop_direct_blend, y_crop) - psnr(tool, a0_crop_direct, y_crop)
                crop_fullslice_delta = psnr(tool, crop_fullslice_blend, y_crop) - psnr(tool, a0_full_slice, y_crop)
                row[f"full_recompute_{label_key}_dPSNR"] = full_delta
                row[f"crop_direct_{label_key}_dPSNR"] = crop_direct_delta
                row[f"crop_fullslice_{label_key}_dPSNR"] = crop_fullslice_delta
                row[f"context_gap_direct_minus_fullslice_{label_key}"] = crop_direct_delta - crop_fullslice_delta
                row[f"location_gap_fullslice_minus_fullimage_{label_key}"] = crop_fullslice_delta - full_delta
            rows.append(row)
            print(f"diag_progress {len(rows)}/{len(manifest)} {name}", flush=True)

    write_csv(out_dir / "v234_p0c_metric_contract_diagnostic.csv", rows)
    summary: dict[str, Any] = {
        "phase": "P0C metric-contract diagnostic",
        "locked_test_touched": False,
        "sample_count": len(rows),
        "source_manifest": str(manifest_path),
        "wdmamba_table": str(args.wdmamba_table),
    }
    for key in [
        "table_a0p375_dPSNR",
        "table_a0p5_dPSNR",
        "full_recompute_a0p375_dPSNR",
        "full_recompute_a0p5_dPSNR",
        "crop_fullslice_a0p375_dPSNR",
        "crop_fullslice_a0p5_dPSNR",
        "crop_direct_a0p375_dPSNR",
        "crop_direct_a0p5_dPSNR",
        "context_gap_direct_minus_fullslice_a0p375",
        "context_gap_direct_minus_fullslice_a0p5",
        "location_gap_fullslice_minus_fullimage_a0p375",
        "location_gap_fullslice_minus_fullimage_a0p5",
        "crop_A0_context_gap_direct_minus_fullslice",
    ]:
        summary.update(summarize(rows, key))
    summary["table_positive_a0p5_but_crop_direct_negative_count"] = sum(
        1 for row in rows
        if (row.get("table_a0p5_dPSNR") is not None and float(row["table_a0p5_dPSNR"]) >= 0.5)
        and float(row["crop_direct_a0p5_dPSNR"]) < 0
    )
    diffs = [
        abs(float(row["crop_direct_a0p5_dPSNR"]) - float(row["v234_recorded_crop_direct_a0p5_dPSNR"]))
        for row in rows
        if row.get("v234_recorded_crop_direct_a0p5_dPSNR") is not None
    ]
    summary["v234_recorded_vs_recomputed_crop_direct_a0p5_mean_abs_diff"] = mean(diffs)
    (out_dir / "v234_p0c_metric_contract_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("V234_P0C_METRIC_CONTRACT_DIAG_OK")


if __name__ == "__main__":
    main()
