#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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


def read_names(path: Path, fold: int, max_images: int = 0) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        names = [r["name"] for r in csv.DictReader(f) if int(r["oof_fold"]) == fold]
    return names[:max_images] if max_images else names


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    return TVF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def pad_to(x: torch.Tensor, factor: int = 32) -> tuple[torch.Tensor, int, int, int, int]:
    _, _, h, w = x.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(x, (0, pw, 0, ph), "reflect"), h, w, h + ph, w + pw


def load_state(path: Path, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def final_output(out: Any, h: int, w: int) -> torch.Tensor:
    pred = out[2] if isinstance(out, (list, tuple)) else out
    return torch.clamp(pred[:, :, :h, :w], 0, 1)


def metric(pred: torch.Tensor, label: torch.Tensor, hp: int, wp: int) -> tuple[float, float]:
    mse = F.mse_loss(pred, label).clamp_min(1e-12)
    psnr = float((10 * torch.log10(1 / mse)).detach().cpu())
    down = max(1, round(min(hp, wp) / 256))
    ss = ssim(
        F.adaptive_avg_pool2d(pred, (int(hp / down), int(wp / down))),
        F.adaptive_avg_pool2d(label, (int(hp / down), int(wp / down))),
        data_range=1,
        size_average=False,
    ).mean().item()
    return psnr, float(ss)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(r["dPSNR"]) for r in rows]
    ssim_deltas = [float(r["dSSIM"]) for r in rows]
    ordered = sorted(rows, key=lambda r: float(r["A0_PSNR"]))
    k = max(1, len(rows) // 4)
    hard = [float(r["dPSNR"]) for r in ordered[:k]]
    easy = [float(r["dPSNR"]) for r in ordered[-k:]]
    strong_cut = percentile([float(r["A0_PSNR"]) for r in rows], 75)
    strong = [r for r in rows if float(r["A0_PSNR"]) >= strong_cut]
    return {
        "count": len(rows),
        "mean_dPSNR": statistics.mean(deltas),
        "median_dPSNR": statistics.median(deltas),
        "p05_dPSNR": percentile(deltas, 5),
        "p95_dPSNR": percentile(deltas, 95),
        "hard_bottom25_dPSNR": statistics.mean(hard),
        "easy_top25_dPSNR": statistics.mean(easy),
        "mean_dSSIM": statistics.mean(ssim_deltas),
        "positive_ratio": sum(v > 0 for v in deltas) / len(deltas),
        "severe_loss_count": sum(v <= -0.20 for v in deltas),
        "strong_reference_cut_psnr": strong_cut,
        "strong_regression_count_delta_le_-0.05": sum(float(r["dPSNR"]) <= -0.05 for r in strong),
    }


def gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "mean_dPSNR_ge_0p05": summary["mean_dPSNR"] >= 0.05,
        "hard_bottom25_dPSNR_ge_0p10": summary["hard_bottom25_dPSNR"] >= 0.10,
        "easy_top25_dPSNR_ge_neg0p05": summary["easy_top25_dPSNR"] >= -0.05,
        "positive_ratio_ge_0p55": summary["positive_ratio"] >= 0.55,
        "severe_loss_count_le_12": summary["severe_loss_count"] <= 12,
        "strong_regression_count_le_48": summary["strong_regression_count_delta_le_-0.05"] <= 48,
    }
    return {"checks": checks, "pass": all(checks.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--official-checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--checkpoint", action="append", required=True, help="label=path")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    from models.ConvIR import build_net as build_official
    from models.NoPostWLDBConvIR import build_net as build_wldb

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = read_names(args.split_csv, args.fold, args.max_images)
    input_dir, gt_dir = train_dirs(args.data_dir)

    official = build_official("base", "Haze4K", "original").to(device)
    official.load_state_dict(load_state(args.official_checkpoint, device))
    official.eval()

    base_rows = []
    with torch.no_grad():
        for name in names:
            hazy = image_tensor(input_dir / name, device)
            gt = image_tensor(label_path(gt_dir, name), device)
            x, h, w, hp, wp = pad_to(hazy, 32)
            pred = final_output(official(x), h, w)
            psnr, ss = metric(pred, gt, hp, wp)
            base_rows.append({"name": name, "A0_PSNR": psnr, "A0_SSIM": ss})
    base = {r["name"]: r for r in base_rows}

    summaries = {}
    for spec in args.checkpoint:
        label, raw_path = spec.split("=", 1)
        ckpt = Path(raw_path)
        model = build_wldb("base", "Haze4K", "original").to(device)
        model.load_state_dict(load_state(ckpt, device), strict=True)
        model.eval()
        rows = []
        with torch.no_grad():
            for idx, name in enumerate(names, 1):
                hazy = image_tensor(input_dir / name, device)
                gt = image_tensor(label_path(gt_dir, name), device)
                x, h, w, hp, wp = pad_to(hazy, 32)
                pred = final_output(model(x), h, w)
                psnr, ss = metric(pred, gt, hp, wp)
                row = {
                    "name": name,
                    "A0_PSNR": base[name]["A0_PSNR"],
                    "A0_SSIM": base[name]["A0_SSIM"],
                    "candidate": label,
                    "candidate_PSNR": psnr,
                    "candidate_SSIM": ss,
                    "dPSNR": psnr - base[name]["A0_PSNR"],
                    "dSSIM": ss - base[name]["A0_SSIM"],
                }
                rows.append(row)
                if idx % 100 == 0:
                    print(f"WLDB_A_EVAL {label} {idx}/{len(names)}", flush=True)
        summary = summarize(rows)
        gate_result = gate(summary)
        summaries[label] = {"checkpoint": str(ckpt), "summary": summary, "gate": gate_result}
        write_csv(args.out_dir / f"v216_wldb_a_eval_{label}_per_image.csv", rows)
        write_json(args.out_dir / f"v216_wldb_a_eval_{label}.json", summaries[label])
        print("WLDB_A_EVAL_SUMMARY", label, json.dumps(summaries[label], sort_keys=True), flush=True)

    passed = [label for label, payload in summaries.items() if payload["gate"]["pass"]]
    ranked = sorted(summaries, key=lambda label: summaries[label]["summary"]["mean_dPSNR"], reverse=True)
    decision = "WLDB_A_SCREEN_PASS_ALLOW_MULTI_SEED" if passed else "WLDB_A_SCREEN_FAIL_STOP_NO_MORE_TRAINING"
    closeout = {
        "route": "haze4k-v2-16-wldb-a-train",
        "fold": args.fold,
        "decision": decision,
        "passed": passed,
        "ranked_by_mean_dPSNR": ranked,
        "summaries": summaries,
        "locked_test_touched": False,
    }
    write_json(args.out_dir / "v216_wldb_a_eval_closeout.json", closeout)
    print("WLDB_A_EVAL_OK", json.dumps(closeout, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
