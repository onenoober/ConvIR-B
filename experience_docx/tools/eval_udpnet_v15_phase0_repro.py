#!/usr/bin/env python3
"""Controlled Phase-0 eval for official UDPNet ConvIR on Haze4K splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import types
from typing import Any

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def load_convir_builders(convir_its_dir: Path):
    sys.path.insert(0, str(convir_its_dir))
    from data import test_dataloader  # type: ignore
    from models.ConvIR import build_net as build_convir_net  # type: ignore

    return test_dataloader, build_convir_net


def load_udpnet_builder(udp_repo: Path):
    models_dir = udp_repo / "Dehazing/ITS/models"
    model_file = models_dir / "ConvIR_UDPNet.py"
    if not model_file.exists():
        raise FileNotFoundError(f"Missing official ConvIR_UDPNet.py: {model_file}")

    # Use a private package name so UDPNet's relative imports do not collide with
    # the local ConvIR-B `models` package.
    package_name = "udpnet_official_models"
    package = types.ModuleType(package_name)
    package.__path__ = [str(models_dir)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.ConvIR_UDPNet", model_file
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_net


def load_a0_model(build_convir_net, checkpoint: Path, device: torch.device):
    model = build_convir_net("base", "Haze4K", "original").to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def load_udpnet_model(build_udpnet, checkpoint: Path, device: torch.device):
    model = build_udpnet().to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise KeyError(f"Official checkpoint lacks state_dict: {checkpoint}")
    state = {}
    for key, value in ckpt["state_dict"].items():
        new_key = key[len("model.") :] if key.startswith("model.") else key
        state[new_key] = value
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"UDPNet checkpoint load mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    model.eval()
    return model, {
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "pytorch_lightning_version": ckpt.get("pytorch-lightning_version"),
        "state_dict_key_count": len(ckpt["state_dict"]),
    }


def infer_one(model, inp: torch.Tensor, h: int, w: int) -> torch.Tensor:
    out = model(inp)
    if isinstance(out, (list, tuple)):
        pred = out[0][2] if isinstance(out[0], (list, tuple)) else out[2]
    else:
        raise TypeError(f"Unexpected model output type: {type(out)!r}")
    return torch.clamp(pred[:, :, :h, :w], 0, 1)


def metric_pair(pred: torch.Tensor, label: torch.Tensor, padded_hw: tuple[int, int]) -> tuple[float, float]:
    mse = F.mse_loss(pred, label)
    psnr_val = (10 * torch.log10(1 / mse)).item()
    h_pad, w_pad = padded_hw
    down_ratio = max(1, round(min(h_pad, w_pad) / 256))
    ssim_val = ssim(
        F.adaptive_avg_pool2d(pred, (int(h_pad / down_ratio), int(w_pad / down_ratio))),
        F.adaptive_avg_pool2d(label, (int(h_pad / down_ratio), int(w_pad / down_ratio))),
        data_range=1,
        size_average=False,
    ).mean().item()
    return psnr_val, ssim_val


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["delta_psnr"]) for row in rows]
    ssim_deltas = [float(row["delta_ssim"]) for row in rows]
    a0_psnr = [float(row["a0_psnr"]) for row in rows]
    sorted_names = sorted(rows, key=lambda row: float(row["a0_psnr"]))
    bucket_count = max(1, len(rows) // 4)
    hard = sorted_names[:bucket_count]
    easy = sorted_names[-bucket_count:]
    strong_cut = percentile(a0_psnr, 75)
    strong = [row for row in rows if strong_cut is not None and float(row["a0_psnr"]) >= strong_cut]
    strong_regressions = [row for row in strong if float(row["delta_psnr"]) <= -0.05]
    worst_regressions = [row for row in rows if float(row["delta_psnr"]) <= -0.20]
    tail_count = max(1, len(deltas) // 10)
    sorted_deltas = sorted(deltas)
    return {
        "count": len(rows),
        "a0_mean_psnr": mean_or_none([float(row["a0_psnr"]) for row in rows]),
        "udpnet_mean_psnr": mean_or_none([float(row["udpnet_psnr"]) for row in rows]),
        "mean_psnr_delta": mean_or_none(deltas),
        "median_psnr_delta": statistics.median(deltas) if deltas else None,
        "p5_psnr_delta": percentile(deltas, 5),
        "p95_psnr_delta": percentile(deltas, 95),
        "hard_bottom25_psnr_delta": mean_or_none([float(row["delta_psnr"]) for row in hard]),
        "easy_top25_psnr_delta": mean_or_none([float(row["delta_psnr"]) for row in easy]),
        "worst10pct_mean_psnr_delta": mean_or_none(sorted_deltas[:tail_count]),
        "best10pct_mean_psnr_delta": mean_or_none(sorted_deltas[-tail_count:]),
        "mean_ssim_delta": mean_or_none(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / len(deltas) if deltas else None,
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_count": len(strong),
        "strong_regression_count_delta_le_-0.05": len(strong_regressions),
        "strong_regression_ratio": len(strong_regressions) / len(strong) if strong else None,
        "worst_regression_count_delta_le_-0.20": len(worst_regressions),
        "worst_regression_ratio": len(worst_regressions) / len(rows) if rows else None,
        "a0_avg_time_sec_sync": mean_or_none([float(row["a0_time_sec"]) for row in rows]),
        "udpnet_avg_time_sec_sync": mean_or_none([float(row["udpnet_time_sec"]) for row in rows]),
        "a0_peak_cuda_mem_mib": max((float(row["a0_peak_cuda_mem_mib"]) for row in rows if row["a0_peak_cuda_mem_mib"] != ""), default=None),
        "udpnet_peak_cuda_mem_mib": max((float(row["udpnet_peak_cuda_mem_mib"]) for row in rows if row["udpnet_peak_cuda_mem_mib"] != ""), default=None),
    }


def mark_buckets(rows: list[dict[str, Any]]) -> None:
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(str(row["split"]), []).append(row)
    for split_rows in by_split.values():
        ordered = sorted(split_rows, key=lambda row: float(row["a0_psnr"]))
        bucket_count = max(1, len(ordered) // 4)
        hard_names = {row["name"] for row in ordered[:bucket_count]}
        easy_names = {row["name"] for row in ordered[-bucket_count:]}
        for row in split_rows:
            if row["name"] in hard_names:
                row["bucket"] = "hard_bottom25_by_a0"
            elif row["name"] in easy_names:
                row["bucket"] = "easy_top25_by_a0"
            else:
                row["bucket"] = "mid_by_a0"


def gate_from_summaries(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    thresholds = {
        "mean_psnr_delta_min": 0.30,
        "hard_bottom25_psnr_delta_min": 0.20,
        "easy_top25_psnr_delta_min": -0.03,
        "mean_ssim_delta_min": 0.0,
        "strong_regression_ratio_max": 0.16,
        "worst_regression_count_max_per_300": 12,
    }

    split_checks: dict[str, dict[str, Any]] = {}
    for split, summary in summaries.items():
        count = int(summary["count"])
        worst_max = thresholds["worst_regression_count_max_per_300"] * count / 300.0
        checks = {
            "mean_psnr_delta": summary["mean_psnr_delta"],
            "mean_psnr_delta_pass": summary["mean_psnr_delta"] is not None
            and summary["mean_psnr_delta"] >= thresholds["mean_psnr_delta_min"],
            "hard_bottom25_psnr_delta": summary["hard_bottom25_psnr_delta"],
            "hard_bottom25_pass": summary["hard_bottom25_psnr_delta"] is not None
            and summary["hard_bottom25_psnr_delta"]
            >= thresholds["hard_bottom25_psnr_delta_min"],
            "easy_top25_psnr_delta": summary["easy_top25_psnr_delta"],
            "easy_top25_pass": summary["easy_top25_psnr_delta"] is not None
            and summary["easy_top25_psnr_delta"] >= thresholds["easy_top25_psnr_delta_min"],
            "mean_ssim_delta": summary["mean_ssim_delta"],
            "mean_ssim_pass": summary["mean_ssim_delta"] is not None
            and summary["mean_ssim_delta"] >= thresholds["mean_ssim_delta_min"],
            "strong_regression_ratio": summary["strong_regression_ratio"],
            "strong_regression_pass": summary["strong_regression_ratio"] is not None
            and summary["strong_regression_ratio"]
            <= thresholds["strong_regression_ratio_max"],
            "worst_regression_count": summary["worst_regression_count_delta_le_-0.20"],
            "worst_regression_pass": summary["worst_regression_count_delta_le_-0.20"]
            <= worst_max,
        }
        checks["split_pass"] = all(value for key, value in checks.items() if key.endswith("_pass"))
        split_checks[split] = checks

    required_splits = ["val_regular", "val_hard"]
    reproduction_pass = all(
        split_checks.get(split, {}).get("split_pass") for split in required_splits
    )
    return {
        "thresholds": thresholds,
        "split_checks": split_checks,
        "reproduction_gate_pass": bool(reproduction_pass),
        "decision": (
            "PHASE0_PASS_AUTHORIZE_PHASE1_TRANSPLANT_DESIGN"
            if reproduction_pass
            else "PHASE0_FAIL_DO_NOT_START_TRANSPLANT"
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    convir_its_dir = Path(args.convir_its_dir)
    udp_repo = Path(args.udp_repo)
    data_dir = Path(args.data_dir)
    depth_cache_dir = Path(args.depth_cache_dir)
    a0_checkpoint = Path(args.a0_checkpoint)
    official_checkpoint = Path(args.official_checkpoint)

    test_dataloader, build_convir_net = load_convir_builders(convir_its_dir)
    build_udpnet = load_udpnet_builder(udp_repo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    a0_model = load_a0_model(build_convir_net, a0_checkpoint, device)
    udpnet_model, ckpt_meta = load_udpnet_model(build_udpnet, official_checkpoint, device)

    all_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    factor = int(args.pad_factor)

    for split in args.splits:
        depth_split = "train" if args.split_json else args.depth_split
        dataloader = test_dataloader(
            str(data_dir),
            "Haze4K",
            batch_size=1,
            num_workers=args.num_workers,
            depth_cache_dir=str(depth_cache_dir),
            depth_split=depth_split,
            split_json=args.split_json,
            split_name=split,
        )
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                input_img, label_img, depth, name = data
                image_name = name[0] if isinstance(name, (list, tuple)) else str(name)
                input_img = input_img.to(device)
                label_img = label_img.to(device)
                depth = depth.to(device)
                h, w = input_img.shape[2], input_img.shape[3]
                h_pad = ((h + factor) // factor) * factor
                w_pad = ((w + factor) // factor) * factor
                padh = h_pad - h if h % factor != 0 else 0
                padw = w_pad - w if w % factor != 0 else 0
                rgb_padded = F.pad(input_img, (0, padw, 0, padh), "reflect")
                depth_padded = F.pad(depth, (0, padw, 0, padh), "reflect")
                udp_input = torch.cat([rgb_padded, depth_padded], dim=1)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.time()
                a0_pred = infer_one(a0_model, rgb_padded, h, w)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                a0_elapsed = time.time() - start
                a0_peak = (
                    torch.cuda.max_memory_allocated() / 1024**2
                    if torch.cuda.is_available()
                    else None
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.time()
                udp_pred = infer_one(udpnet_model, udp_input, h, w)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                udp_elapsed = time.time() - start
                udp_peak = (
                    torch.cuda.max_memory_allocated() / 1024**2
                    if torch.cuda.is_available()
                    else None
                )

                a0_psnr, a0_ssim = metric_pair(a0_pred, label_img, (h_pad, w_pad))
                udp_psnr, udp_ssim = metric_pair(udp_pred, label_img, (h_pad, w_pad))
                row = {
                    "name": image_name,
                    "split": split,
                    "a0_psnr": a0_psnr,
                    "udpnet_psnr": udp_psnr,
                    "delta_psnr": udp_psnr - a0_psnr,
                    "a0_ssim": a0_ssim,
                    "udpnet_ssim": udp_ssim,
                    "delta_ssim": udp_ssim - a0_ssim,
                    "a0_time_sec": a0_elapsed,
                    "udpnet_time_sec": udp_elapsed,
                    "a0_peak_cuda_mem_mib": "" if a0_peak is None else a0_peak,
                    "udpnet_peak_cuda_mem_mib": "" if udp_peak is None else udp_peak,
                    "bucket": "",
                }
                rows.append(row)
                all_rows.append(row)
                if (idx + 1) % args.print_freq == 0:
                    mean_delta = statistics.mean(float(x["delta_psnr"]) for x in rows)
                    print(
                        f"{split} {idx + 1}/{len(dataloader)} "
                        f"mean_delta={mean_delta:.4f}",
                        flush=True,
                    )
                if args.max_images and idx + 1 >= args.max_images:
                    break
        split_summaries[split] = summarize_split(rows)

    mark_buckets(all_rows)
    gate = gate_from_summaries(split_summaries)
    status = (
        "PHASE0_REPRODUCTION_GATE_PASS"
        if gate["reproduction_gate_pass"]
        else "PHASE0_REPRODUCTION_GATE_FAIL"
    )

    payload = {
        "route": "ConvIR-Dehaze-v1.5-FullUDP",
        "phase": "Phase 0 official UDPNet ConvIR reproduction eval",
        "status": status,
        "state_label": "COMPLETED_GATE_PASS" if gate["reproduction_gate_pass"] else "COMPLETED_GATE_FAIL",
        "locked_test_allowed": False,
        "locked_test_touched": False,
        "splits": args.splits,
        "split_json": args.split_json,
        "data_dir": str(data_dir),
        "depth_cache_dir": str(depth_cache_dir),
        "depth_split": "train" if args.split_json else args.depth_split,
        "pad_factor": factor,
        "metric": {
            "psnr": "10*log10(1/MSE) on RGB tensors in [0,1]",
            "ssim": "pytorch_msssim with adaptive average pooling matching existing ConvIR eval",
        },
        "a0_checkpoint_path": str(a0_checkpoint),
        "a0_checkpoint_sha256": sha256_file(a0_checkpoint),
        "official_checkpoint_path": str(official_checkpoint),
        "official_checkpoint_sha256": sha256_file(official_checkpoint),
        "official_checkpoint_meta": ckpt_meta,
        "udp_repo": str(udp_repo),
        "udp_repo_head": os.popen(f"git -C {udp_repo} rev-parse HEAD 2>/dev/null").read().strip() or None,
        "summaries": split_summaries,
        "gate": gate,
        "next_step": gate["decision"],
    }
    return payload, all_rows


def write_outputs(output_dir: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = output_dir / "udpnet_convir_repro_eval.json"
    eval_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = output_dir / "udpnet_convir_bucket_compare.csv"
    fieldnames = [
        "name",
        "split",
        "bucket",
        "a0_psnr",
        "udpnet_psnr",
        "delta_psnr",
        "a0_ssim",
        "udpnet_ssim",
        "delta_ssim",
        "a0_time_sec",
        "udpnet_time_sec",
        "a0_peak_cuda_mem_mib",
        "udpnet_peak_cuda_mem_mib",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    failure_path = output_dir / "udpnet_convir_failure_audit.csv"
    with failure_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "name",
                "bucket",
                "severity",
                "delta_psnr",
                "delta_ssim",
                "reason",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            delta = float(row["delta_psnr"])
            if delta <= -0.20:
                severity = "worst_regression_delta_le_-0.20"
            elif delta <= -0.05 and row["bucket"] == "easy_top25_by_a0":
                severity = "strong_reference_regression_delta_le_-0.05"
            else:
                continue
            writer.writerow(
                {
                    "split": row["split"],
                    "name": row["name"],
                    "bucket": row["bucket"],
                    "severity": severity,
                    "delta_psnr": row["delta_psnr"],
                    "delta_ssim": row["delta_ssim"],
                    "reason": "official UDPNet vs A0 per-image tail audit",
                }
            )

    protocol = f"""# UDPNet ConvIR Phase 0 Official Eval Protocol Diff

Status: `{payload["status"]}`

## Checkpoints

- A0 checkpoint: `{payload["a0_checkpoint_path"]}`
- A0 sha256: `{payload["a0_checkpoint_sha256"]}`
- Official UDPNet checkpoint: `{payload["official_checkpoint_path"]}`
- Official UDPNet sha256: `{payload["official_checkpoint_sha256"]}`
- Official UDPNet metadata: `{payload["official_checkpoint_meta"]}`

## Data And Splits

- Data root: `{payload["data_dir"]}`
- Depth cache: `{payload["depth_cache_dir"]}`
- Depth split: `{payload["depth_split"]}`
- Split JSON: `{payload["split_json"]}`
- Evaluated splits: `{payload["splits"]}`
- Locked Haze4K test touched: `{payload["locked_test_touched"]}`

This run uses the existing train-derived `val_regular` and `val_hard` split
contract before any locked-test decision.

## Entrypoint

- Local ConvIR-B A0 model is loaded from `Dehazing/ITS/models/ConvIR.py`.
- Official UDPNet model is loaded from `UDPNet/Dehazing/ITS/models/ConvIR_UDPNet.py`.
- Official `UDPNet/Dehazing/ITS/test.py` is not used because it imports FSNet by
  default and expects a `test/depth2l` directory. This wrapper evaluates
  `ConvIR_UDPNet` directly and maps the existing DepthAnything cache to the
  official 4-channel RGB+depth input.

## Metric

- PSNR: `{payload["metric"]["psnr"]}`
- SSIM: `{payload["metric"]["ssim"]}`
- Pad factor: `{payload["pad_factor"]}`
"""
    (output_dir / "udpnet_protocol_diff.md").write_text(protocol, encoding="utf-8")

    readme = f"""# Haze4K v1.5 Full UDPNet Phase 0 Official Eval

Status: `{payload["status"]}`.

Primary files:

- `udpnet_convir_repro_eval.json`: split summaries, checkpoint hashes, and gate.
- `udpnet_convir_bucket_compare.csv`: per-image A0 vs official UDPNet compare.
- `udpnet_convir_failure_audit.csv`: per-image strong/worst regression audit.
- `udpnet_protocol_diff.md`: checkpoint, data, depth, split, and metric contract.

Decision: `{payload["gate"]["decision"]}`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convir_its_dir", required=True)
    parser.add_argument("--udp_repo", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--official_checkpoint", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--splits", nargs="+", default=["val_regular", "val_hard"])
    parser.add_argument("--depth_split", default="test")
    parser.add_argument("--pad_factor", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    payload, rows = evaluate(args)
    write_outputs(Path(args.output_dir), payload, rows)
    print(
        "PHASE0_OFFICIAL_EVAL_OK "
        f"status={payload['status']} "
        f"gate={payload['gate']['decision']} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
