#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from c13_common import (
    first_dir,
    image_tensor,
    label_path,
    load_a0_model,
    load_c13_model,
    load_split_manifest,
    metric,
    pad_to,
    summarize_rows,
    write_csv,
    write_json,
)


QUICK_GATE = {
    "mean_dPSNR": 0.25,
    "hard_bottom25_dPSNR": 0.35,
    "easy_top25_dPSNR": 0.10,
    "positive_ratio": 0.80,
    "severe_loss_per_600": 60.0,
    "dSSIM": 0.0,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quick_gate_pass(summary: dict[str, Any]) -> bool:
    return (
        float(summary.get("mean_dPSNR", -999.0)) >= QUICK_GATE["mean_dPSNR"]
        and float(summary.get("hard_bottom25_dPSNR", -999.0)) >= QUICK_GATE["hard_bottom25_dPSNR"]
        and float(summary.get("easy_top25_dPSNR", -999.0)) >= QUICK_GATE["easy_top25_dPSNR"]
        and float(summary.get("positive_ratio", -999.0)) >= QUICK_GATE["positive_ratio"]
        and float(summary.get("severe_loss_per_600", 999.0)) <= QUICK_GATE["severe_loss_per_600"]
        and float(summary.get("dSSIM", -999.0)) >= QUICK_GATE["dSSIM"]
    )


def val_names(split_manifest: Path) -> list[dict[str, str]]:
    payload = load_split_manifest(split_manifest)
    return list(payload["val"])


def parse_haze_params(name: str) -> dict[str, Any]:
    stem = Path(name).stem
    parts = stem.split("_")
    out: dict[str, Any] = {"haze_param_t": "", "haze_param_beta": ""}
    if len(parts) >= 3:
        try:
            out["haze_param_t"] = float(parts[1])
            out["haze_param_beta"] = float(parts[2])
        except ValueError:
            pass
    return out


def dark_channel_mean(x: torch.Tensor) -> float:
    return float(x.detach().float().amin(dim=1).mean().cpu().item())


def low_texture_proxy(x: torch.Tensor) -> float:
    gray = x.detach().float().mean(dim=1, keepdim=True)
    dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]).mean()
    dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]).mean()
    return float((0.5 * (dx + dy)).cpu().item())


def sky_highlight_proxy(x: torch.Tensor) -> float:
    rgb = x.detach().float()
    mx = rgb.max(dim=1, keepdim=True).values
    mn = rgb.min(dim=1, keepdim=True).values
    sat = (mx - mn) / mx.clamp_min(1e-6)
    mask = (mx > 0.78) & (sat < 0.18)
    return float(mask.float().mean().cpu().item())


def color_shift(pred: torch.Tensor, a0: torch.Tensor) -> float:
    return float((pred.mean(dim=(-2, -1)) - a0.mean(dim=(-2, -1))).abs().mean().cpu().item())


def clip_ratio(unclamped: torch.Tensor) -> float:
    return float(((unclamped < 0.0) | (unclamped > 1.0)).float().mean().cpu().item())


def _pad_even(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    ph = h % 2
    pw = w % 2
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


def haar_dwt2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    x, h, w = _pad_even(x)
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh, h, w


def haar_idwt2(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor, h: int, w: int) -> torch.Tensor:
    x00 = (ll + lh + hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll - lh - hl + hh) * 0.5
    out = torch.empty(ll.shape[:-2] + (ll.shape[-2] * 2, ll.shape[-1] * 2), device=ll.device, dtype=ll.dtype)
    out[..., 0::2, 0::2] = x00
    out[..., 0::2, 1::2] = x01
    out[..., 1::2, 0::2] = x10
    out[..., 1::2, 1::2] = x11
    return out[..., :h, :w]


def psnr_tensor(pred: torch.Tensor, label: torch.Tensor) -> float:
    mse = F.mse_loss(pred, label).clamp_min(1e-12)
    return float((10 * torch.log10(1.0 / mse)).item())


def eval_candidate(
    tag: str,
    args: argparse.Namespace,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0 = load_a0_model(args.convir_dir, args.a0_checkpoint, device)
    student = load_c13_model(
        args.convir_dir,
        args.a0_checkpoint,
        args.student_checkpoint,
        device,
        args.feature_mode,
        args.adapter_width,
        args.adapter_depth,
        args.bootstrap_scale,
        residual_mode=args.residual_mode,
        residual_scale=args.residual_scale,
        scale_init=args.scale_init,
        head_init=args.head_init,
        clamp_output=args.clamp_output,
    )
    input_dir = first_dir(args.data_dir / "train", ("IN", "haze", "hazy"))
    gt_dir = first_dir(args.data_dir / "train", ("GT", "gt"))
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for rec in val_names(args.split_manifest):
            name = rec["name"]
            hazy = image_tensor(input_dir / name).to(device)
            label = image_tensor(label_path(gt_dir, name)).to(device)
            x, h, w, hp, wp = pad_to(hazy, 32)
            a0_pred_full = a0(x)[2]
            aux = student.route_forward(x)
            pred_full = aux["outputs"][-1]
            raw_full = aux["raw_residual"]
            residual_full = aux["residual"]
            a0_pred = a0_pred_full[:, :, :h, :w]
            pred = pred_full[:, :, :h, :w]
            raw = raw_full[:, :, :h, :w]
            residual = residual_full[:, :, :h, :w]
            a0_psnr, a0_ssim = metric(a0_pred, label, hp, wp)
            pred_psnr, pred_ssim = metric(pred, label, hp, wp)
            ll, _lh, _hl, _hh, _oh, _ow = haar_dwt2(residual)
            row = {
                "name": name,
                "split": rec.get("split", "val"),
                "A0_PSNR": a0_psnr,
                "A0_SSIM": a0_ssim,
                "student_PSNR": pred_psnr,
                "student_SSIM": pred_ssim,
                "dPSNR": pred_psnr - a0_psnr,
                "dSSIM": pred_ssim - a0_ssim,
                "residual_mean_abs": float(residual.abs().mean().item()),
                "raw_residual_mean_abs": float(raw.abs().mean().item()),
                "ll_residual_mean_abs": float(ll.abs().mean().item()),
                "color_shift": color_shift(pred, a0_pred),
                "clip_ratio": clip_ratio(a0_pred + residual),
                "input_dark_channel_mean": dark_channel_mean(hazy),
                "input_low_texture_proxy": low_texture_proxy(hazy),
                "sky_highlight_proxy": sky_highlight_proxy(hazy),
            }
            row.update(parse_haze_params(name))
            rows.append(row)
    summary = summarize_rows(rows)
    summary.update(
        {
            "variant": tag,
            "checkpoint": args.checkpoint,
            "locked_test_touched": False,
            "locked_per_image_read": False,
            "val_count": len(rows),
            "quick_gate_pass": quick_gate_pass(summary),
            "student_checkpoint": str(args.student_checkpoint),
            "residual_mode": args.residual_mode,
            "residual_scale": args.residual_scale,
            "scale_init": args.scale_init,
            "head_init": args.head_init,
            "clamp_output": args.clamp_output,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"v25_c13_f0_{tag}_full600_per_image.csv", rows)
    write_json(args.out_dir / f"v25_c13_f0_{tag}_full600_summary.json", summary)
    print("C13_F0_EVAL_OK", json.dumps(summary, sort_keys=True))


def read_c8_teacher(args: argparse.Namespace) -> None:
    val = {rec["name"]: rec for rec in val_names(args.split_manifest)}
    rows: list[dict[str, Any]] = []
    for rec in read_csv(args.c8_per_image):
        name = rec.get("name", "")
        if name not in val:
            continue
        a0_psnr = float(rec["A0_PSNR"])
        a0_ssim = float(rec["A0_SSIM"])
        dpsnr = float(rec.get("expert_a0p375_dPSNR", rec.get("WD0375_dPSNR", "nan")))
        dssim = float(rec.get("expert_a0p375_dSSIM", rec.get("WD0375_dSSIM", "0")))
        teacher_psnr = float(rec.get("expert_a0p375_PSNR", a0_psnr + dpsnr))
        teacher_ssim = float(rec.get("expert_a0p375_SSIM", a0_ssim + dssim))
        row = {
            "name": name,
            "split": val[name].get("split", rec.get("split", "val")),
            "A0_PSNR": a0_psnr,
            "A0_SSIM": a0_ssim,
            "student_PSNR": teacher_psnr,
            "student_SSIM": teacher_ssim,
            "dPSNR": dpsnr,
            "dSSIM": dssim,
            "teacher_margin": dpsnr,
        }
        row.update(parse_haze_params(name))
        rows.append(row)
    summary = summarize_rows(rows)
    summary.update(
        {
            "variant": "wd0375_teacher",
            "checkpoint": "c8_per_image_expert_a0p375",
            "locked_test_touched": False,
            "locked_per_image_read": False,
            "val_count": len(rows),
            "quick_gate_pass": quick_gate_pass(summary),
            "source_c8_per_image": str(args.c8_per_image),
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "v25_c13_f0_wd0375_teacher_full600_per_image.csv", rows)
    write_json(args.out_dir / "v25_c13_f0_wd0375_teacher_full600_summary.json", summary)
    print("C13_F0_TEACHER_OK", json.dumps(summary, sort_keys=True))


def best_scaled_psnr(a0: torch.Tensor, residual: torch.Tensor, label: torch.Tensor, scales: list[float]) -> tuple[float, float, torch.Tensor]:
    best_scale = 0.0
    best_psnr = -1e9
    best_pred = a0
    for scale in scales:
        pred = torch.clamp(a0 + float(scale) * residual, 0.0, 1.0)
        score = psnr_tensor(pred, label)
        if score > best_psnr:
            best_psnr = score
            best_scale = float(scale)
            best_pred = pred
    return best_scale, best_psnr, best_pred


def patch_oracle(a0: torch.Tensor, residual: torch.Tensor, label: torch.Tensor, scales: list[float], patch: int) -> tuple[float, torch.Tensor]:
    out = a0.clone()
    _, _, h, w = a0.shape
    for y in range(0, h, patch):
        y2 = min(h, y + patch)
        for x in range(0, w, patch):
            x2 = min(w, x + patch)
            patch_a0 = a0[:, :, y:y2, x:x2]
            patch_res = residual[:, :, y:y2, x:x2]
            patch_label = label[:, :, y:y2, x:x2]
            best_scale, _score, _pred = best_scaled_psnr(patch_a0, patch_res, patch_label, scales)
            out[:, :, y:y2, x:x2] = torch.clamp(patch_a0 + best_scale * patch_res, 0.0, 1.0)
    return psnr_tensor(out, label), out


def run_oracles(args: argparse.Namespace) -> None:
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0 = load_a0_model(args.convir_dir, args.a0_checkpoint, device)
    student = load_c13_model(
        args.convir_dir,
        args.a0_checkpoint,
        args.student_checkpoint,
        device,
        args.feature_mode,
        args.adapter_width,
        args.adapter_depth,
        0.0,
        residual_mode="direct",
        residual_scale=1.0,
        scale_init=0.25,
        head_init="zero",
        clamp_output=False,
    )
    teacher_map = {}
    if args.c8_per_image and args.c8_per_image.is_file():
        for row in read_csv(args.c8_per_image):
            teacher_map[row["name"]] = row
    input_dir = first_dir(args.data_dir / "train", ("IN", "haze", "hazy"))
    gt_dir = first_dir(args.data_dir / "train", ("GT", "gt"))
    rows: list[dict[str, Any]] = []
    taxonomy: list[dict[str, Any]] = []
    with torch.no_grad():
        for rec in val_names(args.split_manifest):
            name = rec["name"]
            hazy = image_tensor(input_dir / name).to(device)
            label = image_tensor(label_path(gt_dir, name)).to(device)
            x, h, w, hp, wp = pad_to(hazy, 32)
            a0_pred = a0(x)[2][:, :, :h, :w]
            aux = student.route_forward(x)
            raw = aux["raw_residual"][:, :, :h, :w]
            a0_psnr, a0_ssim = metric(a0_pred, label, hp, wp)
            actual_unclamped = a0_pred + args.actual_scale * raw
            actual = torch.clamp(actual_unclamped, 0.0, 1.0)
            actual_psnr, actual_ssim = metric(actual, label, hp, wp)
            image_scale, image_psnr, image_pred = best_scaled_psnr(a0_pred, raw, label, scales)
            image_ssim = metric(image_pred, label, hp, wp)[1]
            patch_psnr, patch_pred = patch_oracle(a0_pred, raw, label, scales, args.patch_size)
            patch_ssim = metric(patch_pred, label, hp, wp)[1]

            ll, lh, hl, hh, oh, ow = haar_dwt2(raw)
            zeros = torch.zeros_like(ll)
            bands = {
                "ll": haar_idwt2(ll, zeros, zeros, zeros, oh, ow),
                "lh": haar_idwt2(zeros, lh, zeros, zeros, oh, ow),
                "hl": haar_idwt2(zeros, zeros, hl, zeros, oh, ow),
                "hh": haar_idwt2(zeros, zeros, zeros, hh, oh, ow),
                "high": haar_idwt2(zeros, lh, hl, hh, oh, ow),
            }
            band_results: dict[str, Any] = {}
            band_independent_residual = torch.zeros_like(raw)
            for band_name, band_residual in bands.items():
                band_scale, band_psnr, _band_pred = best_scaled_psnr(a0_pred, band_residual, label, scales)
                band_results[f"{band_name}_best_scale"] = band_scale
                band_results[f"{band_name}_dPSNR"] = band_psnr - a0_psnr
                if band_name in {"ll", "lh", "hl", "hh"}:
                    band_independent_residual = band_independent_residual + band_scale * band_residual
            band_independent_pred = torch.clamp(a0_pred + band_independent_residual, 0.0, 1.0)
            band_independent_psnr, band_independent_ssim = metric(band_independent_pred, label, hp, wp)
            teacher = teacher_map.get(name, {})
            teacher_margin = teacher.get("expert_a0p375_dPSNR", teacher.get("WD0375_dPSNR", ""))
            row = {
                "name": name,
                "split": rec.get("split", "val"),
                "A0_PSNR": a0_psnr,
                "A0_SSIM": a0_ssim,
                "actual_scale": args.actual_scale,
                "actual_PSNR": actual_psnr,
                "actual_SSIM": actual_ssim,
                "actual_dPSNR": actual_psnr - a0_psnr,
                "actual_dSSIM": actual_ssim - a0_ssim,
                "image_oracle_best_scale": image_scale,
                "image_oracle_PSNR": image_psnr,
                "image_oracle_dPSNR": image_psnr - a0_psnr,
                "image_oracle_dSSIM": image_ssim - a0_ssim,
                "patch_oracle_PSNR": patch_psnr,
                "patch_oracle_dPSNR": patch_psnr - a0_psnr,
                "patch_oracle_dSSIM": patch_ssim - a0_ssim,
                "band_independent_PSNR": band_independent_psnr,
                "band_independent_dPSNR": band_independent_psnr - a0_psnr,
                "band_independent_dSSIM": band_independent_ssim - a0_ssim,
                "teacher_margin": float(teacher_margin) if teacher_margin != "" else "",
                "raw_residual_mean_abs": float(raw.abs().mean().item()),
                "ll_residual_mean_abs": float(ll.abs().mean().item()),
                "lh_residual_mean_abs": float(lh.abs().mean().item()),
                "hl_residual_mean_abs": float(hl.abs().mean().item()),
                "hh_residual_mean_abs": float(hh.abs().mean().item()),
                "actual_color_shift": color_shift(actual, a0_pred),
                "actual_clip_ratio": clip_ratio(actual_unclamped),
                "input_dark_channel_mean": dark_channel_mean(hazy),
                "input_low_texture_proxy": low_texture_proxy(hazy),
                "sky_highlight_proxy": sky_highlight_proxy(hazy),
            }
            row.update(band_results)
            row.update(parse_haze_params(name))
            rows.append(row)
            if row["actual_dPSNR"] <= -0.20:
                taxonomy.append(row)

    def projection_summary(key: str, dssim_key: str | None = None) -> dict[str, Any]:
        mapped = []
        for row in rows:
            mapped.append(
                {
                    "name": row["name"],
                    "split": row["split"],
                    "A0_PSNR": row["A0_PSNR"],
                    "dPSNR": row[key],
                    "dSSIM": row.get(dssim_key or "", 0.0),
                }
            )
        summary = summarize_rows(mapped)
        summary["quick_gate_pass"] = quick_gate_pass(summary)
        return summary

    summaries = {
        "actual_scale": projection_summary("actual_dPSNR", "actual_dSSIM"),
        "per_image_scale_oracle": projection_summary("image_oracle_dPSNR", "image_oracle_dSSIM"),
        "patch_scale_oracle": projection_summary("patch_oracle_dPSNR", "patch_oracle_dSSIM"),
        "band_independent_oracle": projection_summary("band_independent_dPSNR", "band_independent_dSSIM"),
        "ll_only_oracle": projection_summary("ll_dPSNR"),
        "lh_only_oracle": projection_summary("lh_dPSNR"),
        "hl_only_oracle": projection_summary("hl_dPSNR"),
        "hh_only_oracle": projection_summary("hh_dPSNR"),
        "high_only_oracle": projection_summary("high_dPSNR"),
    }
    failure_summary = {
        "severe_count": len(taxonomy),
        "severe_per_600": len(taxonomy) / max(1, len(rows)) * 600.0,
        "mean_actual_dPSNR": sum(float(r["actual_dPSNR"]) for r in taxonomy) / max(1, len(taxonomy)),
        "mean_teacher_margin": sum(float(r["teacher_margin"]) for r in taxonomy if r["teacher_margin"] != "") / max(1, sum(1 for r in taxonomy if r["teacher_margin"] != "")),
        "mean_raw_residual_mean_abs": sum(float(r["raw_residual_mean_abs"]) for r in taxonomy) / max(1, len(taxonomy)),
        "mean_ll_residual_mean_abs": sum(float(r["ll_residual_mean_abs"]) for r in taxonomy) / max(1, len(taxonomy)),
        "mean_color_shift": sum(float(r["actual_color_shift"]) for r in taxonomy) / max(1, len(taxonomy)),
        "mean_clip_ratio": sum(float(r["actual_clip_ratio"]) for r in taxonomy) / max(1, len(taxonomy)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "v25_c13_f1_f2_oracle_per_image.csv", rows)
    write_csv(args.out_dir / "v25_c13_f3_failure_taxonomy.csv", taxonomy)
    write_json(
        args.out_dir / "v25_c13_f1_f2_oracle_summary.json",
        {
            "route": "Haze4K v2.5 C13-F oracle diagnostics",
            "locked_test_touched": False,
            "locked_per_image_read": False,
            "scale_grid": scales,
            "patch_size": args.patch_size,
            "actual_scale": args.actual_scale,
            "summaries": summaries,
        },
    )
    write_json(args.out_dir / "v25_c13_f3_failure_taxonomy_summary.json", failure_summary)
    print("C13_F1_F2_F3_ORACLE_OK", json.dumps({"summaries": summaries, "failure_summary": failure_summary}, sort_keys=True))


def quantile_groups(rows: list[dict[str, Any]], key: str) -> list[tuple[str, list[dict[str, Any]]]]:
    usable = [row for row in rows if row.get(key, "") not in ("", None)]
    if not usable:
        return []
    ordered = sorted(usable, key=lambda row: float(row[key]))
    groups = []
    n = len(ordered)
    for idx in range(4):
        start = round(n * idx / 4)
        end = round(n * (idx + 1) / 4)
        part = ordered[start:end]
        if part:
            groups.append((f"{key}_q{idx + 1}", part))
    return groups


def aggregate(args: argparse.Namespace) -> None:
    teacher_rows = {}
    teacher_path = args.out_dir / "v25_c13_f0_wd0375_teacher_full600_per_image.csv"
    if teacher_path.is_file():
        teacher_rows = {row["name"]: row for row in read_csv(teacher_path)}
    leaderboard: list[dict[str, Any]] = []
    group_a0: list[dict[str, Any]] = []
    group_teacher: list[dict[str, Any]] = []
    group_residual: list[dict[str, Any]] = []
    failure_manifest: list[dict[str, Any]] = []
    a0_rows: list[dict[str, Any]] = []
    for summary_path in sorted(args.out_dir.glob("v25_c13_f0_*_full600_summary.json")):
        summary = read_json(summary_path)
        variant = str(summary["variant"])
        per_image = args.out_dir / f"v25_c13_f0_{variant}_full600_per_image.csv"
        if not per_image.is_file():
            continue
        rows = read_csv(per_image)
        if not a0_rows:
            a0_rows = rows
        row = {
            "variant": variant,
            "val_count": summary.get("val_count", summary.get("count", "")),
            "mean_dPSNR": summary.get("mean_dPSNR"),
            "hard_bottom25_dPSNR": summary.get("hard_bottom25_dPSNR"),
            "easy_top25_dPSNR": summary.get("easy_top25_dPSNR"),
            "positive_ratio": summary.get("positive_ratio"),
            "severe_loss_per_600": summary.get("severe_loss_per_600"),
            "dSSIM": summary.get("dSSIM"),
            "quick_gate_pass": quick_gate_pass(summary),
            "summary_json": summary_path.name,
            "per_image_csv": per_image.name,
        }
        leaderboard.append(row)
        merged: list[dict[str, Any]] = []
        for src in rows:
            rec: dict[str, Any] = dict(src)
            teacher = teacher_rows.get(src["name"], {})
            rec["teacher_margin"] = teacher.get("dPSNR", rec.get("teacher_margin", ""))
            rec["residual_energy"] = rec.get("raw_residual_mean_abs", rec.get("residual_mean_abs", ""))
            merged.append(rec)
        for label, part in quantile_groups(merged, "A0_PSNR"):
            s = summarize_rows(part)
            s.update({"variant": variant, "group": label})
            group_a0.append(s)
        for label, part in quantile_groups(merged, "teacher_margin"):
            s = summarize_rows(part)
            s.update({"variant": variant, "group": label})
            group_teacher.append(s)
        for label, part in quantile_groups(merged, "residual_energy"):
            s = summarize_rows(part)
            s.update({"variant": variant, "group": label})
            group_residual.append(s)
        worst = sorted(merged, key=lambda r: float(r["dPSNR"]))[:20]
        for idx, rec in enumerate(worst, 1):
            failure_manifest.append(
                {
                    "variant": variant,
                    "rank": idx,
                    "name": rec["name"],
                    "split": rec.get("split", ""),
                    "A0_PSNR": rec.get("A0_PSNR", ""),
                    "dPSNR": rec.get("dPSNR", ""),
                    "teacher_margin": rec.get("teacher_margin", ""),
                    "residual_energy": rec.get("residual_energy", ""),
                    "color_shift": rec.get("color_shift", ""),
                    "clip_ratio": rec.get("clip_ratio", ""),
                    "haze_param_t": rec.get("haze_param_t", ""),
                    "haze_param_beta": rec.get("haze_param_beta", ""),
                }
            )
    leaderboard.sort(
        key=lambda r: (
            bool(r["quick_gate_pass"]),
            float(r["mean_dPSNR"]),
            float(r["hard_bottom25_dPSNR"]),
            -float(r["severe_loss_per_600"]),
        ),
        reverse=True,
    )
    write_csv(args.out_dir / "v25_c13_f0_full600_leaderboard.csv", leaderboard)
    write_csv(args.out_dir / "v25_c13_f0_group_by_a0psnr.csv", group_a0)
    write_csv(args.out_dir / "v25_c13_f0_group_by_teacher_margin.csv", group_teacher)
    write_csv(args.out_dir / "v25_c13_f0_group_by_residual_energy.csv", group_residual)
    write_csv(args.out_dir / "v25_c13_f0_failure_gallery_manifest.csv", failure_manifest)
    oracle_summary = {}
    oracle_path = args.out_dir / "v25_c13_f1_f2_oracle_summary.json"
    if oracle_path.is_file():
        oracle_summary = read_json(oracle_path)
    a0_reference = {}
    if a0_rows:
        a0_psnr_vals = [float(r["A0_PSNR"]) for r in a0_rows]
        a0_ssim_vals = [float(r["A0_SSIM"]) for r in a0_rows]
        a0_reference = {
            "count": len(a0_rows),
            "mean_A0_PSNR": sum(a0_psnr_vals) / len(a0_psnr_vals),
            "mean_A0_SSIM": sum(a0_ssim_vals) / len(a0_ssim_vals),
        }
    teacher_reference = {}
    if teacher_rows:
        teacher_list = list(teacher_rows.values())
        teacher_dpsnr = [float(r.get("dPSNR", 0.0)) for r in teacher_list]
        teacher_dssim = [float(r.get("dSSIM", 0.0)) for r in teacher_list]
        teacher_reference = {
            "count": len(teacher_list),
            "mean_teacher_dPSNR": sum(teacher_dpsnr) / len(teacher_dpsnr),
            "mean_teacher_dSSIM": sum(teacher_dssim) / len(teacher_dssim),
        }
    decision = {
        "route": "Haze4K v2.5 C13-F diagnostics",
        "locked_test_touched": False,
        "locked_per_image_read": False,
        "f0_rows": leaderboard,
        "f0_quick_gate_pass_variants": [row["variant"] for row in leaderboard if row["quick_gate_pass"]],
        "oracle_summary": oracle_summary,
        "a0_reference": a0_reference,
        "teacher_reference": teacher_reference,
        "decision": "C13_F_DIAGNOSTIC_COMPLETE_GATE_VS_RESIDUAL_DIRECTION_REVIEW",
    }
    write_json(args.out_dir / "v25_c13_f_diagnostic_decision.json", decision)
    lines = [
        "# Haze4K v2.5 C13-F Diagnostic Decision",
        "",
        "Decision: `C13_F_DIAGNOSTIC_COMPLETE_GATE_VS_RESIDUAL_DIRECTION_REVIEW`",
        "",
        "Locked Haze4K test remained untouched.",
        "",
        "## F0 Full-600 Leaderboard",
        "",
        "| Variant | mean | hard | easy | positive | severe/600 | dSSIM | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in leaderboard:
        lines.append(
            "| {variant} | {mean_dPSNR:.6f} | {hard_bottom25_dPSNR:.6f} | {easy_top25_dPSNR:.6f} | "
            "{positive_ratio:.6f} | {severe_loss_per_600:.2f} | {dSSIM:.8f} | {quick_gate_pass} |".format(
                variant=row["variant"],
                mean_dPSNR=float(row["mean_dPSNR"]),
                hard_bottom25_dPSNR=float(row["hard_bottom25_dPSNR"]),
                easy_top25_dPSNR=float(row["easy_top25_dPSNR"]),
                positive_ratio=float(row["positive_ratio"]),
                severe_loss_per_600=float(row["severe_loss_per_600"]),
                dSSIM=float(row["dSSIM"]),
                quick_gate_pass=row["quick_gate_pass"],
            )
        )
    if oracle_summary:
        lines.extend(["", "## Oracle Summary", ""])
        for name, summary in oracle_summary.get("summaries", {}).items():
            lines.append(
                "- `{}`: mean `{:.6f}`, hard `{:.6f}`, easy `{:.6f}`, positive `{:.6f}`, severe/600 `{:.2f}`, pass `{}`".format(
                    name,
                    float(summary.get("mean_dPSNR", 0.0)),
                    float(summary.get("hard_bottom25_dPSNR", 0.0)),
                    float(summary.get("easy_top25_dPSNR", 0.0)),
                    float(summary.get("positive_ratio", 0.0)),
                    float(summary.get("severe_loss_per_600", 0.0)),
                    summary.get("quick_gate_pass", False),
                )
            )
    if a0_reference:
        lines.extend(
            [
                "",
                "## A0 Reference",
                "",
                f"- count: `{a0_reference['count']}`",
                f"- mean A0 PSNR: `{a0_reference['mean_A0_PSNR']:.6f}`",
                f"- mean A0 SSIM: `{a0_reference['mean_A0_SSIM']:.6f}`",
            ]
        )
    if teacher_reference:
        lines.extend(
            [
                "",
                "## Teacher Reference",
                "",
                f"- count: `{teacher_reference['count']}`",
                f"- mean teacher dPSNR: `{teacher_reference['mean_teacher_dPSNR']:.6f}`",
                f"- mean teacher dSSIM: `{teacher_reference['mean_teacher_dSSIM']:.6f}`",
            ]
        )
    (args.out_dir / "v25_c13_f_diagnostic_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("C13_F_AGGREGATE_OK", json.dumps({"leaderboard_count": len(leaderboard)}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    eval_ap = sub.add_parser("eval-c13")
    eval_ap.add_argument("--tag", required=True)
    eval_ap.add_argument("--checkpoint", default="Best")
    eval_ap.add_argument("--convir-dir", type=Path, required=True)
    eval_ap.add_argument("--data-dir", type=Path, required=True)
    eval_ap.add_argument("--split-manifest", type=Path, required=True)
    eval_ap.add_argument("--a0-checkpoint", type=Path, required=True)
    eval_ap.add_argument("--student-checkpoint", type=Path, required=True)
    eval_ap.add_argument("--out-dir", type=Path, required=True)
    eval_ap.add_argument("--feature-mode", default="rgb_wavelet", choices=["rgb", "rgb_wavelet"])
    eval_ap.add_argument("--adapter-width", type=int, default=32)
    eval_ap.add_argument("--adapter-depth", type=int, default=3)
    eval_ap.add_argument("--bootstrap-scale", type=float, default=0.0)
    eval_ap.add_argument("--residual-mode", default="direct", choices=["gated_bootstrap", "direct", "adaptive_scalar"])
    eval_ap.add_argument("--residual-scale", type=float, default=1.0)
    eval_ap.add_argument("--scale-init", type=float, default=0.25)
    eval_ap.add_argument("--head-init", default="zero", choices=["kaiming", "zero"])
    eval_ap.add_argument("--clamp-output", action="store_true")

    teacher_ap = sub.add_parser("teacher-c8")
    teacher_ap.add_argument("--split-manifest", type=Path, required=True)
    teacher_ap.add_argument("--c8-per-image", type=Path, required=True)
    teacher_ap.add_argument("--out-dir", type=Path, required=True)

    oracle_ap = sub.add_parser("oracle")
    oracle_ap.add_argument("--convir-dir", type=Path, required=True)
    oracle_ap.add_argument("--data-dir", type=Path, required=True)
    oracle_ap.add_argument("--split-manifest", type=Path, required=True)
    oracle_ap.add_argument("--a0-checkpoint", type=Path, required=True)
    oracle_ap.add_argument("--student-checkpoint", type=Path, required=True)
    oracle_ap.add_argument("--c8-per-image", type=Path, required=False)
    oracle_ap.add_argument("--out-dir", type=Path, required=True)
    oracle_ap.add_argument("--feature-mode", default="rgb_wavelet", choices=["rgb", "rgb_wavelet"])
    oracle_ap.add_argument("--adapter-width", type=int, default=32)
    oracle_ap.add_argument("--adapter-depth", type=int, default=3)
    oracle_ap.add_argument("--actual-scale", type=float, default=0.50)
    oracle_ap.add_argument("--scales", default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    oracle_ap.add_argument("--patch-size", type=int, default=64)

    agg_ap = sub.add_parser("aggregate")
    agg_ap.add_argument("--out-dir", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "eval-c13":
        eval_candidate(args.tag, args)
    elif args.cmd == "teacher-c8":
        read_c8_teacher(args)
    elif args.cmd == "oracle":
        run_oracles(args)
    elif args.cmd == "aggregate":
        aggregate(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
