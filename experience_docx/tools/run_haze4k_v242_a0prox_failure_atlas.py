#!/usr/bin/env python3
"""Run the v2.42 diagnostic-only A0PROX failure atlas.

This tool is cloud-only. It performs read-only evaluation of the completed
v2.41 canary32 checkpoints and writes compact diagnostic summaries. It does not
train, update parameters, select a deployable threshold, launch canary80, or
touch locked test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as VF


IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")
GAMMAS = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)


def first_existing_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"none of {names} exists under {root}")


def image_names(path: Path) -> list[str]:
    return sorted(
        p.name
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )


def label_path(label_dir: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    ext = Path(image_name).suffix
    candidates = [image_name]
    if "_" in stem:
        candidates.append(f"{stem.split('_')[0]}{ext}")
        candidates.append(f"{stem.split('_')[0]}.png")
    for candidate in candidates:
        path = label_dir / candidate
        if path.is_file():
            return path
    return None


def list_haze4k_records(data_dir: Path, split: str) -> tuple[list[dict[str, str]], list[str]]:
    split_dir = data_dir / split
    input_dir = first_existing_dir(split_dir, ("IN", "haze", "hazy"))
    label_dir = first_existing_dir(split_dir, ("GT", "gt"))
    records = []
    skipped = []
    for name in image_names(input_dir):
        gt = label_path(label_dir, name)
        if gt is None:
            skipped.append(name)
            continue
        records.append({"name": name, "input": str(input_dir / name), "label": str(gt)})
    if not records:
        raise RuntimeError(f"no matched Haze4K records under {split_dir}")
    return records, skipped


def load_checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint is not a state dict: {path}")
    return state


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1e-12))


def pad_to_factor(x: torch.Tensor, factor: int = 32) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    h2 = ((h + factor) // factor) * factor
    w2 = ((w + factor) // factor) * factor
    padh = 0 if h % factor == 0 else h2 - h
    padw = 0 if w % factor == 0 else w2 - w
    if padh == 0 and padw == 0:
        return x, h, w
    return F.pad(x, (0, padw, 0, padh), "reflect"), h, w


def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return VF.to_tensor(image).unsqueeze(0).to(device)


def build_models(
    repo_root: Path,
    official_checkpoint: Path,
    route_checkpoint: Path,
    beta: float,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    sys.path.insert(0, str(repo_root / "Dehazing" / "ITS"))
    from models.A0ProxResidualConvIR import build_a0prox_residual_net
    from models.ConvIR import build_net

    base = build_net("base", "Haze4K")
    route = build_a0prox_residual_net("base", "Haze4K", beta=beta)
    base.load_state_dict(load_checkpoint_model(official_checkpoint), strict=True)
    route.load_state_dict(load_checkpoint_model(route_checkpoint), strict=True)
    base.eval().to(device)
    route.eval().to(device)
    for param in base.parameters():
        param.requires_grad = False
    for param in route.parameters():
        param.requires_grad = False
    return route, base


def bool_from_csv(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def to_float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    return float(value)


def maybe_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_rows(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, Any]:
    if not rows:
        raise RuntimeError("cannot summarize empty rows")
    ordered_by_a0 = sorted(rows, key=lambda r: float(r["a0_psnr"]))
    bucket_n = max(1, int(math.ceil(0.25 * len(rows))))
    hard_names = {r["image"] for r in ordered_by_a0[:bucket_n]}
    easy_names = {r["image"] for r in ordered_by_a0[-bucket_n:]}
    deltas = [float(r["delta_psnr"]) for r in rows]
    hard = [float(r["delta_psnr"]) for r in rows if r["image"] in hard_names]
    easy = [float(r["delta_psnr"]) for r in rows if r["image"] in easy_names]
    residual_hard = [
        float(r.get("residual_energy", 0.0)) for r in rows if r["image"] in hard_names
    ]
    residual_easy = [
        float(r.get("residual_energy", 0.0)) for r in rows if r["image"] in easy_names
    ]
    bottom = sorted(deltas)[: max(1, int(math.ceil(0.05 * len(deltas))))]
    severe = sum(1 for d in deltas if d <= -0.20)
    strong_ref = sum(
        1 for r in rows if r["image"] in easy_names and float(r["delta_psnr"]) < -0.01
    )
    hard_energy = sum(residual_hard) / len(residual_hard)
    easy_energy = sum(residual_easy) / len(residual_easy)
    out = {
        f"{prefix}count": len(rows),
        f"{prefix}mean_delta": sum(deltas) / len(deltas),
        f"{prefix}hard_delta": sum(hard) / len(hard),
        f"{prefix}easy_delta": sum(easy) / len(easy),
        f"{prefix}p05_delta": percentile(deltas, 0.05),
        f"{prefix}cvar5_delta": sum(bottom) / len(bottom),
        f"{prefix}worst_delta": min(deltas),
        f"{prefix}severe_count": severe,
        f"{prefix}strong_reference_regression_count": strong_ref,
        f"{prefix}residual_energy_hard_mean": hard_energy,
        f"{prefix}residual_energy_easy_mean": easy_energy,
        f"{prefix}residual_energy_easy_to_hard_ratio": (
            easy_energy / hard_energy if hard_energy > 0 else 1e12
        ),
    }
    return out


def fold_gate(summary: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        summary["mean_delta"] >= args.gate_mean_delta
        and summary["hard_delta"] >= args.gate_hard_delta
        and summary["easy_delta"] >= args.gate_easy_delta
        and summary["p05_delta"] >= args.gate_p05_delta
        and summary["cvar5_delta"] >= args.gate_cvar5_delta
        and summary["severe_count"] == 0
        and summary["strong_reference_regression_count"] == 0
    )


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_records(
    model: torch.nn.Module,
    base: torch.nn.Module,
    records: list[dict[str, str]],
    fold: int,
    split: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    with torch.no_grad():
        for rec in records:
            input_img = load_image_tensor(rec["input"], device)
            label_img = load_image_tensor(rec["label"], device)
            padded, h, w = pad_to_factor(input_img)
            a0_raw = base(padded)[2][:, :, :h, :w]
            pred_raw = model(padded)[2][:, :, :h, :w]
            a0 = a0_raw.clamp(0.0, 1.0)
            pred = pred_raw.clamp(0.0, 1.0)
            e0 = a0 - label_img
            residual = pred - a0
            a0_mse = float(e0.pow(2).mean().item())
            pred_mse = float((pred - label_img).pow(2).mean().item())
            residual_energy = float(residual.pow(2).mean().item())
            gt_error_energy = a0_mse
            alignment_dot = float((e0 * residual).mean().item())
            denom = math.sqrt(max(gt_error_energy * residual_energy, 0.0))
            alignment_cos = alignment_dot / denom if denom > 0 else None
            alpha_safe_upper = (
                -2.0 * alignment_dot / residual_energy
                if residual_energy > 1.0e-20
                else None
            )
            alpha_opt = (
                -alignment_dot / residual_energy
                if residual_energy > 1.0e-20
                else None
            )
            mse_delta = pred_mse - a0_mse
            identity_mse_delta = 2.0 * alignment_dot + residual_energy
            direction_bad = alignment_dot >= 0.0
            overshoot_bad = (
                alignment_dot < 0.0
                and alpha_safe_upper is not None
                and alpha_safe_upper < 1.0
            )
            pred_raw_oob = ((pred_raw < 0.0) | (pred_raw > 1.0)).float().mean()
            a0_raw_oob = ((a0_raw < 0.0) | (a0_raw > 1.0)).float().mean()
            a0_psnr = psnr_from_mse(a0_mse)
            pred_psnr = psnr_from_mse(pred_mse)
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "image": rec["name"],
                    "a0_mse": a0_mse,
                    "pred_mse": pred_mse,
                    "mse_delta": mse_delta,
                    "identity_mse_delta": identity_mse_delta,
                    "mse_identity_abs_error": abs(mse_delta - identity_mse_delta),
                    "a0_psnr": a0_psnr,
                    "pred_psnr": pred_psnr,
                    "delta_psnr": pred_psnr - a0_psnr,
                    "residual_energy": residual_energy,
                    "gt_error_energy": gt_error_energy,
                    "alignment_dot": alignment_dot,
                    "alignment_cos": maybe_float(alignment_cos),
                    "alpha_safe_upper": maybe_float(alpha_safe_upper),
                    "alpha_opt": maybe_float(alpha_opt),
                    "direction_bad": bool(direction_bad),
                    "overshoot_bad": bool(overshoot_bad),
                    "pred_raw_min": float(pred_raw.min().item()),
                    "pred_raw_max": float(pred_raw.max().item()),
                    "a0_raw_min": float(a0_raw.min().item()),
                    "a0_raw_max": float(a0_raw.max().item()),
                    "pred_raw_oob_fraction": float(pred_raw_oob.item()),
                    "a0_raw_oob_fraction": float(a0_raw_oob.item()),
                }
            )
            del input_img, label_img, padded, a0_raw, pred_raw, a0, pred
    return rows


def compare_recompute(
    original_rows: list[dict[str, str]], recomputed_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    recomputed_by_key = {
        (str(row["fold"]), row["image"]): row for row in recomputed_rows
    }
    numeric_tolerances = {
        "a0_mse": 1.0e-9,
        "pred_mse": 1.0e-9,
        "a0_psnr": 1.0e-4,
        "pred_psnr": 1.0e-4,
        "delta_psnr": 1.0e-4,
        "residual_energy": 1.0e-9,
    }
    mismatches = []
    max_abs = {key: 0.0 for key in numeric_tolerances}
    missing = []
    for old in original_rows:
        key = (str(old["fold"]), old["image"])
        new = recomputed_by_key.get(key)
        if new is None:
            missing.append({"fold": old["fold"], "image": old["image"]})
            continue
        for field, tol in numeric_tolerances.items():
            diff = abs(to_float(old[field]) - float(new[field]))
            max_abs[field] = max(max_abs[field], diff)
            if diff > tol:
                mismatches.append(
                    {
                        "fold": old["fold"],
                        "image": old["image"],
                        "field": field,
                        "old": old[field],
                        "recomputed": new[field],
                        "abs_diff": diff,
                        "tolerance": tol,
                    }
                )
    audit = {
        "original_count": len(original_rows),
        "recomputed_count": len(recomputed_rows),
        "missing_count": len(missing),
        "mismatch_count": len(mismatches),
        "max_abs_diff": max_abs,
        "gate_pass": len(missing) == 0 and len(mismatches) == 0,
        "missing": missing[:20],
    }
    return audit, mismatches


def residual_geometry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    severe_rows = [r for r in rows if float(r["delta_psnr"]) <= -0.20]
    positive_rows = [r for r in rows if float(r["delta_psnr"]) > 0.0]
    negative_rows = [r for r in rows if float(r["delta_psnr"]) < 0.0]

    def count_flag(subset: list[dict[str, Any]], field: str) -> int:
        return sum(1 for r in subset if bool(r[field]))

    def mean_of(subset: list[dict[str, Any]], field: str) -> float | None:
        vals = [float(r[field]) for r in subset if r.get(field) is not None]
        return sum(vals) / len(vals) if vals else None

    def q_of(subset: list[dict[str, Any]], field: str, q: float) -> float | None:
        vals = [float(r[field]) for r in subset if r.get(field) is not None]
        return percentile(vals, q)

    out = {
        "count": len(rows),
        "global_summary": summarize_rows(rows),
        "positive_count": len(positive_rows),
        "negative_count": len(negative_rows),
        "severe_count": len(severe_rows),
        "direction_bad_count": count_flag(rows, "direction_bad"),
        "overshoot_bad_count": count_flag(rows, "overshoot_bad"),
        "severe_direction_bad_count": count_flag(severe_rows, "direction_bad"),
        "severe_overshoot_bad_count": count_flag(severe_rows, "overshoot_bad"),
        "severe_direction_bad_rate": (
            count_flag(severe_rows, "direction_bad") / len(severe_rows)
            if severe_rows
            else None
        ),
        "severe_overshoot_bad_rate": (
            count_flag(severe_rows, "overshoot_bad") / len(severe_rows)
            if severe_rows
            else None
        ),
        "alignment_dot_mean": mean_of(rows, "alignment_dot"),
        "alignment_dot_p05": q_of(rows, "alignment_dot", 0.05),
        "alignment_dot_p50": q_of(rows, "alignment_dot", 0.50),
        "alignment_cos_mean": mean_of(rows, "alignment_cos"),
        "alpha_safe_upper_p05": q_of(rows, "alpha_safe_upper", 0.05),
        "alpha_safe_upper_p50": q_of(rows, "alpha_safe_upper", 0.50),
        "severe_alpha_safe_upper_p50": q_of(severe_rows, "alpha_safe_upper", 0.50),
        "residual_energy_mean": mean_of(rows, "residual_energy"),
        "severe_residual_energy_mean": mean_of(severe_rows, "residual_energy"),
        "mse_identity_abs_error_max": max(
            float(r["mse_identity_abs_error"]) for r in rows
        ),
        "pred_raw_oob_fraction_max": max(
            float(r["pred_raw_oob_fraction"]) for r in rows
        ),
        "a0_raw_oob_fraction_max": max(float(r["a0_raw_oob_fraction"]) for r in rows),
        "severe_pred_raw_oob_fraction_mean": mean_of(
            severe_rows, "pred_raw_oob_fraction"
        ),
        "worst_images": [
            {
                "fold": r["fold"],
                "image": r["image"],
                "delta_psnr": r["delta_psnr"],
                "direction_bad": r["direction_bad"],
                "overshoot_bad": r["overshoot_bad"],
                "alignment_dot": r["alignment_dot"],
                "alpha_safe_upper": r["alpha_safe_upper"],
                "residual_energy": r["residual_energy"],
            }
            for r in sorted(rows, key=lambda x: float(x["delta_psnr"]))[:10]
        ],
    }
    return out


def shrink_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    output = []
    for gamma in GAMMAS:
        gamma_rows = []
        for row in rows:
            mse = (
                float(row["gt_error_energy"])
                + 2.0 * gamma * float(row["alignment_dot"])
                + gamma * gamma * float(row["residual_energy"])
            )
            psnr = psnr_from_mse(mse)
            gamma_rows.append(
                {
                    "fold": row["fold"],
                    "image": row["image"],
                    "a0_psnr": row["a0_psnr"],
                    "pred_psnr": psnr,
                    "delta_psnr": psnr - float(row["a0_psnr"]),
                    "residual_energy": gamma * gamma * float(row["residual_energy"]),
                }
            )
        global_summary = summarize_rows(gamma_rows)
        fold_pass_count = 0
        for fold in sorted({int(r["fold"]) for r in gamma_rows}):
            fold_summary = summarize_rows([r for r in gamma_rows if int(r["fold"]) == fold])
            if fold_gate(fold_summary, args):
                fold_pass_count += 1
        global_gate = fold_gate(global_summary, args)
        route_gate = global_gate and fold_pass_count >= args.gate_min_fold_pass
        output.append(
            {
                "gamma": gamma,
                **global_summary,
                "fold_pass_count": fold_pass_count,
                "global_gate_pass": global_gate,
                "gate_pass": route_gate,
            }
        )
    return output


def oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    oracle_rows = []
    chosen = 0
    for row in rows:
        use_pred = float(row["delta_psnr"]) > 0.0
        chosen += int(use_pred)
        delta = float(row["delta_psnr"]) if use_pred else 0.0
        oracle_rows.append(
            {
                "fold": row["fold"],
                "image": row["image"],
                "a0_psnr": row["a0_psnr"],
                "pred_psnr": float(row["a0_psnr"]) + delta,
                "delta_psnr": delta,
                "residual_energy": row["residual_energy"] if use_pred else 0.0,
            }
        )
    summary = summarize_rows(oracle_rows)
    summary.update(
        {
            "oracle_positive_rate": chosen / len(rows),
            "oracle_positive_count": chosen,
            "oracle_severe": summary["severe_count"],
            "oracle_p05_delta": summary["p05_delta"],
            "oracle_cvar5_delta": summary["cvar5_delta"],
            "oracle_mean_delta": summary["mean_delta"],
            "oracle_hard_delta": summary["hard_delta"],
            "oracle_easy_delta": summary["easy_delta"],
        }
    )
    return summary


def train_vs_oof_rows(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    for split, source in (("train32", train_rows), ("oof32", val_rows)):
        global_summary = summarize_rows(source)
        rows.append(
            {
                "fold": "global",
                "split": split,
                **global_summary,
                "gate_pass": fold_gate(global_summary, args),
            }
        )
        for fold in sorted({int(r["fold"]) for r in source}):
            subset = [r for r in source if int(r["fold"]) == fold]
            summary = summarize_rows(subset)
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    **summary,
                    "gate_pass": fold_gate(summary, args),
                }
            )
    return rows


def filename_params(name: str) -> dict[str, float | None]:
    parts = Path(name).stem.split("_")
    out: dict[str, float | None] = {"filename_param_1": None, "filename_param_2": None}
    if len(parts) >= 3:
        try:
            out["filename_param_1"] = float(parts[1])
            out["filename_param_2"] = float(parts[2])
        except ValueError:
            pass
    return out


def numeric_range(rows: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    vals = [to_float(r.get(field), default=float("nan")) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(vals), "mean": sum(vals) / len(vals), "max": max(vals)}


def group_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(r.get(field, "")) for r in rows).items()))


def v240_cross_summary(rows: list[dict[str, Any]], v240_rows: list[dict[str, str]]) -> dict[str, Any]:
    v240_by_name = {r["image_name"]: r for r in v240_rows}
    joined = []
    missing = []
    for row in rows:
        ref = v240_by_name.get(row["image"])
        if ref is None:
            missing.append(row["image"])
            continue
        params = filename_params(row["image"])
        joined.append(
            {
                **row,
                **params,
                "v240_hardness_bucket": ref["hardness_bucket"],
                "v240_easy_bucket": ref["easy_bucket"],
                "v240_strong_reference_bucket": ref["strong_reference_bucket"],
                "v240_wdmamba_headroom_mse": to_float(ref["wdmamba_headroom_mse"]),
                "v240_wdmamba_alpha_safe_upper": to_float(ref["wdmamba_alpha_safe_upper"]),
                "v240_wdmamba_anti_aligned": bool_from_csv(ref["wdmamba_anti_aligned"]),
                "v240_wdmamba_unsafe": bool_from_csv(ref["unsafe_wdmamba_alpha003125"])
                or bool_from_csv(ref["unsafe_wdmamba_alpha0125"]),
                "v240_convirl_unsafe": bool_from_csv(ref["unsafe_convirl_alpha0015625"])
                or bool_from_csv(ref["unsafe_convirl_alpha025"]),
                "v240_union_useful_alpha_unsafe": bool_from_csv(
                    ref["union_useful_alpha_unsafe"]
                ),
                "v240_shared_useful_alpha_unsafe": bool_from_csv(
                    ref["shared_useful_alpha_unsafe"]
                ),
                "v240_delta_wdmamba_vs_A0": to_float(ref["delta_wdmamba_vs_A0"]),
                "v240_delta_convirl_vs_A0": to_float(ref["delta_convirl_vs_A0"]),
            }
        )
    severe = [r for r in joined if float(r["delta_psnr"]) <= -0.20]
    positives = [r for r in joined if float(r["delta_psnr"]) > 0.0]

    def count_bool(subset: list[dict[str, Any]], field: str) -> int:
        return sum(1 for r in subset if bool(r[field]))

    out = {
        "joined_count": len(joined),
        "missing_count": len(missing),
        "coverage_fraction": len(joined) / len(rows) if rows else 0.0,
        "coverage_limited": len(joined) < len(rows),
        "coverage_note": (
            "v2.40 atlas covers only the joined subset; cross-over findings are "
            "supporting diagnostics, not a full v2.41 OOF explanation."
        ),
        "missing_examples": missing[:20],
        "v241_severe_count": len(severe),
        "v241_positive_count": len(positives),
        "severe_intersections": {
            "wdmamba_unsafe": count_bool(severe, "v240_wdmamba_unsafe"),
            "convirl_unsafe": count_bool(severe, "v240_convirl_unsafe"),
            "union_useful_alpha_unsafe": count_bool(
                severe, "v240_union_useful_alpha_unsafe"
            ),
            "shared_useful_alpha_unsafe": count_bool(
                severe, "v240_shared_useful_alpha_unsafe"
            ),
            "wdmamba_anti_aligned": count_bool(severe, "v240_wdmamba_anti_aligned"),
        },
        "positive_intersections": {
            "wdmamba_aligned_positive": sum(
                1
                for r in positives
                if not r["v240_wdmamba_anti_aligned"]
                and float(r["v240_delta_wdmamba_vs_A0"]) > 0.0
            ),
            "wdmamba_unsafe": count_bool(positives, "v240_wdmamba_unsafe"),
            "convirl_unsafe": count_bool(positives, "v240_convirl_unsafe"),
        },
        "severe_by_v240_hardness_bucket": group_counts(severe, "v240_hardness_bucket"),
        "severe_by_v240_easy_bucket": group_counts(severe, "v240_easy_bucket"),
        "severe_by_v240_strong_reference_bucket": group_counts(
            severe, "v240_strong_reference_bucket"
        ),
        "severe_by_v241_fold": group_counts(severe, "fold"),
        "all_filename_param_1": numeric_range(joined, "filename_param_1"),
        "all_filename_param_2": numeric_range(joined, "filename_param_2"),
        "severe_filename_param_1": numeric_range(severe, "filename_param_1"),
        "severe_filename_param_2": numeric_range(severe, "filename_param_2"),
        "all_v240_wdmamba_headroom_mse": numeric_range(
            joined, "v240_wdmamba_headroom_mse"
        ),
        "severe_v240_wdmamba_headroom_mse": numeric_range(
            severe, "v240_wdmamba_headroom_mse"
        ),
        "all_v240_wdmamba_alpha_safe_upper": numeric_range(
            joined, "v240_wdmamba_alpha_safe_upper"
        ),
        "severe_v240_wdmamba_alpha_safe_upper": numeric_range(
            severe, "v240_wdmamba_alpha_safe_upper"
        ),
    }
    return out


def decide_label(
    geometry: dict[str, Any],
    shrink_curve: list[dict[str, Any]],
    oracle: dict[str, Any],
    train_gap: list[dict[str, Any]],
) -> dict[str, Any]:
    severe = int(geometry["severe_count"])
    direction_rate = geometry["severe_direction_bad_rate"] or 0.0
    overshoot_rate = geometry["severe_overshoot_bad_rate"] or 0.0
    shrink_gate = any(bool(r["gate_pass"]) for r in shrink_curve)
    best_hard = max(float(r["hard_delta"]) for r in shrink_curve)
    oracle_strong = (
        float(oracle["oracle_mean_delta"]) >= 0.15
        and float(oracle["oracle_hard_delta"]) >= 0.30
    )
    train_global = next(
        r for r in train_gap if r["fold"] == "global" and r["split"] == "train32"
    )
    oof_global = next(
        r for r in train_gap if r["fold"] == "global" and r["split"] == "oof32"
    )
    train_gate = bool(train_global["gate_pass"])
    oof_gate = bool(oof_global["gate_pass"])
    if train_gate and not oof_gate:
        label = "A0PROX_OVERFIT_VARIANCE_FAIL"
        reason = "train32 global gate passed while OOF global gate failed"
    elif severe > 0 and direction_rate >= 0.50 and not shrink_gate and not oracle_strong:
        label = "A0PROX_DIRECTION_FAIL"
        reason = "severe regressions are mostly direction_bad and no gamma/oracle rescue is strong"
    elif severe > 0 and overshoot_rate > direction_rate and oracle_strong:
        label = "A0PROX_SCALE_FAIL_BUT_DIRECTION_EXISTS"
        reason = "severe regressions are mostly overshoot_bad and oracle upper bound is strong"
    elif oracle_strong:
        label = "A0PROX_SELECTION_FAIL_AGAIN"
        reason = "oracle upper bound is strong but direct OOF gate failed"
    else:
        label = "A0PROX_DIRECTION_FAIL"
        reason = "no shrink gate and oracle upper bound is not strong enough for a scale or selection rescue"
    return {
        "decision_label": label,
        "reason": reason,
        "severe_count": severe,
        "severe_direction_bad_rate": direction_rate,
        "severe_overshoot_bad_rate": overshoot_rate,
        "shrink_gate_exists": shrink_gate,
        "best_shrink_hard_delta": best_hard,
        "oracle_strong": oracle_strong,
        "train_global_gate_pass": train_gate,
        "oof_global_gate_pass": oof_gate,
    }


def run(args: argparse.Namespace) -> None:
    evidence_root = Path(args.evidence_root)
    evidence_root.mkdir(parents=True, exist_ok=True)
    runtime_dir = evidence_root / "runtime_logs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    status_path = evidence_root / "status.txt"
    status_path.write_text(
        f"RUNNING_DIAGNOSTIC start={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n",
        encoding="utf-8",
    )

    repo_root = Path(args.repo_root)
    data_dir = Path(args.data_dir)
    official_checkpoint = Path(args.official_checkpoint)
    v241_evidence = Path(args.v241_evidence_root)
    v241_summary = json.loads((v241_evidence / "v241_canary32_oof_summary.json").read_text())
    original_oof_rows = read_csv(v241_evidence / "v241_canary32_oof_per_image.csv")
    v240_rows = read_csv(Path(args.v240_atlas_csv))
    records, skipped = list_haze4k_records(data_dir, "train")
    record_by_name = {r["name"]: r for r in records}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_summaries = {int(r["fold"]): r for r in v241_summary["fold_summaries"]}
    val_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    checkpoint_manifest = []
    for detail in v241_summary["fold_details"]:
        fold = int(detail["fold"])
        best_checkpoint = Path(fold_summaries[fold]["best_checkpoint"])
        if not best_checkpoint.is_file():
            raise FileNotFoundError(f"missing v2.41 checkpoint: {best_checkpoint}")
        train_records = [record_by_name[name] for name in detail["train_images"]]
        val_records = [record_by_name[name] for name in detail["val_images"]]
        model, base = build_models(
            repo_root, official_checkpoint, best_checkpoint, args.beta, device
        )
        train_rows.extend(evaluate_records(model, base, train_records, fold, "train32", device))
        val_rows.extend(evaluate_records(model, base, val_records, fold, "oof32", device))
        checkpoint_manifest.append(
            {
                "fold": fold,
                "best_checkpoint": str(best_checkpoint),
                "best_checkpoint_sha256": sha256(best_checkpoint),
                "train_count": len(train_records),
                "val_count": len(val_records),
            }
        )
        del model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with status_path.open("a", encoding="utf-8") as f:
            f.write(f"FOLD_DONE fold={fold} time={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")

    audit, mismatches = compare_recompute(original_oof_rows, val_rows)
    geometry = residual_geometry_summary(val_rows)
    shrink_curve = shrink_rows(val_rows, args)
    oracle = oracle_summary(val_rows)
    gap_rows = train_vs_oof_rows(train_rows, val_rows, args)
    cross = v240_cross_summary(val_rows, v240_rows)
    decision = decide_label(geometry, shrink_curve, oracle, gap_rows)

    write_json(
        evidence_root / "v242_recompute_audit.json",
        {
            **audit,
            "route": "haze4k_v2_42_a0prox_failure_atlas",
            "stage": "DIAGNOSTIC_ONLY_RECOMPUTE",
            "v241_summary_path": str(v241_evidence / "v241_canary32_oof_summary.json"),
            "v241_per_image_path": str(v241_evidence / "v241_canary32_oof_per_image.csv"),
            "locked_test_touched": False,
            "canary80_touched": False,
        },
    )
    write_csv(
        evidence_root / "v242_recompute_mismatch.csv",
        mismatches,
        ["fold", "image", "field", "old", "recomputed", "abs_diff", "tolerance"],
    )
    write_json(evidence_root / "v242_a0prox_residual_geometry_summary.json", geometry)
    write_csv(
        evidence_root / "v242_a0prox_residual_geometry_per_image.csv",
        val_rows,
        [
            "fold",
            "split",
            "image",
            "a0_mse",
            "pred_mse",
            "mse_delta",
            "identity_mse_delta",
            "mse_identity_abs_error",
            "a0_psnr",
            "pred_psnr",
            "delta_psnr",
            "residual_energy",
            "gt_error_energy",
            "alignment_dot",
            "alignment_cos",
            "alpha_safe_upper",
            "alpha_opt",
            "direction_bad",
            "overshoot_bad",
            "pred_raw_min",
            "pred_raw_max",
            "a0_raw_min",
            "a0_raw_max",
            "pred_raw_oob_fraction",
            "a0_raw_oob_fraction",
        ],
    )
    write_csv(
        evidence_root / "v242_global_shrink_curve.csv",
        shrink_curve,
        [
            "gamma",
            "count",
            "mean_delta",
            "hard_delta",
            "easy_delta",
            "p05_delta",
            "cvar5_delta",
            "worst_delta",
            "severe_count",
            "strong_reference_regression_count",
            "residual_energy_hard_mean",
            "residual_energy_easy_mean",
            "residual_energy_easy_to_hard_ratio",
            "fold_pass_count",
            "global_gate_pass",
            "gate_pass",
        ],
    )
    write_json(evidence_root / "v242_oracle_clamp_upper_bound.json", oracle)
    write_csv(
        evidence_root / "v242_train_vs_oof_gap.csv",
        gap_rows,
        [
            "fold",
            "split",
            "count",
            "mean_delta",
            "hard_delta",
            "easy_delta",
            "p05_delta",
            "cvar5_delta",
            "worst_delta",
            "severe_count",
            "strong_reference_regression_count",
            "residual_energy_hard_mean",
            "residual_energy_easy_mean",
            "residual_energy_easy_to_hard_ratio",
            "gate_pass",
        ],
    )
    write_json(evidence_root / "v242_v240_cross_overlap_summary.json", cross)
    closeout = {
        "route": "haze4k_v2_42_a0prox_failure_atlas",
        "status": "COMPLETED_DIAGNOSTIC",
        "decision": decision["decision_label"],
        "decision_details": decision,
        "diagnostic_only": True,
        "training_touched": False,
        "parameter_updates": False,
        "canary80_touched": False,
        "locked_test_touched": False,
        "threshold_selected_for_deployment": False,
        "new_model_claim": False,
        "device": str(device),
        "data_dir": str(data_dir),
        "official_checkpoint": str(official_checkpoint),
        "official_checkpoint_sha256": sha256(official_checkpoint),
        "checkpoint_manifest": checkpoint_manifest,
        "skipped_unmatched_train_images": skipped,
        "v241_decision": v241_summary.get("decision"),
        "recompute_gate_pass": audit["gate_pass"],
        "recompute_mismatch_count": audit["mismatch_count"],
        "geometry_summary_path": str(
            evidence_root / "v242_a0prox_residual_geometry_summary.json"
        ),
        "shrink_curve_path": str(evidence_root / "v242_global_shrink_curve.csv"),
        "oracle_path": str(evidence_root / "v242_oracle_clamp_upper_bound.json"),
        "train_vs_oof_path": str(evidence_root / "v242_train_vs_oof_gap.csv"),
        "v240_cross_path": str(evidence_root / "v242_v240_cross_overlap_summary.json"),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(evidence_root / "v242_closeout.json", closeout)
    with status_path.open("a", encoding="utf-8") as f:
        f.write(
            "COMPLETED_DIAGNOSTIC "
            f"decision={decision['decision_label']} "
            f"time={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        )
    print(json.dumps(closeout, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--official-checkpoint", required=True)
    parser.add_argument("--v241-evidence-root", required=True)
    parser.add_argument("--v240-atlas-csv", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--gate-mean-delta", type=float, default=0.15)
    parser.add_argument("--gate-hard-delta", type=float, default=0.30)
    parser.add_argument("--gate-easy-delta", type=float, default=0.0)
    parser.add_argument("--gate-p05-delta", type=float, default=-0.01)
    parser.add_argument("--gate-cvar5-delta", type=float, default=-0.02)
    parser.add_argument("--gate-min-fold-pass", type=int, default=4)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
