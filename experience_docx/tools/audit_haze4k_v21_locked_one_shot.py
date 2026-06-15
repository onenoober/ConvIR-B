#!/usr/bin/env python3
"""Authorized locked one-shot replay for Haze4K v2.1 SEG-Mix.

This script is only valid after C10 formal 5x3 writes
`C10_FORMAL_5X3_STRONG_PASS_AUTHORIZE_LOCKED_ONE_SHOT`. It applies the sealed
`riskcap36_no075` C10 policy family to the locked Haze4K test split once and
writes text evidence only. Locked results must not feed any policy selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TVF

import audit_haze4k_v20_c2_outputdiff_router as c2
from audit_haze4k_v20_c2b_multirule_router import fnum, write_csv
from audit_haze4k_v21_c10_formal_5x3 import SEEDS, choose_seed_fold_policies, seeded_fold_id
from audit_haze4k_v21_c7b_local_alpha_prototype import (
    ALPHAS,
    C7B_STRONG_GATE,
    PatchTable,
    alpha_key,
    gate_pass,
    patch_rows_for_image_tensors,
    summarize_actual_rows,
)
from audit_haze4k_v21_c7c_local_alpha_risk_tighten import iter_patches


MAX_SEED_SEVERE = 60.0
AUTHORIZED_DECISION = "C10_FORMAL_5X3_STRONG_PASS_AUTHORIZE_LOCKED_ONE_SHOT"


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_existing_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"none of {names} exists under {root}")


def list_split_images(data_dir: Path, split: str) -> list[str]:
    input_dir = first_existing_dir(data_dir / split, ("IN", "haze", "hazy"))
    names = [path.name for path in input_dir.iterdir() if path.is_file()]
    return sorted(names)


def label_path(label_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    ext = Path(image_name).suffix
    candidates = [image_name]
    if "_" in stem:
        candidates.extend([f"{stem.split('_')[0]}{ext}", f"{stem.split('_')[0]}.png"])
    for candidate in candidates:
        path = label_dir / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"no label for {image_name} in {label_dir}; tried {candidates}")


def depth_path(depth_cache: Path, depth_split: str, image_name: str) -> Path:
    candidates = [
        depth_cache / depth_split / f"{image_name.replace('/', '__')}.npy",
        depth_cache / depth_split / f"{image_name}.npy",
        depth_cache / f"{image_name.replace('/', '__')}.npy",
        depth_cache / f"{image_name}.npy",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing depth cache for {image_name}; tried {candidates}")


def normalize_depth_minmax(depth: np.ndarray) -> np.ndarray:
    lo = float(np.nanmin(depth))
    hi = float(np.nanmax(depth))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - lo) / (hi - lo + 1e-6)).astype(np.float32)


def load_locked_sample(
    data_dir: Path, depth_cache: Path, data_split: str, depth_split: str, image_name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_root = data_dir / data_split
    input_dir = first_existing_dir(split_root, ("IN", "haze", "hazy"))
    gt_dir = first_existing_dir(split_root, ("GT", "gt"))
    image = Image.open(input_dir / image_name).convert("RGB")
    label = Image.open(label_path(gt_dir, image_name)).convert("RGB")
    depth_arr = np.load(depth_path(depth_cache, depth_split, image_name)).astype(np.float32)
    depth_arr = np.nan_to_num(depth_arr, nan=0.0, posinf=0.0, neginf=0.0)
    if depth_arr.ndim == 3:
        depth_arr = np.squeeze(depth_arr)
    depth_arr = normalize_depth_minmax(depth_arr)
    depth_img = Image.fromarray(depth_arr, mode="F")
    if depth_img.size != image.size:
        depth_img = depth_img.resize(image.size, resample=Image.BICUBIC)
    return TVF.to_tensor(image), TVF.to_tensor(label), TVF.to_tensor(depth_img).float()


def actions_for_patch_rows(policy_id: str, rows: list[dict[str, Any]]) -> np.ndarray:
    from audit_haze4k_v21_c7b_local_alpha_prototype import FEATURES, apply_policy

    image_rows = [{"name": "one", "split": "locked_test", "A0_PSNR": 0.0}]
    table_rows = []
    for row in rows:
        rec: dict[str, Any] = {"name": "one", "pixel_count": 1}
        rec.update({feat: row[feat] for feat in FEATURES})
        for alpha in ALPHAS:
            rec[f"sse_{alpha_key(alpha)}"] = 0.0
        table_rows.append(rec)
    return apply_policy(PatchTable(table_rows, image_rows), policy_id)


def assert_c10_authorized(summary_path: Path, fixed_profile: str) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("decision") != AUTHORIZED_DECISION:
        raise RuntimeError(f"C10 decision does not authorize locked: {payload.get('decision')}")
    if not payload.get("aggregate", {}).get("strong_formal_gate_pass"):
        raise RuntimeError("C10 strong_formal_gate_pass is not true")
    if payload.get("locked_test_touched") is not False:
        raise RuntimeError("C10 summary must state locked_test_touched=false")
    if payload.get("fixed_profile") != fixed_profile:
        raise RuntimeError(
            f"fixed profile mismatch: summary={payload.get('fixed_profile')} requested={fixed_profile}"
        )
    return payload


def eval_locked(args: argparse.Namespace, policies: dict[int, dict[int, str]]) -> list[dict[str, Any]]:
    _loader, build_convir_net = c2.load_convir_builders(Path(args.convir_its_dir))
    build_udpnet = c2.load_udpnet_builder(Path(args.udp_repo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    a0_model = c2.load_a0_model(build_convir_net, Path(args.a0_checkpoint), device)
    udp_model, _ckpt_meta = c2.load_udpnet_model(build_udpnet, Path(args.official_checkpoint), device)
    names = list_split_images(Path(args.data_dir), args.data_split)
    if args.max_images:
        names = names[: args.max_images]
    rows: list[dict[str, Any]] = []
    factor = int(args.pad_factor)
    with torch.no_grad():
        for idx, image_name in enumerate(names):
            input_img, label_img, depth = load_locked_sample(
                Path(args.data_dir),
                Path(args.depth_cache_dir),
                args.data_split,
                args.depth_split,
                image_name,
            )
            input_img = input_img.unsqueeze(0).to(device)
            label_img = label_img.unsqueeze(0).to(device)
            depth = depth.unsqueeze(0).to(device)
            h, w = input_img.shape[2], input_img.shape[3]
            h_pad = ((h + factor) // factor) * factor
            w_pad = ((w + factor) // factor) * factor
            padh = h_pad - h if h % factor != 0 else 0
            padw = w_pad - w if w % factor != 0 else 0
            rgb_padded = F.pad(input_img, (0, padw, 0, padh), "reflect")
            depth_padded = F.pad(depth, (0, padw, 0, padh), "reflect")
            a0_pred = c2.infer_one(a0_model, rgb_padded, h, w)
            udp_pred = c2.infer_one(udp_model, torch.cat([rgb_padded, depth_padded], dim=1), h, w)
            a0_psnr, a0_ssim = c2.metric_pair(a0_pred, label_img, (h_pad, w_pad))
            patch_rows = patch_rows_for_image_tensors(
                image_name,
                "locked_test",
                input_img,
                depth,
                a0_pred,
                udp_pred,
                int(args.patch_size),
            )
            residual = udp_pred - a0_pred
            for seed in SEEDS:
                fold = seeded_fold_id(image_name, seed)
                policy_id = policies[seed][fold]
                actions = actions_for_patch_rows(policy_id, patch_rows)
                pred = a0_pred.clone()
                counts = {alpha: 0 for alpha in ALPHAS}
                for (_pid, y, y2, x, x2), action in zip(
                    iter_patches(h, w, int(args.patch_size)), actions, strict=False
                ):
                    alpha = ALPHAS[int(action)]
                    counts[alpha] += 1
                    pred[..., y:y2, x:x2] = torch.clamp(
                        a0_pred[..., y:y2, x:x2] + alpha * residual[..., y:y2, x:x2],
                        0,
                        1,
                    )
                psnr, ssim = c2.metric_pair(pred, label_img, (h_pad, w_pad))
                rec: dict[str, Any] = {
                    "seed": seed,
                    "name": image_name,
                    "split": "locked_test",
                    "fold": fold,
                    "policy_id": policy_id,
                    "A0_PSNR": a0_psnr,
                    "A0_SSIM": a0_ssim,
                    "locked_PSNR": psnr,
                    "locked_SSIM": ssim,
                    "dPSNR": psnr - a0_psnr,
                    "dSSIM": ssim - a0_ssim,
                    "patch_count": sum(counts.values()),
                }
                for alpha in ALPHAS:
                    rec[f"patch_action_fraction_{alpha_key(alpha)}"] = counts[alpha] / max(1, rec["patch_count"])
                rows.append(rec)
            if (idx + 1) % args.print_freq == 0:
                print(f"locked_actual {idx + 1}/{len(names)} rows={len(rows)}", flush=True)
    return rows


def mean_std(vals: list[float]) -> tuple[float, float]:
    return (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_rows", type=Path, required=True)
    parser.add_argument("--image_rows", type=Path, required=True)
    parser.add_argument("--c10_summary", type=Path, required=True)
    parser.add_argument("--fixed_profile", default="riskcap36_no075")
    parser.add_argument("--convir_its_dir", required=True)
    parser.add_argument("--udp_repo", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_split", default="test")
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--depth_split", default="test")
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--official_checkpoint", required=True)
    parser.add_argument("--pad_factor", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--low_pool_limit", type=int, default=80)
    parser.add_argument("--high_pool_limit", type=int, default=120)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    c10_payload = assert_c10_authorized(args.c10_summary, args.fixed_profile)
    patch_rows = read_csv(args.patch_rows)
    image_rows = read_csv(args.image_rows)
    table = PatchTable(patch_rows, image_rows)
    policies, fold_rows = choose_seed_fold_policies(table, image_rows, args.fixed_profile, args)
    actual = eval_locked(args, policies)

    seed_rows = []
    for seed in SEEDS:
        rec = {"seed": seed, **summarize_actual_rows([r for r in actual if int(r["seed"]) == seed])}
        rec["strong_gate_pass"] = gate_pass(rec, C7B_STRONG_GATE)
        seed_rows.append(rec)
    metrics = [
        "mean_dPSNR",
        "hard_bottom25_dPSNR",
        "easy_top25_dPSNR",
        "dSSIM",
        "positive_ratio",
        "nonnegative_ratio",
        "severe_loss_per_600",
        "selected_precision",
    ]
    aggregate: dict[str, Any] = {
        "seed_count": len(seed_rows),
        "locked_image_count_per_seed": int(seed_rows[0]["count"]) if seed_rows else 0,
        "fold_count": len(fold_rows),
    }
    for metric in metrics:
        mean, sd = mean_std([fnum(row[metric]) for row in seed_rows])
        aggregate[f"{metric}_mean"] = mean
        aggregate[f"{metric}_std"] = sd
    aggregate["max_seed_severe_loss_per_600"] = max(fnum(row["severe_loss_per_600"]) for row in seed_rows)
    aggregate["all_seed_strong_gate_pass"] = all(bool(row["strong_gate_pass"]) for row in seed_rows)
    strong_mean = {
        "mean_dPSNR": aggregate["mean_dPSNR_mean"],
        "hard_bottom25_dPSNR": aggregate["hard_bottom25_dPSNR_mean"],
        "easy_top25_dPSNR": aggregate["easy_top25_dPSNR_mean"],
        "dSSIM": aggregate["dSSIM_mean"],
        "positive_ratio": aggregate["positive_ratio_mean"],
        "severe_loss_per_600": aggregate["severe_loss_per_600_mean"],
    }
    aggregate["locked_strong_gate_pass"] = (
        gate_pass(strong_mean, C7B_STRONG_GATE)
        and aggregate["max_seed_severe_loss_per_600"] <= MAX_SEED_SEVERE
        and aggregate["all_seed_strong_gate_pass"]
    )
    decision = (
        "LOCKED_ONE_SHOT_STRONG_PASS_REVIEW_DISTILLATION"
        if aggregate["locked_strong_gate_pass"]
        else "LOCKED_ONE_SHOT_FAIL_NO_TUNING"
    )

    write_csv(args.out_dir / "v21_locked_one_shot_fold_proxy.csv", fold_rows, sorted({k for r in fold_rows for k in r}))
    write_csv(args.out_dir / "v21_locked_one_shot_per_image.csv", actual, sorted({k for r in actual for k in r}))
    write_csv(args.out_dir / "v21_locked_one_shot_seed_summary.csv", seed_rows, sorted({k for r in seed_rows for k in r}))

    payload = {
        "route": "Haze4K-v2.1 SEG-Mix",
        "phase": "Locked One-Shot Sealed C10 Policy Replay",
        "decision": decision,
        "locked_test_touched": True,
        "locked_one_shot": True,
        "no_tuning_from_locked": True,
        "authorized_by": AUTHORIZED_DECISION,
        "c10_summary": str(args.c10_summary),
        "c10_source_decision": c10_payload.get("decision"),
        "fixed_profile": args.fixed_profile,
        "seeds": SEEDS,
        "strong_gate": C7B_STRONG_GATE,
        "max_seed_severe_gate": MAX_SEED_SEVERE,
        "data_split": args.data_split,
        "depth_split": args.depth_split,
        "a0_checkpoint": str(args.a0_checkpoint),
        "a0_sha256": sha256_file(Path(args.a0_checkpoint)),
        "official_checkpoint": str(args.official_checkpoint),
        "official_sha256": sha256_file(Path(args.official_checkpoint)),
        "aggregate": aggregate,
        "seed_rows": seed_rows,
    }
    (args.out_dir / "v21_locked_one_shot_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Haze4K v2.1 Locked One-Shot Sealed C10 Policy Replay",
        "",
        f"Decision: `{decision}`",
        "",
        f"Authorized by: `{AUTHORIZED_DECISION}`",
        f"Fixed profile: `{args.fixed_profile}`",
        "",
        "Locked output is evidence only and must not tune thresholds, profiles, features, action sets, checkpoints, or distillation targets.",
        "",
        "## Aggregate",
        "",
    ]
    for key, value in aggregate.items():
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Seed Summary", ""]
    for row in seed_rows:
        lines.append(
            f"- seed `{row['seed']}`: mean `{row['mean_dPSNR']}`, hard `{row['hard_bottom25_dPSNR']}`, "
            f"easy `{row['easy_top25_dPSNR']}`, positive `{row['positive_ratio']}`, "
            f"severe `{row['severe_loss_per_600']}`, strong `{row['strong_gate_pass']}`"
        )
    lines += [
        "",
        "## Closeout Rule",
        "",
        "If this one-shot fails, do not tune from locked output. If it passes, sync evidence first, then review whether distillation may begin from train-derived teacher definitions only.",
    ]
    (args.out_dir / "v21_locked_one_shot_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V21_LOCKED_ONE_SHOT_OK decision={decision} out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
