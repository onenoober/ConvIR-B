#!/usr/bin/env python3
"""C7c train-derived severe-risk tightening for local alpha.

C7c reuses C7b patch feature/SSE text rows, selects stricter fold policies under
several predeclared risk profiles, and re-renders held-out images once to obtain
true PSNR/SSIM for every profile. No locked Haze4K test data is touched.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import audit_haze4k_v20_c2_outputdiff_router as c2
from audit_haze4k_v20_c2b_multirule_router import fnum, fold_id, write_csv
from audit_haze4k_v21_c7b_local_alpha_prototype import (
    ALPHAS,
    C7B_SCREEN_GATE,
    C7B_STRONG_GATE,
    PatchTable,
    alpha_key,
    apply_policy,
    choose_policy,
    gate_pass,
    patch_rows_for_image_tensors,
    policy_grid,
    score,
    summarize_actual_rows,
)


PROFILES = [
    {"profile": "riskcap48_no075", "proxy_severe_cap": 48.0, "allow_alpha075": False, "hard_floor": 0.28, "positive_floor": 0.70},
    {"profile": "riskcap42_no075", "proxy_severe_cap": 42.0, "allow_alpha075": False, "hard_floor": 0.28, "positive_floor": 0.69},
    {"profile": "riskcap36_no075", "proxy_severe_cap": 36.0, "allow_alpha075": False, "hard_floor": 0.25, "positive_floor": 0.68},
    {"profile": "riskcap48_allow075", "proxy_severe_cap": 48.0, "allow_alpha075": True, "hard_floor": 0.30, "positive_floor": 0.70},
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def policy_allows_profile(policy_id: str, profile: dict[str, Any]) -> bool:
    if profile["allow_alpha075"]:
        return True
    return "alpha=0.75" not in policy_id and "high_alpha=0.75" not in policy_id and "low_alpha=0.75" not in policy_id


def profile_score(row: dict[str, Any], profile: dict[str, Any]) -> float:
    severe = fnum(row.get("severe_loss_per_600"))
    hard = fnum(row.get("hard_bottom25_dPSNR"))
    pos = fnum(row.get("positive_ratio"))
    penalty = 0.03 * max(0.0, severe - float(profile["proxy_severe_cap"]))
    penalty += 2.0 * max(0.0, float(profile["hard_floor"]) - hard)
    penalty += 1.5 * max(0.0, float(profile["positive_floor"]) - pos)
    return score(row) - penalty


def choose_profile_from_grid(grid: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    allowed = [row for row in grid if policy_allows_profile(str(row["policy_id"]), profile)]
    if not allowed:
        allowed = grid
    cap = float(profile["proxy_severe_cap"])
    hard_floor = float(profile["hard_floor"])
    positive_floor = float(profile["positive_floor"])
    feasible = [
        row
        for row in allowed
        if fnum(row.get("severe_loss_per_600")) <= cap
        and fnum(row.get("hard_bottom25_dPSNR")) >= hard_floor
        and fnum(row.get("positive_ratio")) >= positive_floor
        and fnum(row.get("easy_top25_dPSNR")) >= 0.0
    ]
    pool = feasible if feasible else allowed
    pool.sort(key=lambda row: (bool(row.get("strong_proxy_pass")), bool(row.get("screen_proxy_pass")), profile_score(row, profile)), reverse=True)
    chosen = dict(pool[0])
    chosen["profile_feasible_count"] = len(feasible)
    return chosen


def choose_profile_policy(train_table: PatchTable, profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    grid = policy_grid(train_table, args.top_k, args.low_pool_limit, args.high_pool_limit)
    if not grid:
        return choose_policy(train_table, args.top_k, args.low_pool_limit, args.high_pool_limit)
    return choose_profile_from_grid(grid, profile)


def build_profile_fold_policies(table: PatchTable, image_rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, dict[int, str]], list[dict[str, Any]]]:
    image_folds = np.array([fold_id(str(row["name"])) for row in image_rows], dtype=np.int64)
    profile_policies: dict[str, dict[int, str]] = {str(p["profile"]): {} for p in PROFILES}
    fold_rows: list[dict[str, Any]] = []
    from audit_haze4k_v21_c7b_local_alpha_prototype import summarize_patch_actions

    for fold in range(5):
        train_table = table.subset_images(image_folds != fold)
        heldout_table = table.subset_images(image_folds == fold)
        grid = policy_grid(train_table, args.top_k, args.low_pool_limit, args.high_pool_limit)
        print(f"c7c_policy_grid fold={fold} candidates={len(grid)}", flush=True)
        for profile in PROFILES:
            pname = str(profile["profile"])
            chosen = choose_profile_from_grid(grid, profile)
            profile_policies[pname][fold] = str(chosen["policy_id"])
            heldout_actions = apply_policy(heldout_table, str(chosen["policy_id"]))
            rec = {
                "profile": pname,
                "fold": fold,
                "train_policy_id": chosen["policy_id"],
                "profile_feasible_count": chosen.get("profile_feasible_count", ""),
                "train_score": chosen.get("score", ""),
                "train_proxy_severe": chosen.get("severe_loss_per_600", ""),
                "train_proxy_hard": chosen.get("hard_bottom25_dPSNR", ""),
                "train_proxy_positive": chosen.get("positive_ratio", ""),
                "heldout_count": len(heldout_table.image_rows),
            }
            rec.update(summarize_patch_actions(heldout_table, heldout_actions))
            fold_rows.append(rec)
    return profile_policies, fold_rows


def actions_for_patch_rows(policy_id: str, rows: list[dict[str, Any]]) -> np.ndarray:
    image_rows = [{"name": "one", "split": "one", "A0_PSNR": 0.0}]
    table_rows = []
    from audit_haze4k_v21_c7b_local_alpha_prototype import FEATURES

    for row in rows:
        rec = {"name": "one", "pixel_count": 1}
        rec.update({feat: row[feat] for feat in FEATURES})
        for alpha in ALPHAS:
            rec[f"sse_{alpha_key(alpha)}"] = 0.0
        table_rows.append(rec)
    table = PatchTable(table_rows, image_rows)
    return apply_policy(table, policy_id)


def eval_profiles_actual(args: argparse.Namespace, profile_policies: dict[str, dict[int, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _loader, build_convir_net = c2.load_convir_builders(Path(args.convir_its_dir))
    build_udpnet = c2.load_udpnet_builder(Path(args.udp_repo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0_model = c2.load_a0_model(build_convir_net, Path(args.a0_checkpoint), device)
    udp_model, _ = c2.load_udpnet_model(build_udpnet, Path(args.official_checkpoint), device)
    rows: list[dict[str, Any]] = []
    factor = int(args.pad_factor)
    for split in args.splits:
        names = c2.load_split_names(Path(args.split_json), split)
        depth_split = "train" if args.split_json else args.depth_split
        with torch.no_grad():
            for idx, image_name in enumerate(names):
                input_img, label_img, depth = c2.load_sample(Path(args.data_dir), Path(args.depth_cache_dir), image_name, depth_split)
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
                udp_input = torch.cat([rgb_padded, depth_padded], dim=1)
                a0_pred = c2.infer_one(a0_model, rgb_padded, h, w)
                udp_pred = c2.infer_one(udp_model, udp_input, h, w)
                a0_psnr, a0_ssim = c2.metric_pair(a0_pred, label_img, (h_pad, w_pad))
                fold = fold_id(image_name)
                p_rows = patch_rows_for_image_tensors(image_name, split, input_img, depth, a0_pred, udp_pred, int(args.patch_size))
                residual = udp_pred - a0_pred
                for pname, fold_map in profile_policies.items():
                    policy_id = fold_map[fold]
                    actions = actions_for_patch_rows(policy_id, p_rows)
                    out = a0_pred.clone()
                    alpha_counts = {alpha: 0 for alpha in ALPHAS}
                    patch_iter = list(iter_patches(h, w, int(args.patch_size)))
                    for (_patch_id, y, y2, x, x2), action in zip(patch_iter, actions, strict=False):
                        alpha = ALPHAS[int(action)]
                        alpha_counts[alpha] += 1
                        out[..., y:y2, x:x2] = torch.clamp(a0_pred[..., y:y2, x:x2] + alpha * residual[..., y:y2, x:x2], 0.0, 1.0)
                    psnr, ssim = c2.metric_pair(out, label_img, (h_pad, w_pad))
                    rec: dict[str, Any] = {
                        "profile": pname,
                        "name": image_name,
                        "split": split,
                        "fold": fold,
                        "policy_id": policy_id,
                        "A0_PSNR": a0_psnr,
                        "A0_SSIM": a0_ssim,
                        "local_alpha_PSNR": psnr,
                        "local_alpha_SSIM": ssim,
                        "dPSNR": psnr - a0_psnr,
                        "dSSIM": ssim - a0_ssim,
                        "patch_count": int(sum(alpha_counts.values())),
                    }
                    for alpha in ALPHAS:
                        rec[f"patch_action_count_{alpha_key(alpha)}"] = alpha_counts[alpha]
                        rec[f"patch_action_fraction_{alpha_key(alpha)}"] = alpha_counts[alpha] / max(1, rec["patch_count"])
                    rows.append(rec)
                if (idx + 1) % args.print_freq == 0:
                    print(f"c7c_actual {split} {idx + 1}/{len(names)} rows={len(rows)}", flush=True)
                if args.max_images and idx + 1 >= args.max_images:
                    break
    summary_rows: list[dict[str, Any]] = []
    for pname in sorted(profile_policies):
        subset = [row for row in rows if row["profile"] == pname]
        rec = {"profile": pname}
        rec.update(summarize_actual_rows(subset))
        summary_rows.append(rec)
    return rows, summary_rows


def iter_patches(h: int, w: int, patch_size: int):
    patch_id = 0
    for y in range(0, h, patch_size):
        y2 = min(h, y + patch_size)
        for x in range(0, w, patch_size):
            x2 = min(w, x + patch_size)
            yield patch_id, y, y2, x, x2
            patch_id += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_rows", type=Path, required=True)
    parser.add_argument("--image_rows", type=Path, required=True)
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
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--low_pool_limit", type=int, default=80)
    parser.add_argument("--high_pool_limit", type=int, default=120)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    patch_rows = read_csv(args.patch_rows)
    image_rows = read_csv(args.image_rows)
    table = PatchTable(patch_rows, image_rows)
    profile_policies, fold_rows = build_profile_fold_policies(table, image_rows, args)
    fold_fields = sorted({key for row in fold_rows for key in row.keys()})
    write_csv(args.out_dir / "v21_c7c_risk_profile_fold_policies.csv", fold_rows, fold_fields)

    per_image_rows, summary_rows = eval_profiles_actual(args, profile_policies)
    per_image_fields = sorted({key for row in per_image_rows for key in row.keys()})
    summary_fields = sorted({key for row in summary_rows for key in row.keys()})
    write_csv(args.out_dir / "v21_c7c_risk_profile_per_image.csv", per_image_rows, per_image_fields)
    write_csv(args.out_dir / "v21_c7c_risk_profile_summary.csv", summary_rows, summary_fields)

    strong_profiles = [row for row in summary_rows if row.get("strong_gate_pass")]
    screen_profiles = [row for row in summary_rows if row.get("screen_gate_pass")]
    best = max(summary_rows, key=lambda row: (bool(row.get("strong_gate_pass")), bool(row.get("screen_gate_pass")), fnum(row.get("score"))))
    if strong_profiles:
        decision = "C7C_RISK_TIGHTEN_STRONG_PASS_START_C9_SHIFTED_STRONG"
    elif screen_profiles:
        decision = "C7C_RISK_TIGHTEN_SCREEN_PASS_STRONG_NOT_YET"
    else:
        decision = "C7C_RISK_TIGHTEN_FAIL_START_C8_MULTIEXPERT"
    payload = {
        "route": "Haze4K-v2.1 SEG-Mix",
        "phase": "C7c Local-Alpha Severe-Risk Tightening",
        "locked_test_touched": False,
        "profiles": PROFILES,
        "profile_policies": profile_policies,
        "summary_rows": summary_rows,
        "best_profile": best,
        "decision": decision,
    }
    (args.out_dir / "v21_c7c_risk_tighten_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Haze4K v2.1 C7c Local-Alpha Severe-Risk Tightening",
        "",
        f"Decision: `{decision}`",
        "",
        "C7c reuses C7b patch feature/SSE rows, selects stricter train-fold policies, and re-renders held-out images once for all risk profiles. Locked test data was not touched.",
        "",
        "| Profile | Mean | Hard | Easy | dSSIM | Positive | Severe/600 | Screen | Strong |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['profile']}` | `{fnum(row['mean_dPSNR']):.6f}` | `{fnum(row['hard_bottom25_dPSNR']):.6f}` | "
            f"`{fnum(row['easy_top25_dPSNR']):.6f}` | `{fnum(row['dSSIM']):.8f}` | `{fnum(row['positive_ratio']):.6f}` | "
            f"`{fnum(row['severe_loss_per_600']):.1f}` | `{row['screen_gate_pass']}` | `{row['strong_gate_pass']}` |"
        )
    lines.extend(["", "## Interpretation", "", "C9 shifted-strong validation is authorized only if a risk profile passes the strong gate. Locked test remains blocked."])
    (args.out_dir / "v21_c7c_risk_tighten_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V21_C7C_RISK_TIGHTEN_OK decision={decision} best={best['profile']} out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
