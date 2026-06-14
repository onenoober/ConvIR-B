#!/usr/bin/env python3
"""C2d alpha-shrink StrongExpert router screen.

Renders A0 and FullUDP in memory, evaluates A0 + alpha*(FullUDP-A0) for a small
alpha grid, and runs the same leakage-safe one/two-rule OOF router screen with
alpha as a train-selected action. No raw images/tensors are written.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import audit_haze4k_v20_c2_outputdiff_router as c2
from audit_haze4k_v20_c2b_multirule_router import (  # noqa: E402
    ABSTENTION_GATE,
    POLICY_FEATURES,
    QUANTILES,
    STRICT_GATE,
    Table,
    abstention_gate_pass,
    fnum,
    fold_id,
    policy_grid,
    score,
    strict_gate_pass,
    summarize_mask,
    write_csv,
)


ALPHAS = [0.25, 0.50, 0.75, 1.00]


def alpha_key(alpha: float) -> str:
    return f"a{str(alpha).replace('.', 'p')}"


def alpha_rows(rows: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    key = alpha_key(alpha)
    out: list[dict[str, Any]] = []
    for row in rows:
        clone = dict(row)
        clone["dPSNR"] = row[f"dPSNR_{key}"]
        clone["dSSIM"] = row[f"dSSIM_{key}"]
        clone["alpha"] = alpha
        out.append(clone)
    return out


def render_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    convir_its_dir = Path(args.convir_its_dir)
    _test_dataloader, build_convir_net = c2.load_convir_builders(convir_its_dir)
    build_udpnet = c2.load_udpnet_builder(Path(args.udp_repo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    a0_model = c2.load_a0_model(build_convir_net, Path(args.a0_checkpoint), device)
    udpnet_model, ckpt_meta = c2.load_udpnet_model(build_udpnet, Path(args.official_checkpoint), device)

    rows: list[dict[str, Any]] = []
    factor = int(args.pad_factor)
    start_time = time.time()
    for split in args.splits:
        names = c2.load_split_names(Path(args.split_json), split)
        depth_split = "train" if args.split_json else args.depth_split
        with torch.no_grad():
            for idx, image_name in enumerate(names):
                input_img, label_img, depth = c2.load_sample(
                    Path(args.data_dir),
                    Path(args.depth_cache_dir),
                    image_name,
                    depth_split,
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
                udp_input = torch.cat([rgb_padded, depth_padded], dim=1)
                a0_pred = c2.infer_one(a0_model, rgb_padded, h, w)
                udp_pred = c2.infer_one(udpnet_model, udp_input, h, w)
                a0_psnr, a0_ssim = c2.metric_pair(a0_pred, label_img, (h_pad, w_pad))
                rec: dict[str, Any] = {
                    "name": image_name,
                    "split": split,
                    "A0_PSNR": a0_psnr,
                    "A0_SSIM": a0_ssim,
                }
                rec.update(c2.feature_dict(input_img, depth, a0_pred, udp_pred))
                for alpha in ALPHAS:
                    key = alpha_key(alpha)
                    blend = torch.clamp(a0_pred + alpha * (udp_pred - a0_pred), 0, 1)
                    psnr_val, ssim_val = c2.metric_pair(blend, label_img, (h_pad, w_pad))
                    rec[f"blend_PSNR_{key}"] = psnr_val
                    rec[f"blend_SSIM_{key}"] = ssim_val
                    rec[f"dPSNR_{key}"] = psnr_val - a0_psnr
                    rec[f"dSSIM_{key}"] = ssim_val - a0_ssim
                rows.append(rec)
                if (idx + 1) % args.print_freq == 0:
                    endpoint_key = alpha_key(1.0)
                    mean_delta = statistics.mean(float(r[f"dPSNR_{endpoint_key}"]) for r in rows)
                    print(f"{split} {idx + 1}/{len(names)} rows={len(rows)} endpoint_mean_delta={mean_delta:.4f}", flush=True)
                if args.max_images and idx + 1 >= args.max_images:
                    break
    meta = {
        "elapsed_sec": time.time() - start_time,
        "device": str(device),
        "a0_checkpoint": str(args.a0_checkpoint),
        "a0_sha256": c2.sha256_file(Path(args.a0_checkpoint)),
        "official_checkpoint": str(args.official_checkpoint),
        "official_sha256": c2.sha256_file(Path(args.official_checkpoint)),
        "official_checkpoint_meta": ckpt_meta,
        "udp_repo": str(args.udp_repo),
        "depth_normalization": "per_image_minmax_to_udpnet_depth2l_contract",
        "alphas": ALPHAS,
        "splits": args.splits,
        "split_json": args.split_json,
        "locked_test_touched": False,
    }
    return rows, meta


def choose_alpha_policy(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        table = Table(alpha_rows(train_rows, alpha), POLICY_FEATURES)
        grid = policy_grid(table, POLICY_FEATURES, QUANTILES, top_k=200)
        for rec in grid:
            clone = dict(rec)
            clone["alpha"] = alpha
            clone["policy_id"] = f"alpha_{alpha_key(alpha)}__{rec['policy_id']}"
            candidates.append(clone)
    candidates.sort(key=lambda row: (bool(row["strict_gate_pass"]), bool(row["abstention_gate_pass"]), fnum(row["score"])), reverse=True)
    return candidates[0]


def strip_alpha_policy(policy_id: str) -> tuple[float, str]:
    prefix, rest = policy_id.split("__", 1)
    key = prefix.replace("alpha_", "")
    alpha = float(key[1:].replace("p", "."))
    return alpha, rest


def main() -> int:
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
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, meta = render_rows(args)
    fields = ["name", "split", "A0_PSNR", "A0_SSIM"] + POLICY_FEATURES
    for alpha in ALPHAS:
        key = alpha_key(alpha)
        fields.extend([f"blend_PSNR_{key}", f"blend_SSIM_{key}", f"dPSNR_{key}", f"dSSIM_{key}"])
    write_csv(out_dir / "v20_c2d_alpha_feature_rows.csv", rows, fields)

    in_sample_candidates: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        table = Table(alpha_rows(rows, alpha), POLICY_FEATURES)
        for rec in policy_grid(table, POLICY_FEATURES, QUANTILES, top_k=100):
            clone = dict(rec)
            clone["alpha"] = alpha
            clone["policy_id"] = f"alpha_{alpha_key(alpha)}__{rec['policy_id']}"
            in_sample_candidates.append(clone)
    in_sample_candidates.sort(key=lambda row: (bool(row["strict_gate_pass"]), bool(row["abstention_gate_pass"]), fnum(row["score"])), reverse=True)
    policy_fields = [
        "policy_id",
        "alpha",
        "complexity",
        "count",
        "selected_count",
        "coverage",
        "mean_dPSNR",
        "hard_bottom25_dPSNR",
        "easy_top25_dPSNR",
        "dSSIM",
        "positive_ratio",
        "nonnegative_ratio",
        "severe_loss_count",
        "severe_loss_per_600",
        "strong_loss_count",
        "strong_loss_per_600",
        "selected_precision",
        "selected_nonnegative_ratio",
        "selected_severe_count",
        "strict_gate_pass",
        "abstention_gate_pass",
        "score",
    ]
    write_csv(out_dir / "v20_c2d_alpha_policy_grid.csv", in_sample_candidates[:800], policy_fields)

    fold_rows: list[dict[str, Any]] = []
    selected_oof: dict[str, float] = {}
    for fold in range(5):
        train_rows = [r for r in rows if fold_id(str(r["name"])) != fold]
        heldout_rows = [r for r in rows if fold_id(str(r["name"])) == fold]
        chosen = choose_alpha_policy(train_rows)
        alpha, inner_policy = strip_alpha_policy(str(chosen["policy_id"]))
        heldout_table = Table(alpha_rows(heldout_rows, alpha), POLICY_FEATURES)
        heldout_mask = heldout_table.mask_for_policy(inner_policy)
        eval_rec: dict[str, Any] = {
            "fold": fold,
            "train_policy_id": chosen["policy_id"],
            "train_strict_gate_pass": chosen["strict_gate_pass"],
            "train_abstention_gate_pass": chosen["abstention_gate_pass"],
        }
        eval_rec.update(summarize_mask(heldout_table, heldout_mask))
        eval_rec["strict_gate_pass"] = strict_gate_pass(eval_rec)
        eval_rec["abstention_gate_pass"] = abstention_gate_pass(eval_rec)
        eval_rec["score"] = score(eval_rec)
        fold_rows.append(eval_rec)
        for row, selected in zip(heldout_rows, heldout_mask, strict=False):
            if bool(selected):
                selected_oof[str(row["name"])] = alpha
    write_csv(out_dir / "v20_c2d_alpha_oof_fold_metrics.csv", fold_rows, ["fold", "train_policy_id", "train_strict_gate_pass", "train_abstention_gate_pass"] + policy_fields[3:])

    # OOF uses per-sample fold-selected alpha, so materialize rows with the chosen alpha-specific deltas.
    oof_rows: list[dict[str, Any]] = []
    for row in rows:
        alpha = selected_oof.get(str(row["name"]))
        if alpha is None:
            clone = dict(row)
            clone["dPSNR"] = 0.0
            clone["dSSIM"] = 0.0
            clone["selected_oof"] = False
            clone["alpha_oof"] = 0.0
        else:
            key = alpha_key(alpha)
            clone = dict(row)
            clone["dPSNR"] = row[f"dPSNR_{key}"]
            clone["dSSIM"] = row[f"dSSIM_{key}"]
            clone["selected_oof"] = True
            clone["alpha_oof"] = alpha
        oof_rows.append(clone)
    oof_table = Table(oof_rows, POLICY_FEATURES)
    oof_summary = summarize_mask(oof_table, [bool(row["selected_oof"]) for row in oof_rows])
    oof_summary["strict_gate_pass"] = strict_gate_pass(oof_summary)
    oof_summary["abstention_gate_pass"] = abstention_gate_pass(oof_summary)
    oof_summary["score"] = score(oof_summary)

    if oof_summary["strict_gate_pass"]:
        decision = "C2D_ALPHA_STRICT_SCREEN_PASS_START_C3_SHIFTED"
    elif oof_summary["abstention_gate_pass"]:
        decision = "C2D_ALPHA_ABSTENTION_SCREEN_PASS_START_C3_SHIFTED"
    elif any(row["abstention_gate_pass"] for row in in_sample_candidates):
        decision = "C2D_ALPHA_IN_SAMPLE_ONLY_FAIL_OOF"
    else:
        decision = "C2D_ALPHA_ROUTER_SCREEN_FAIL_REASSESS_FEATURES_OR_EXPERT"

    summary = {
        "route": "Haze4K-v2.0 StrongExpert-GainMix",
        "phase": "C2d Alpha-Blend OutputDiff Router Screen",
        "locked_test_touched": False,
        "meta": meta,
        "rows": len(rows),
        "policy_features": POLICY_FEATURES,
        "strict_gate": STRICT_GATE,
        "abstention_gate": ABSTENTION_GATE,
        "best_policy": in_sample_candidates[0] if in_sample_candidates else None,
        "best_abstention_policy": next((row for row in in_sample_candidates if row["abstention_gate_pass"]), None),
        "fold_rows": fold_rows,
        "oof_summary": oof_summary,
        "decision": decision,
    }
    (out_dir / "v20_c2d_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Haze4K v2.0 C2d Alpha-Blend OutputDiff Router Screen",
        "",
        f"Decision: `{decision}`",
        "",
        "C2d evaluates fixed alpha shrink for selected FullUDP residuals and chooses alpha+policy inside each train fold.",
        "No raw images/tensors were written, and locked test data was not touched.",
        "",
        "## Best In-Sample Policy",
        "",
    ]
    for key in policy_fields:
        if in_sample_candidates and key in in_sample_candidates[0]:
            lines.append(f"- `{key}`: `{in_sample_candidates[0][key]}`")
    lines.extend(["", "## OOF Replay", ""])
    for key, value in oof_summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Fold Policies", ""])
    for row in fold_rows:
        lines.append(f"- fold `{row['fold']}`: `{row['train_policy_id']}`, pass `{row['abstention_gate_pass']}`")
    (out_dir / "v20_c2d_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V20_C2D_ALPHA_ROUTER_OK decision={decision} rows={len(rows)} out={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
