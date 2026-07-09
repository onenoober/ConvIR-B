import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
for path in (str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

V2D_TOOL = REPO_ROOT / "experience_docx" / "tools" / "run_chd_rm_v2d_need_spatial_hard_negative.py"
spec = importlib.util.spec_from_file_location("chdrm_v2d_tool", V2D_TOOL)
v2d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2d)


def precompute_base_metrics(pack, q33, q66, density_q33):
    target = pack["target"]
    pred = pack["pred"]
    density = pack["density"]
    high = target >= q66
    low_or_high = (target <= q33) | high
    hard_negative = v2d.hard_negative_mask(target, density, q33, density_q33)
    positives_or_hn = high | hard_negative
    hn_scores = pred[hard_negative]
    if hn_scores.size:
        k = max(1, int(math.ceil(hn_scores.size * 0.05)))
        hn_top5 = float(np.mean(np.sort(hn_scores)[-k:]))
    else:
        hn_top5 = math.nan
    return {
        "AUPRC_high": v2d.v2b.auprc(high, pred),
        "AUROC_high_vs_low": v2d.v2b.auroc(high[low_or_high], pred[low_or_high]),
        "AUROC_high_vs_hard_negative": (
            v2d.v2b.auroc(high[positives_or_hn], pred[positives_or_hn])
            if int(positives_or_hn.sum())
            else math.nan
        ),
        "hard_negative_top5_mean": hn_top5,
    }


def nearest_row(rows, threshold):
    return min(rows, key=lambda row: abs(row["threshold"] - threshold))


def audit_variant(model, head, train_names, val_names, device, target_info, density_stats, args, variant):
    print(f"AUDIT_{variant}_COLLECT_TRAIN", flush=True)
    train_pack = v2d.collect_sample_records(
        model, head, None, train_names, args.data_dir, device, target_info, density_stats, args, limit=args.train_limit
    )
    print(f"AUDIT_{variant}_COLLECT_VAL", flush=True)
    val_pack = v2d.collect_sample_records(
        model, head, None, val_names, args.data_dir, device, target_info, density_stats, args, limit=args.val_limit
    )
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    density_q33 = float(density_stats["density"]["q33"])
    train_pred = train_pack["pred"]
    grid = np.unique(
        np.concatenate(
            [
                np.quantile(train_pred, np.linspace(0.0, 1.0, args.threshold_grid)),
                np.linspace(0.05, 0.95, 19),
                np.asarray([v2d.STRONG_PRED_THRESHOLD], dtype=np.float32),
            ]
        )
    )
    base = precompute_base_metrics(val_pack, q33, q66, density_q33)
    rows = [
        v2d.threshold_metrics(
            t,
            val_pack["target"],
            val_pack["pred"],
            val_pack["density"],
            val_pack["records"],
            q33,
            q66,
            density_q33,
            base,
        )
        for t in sorted(grid)
    ]
    v2d.write_csv(args.output_dir / f"need_rank_safety_curve_{variant}.csv", rows)
    safe_global = [
        row
        for row in rows
        if v2d.MIN_COVERAGE <= row["pred_high_coverage"] <= v2d.MAX_COVERAGE
        and row["low_context_false_strong_global"] <= v2d.GATE["false_strong_global"]
    ]
    safe_p90 = [
        row
        for row in safe_global
        if row["low_context_false_strong_per_image_p90"] <= v2d.GATE["false_strong_p90"]
    ]
    selected_pool = safe_p90 or safe_global or rows
    selected = sorted(
        selected_pool,
        key=lambda row: (
            0 if row in safe_p90 else 1,
            0 if row in safe_global else 1,
            abs(row["pred_high_coverage"] - 0.30),
            row["low_context_false_strong_per_image_p90"],
        ),
    )[0]
    raw66 = nearest_row(rows, v2d.STRONG_PRED_THRESHOLD)
    target = val_pack["target"]
    pred = val_pack["pred"]
    summary = {
        "variant": variant,
        "need_pearson": v2d.v2b.pearson(pred, target),
        "need_spearman": v2d.v2b.spearman(pred, target),
        "need_auroc_high_vs_low": base["AUROC_high_vs_low"],
        "need_auprc_high": base["AUPRC_high"],
        "safe_global_points": len(safe_global),
        "safe_p90_points": len(safe_p90),
        "selected_threshold": selected["threshold"],
        "selected_coverage": selected["pred_high_coverage"],
        "selected_recall": selected["target_high_recall"],
        "selected_precision": selected["target_high_precision"],
        "selected_false_global": selected["low_context_false_strong_global"],
        "selected_false_p90": selected["low_context_false_strong_per_image_p90"],
        "selected_false_p95": selected["low_context_false_strong_per_image_p95"],
        "selected_low_density_high_recall": selected["high_need_recall_in_low_density_regions"],
        "raw66_coverage": raw66["pred_high_coverage"],
        "raw66_false_global": raw66["low_context_false_strong_global"],
        "hard_negative_top5_mean": base["hard_negative_top5_mean"],
    }
    print(json.dumps(summary), flush=True)
    return summary


def write_summary(output_dir, summaries):
    v2d.write_csv(output_dir / "trained_head_safety_audit_summary.csv", summaries)
    lines = [
        "# v2d Trained-Head Threshold Safety Audit",
        "",
        "| Variant | Spearman | AUROC | AUPRC | safe_global | safe_p90 | selected_threshold | coverage | recall | false_global | false_p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {row['need_spearman']:.4f} | {row['need_auroc_high_vs_low']:.4f} | "
            f"{row['need_auprc_high']:.4f} | {row['safe_global_points']} | {row['safe_p90_points']} | "
            f"{row['selected_threshold']:.6f} | {row['selected_coverage']:.4f} | {row['selected_recall']:.4f} | "
            f"{row['selected_false_global']:.4f} | {row['selected_false_p90']:.4f} |"
        )
    lines += [
        "",
        "Thresholds are generated from train_inner predictions and evaluated on val_inner.",
        "Locked Haze4K test usage: none.",
        "",
    ]
    (output_dir / "trained_head_safety_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--v2_thresholds", required=True)
    ap.add_argument("--v2b_thresholds", required=True)
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--blur_kernel", type=int, default=9)
    ap.add_argument("--metric_sample_size", type=int, default=64)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--threshold_grid", type=int, default=121)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["d7a_hn_ordinal", "d7b_topk_hn_ordinal", "d7s_shuffled_topk"],
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_names, val_names, device, model, density_stats, target_info = v2d.load_runtime(args)
    summaries = []
    for variant in args.variants:
        head_path = Path(args.artifact_dir) / f"{variant}_head.pt"
        head = v2d.load_need_head(head_path, device)
        head.eval()
        for param in head.parameters():
            param.requires_grad_(False)
        summaries.append(audit_variant(model, head, train_names, val_names, device, target_info, density_stats, args, variant))
    write_summary(args.output_dir, summaries)
    print(json.dumps({"output_dir": str(args.output_dir), "summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
