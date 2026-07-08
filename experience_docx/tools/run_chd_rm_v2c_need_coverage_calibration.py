import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

V2B_TOOL = REPO_ROOT / "experience_docx" / "tools" / "run_chd_rm_v2b_need_calibration_repair.py"
spec = importlib.util.spec_from_file_location("chdrm_v2b_tool", V2B_TOOL)
v2b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2b)

STRONG_PRED_THRESHOLD = 0.66
MIN_STRONG_COVERAGE = 0.01
MAX_STRONG_COVERAGE = 0.90


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_heads(artifact_dir, variants, device):
    heads = {}
    for variant in variants:
        ckpt_path = Path(artifact_dir) / f"{variant}_head.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        out_channels = 4 if variant == "d6c_ordinal_quantile" else 1
        head = v2b.ScalarNeedHead(out_channels).to(device)
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        for param in head.parameters():
            param.requires_grad_(False)
        heads[variant] = head
    return heads


def target_mode_for_variant(variant):
    return "log" if variant == "d6b_log" else "quantile"


def sample_np(x, sample_size):
    return v2b.sample_values(x, sample_size).astype(np.float32, copy=False)


def collect_split(model, heads, names, data_dir, device, target_info, density_stats, args, split_name, limit=0):
    dataset = v2b.Haze4KPairDataset(names, data_dir, max_items=limit, seed=args.seed)
    train_values = {
        variant: {"target": [], "pred": []}
        for variant in heads
    }
    val_records = {variant: [] for variant in heads}
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = v2b.v2.pad32(hazy)
            a0, res1 = v2b.v2.convir_a0_and_res1(model, padded)
            a0 = a0[:, :, :h, :w]
            res1 = res1[:, :, :h, :w]
            raw_need = v2b.v2.raw_need(a0, gt, args.blur_kernel)
            target_maps = {
                "quantile": v2b.make_target(raw_need, target_info, "quantile"),
                "log": v2b.make_target(raw_need, target_info, "log"),
            }
            density_target = v2b.v2.normalize(
                v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                density_stats["density"]["raw_p1"],
                density_stats["density"]["raw_p99"],
            )
            density_sample = sample_np(density_target, args.metric_sample_size)
            for variant, head in heads.items():
                target_mode = target_mode_for_variant(variant)
                target_sample = sample_np(target_maps[target_mode], args.metric_sample_size)
                pred = v2b.predict(head, res1, variant)
                pred_sample = sample_np(pred, args.metric_sample_size)
                if split_name == "train_inner":
                    train_values[variant]["target"].append(target_sample)
                    train_values[variant]["pred"].append(pred_sample)
                else:
                    val_records[variant].append(
                        {
                            "name": name,
                            "target": target_sample,
                            "pred": pred_sample,
                            "density": density_sample,
                        }
                    )
            if (idx + 1) % args.progress_every == 0:
                print(f"collect {split_name} {idx + 1}/{len(dataset)}", flush=True)
    if split_name == "train_inner":
        return {
            variant: {
                "target": np.concatenate(parts["target"]).astype(np.float32, copy=False),
                "pred": np.concatenate(parts["pred"]).astype(np.float32, copy=False),
            }
            for variant, parts in train_values.items()
        }
    return val_records


def fit_calibrator(method, train_pred, train_target):
    pred = train_pred.astype(np.float64, copy=False)
    target = train_target.astype(np.float64, copy=False)
    eps = 1e-5
    if method == "identity":
        return {"method": method}
    if method in {"affine_p01_p99", "affine_p10_p90"}:
        lo_pct, hi_pct = (1, 99) if method == "affine_p01_p99" else (10, 90)
        return {
            "method": method,
            "pred_lo": float(np.percentile(pred, lo_pct)),
            "pred_hi": float(np.percentile(pred, hi_pct)),
            "target_lo": float(np.percentile(target, lo_pct)),
            "target_hi": float(np.percentile(target, hi_pct)),
        }
    if method == "mean_std":
        return {
            "method": method,
            "pred_mean": float(np.mean(pred)),
            "pred_std": float(np.std(pred) + eps),
            "target_mean": float(np.mean(target)),
            "target_std": float(np.std(target) + eps),
        }
    if method == "logit_mean_std":
        pred_l = np.log(np.clip(pred, eps, 1.0 - eps) / (1.0 - np.clip(pred, eps, 1.0 - eps)))
        target_l = np.log(np.clip(target, eps, 1.0 - eps) / (1.0 - np.clip(target, eps, 1.0 - eps)))
        return {
            "method": method,
            "pred_mean": float(np.mean(pred_l)),
            "pred_std": float(np.std(pred_l) + eps),
            "target_mean": float(np.mean(target_l)),
            "target_std": float(np.std(target_l) + eps),
        }
    if method == "quantile_map_1001":
        grid = np.linspace(0.0, 1.0, 1001)
        pred_q = np.quantile(pred, grid)
        target_q = np.quantile(target, grid)
        unique_pred, unique_idx = np.unique(pred_q, return_index=True)
        unique_target = target_q[unique_idx]
        return {
            "method": method,
            "pred_q": unique_pred.astype(float).tolist(),
            "target_q": unique_target.astype(float).tolist(),
        }
    raise ValueError(method)


def apply_calibrator(calibrator, pred):
    method = calibrator["method"]
    x = pred.astype(np.float64, copy=False)
    eps = 1e-5
    if method == "identity":
        y = x
    elif method in {"affine_p01_p99", "affine_p10_p90"}:
        scale = (calibrator["target_hi"] - calibrator["target_lo"]) / max(
            calibrator["pred_hi"] - calibrator["pred_lo"], eps
        )
        y = (x - calibrator["pred_lo"]) * scale + calibrator["target_lo"]
    elif method == "mean_std":
        y = (x - calibrator["pred_mean"]) / calibrator["pred_std"]
        y = y * calibrator["target_std"] + calibrator["target_mean"]
    elif method == "logit_mean_std":
        clipped = np.clip(x, eps, 1.0 - eps)
        z = np.log(clipped / (1.0 - clipped))
        z = (z - calibrator["pred_mean"]) / calibrator["pred_std"]
        z = z * calibrator["target_std"] + calibrator["target_mean"]
        y = 1.0 / (1.0 + np.exp(-z))
    elif method == "quantile_map_1001":
        y = np.interp(x, np.asarray(calibrator["pred_q"]), np.asarray(calibrator["target_q"]))
    else:
        raise ValueError(method)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def summarize_variant_method(variant, method, target_mode, target_all, pred_all, val_records, density_q33, q33, q66):
    low_or_high = (target_all <= q33) | (target_all >= q66)
    high_label = target_all >= q66
    summary = {
        "variant": variant,
        "calibration_method": method,
        "target_mode": target_mode,
        "eval_pixels": int(target_all.size),
        "need_pearson": v2b.pearson(pred_all, target_all),
        "need_spearman": v2b.spearman(pred_all, target_all),
        "need_auroc_high_vs_low": v2b.auroc(high_label[low_or_high], pred_all[low_or_high]),
        "need_auprc_high": v2b.auprc(high_label, pred_all),
        "need_pred_high_coverage": float(np.mean(pred_all >= STRONG_PRED_THRESHOLD)),
        "need_target_high_coverage": float(np.mean(target_all >= q66)),
    }
    bins, mono, valid = v2b.bin_rows(target_all, pred_all, [0.20, q33, q66, 0.80], variant)
    summary["need_monotonic_pairs"] = mono
    summary["need_monotonic_valid_pairs"] = valid
    false_rates = []
    per_rows = []
    for record in val_records:
        cal_pred = record["calibrated_pred"]
        target = record["target"]
        low_context = (record["density"] <= density_q33) & (target <= q33)
        false_strong = cal_pred >= STRONG_PRED_THRESHOLD
        false_rate = float(np.mean(false_strong[low_context])) if int(low_context.sum()) else math.nan
        if not math.isnan(false_rate):
            false_rates.append(false_rate)
        per_rows.append(
            {
                "variant": variant,
                "calibration_method": method,
                "name": record["name"],
                "need_pearson": v2b.pearson(cal_pred, target),
                "need_mae": float(np.mean(np.abs(cal_pred - target))),
                "need_target_mean": float(np.mean(target)),
                "need_pred_mean": float(np.mean(cal_pred)),
                "need_target_max": float(np.max(target)),
                "need_pred_max": float(np.max(cal_pred)),
                "low_context_false_strong_rate": false_rate,
            }
        )
    summary["low_context_false_strong_rate"] = statistics.mean(false_rates) if false_rates else math.nan
    for row in bins:
        row["calibration_method"] = method
    return summary, bins, per_rows


def gate_pass(summary):
    return bool(
        summary.get("need_pearson", math.nan) >= 0.20
        and summary.get("need_spearman", math.nan) >= 0.25
        and summary.get("need_auroc_high_vs_low", math.nan) >= 0.65
        and summary.get("need_monotonic_pairs", 0) >= 4
        and MIN_STRONG_COVERAGE
        <= summary.get("need_pred_high_coverage", 0.0)
        <= MAX_STRONG_COVERAGE
        and summary.get("low_context_false_strong_rate", 1.0) <= 0.10
    )


def write_decision(output_dir, summaries):
    candidate_pass = [
        s for s in summaries
        if s["variant"] != "d6s_shuffled_quantile" and s["calibration_method"] != "identity" and gate_pass(s)
    ]
    control_pass = any(
        s["variant"] == "d6s_shuffled_quantile" and s["calibration_method"] != "identity" and gate_pass(s)
        for s in summaries
    )
    if candidate_pass and not control_pass:
        decision = "COMPLETED_V2C_SCALE_CALIBRATION_PASS_PAUSE_BEFORE_LEARNABLE_HEAD"
        next_step = "Do not run D2/RARM yet; implement a train-inner-fitted learnable or frozen monotonic calibration confirmation."
    elif control_pass:
        decision = "PAUSE_V2C_CONTROL_INVALID"
        next_step = "Do not proceed; shuffled control passed a calibrated gate, so the metric contract or calibration is invalid."
    else:
        decision = "PAUSE_V2C_SCALE_CALIBRATION_NOT_ENOUGH"
        next_step = "Do not run D2/RARM; repair spatial ranking or head capacity before modulation."
    payload = {
        "decision": decision,
        "passing": [
            {"variant": s["variant"], "calibration_method": s["calibration_method"]}
            for s in candidate_pass
        ],
        "control_pass": control_pass,
        "next_step": next_step,
    }
    lines = [
        "# CHD-RM v2c Decision Record",
        "",
        f"Decision: `{decision}`",
        "",
        f"Next step: {next_step}",
        "",
        "Forbidden in this stage: D2, RARM connection, v3 expansion, locked Haze4K test.",
        "Locked Haze4K test usage: none.",
        "",
    ]
    (output_dir / "decision_record.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--v2_thresholds", required=True)
    ap.add_argument("--v2b_thresholds", required=True)
    ap.add_argument("--v2b_artifact_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--blur_kernel", type=int, default=9)
    ap.add_argument("--metric_sample_size", type=int, default=64)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["d6a_quantile", "d6b_log", "d6c_ordinal_quantile", "d6s_shuffled_quantile"],
    )
    ap.add_argument(
        "--methods",
        nargs="*",
        default=["identity", "affine_p01_p99", "affine_p10_p90", "mean_std", "logit_mean_std", "quantile_map_1001"],
    )
    args = ap.parse_args()

    v2b.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_names = v2b.load_split_names(args.split_json, "train_inner")
    val_names = v2b.load_split_names(args.split_json, "val_inner")
    if len(train_names) != 2400 or len(val_names) != 600:
        raise ValueError(f"Expected 2400/600 split, got {len(train_names)}/{len(val_names)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = v2b.load_model(args.checkpoint, device)
    heads = load_heads(args.v2b_artifact_dir, args.variants, device)
    density_stats = json.loads(Path(args.v2_thresholds).read_text(encoding="utf-8"))
    target_info = json.loads(Path(args.v2b_thresholds).read_text(encoding="utf-8"))

    train_values = collect_split(
        model, heads, train_names, args.data_dir, device, target_info, density_stats, args, "train_inner", args.train_limit
    )
    val_records = collect_split(
        model, heads, val_names, args.data_dir, device, target_info, density_stats, args, "val_inner", args.val_limit
    )

    summaries = []
    bin_rows = []
    per_rows = []
    calibrator_rows = []
    calibrators = {}
    for variant in args.variants:
        target_mode = target_mode_for_variant(variant)
        q33 = float(target_info[target_mode]["q33"])
        q66 = float(target_info[target_mode]["q66"])
        density_q33 = float(density_stats["density"]["q33"])
        train_target = train_values[variant]["target"]
        train_pred = train_values[variant]["pred"]
        for method in args.methods:
            calibrator = fit_calibrator(method, train_pred, train_target)
            calibrators[f"{variant}:{method}"] = calibrator
            calibrator_rows.append(
                {
                    "variant": variant,
                    "calibration_method": method,
                    "target_mode": target_mode,
                    "train_pred_mean": float(np.mean(train_pred)),
                    "train_target_mean": float(np.mean(train_target)),
                    "train_pred_max": float(np.max(train_pred)),
                    "train_target_max": float(np.max(train_target)),
                    "calibrator": json.dumps(calibrator)[:1000],
                }
            )
            method_records = []
            pred_all_parts = []
            target_all_parts = []
            for record in val_records[variant]:
                cal_pred = apply_calibrator(calibrator, record["pred"])
                method_record = dict(record)
                method_record["calibrated_pred"] = cal_pred
                method_records.append(method_record)
                pred_all_parts.append(cal_pred)
                target_all_parts.append(record["target"])
            pred_all = np.concatenate(pred_all_parts)
            target_all = np.concatenate(target_all_parts)
            summary, bins, per = summarize_variant_method(
                variant, method, target_mode, target_all, pred_all, method_records, density_q33, q33, q66
            )
            summaries.append(summary)
            bin_rows.extend(bins)
            per_rows.extend(per)
            print(json.dumps(summary), flush=True)

    write_csv(output_dir / "need_coverage_calibration_summary.csv", summaries)
    write_csv(output_dir / "need_coverage_calibration_bins.csv", bin_rows)
    write_csv(output_dir / "need_coverage_calibration_per_image_metrics.csv", per_rows)
    write_csv(output_dir / "need_coverage_calibrator_params.csv", calibrator_rows)
    (output_dir / "need_coverage_calibrators.json").write_text(json.dumps(calibrators, indent=2), encoding="utf-8")
    decision = write_decision(output_dir, summaries)
    run_summary = {"decision": decision, "summaries": summaries, "args": vars(args)}
    (output_dir / "v2c_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    lines = [
        "# CHD-RM v2c Need Coverage Calibration Summary",
        "",
        f"Decision: `{decision['decision']}`",
        "",
        "| Variant | Method | Pearson | Spearman | AUROC | Coverage | False-strong | Monotonic |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {s['variant']} | {s['calibration_method']} | {s['need_pearson']:.4f} | "
            f"{s['need_spearman']:.4f} | {s['need_auroc_high_vs_low']:.4f} | "
            f"{s['need_pred_high_coverage']:.4f} | {s['low_context_false_strong_rate']:.4f} | "
            f"{s['need_monotonic_pairs']}/{s['need_monotonic_valid_pairs']} |"
        )
    lines += [
        "",
        "Calibration is fitted on train_inner predictions only and evaluated on val_inner.",
        "Forbidden in this stage: D2, RARM connection, v3 expansion, locked Haze4K test.",
        "Locked Haze4K test usage: none.",
        "",
    ]
    (output_dir / "v2c_result_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# CHD-RM v2c Need Coverage Calibration Evidence\n\n"
        f"Status: `{decision['decision']}`\n\n"
        "This stage tests whether v2b's nonzero R_need ranking signal is blocked mainly by output-scale compression. "
        "It fits monotone/affine calibrators on train_inner only and evaluates on val_inner with shuffled control retained.\n\n"
        "Start with `v2c_result_summary.md`, `decision_record.md`, and `v2c_run_summary.json`.\n\n"
        "Locked Haze4K test usage: none.\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "output_dir": str(output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
