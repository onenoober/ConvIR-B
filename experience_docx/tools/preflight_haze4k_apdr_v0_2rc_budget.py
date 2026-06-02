import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_ROOT))

import preflight_haze4k_apdr_v0_2r_selector as v02r


def sigmoid(value):
    value = max(min(float(value), 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-value))


def psnr(pred, label):
    mse = F.mse_loss(torch.clamp(pred, 0, 1), label).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def bce_from_probs(rows, candidate):
    eps = 1e-6
    losses = []
    for row in rows:
        target = row["hard_soft"]
        prob = min(max(apply_candidate(row["z_img"], candidate), eps), 1.0 - eps)
        losses.append(-(target * math.log(prob) + (1.0 - target) * math.log(1.0 - prob)))
    return statistics.mean(losses)


def percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute percentile from empty values.")
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def apply_candidate(z_img, candidate):
    kind = candidate["kind"]
    if kind in ("base_power", "shifted_power"):
        base = sigmoid((z_img - candidate["tau_base"]) / candidate["temperature_base"])
        if kind == "base_power":
            return base ** candidate["gamma"]
        shifted = max((base - candidate["b0"]) / max(1.0 - candidate["b0"], 1e-6), 0.0)
        return shifted ** candidate["gamma"]
    if kind == "platt_power":
        base = sigmoid((z_img - candidate["tau"]) / candidate["temperature"])
        return base ** candidate["gamma"]
    raise ValueError(f"Unknown candidate kind: {kind}")


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        rank = (pos + end - 1) / 2.0 + 1.0
        for idx in order[pos:end]:
            ranks[idx] = rank
        pos = end
    return ranks


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def spearman(xs, ys):
    return pearson(average_ranks(xs), average_ranks(ys))


def auc_score(pos_scores, neg_scores):
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    total = len(pos_scores) * len(neg_scores)
    for pos in pos_scores:
        for neg in neg_scores:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def mean_or_none(values):
    return statistics.mean(values) if values else None


def rows_from_train_scores(train_scores, budget_calibration):
    rows = []
    tau = budget_calibration["tau_train"]
    temperature = max(budget_calibration["temperature_train"], 1e-6)
    for row in train_scores:
        rows.append(
            {
                "name": row["name"],
                "split": "train",
                "a0_psnr": None,
                "z_img": row["z_img"],
                "b_base": sigmoid((row["z_img"] - tau) / temperature),
                "rmse0": row["rmse"],
                "hard_soft": row["hard_soft"],
                "hard_binary": row["hard_binary"],
                "spatial_bce": None,
                "zero_residual_max_abs_diff_vs_a0": None,
            }
        )
    return rows


def collect_test_rows(original, apdr, args, device, calibration, csv_path):
    loader = v02r.test_dataloader(args.data_dir, "Haze4K", batch_size=1, num_workers=0)
    rows = []
    v02r.set_selector_eval_mode(apdr)
    original.eval()
    max_abs_diff = 0.0
    with torch.no_grad(), csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "name",
            "split",
            "a0_psnr",
            "z_img",
            "b_base",
            "rmse0",
            "hard_soft",
            "hard_binary",
            "spatial_bce",
            "zero_residual_max_abs_diff_vs_a0",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        summary = calibration["summary"]
        for idx, (input_img, label_img, name) in enumerate(loader):
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            padded, h, w = v02r.pad_to_factor(input_img)
            a0 = v02r.crop_to(original(padded)[2], h, w)
            apdr_out = v02r.crop_to(apdr(padded)[2], h, w)
            item = v02r.full_item(apdr)
            anchor = v02r.crop_to(item["anchor"], h, w)
            error = anchor - label_img
            rmse = torch.sqrt(error.square().mean()).item()
            denom = max(summary["rmse_q90_train"] - summary["rmse_q50_train"], 1e-8)
            hard_soft = min(max((rmse - summary["rmse_q50_train"]) / denom, 0.0), 1.0)
            hard_binary = None
            if rmse >= summary["rmse_q75_train"]:
                hard_binary = 1.0
            elif rmse <= summary["rmse_q25_train"]:
                hard_binary = 0.0
            spatial_target = v02r.make_spatial_target(anchor, label_img, calibration)
            spatial_logits = v02r.crop_to(item["spatial_logits"], h, w)
            spatial_bce = F.binary_cross_entropy_with_logits(spatial_logits, spatial_target).item()
            diff = (apdr_out - a0).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            row = {
                "name": name[0],
                "split": "test",
                "a0_psnr": psnr(a0, label_img),
                "z_img": item["global_logits"].view(-1).item(),
                "b_base": item["global_budget_unit"].view(-1).item(),
                "rmse0": rmse,
                "hard_soft": hard_soft,
                "hard_binary": hard_binary,
                "spatial_bce": spatial_bce,
                "zero_residual_max_abs_diff_vs_a0": diff,
            }
            rows.append(row)
            writer.writerow(row)
            if (idx + 1) % args.progress_freq == 0:
                print(f"test_score {idx + 1}/{len(loader)}", flush=True)
    return rows, max_abs_diff


def write_train_rows(rows, path):
    fieldnames = [
        "name",
        "split",
        "a0_psnr",
        "z_img",
        "b_base",
        "rmse0",
        "hard_soft",
        "hard_binary",
        "spatial_bce",
        "zero_residual_max_abs_diff_vs_a0",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_candidates(train_rows, budget_calibration):
    tau_base = budget_calibration["tau_train"]
    temperature_base = max(budget_calibration["temperature_train"], 1e-6)
    candidates = []
    for gamma in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
        candidates.append(
            {
                "name": f"base_power_g{gamma:g}",
                "kind": "base_power",
                "tau_base": tau_base,
                "temperature_base": temperature_base,
                "gamma": gamma,
            }
        )
    for b0 in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25):
        for gamma in (1.0, 1.5, 2.0, 3.0, 4.0):
            candidates.append(
                {
                    "name": f"shifted_b{b0:g}_g{gamma:g}",
                    "kind": "shifted_power",
                    "tau_base": tau_base,
                    "temperature_base": temperature_base,
                    "b0": b0,
                    "gamma": gamma,
                }
            )
    z_values = [row["z_img"] for row in train_rows]
    tau_options = sorted(
        {
            percentile(z_values, pct)
            for pct in (45, 50, 55, 60, 65, 70, 75, 80, 85, 90)
        }
    )
    for tau in tau_options:
        for temperature in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50):
            for gamma in (1.0, 1.5, 2.0, 3.0, 4.0):
                candidates.append(
                    {
                        "name": f"platt_tau{tau:.4f}_t{temperature:g}_g{gamma:g}",
                        "kind": "platt_power",
                        "tau": tau,
                        "temperature": temperature,
                        "gamma": gamma,
                    }
                )
    return candidates


def summarize_candidate(rows, candidate, group_mode):
    budgets = [apply_candidate(row["z_img"], candidate) for row in rows]
    z_values = [row["z_img"] for row in rows]
    if group_mode == "binary":
        hard = [(row, b) for row, b in zip(rows, budgets) if row["hard_binary"] == 1.0]
        easy = [(row, b) for row, b in zip(rows, budgets) if row["hard_binary"] == 0.0]
    elif group_mode == "a0_psnr":
        ordered = sorted(zip(rows, budgets), key=lambda item: item[0]["a0_psnr"])
        k = max(1, len(ordered) // 4)
        hard = ordered[:k]
        easy = ordered[-k:]
    else:
        raise ValueError(f"Unsupported group_mode: {group_mode}")
    hard_budgets = [budget for _, budget in hard]
    easy_budgets = [budget for _, budget in easy]
    ratio = mean_or_none(hard_budgets) / max(mean_or_none(easy_budgets), 1e-12)
    result = {
        "candidate": candidate["name"],
        "kind": candidate["kind"],
        "hard_mean_b": mean_or_none(hard_budgets),
        "easy_mean_b": mean_or_none(easy_budgets),
        "strong_reference_mean_b": mean_or_none(easy_budgets),
        "hard_easy_ratio": ratio,
        "hard_bce_after_calibration": bce_from_probs(rows, candidate),
        "count": len(rows),
    }
    if group_mode == "a0_psnr":
        hard_scores = [row["z_img"] for row, _ in hard]
        easy_scores = [row["z_img"] for row, _ in easy]
        result.update(
            {
                "auc_hard_vs_easy_by_z_img": auc_score(hard_scores, easy_scores),
                "spearman_z_img_vs_a0_psnr": spearman(
                    z_values,
                    [row["a0_psnr"] for row in rows],
                ),
            }
        )
    return result


def train_constraints_pass(metrics, args):
    return (
        metrics["hard_mean_b"] is not None
        and metrics["easy_mean_b"] is not None
        and metrics["hard_mean_b"] >= args.train_hard_mean_min
        and metrics["easy_mean_b"] <= args.train_easy_mean_max
        and metrics["hard_easy_ratio"] >= args.train_ratio_min
        and metrics["hard_bce_after_calibration"] <= args.train_hard_bce_max
    )


def choose_candidate(candidates, train_rows, args):
    scored = []
    for candidate in candidates:
        metrics = summarize_candidate(train_rows, candidate, "binary")
        passed = train_constraints_pass(metrics, args)
        fail_count = 0
        fail_count += int(metrics["hard_mean_b"] < args.train_hard_mean_min)
        fail_count += int(metrics["easy_mean_b"] > args.train_easy_mean_max)
        fail_count += int(metrics["hard_easy_ratio"] < args.train_ratio_min)
        fail_count += int(metrics["hard_bce_after_calibration"] > args.train_hard_bce_max)
        metrics["train_pass"] = passed
        metrics["train_fail_count"] = fail_count
        scored.append((candidate, metrics))
    passing = [(candidate, metrics) for candidate, metrics in scored if metrics["train_pass"]]
    if passing:
        passing.sort(
            key=lambda item: (
                item[1]["easy_mean_b"],
                item[1]["hard_bce_after_calibration"],
                -item[1]["hard_mean_b"],
            )
        )
        return passing[0][0], scored
    scored.sort(
        key=lambda item: (
            item[1]["train_fail_count"],
            item[1]["easy_mean_b"],
            item[1]["hard_bce_after_calibration"],
            -item[1]["hard_easy_ratio"],
        )
    )
    return scored[0][0], scored


def build_replay_gate(train_metrics, test_metrics, max_abs_diff, args):
    checks = {
        "selected_train_constraints": {
            "observed": {
                "hard_mean_b": train_metrics["hard_mean_b"],
                "easy_mean_b": train_metrics["easy_mean_b"],
                "hard_easy_ratio": train_metrics["hard_easy_ratio"],
                "hard_bce_after_calibration": train_metrics["hard_bce_after_calibration"],
            },
            "required": (
                f"hard_mean_b >= {args.train_hard_mean_min}; "
                f"easy_mean_b <= {args.train_easy_mean_max}; "
                f"ratio >= {args.train_ratio_min}; "
                f"hard_bce <= {args.train_hard_bce_max}"
            ),
            "pass": train_constraints_pass(train_metrics, args),
        },
        "zero_residual_output": {
            "observed": max_abs_diff,
            "required": f"< {args.zero_diff_threshold}",
            "pass": max_abs_diff < args.zero_diff_threshold,
        },
        "auc_hard_vs_easy_by_z_img": {
            "observed": test_metrics["auc_hard_vs_easy_by_z_img"],
            "required": ">= 0.95",
            "pass": test_metrics["auc_hard_vs_easy_by_z_img"] is not None
            and test_metrics["auc_hard_vs_easy_by_z_img"] >= 0.95,
        },
        "spearman_z_img_vs_a0_psnr": {
            "observed": test_metrics["spearman_z_img_vs_a0_psnr"],
            "required": "<= -0.70",
            "pass": test_metrics["spearman_z_img_vs_a0_psnr"] is not None
            and test_metrics["spearman_z_img_vs_a0_psnr"] <= -0.70,
        },
        "hard_bottom25_mean_b_cons": {
            "observed": test_metrics["hard_mean_b"],
            "required": ">= 0.35",
            "pass": test_metrics["hard_mean_b"] is not None and test_metrics["hard_mean_b"] >= 0.35,
        },
        "easy_top25_mean_b_cons": {
            "observed": test_metrics["easy_mean_b"],
            "required": "<= 0.03",
            "pass": test_metrics["easy_mean_b"] is not None and test_metrics["easy_mean_b"] <= 0.03,
        },
        "strong_reference_mean_b_cons": {
            "observed": test_metrics["strong_reference_mean_b"],
            "required": "<= 0.03",
            "pass": test_metrics["strong_reference_mean_b"] is not None
            and test_metrics["strong_reference_mean_b"] <= 0.03,
        },
        "hard_easy_b_cons_ratio": {
            "observed": test_metrics["hard_easy_ratio"],
            "required": ">= 10.0",
            "pass": test_metrics["hard_easy_ratio"] >= 10.0,
        },
        "hard_bce_after_calibration": {
            "observed": test_metrics["hard_bce_after_calibration"],
            "required": "<= 0.55",
            "pass": test_metrics["hard_bce_after_calibration"] <= 0.55,
        },
    }
    return {
        "stage": "APDR-v0.2RC conservative budget replay gate",
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
    }


def evaluate_oracle(original, apdr, args, device, calibration, candidate, csv_path):
    loader = v02r.test_dataloader(args.data_dir, "Haze4K", batch_size=1, num_workers=0)
    rows = []
    v02r.set_selector_eval_mode(apdr)
    original.eval()
    with torch.no_grad(), csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["name", "a0_psnr", "oracle_psnr", "delta_psnr", "b_cons", "s_pixel_mean"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (input_img, label_img, name) in enumerate(loader):
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            padded, h, w = v02r.pad_to_factor(input_img)
            a0 = v02r.crop_to(original(padded)[2], h, w)
            apdr(padded)
            item = v02r.full_item(apdr)
            anchor = v02r.crop_to(item["anchor"], h, w)
            spatial = v02r.crop_to(item["spatial_gate_unit"], h, w)
            z_img = item["global_logits"].view(-1).item()
            b_cons = apply_candidate(z_img, candidate)
            delta_star = (label_img - anchor).clamp(
                -float(args.apdr_residual_max),
                float(args.apdr_residual_max),
            )
            oracle = (anchor + b_cons * spatial * delta_star).clamp(0.0, 1.0)
            a0_psnr = psnr(a0, label_img)
            oracle_psnr = psnr(oracle, label_img)
            row = {
                "name": name[0],
                "a0_psnr": a0_psnr,
                "oracle_psnr": oracle_psnr,
                "delta_psnr": oracle_psnr - a0_psnr,
                "b_cons": b_cons,
                "s_pixel_mean": spatial.mean().item(),
            }
            rows.append(row)
            writer.writerow(row)
            if (idx + 1) % args.progress_freq == 0:
                print(f"oracle_eval {idx + 1}/{len(loader)}", flush=True)

    by_psnr = sorted(rows, key=lambda row: row["a0_psnr"])
    count = len(by_psnr)
    k = max(1, count // 4)
    hard = by_psnr[:k]
    easy = by_psnr[-k:]
    deltas = [row["delta_psnr"] for row in rows]
    strong_regressions = [row for row in easy if row["delta_psnr"] <= -0.05]
    severe_regressions = [row for row in rows if row["delta_psnr"] <= -0.20]
    worst10 = sorted(deltas)[:10]
    summary = {
        "count": count,
        "mean_psnr_delta": statistics.mean(deltas),
        "hard_bottom25_mean_delta": statistics.mean(row["delta_psnr"] for row in hard),
        "easy_top25_mean_delta": statistics.mean(row["delta_psnr"] for row in easy),
        "strong_reference_regressions": len(strong_regressions),
        "severe_regressions": len(severe_regressions),
        "worst10_image_mean_delta": statistics.mean(worst10),
    }
    checks = {
        "oracle_mean_psnr_delta": {
            "observed": summary["mean_psnr_delta"],
            "required": ">= +0.050",
            "pass": summary["mean_psnr_delta"] >= 0.050,
        },
        "oracle_hard_bottom25_delta": {
            "observed": summary["hard_bottom25_mean_delta"],
            "required": ">= +0.150",
            "pass": summary["hard_bottom25_mean_delta"] >= 0.150,
        },
        "oracle_easy_top25_delta": {
            "observed": summary["easy_top25_mean_delta"],
            "required": ">= -0.005",
            "pass": summary["easy_top25_mean_delta"] >= -0.005,
        },
        "oracle_strong_reference_regressions": {
            "observed": summary["strong_reference_regressions"],
            "required": "<= 5 / 250",
            "pass": summary["strong_reference_regressions"] <= 5,
        },
        "oracle_severe_regressions": {
            "observed": summary["severe_regressions"],
            "required": "== 0 / 1000",
            "pass": summary["severe_regressions"] == 0,
        },
    }
    return {
        "summary": summary,
        "gate": {
            "stage": "APDR-v0.2RC oracle residual ceiling",
            "checks": checks,
            "pass": all(item["pass"] for item in checks.values()),
        },
    }


def write_candidate_csv(scored, test_rows, selected, path):
    fieldnames = [
        "candidate",
        "kind",
        "selected",
        "train_pass",
        "train_fail_count",
        "train_hard_mean_b",
        "train_easy_mean_b",
        "train_ratio",
        "train_hard_bce",
        "test_hard_mean_b",
        "test_easy_mean_b",
        "test_ratio",
        "test_hard_bce",
        "test_auc_z",
        "test_spearman_z",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate, train_metrics in scored:
            test_metrics = summarize_candidate(test_rows, candidate, "a0_psnr")
            writer.writerow(
                {
                    "candidate": candidate["name"],
                    "kind": candidate["kind"],
                    "selected": candidate["name"] == selected["name"],
                    "train_pass": train_metrics["train_pass"],
                    "train_fail_count": train_metrics["train_fail_count"],
                    "train_hard_mean_b": train_metrics["hard_mean_b"],
                    "train_easy_mean_b": train_metrics["easy_mean_b"],
                    "train_ratio": train_metrics["hard_easy_ratio"],
                    "train_hard_bce": train_metrics["hard_bce_after_calibration"],
                    "test_hard_mean_b": test_metrics["hard_mean_b"],
                    "test_easy_mean_b": test_metrics["easy_mean_b"],
                    "test_ratio": test_metrics["hard_easy_ratio"],
                    "test_hard_bce": test_metrics["hard_bce_after_calibration"],
                    "test_auc_z": test_metrics.get("auc_hard_vs_easy_by_z_img"),
                    "test_spearman_z": test_metrics.get("spearman_z_img_vs_a0_psnr"),
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_2rc_budget_seed3407")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--global_epochs", type=int, default=5)
    parser.add_argument("--spatial_epochs", type=int, default=3)
    parser.add_argument("--global_batch_size", type=int, default=4)
    parser.add_argument("--spatial_batch_size", type=int, default=8)
    parser.add_argument("--global_resize", type=int, default=384)
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--global_learning_rate", type=float, default=2e-4)
    parser.add_argument("--spatial_learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--global_train_batches_per_epoch", type=int, default=0)
    parser.add_argument("--spatial_train_batches_per_epoch", type=int, default=0)
    parser.add_argument("--calibration_images", type=int, default=0)
    parser.add_argument("--budget_calibration_images", type=int, default=0)
    parser.add_argument("--pixel_samples_per_image", type=int, default=2048)
    parser.add_argument("--loss_eval_images", type=int, default=256)
    parser.add_argument("--spatial_tau", type=float, default=0.0)
    parser.add_argument("--global_bce_lambda", type=float, default=1.0)
    parser.add_argument("--global_focal_lambda", type=float, default=0.5)
    parser.add_argument("--global_rank_lambda", type=float, default=0.2)
    parser.add_argument("--spatial_bce_lambda", type=float, default=1.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--rank_margin", type=float, default=1.0)
    parser.add_argument("--budget_temperature_floor", type=float, default=0.05)
    parser.add_argument("--apdr_residual_max", type=float, default=0.04)
    parser.add_argument("--apdr_gate_max", type=float, default=0.5)
    parser.add_argument("--apdr_gate_init", type=float, default=0.01)
    parser.add_argument("--zero_diff_threshold", type=float, default=1e-6)
    parser.add_argument("--progress_freq", type=int, default=100)
    parser.add_argument("--train_hard_mean_min", type=float, default=0.35)
    parser.add_argument("--train_easy_mean_max", type=float, default=0.03)
    parser.add_argument("--train_ratio_min", type=float, default=10.0)
    parser.add_argument("--train_hard_bce_max", type=float, default=0.55)
    parser.add_argument("--run_oracle_on_replay_fail", action="store_true")
    parser.add_argument("--fail_on_gate", action="store_true")
    args = parser.parse_args()

    v02r.set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original, apdr, checkpoint_load = v02r.build_models(args, device)
    print("checkpoint loaded", json.dumps(checkpoint_load), flush=True)

    calibration = v02r.compute_calibration(apdr, args, device)
    print("calibration", json.dumps(calibration["summary"], indent=2), flush=True)

    global_history, global_scope = v02r.train_global_router(apdr, args, device, calibration)
    train_scores = v02r.collect_train_scores(apdr, args, device, calibration)
    base_budget_calibration = v02r.calibrate_budget(apdr, args, train_scores)
    print("base_budget_calibration", json.dumps(base_budget_calibration, indent=2), flush=True)
    spatial_history, spatial_scope = v02r.train_spatial_gate(apdr, args, device, calibration)

    train_rows = rows_from_train_scores(train_scores, base_budget_calibration)
    train_csv = output_dir / f"budget_train_scores_{args.tag}.csv"
    write_train_rows(train_rows, train_csv)
    test_csv = output_dir / f"budget_test_scores_{args.tag}.csv"
    test_rows, max_abs_diff = collect_test_rows(original, apdr, args, device, calibration, test_csv)

    candidates = generate_candidates(train_rows, base_budget_calibration)
    selected, scored = choose_candidate(candidates, train_rows, args)
    selected_train_metrics = summarize_candidate(train_rows, selected, "binary")
    selected_test_metrics = summarize_candidate(test_rows, selected, "a0_psnr")
    replay_gate = build_replay_gate(selected_train_metrics, selected_test_metrics, max_abs_diff, args)
    candidate_csv = output_dir / f"budget_candidates_{args.tag}.csv"
    write_candidate_csv(scored, test_rows, selected, candidate_csv)

    oracle = None
    oracle_csv = None
    if replay_gate["pass"] or args.run_oracle_on_replay_fail:
        oracle_csv = output_dir / f"oracle_per_image_{args.tag}.csv"
        oracle = evaluate_oracle(original, apdr, args, device, calibration, selected, oracle_csv)

    gate = {
        "stage": "APDR-v0.2RC conservative budget replay",
        "replay_gate": replay_gate,
        "oracle_gate": oracle["gate"] if oracle else None,
        "pass": replay_gate["pass"] and (oracle is None or oracle["gate"]["pass"]),
    }
    result = {
        "stage": "apdr_v0_2rc_conservative_budget_replay",
        "tag": args.tag,
        "seed": args.seed,
        "device": str(device),
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "checkpoint_load": checkpoint_load,
        "calibration": calibration["summary"],
        "base_budget_calibration": base_budget_calibration,
        "selected_candidate": selected,
        "selected_train_metrics": selected_train_metrics,
        "selected_test_metrics": selected_test_metrics,
        "max_abs_diff_vs_a0": max_abs_diff,
        "global_history": global_history,
        "spatial_history": spatial_history,
        "train_scope": {"global": global_scope, "spatial": spatial_scope},
        "replay_gate": replay_gate,
        "oracle": oracle,
        "gate": gate,
        "artifacts": {
            "train_scores_csv": str(train_csv),
            "test_scores_csv": str(test_csv),
            "candidate_csv": str(candidate_csv),
            "oracle_csv": str(oracle_csv) if oracle_csv else None,
        },
        "pass": gate["pass"],
    }
    summary_json = output_dir / f"budget_summary_{args.tag}.json"
    gate_json = output_dir / f"gate_{args.tag}.json"
    summary_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    gate_json.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if args.fail_on_gate and not gate["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
