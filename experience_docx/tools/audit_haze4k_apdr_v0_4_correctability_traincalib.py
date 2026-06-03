import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))

from audit_haze4k_apdr_correctability_proxy import (  # noqa: E402
    TabularBenefitProxy,
    auc_score,
    build_apdr_model,
    build_loader,
    collect_rows,
    matrix_from_rows,
    spearman,
)


def percentile(values, pct):
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


def labeled_indices(rows):
    return [idx for idx, row in enumerate(rows) if row["benefit_label"] in (0, 1)]


def split_folds(indices, folds, seed):
    shuffled = list(indices)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    fold_count = max(2, min(folds, len(shuffled)))
    return [shuffled[idx::fold_count] for idx in range(fold_count)]


def feature_names_from_rows(rows):
    excluded = {"name", "index", "anchor_psnr", "low_oracle_gain", "benefit_label"}
    return [name for name in rows[0].keys() if name not in excluded]


def train_model(rows, feature_names, train_indices, args, device):
    labeled = [idx for idx in train_indices if rows[idx]["benefit_label"] in (0, 1)]
    if not labeled:
        raise RuntimeError("No labeled rows available for correctability training.")
    y = torch.tensor([rows[idx]["benefit_label"] for idx in labeled], dtype=torch.float32, device=device)
    pos = y.sum().item()
    neg = y.numel() - pos
    if pos == 0 or neg == 0:
        raise RuntimeError("Correctability training labels contain only one class.")
    norm_source = matrix_from_rows(rows, feature_names, train_indices, device)
    mean = norm_source.mean(dim=0, keepdim=True)
    std = norm_source.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x = (matrix_from_rows(rows, feature_names, labeled, device) - mean) / std
    model = TabularBenefitProxy(len(feature_names), args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    )
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if args.progress_freq and step % args.progress_freq == 0:
            model.eval()
            with torch.no_grad():
                scores = torch.sigmoid(model(x)).detach().cpu().tolist()
            history.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "train_auc": auc_score(scores, [rows[idx]["benefit_label"] for idx in labeled]),
                }
            )
    return model, mean, std, history, labeled


def score_rows(model, mean, std, rows, feature_names, device):
    model.eval()
    with torch.no_grad():
        x = (matrix_from_rows(rows, feature_names, list(range(len(rows))), device) - mean) / std
        return torch.sigmoid(model(x)).detach().cpu().tolist()


def oof_scores(rows, feature_names, args, device):
    all_indices = list(range(len(rows)))
    folds = split_folds(all_indices, args.folds, args.seed)
    scores = [None] * len(rows)
    histories = []
    for fold_idx, valid_fold in enumerate(folds):
        valid_set = set(valid_fold)
        train_indices = [idx for idx in all_indices if idx not in valid_set]
        model, mean, std, history, _ = train_model(rows, feature_names, train_indices, args, device)
        fold_scores = score_rows(model, mean, std, rows, feature_names, device)
        for idx in valid_fold:
            scores[idx] = fold_scores[idx]
        for item in history:
            histories.append({"fold": fold_idx, **item})
        print(f"fold={fold_idx} valid_count={len(valid_fold)}", flush=True)
    return scores, histories


def threshold_metrics(rows, scores, tau, prefix):
    valid = [idx for idx, score in enumerate(scores) if score is not None]
    labeled = [idx for idx in valid if rows[idx]["benefit_label"] in (0, 1)]
    labels = [rows[idx]["benefit_label"] for idx in labeled]
    labeled_scores = [scores[idx] for idx in labeled]
    gains = [rows[idx]["low_oracle_gain"] for idx in valid]
    valid_scores = [scores[idx] for idx in valid]
    hard_cut = percentile([rows[idx]["anchor_psnr"] for idx in valid], 25)
    easy_cut = percentile([rows[idx]["anchor_psnr"] for idx in valid], 75)
    easy = [idx for idx in valid if rows[idx]["anchor_psnr"] >= easy_cut]
    hard_positive = [
        idx
        for idx in valid
        if rows[idx]["anchor_psnr"] <= hard_cut and rows[idx]["low_oracle_gain"] >= 0.10
    ]
    negative = [idx for idx in labeled if rows[idx]["benefit_label"] == 0]
    positive = [idx for idx in labeled if rows[idx]["benefit_label"] == 1]
    opened = [idx for idx in valid if scores[idx] >= tau]
    def rate(items):
        return sum(scores[idx] >= tau for idx in items) / max(len(items), 1)

    return {
        f"{prefix}_count": len(valid),
        f"{prefix}_labeled_count": len(labeled),
        f"{prefix}_positive_count": len(positive),
        f"{prefix}_negative_count": len(negative),
        f"{prefix}_auc": auc_score(labeled_scores, labels),
        f"{prefix}_spearman_score_low_gain": spearman(valid_scores, gains),
        f"{prefix}_tau": tau,
        f"{prefix}_open_rate": len(opened) / max(len(valid), 1),
        f"{prefix}_easy_top25_open_rate": rate(easy),
        f"{prefix}_negative_false_open": rate(negative),
        f"{prefix}_positive_recall": rate(positive),
        f"{prefix}_positive_hard_recall": rate(hard_positive),
        f"{prefix}_positive_hard_count": len(hard_positive),
        f"{prefix}_mean_score_easy_top25": statistics.mean(scores[idx] for idx in easy) if easy else None,
        f"{prefix}_mean_score_oracle_positive_hard": (
            statistics.mean(scores[idx] for idx in hard_positive) if hard_positive else None
        ),
        f"{prefix}_mean_low_oracle_gain": statistics.mean(gains),
        f"{prefix}_opened_low_oracle_gain_proxy": (
            statistics.mean(rows[idx]["low_oracle_gain"] for idx in opened) if opened else 0.0
        ),
        f"{prefix}_zero_closed_low_oracle_gain_proxy": statistics.mean(
            rows[idx]["low_oracle_gain"] if scores[idx] >= tau else 0.0 for idx in valid
        ),
        f"{prefix}_hard_anchor_psnr_cut": hard_cut,
        f"{prefix}_easy_anchor_psnr_cut": easy_cut,
    }


def choose_tau(rows, scores, args):
    candidates = sorted({score for score in scores if score is not None})
    if not candidates:
        raise RuntimeError("No scores available for tau calibration.")
    candidates = [min(candidates) - 1e-6] + candidates + [max(candidates) + 1e-6]
    feasible = []
    fallback = []
    for tau in candidates:
        metrics = threshold_metrics(rows, scores, tau, "train")
        easy_ok = metrics["train_easy_top25_open_rate"] <= args.max_easy_open
        neg_ok = metrics["train_negative_false_open"] <= args.max_negative_false_open
        recall = metrics["train_positive_hard_recall"]
        record = (recall, metrics["train_open_rate"], -tau, tau, metrics)
        if easy_ok and neg_ok and recall >= args.min_positive_hard_recall:
            feasible.append(record)
        if easy_ok and neg_ok:
            fallback.append(record)
    if feasible:
        chosen = max(feasible)[3]
        status = "pass_constraints"
    elif fallback:
        chosen = max(fallback)[3]
        status = "fallback_best_recall_under_safety"
    else:
        chosen = max(candidates)
        status = "fallback_close_all"
    return chosen, status


def attach_scores(rows, scores):
    for row, score in zip(rows, scores):
        row["proxy_score"] = score
    return rows


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_4_correctability_traincalib")
    parser.add_argument("--train_max_images", type=int, default=0)
    parser.add_argument("--test_max_images", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--sigma", type=float, default=7.0)
    parser.add_argument("--positive_gain", type=float, default=0.10)
    parser.add_argument("--negative_gain", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_easy_open", type=float, default=0.05)
    parser.add_argument("--max_negative_false_open", type=float, default=0.02)
    parser.add_argument("--min_positive_hard_recall", type=float, default=0.95)
    parser.add_argument("--progress_freq", type=int, default=200)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    model = build_apdr_model(args.selector_checkpoint, args.residual_max, device)
    train_rows = collect_rows(
        model,
        build_loader(args.data_dir, "train", args.train_max_images, args.num_workers),
        device,
        args,
    )
    test_rows = collect_rows(
        model,
        build_loader(args.data_dir, "test", args.test_max_images, args.num_workers),
        device,
        args,
    )
    if len(train_rows) < 20 or len(test_rows) < 20:
        raise RuntimeError("Too few rows for train calibration and test evaluation.")

    feature_names = feature_names_from_rows(train_rows)
    train_scores, fold_history = oof_scores(train_rows, feature_names, args, device)
    tau_train, tau_status = choose_tau(train_rows, train_scores, args)

    final_model, mean, std, final_history, final_labeled = train_model(
        train_rows,
        feature_names,
        labeled_indices(train_rows),
        args,
        device,
    )
    test_scores = score_rows(final_model, mean, std, test_rows, feature_names, device)
    train_metrics = threshold_metrics(train_rows, train_scores, tau_train, "train_oof")
    test_metrics = threshold_metrics(test_rows, test_scores, tau_train, "test")
    checks = {
        "train_easy_open": {
            "observed": train_metrics["train_oof_easy_top25_open_rate"],
            "required": f"<= {args.max_easy_open}",
            "pass": train_metrics["train_oof_easy_top25_open_rate"] <= args.max_easy_open,
        },
        "train_positive_hard_recall": {
            "observed": train_metrics["train_oof_positive_hard_recall"],
            "required": f">= {args.min_positive_hard_recall}",
            "pass": train_metrics["train_oof_positive_hard_recall"] >= args.min_positive_hard_recall,
        },
        "train_negative_false_open": {
            "observed": train_metrics["train_oof_negative_false_open"],
            "required": f"<= {args.max_negative_false_open}",
            "pass": train_metrics["train_oof_negative_false_open"] <= args.max_negative_false_open,
        },
        "test_auc": {
            "observed": test_metrics["test_auc"],
            "required": ">= 0.90",
            "pass": test_metrics["test_auc"] is not None and test_metrics["test_auc"] >= 0.90,
        },
        "test_spearman": {
            "observed": test_metrics["test_spearman_score_low_gain"],
            "required": ">= 0.70",
            "pass": (
                test_metrics["test_spearman_score_low_gain"] is not None
                and test_metrics["test_spearman_score_low_gain"] >= 0.70
            ),
        },
        "test_easy_open": {
            "observed": test_metrics["test_easy_top25_open_rate"],
            "required": f"<= {args.max_easy_open}",
            "pass": test_metrics["test_easy_top25_open_rate"] <= args.max_easy_open,
        },
        "test_positive_hard_recall": {
            "observed": test_metrics["test_positive_hard_recall"],
            "required": f">= {args.min_positive_hard_recall}",
            "pass": test_metrics["test_positive_hard_recall"] >= args.min_positive_hard_recall,
        },
        "test_negative_false_open": {
            "observed": test_metrics["test_negative_false_open"],
            "required": f"<= {args.max_negative_false_open}",
            "pass": test_metrics["test_negative_false_open"] <= args.max_negative_false_open,
        },
    }
    result = {
        "stage": "APDR-v0.4 CCLF train-calibrated CorrectabilityOpen audit",
        "tag": args.tag,
        "tau_train": tau_train,
        "tau_status": tau_status,
        "feature_names": feature_names,
        "summary": {
            **train_metrics,
            **test_metrics,
            "final_train_labeled_count": len(final_labeled),
        },
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
        "fold_history": fold_history,
        "final_history": final_history,
        "args": vars(args),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"correctability_traincalib_{args.tag}.json"
    train_csv = output_dir / f"correctability_traincalib_train_oof_{args.tag}.csv"
    test_csv = output_dir / f"correctability_traincalib_test_{args.tag}.csv"
    history_csv = output_dir / f"correctability_traincalib_history_{args.tag}.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_rows(train_csv, attach_scores(train_rows, train_scores))
    write_rows(test_csv, attach_scores(test_rows, test_scores))
    history = [{"phase": "fold", **item} for item in fold_history] + [
        {"phase": "final", **item} for item in final_history
    ]
    if history:
        with history_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {train_csv}")
    print(f"wrote {test_csv}")
    if history:
        print(f"wrote {history_csv}")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
