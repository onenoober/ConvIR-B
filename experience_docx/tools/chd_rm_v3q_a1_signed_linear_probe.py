#!/usr/bin/env python3
"""Cross-fitted signed-value linear-probe audit for v3q A1."""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROUTE_ID = "haze4k_v5_chd_rm_v3q_active_signed_value_20260712"
OPERATORS = ("D_ref", "D_rep")
FEATURE_COLUMNS = (
    "direct_step_energy", "d7c_score_mean", "delta_l1_mean",
    "delta_r_mean", "delta_g_mean", "delta_b_mean",
    "delta_r_std", "delta_g_std", "delta_b_std",
    "midpoint_r_mean", "midpoint_g_mean", "midpoint_b_mean",
    "hazy_minus_midpoint_dot_delta", "gradient_alignment",
    "cov_hazy_minus_midpoint_delta_r", "cov_hazy_minus_midpoint_delta_g",
    "cov_hazy_minus_midpoint_delta_b", "hazy_luminance_mean",
    "hazy_luminance_std", "hazy_saturation_mean", "clip_fraction_0p125",
    "clip_fraction_0p25", "signed_distance_to_clip_0p125",
    "signed_distance_to_clip_0p25",
)
EXPECTED_ROWS = {
    "smoke": {"D_ref": 14151, "D_rep": 14151},
    "formal": {"D_ref": 503973, "D_rep": 503987},
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path, rows):
    if not rows:
        raise ValueError("cannot write an empty compact CSV")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--a0b-closeout", required=True)
    parser.add_argument("--expected-a0b-closeout-sha256", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--expected-schema-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--run-mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--route-commit", required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_contract(args):
    if args.fold_count != 5:
        raise ValueError("A1 is fixed to five clean-reference-grouped outer folds")
    if sha256_file(args.features) != args.expected_features_sha256:
        raise RuntimeError("A1 feature-table hash mismatch")
    if sha256_file(args.a0b_closeout) != args.expected_a0b_closeout_sha256:
        raise RuntimeError("A0b closeout hash mismatch")
    if sha256_file(args.schema) != args.expected_schema_sha256:
        raise RuntimeError("A0b schema hash mismatch")
    closeout = json.loads(Path(args.a0b_closeout).read_text(encoding="utf-8"))
    if closeout.get("decision") != "V3Q_A0B_FORMAL_PASS_AUTHORIZE_A1_SIGNED_LINEAR_PROBE_ONLY":
        raise RuntimeError("A0b formal does not authorize A1 only")
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    if tuple(schema.get("model_feature_columns", ())) != FEATURE_COLUMNS:
        raise RuntimeError("A0b model-feature schema mismatch")
    forbidden = set(schema.get("forbidden_model_features", ()))
    if forbidden.intersection(FEATURE_COLUMNS):
        raise RuntimeError("forbidden feature included in A1 schema")
    return closeout, schema


def count_images(path):
    counts = Counter()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["g1_state"] in {"beneficial", "harmful"}:
                counts[(row["operator_label"], row["name"])] += 1
    return counts


def load_data(path, image_counts):
    counts = Counter()
    for (operator, _), value in image_counts.items():
        counts[operator] += value
    data = {
        operator: {
            "x": np.empty((counts[operator], len(FEATURE_COLUMNS)), dtype=np.float32),
            "y": np.empty(counts[operator], dtype=np.float32),
            "fold": np.empty(counts[operator], dtype=np.int8),
            "image": np.empty(counts[operator], dtype=np.int32),
            "meta": np.empty((counts[operator], 3), dtype=np.float32),
        }
        for operator in OPERATORS
    }
    image_ids = {operator: {} for operator in OPERATORS}
    offsets = Counter()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            operator = row["operator_label"]
            state = row["g1_state"]
            if operator not in OPERATORS or state not in {"beneficial", "harmful"}:
                continue
            index = offsets[operator]
            item = data[operator]
            item["x"][index] = [float(row[column]) for column in FEATURE_COLUMNS]
            item["y"][index] = float(state == "beneficial")
            item["fold"][index] = int(row["fold"])
            name = row["name"]
            image_id = image_ids[operator].setdefault(name, len(image_ids[operator]))
            item["image"][index] = image_id
            # Explicitly non-deployable diagnostic control; never a model feature.
            item["meta"][index] = (float(row["fold"]), float(row["block_y"]), float(row["block_x"]))
            offsets[operator] += 1
    for operator in OPERATORS:
        if offsets[operator] != counts[operator]:
            raise RuntimeError(f"A1 row-load mismatch for {operator}")
        if np.any(data[operator]["fold"] < 0) or np.any(data[operator]["fold"] >= 5):
            raise RuntimeError(f"A1 fold range mismatch for {operator}")
    return data


def image_weights(image_ids, image_counts):
    counts = np.bincount(image_ids, minlength=len(image_counts)).astype(np.float32)
    return 1.0 / counts[image_ids]


def shuffled_within_image(y, image_ids, seed):
    result = y.copy()
    order = np.argsort(image_ids, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(image_ids[order])) + 1, len(order)]
    rng = np.random.default_rng(seed)
    for start, end in zip(starts[:-1], starts[1:]):
        indices = order[start:end]
        result[indices] = result[indices][rng.permutation(len(indices))]
    return result


def weighted_standardize(train_x, train_w, test_x):
    denominator = float(train_w.sum())
    mean = (train_x * train_w[:, None]).sum(axis=0) / denominator
    variance = ((train_x - mean) ** 2 * train_w[:, None]).sum(axis=0) / denominator
    scale = np.sqrt(np.maximum(variance, 1e-8))
    return (train_x - mean) / scale, (test_x - mean) / scale


def fit_predict(train_x, train_y, train_w, test_x, args, seed):
    train_x, test_x = weighted_standardize(train_x, train_w, test_x)
    device = torch.device(args.device)
    torch.manual_seed(seed)
    x = torch.from_numpy(np.ascontiguousarray(train_x)).to(device)
    y = torch.from_numpy(np.ascontiguousarray(train_y)).to(device)
    w = torch.from_numpy(np.ascontiguousarray(train_w)).to(device)
    coefficient = torch.zeros(x.shape[1], device=device, requires_grad=True)
    intercept = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.Adam((coefficient, intercept), lr=args.learning_rate)
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = x @ coefficient + intercept
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / w.sum()
        loss = loss + args.l2 * coefficient.square().sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        test = torch.from_numpy(np.ascontiguousarray(test_x)).to(device)
        probability = torch.sigmoid(test @ coefficient + intercept).cpu().numpy()
    del x, y, w, test, coefficient, intercept
    torch.cuda.empty_cache()
    return probability.astype(np.float32, copy=False)


def weighted_auc(y, score, weight):
    order = np.argsort(score, kind="stable")
    y, score, weight = y[order], score[order], weight[order]
    positive = weight[y > 0.5].sum()
    negative = weight[y <= 0.5].sum()
    if positive <= 0.0 or negative <= 0.0:
        return float("nan")
    numerator = 0.0
    negative_before = 0.0
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[end] == score[start]:
            end += 1
        group_y, group_w = y[start:end], weight[start:end]
        group_positive = group_w[group_y > 0.5].sum()
        group_negative = group_w[group_y <= 0.5].sum()
        numerator += group_positive * (negative_before + 0.5 * group_negative)
        negative_before += group_negative
        start = end
    return float(numerator / (positive * negative))


def metrics(y, score, weight):
    clipped = np.clip(score, 1e-7, 1.0 - 1e-7)
    total = weight.sum()
    return {
        "weighted_auc": weighted_auc(y, score, weight),
        "weighted_logloss": float((weight * (-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))).sum() / total),
        "weighted_brier": float((weight * (score - y) ** 2).sum() / total),
        "weighted_prevalence": float((weight * y).sum() / total),
    }


def run_config(operator, item, config, args, seed):
    x = item["x"]
    if config == "energy_only":
        x = x[:, :1]
        train_y = item["y"]
    elif config == "unsigned_magnitude_only":
        x = np.abs(x)
        train_y = item["y"]
    elif config == "within_image_shuffled_label":
        train_y = shuffled_within_image(item["y"], item["image"], seed)
    elif config == "metadata_only_nondeployable_control":
        x = item["meta"]
        train_y = item["y"]
    else:
        train_y = item["y"]
    weight = image_weights(item["image"], np.unique(item["image"]))
    oof = np.empty(len(train_y), dtype=np.float32)
    fold_rows = []
    for fold in range(args.fold_count):
        test_mask = item["fold"] == fold
        train_mask = ~test_mask
        probability = fit_predict(x[train_mask], train_y[train_mask], weight[train_mask], x[test_mask], args, seed + fold)
        oof[test_mask] = probability
        fold_rows.append({
            "operator_label": operator,
            "config": config,
            "outer_fold": fold,
            "row_count": int(test_mask.sum()),
            **metrics(item["y"][test_mask], probability, weight[test_mask]),
        })
    return metrics(item["y"], oof, weight), fold_rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    closeout, schema = validate_contract(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    image_counts = count_images(args.features)
    data = load_data(args.features, image_counts)
    observed = {operator: int(len(data[operator]["y"])) for operator in OPERATORS}
    if observed != EXPECTED_ROWS[args.run_mode]:
        raise RuntimeError(f"A1 eligible-row mismatch: {observed}")
    configs = (
        "full_signed_features", "energy_only", "unsigned_magnitude_only",
        "within_image_shuffled_label", "metadata_only_nondeployable_control",
    )
    summary_rows, fold_rows = [], []
    for operator_index, operator in enumerate(OPERATORS):
        for config_index, config in enumerate(configs):
            value, by_fold = run_config(
                operator, data[operator], config, args,
                args.seed + 1000 * operator_index + 100 * config_index,
            )
            summary_rows.append({"operator_label": operator, "config": config, "eligible_row_count": observed[operator], **value})
            fold_rows.extend(by_fold)
            print(f"{args.run_tag}_{operator}_{config}", flush=True)
    lookup = {(row["operator_label"], row["config"]): row for row in summary_rows}
    full_pass = all(
        lookup[(operator, "full_signed_features")]["weighted_auc"] >= 0.60
        and lookup[(operator, "full_signed_features")]["weighted_auc"]
            - max(lookup[(operator, "energy_only")]["weighted_auc"], lookup[(operator, "unsigned_magnitude_only")]["weighted_auc"])
            >= 0.02
        and lookup[(operator, "within_image_shuffled_label")]["weighted_auc"] <= 0.53
        and lookup[(operator, "metadata_only_nondeployable_control")]["weighted_auc"] <= 0.53
        for operator in OPERATORS
    )
    if args.run_mode == "smoke":
        decision = "V3Q_A1_SMOKE_PASS_AUTHORIZE_FORMAL_ONLY"
        state = "COMPLETED_GATE_PASS"
        authorizes = "v3q-A1-formal"
        reason = "pinned source, grouped-fold, per-image-weight, and compact-output contracts passed"
    else:
        decision = "V3Q_A1_PASS_AUTHORIZE_A2_SIGNED_FEATURE_ABLATION_ONLY" if full_pass else "V3Q_A1_FAIL_STOP_LEARNED_SIGNED_SCORING"
        state = "COMPLETED_GATE_PASS" if full_pass else "COMPLETED_GATE_FAIL"
        authorizes = "v3q-A2 signed feature ablation only" if full_pass else "none"
        reason = "both operators cleared the signed utility and negative-control gate" if full_pass else "signed probe did not clear both-operator utility and negative-control gate"
    source_manifest = {
        "route_id": ROUTE_ID, "run_tag": args.run_tag, "run_mode": args.run_mode,
        "route_commit": args.route_commit, "feature_table": str(Path(args.features)),
        "feature_table_sha256": args.expected_features_sha256,
        "a0b_closeout_sha256": args.expected_a0b_closeout_sha256,
        "schema_sha256": args.expected_schema_sha256, "schema": schema,
        "outer_fold_contract": "frozen clean-reference grouped five-fold labels from A0b; no fold reassignment",
        "weight_contract": "each eligible block has weight 1 / eligible blocks in its image",
        "controls": list(configs), "locked_test_touched": False,
        "canary_touched": False, "policy_replay_occurred": False,
    }
    closeout_value = {
        "route_id": ROUTE_ID, "run_id": args.run_tag, "stage": "v3q-A1-signed-linear-probe",
        "state": state, "gate_type": "scientific_utility", "decision": decision,
        "metric_contract": "v3q route card A1 grouped OOF per-image-weighted signed probe",
        "authorizes": authorizes, "reason": reason,
        "locked_test_touched": False, "canary_touched": False, "policy_replay_occurred": False,
    }
    write_rows(output_dir / f"{args.run_tag}_summary.csv", summary_rows)
    write_rows(output_dir / f"{args.run_tag}_by_fold.csv", fold_rows)
    write_json(output_dir / f"{args.run_tag}_summary.json", {"rows": summary_rows, "gate_pass": full_pass})
    write_json(output_dir / f"{args.run_tag}_source_manifest.json", source_manifest)
    write_json(output_dir / f"{args.run_tag}_closeout.json", closeout_value)
    print(decision)


if __name__ == "__main__":
    main()
