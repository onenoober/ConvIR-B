#!/usr/bin/env python3
"""Freeze the R3 development/confirmation identity ledger without model runtime."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from route_program_api import (
    asset_path,
    atomic_json,
    load_context,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
)


ROUTE_ID = "haze4k_v5_r3_proposal_first_acv_20260717"
OPERATION_ID = "R3_S0_LEDGER_FREEZE"
EXPECTED_TRAIN_INNER = 2400
EXPECTED_VAL_INNER = 600
EXPECTED_HISTORICAL = 1200
EXPECTED_ELIGIBLE = 1200
EXPECTED_OPERATORS = {"D_ref", "D_rep"}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_value(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_reference_group(name):
    return Path(name).stem.split("_", 1)[0]


def haze_signature(name):
    stem = Path(name).stem
    return stem.split("_", 1)[1] if "_" in stem else "__none__"


def stable_token(scope, value, seed):
    raw = f"{ROUTE_ID}:{seed}:{scope}:{value}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def group_names(names):
    groups = defaultdict(list)
    for name in sorted(names):
        groups[clean_reference_group(name)].append(name)
    return {group: sorted(items) for group, items in sorted(groups.items())}


def group_profile(items):
    return tuple(sorted(Counter(haze_signature(name) for name in items).items()))


def profile_key(profile):
    return json.dumps(profile, separators=(",", ":"))


def profile_penalty(selected, groups_by_profile, target_ratio):
    selected_counts = Counter()
    for group in selected:
        selected_counts[group_profile(groups_by_profile["groups"][group])] += len(
            groups_by_profile["groups"][group]
        )
    return sum(
        abs(selected_counts[profile] - total * target_ratio)
        for profile, total in groups_by_profile["profile_image_counts"].items()
    )


def select_confirmation(groups, target_count, seed):
    total_count = sum(len(items) for items in groups.values())
    ratio = target_count / total_count
    by_profile = defaultdict(list)
    for group, items in groups.items():
        by_profile[group_profile(items)].append(group)
    selected = set()
    for profile, profile_groups in sorted(by_profile.items(), key=lambda item: profile_key(item[0])):
        ordered = sorted(
            profile_groups,
            key=lambda group: (stable_token("confirmation", group, seed), group),
        )
        take = int(math.floor(len(ordered) * ratio))
        selected.update(ordered[:take])
    profile_totals = {
        profile: sum(len(groups[group]) for group in profile_groups)
        for profile, profile_groups in by_profile.items()
    }
    profile_state = {
        "groups": groups,
        "profile_image_counts": profile_totals,
    }

    def score(candidate):
        count = sum(len(groups[group]) for group in candidate)
        return (
            abs(count - target_count),
            profile_penalty(candidate, profile_state, ratio),
        )

    while True:
        current_score = score(selected)
        best_score = current_score
        best_selected = None
        best_token = None
        for group in sorted(groups):
            candidate = set(selected)
            if group in candidate:
                candidate.remove(group)
                action = "remove"
            else:
                candidate.add(group)
                action = "add"
            candidate_score = score(candidate)
            token = stable_token(f"adjust-{action}", group, seed)
            if candidate_score < best_score or (
                candidate_score == best_score
                and best_selected is not None
                and token < best_token
            ):
                best_score = candidate_score
                best_selected = candidate
                best_token = token
        if best_selected is None or best_score >= current_score:
            break
        selected = best_selected

    confirmation = sorted(selected)
    development = sorted(set(groups) - selected)
    return development, confirmation


def assign_folds(development_groups, groups, fold_count, seed):
    all_names = [name for group in development_groups for name in groups[group]]
    target_count = len(all_names) / fold_count
    signatures = Counter(haze_signature(name) for name in all_names)
    target_signatures = {
        signature: count / fold_count for signature, count in signatures.items()
    }
    fold_groups = {fold: [] for fold in range(fold_count)}
    fold_counts = Counter()
    fold_signatures = {fold: Counter() for fold in range(fold_count)}
    ordered = sorted(
        development_groups,
        key=lambda group: (
            -len(groups[group]),
            stable_token("fold-order", group, seed),
            group,
        ),
    )
    for group in ordered:
        group_signatures = Counter(haze_signature(name) for name in groups[group])
        candidates = []
        rotation = int(stable_token("fold-tie", group, seed)[:8], 16) % fold_count
        for fold in range(fold_count):
            new_count = fold_counts[fold] + len(groups[group])
            count_error = abs(new_count - target_count)
            signature_error = sum(
                abs(
                    fold_signatures[fold][signature]
                    + group_signatures[signature]
                    - target_signatures[signature]
                )
                / max(1.0, target_signatures[signature])
                for signature in signatures
            )
            tie_rank = (fold - rotation) % fold_count
            candidates.append((count_error + signature_error, count_error, tie_rank, fold))
        fold = min(candidates)[-1]
        fold_groups[fold].append(group)
        fold_counts[fold] += len(groups[group])
        fold_signatures[fold].update(group_signatures)
    return {fold: sorted(items) for fold, items in fold_groups.items()}


def build_ledger(train_inner, val_inner, historical_names, seed, confirmation_target, fold_count):
    train_set = set(train_inner)
    val_set = set(val_inner)
    historical_set = set(historical_names)
    eligible = sorted(train_set - historical_set)
    groups = group_names(eligible)
    development_groups, confirmation_groups = select_confirmation(
        groups, confirmation_target, seed
    )
    folds = assign_folds(development_groups, groups, fold_count, seed)
    development_names = sorted(
        name for group in development_groups for name in groups[group]
    )
    confirmation_names = sorted(
        name for group in confirmation_groups for name in groups[group]
    )
    fold_names = {
        str(fold): sorted(name for group in fold_groups for name in groups[group])
        for fold, fold_groups in folds.items()
    }
    ledger = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "seed": seed,
        "confirmation_target": confirmation_target,
        "fold_count": fold_count,
        "group_rule": "filename stem before first underscore",
        "haze_signature_rule": "filename stem after first underscore",
        "roles": {
            "development": development_names,
            "confirmation": confirmation_names,
        },
        "development_folds": fold_names,
        "groups": {
            "development": development_groups,
            "confirmation": confirmation_groups,
        },
    }
    return ledger, groups, train_set, val_set, historical_set


def write_csv(path, rows, fieldnames):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_names(split_path, v3p_path):
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if not isinstance(split, dict) or not isinstance(split.get("splits"), dict):
        raise ValueError("v1 split has no splits object")
    train_inner = split["splits"].get("train_inner")
    val_inner = split["splits"].get("val_inner")
    if not isinstance(train_inner, list) or not all(isinstance(item, str) for item in train_inner):
        raise ValueError("v1 train_inner is invalid")
    if not isinstance(val_inner, list) or not all(isinstance(item, str) for item in val_inner):
        raise ValueError("v1 val_inner is invalid")

    rows_by_name = defaultdict(list)
    row_count = 0
    with Path(v3p_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "clean_reference_group", "operator_label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("v3p image table is missing identity columns")
        name_index = reader.fieldnames.index("name")
        group_index = reader.fieldnames.index("clean_reference_group")
        operator_index = reader.fieldnames.index("operator_label")
    with Path(v3p_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            row_count += 1
            name = row[name_index]
            group = row[group_index]
            operator = row[operator_index]
            if clean_reference_group(name) != group:
                raise ValueError(f"v3p group mismatch for {name}")
            rows_by_name[name].append(operator)
    operators_valid = all(
        len(operators) == len(EXPECTED_OPERATORS)
        and set(operators) == EXPECTED_OPERATORS
        for operators in rows_by_name.values()
    )
    return sorted(train_inner), sorted(val_inner), rows_by_name, row_count, operators_valid


def compact_outputs(ledger, groups, train_set, val_set, historical_set, row_count, operators_valid):
    development = set(ledger["roles"]["development"])
    confirmation = set(ledger["roles"]["confirmation"])
    all_eligible = development | confirmation
    development_groups = set(ledger["groups"]["development"])
    confirmation_groups = set(ledger["groups"]["confirmation"])
    fold_sets = {
        fold: set(names) for fold, names in ledger["development_folds"].items()
    }
    max_group_size = max(len(items) for items in groups.values())
    fold_counts = {fold: len(names) for fold, names in fold_sets.items()}
    signatures = Counter(haze_signature(name) for name in all_eligible)
    confirmation_signatures = Counter(haze_signature(name) for name in confirmation)
    confirmation_ratio = len(confirmation) / max(1, len(all_eligible))
    max_signature_per_group = {
        signature: max(
            Counter(haze_signature(name) for name in names)[signature]
            for names in groups.values()
        )
        for signature in signatures
    }
    signature_rows = []
    signature_balance_pass = True
    for signature in sorted(signatures):
        expected = signatures[signature] * confirmation_ratio
        deviation = abs(confirmation_signatures[signature] - expected)
        tolerance = max(1, max_signature_per_group[signature])
        passed = deviation <= tolerance + 1e-12
        signature_balance_pass = signature_balance_pass and passed
        signature_rows.append(
            {
                "haze_signature": signature,
                "eligible_count": signatures[signature],
                "confirmation_count": confirmation_signatures[signature],
                "expected_confirmation_count": f"{expected:.12f}",
                "absolute_deviation": f"{deviation:.12f}",
                "group_tolerance": tolerance,
                "pass": str(passed).lower(),
            }
        )
    fold_union = set().union(*fold_sets.values()) if fold_sets else set()
    fold_pair_overlap = sum(
        len(fold_sets[left] & fold_sets[right])
        for left in fold_sets
        for right in fold_sets
        if left < right
    )
    count_contract = {
        "train_inner_2400": len(train_set) == EXPECTED_TRAIN_INNER,
        "val_inner_600": len(val_set) == EXPECTED_VAL_INNER,
        "train_val_overlap_zero": not train_set & val_set,
        "historical_1200": len(historical_set) == EXPECTED_HISTORICAL,
        "historical_inside_train": historical_set <= train_set,
        "eligible_1200": len(all_eligible) == EXPECTED_ELIGIBLE,
        "eligible_exact_difference": all_eligible == train_set - historical_set,
        "development_confirmation_overlap_zero": not development & confirmation,
        "development_confirmation_group_overlap_zero": not development_groups
        & confirmation_groups,
        "confirmation_near_target": abs(
            len(confirmation) - ledger["confirmation_target"]
        )
        <= max_group_size,
        "fold_count_exact": len(fold_sets) == ledger["fold_count"],
        "fold_pair_overlap_zero": fold_pair_overlap == 0,
        "fold_union_exact_development": fold_union == development,
        "fold_count_balance": max(fold_counts.values()) - min(fold_counts.values())
        <= max_group_size,
        "paired_v3p_operator_rows": operators_valid
        and row_count == EXPECTED_HISTORICAL * len(EXPECTED_OPERATORS),
        "haze_signature_balance": signature_balance_pass,
    }
    structural_valid = all(count_contract.values())
    role_rows = [
        {
            "role": "train_inner",
            "image_count": len(train_set),
            "group_count": len(group_names(train_set)),
            "published_names": "false",
        },
        {
            "role": "historical_v3p",
            "image_count": len(historical_set),
            "group_count": len(group_names(historical_set)),
            "published_names": "false",
        },
        {
            "role": "r3_development",
            "image_count": len(development),
            "group_count": len(development_groups),
            "published_names": "false",
        },
        {
            "role": "r3_confirmation",
            "image_count": len(confirmation),
            "group_count": len(confirmation_groups),
            "published_names": "false",
        },
        {
            "role": "val_inner_historical",
            "image_count": len(val_set),
            "group_count": len(group_names(val_set)),
            "published_names": "false",
        },
    ]
    fold_rows = [
        {
            "fold": fold,
            "image_count": len(names),
            "group_count": len(
                {clean_reference_group(name) for name in names}
            ),
            "names_sha256": sha256_value(sorted(names)),
        }
        for fold, names in sorted(fold_sets.items())
    ]
    summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "state": "COMPLETED_GATE_PASS"
        if structural_valid
        else "COMPLETED_GATE_FAIL",
        "decision": "R3_S0_LEDGER_FREEZE_PASS"
        if structural_valid
        else "R3_S0_LEDGER_FREEZE_FAIL_STOP",
        "authorizes": "R3_A0_GT_FREE_PROPOSAL_ORACLE"
        if structural_valid
        else "NONE",
        "structural_valid": structural_valid,
        "checks": count_contract,
        "counts": {
            "train_inner": len(train_set),
            "val_inner": len(val_set),
            "historical_v3p": len(historical_set),
            "eligible": len(all_eligible),
            "development": len(development),
            "confirmation": len(confirmation),
            "confirmation_target": ledger["confirmation_target"],
            "max_group_size": max_group_size,
            "v3p_identity_rows": row_count,
        },
        "group_counts": {
            "eligible": len(groups),
            "development": len(development_groups),
            "confirmation": len(confirmation_groups),
        },
        "fold_counts": fold_counts,
        "hashes": {
            "ledger": sha256_value(ledger),
            "development_names": sha256_value(sorted(development)),
            "confirmation_names": sha256_value(sorted(confirmation)),
            "development_groups": sha256_value(sorted(development_groups)),
            "confirmation_groups": sha256_value(sorted(confirmation_groups)),
        },
        "identity_only_v3p_columns": [
            "name",
            "clean_reference_group",
            "operator_label",
        ],
        "confirmation_images_targets_outcomes_touched": False,
        "canary_touched": False,
        "locked_test_touched": False,
        "model_calls": 0,
        "gpu_accessed": False,
        "checkpoint_accessed": False,
        "image_decoded": False,
        "gt_decoded": False,
        "training_occurred": False,
        "inference_occurred": False,
        "name_level_ledger_cloud_only": True,
    }
    return summary, role_rows, fold_rows, signature_rows


def contract(context_path):
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    train = [
        "g00_a.png",
        "g00_b.png",
        "g01_a.png",
        "g01_b.png",
        "g02_a.png",
        "g02_b.png",
        "g03_a.png",
        "g03_b.png",
        "g04_a.png",
        "g04_b.png",
        "g05_a.png",
        "g05_b.png",
    ]
    historical = train[:4]
    first = build_ledger(train, ["v00_a.png"], historical, 3407, 3, 2)[0]
    second = build_ledger(train, ["v00_a.png"], historical, 3407, 3, 2)[0]
    development = set(first["roles"]["development"])
    confirmation = set(first["roles"]["confirmation"])
    checks = {
        "contract_environment": os.environ.get("CONVIR_CONTRACT_ONLY") == "1",
        "cpu_only_contract": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "deterministic_ledger": canonical_bytes(first) == canonical_bytes(second),
        "synthetic_role_overlap_zero": not development & confirmation,
        "synthetic_partition_complete": development | confirmation
        == set(train) - set(historical),
        "workload_output_absent": not (context.output_path / "workload").exists(),
        "protected_assets_unavailable": not context.assets,
    }
    atomic_json(
        context.phase_output_path / "synthetic_summary.json",
        {
            "schema_version": 1,
            "checks": checks,
            "synthetic_ledger_sha256": sha256_value(first),
        },
    )
    write_contract_result(context, checks=checks)
    if not all(checks.values()):
        raise SystemExit("R3 S0 synthetic contract failed")


def run(context_path):
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    split_path = asset_path(context, "v1_split", kind="file")
    v3p_path = asset_path(context, "v3p_image_rows", kind="file")
    seed = int(os.environ["CONVIR_ROUTE_LEDGER_SEED"])
    confirmation_target = int(os.environ["CONVIR_ROUTE_CONFIRMATION_TARGET"])
    fold_count = int(os.environ["CONVIR_ROUTE_FOLD_COUNT"])
    train_inner, val_inner, rows_by_name, row_count, operators_valid = source_names(
        split_path, v3p_path
    )
    historical_names = sorted(rows_by_name)
    ledger, groups, train_set, val_set, historical_set = build_ledger(
        train_inner,
        val_inner,
        historical_names,
        seed,
        confirmation_target,
        fold_count,
    )
    summary, role_rows, fold_rows, signature_rows = compact_outputs(
        ledger,
        groups,
        train_set,
        val_set,
        historical_set,
        row_count,
        operators_valid,
    )
    atomic_json(context.phase_output_path / "ledger_cloud_only.json", ledger)
    atomic_json(context.phase_output_path / "s0_ledger_summary.json", summary)
    write_csv(
        context.phase_output_path / "s0_data_role_matrix.csv",
        role_rows,
        ["role", "image_count", "group_count", "published_names"],
    )
    write_csv(
        context.phase_output_path / "s0_fold_summary.csv",
        fold_rows,
        ["fold", "image_count", "group_count", "names_sha256"],
    )
    write_csv(
        context.phase_output_path / "s0_signature_balance.csv",
        signature_rows,
        [
            "haze_signature",
            "eligible_count",
            "confirmation_count",
            "expected_confirmation_count",
            "absolute_deviation",
            "group_tolerance",
            "pass",
        ],
    )
    atomic_json(
        context.phase_output_path / "s0_source_identity.json",
        {
            "schema_version": 1,
            "route_id": ROUTE_ID,
            "operation_id": OPERATION_ID,
            "route_commit": context.route_commit,
            "v1_split": {
                "path": str(split_path),
                "bytes": split_path.stat().st_size,
                "sha256": sha256_file(split_path),
            },
            "v3p_image_rows": {
                "path": str(v3p_path),
                "bytes": v3p_path.stat().st_size,
                "sha256": sha256_file(v3p_path),
                "read_columns": [
                    "name",
                    "clean_reference_group",
                    "operator_label",
                ],
            },
            "entrypoint_sha256": sha256_file(__file__),
        },
    )
    atomic_json(
        context.phase_output_path / "s0_access_audit.json",
        {
            "schema_version": 1,
            "route_id": ROUTE_ID,
            "operation_id": OPERATION_ID,
            "historical_development_assets_read": True,
            "historical_v3p_identity_columns_only": True,
            "dataset_directory_accessed": False,
            "image_decoded": False,
            "gt_decoded": False,
            "candidate_outcomes_used_for_assignment": False,
            "confirmation_images_targets_outcomes_touched": False,
            "canary_touched": False,
            "locked_test_touched": False,
            "checkpoint_loaded": False,
            "model_calls": 0,
            "gpu_accessed": False,
            "training_occurred": False,
            "inference_occurred": False,
        },
    )
    write_run_result(
        context,
        state=summary["state"],
        decision=summary["decision"],
        authorizes=summary["authorizes"],
        details={
            "structural_valid": summary["structural_valid"],
            "eligible_count": summary["counts"]["eligible"],
            "development_count": summary["counts"]["development"],
            "confirmation_count": summary["counts"]["confirmation"],
            "ledger_sha256": summary["hashes"]["ledger"],
            "model_calls": 0,
            "gpu_accessed": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "contract":
        contract(args.context)
    else:
        run(args.context)


if __name__ == "__main__":
    main()
