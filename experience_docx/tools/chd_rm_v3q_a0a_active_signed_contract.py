#!/usr/bin/env python3
"""Read-only v3q-A0a audit of the pinned v3p active signed-G1 contract."""

import argparse
import csv
import hashlib
import json
import math
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROUTE_ID = "haze4k_v5_chd_rm_v3q_active_signed_value_20260712"
OPERATORS = ("D_ref", "D_rep")
REQUIRED_BLOCK_COLUMNS = {
    "operator_label",
    "fold",
    "name",
    "clean_reference_group",
    "block_y",
    "block_x",
    "direct_step_energy",
    "gain_0125_to_025",
    "gain_0125_to_025_state",
}
REQUIRED_IMAGE_COLUMNS = {"operator_label", "name", "fold", "clean_reference_group"}
EXPECTED_FORMAL = {
    "D_ref": {
        "rows": 1088675,
        "zero_energy": 584680,
        "active": 503995,
        "beneficial": 293415,
        "harmful": 210558,
        "active_abstain": 22,
    },
    "D_rep": {
        "rows": 1088675,
        "zero_energy": 584680,
        "active": 503995,
        "beneficial": 293232,
        "harmful": 210755,
        "active_abstain": 8,
    },
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
        raise ValueError(f"cannot write an empty table: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def assert_columns(path, required):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"missing header: {path}")
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise RuntimeError(f"missing required columns in {path}: {missing}")
        return tuple(reader.fieldnames)


def read_selected_names(image_path, max_images):
    assert_columns(image_path, REQUIRED_IMAGE_COLUMNS)
    names = {operator: set() for operator in OPERATORS}
    groups = {operator: {} for operator in OPERATORS}
    with Path(image_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            operator = row["operator_label"]
            if operator not in names:
                raise RuntimeError(f"unexpected operator in image table: {operator}")
            name = row["name"]
            if name in groups[operator]:
                raise RuntimeError(f"duplicate image row: {operator}/{name}")
            names[operator].add(name)
            groups[operator][name] = row["clean_reference_group"]
    if names["D_ref"] != names["D_rep"]:
        raise RuntimeError("D_ref/D_rep image-name sets differ")
    ordered = sorted(names["D_ref"])
    if max_images:
        ordered = ordered[:max_images]
    if not ordered:
        raise RuntimeError("selected image set is empty")
    selected = set(ordered)
    selected_groups = {
        operator: {name: groups[operator][name] for name in selected}
        for operator in OPERATORS
    }
    return selected, selected_groups


def auc(pos_scores, neg_scores):
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.sort(np.asarray(neg_scores, dtype=np.float64))
    if not len(pos) or not len(neg):
        return float("nan")
    left = np.searchsorted(neg, pos, side="left")
    right = np.searchsorted(neg, pos, side="right")
    return float(np.mean((left + right) * 0.5) / len(neg))


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-blocks", required=True)
    parser.add_argument("--canonical-images", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--run-mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--expected-canonical-blocks-sha256", required=True)
    parser.add_argument("--expected-route-commit", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_mode == "smoke" and args.max_images != 32:
        raise ValueError("A0a smoke must use exactly 32 selected images")
    if args.run_mode == "formal" and args.max_images:
        raise ValueError("A0a formal must use all selected images")

    blocks_path = Path(args.canonical_blocks)
    images_path = Path(args.canonical_images)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    block_columns = assert_columns(blocks_path, REQUIRED_BLOCK_COLUMNS)
    image_columns = assert_columns(images_path, REQUIRED_IMAGE_COLUMNS)
    observed_hash = sha256_file(blocks_path)
    if observed_hash != args.expected_canonical_blocks_sha256:
        raise RuntimeError(
            "canonical block hash mismatch: "
            f"expected={args.expected_canonical_blocks_sha256} observed={observed_hash}"
        )

    selected_names, selected_groups = read_selected_names(images_path, args.max_images)
    expected_group_set = set(selected_groups["D_ref"].values())
    if len(expected_group_set) != len(selected_names):
        raise RuntimeError("selected OOF names do not map one-to-one to clean-reference groups")

    stats = {
        operator: {
            "rows": 0,
            "zero_energy": 0,
            "active": 0,
            "state": Counter(),
            "active_state": Counter(),
            "energy": array("d"),
            "gain": array("d"),
            "state_code": bytearray(),
            "keys": set(),
            "seen_names": set(),
            "per_image": defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        for operator in OPERATORS
    }
    state_code = {"abstain": 0, "beneficial": 1, "harmful": 2}

    with blocks_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            operator = row["operator_label"]
            if operator not in stats:
                raise RuntimeError(f"unexpected operator in block table: {operator}")
            name = row["name"]
            if name not in selected_names:
                continue
            if row["clean_reference_group"] != selected_groups[operator][name]:
                raise RuntimeError(f"clean-reference group mismatch: {operator}/{name}")
            state = row["gain_0125_to_025_state"]
            if state not in state_code:
                raise RuntimeError(f"unexpected canonical state: {state}")
            energy = float(row["direct_step_energy"])
            gain = float(row["gain_0125_to_025"])
            if energy < 0.0 or not math.isfinite(energy) or not math.isfinite(gain):
                raise RuntimeError(f"non-finite or negative value for {operator}/{name}")
            key = (name, int(row["block_y"]), int(row["block_x"]))
            data = stats[operator]
            if key in data["keys"]:
                raise RuntimeError(f"duplicate block key: {operator}/{key}")
            data["keys"].add(key)
            data["seen_names"].add(name)
            data["rows"] += 1
            data["state"][state] += 1
            image = data["per_image"][name]
            image[0] += 1
            image[1] += energy
            image[2] += gain
            image[3] += energy * energy
            image[4] += gain * gain
            image[5] += energy * gain
            if energy == 0.0:
                if state != "abstain":
                    raise RuntimeError(f"zero-energy row is not abstain: {operator}/{key}")
                data["zero_energy"] += 1
                continue
            data["active"] += 1
            data["active_state"][state] += 1
            data["energy"].append(energy)
            data["gain"].append(gain)
            data["state_code"].append(state_code[state])

    compact_rows = []
    summaries = {}
    image_summaries = {}
    for operator, data in stats.items():
        if data["seen_names"] != selected_names:
            missing = sorted(selected_names.difference(data["seen_names"]))
            raise RuntimeError(f"missing block rows for {operator}: {missing[:5]}")
        if data["state"]["abstain"] < data["zero_energy"]:
            raise RuntimeError(f"zero-energy abstain count mismatch for {operator}")

        energy = np.frombuffer(data["energy"], dtype=np.float64)
        gain = np.frombuffer(data["gain"], dtype=np.float64)
        labels = np.frombuffer(data["state_code"], dtype=np.uint8)
        if len(energy) != data["active"]:
            raise RuntimeError(f"active array count mismatch for {operator}")
        max_energy = float(energy.max())
        max_mask = energy == max_energy
        image_rows = []
        for name, values in sorted(data["per_image"].items()):
            count, sum_energy, sum_gain, sum_e2, sum_g2, sum_eg = values
            denominator = math.sqrt(max((count * sum_e2 - sum_energy**2) * (count * sum_g2 - sum_gain**2), 0.0))
            image_rows.append(
                {
                    "operator_label": operator,
                    "name": name,
                    "clean_reference_group": selected_groups[operator][name],
                    "block_count": count,
                    "sum_active_energy": sum_energy,
                    "sum_G1": sum_gain,
                    "within_image_energy_G1_pearson": (
                        (count * sum_eg - sum_energy * sum_gain) / denominator
                        if denominator > 0.0
                        else float("nan")
                    ),
                }
            )
        image_summaries[operator] = image_rows
        active_harm = labels == 2
        active_benefit = labels == 1
        summary = {
            "operator_label": operator,
            "row_count": data["rows"],
            "zero_energy_abstain_count": data["zero_energy"],
            "active_count": data["active"],
            "beneficial_active_count": int(active_benefit.sum()),
            "harmful_active_count": int(active_harm.sum()),
            "numerical_gray_active_count": int((labels == 0).sum()),
            "abstain_count": data["state"]["abstain"],
            "energy_auc_beneficial_vs_all_nonbeneficial": auc(
                energy[active_benefit],
                np.concatenate((energy[~active_benefit], np.zeros(data["zero_energy"]))),
            ),
            "energy_auc_beneficial_vs_harmful_only": auc(energy[active_benefit], energy[active_harm]),
            "pearson_energy_G1_active": pearson(energy, gain),
            "pearson_energy_abs_G1_active": pearson(energy, np.abs(gain)),
            "max_active_energy": max_energy,
            "max_energy_active_count": int(max_mask.sum()),
            "max_energy_harmful_rate": float(np.mean(active_harm[max_mask])),
            "max_energy_beneficial_rate": float(np.mean(active_benefit[max_mask])),
            "image_count": len(image_rows),
        }
        if args.run_mode == "formal":
            expected = EXPECTED_FORMAL[operator]
            observed = {
                "rows": summary["row_count"],
                "zero_energy": summary["zero_energy_abstain_count"],
                "active": summary["active_count"],
                "beneficial": summary["beneficial_active_count"],
                "harmful": summary["harmful_active_count"],
                "active_abstain": summary["numerical_gray_active_count"],
            }
            if observed != expected:
                raise RuntimeError(f"formal count mismatch for {operator}: expected={expected} observed={observed}")
        summaries[operator] = summary
        compact_rows.append(summary)

    ref_images = {row["name"]: row for row in image_summaries["D_ref"]}
    rep_images = {row["name"]: row for row in image_summaries["D_rep"]}
    paired_names = sorted(set(ref_images).intersection(rep_images))
    if len(paired_names) != len(selected_names):
        raise RuntimeError("cross-operator image pairing is incomplete")
    cross_operator = {
        "paired_image_count": len(paired_names),
        "sum_G1_pearson": pearson(
            [ref_images[name]["sum_G1"] for name in paired_names],
            [rep_images[name]["sum_G1"] for name in paired_names],
        ),
        "sum_active_energy_pearson": pearson(
            [ref_images[name]["sum_active_energy"] for name in paired_names],
            [rep_images[name]["sum_active_energy"] for name in paired_names],
        ),
        "within_image_energy_G1_pearson": pearson(
            [ref_images[name]["within_image_energy_G1_pearson"] for name in paired_names],
            [rep_images[name]["within_image_energy_G1_pearson"] for name in paired_names],
        ),
    }

    manifest = {
        "route_id": ROUTE_ID,
        "run_tag": args.run_tag,
        "run_mode": args.run_mode,
        "runnable_route_commit": args.expected_route_commit,
        "canonical_blocks": str(blocks_path),
        "canonical_blocks_sha256": observed_hash,
        "canonical_images": str(images_path),
        "selected_image_count": len(selected_names),
        "selected_clean_reference_group_count": len(expected_group_set),
        "required_block_columns": sorted(REQUIRED_BLOCK_COLUMNS),
        "forbidden_model_features": ["name", "fold", "clean_reference_group", "index", "numeric_image_id"],
        "gpu_used": False,
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    decision = (
        "V3Q_A0A_SMOKE_PASS_AUTHORIZE_FORMAL_ONLY"
        if args.run_mode == "smoke"
        else "V3Q_A0A_FORMAL_PASS_AUTHORIZE_A0B_FEATURE_CONTRACT_ONLY"
    )
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": "v3q-A0a-active-signed-contract",
        "state": "COMPLETED_GATE_PASS",
        "gate_type": "structural_integrity",
        "decision": decision,
        "metric_contract": "route card frozen label contract",
        "authorizes": "v3q-A0a-formal" if args.run_mode == "smoke" else "v3q-A0b-feature-contract-only",
        "reason": "pinned canonical source hash, active-stratum counts, one-to-one OOF grouping, and no forbidden runtime data",
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    write_json(output_dir / f"{args.run_tag}_source_manifest.json", manifest)
    write_json(output_dir / f"{args.run_tag}_summary.json", {"operators": compact_rows, "cross_operator": cross_operator})
    write_json(output_dir / f"{args.run_tag}_closeout.json", closeout)
    write_rows(output_dir / f"{args.run_tag}_by_operator.csv", compact_rows)
    print(decision)


if __name__ == "__main__":
    main()
