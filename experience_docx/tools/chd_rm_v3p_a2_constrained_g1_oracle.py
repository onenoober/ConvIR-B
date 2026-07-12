#!/usr/bin/env python3
"""v3p-A2 fixed-cap first-step hard-block oracle from canonical v3p losses."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROUTE_ID = "haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712"


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
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def psnr(sse, pixels):
    return 10.0 * math.log10(pixels / max(sse, 1e-30))


def bootstrap_lcb(values, draws, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        means[draw] = values[rng.integers(0, len(values), size=len(values))].mean()
    return float(np.quantile(means, 0.025))


def finalize_image(state, cap_fraction):
    block_count = len(state["blocks"])
    cap = int(math.floor(cap_fraction * block_count))
    eligible = [item for item in state["blocks"] if item[0] > item[1]]
    eligible.sort(key=lambda item: (-item[0], item[2], item[3]))
    selected = eligible[:cap]
    selected_coordinates = {(item[2], item[3]) for item in selected}
    fixed_sse = math.fsum(item[4] for item in state["blocks"])
    uniform_sse = math.fsum(item[5] for item in state["blocks"])
    # Blocks partition the image, so this direct hard-mosaic sum is the
    # authoritative oracle executor. The additive form is retained as a check.
    oracle_sse = math.fsum(
        item[5] if (item[2], item[3]) in selected_coordinates else item[4]
        for item in state["blocks"]
    )
    oracle_sse_additive = fixed_sse - math.fsum(item[0] for item in selected)
    pixels = state["pixels"]
    return {
        "operator_label": state["operator"],
        "fold": state["fold"],
        "name": state["name"],
        "block_count": block_count,
        "selected_block_count": len(selected),
        "selected_block_coverage": len(selected) / block_count,
        "selected_pixel_coverage": math.fsum(item[6] for item in selected) / pixels,
        "fixed_sse": fixed_sse,
        "uniform_025_sse": uniform_sse,
        "oracle_hard_sse": oracle_sse,
        "hard_mosaic_additive_abs_error": abs(oracle_sse - oracle_sse_additive),
        "fixed_psnr": psnr(fixed_sse, pixels),
        "uniform_025_psnr": psnr(uniform_sse, pixels),
        "oracle_hard_psnr": psnr(oracle_sse, pixels),
        "eligible_block_count": len(eligible),
        "selected_harmful_sse": math.fsum(max(-item[0], 0.0) for item in selected),
    }


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "closeout": output_dir / f"{args.run_tag}_closeout.json",
        "summary": output_dir / f"{args.run_tag}_summary.json",
        "images": output_dir / f"{args.run_tag}_per_image_cloud_only.csv",
        "folds": output_dir / f"{args.run_tag}_by_fold.csv",
        "source": output_dir / f"{args.run_tag}_source_manifest.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite v3p-A2 outputs")
    a1 = json.loads(Path(args.a1_closeout).read_text(encoding="utf-8"))
    if a1.get("decision") != "V3P_A1_RECONSTRUCTION_PASS_AUTHORIZE_A2_CONSTRAINED_G1_ORACLE_ONLY":
        raise ValueError("v3p-A2 requires A1r reconstruction pass")
    rows = []
    state = None

    def flush():
        nonlocal state
        if state is not None:
            rows.append(finalize_image(state, args.max_block_coverage))
            state = None

    with Path(args.canonical_blocks).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["operator_label"], row["name"])
            if state is None or key != (state["operator"], state["name"]):
                flush()
                state = {"operator": key[0], "name": key[1], "fold": int(row["fold"]), "blocks": [], "pixels": 0}
            fixed = float(row["candidate_sse_0p125"])
            upper = float(row["candidate_sse_0p25"])
            gain = fixed - upper
            epsilon = float(row["gain_0125_to_025_epsilon"])
            pixels = int(row["candidate_pixel_count"])
            state["blocks"].append((gain, epsilon, int(row["block_y"]), int(row["block_x"]), fixed, upper, pixels))
            state["pixels"] += pixels
    flush()
    if len(rows) != args.expected_images_per_operator * 2:
        raise ValueError(f"expected {args.expected_images_per_operator * 2} image rows, got {len(rows)}")
    for operator in ("D_ref", "D_rep"):
        count = sum(row["operator_label"] == operator for row in rows)
        if count != args.expected_images_per_operator:
            raise ValueError(f"expected {args.expected_images_per_operator} {operator} rows, got {count}")
    for row in rows:
        row["oracle_lift_vs_fixed_db"] = row["oracle_hard_psnr"] - row["fixed_psnr"]
        row["oracle_lift_vs_uniform_025_db"] = row["oracle_hard_psnr"] - row["uniform_025_psnr"]
    operator_rows = []
    for operator in ("D_ref", "D_rep"):
        subset = [row for row in rows if row["operator_label"] == operator]
        fixed_lifts = [row["oracle_lift_vs_fixed_db"] for row in subset]
        uniform_lifts = [row["oracle_lift_vs_uniform_025_db"] for row in subset]
        coverage = [row["selected_pixel_coverage"] for row in subset]
        severe = sum(value <= -0.2 for value in fixed_lifts)
        operator_rows.append({
            "operator_label": operator,
            "image_count": len(subset),
            "oracle_mean_lift_vs_fixed_db": float(np.mean(fixed_lifts)),
            "oracle_mean_lift_vs_fixed_lcb95_db": bootstrap_lcb(fixed_lifts, args.bootstrap_draws, args.seed),
            "oracle_mean_lift_vs_uniform_025_db": float(np.mean(uniform_lifts)),
            "oracle_mean_lift_vs_uniform_025_lcb95_db": bootstrap_lcb(uniform_lifts, args.bootstrap_draws, args.seed + 1),
            "selected_pixel_coverage_mean": float(np.mean(coverage)),
            "selected_pixel_coverage_lcb95": bootstrap_lcb(coverage, args.bootstrap_draws, args.seed + 2),
            "severe_lift_vs_fixed_count": severe,
            "selected_harmful_sse": float(math.fsum(row["selected_harmful_sse"] for row in subset)),
            "hard_mosaic_additive_abs_error_max": float(max(row["hard_mosaic_additive_abs_error"] for row in subset)),
        })
    passes = []
    for row in operator_rows:
        passes.append(
            row["oracle_mean_lift_vs_fixed_lcb95_db"] > args.minimum_lift_vs_fixed_db
            and row["oracle_mean_lift_vs_uniform_025_lcb95_db"] > args.minimum_lift_vs_uniform_025_db
            and row["selected_pixel_coverage_lcb95"] > args.minimum_coverage
            and row["severe_lift_vs_fixed_count"] == 0
            and row["selected_harmful_sse"] == 0.0
        )
    decision = "V3P_A2_CONSTRAINED_G1_ORACLE_PASS_AUTHORIZE_B0_PHYSICS_FORWARD_CONTRACT_ONLY" if all(passes) else "V3P_A2_CONSTRAINED_G1_ORACLE_FAIL_UNIFORM_FRONTIER"
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": "v3p-A2-constrained-G1-oracle",
        "state": "COMPLETED_GATE_PASS" if all(passes) else "COMPLETED_GATE_FAIL",
        "gate_type": "scientific_utility",
        "decision": decision,
        "authorizes": "v3p-B0 physics forward contract only" if all(passes) else "none",
        "reason": (
            "Both canonical operators clear every preregistered constrained-G1 utility and safety gate."
            if all(passes)
            else "At least one canonical operator does not clear the preregistered constrained-G1 utility or safety gate."
        ),
        "metric_contract": "default .125, only beneficial .125->.25 canonical G1 blocks, hard non-overlap executor, fixed 25% block cap",
        "max_block_coverage": args.max_block_coverage,
        "minimum_lift_vs_fixed_db": args.minimum_lift_vs_fixed_db,
        "minimum_lift_vs_uniform_025_db": args.minimum_lift_vs_uniform_025_db,
        "minimum_coverage": args.minimum_coverage,
        "operator_rows": operator_rows,
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    fold_rows = []
    for operator in ("D_ref", "D_rep"):
        for fold in range(5):
            subset = [row for row in rows if row["operator_label"] == operator and row["fold"] == fold]
            fold_rows.append({
                "operator_label": operator,
                "fold": fold,
                "image_count": len(subset),
                "mean_lift_vs_fixed_db": float(np.mean([row["oracle_lift_vs_fixed_db"] for row in subset])),
                "mean_lift_vs_uniform_025_db": float(np.mean([row["oracle_lift_vs_uniform_025_db"] for row in subset])),
                "mean_pixel_coverage": float(np.mean([row["selected_pixel_coverage"] for row in subset])),
            })
    source = {"route_id": ROUTE_ID, "canonical_blocks_sha256": sha256_file(args.canonical_blocks), "a1_closeout_sha256": sha256_file(args.a1_closeout), "raw_per_image_cloud_only": str(outputs["images"])}
    summary = {**closeout, "source_manifest": source}
    write_rows(outputs["images"], rows)
    write_rows(outputs["folds"], fold_rows)
    write_json(outputs["closeout"], closeout)
    write_json(outputs["summary"], summary)
    write_json(outputs["source"], source)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical_blocks", required=True)
    parser.add_argument("--a1_closeout", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_tag", default="v3p_a2")
    parser.add_argument("--expected_images_per_operator", type=int, default=1200)
    parser.add_argument("--max_block_coverage", type=float, default=0.25)
    parser.add_argument("--minimum_lift_vs_fixed_db", type=float, default=0.02)
    parser.add_argument("--minimum_lift_vs_uniform_025_db", type=float, default=0.01)
    parser.add_argument("--minimum_coverage", type=float, default=0.01)
    parser.add_argument("--bootstrap_draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
