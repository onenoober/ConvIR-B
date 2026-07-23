#!/usr/bin/env python3
"""Freeze checkpoint and SOTS-Outdoor identities without model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from route_program_api import (
    asset_path,
    atomic_json,
    load_context,
    output_file,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
    write_workload_progress,
)


EXPECTED_GT = 492
EXPECTED_HAZY = 500
TOTAL_UNITS = 993
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def image_files(path: Path) -> list[Path]:
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def aggregate_digest(root: Path, files: list[Path], context, completed: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    for index, path in enumerate(files, start=1):
        identity = sha256_file(path)
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\t{path.stat().st_size}\t{identity}\n".encode("utf-8"))
        completed += 1
        if index % 25 == 0:
            write_workload_progress(context, completed_units=completed, stage="dataset_hash")
    return digest.hexdigest(), completed


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    checks = {
        "metadata_only_contract": True,
        "protected_permissions_disabled": not any(context.protected_data_permissions.values()),
        "no_scientific_asset_exposed": not any(asset.contract_access for asset in context.assets.values()),
    }
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "metadata_only",
            "device": context.device,
            "fixture": None,
            "production_path_exercised": False,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)

    checkpoint = asset_path(context, "checkpoint_root", kind="directory") / "ots-base.pkl"
    reside = asset_path(context, "reside_root", kind="directory")
    sots = reside / "official" / "SOTS" / "outdoor"
    gt_dir = sots / "gt"
    hazy_dir = sots / "hazy"

    checkpoint_exists = checkpoint.is_file() and checkpoint.stat().st_size > 0
    checkpoint_sha = sha256_file(checkpoint) if checkpoint_exists else None
    checkpoint_bytes = checkpoint.stat().st_size if checkpoint_exists else 0
    write_workload_progress(context, completed_units=1, stage="checkpoint_hash")

    gt = image_files(gt_dir) if gt_dir.is_dir() else []
    hazy = image_files(hazy_dir) if hazy_dir.is_dir() else []
    gt_by_stem = {path.stem: path for path in gt}
    duplicate_gt_stems = len(gt_by_stem) != len(gt)
    source_counts = Counter()
    unmapped_hazy = []
    for path in hazy:
        source = path.stem.split("_", 1)[0]
        if source not in gt_by_stem:
            unmapped_hazy.append(path.name)
        else:
            source_counts[source] += 1
    unused_gt = sorted(set(gt_by_stem) - set(source_counts))

    ordered = sorted(gt + hazy, key=lambda path: path.relative_to(sots).as_posix())
    dataset_sha, completed = aggregate_digest(sots, ordered, context, 1)
    write_workload_progress(context, completed_units=TOTAL_UNITS, stage="asset_audit_finalize")

    checks = {
        "checkpoint_identity": bool(
            checkpoint_exists and checkpoint_sha is not None and len(checkpoint_sha) == 64
        ),
        "sots_counts": len(gt) == EXPECTED_GT and len(hazy) == EXPECTED_HAZY,
        "source_pairing": bool(
            not duplicate_gt_stems
            and not unmapped_hazy
            and not unused_gt
            and len(source_counts) == EXPECTED_GT
            and sum(source_counts.values()) == EXPECTED_HAZY
        ),
        "aggregate_digest": len(dataset_sha) == 64 and completed == TOTAL_UNITS,
    }
    passed = all(checks.values())
    if passed:
        state = "COMPLETED_GATE_PASS"
        decision = "SOTS_OTS_ASSET_AUDIT_PASS"
        authorizes = "SOTS_OTS_LOCAL_ERROR_MEASUREMENT_CONTRACT"
        reasons = ["checkpoint identity, exact SOTS counts, complete pairing and aggregate digest passed"]
    else:
        state = "COMPLETED_GATE_FAIL"
        decision = "SOTS_OTS_ASSET_AUDIT_FAIL"
        authorizes = "NONE"
        reasons = [name for name, value in checks.items() if not value]

    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "checks": checks,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha,
        },
        "sots_outdoor": {
            "root": str(sots),
            "gt_images": len(gt),
            "hazy_images": len(hazy),
            "unique_source_groups": len(source_counts),
            "variant_count_histogram": {
                str(key): value for key, value in sorted(Counter(source_counts.values()).items())
            },
            "unmapped_hazy_count": len(unmapped_hazy),
            "unused_gt_count": len(unused_gt),
            "aggregate_sha256": dataset_sha,
        },
        "terminal": {
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "gate_reasons": reasons,
        },
        "limitations": [
            "This is an engineering identity audit and contains no model-quality result.",
            "The audit does not establish scene independence from OTS training sources.",
            "No model load, inference, training or restoration metric occurred."
        ],
        "marker": "SOTS_OTS_ASSET_AUDIT_V1_COMPLETE",
    }
    atomic_json(output_file(context, "asset_audit.json"), summary)
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "asset_audit.json",
            "checkpoint_sha256": checkpoint_sha,
            "dataset_sha256": dataset_sha,
            "gt_images": len(gt),
            "hazy_images": len(hazy),
            "source_groups": len(source_counts),
            "gate_reasons": reasons,
            "model_execution_occurred": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.phase == "contract":
        contract(arguments.context)
    else:
        run(arguments.context)


if __name__ == "__main__":
    main()
