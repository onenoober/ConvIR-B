#!/usr/bin/env python3
"""Model-free identity and pairing qualification for RESIDE SOTS Indoor."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from route_program_api import (
    asset_path,
    atomic_json,
    load_completed_unit_ledger,
    load_context,
    output_file,
    prepare_phase_output,
    record_completed_unit,
    write_contract_progress,
    write_contract_result,
    write_gate_result,
    write_scientific_review_facts,
    write_workload_progress,
)


EXPECTED_ROOT = Path(
    "/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/SOTS/indoor"
)
EXPECTED_GT = 50
EXPECTED_HAZY = 500
EXPECTED_VARIANTS = 10
EXPECTED_DECODED = EXPECTED_GT + EXPECTED_HAZY
SUMMARY_FILENAME = "reside_sots_indoor_asset_qualification_v1_summary.json"
PUBLISHED_SUMMARY_FILENAME = SUMMARY_FILENAME
REVIEW_FACTS_FILENAME = "reside_sots_indoor_asset_qualification_v1_review_facts.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def direct_png_files(directory: Path) -> tuple[list[Path], int]:
    if not directory.is_dir() or directory.is_symlink():
        return [], 1
    files: list[Path] = []
    unexpected_images = 0
    for item in directory.iterdir():
        if item.is_symlink() or item.is_dir():
            unexpected_images += 1
        elif item.is_file() and item.suffix.lower() == ".png":
            files.append(item)
        elif item.is_file() and item.suffix.lower() in {
            ".bmp", ".jpeg", ".jpg", ".tif", ".tiff"
        }:
            unexpected_images += 1
    return sorted(files), unexpected_images


def aggregate_digest(
    root: Path, files: list[Path], identities: dict[Path, str]
) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(
            f"{relative}\t{path.stat().st_size}\t{identities[path]}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def decode_image(path: Path) -> tuple[tuple[int, int], str, str]:
    with Image.open(path) as image:
        image.load()
        size = tuple(int(value) for value in image.size)
        image_format = str(image.format or "unknown")
        mode = str(image.mode)
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("decoded image has an invalid size")
    return size, image_format, mode


def digest_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8") + b"\n")
    return digest.hexdigest()


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    write_contract_progress(
        context,
        completed_iterations=1,
        total_iterations=1,
        stage="asset_identity_contract",
    )
    write_contract_result(
        context,
        checks={
            "cpu_contract": context.device == "cpu",
            "metadata_only_engineering_contract": (
                context.engineering_contract.get("mode") == "metadata_only"
            ),
            "protected_permissions_disabled": not any(
                context.protected_data_permissions.values()
            ),
            "dataset_hidden_from_contract_phase": "sots_root" not in context.assets,
        },
        engineering={
            "mode": "metadata_only",
            "device": "cpu",
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
    if context.total_units != 1 or context.device != "cpu":
        raise RuntimeError("SOTS Indoor qualification runtime contract mismatch")
    if any(context.protected_data_permissions.values()):
        raise RuntimeError("SOTS Indoor qualification forbids protected permissions")
    if load_completed_unit_ledger(context):
        raise RuntimeError("fresh qualification unexpectedly has completed units")

    sots_root = asset_path(context, "sots_root", kind="directory")
    indoor_root = sots_root / "indoor"
    gt_root = indoor_root / "gt"
    hazy_root = indoor_root / "hazy"
    gt_files, unexpected_gt_images = direct_png_files(gt_root)
    hazy_files, unexpected_hazy_images = direct_png_files(hazy_root)

    gt_by_stem = {path.stem: path for path in gt_files}
    duplicate_gt_stems = len(gt_by_stem) != len(gt_files)
    hazy_by_source: dict[str, list[Path]] = defaultdict(list)
    unmapped_hazy = []
    for path in hazy_files:
        source = path.stem.split("_", 1)[0]
        if source not in gt_by_stem:
            unmapped_hazy.append(path.name)
        else:
            hazy_by_source[source].append(path)
    unused_gt = sorted(set(gt_by_stem) - set(hazy_by_source))
    variant_counts = Counter(len(paths) for paths in hazy_by_source.values())

    all_files = [*gt_files, *hazy_files]
    identities: dict[Path, str] = {}
    dimensions: dict[Path, tuple[int, int]] = {}
    formats: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    decode_failures: list[str] = []
    for index, path in enumerate(all_files, start=1):
        identities[path] = sha256_file(path)
        try:
            size, image_format, mode = decode_image(path)
            dimensions[path] = size
            formats[image_format] += 1
            modes[mode] += 1
        except Exception:
            decode_failures.append(path.relative_to(indoor_root).as_posix())
        if index % 25 == 0:
            write_workload_progress(
                context,
                completed_units=0,
                stage=f"asset_files_checked_{index}_of_{len(all_files)}",
            )

    dimension_mismatches: list[str] = []
    matched_dimensions = 0
    for source, paths in hazy_by_source.items():
        clear_path = gt_by_stem[source]
        clear_size = dimensions.get(clear_path)
        for hazy_path in paths:
            if clear_size is not None and dimensions.get(hazy_path) == clear_size:
                matched_dimensions += 1
            elif clear_size is not None and hazy_path in dimensions:
                dimension_mismatches.append(hazy_path.name)

    exact_root = (
        indoor_root == EXPECTED_ROOT
        and sots_root == EXPECTED_ROOT.parent
        and indoor_root.is_dir()
        and not indoor_root.is_symlink()
        and gt_root.is_dir()
        and not gt_root.is_symlink()
        and hazy_root.is_dir()
        and not hazy_root.is_symlink()
    )
    asset_identity_ok = (
        exact_root
        and unexpected_gt_images == 0
        and unexpected_hazy_images == 0
        and len(identities) == len(all_files)
    )
    pairing_ok = (
        len(gt_files) == EXPECTED_GT
        and len(hazy_files) == EXPECTED_HAZY
        and not duplicate_gt_stems
        and not unmapped_hazy
        and not unused_gt
        and len(hazy_by_source) == EXPECTED_GT
        and variant_counts == Counter({EXPECTED_VARIANTS: EXPECTED_GT})
    )
    decode_dimensions_ok = (
        len(dimensions) == EXPECTED_DECODED
        and not decode_failures
        and matched_dimensions == EXPECTED_HAZY
        and not dimension_mismatches
    )
    forbidden_operation_count = 0
    gate_outcomes = {
        "asset_identity": "pass" if asset_identity_ok else "fail",
        "pairing_coverage": "pass" if pairing_ok else "fail",
        "decode_dimensions": "pass" if decode_dimensions_ok else "fail",
        "isolation": "safe" if forbidden_operation_count == 0 else "unsafe",
    }

    gt_digest = aggregate_digest(indoor_root, gt_files, identities)
    hazy_digest = aggregate_digest(indoor_root, hazy_files, identities)
    combined_digest = aggregate_digest(indoor_root, all_files, identities)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "model-free RESIDE SOTS Indoor asset identity qualification",
        "target_root": str(indoor_root),
        "qualification": {
            "expected_gt_images": EXPECTED_GT,
            "observed_gt_images": len(gt_files),
            "expected_hazy_images": EXPECTED_HAZY,
            "observed_hazy_images": len(hazy_files),
            "expected_hazy_variants_per_clear": EXPECTED_VARIANTS,
            "observed_clear_source_groups": len(hazy_by_source),
            "variant_count_histogram": {
                str(count): scenes for count, scenes in sorted(variant_counts.items())
            },
            "unmapped_hazy_count": len(unmapped_hazy),
            "unused_gt_count": len(unused_gt),
            "unexpected_image_entry_count": (
                unexpected_gt_images + unexpected_hazy_images
            ),
            "expected_decoded_images": EXPECTED_DECODED,
            "decoded_images": len(dimensions),
            "decode_failure_count": len(decode_failures),
            "expected_dimension_matched_pairs": EXPECTED_HAZY,
            "dimension_matched_pairs": matched_dimensions,
            "dimension_mismatch_count": len(dimension_mismatches),
        },
        "aggregate_identity": {
            "method": "SHA-256 over sorted UTF-8 records: relative_path TAB byte_size TAB file_sha256 LF",
            "gt_sha256": gt_digest,
            "hazy_sha256": hazy_digest,
            "combined_sha256": combined_digest,
            "file_count": len(all_files),
        },
        "decode_profile": {
            "format_histogram": dict(sorted(formats.items())),
            "mode_histogram": dict(sorted(modes.items())),
            "decode_failure_name_digest_sha256": digest_strings(decode_failures),
            "dimension_mismatch_name_digest_sha256": digest_strings(
                dimension_mismatches
            ),
        },
        "isolation": {
            "model_loads": 0,
            "checkpoint_reads": 0,
            "training_calls": 0,
            "inference_calls": 0,
            "restoration_metric_calls": 0,
            "image_display_calls": 0,
            "protected_data_operations": 0,
            "forbidden_operation_count": forbidden_operation_count,
            "forbidden_operation_limit": 0,
        },
        "gate_outcomes": gate_outcomes,
        "limitations": [
            "This qualification establishes local asset identity and pairing only.",
            "Image decoding is used only for file validity and dimensions; pixels are not displayed or scored.",
            "No model-quality, restoration, promotion, or deployment conclusion is authorized.",
        ],
        "marker": "RESIDE_SOTS_INDOOR_ASSET_QUALIFICATION_V1_COMPLETE",
    }
    summary_path = output_file(context, SUMMARY_FILENAME)
    atomic_json(summary_path, summary)
    summary_sha256 = hashlib.sha256(summary_path.read_bytes()).hexdigest()

    write_scientific_review_facts(
        context,
        relpath=REVIEW_FACTS_FILENAME,
        facts=[
            {
                "fact_id": "gt_image_count",
                "claim_id": "sots_indoor_exact_counts",
                "metric": "clear PNG file count",
                "unit": "files",
                "population": "direct children of SOTS Indoor gt",
                "grouping": "file",
                "point": len(gt_files),
                "ci_lower": None,
                "ci_upper": None,
                "confidence_level": None,
                "threshold": EXPECTED_GT,
                "threshold_operator": "==",
                "gate_outcome": gate_outcomes["pairing_coverage"],
                "source_filename": PUBLISHED_SUMMARY_FILENAME,
                "source_sha256": summary_sha256,
                "json_pointers": {
                    "point": "/qualification/observed_gt_images",
                    "ci_lower": None,
                    "ci_upper": None,
                    "confidence_level": None,
                    "threshold": "/qualification/expected_gt_images",
                    "gate_outcome": "/gate_outcomes/pairing_coverage"
                }
            },
            {
                "fact_id": "hazy_image_count",
                "claim_id": "sots_indoor_exact_counts",
                "metric": "hazy PNG file count",
                "unit": "files",
                "population": "direct children of SOTS Indoor hazy",
                "grouping": "file",
                "point": len(hazy_files),
                "ci_lower": None,
                "ci_upper": None,
                "confidence_level": None,
                "threshold": EXPECTED_HAZY,
                "threshold_operator": "==",
                "gate_outcome": gate_outcomes["pairing_coverage"],
                "source_filename": PUBLISHED_SUMMARY_FILENAME,
                "source_sha256": summary_sha256,
                "json_pointers": {
                    "point": "/qualification/observed_hazy_images",
                    "ci_lower": None,
                    "ci_upper": None,
                    "confidence_level": None,
                    "threshold": "/qualification/expected_hazy_images",
                    "gate_outcome": "/gate_outcomes/pairing_coverage"
                }
            },
            {
                "fact_id": "decoded_image_count",
                "claim_id": "sots_indoor_decode_integrity",
                "metric": "fully decoded image count",
                "unit": "files",
                "population": "all SOTS Indoor gt and hazy PNG files",
                "grouping": "file",
                "point": len(dimensions),
                "ci_lower": None,
                "ci_upper": None,
                "confidence_level": None,
                "threshold": EXPECTED_DECODED,
                "threshold_operator": "==",
                "gate_outcome": gate_outcomes["decode_dimensions"],
                "source_filename": PUBLISHED_SUMMARY_FILENAME,
                "source_sha256": summary_sha256,
                "json_pointers": {
                    "point": "/qualification/decoded_images",
                    "ci_lower": None,
                    "ci_upper": None,
                    "confidence_level": None,
                    "threshold": "/qualification/expected_decoded_images",
                    "gate_outcome": "/gate_outcomes/decode_dimensions"
                }
            },
            {
                "fact_id": "dimension_matched_pair_count",
                "claim_id": "sots_indoor_pair_dimensions",
                "metric": "dimension-matched hazy-clear pair count",
                "unit": "pairs",
                "population": "all mapped SOTS Indoor hazy-clear pairs",
                "grouping": "hazy_variant",
                "point": matched_dimensions,
                "ci_lower": None,
                "ci_upper": None,
                "confidence_level": None,
                "threshold": EXPECTED_HAZY,
                "threshold_operator": "==",
                "gate_outcome": gate_outcomes["decode_dimensions"],
                "source_filename": PUBLISHED_SUMMARY_FILENAME,
                "source_sha256": summary_sha256,
                "json_pointers": {
                    "point": "/qualification/dimension_matched_pairs",
                    "ci_lower": None,
                    "ci_upper": None,
                    "confidence_level": None,
                    "threshold": "/qualification/expected_dimension_matched_pairs",
                    "gate_outcome": "/gate_outcomes/decode_dimensions"
                }
            },
            {
                "fact_id": "forbidden_operation_count",
                "claim_id": "model_free_asset_qualification",
                "metric": "forbidden operation count",
                "unit": "operations",
                "population": "one SOTS Indoor asset qualification run",
                "grouping": "run",
                "point": forbidden_operation_count,
                "ci_lower": None,
                "ci_upper": None,
                "confidence_level": None,
                "threshold": 0,
                "threshold_operator": "==",
                "gate_outcome": gate_outcomes["isolation"],
                "source_filename": PUBLISHED_SUMMARY_FILENAME,
                "source_sha256": summary_sha256,
                "json_pointers": {
                    "point": "/isolation/forbidden_operation_count",
                    "ci_lower": None,
                    "ci_upper": None,
                    "confidence_level": None,
                    "threshold": "/isolation/forbidden_operation_limit",
                    "gate_outcome": "/gate_outcomes/isolation"
                }
            }
        ],
    )
    record_completed_unit(
        context,
        unit_id="sots_indoor_asset_qualification",
        input_sha256=combined_digest,
        output_relpath=SUMMARY_FILENAME,
    )
    write_workload_progress(
        context,
        completed_units=1,
        stage="sots_indoor_asset_qualified",
    )
    write_gate_result(
        context,
        gate_outcomes=gate_outcomes,
        details={
            "summary_file": SUMMARY_FILENAME,
            "combined_asset_sha256": combined_digest,
            "model_or_checkpoint_accessed": False,
            "training_or_inference_occurred": False,
            "restoration_metrics_computed": False,
            "protected_data_touched": False,
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
