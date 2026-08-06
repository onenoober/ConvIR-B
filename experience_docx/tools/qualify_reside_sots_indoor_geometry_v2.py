#!/usr/bin/env python3
"""Model-free geometry qualification for the exact RESIDE SOTS Indoor asset."""

from __future__ import annotations

import argparse
import hashlib
import json
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
EXPECTED_V1_ROUTE = "reside-sots-indoor-asset-qualification-v1"
EXPECTED_V1_SUMMARY_SHA256 = (
    "f08eb4738ba13b7f20256a5e95fade8834f44842e0d5dfa4df35b1fc199527ea"
)
EXPECTED_V1_CLOSEOUT_SHA256 = (
    "fb9cca354bbbd2987b0a4c415f518fe1456d1443fa9f54e03418643c361f58c1"
)
EXPECTED_GT_SHA256 = (
    "f87015ecff62c83a7286017fb1641589df6445556e35737196d81e4fb7dda950"
)
EXPECTED_HAZY_SHA256 = (
    "18543d375ac814208b71844748e85ca8685317da4a52403dfb172ecfeee6f3ee"
)
EXPECTED_COMBINED_SHA256 = (
    "31efa342c4224585772ff15eaad9285e1d4134f9dd3f355eb70e689e19fdbfb4"
)
MAX_CROP_PER_SIDE = 32
SUMMARY_FILENAME = "reside_sots_indoor_geometry_qualification_v2_summary.json"
REVIEW_FACTS_FILENAME = (
    "reside_sots_indoor_geometry_qualification_v2_review_facts.json"
)


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


def digest_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8") + b"\n")
    return digest.hexdigest()


def image_metadata(path: Path) -> tuple[tuple[int, int], str, str]:
    with Image.open(path) as image:
        size = tuple(int(value) for value in image.size)
        image_format = str(image.format or "unknown")
        mode = str(image.mode)
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("image metadata has an invalid size")
    return size, image_format, mode


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def v1_evidence_matches(summary: dict[str, Any], closeout: dict[str, Any]) -> bool:
    aggregate = summary.get("aggregate_identity", {})
    qualification = summary.get("qualification", {})
    decode_profile = summary.get("decode_profile", {})
    return (
        summary.get("route_id") == EXPECTED_V1_ROUTE
        and summary.get("marker")
        == "RESIDE_SOTS_INDOOR_ASSET_QUALIFICATION_V1_COMPLETE"
        and aggregate.get("gt_sha256") == EXPECTED_GT_SHA256
        and aggregate.get("hazy_sha256") == EXPECTED_HAZY_SHA256
        and aggregate.get("combined_sha256") == EXPECTED_COMBINED_SHA256
        and aggregate.get("file_count") == EXPECTED_DECODED
        and qualification.get("observed_gt_images") == EXPECTED_GT
        and qualification.get("observed_hazy_images") == EXPECTED_HAZY
        and qualification.get("decoded_images") == EXPECTED_DECODED
        and qualification.get("decode_failure_count") == 0
        and qualification.get("dimension_mismatch_count") == EXPECTED_HAZY
        and decode_profile.get("format_histogram") == {"PNG": EXPECTED_DECODED}
        and decode_profile.get("mode_histogram") == {"RGB": EXPECTED_DECODED}
        and closeout.get("route_id") == EXPECTED_V1_ROUTE
        and closeout.get("state") == "COMPLETED_INCONCLUSIVE"
        and closeout.get("decision")
        == "RESIDE_SOTS_INDOOR_ASSET_QUALIFICATION_INCONCLUSIVE"
        and closeout.get("authorizes") == "ASSET_REPAIR_REVIEW_ONLY"
        and closeout.get("evidence_sha256", {}).get(
            "reside_sots_indoor_asset_qualification_v1_summary.json"
        )
        == EXPECTED_V1_SUMMARY_SHA256
    )


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    v1_summary = load_json(asset_path(context, "v1_summary", kind="file"))
    v1_closeout = load_json(asset_path(context, "v1_closeout", kind="file"))
    write_contract_progress(
        context,
        completed_iterations=1,
        total_iterations=1,
        stage="geometry_metadata_contract",
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
            "v1_evidence_available": v1_evidence_matches(v1_summary, v1_closeout),
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
        raise RuntimeError("SOTS Indoor geometry runtime contract mismatch")
    if any(context.protected_data_permissions.values()):
        raise RuntimeError("SOTS Indoor geometry qualification forbids protected data")
    if load_completed_unit_ledger(context):
        raise RuntimeError("fresh geometry qualification has completed units")

    v1_summary_path = asset_path(context, "v1_summary", kind="file")
    v1_closeout_path = asset_path(context, "v1_closeout", kind="file")
    v1_summary = load_json(v1_summary_path)
    v1_closeout = load_json(v1_closeout_path)
    v1_evidence_ok = (
        sha256_file(v1_summary_path) == EXPECTED_V1_SUMMARY_SHA256
        and sha256_file(v1_closeout_path) == EXPECTED_V1_CLOSEOUT_SHA256
        and v1_evidence_matches(v1_summary, v1_closeout)
    )

    sots_root = asset_path(context, "sots_root", kind="directory")
    indoor_root = sots_root / "indoor"
    gt_root = indoor_root / "gt"
    hazy_root = indoor_root / "hazy"
    gt_files, unexpected_gt_images = direct_png_files(gt_root)
    hazy_files, unexpected_hazy_images = direct_png_files(hazy_root)

    gt_by_stem = {path.stem: path for path in gt_files}
    duplicate_gt_stems = len(gt_by_stem) != len(gt_files)
    hazy_by_source: dict[str, list[Path]] = defaultdict(list)
    unmapped_hazy: list[str] = []
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
    metadata_failures: list[str] = []
    for index, path in enumerate(all_files, start=1):
        identities[path] = sha256_file(path)
        try:
            size, image_format, mode = image_metadata(path)
            dimensions[path] = size
            formats[image_format] += 1
            modes[mode] += 1
        except Exception:
            metadata_failures.append(path.relative_to(indoor_root).as_posix())
        if index % 50 == 0:
            write_workload_progress(
                context,
                completed_units=0,
                stage=f"geometry_headers_checked_{index}_of_{len(all_files)}",
            )

    gt_digest = aggregate_digest(indoor_root, gt_files, identities)
    hazy_digest = aggregate_digest(indoor_root, hazy_files, identities)
    combined_digest = aggregate_digest(indoor_root, all_files, identities)
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
    pairing_ok = (
        len(gt_files) == EXPECTED_GT
        and len(hazy_files) == EXPECTED_HAZY
        and not duplicate_gt_stems
        and not unmapped_hazy
        and not unused_gt
        and len(hazy_by_source) == EXPECTED_GT
        and variant_counts == Counter({EXPECTED_VARIANTS: EXPECTED_GT})
    )
    current_identity_ok = (
        exact_root
        and pairing_ok
        and unexpected_gt_images == 0
        and unexpected_hazy_images == 0
        and gt_digest == EXPECTED_GT_SHA256
        and hazy_digest == EXPECTED_HAZY_SHA256
        and combined_digest == EXPECTED_COMBINED_SHA256
    )

    clear_size_histogram: Counter[str] = Counter()
    hazy_size_histogram: Counter[str] = Counter()
    crop_delta_histogram: Counter[str] = Counter()
    per_side_crop_histogram: Counter[str] = Counter()
    pair_metadata_count = 0
    locally_compatible_pairs = 0
    nonnegative_delta_pairs = 0
    even_delta_pairs = 0
    positive_delta_pairs = 0
    bounded_crop_pairs = 0
    max_per_side_crop_pixels = 0.0

    for source, paths in hazy_by_source.items():
        clear_path = gt_by_stem[source]
        clear_size = dimensions.get(clear_path)
        if clear_size is not None:
            clear_size_histogram[f"{clear_size[0]}x{clear_size[1]}"] += 1
        for hazy_path in paths:
            hazy_size = dimensions.get(hazy_path)
            if hazy_size is not None:
                hazy_size_histogram[f"{hazy_size[0]}x{hazy_size[1]}"] += 1
            if clear_size is None or hazy_size is None:
                continue
            pair_metadata_count += 1
            delta_width = clear_size[0] - hazy_size[0]
            delta_height = clear_size[1] - hazy_size[1]
            crop_delta_histogram[f"{delta_width}x{delta_height}"] += 1
            max_per_side_crop_pixels = max(
                max_per_side_crop_pixels,
                abs(delta_width) / 2.0,
                abs(delta_height) / 2.0,
            )
            nonnegative = delta_width >= 0 and delta_height >= 0
            even = delta_width % 2 == 0 and delta_height % 2 == 0
            positive = delta_width > 0 or delta_height > 0
            bounded = (
                nonnegative
                and even
                and delta_width // 2 <= MAX_CROP_PER_SIDE
                and delta_height // 2 <= MAX_CROP_PER_SIDE
            )
            if nonnegative:
                nonnegative_delta_pairs += 1
            if even:
                even_delta_pairs += 1
            if positive:
                positive_delta_pairs += 1
            if bounded:
                bounded_crop_pairs += 1
                per_side_crop_histogram[
                    f"{delta_width // 2}x{delta_height // 2}"
                ] += 1
            if nonnegative and even and positive and bounded:
                locally_compatible_pairs += 1

    metadata_ok = (
        len(dimensions) == EXPECTED_DECODED
        and not metadata_failures
        and formats == Counter({"PNG": EXPECTED_DECODED})
        and modes == Counter({"RGB": EXPECTED_DECODED})
        and pair_metadata_count == EXPECTED_HAZY
    )
    uniform_crop_delta = len(crop_delta_histogram) == 1
    geometry_ok = (
        current_identity_ok
        and metadata_ok
        and uniform_crop_delta
        and nonnegative_delta_pairs == EXPECTED_HAZY
        and even_delta_pairs == EXPECTED_HAZY
        and positive_delta_pairs == EXPECTED_HAZY
        and bounded_crop_pairs == EXPECTED_HAZY
        and locally_compatible_pairs == EXPECTED_HAZY
    )
    geometry_outcome = (
        "invalid"
        if not current_identity_ok or not metadata_ok
        else "favorable" if geometry_ok else "unfavorable"
    )
    forbidden_operation_count = 0
    gate_outcomes = {
        "v1_evidence_identity": "pass" if v1_evidence_ok else "fail",
        "current_asset_identity": "pass" if current_identity_ok else "fail",
        "geometry_compatibility": geometry_outcome,
        "isolation": "safe" if forbidden_operation_count == 0 else "unsafe",
    }
    geometry_qualified_pairs = EXPECTED_HAZY if geometry_ok else 0
    geometry_qualified_clear_scenes = EXPECTED_GT if geometry_ok else 0

    summary: dict[str, Any] = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "model-free RESIDE SOTS Indoor bounded center-crop geometry qualification",
        "target_root": str(indoor_root),
        "v1_evidence_binding": {
            "terminal_record_sha256": "f2d26c3af6fbfb4a294c6e6b1e745d87f6b69b6e66ced4e15936fd81ee12d460",
            "summary_sha256": EXPECTED_V1_SUMMARY_SHA256,
            "closeout_sha256": EXPECTED_V1_CLOSEOUT_SHA256,
            "evidence_matches": v1_evidence_ok,
        },
        "aggregate_identity": {
            "method": "SHA-256 over sorted UTF-8 records: relative_path TAB byte_size TAB file_sha256 LF",
            "gt_sha256": gt_digest,
            "hazy_sha256": hazy_digest,
            "combined_sha256": combined_digest,
            "file_count": len(all_files),
            "matches_v1": current_identity_ok,
        },
        "pairing": {
            "observed_gt_images": len(gt_files),
            "observed_hazy_images": len(hazy_files),
            "observed_clear_source_groups": len(hazy_by_source),
            "variant_count_histogram": {
                str(count): scenes for count, scenes in sorted(variant_counts.items())
            },
            "unmapped_hazy_count": len(unmapped_hazy),
            "unused_gt_count": len(unused_gt),
            "unexpected_image_entry_count": (
                unexpected_gt_images + unexpected_hazy_images
            ),
        },
        "geometry": {
            "expected_pair_count": EXPECTED_HAZY,
            "expected_clear_scene_count": EXPECTED_GT,
            "pair_metadata_count": pair_metadata_count,
            "locally_compatible_pair_count": locally_compatible_pairs,
            "geometry_qualified_pair_count": geometry_qualified_pairs,
            "geometry_qualified_clear_scene_count": (
                geometry_qualified_clear_scenes
            ),
            "nonnegative_delta_pair_count": nonnegative_delta_pairs,
            "even_delta_pair_count": even_delta_pairs,
            "positive_delta_pair_count": positive_delta_pairs,
            "bounded_crop_pair_count": bounded_crop_pairs,
            "crop_delta_class_count": len(crop_delta_histogram),
            "expected_crop_delta_class_count": 1,
            "uniform_crop_delta": uniform_crop_delta,
            "max_crop_per_side_limit_pixels": MAX_CROP_PER_SIDE,
            "max_per_side_crop_pixels": max_per_side_crop_pixels,
            "clear_size_histogram": dict(sorted(clear_size_histogram.items())),
            "hazy_size_histogram": dict(sorted(hazy_size_histogram.items())),
            "crop_delta_histogram": dict(sorted(crop_delta_histogram.items())),
            "per_side_crop_histogram": dict(
                sorted(per_side_crop_histogram.items())
            ),
        },
        "metadata_profile": {
            "metadata_failure_count": len(metadata_failures),
            "metadata_failure_name_digest_sha256": digest_strings(metadata_failures),
            "format_histogram": dict(sorted(formats.items())),
            "mode_histogram": dict(sorted(modes.items())),
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
            "This route qualifies only exact asset identity and bounded symmetric center-crop dimension compatibility.",
            "It does not compare pixel content, prove pixel registration, or authorize a restoration-quality claim.",
            "No model, checkpoint, inference, training, restoration metric, image display, or protected-data access occurs.",
        ],
        "marker": "RESIDE_SOTS_INDOOR_GEOMETRY_QUALIFICATION_V2_COMPLETE",
    }
    summary_path = output_file(context, SUMMARY_FILENAME)
    atomic_json(summary_path, summary)
    summary_sha256 = hashlib.sha256(summary_path.read_bytes()).hexdigest()

    def fact(
        fact_id: str,
        metric: str,
        unit: str,
        point: int | float,
        point_pointer: str,
        threshold: int | float,
        threshold_operator: str,
        threshold_pointer: str,
        gate_id: str,
    ) -> dict[str, Any]:
        return {
            "fact_id": fact_id,
            "claim_id": "sots_indoor_geometry_qualification",
            "metric": metric,
            "unit": unit,
            "population": "all 500 mapped SOTS Indoor hazy-clear pairs",
            "grouping": "hazy-clear pair",
            "point": point,
            "ci_lower": None,
            "ci_upper": None,
            "confidence_level": None,
            "threshold": threshold,
            "threshold_operator": threshold_operator,
            "gate_outcome": gate_outcomes[gate_id],
            "source_filename": SUMMARY_FILENAME,
            "source_sha256": summary_sha256,
            "json_pointers": {
                "point": point_pointer,
                "ci_lower": None,
                "ci_upper": None,
                "confidence_level": None,
                "threshold": threshold_pointer,
                "gate_outcome": f"/gate_outcomes/{gate_id}",
            },
        }

    write_scientific_review_facts(
        context,
        relpath=REVIEW_FACTS_FILENAME,
        facts=[
            fact(
                "geometry_qualified_clear_scene_count",
                "clear scenes whose ten variants satisfy the complete dataset-wide geometry contract",
                "clear scenes",
                geometry_qualified_clear_scenes,
                "/geometry/geometry_qualified_clear_scene_count",
                EXPECTED_GT,
                "==",
                "/geometry/expected_clear_scene_count",
                "geometry_compatibility",
            ),
            fact(
                "geometry_qualified_pair_count",
                "pairs satisfying the complete bounded center-crop geometry contract",
                "pairs",
                geometry_qualified_pairs,
                "/geometry/geometry_qualified_pair_count",
                EXPECTED_HAZY,
                "==",
                "/geometry/expected_pair_count",
                "geometry_compatibility",
            ),
            fact(
                "crop_delta_class_count",
                "distinct clear-minus-hazy dimension delta classes",
                "classes",
                len(crop_delta_histogram),
                "/geometry/crop_delta_class_count",
                1,
                "==",
                "/geometry/expected_crop_delta_class_count",
                "geometry_compatibility",
            ),
            fact(
                "max_per_side_crop_pixels",
                "maximum implied symmetric crop on any image side",
                "pixels",
                max_per_side_crop_pixels,
                "/geometry/max_per_side_crop_pixels",
                MAX_CROP_PER_SIDE,
                "<=",
                "/geometry/max_crop_per_side_limit_pixels",
                "geometry_compatibility",
            ),
            fact(
                "forbidden_operation_count",
                "model, metric, display, or protected-data operation count",
                "operations",
                forbidden_operation_count,
                "/isolation/forbidden_operation_count",
                0,
                "==",
                "/isolation/forbidden_operation_limit",
                "isolation",
            ),
        ],
    )
    record_completed_unit(
        context,
        unit_id="sots_indoor_geometry_qualification",
        input_sha256=combined_digest,
        output_relpath=SUMMARY_FILENAME,
    )
    write_workload_progress(
        context,
        completed_units=1,
        stage="sots_indoor_geometry_qualified",
    )
    write_gate_result(
        context,
        gate_outcomes=gate_outcomes,
        details={
            "summary_file": SUMMARY_FILENAME,
            "combined_asset_sha256": combined_digest,
            "v1_summary_sha256": EXPECTED_V1_SUMMARY_SHA256,
            "v1_closeout_sha256": EXPECTED_V1_CLOSEOUT_SHA256,
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
