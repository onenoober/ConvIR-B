#!/usr/bin/env python3
"""Bind the frozen ITS/OTS S1 rosters to outcome-blind paired-image evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

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
    write_workload_progress,
)


ROUTE_ID = "daytime-dehazing-reside-pairing-contract-v2"
OPERATION_ID = "DAYTIME_DEHAZING_RESIDE_PAIRING_CONTRACT_QUALIFY"
SUMMARY_NAME = "daytime_dehazing_reside_pairing_contract_v2_summary.json"
IDENTITY_NAME = "daytime_dehazing_reside_pairing_contract_v2_identity_summary.json"
REVIEW_FACTS_NAME = "daytime_dehazing_reside_pairing_contract_v2_review_facts.json"
RAW_LEDGER_NAME = "daytime_dehazing_reside_pairing_contract_v2_mapping_ledger.jsonl"

DATASETS = ("ITS", "OTS")
DEVELOPMENT_COUNTS = {"ITS": 869, "OTS": 801}
PLANNED_SCENES = {"ITS": 150, "OTS": 150}
EXPECTED_PLANNED_DIGESTS = {
    "ITS": "e61a53e819f36f0ee718d1bf4d6d54cfa5982501397a46cbaa005f9877e62882",
    "OTS": "6031862cc97f33bb2995df4cb23ad7db522cf0ed992ad9deb5c76003fc17acfb",
}
SCENE_SELECTION_SALT = "daytime-dehazing-local-restoration-need-v1|scenes"
VARIANT_SELECTION_SALT = "daytime-dehazing-local-restoration-need-v1|variants"
VARIANTS_PER_SCENE = 2
TOTAL_UNITS = 3
WORKERS = 8
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MAX_WIDTH = 4096
MAX_HEIGHT = 4096
MAX_PADDED_PIXELS = 4096 * 4096
MIN_CROP_SIDE = 256
MAX_CROP_SIDE = 512
CROP_MULTIPLE = 32

PARENT_CLOSEOUT_SHA256 = "8272c22034ad827c39ac2ff7d7c1903f39a9872f8a7a81ca854bc95b045af98b"
PARENT_SUMMARY_SHA256 = "5e3b678c8968a0c243f6bf79fe40b8ca80e860a376e475b61f90599eded0df1f"
S0_CLOSEOUT_SHA256 = "9e132828fb98615241d5e8dea0b0fecffa542397f4ff71bf686a901ca8959346"
S0_ROLE_SUMMARY_SHA256 = "d2262c8ba28c56a21b992c8f2c445d92099d7c9861f4263171c522c2efd8e7b1"
S0_LEDGER_SHA256 = "4cff8e7aecea5d8e19165ac4e725f69746342521115a3ab1f03ea1474f280960"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_lines(values: Iterable[str]) -> str:
    return sha256_text("\n".join(sorted(values)))


def deterministic_rank(values: Iterable[str], salt: str) -> list[str]:
    return sorted(values, key=lambda value: (sha256_text(f"{salt}|{value}"), value))


def image_files(directory: Path, *, recursive: bool = False) -> list[Path]:
    if not directory.is_dir():
        return []
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def canonical_rgb_digest(width: int, height: int, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(width.to_bytes(8, "big"))
    digest.update(height.to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict[str, Any]:
    try:
        file_sha256 = sha256_file(path)
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            rgb.load()
            width, height = rgb.size
            rgb_sha256 = canonical_rgb_digest(width, height, rgb.tobytes())
        return {
            "file_sha256": file_sha256,
            "rgb_sha256": rgb_sha256,
            "width": width,
            "height": height,
            "error": None,
        }
    except Exception as exc:
        return {
            "file_sha256": None,
            "rgb_sha256": None,
            "width": None,
            "height": None,
            "error": f"{type(exc).__name__}: {exc}"[:240],
        }


def inspect_paths(paths: Iterable[Path]) -> dict[Path, dict[str, Any]]:
    ordered = sorted(set(paths))
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        return dict(zip(ordered, executor.map(inspect_image, ordered, chunksize=8)))


def read_json_asset(context: Any, asset_id: str) -> dict[str, Any]:
    value = json.loads(
        asset_path(context, asset_id, kind="file").read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{asset_id} must contain a JSON object")
    return value


def read_s0_development(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    roles: dict[str, dict[str, str]] = defaultdict(dict)
    row_count = 0
    excluded_count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row_count += 1
            if (
                row.get("schema_version") != 1
                or row.get("program_id")
                != "daytime_dehazing_spatially_adaptive_restoration_v1"
                or row.get("independent_unit") != "original_clear_scene"
            ):
                raise RuntimeError("S0 ledger schema or program identity changed")
            if row.get("exclusion_reason") is not None:
                excluded_count += 1
                continue
            dataset = str(row.get("dataset"))
            if dataset not in DATASETS or row.get("role") != "development_screening":
                continue
            scene_id = str(row.get("scene_id"))
            canonical_digest = str(row.get("canonical_digest"))
            if len(canonical_digest) != 64:
                raise RuntimeError("S0 ledger contains an invalid canonical digest")
            if scene_id in roles[dataset]:
                raise RuntimeError("S0 ledger repeats a retained development scene")
            roles[dataset][scene_id] = canonical_digest
    counts = {dataset: len(roles[dataset]) for dataset in DATASETS}
    return roles, {
        "row_count": row_count,
        "excluded_count": excluded_count,
        "development_counts": counts,
        "expected_counts_match": counts == DEVELOPMENT_COUNTS,
    }


def stem_map(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in image_files(directory):
        if path.stem in result:
            return {}
        result[path.stem] = path
    return result


def match_haze_candidates(
    hazy_root: Path,
    clear_stems: Iterable[str],
) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    stems = sorted(clear_stems, key=lambda value: (-len(value), value))
    groups: dict[str, list[Path]] = defaultdict(list)
    ambiguous = 0
    unmatched = 0
    files = image_files(hazy_root, recursive=True)
    for path in files:
        matches = [
            stem
            for stem in stems
            if path.stem == stem or path.stem.startswith(f"{stem}_")
        ]
        if not matches:
            unmatched += 1
            continue
        longest = len(matches[0])
        best = [stem for stem in matches if len(stem) == longest]
        if len(best) != 1:
            ambiguous += 1
            continue
        groups[best[0]].append(path)
    return dict(groups), {
        "hazy_root_available": hazy_root.is_dir(),
        "hazy_file_count": len(files),
        "matched_file_count": sum(len(paths) for paths in groups.values()),
        "unmatched_file_count": unmatched,
        "ambiguous_file_count": ambiguous,
        "mapping_rule": "unique longest clear stem followed by an underscore boundary",
    }


def crop_side(height: int, width: int) -> int:
    bounded = min(height, width, MAX_CROP_SIDE)
    return (bounded // CROP_MULTIPLE) * CROP_MULTIPLE


def padded_size(value: int) -> int:
    return ((value + CROP_MULTIPLE - 1) // CROP_MULTIPLE) * CROP_MULTIPLE


def formation_token(clear_stem: str, hazy_stem: str) -> str:
    if hazy_stem == clear_stem:
        return ""
    return hazy_stem[len(clear_stem):].lstrip("_")


def build_dataset_contract(
    dataset: str,
    reside_root: Path,
    allowed_scenes: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    planned = deterministic_rank(
        allowed_scenes,
        f"{SCENE_SELECTION_SALT}|{dataset}",
    )[:PLANNED_SCENES[dataset]]
    planned_digest = sha256_text("\n".join(planned))

    if dataset == "ITS":
        specifications = (
            (
                "ITS_TRAIN",
                reside_root / "official/ITS/train/ITS_clear",
                reside_root / "official/ITS/train/ITS_haze",
            ),
            (
                "ITS_VALIDATION",
                reside_root / "official/ITS/val/clear",
                reside_root / "official/ITS/val/haze",
            ),
        )
    else:
        specifications = (
            (
                "OTS",
                reside_root / "official/OTS_ALPHA/clear_images",
                reside_root / "official/OTS_ALPHA/OTS",
            ),
        )

    planned_set = set(planned)
    clear_by_scene: dict[str, Path] = {}
    candidates_by_scene: dict[str, list[Path]] = {}
    namespace_summaries: dict[str, Any] = {}
    for namespace, clear_root, hazy_root in specifications:
        clears = stem_map(clear_root)
        selected_stems = {
            scene.split(":", 1)[1] if dataset == "ITS" else scene
            for scene in planned
            if (dataset != "ITS" or scene.startswith(f"{namespace}:"))
        }
        haze_groups, haze_summary = match_haze_candidates(hazy_root, selected_stems)
        namespace_summaries[namespace] = {
            "clear_root_available": clear_root.is_dir(),
            "selected_clear_count": sum(stem in clears for stem in selected_stems),
            **haze_summary,
        }
        for stem in selected_stems:
            scene = f"{namespace}:{stem}" if dataset == "ITS" else stem
            if scene not in planned_set:
                continue
            if stem in clears:
                clear_by_scene[scene] = clears[stem]
            candidates_by_scene[scene] = haze_groups.get(stem, [])

    selected_haze_by_scene: dict[str, list[Path]] = {}
    selected_paths: list[Path] = list(clear_by_scene.values())
    for scene in planned:
        candidates = sorted(
            candidates_by_scene.get(scene, []),
            key=lambda path: (
                sha256_text(
                    f"{VARIANT_SELECTION_SALT}|{dataset}|{scene}|{path.name}"
                ),
                path.name,
                path.as_posix(),
            ),
        )[:VARIANTS_PER_SCENE]
        selected_haze_by_scene[scene] = candidates
        selected_paths.extend(candidates)
    inspected = inspect_paths(selected_paths)

    rows: list[dict[str, Any]] = []
    max_width = 0
    max_height = 0
    max_padded_pixels = 0
    for scene in planned:
        clear_path = clear_by_scene.get(scene)
        variants = selected_haze_by_scene[scene]
        clear = inspected.get(clear_path) if clear_path is not None else None
        haze = [inspected[path] for path in variants]
        clear_stem = clear_path.stem if clear_path is not None else (
            scene.split(":", 1)[1] if dataset == "ITS" else scene
        )
        tokens = [formation_token(clear_stem, path.stem) for path in variants]
        all_decoded = clear is not None and not clear["error"] and all(
            not item["error"] for item in haze
        )
        canonical_match = bool(
            all_decoded and clear["rgb_sha256"] == allowed_scenes[scene]
        )
        aligned = bool(
            all_decoded
            and all(
                (item["width"], item["height"])
                == (clear["width"], clear["height"])
                for item in haze
            )
        )
        distinct_files = len(haze) == VARIANTS_PER_SCENE and len(
            {item["file_sha256"] for item in haze}
        ) == VARIANTS_PER_SCENE
        distinct_rgb = len(haze) == VARIANTS_PER_SCENE and len(
            {item["rgb_sha256"] for item in haze}
        ) == VARIANTS_PER_SCENE
        distinct_tokens = len(tokens) == VARIANTS_PER_SCENE and all(tokens) and len(
            set(tokens)
        ) == VARIANTS_PER_SCENE
        width = int(clear["width"]) if all_decoded else 0
        height = int(clear["height"]) if all_decoded else 0
        padded_pixels = padded_size(width) * padded_size(height) if all_decoded else 0
        score_side = crop_side(height, width) if all_decoded else 0
        geometry_ok = bool(
            all_decoded
            and score_side >= MIN_CROP_SIDE
            and score_side % CROP_MULTIPLE == 0
        )
        resource_ok = bool(
            all_decoded
            and width <= MAX_WIDTH
            and height <= MAX_HEIGHT
            and padded_pixels <= MAX_PADDED_PIXELS
        )
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        max_padded_pixels = max(max_padded_pixels, padded_pixels)
        rows.append({
            "schema_version": 1,
            "route_id": ROUTE_ID,
            "operation_id": OPERATION_ID,
            "dataset": dataset,
            "independent_unit": "original_clear_scene",
            "scene_id": scene,
            "content_identity": {
                "canonical_clear_rgb_sha256": clear["rgb_sha256"] if clear else None,
                "expected_s0_rgb_sha256": allowed_scenes[scene],
                "clear_file_sha256": clear["file_sha256"] if clear else None,
                "clear_relpath": clear_path.relative_to(reside_root).as_posix()
                if clear_path is not None else None,
            },
            "formation_observations": [
                {
                    "ordinal": index + 1,
                    "source_token": tokens[index],
                    "haze_file_sha256": info["file_sha256"],
                    "haze_rgb_sha256": info["rgb_sha256"],
                    "haze_relpath": path.relative_to(reside_root).as_posix(),
                }
                for index, (path, info) in enumerate(zip(variants, haze))
            ],
            "width": width or None,
            "height": height or None,
            "padded_width": padded_size(width) if width else None,
            "padded_height": padded_size(height) if height else None,
            "scoring_crop_side": score_side or None,
            "checks": {
                "clear_present_and_decoded": bool(clear and not clear["error"]),
                "canonical_clear_identity": canonical_match,
                "exactly_two_selected_haze_observations": len(haze)
                == VARIANTS_PER_SCENE,
                "haze_observations_decoded": len(haze) == VARIANTS_PER_SCENE
                and all(not item["error"] for item in haze),
                "haze_payloads_distinct": distinct_files and distinct_rgb,
                "formation_tokens_distinct": distinct_tokens,
                "clear_haze_dimensions_aligned": aligned,
                "scoring_geometry_valid": geometry_ok,
                "whole_image_resource_feasible": resource_ok,
            },
        })

    check_names = tuple(rows[0]["checks"]) if rows else ()
    check_counts = {
        check: sum(bool(row["checks"][check]) for row in rows)
        for check in check_names
    }
    roster_rows = [
        row
        for row in rows
        if row["content_identity"]["clear_relpath"] is not None
        and len(row["formation_observations"]) == VARIANTS_PER_SCENE
    ]
    roster_lines = []
    for row in rows:
        roster_lines.append(
            json.dumps(
                {
                    "dataset": row["dataset"],
                    "scene_id": row["scene_id"],
                    "content_identity": row["content_identity"],
                    "formation_observations": row["formation_observations"],
                    "width": row["width"],
                    "height": row["height"],
                    "scoring_crop_side": row["scoring_crop_side"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    summary = {
        "dataset": dataset,
        "allowed_development_scene_count": len(allowed_scenes),
        "planned_scene_count": len(planned),
        "planned_scene_digest": planned_digest,
        "expected_planned_scene_digest": EXPECTED_PLANNED_DIGESTS[dataset],
        "planned_digest_matches_parent": planned_digest
        == EXPECTED_PLANNED_DIGESTS[dataset],
        "selected_scene_count": len(roster_rows),
        "exact_planned_roster_without_backfill": len(roster_rows) == len(planned)
        and {row["scene_id"] for row in roster_rows} == set(planned),
        "mapping_ledger_slice_digest": digest_lines(roster_lines),
        "check_counts": check_counts,
        "namespace_summaries": namespace_summaries,
        "shape_envelope": {
            "max_observed_width": max_width,
            "max_observed_height": max_height,
            "max_observed_padded_pixels": max_padded_pixels,
            "max_allowed_width": MAX_WIDTH,
            "max_allowed_height": MAX_HEIGHT,
            "max_allowed_padded_pixels": MAX_PADDED_PIXELS,
            "scoring_crop_rule": "largest square multiple of 32 not exceeding 512 or either image dimension, with minimum 256",
        },
        "independence_contract": {
            "independent_unit": "original_clear_scene",
            "content_equivalence": "exact decoded canonical clear RGB digest",
            "formation_distinction": "distinct haze file SHA-256, decoded haze RGB SHA-256, and nonempty source filename token",
            "haze_observations_are_nested": True,
            "haze_observations_are_independent_replicates": False,
        },
    }
    return rows, summary


def write_once_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"write-once ledger already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def unit_input_sha256(unit_id: str, payload: dict[str, Any]) -> str:
    value = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "unit_id": unit_id,
        "scene_selection_salt": SCENE_SELECTION_SALT,
        "variant_selection_salt": VARIANT_SELECTION_SALT,
        "payload": payload,
    }
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def parent_evidence_checks(context: Any) -> dict[str, bool]:
    closeout = read_json_asset(context, "s1_v1_closeout")
    summary = read_json_asset(context, "s1_v1_summary")
    s0_closeout = read_json_asset(context, "s0_closeout")
    s0_roles = read_json_asset(context, "s0_role_summary")
    return {
        "s1_v1_typed_parent": (
            closeout.get("state") == "COMPLETED_INCONCLUSIVE"
            and closeout.get("decision")
            == "DAYTIME_DEHAZING_LOCAL_RESTORATION_NEED_INCONCLUSIVE"
            and closeout.get("authorizes")
            == "S1_MEASUREMENT_VALIDITY_OR_PRECISION_REVIEW_ONLY"
            and closeout.get("next_action_id")
            == "review_s1_measurement_validity_or_precision"
        ),
        "s1_v1_pre_inference_pairing_defect": all(
            summary.get("coverage", {}).get(dataset, {}).get("selected_scene_count") == 0
            and summary.get("coverage", {}).get(dataset, {}).get("planned_scene_count")
            == PLANNED_SCENES[dataset]
            and summary.get("coverage", {}).get(dataset, {}).get("planned_scene_digest")
            == EXPECTED_PLANNED_DIGESTS[dataset]
            for dataset in DATASETS
        ),
        "s0_parent_pass": (
            s0_closeout.get("state") == "COMPLETED_GATE_PASS"
            and s0_closeout.get("decision")
            == "DAYTIME_DEHAZING_PROGRAM_FOUNDATION_PASS"
        ),
        "s0_development_capacity": all(
            s0_roles.get("roles", {}).get("datasets", {}).get(dataset, {}).get(
                "role_counts", {}
            ).get("development_screening") == DEVELOPMENT_COUNTS[dataset]
            for dataset in DATASETS
        ),
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    write_contract_progress(
        context,
        completed_iterations=1,
        total_iterations=1,
        stage="outcome_blind_pairing_contract",
    )
    write_contract_result(
        context,
        checks={
            "cpu_contract": context.device == "cpu",
            "metadata_only_mode": context.engineering_contract.get("mode")
            == "metadata_only",
            "protected_permissions_disabled": not any(
                context.protected_data_permissions.values()
            ),
            "dataset_and_s0_ledger_hidden": all(
                asset_id not in context.assets
                for asset_id in ("reside_root", "s0_scene_role_ledger")
            ),
            "no_model_checkpoint_or_metric_asset": all(
                token not in asset_id
                for asset_id in context.assets
                for token in ("model", "checkpoint", "prediction", "metric")
            ),
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
    if context.total_units != TOTAL_UNITS or context.device != "cpu":
        raise RuntimeError("RESIDE pairing runtime contract mismatch")
    if any(context.protected_data_permissions.values()):
        raise RuntimeError("RESIDE pairing contract forbids protected permissions")
    if load_completed_unit_ledger(context):
        raise RuntimeError("fresh RESIDE pairing workload has completed units")

    evidence_checks = parent_evidence_checks(context)
    parent_relpath = "units/parent_evidence.json"
    atomic_json(output_file(context, parent_relpath), evidence_checks)
    record_completed_unit(
        context,
        unit_id="parent_evidence",
        input_sha256=unit_input_sha256("PARENT_EVIDENCE", evidence_checks),
        output_relpath=parent_relpath,
    )
    write_workload_progress(context, completed_units=1, stage="parent_evidence_bound")

    s0_ledger = asset_path(context, "s0_scene_role_ledger", kind="file")
    roles, ledger_summary = read_s0_development(s0_ledger)
    reside_root = asset_path(context, "reside_root", kind="directory")
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for completed_units, dataset in enumerate(DATASETS, start=2):
        rows, dataset_summary = build_dataset_contract(
            dataset, reside_root, roles[dataset]
        )
        rows_by_dataset[dataset] = rows
        summaries[dataset] = dataset_summary
        relpath = f"units/{dataset.lower()}_pairing_contract.json"
        atomic_json(output_file(context, relpath), dataset_summary)
        record_completed_unit(
            context,
            unit_id=f"{dataset.lower()}_pairing_contract",
            input_sha256=unit_input_sha256(dataset, dataset_summary),
            output_relpath=relpath,
        )
        write_workload_progress(
            context,
            completed_units=completed_units,
            stage=f"{dataset.lower()}_pairing_contract_complete",
        )

    mapping_rows = [
        row
        for dataset in DATASETS
        for row in rows_by_dataset[dataset]
    ]
    mapping_path = output_file(context, RAW_LEDGER_NAME)
    write_once_jsonl(mapping_path, mapping_rows)
    mapping_sha256 = sha256_file(mapping_path)

    evidence_identity_ok = all(evidence_checks.values()) and (
        sha256_file(s0_ledger) == S0_LEDGER_SHA256
    ) and ledger_summary["expected_counts_match"]
    planned_roster_ok = all(
        summaries[dataset]["planned_digest_matches_parent"]
        and summaries[dataset]["exact_planned_roster_without_backfill"]
        for dataset in DATASETS
    )
    clear_identity_ok = all(
        summaries[dataset]["check_counts"].get("canonical_clear_identity")
        == PLANNED_SCENES[dataset]
        for dataset in DATASETS
    )
    nested_pairing_ok = all(
        summaries[dataset]["check_counts"].get(
            "exactly_two_selected_haze_observations"
        ) == PLANNED_SCENES[dataset]
        and summaries[dataset]["check_counts"].get("haze_payloads_distinct")
        == PLANNED_SCENES[dataset]
        for dataset in DATASETS
    )
    separation_ok = all(
        summaries[dataset]["check_counts"].get("formation_tokens_distinct")
        == PLANNED_SCENES[dataset]
        and summaries[dataset]["check_counts"].get(
            "clear_haze_dimensions_aligned"
        ) == PLANNED_SCENES[dataset]
        for dataset in DATASETS
    )
    geometry_ok = all(
        summaries[dataset]["check_counts"].get(
            "scoring_geometry_valid"
        ) == PLANNED_SCENES[dataset]
        for dataset in DATASETS
    )
    resource_feasibility_ok = all(
        summaries[dataset]["check_counts"].get(
            "whole_image_resource_feasible"
        ) == PLANNED_SCENES[dataset]
        for dataset in DATASETS
    )
    all_validity_ok = all((
        evidence_identity_ok,
        planned_roster_ok,
        clear_identity_ok,
        nested_pairing_ok,
        separation_ok,
        geometry_ok,
    ))
    gate_outcomes = {
        "evidence_identity": "pass" if evidence_identity_ok else "fail",
        "planned_roster_coverage": "pass" if planned_roster_ok else "fail",
        "canonical_clear_identity": "pass" if clear_identity_ok else "fail",
        "nested_haze_pairing": "pass" if nested_pairing_ok else "fail",
        "formation_content_separation": "pass" if separation_ok else "fail",
        "scoring_geometry_validity": "pass" if geometry_ok else "fail",
        "whole_image_resource_feasibility": (
            "favorable" if resource_feasibility_ok else "unfavorable"
        ) if all_validity_ok else "invalid",
        "identity_contract_precision": "met" if all_validity_ok else "invalid",
    }
    valid_scene_count = sum(
        summaries[dataset]["selected_scene_count"] for dataset in DATASETS
    )
    summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "program_id": "daytime_dehazing_spatially_adaptive_restoration_v1",
        "scope": "outcome-blind ITS/OTS pairing and shape contract for revised S1 authoring",
        "gate_outcomes": gate_outcomes,
        "parent_evidence_checks": evidence_checks,
        "s0_ledger_summary": ledger_summary,
        "datasets": summaries,
        "valid_scene_count": valid_scene_count,
        "required_scene_count": sum(PLANNED_SCENES.values()),
        "mapping_ledger": {
            "filename": RAW_LEDGER_NAME,
            "rows": len(mapping_rows),
            "sha256": mapping_sha256,
            "archived_to_github": False,
        },
        "future_s1_constraints": {
            "whole_image_inference_before_scoring_crop": True,
            "scoring_crop_multiple": CROP_MULTIPLE,
            "scoring_crop_minimum": MIN_CROP_SIDE,
            "scoring_crop_maximum": MAX_CROP_SIDE,
            "independent_unit": "original_clear_scene",
            "haze_variants_nested_within_scene": True,
            "capability_profile_must_match_observed_shape_envelope": True,
        },
        "forbidden_activity_receipt": {
            "model_or_checkpoint_accessed": False,
            "training_or_inference_occurred": False,
            "restoration_outcome_or_metric_computed": False,
            "confirmation_or_sealed_final_image_accessed": False,
            "planned_scene_backfill_occurred": False,
        },
        "limitations": [
            "The two haze files are nested formation observations of one original clear scene and are not independent replicates.",
            "Distinct source tokens and decoded haze payloads verify a formation contrast while exact clear RGB identity holds content fixed; they do not validate a physical atmospheric parameter model.",
            "The whole-image shape envelope is an input contract only. Revised S1 must separately qualify the exact full-image ConvIR-B production path at the bound device class before inference.",
            "PASS authorizes only revised S1 contract authoring and provides no evidence about restoration need, local utility, near-clear safety, or S2 readiness.",
        ],
        "marker": "DAYTIME_DEHAZING_RESIDE_PAIRING_CONTRACT_V1_COMPLETE",
    }
    summary_path = output_file(context, SUMMARY_NAME)
    atomic_json(summary_path, summary)
    summary_sha256 = sha256_file(summary_path)
    identity_summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "independent_unit": "original_clear_scene",
        "content_definition": "exact decoded canonical clear RGB digest",
        "formation_definition": "two distinct nested haze file and decoded RGB payloads with distinct nonempty source tokens",
        "planned_rosters": {
            dataset: {
                key: summaries[dataset][key]
                for key in (
                    "planned_scene_count",
                    "planned_scene_digest",
                    "selected_scene_count",
                    "mapping_ledger_slice_digest",
                    "shape_envelope",
                    "independence_contract",
                )
            }
            for dataset in DATASETS
        },
        "mapping_ledger_sha256": mapping_sha256,
        "mapping_ledger_rows": len(mapping_rows),
    }
    atomic_json(output_file(context, IDENTITY_NAME), identity_summary)

    review_facts = {
        "schema_version": 2,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "facts": [
            {
                "fact_id": gate_id,
                "claim_id": gate_id,
                "metric": (
                    "valid frozen planned original-clear scenes"
                    if gate_id == "planned_roster_coverage"
                    else f"{gate_id} typed gate outcome"
                ),
                "unit": "original_clear_scene"
                if gate_id == "planned_roster_coverage" else "typed outcome",
                "population": "the frozen 150 ITS and 150 OTS development-screening scenes",
                "grouping": "original clear scene; two haze observations nested within scene",
                "point": valid_scene_count
                if gate_id == "planned_roster_coverage" else None,
                "ci_lower": None,
                "ci_upper": None,
                "confidence_level": None,
                "threshold": sum(PLANNED_SCENES.values())
                if gate_id == "planned_roster_coverage" else None,
                "threshold_operator": ">="
                if gate_id == "planned_roster_coverage" else None,
                "gate_outcome": gate_outcomes[gate_id],
                "source_filename": SUMMARY_NAME,
                "source_sha256": summary_sha256,
                "json_pointers": {
                    "point": "/valid_scene_count"
                    if gate_id == "planned_roster_coverage" else None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "confidence_level": None,
                    "threshold": "/required_scene_count"
                    if gate_id == "planned_roster_coverage" else None,
                    "gate_outcome": f"/gate_outcomes/{gate_id}",
                },
            }
            for gate_id in gate_outcomes
        ],
    }
    atomic_json(output_file(context, REVIEW_FACTS_NAME), review_facts)
    write_workload_progress(
        context,
        completed_units=TOTAL_UNITS,
        stage="pairing_contract_finalized",
    )
    write_gate_result(
        context,
        gate_outcomes=gate_outcomes,
        details={
            "summary_file": SUMMARY_NAME,
            "identity_summary_file": IDENTITY_NAME,
            "mapping_ledger_file": RAW_LEDGER_NAME,
            "mapping_ledger_sha256": mapping_sha256,
            "independent_scene_count": valid_scene_count,
            "nested_variant_count": valid_scene_count * VARIANTS_PER_SCENE,
            "training_or_inference_occurred": False,
            "restoration_outcomes_accessed": False,
            "confirmation_or_sealed_data_touched": False,
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
