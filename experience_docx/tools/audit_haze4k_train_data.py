#!/usr/bin/env python3
"""Audit Haze4K train pairing, canonical scenes, and a frozen split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from route_program_api import (
    asset_path,
    load_context,
    output_file,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
    write_workload_progress,
)


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
INPUT_DIRECTORY_NAMES = ("IN", "haze", "hazy")
LABEL_DIRECTORY_NAMES = ("GT", "gt")
EXPECTED_HAZY = 3000
EXPECTED_CLEAR = 3000
INTERNAL_DEV_SCENES = 600
SPLIT_SALT = "haze4k-local-error-qualification-v1"


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def supported_directories(root: Path, names: tuple[str, ...]) -> list[Path]:
    return [root / name for name in names if (root / name).is_dir()]


def canonical_rgb_digest(width: int, height: int, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(width.to_bytes(8, "big"))
    digest.update(height.to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            rgb.load()
            width, height = rgb.size
            digest = canonical_rgb_digest(width, height, rgb.tobytes())
        return {
            "path": path,
            "width": width,
            "height": height,
            "digest": digest,
            "error": None,
        }
    except Exception as exc:  # PIL exposes several format-specific exceptions.
        return {
            "path": path,
            "width": None,
            "height": None,
            "digest": None,
            "error": f"{type(exc).__name__}: {exc}"[:240],
        }


def inspect_images(paths: list[Path], workers: int) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(inspect_image, paths, chunksize=16))


def label_candidates(image_name: str, label_dir: Path) -> tuple[Path | None, int]:
    """Return the loader-selected label and the number of existing candidates.

    The official loader uses deterministic first-match precedence.  Counting all
    existing candidates separately preserves an audit signal for redundant names
    without changing the target selected by the production loader.
    """
    stem, extension = os.path.splitext(image_name)
    names = [image_name]
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        names.extend((f"{prefix}{extension}", f"{prefix}.png"))
    unique: list[Path] = []
    seen: set[Path] = set()
    for name in names:
        candidate = label_dir / name
        if candidate.is_file() and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return (unique[0] if unique else None, len(unique))


def numeric_filename_tokens(path: Path) -> tuple[str, tuple[float, float] | None]:
    parts = path.stem.split("_")
    scene_token = parts[0]
    if len(parts) != 3:
        return scene_token, None
    try:
        values = (float(parts[1]), float(parts[2]))
    except ValueError:
        return scene_token, None
    if not all(math.isfinite(value) for value in values):
        return scene_token, None
    return scene_token, values


def digest_lines(lines: list[str]) -> str:
    payload = "\n".join(sorted(lines)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def range_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "unique": 0, "min": None, "max": None}
    return {
        "count": len(values),
        "unique": len(set(values)),
        "min": min(values),
        "max": max(values),
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    parent = context.assets.get("sots_parent_conclusion")
    checks = {
        "metadata_only_mode": context.engineering_contract["mode"] == "metadata_only",
        "cpu_contract": context.device == "cpu",
        "haze4k_train_hidden_from_contract": (
            "haze4k_train" not in context.assets
            or (
                context.assets["haze4k_train"].contract_access is False
                and context.assets["haze4k_train"].access_role == "development_screening"
            )
        ),
        "parent_conclusion_identity_bound": (
            parent is not None and parent.contract_access is True
        ),
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
        "no_model_checkpoint_or_inference_path": True,
        "aggregate_evidence_only": True,
        "output_and_finalizer_contract": True,
    }
    write_contract_result(
        context,
        checks=checks,
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
    workers = int(os.environ.get("CONVIR_ROUTE_HAZE4K_AUDIT_WORKERS", "8"))
    expected_hazy = int(os.environ.get(
        "CONVIR_ROUTE_HAZE4K_EXPECTED_HAZY", str(EXPECTED_HAZY),
    ))
    expected_clear = int(os.environ.get(
        "CONVIR_ROUTE_HAZE4K_EXPECTED_CLEAR", str(EXPECTED_CLEAR),
    ))
    internal_dev_scenes = int(os.environ.get(
        "CONVIR_ROUTE_HAZE4K_INTERNAL_DEV_SCENES", str(INTERNAL_DEV_SCENES),
    ))
    split_salt = os.environ.get("CONVIR_ROUTE_HAZE4K_SPLIT_SALT", SPLIT_SALT)
    if (workers, expected_hazy, expected_clear, internal_dev_scenes, split_salt) != (
        8, EXPECTED_HAZY, EXPECTED_CLEAR, INTERNAL_DEV_SCENES, SPLIT_SALT,
    ):
        raise ValueError("Haze4K audit environment differs from the frozen contract")

    train_root = asset_path(context, "haze4k_train", kind="directory")
    scope_ok = (
        train_root.name.lower() == "train"
        and "test" not in {part.lower() for part in train_root.parts}
        and not any(context.protected_data_permissions.values())
    )
    input_directories = supported_directories(train_root, INPUT_DIRECTORY_NAMES) if scope_ok else []
    label_directories = supported_directories(train_root, LABEL_DIRECTORY_NAMES) if scope_ok else []
    directory_ok = len(input_directories) == 1 and len(label_directories) == 1
    write_workload_progress(context, completed_units=1, stage="train_scope_and_structure")

    hazy_paths = image_files(input_directories[0]) if directory_ok else []
    clear_paths = image_files(label_directories[0]) if directory_ok else []
    label_dir = label_directories[0] if directory_ok else None
    pairing: dict[Path, Path] = {}
    missing_pair_count = 0
    redundant_candidate_count = 0
    if label_dir is not None:
        for hazy_path in hazy_paths:
            selected, candidate_count = label_candidates(hazy_path.name, label_dir)
            if selected is not None:
                pairing[hazy_path] = selected
                if candidate_count > 1:
                    redundant_candidate_count += 1
            else:
                missing_pair_count += 1

    clear_info = inspect_images(clear_paths, workers) if clear_paths else []
    clear_by_path = {item["path"]: item for item in clear_info}
    clear_decode_failures = sum(item["error"] is not None for item in clear_info)
    write_workload_progress(context, completed_units=2, stage="decode_clear_train")

    hazy_info = inspect_images(hazy_paths, workers) if hazy_paths else []
    hazy_by_path = {item["path"]: item for item in hazy_info}
    hazy_decode_failures = sum(item["error"] is not None for item in hazy_info)
    write_workload_progress(context, completed_units=3, stage="decode_hazy_train")

    aligned_pair_count = 0
    decodable_pair_count = 0
    referenced_labels: set[Path] = set()
    scene_variants: Counter[str] = Counter()
    scene_digest_to_labels: dict[str, set[str]] = defaultdict(set)
    scene_token_to_digests: dict[str, set[str]] = defaultdict(set)
    parsed_token_count = 0
    token_values: tuple[list[float], list[float]] = ([], [])
    for hazy_path, clear_path in pairing.items():
        hazy = hazy_by_path.get(hazy_path)
        clear = clear_by_path.get(clear_path)
        referenced_labels.add(clear_path)
        scene_token, numeric_tokens = numeric_filename_tokens(hazy_path)
        if numeric_tokens is not None:
            parsed_token_count += 1
            token_values[0].append(numeric_tokens[0])
            token_values[1].append(numeric_tokens[1])
        if not hazy or not clear or hazy["error"] is not None or clear["error"] is not None:
            continue
        decodable_pair_count += 1
        if (hazy["width"], hazy["height"]) == (clear["width"], clear["height"]):
            aligned_pair_count += 1
        clear_digest = clear["digest"]
        scene_variants[clear_digest] += 1
        scene_digest_to_labels[clear_digest].add(clear_path.name)
        scene_token_to_digests[scene_token].add(clear_digest)

    all_clear_digest_to_labels: dict[str, set[str]] = defaultdict(set)
    for item in clear_info:
        if item["digest"] is not None:
            all_clear_digest_to_labels[item["digest"]].add(item["path"].name)
    canonical_scene_digests = sorted(all_clear_digest_to_labels)
    clear_alias_count = sum(
        max(0, len(names) - 1) for names in all_clear_digest_to_labels.values()
    )
    scene_token_collision_count = sum(
        len(digests) > 1 for digests in scene_token_to_digests.values()
    )
    orphan_clear_count = len(set(clear_paths) - referenced_labels)

    ranked_scenes = sorted(
        canonical_scene_digests,
        key=lambda digest: (hashlib.sha256(
            f"{split_salt}|{digest}".encode("utf-8")
        ).hexdigest(), digest),
    )
    development = set(ranked_scenes[:internal_dev_scenes])
    split_lines = [
        f"{digest},{'internal_development' if digest in development else 'training'}"
        for digest in canonical_scene_digests
    ]
    internal_development_count = len(development)
    training_count = len(canonical_scene_digests) - internal_development_count
    split_digest = digest_lines(split_lines)

    metadata_paths = []
    if directory_ok:
        metadata_paths = sorted(
            path.relative_to(train_root).as_posix()
            for path in train_root.rglob("*")
            if path.is_file() and path.suffix.lower() not in IMAGE_EXTENSIONS
        )
    variants_histogram = dict(sorted(Counter(scene_variants.values()).items()))
    total_images = len(hazy_paths) + len(clear_paths)
    total_decode_failures = hazy_decode_failures + clear_decode_failures
    unique_pair_rate = len(pairing) / len(hazy_paths) if hazy_paths else 0.0
    decode_rate = (
        (total_images - total_decode_failures) / total_images if total_images else 0.0
    )
    dimension_alignment_rate = (
        aligned_pair_count / decodable_pair_count if decodable_pair_count else 0.0
    )
    filename_token_parse_rate = (
        parsed_token_count / len(hazy_paths) if hazy_paths else 0.0
    )
    gates = {
        "train_only_asset_scope": scope_ok,
        "directory_contract": directory_ok,
        "hazy_file_count": len(hazy_paths) == EXPECTED_HAZY,
        "clear_file_count": len(clear_paths) == EXPECTED_CLEAR,
        "unique_pairing_rate": unique_pair_rate == 1.0,
        "decode_rate": decode_rate == 1.0,
        "dimension_alignment_rate": dimension_alignment_rate == 1.0,
        "canonical_scene_count": len(canonical_scene_digests) == EXPECTED_CLEAR,
        "clear_alias_count": clear_alias_count == 0,
        "filename_token_parse_rate": filename_token_parse_rate == 1.0,
        "scene_token_collision_count": scene_token_collision_count == 0,
        "frozen_split_counts": (
            training_count == 2400 and internal_development_count == 600
        ),
    }
    safety_gates = ("train_only_asset_scope",)
    identity_gates = ("directory_contract", "hazy_file_count", "clear_file_count")
    if not all(gates[name] for name in safety_gates):
        state = "COMPLETED_GATE_FAIL"
        decision = "HAZE4K_TRAIN_DATA_AUDIT_FAIL"
        authorizes = "NONE"
        gate_reasons = [name for name in safety_gates if not gates[name]]
    elif not all(gates[name] for name in identity_gates):
        state = "COMPLETED_INCONCLUSIVE"
        decision = "HAZE4K_TRAIN_DATA_AUDIT_INCONCLUSIVE"
        authorizes = "HAZE4K_TRAIN_DATA_AUDIT_SUPPLEMENT_ONLY"
        gate_reasons = [name for name in identity_gates if not gates[name]]
    elif all(gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "HAZE4K_TRAIN_DATA_AUDIT_PASS"
        authorizes = "HAZE4K_TRAIN_BASELINE_LOCAL_ERROR_MEASUREMENT"
        gate_reasons = [
            "all frozen train-scope, pairing, decode, alignment, scene-identity, token-syntax, and split gates passed"
        ]
    else:
        state = "COMPLETED_GATE_FAIL"
        decision = "HAZE4K_TRAIN_DATA_AUDIT_FAIL"
        authorizes = "NONE"
        gate_reasons = [name for name, passed in gates.items() if not passed]

    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "Haze4K train-only data qualification; official test excluded",
        "asset_scope": {
            "train_directory_name": train_root.name,
            "scope_ok": scope_ok,
            "confirmation_permissions_enabled": any(
                context.protected_data_permissions.values()
            ),
            "official_test_asset_delivered": False,
        },
        "structure": {
            "input_directory_candidates": [path.name for path in input_directories],
            "label_directory_candidates": [path.name for path in label_directories],
            "hazy_file_count": len(hazy_paths),
            "clear_file_count": len(clear_paths),
            "metadata_file_count": len(metadata_paths),
            "metadata_files": metadata_paths[:64],
            "metadata_files_truncated": len(metadata_paths) > 64,
        },
        "pairing": {
            "uniquely_paired": len(pairing),
            "missing_pair_count": missing_pair_count,
            "redundant_candidate_count": redundant_candidate_count,
            "orphan_clear_count": orphan_clear_count,
            "hazy_decode_failures": hazy_decode_failures,
            "clear_decode_failures": clear_decode_failures,
            "decodable_pair_count": decodable_pair_count,
            "dimension_aligned_pair_count": aligned_pair_count,
            "unique_pair_rate": unique_pair_rate,
            "decode_rate": decode_rate,
            "dimension_alignment_rate": dimension_alignment_rate,
        },
        "scene_grouping": {
            "canonical_scene_definition": "SHA-256 over width, height, and decoded RGB bytes of the clear target",
            "canonical_scene_count": len(canonical_scene_digests),
            "clear_alias_count": clear_alias_count,
            "scene_token_collision_count": scene_token_collision_count,
            "variant_count_histogram": variants_histogram,
        },
        "frozen_split": {
            "salt": split_salt,
            "ranking": "SHA-256(salt + vertical-bar + canonical_scene_digest), then digest",
            "training_scene_count": training_count,
            "internal_development_scene_count": internal_development_count,
            "assignment_digest": split_digest,
            "assignment_rows_archived": False,
        },
        "filename_generation_metadata": {
            "parsed_two_numeric_suffix_count": parsed_token_count,
            "parse_rate": filename_token_parse_rate,
            "suffix_position_1": range_summary(token_values[0]),
            "suffix_position_2": range_summary(token_values[1]),
            "semantic_status": "observed_numeric_tokens_only_no_physical_semantics_assigned",
            "interpretation_limit": (
                "Filename tokens and depth-induced transmission variation do not by themselves "
                "establish spatially varying atmospheric-light or scattering coefficients."
            ),
        },
        "gates": gates,
        "terminal": {
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "limitations": [
            "This operation audits Haze4K train only and makes no Haze4K official-test claim.",
            "The official baseline test metric was reproduced before this program; candidate test outcomes remain unseen and receive no access here.",
            "No model, checkpoint, training, inference, local-error measurement, oracle, proxy, module, NH-HAZE, confirmation, canary, or locked-test data is used.",
            "The audit reports filename-token syntax but does not identify their physical meaning without authoritative generation metadata.",
            "Only aggregate evidence and a split-assignment digest are archived; no per-image or per-scene table is published.",
        ],
        "marker": "HAZE4K_TRAIN_DATA_AUDIT_COMPLETE",
    }
    output_file(context, "haze4k_train_data_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    rows = [
        ("hazy_files", len(hazy_paths), EXPECTED_HAZY),
        ("clear_files", len(clear_paths), EXPECTED_CLEAR),
        ("uniquely_paired", len(pairing), len(hazy_paths)),
        ("decoded_images", total_images - total_decode_failures, total_images),
        ("dimension_aligned_pairs", aligned_pair_count, decodable_pair_count),
        ("canonical_clear_scenes", len(canonical_scene_digests), EXPECTED_CLEAR),
        ("filename_tokens_parsed", parsed_token_count, len(hazy_paths)),
        ("training_scenes", training_count, 2400),
        ("internal_development_scenes", internal_development_count, 600),
    ]
    strata_path = output_file(context, "haze4k_train_data_audit_strata.csv")
    with strata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("stratum", "observed", "reference", "rate"))
        for name, observed, reference in rows:
            rate = observed / reference if reference else 0.0
            writer.writerow((name, observed, reference, rate))

    write_workload_progress(context, completed_units=4, stage="aggregate_and_finalize")
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "haze4k_train_data_audit_summary.json",
            "strata_file": "haze4k_train_data_audit_strata.csv",
            "hazy_files": len(hazy_paths),
            "clear_files": len(clear_paths),
            "canonical_scenes": len(canonical_scene_digests),
            "training_scenes": training_count,
            "internal_development_scenes": internal_development_count,
            "split_assignment_digest": split_digest,
            "gate_reasons": gate_reasons,
            "official_test_accessed": False,
            "model_or_checkpoint_accessed": False,
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
