#!/usr/bin/env python3
"""Qualify a deterministic 600/150 scene split for Haze4K train."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
EXPECTED_FILES = 3000
EXPECTED_SCENES = 750
TRAINING_SCENES = 600
DEVELOPMENT_SCENES = 150
VARIANTS_PER_SCENE = 4
SPLIT_SALT = "haze4k-local-error-qualification-v2"


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
    except Exception as exc:  # PIL uses several format-specific exceptions.
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


def selected_label(image_name: str, label_dir: Path) -> Path | None:
    stem, extension = os.path.splitext(image_name)
    names = [image_name]
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        names.extend((f"{prefix}{extension}", f"{prefix}.png"))
    seen: set[Path] = set()
    for name in names:
        candidate = label_dir / name
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def digest_lines(lines: list[str]) -> str:
    payload = "\n".join(sorted(lines)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    parent = context.assets.get("prior_scene_audit_conclusion")
    hidden_train = context.assets.get("haze4k_train")
    checks = {
        "metadata_only_mode": context.engineering_contract["mode"] == "metadata_only",
        "cpu_contract": context.device == "cpu",
        "haze4k_train_hidden_from_contract": (
            hidden_train is None or hidden_train.contract_access is False
        ),
        "prior_scene_audit_identity_bound": (
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
    frozen = (
        int(os.environ.get("CONVIR_ROUTE_HAZE4K_SPLIT_WORKERS", "8")),
        int(os.environ.get("CONVIR_ROUTE_HAZE4K_EXPECTED_SCENES", "750")),
        int(os.environ.get("CONVIR_ROUTE_HAZE4K_TRAINING_SCENES", "600")),
        int(os.environ.get("CONVIR_ROUTE_HAZE4K_DEVELOPMENT_SCENES", "150")),
        os.environ.get("CONVIR_ROUTE_HAZE4K_SPLIT_SALT", SPLIT_SALT),
    )
    expected = (8, EXPECTED_SCENES, TRAINING_SCENES, DEVELOPMENT_SCENES, SPLIT_SALT)
    if frozen != expected:
        raise ValueError("Haze4K split environment differs from the frozen contract")
    workers = frozen[0]

    train_root = asset_path(context, "haze4k_train", kind="directory")
    scope_ok = (
        train_root.name.lower() == "train"
        and "test" not in {part.lower() for part in train_root.parts}
        and not any(context.protected_data_permissions.values())
    )
    input_dirs = supported_directories(train_root, INPUT_DIRECTORY_NAMES) if scope_ok else []
    label_dirs = supported_directories(train_root, LABEL_DIRECTORY_NAMES) if scope_ok else []
    directory_ok = len(input_dirs) == 1 and len(label_dirs) == 1
    write_workload_progress(context, completed_units=1, stage="scope_and_pairing")

    hazy_paths = image_files(input_dirs[0]) if directory_ok else []
    clear_paths = image_files(label_dirs[0]) if directory_ok else []
    pairing: dict[Path, Path] = {}
    if directory_ok:
        for hazy_path in hazy_paths:
            clear_path = selected_label(hazy_path.name, label_dirs[0])
            if clear_path is not None:
                pairing[hazy_path] = clear_path

    clear_info = inspect_images(clear_paths, workers) if clear_paths else []
    clear_by_path = {item["path"]: item for item in clear_info}
    write_workload_progress(context, completed_units=2, stage="decode_clear_train")
    hazy_info = inspect_images(hazy_paths, workers) if hazy_paths else []
    hazy_by_path = {item["path"]: item for item in hazy_info}
    write_workload_progress(context, completed_units=3, stage="decode_hazy_train")

    clear_decode_failures = sum(item["error"] is not None for item in clear_info)
    hazy_decode_failures = sum(item["error"] is not None for item in hazy_info)
    aligned_pairs = 0
    decodable_pairs = 0
    scene_observations: Counter[str] = Counter()
    all_clear_digest_to_names: dict[str, set[str]] = defaultdict(set)
    for clear in clear_info:
        if clear["digest"] is not None:
            all_clear_digest_to_names[clear["digest"]].add(clear["path"].name)
    for hazy_path, clear_path in pairing.items():
        hazy = hazy_by_path.get(hazy_path)
        clear = clear_by_path.get(clear_path)
        if not hazy or not clear or hazy["error"] is not None or clear["error"] is not None:
            continue
        decodable_pairs += 1
        if (hazy["width"], hazy["height"]) == (clear["width"], clear["height"]):
            aligned_pairs += 1
        scene_observations[clear["digest"]] += 1

    scene_digests = sorted(all_clear_digest_to_names)
    clear_alias_count = sum(
        max(0, len(names) - 1) for names in all_clear_digest_to_names.values()
    )
    variant_histogram = dict(sorted(Counter(scene_observations.values()).items()))
    four_variant_scenes = sum(
        count == VARIANTS_PER_SCENE for count in scene_observations.values()
    )
    ranked_scenes = sorted(
        scene_digests,
        key=lambda digest: (
            hashlib.sha256(f"{SPLIT_SALT}|{digest}".encode("utf-8")).hexdigest(),
            digest,
        ),
    )
    development = set(ranked_scenes[:DEVELOPMENT_SCENES])
    training = set(ranked_scenes[DEVELOPMENT_SCENES:])
    assignment_lines = [
        f"{digest},{'internal_development' if digest in development else 'training'}"
        for digest in scene_digests
    ]
    assignment_digest = digest_lines(assignment_lines)
    development_observations = sum(scene_observations[digest] for digest in development)
    training_observations = sum(scene_observations[digest] for digest in training)
    all_observations = sum(scene_observations.values())

    complete_census = (
        directory_ok
        and len(hazy_paths) == EXPECTED_FILES
        and len(clear_paths) == EXPECTED_FILES
        and len(pairing) == EXPECTED_FILES
        and clear_decode_failures == 0
        and hazy_decode_failures == 0
        and decodable_pairs == EXPECTED_FILES
        and aligned_pairs == EXPECTED_FILES
    )
    gates = {
        "train_only_asset_scope": scope_ok,
        "complete_file_and_pair_census": complete_census,
        "canonical_scene_count": len(scene_digests) == EXPECTED_SCENES,
        "four_variants_per_scene": (
            four_variant_scenes == EXPECTED_SCENES
            and variant_histogram == {VARIANTS_PER_SCENE: EXPECTED_SCENES}
        ),
        "frozen_role_counts": (
            len(training) == TRAINING_SCENES
            and len(development) == DEVELOPMENT_SCENES
            and not (training & development)
        ),
        "variant_containment_and_coverage": (
            all_observations == EXPECTED_FILES
            and training_observations == TRAINING_SCENES * VARIANTS_PER_SCENE
            and development_observations == DEVELOPMENT_SCENES * VARIANTS_PER_SCENE
            and training | development == set(scene_digests)
        ),
    }
    if all(gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "HAZE4K_TRAIN_SCENE_SPLIT_PASS"
        authorizes = "HAZE4K_TRAIN_BASELINE_LOCAL_ERROR_MEASUREMENT"
        gate_reasons = ["all frozen 750-scene grouping, 600/150 role, containment, and coverage gates passed"]
    elif not scope_ok:
        state = "COMPLETED_GATE_FAIL"
        decision = "HAZE4K_TRAIN_SCENE_SPLIT_FAIL"
        authorizes = "NONE"
        gate_reasons = ["train_only_asset_scope"]
    elif not directory_ok or len(hazy_paths) != EXPECTED_FILES or len(clear_paths) != EXPECTED_FILES:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "HAZE4K_TRAIN_SCENE_SPLIT_INCONCLUSIVE"
        authorizes = "HAZE4K_TRAIN_SCENE_SPLIT_SUPPLEMENT_ONLY"
        gate_reasons = [name for name, passed in gates.items() if not passed]
    else:
        state = "COMPLETED_GATE_FAIL"
        decision = "HAZE4K_TRAIN_SCENE_SPLIT_FAIL"
        authorizes = "NONE"
        gate_reasons = [name for name, passed in gates.items() if not passed]

    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "Haze4K train-only 750-scene split qualification; official test excluded",
        "asset_scope": {
            "scope_ok": scope_ok,
            "train_directory_name": train_root.name,
            "official_test_asset_delivered": False,
            "protected_permissions_enabled": any(context.protected_data_permissions.values()),
        },
        "census": {
            "hazy_files": len(hazy_paths),
            "clear_files": len(clear_paths),
            "loader_selected_pairs": len(pairing),
            "hazy_decode_failures": hazy_decode_failures,
            "clear_decode_failures": clear_decode_failures,
            "decodable_pairs": decodable_pairs,
            "dimension_aligned_pairs": aligned_pairs,
        },
        "scene_grouping": {
            "canonical_definition": "SHA-256 over width, height, and decoded RGB bytes of the clear target",
            "canonical_scene_count": len(scene_digests),
            "clear_alias_count": clear_alias_count,
            "variant_count_histogram": variant_histogram,
            "four_variant_scene_count": four_variant_scenes,
        },
        "frozen_split": {
            "salt": SPLIT_SALT,
            "ranking": "SHA-256(salt + vertical-bar + canonical_scene_digest), then digest",
            "training_scene_count": len(training),
            "internal_development_scene_count": len(development),
            "training_observation_count": training_observations,
            "internal_development_observation_count": development_observations,
            "assignment_digest": assignment_digest,
            "assignment_rows_archived": False,
        },
        "gates": gates,
        "terminal": {
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "limitations": [
            "The operation uses Haze4K train only and makes no official-test or model-performance claim.",
            "Decoded RGB identity defines canonical grouping but does not establish capture provenance or filename-token physics.",
            "All 3,000 haze observations are retained, while variants remain nested within 750 independent scenes.",
            "Only aggregate counts and an assignment digest are archived; no per-image assignment table is published.",
        ],
        "marker": "HAZE4K_TRAIN_SCENE_SPLIT_QUALIFICATION_COMPLETE",
    }
    output_file(context, "haze4k_train_scene_split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    rows = [
        ("hazy_files", len(hazy_paths), EXPECTED_FILES),
        ("clear_files", len(clear_paths), EXPECTED_FILES),
        ("loader_selected_pairs", len(pairing), EXPECTED_FILES),
        ("canonical_scenes", len(scene_digests), EXPECTED_SCENES),
        ("four_variant_scenes", four_variant_scenes, EXPECTED_SCENES),
        ("training_scenes", len(training), TRAINING_SCENES),
        ("internal_development_scenes", len(development), DEVELOPMENT_SCENES),
        ("training_observations", training_observations, TRAINING_SCENES * VARIANTS_PER_SCENE),
        ("internal_development_observations", development_observations, DEVELOPMENT_SCENES * VARIANTS_PER_SCENE),
    ]
    with output_file(context, "haze4k_train_scene_split_strata.csv").open(
        "w", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("stratum", "observed", "reference", "rate"))
        for name, observed, reference in rows:
            writer.writerow((name, observed, reference, observed / reference if reference else 0.0))

    write_workload_progress(context, completed_units=4, stage="aggregate_and_finalize")
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "haze4k_train_scene_split_summary.json",
            "strata_file": "haze4k_train_scene_split_strata.csv",
            "canonical_scenes": len(scene_digests),
            "training_scenes": len(training),
            "internal_development_scenes": len(development),
            "training_observations": training_observations,
            "internal_development_observations": development_observations,
            "split_assignment_digest": assignment_digest,
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
