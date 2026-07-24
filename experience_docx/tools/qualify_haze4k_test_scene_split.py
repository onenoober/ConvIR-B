#!/usr/bin/env python3
"""Outcome-blind structure census and role isolation for Haze4K official test."""

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
EXPECTED_FILES = 1000
EXPECTED_SCENES = 250
DEVELOPMENT_SCENES = 100
CONFIRMATION_SCENES = 150
VARIANTS_PER_SCENE = 4
SPLIT_SALT = "haze4k-test-local-error-replication-v1"


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


def hardlink_role_asset(
    root: Path,
    role: str,
    pairs: list[tuple[Path, Path, str]],
    assignment_digest: str,
) -> dict[str, Any]:
    role_root = root / role
    haze_root = role_root / "haze"
    clear_root = role_root / "gt"
    haze_root.mkdir(parents=True)
    clear_root.mkdir()
    samefile_checks = 0
    for haze_path, clear_path, _scene_digest in sorted(pairs, key=lambda item: item[0].name):
        haze_destination = haze_root / haze_path.name
        clear_destination = clear_root / haze_path.name
        os.link(haze_path, haze_destination)
        os.link(clear_path, clear_destination)
        samefile_checks += int(
            os.path.samefile(haze_path, haze_destination)
            and os.path.samefile(clear_path, clear_destination)
        )
    scene_digests = {item[2] for item in pairs}
    metadata = {
        "schema_version": 1,
        "role": role,
        "canonical_scene_count": len(scene_digests),
        "observation_count": len(pairs),
        "haze_file_count": len(image_files(haze_root)),
        "clear_file_count": len(image_files(clear_root)),
        "assignment_digest": assignment_digest,
        "split_salt": SPLIT_SALT,
        "per_image_rows_included": False,
    }
    (role_root / "role_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {
        **metadata,
        "samefile_pair_count": samefile_checks,
        "exact_name_pairing": (
            {path.name for path in image_files(haze_root)}
            == {path.name for path in image_files(clear_root)}
        ),
        "asset_relpath": f"role_assets/{role}",
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    hidden_test = context.assets.get("haze4k_official_test")
    permissions = context.protected_data_permissions
    checks = {
        "metadata_only_mode": context.engineering_contract["mode"] == "metadata_only",
        "cpu_contract": context.device == "cpu",
        "official_test_hidden_from_contract": (
            hidden_test is None or hidden_test.contract_access is False
        ),
        "confirmation_permission_only": (
            permissions["allow_confirmation"]
            and not permissions["allow_canary"]
            and not permissions["allow_locked_test"]
        ),
        "no_model_checkpoint_inference_or_metric_path": True,
        "aggregate_publishable_evidence_only": True,
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
        int(os.environ.get("CONVIR_ROUTE_HAZE4K_TEST_SPLIT_WORKERS", "8")),
        int(os.environ.get("CONVIR_ROUTE_HAZE4K_TEST_EXPECTED_FILES", "1000")),
        int(os.environ.get("CONVIR_ROUTE_HAZE4K_TEST_EXPECTED_SCENES", "250")),
        int(os.environ.get("CONVIR_ROUTE_HAZE4K_TEST_DEVELOPMENT_SCENES", "100")),
        int(os.environ.get("CONVIR_ROUTE_HAZE4K_TEST_CONFIRMATION_SCENES", "150")),
        os.environ.get("CONVIR_ROUTE_HAZE4K_TEST_SPLIT_SALT", SPLIT_SALT),
    )
    expected = (
        8, EXPECTED_FILES, EXPECTED_SCENES, DEVELOPMENT_SCENES,
        CONFIRMATION_SCENES, SPLIT_SALT,
    )
    if frozen != expected:
        raise ValueError("Haze4K official-test split environment differs from contract")
    workers = frozen[0]

    test_root = asset_path(context, "haze4k_official_test", kind="directory")
    permissions = context.protected_data_permissions
    scope_ok = (
        test_root.name.lower() == "test"
        and permissions["allow_confirmation"]
        and not permissions["allow_canary"]
        and not permissions["allow_locked_test"]
        and len(context.assets) == 1
    )
    input_dirs = supported_directories(test_root, INPUT_DIRECTORY_NAMES) if scope_ok else []
    label_dirs = supported_directories(test_root, LABEL_DIRECTORY_NAMES) if scope_ok else []
    directory_ok = len(input_dirs) == 1 and len(label_dirs) == 1
    hazy_paths = image_files(input_dirs[0]) if directory_ok else []
    clear_paths = image_files(label_dirs[0]) if directory_ok else []
    pairing: dict[Path, Path] = {}
    if directory_ok:
        for hazy_path in hazy_paths:
            clear_path = selected_label(hazy_path.name, label_dirs[0])
            if clear_path is not None:
                pairing[hazy_path] = clear_path
    write_workload_progress(context, completed_units=1, stage="scope_and_pairing")

    clear_info = inspect_images(clear_paths, workers) if clear_paths else []
    clear_by_path = {item["path"]: item for item in clear_info}
    write_workload_progress(context, completed_units=2, stage="decode_clear_test")
    hazy_info = inspect_images(hazy_paths, workers) if hazy_paths else []
    hazy_by_path = {item["path"]: item for item in hazy_info}
    write_workload_progress(context, completed_units=3, stage="decode_hazy_test")

    clear_decode_failures = sum(item["error"] is not None for item in clear_info)
    hazy_decode_failures = sum(item["error"] is not None for item in hazy_info)
    aligned_pairs = 0
    decodable_pairs = 0
    scene_observations: Counter[str] = Counter()
    paired_rows: list[tuple[Path, Path, str]] = []
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
        scene_digest = clear["digest"]
        scene_observations[scene_digest] += 1
        paired_rows.append((hazy_path, clear_path, scene_digest))

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
    confirmation = set(ranked_scenes[DEVELOPMENT_SCENES:])
    assignment_lines = [
        f"{digest},{'development_screening' if digest in development else 'candidate_confirmation'}"
        for digest in scene_digests
    ]
    assignment_digest = digest_lines(assignment_lines)
    development_rows = [row for row in paired_rows if row[2] in development]
    confirmation_rows = [row for row in paired_rows if row[2] in confirmation]

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
    structure_ok = (
        len(scene_digests) == EXPECTED_SCENES
        and four_variant_scenes == EXPECTED_SCENES
        and variant_histogram == {VARIANTS_PER_SCENE: EXPECTED_SCENES}
    )
    role_counts_ok = (
        len(development) == DEVELOPMENT_SCENES
        and len(confirmation) == CONFIRMATION_SCENES
        and not (development & confirmation)
        and development | confirmation == set(scene_digests)
        and len(development_rows) == DEVELOPMENT_SCENES * VARIANTS_PER_SCENE
        and len(confirmation_rows) == CONFIRMATION_SCENES * VARIANTS_PER_SCENE
    )

    role_asset_results: dict[str, Any] = {}
    isolated_assets_ok = False
    if scope_ok and complete_census and structure_ok and role_counts_ok:
        role_asset_root = output_file(context, "role_assets")
        role_asset_root.mkdir()
        role_asset_results["development_screening"] = hardlink_role_asset(
            role_asset_root, "development_screening", development_rows, assignment_digest,
        )
        role_asset_results["candidate_confirmation"] = hardlink_role_asset(
            role_asset_root, "candidate_confirmation", confirmation_rows, assignment_digest,
        )
        expected_role_pairs = {
            "development_screening": DEVELOPMENT_SCENES * VARIANTS_PER_SCENE,
            "candidate_confirmation": CONFIRMATION_SCENES * VARIANTS_PER_SCENE,
        }
        isolated_assets_ok = all(
            result["observation_count"] == expected_role_pairs[role]
            and result["haze_file_count"] == expected_role_pairs[role]
            and result["clear_file_count"] == expected_role_pairs[role]
            and result["samefile_pair_count"] == expected_role_pairs[role]
            and result["exact_name_pairing"]
            for role, result in role_asset_results.items()
        ) and not (development & confirmation)

        with output_file(context, "role_assignment_cloud_only.csv").open(
            "w", encoding="utf-8", newline="",
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(("haze_filename", "selected_clear_filename", "scene_digest", "role"))
            for haze_path, clear_path, scene_digest in sorted(paired_rows):
                writer.writerow((
                    haze_path.name, clear_path.name, scene_digest,
                    "development_screening" if scene_digest in development else "candidate_confirmation",
                ))
    write_workload_progress(context, completed_units=4, stage="role_isolation")

    gates = {
        "confirmation_structure_only_scope": scope_ok,
        "complete_file_pair_decode_alignment_census": complete_census,
        "canonical_scene_and_variant_structure": structure_ok,
        "frozen_role_counts_and_containment": role_counts_ok,
        "isolated_role_asset_integrity": isolated_assets_ok,
    }
    raw_layout_complete = (
        directory_ok
        and len(hazy_paths) == EXPECTED_FILES
        and len(clear_paths) == EXPECTED_FILES
    )
    if all(gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "HAZE4K_TEST_SCENE_SPLIT_PASS"
        authorizes = "HAZE4K_TEST_DEVELOPMENT_LOCAL_ERROR_REPLICATION"
        gate_reasons = [
            "all frozen outcome-blind census, 250-scene structure, 100/150 containment, and isolated-role asset gates passed"
        ]
    elif not scope_ok:
        state = "COMPLETED_GATE_FAIL"
        decision = "HAZE4K_TEST_SCENE_SPLIT_FAIL"
        authorizes = "NONE"
        gate_reasons = ["confirmation_structure_only_scope"]
    elif not raw_layout_complete or not complete_census:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "HAZE4K_TEST_SCENE_SPLIT_INCONCLUSIVE"
        authorizes = "HAZE4K_TEST_SCENE_SPLIT_SUPPLEMENT_ONLY"
        gate_reasons = [name for name, passed in gates.items() if not passed]
    else:
        state = "COMPLETED_GATE_FAIL"
        decision = "HAZE4K_TEST_SCENE_SPLIT_FAIL"
        authorizes = "NONE"
        gate_reasons = [name for name, passed in gates.items() if not passed]

    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "Haze4K official-test outcome-blind structure census and 100/150 role isolation",
        "exposure_class": "baseline_exposed_candidate_unseen",
        "asset_scope": {
            "scope_ok": scope_ok,
            "test_directory_name": test_root.name,
            "confirmation_permission_enabled": permissions["allow_confirmation"],
            "canary_or_locked_test_permission_enabled": (
                permissions["allow_canary"] or permissions["allow_locked_test"]
            ),
            "model_or_checkpoint_asset_delivered": False,
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
            "development_scene_count": len(development),
            "candidate_confirmation_scene_count": len(confirmation),
            "development_observation_count": len(development_rows),
            "candidate_confirmation_observation_count": len(confirmation_rows),
            "assignment_digest": assignment_digest,
            "assignment_rows_published": False,
        },
        "isolated_role_assets": role_asset_results,
        "gates": gates,
        "terminal": {
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "forbidden_outcomes": {
            "model_or_checkpoint_accessed": False,
            "inference_or_prediction_accessed": False,
            "restoration_metric_computed": False,
            "local_error_or_threshold_computed": False,
            "candidate_comparison_performed": False,
        },
        "limitations": [
            "The full official test is baseline-exposed because its aggregate official ConvIR-B metric was previously reproduced; this operation preserves only candidate-unseen status.",
            "Decoded RGB identity defines the leakage-control group but does not establish capture provenance or filename-token physics.",
            "The qualification is structure-only and makes no local-error or restoration-quality claim.",
            "Only the 100-scene development asset may be delivered to the authorized replication; the 150-scene candidate-confirmation asset remains prohibited until a frozen-candidate confirmation route.",
            "Per-image assignments remain cloud-only and are not part of compact GitHub evidence.",
        ],
        "marker": "HAZE4K_TEST_SCENE_SPLIT_QUALIFICATION_COMPLETE",
    }
    output_file(context, "haze4k_test_scene_split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    rows = [
        ("hazy_files", len(hazy_paths), EXPECTED_FILES),
        ("clear_files", len(clear_paths), EXPECTED_FILES),
        ("loader_selected_pairs", len(pairing), EXPECTED_FILES),
        ("canonical_scenes", len(scene_digests), EXPECTED_SCENES),
        ("four_variant_scenes", four_variant_scenes, EXPECTED_SCENES),
        ("development_scenes", len(development), DEVELOPMENT_SCENES),
        ("candidate_confirmation_scenes", len(confirmation), CONFIRMATION_SCENES),
        ("development_observations", len(development_rows), DEVELOPMENT_SCENES * VARIANTS_PER_SCENE),
        ("candidate_confirmation_observations", len(confirmation_rows), CONFIRMATION_SCENES * VARIANTS_PER_SCENE),
    ]
    with output_file(context, "haze4k_test_scene_split_strata.csv").open(
        "w", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("stratum", "observed", "reference", "rate"))
        for name, observed, reference in rows:
            writer.writerow((name, observed, reference, observed / reference if reference else 0.0))

    write_workload_progress(context, completed_units=5, stage="aggregate_and_finalize")
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "haze4k_test_scene_split_summary.json",
            "strata_file": "haze4k_test_scene_split_strata.csv",
            "canonical_scenes": len(scene_digests),
            "development_scenes": len(development),
            "candidate_confirmation_scenes": len(confirmation),
            "development_observations": len(development_rows),
            "candidate_confirmation_observations": len(confirmation_rows),
            "split_assignment_digest": assignment_digest,
            "development_asset_relpath": "workload/role_assets/development_screening",
            "candidate_confirmation_asset_relpath": "workload/role_assets/candidate_confirmation",
            "gate_reasons": gate_reasons,
            "model_or_checkpoint_accessed": False,
            "restoration_outcomes_accessed": False,
        },
        confirmation_images_targets_outcomes_touched=True,
        canary_touched=False,
        locked_test_touched=False,
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
