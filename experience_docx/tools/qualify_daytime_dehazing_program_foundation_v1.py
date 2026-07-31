#!/usr/bin/env python3
"""Build the outcome-blind scene and evidence-role ledger for the new program."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
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


ROUTE_ID = "daytime-dehazing-program-foundation-v1"
OPERATION_ID = "DAYTIME_DEHAZING_PROGRAM_FOUNDATION_QUALIFY"
SUMMARY_NAME = "daytime_dehazing_program_foundation_v1_summary.json"
IDENTITY_NAME = "daytime_dehazing_program_foundation_v1_identity_summary.json"
ROLE_NAME = "daytime_dehazing_program_foundation_v1_role_summary.json"
RAW_LEDGER_NAME = "daytime_dehazing_program_foundation_v1_scene_role_ledger.jsonl"

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
INPUT_DIRECTORY_NAMES = ("IN", "haze", "hazy")
LABEL_DIRECTORY_NAMES = ("GT", "gt")
WORKERS = 8

HAZE4K_TRAIN_SALT = "haze4k-local-error-qualification-v2"
HAZE4K_TEST_SALT = "haze4k-test-local-error-replication-v1"
PROGRAM_ROLE_SALT = "daytime-dehazing-program-foundation-v1|scene-roles-v1"
ROLE_WEIGHTS = {
    "training": 0.40,
    "development_screening": 0.10,
    "confirmation": 0.25,
    "sealed_final": 0.25,
}
HAMILTON_TIE_PRIORITY = {
    "sealed_final": 0,
    "confirmation": 1,
    "development_screening": 2,
    "training": 3,
}

EXPECTED = {
    "HAZE4K_TRAIN": 750,
    "HAZE4K_TEST": 250,
    "ITS": 11000,
    "OTS": 8970,
    "NH_HAZE": 55,
}
EXPECTED_KNOWN_EXCLUSIONS = {"ITS": 2187, "OTS": 964}
MINIMUM_ROLE_COUNTS = {
    "ITS": {
        "training": 3000,
        "development_screening": 700,
        "confirmation": 1900,
        "sealed_final": 1900,
    },
    "OTS": {
        "training": 3000,
        "development_screening": 700,
        "confirmation": 1900,
        "sealed_final": 1900,
    },
}


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory is unavailable: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
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
    except Exception as exc:
        return {
            "path": path,
            "width": None,
            "height": None,
            "digest": None,
            "error": f"{type(exc).__name__}: {exc}"[:240],
        }


def inspect_images(paths: list[Path]) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
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


def digest_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def digest_order(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def salted_order(values: Iterable[str], salt: str, namespace: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{salt}|{namespace}|{value}".encode()).hexdigest(),
            value,
        ),
    )


def historical_haze4k_order(values: Iterable[str], salt: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{salt}|{value}".encode()).hexdigest(),
            value,
        ),
    )


def read_json_asset(context: Any, asset_id: str) -> dict[str, Any]:
    value = json.loads(
        asset_path(context, asset_id, kind="file").read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise ValueError(f"{asset_id} must contain a JSON object")
    return value


def read_unique_lines(path: Path) -> tuple[list[str], set[str]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines, set(lines)


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
                stream.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"write-once ledger already exists: {path}"
            ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit_input_sha256(unit_id: str, payload: dict[str, Any]) -> str:
    value = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "unit_id": unit_id,
        "program_role_salt": PROGRAM_ROLE_SALT,
        "payload": payload,
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def inventory_haze4k(root: Path, dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_dirs = supported_directories(root, INPUT_DIRECTORY_NAMES)
    label_dirs = supported_directories(root, LABEL_DIRECTORY_NAMES)
    directory_ok = len(input_dirs) == 1 and len(label_dirs) == 1
    hazy_paths = image_files(input_dirs[0]) if directory_ok else []
    clear_paths = image_files(label_dirs[0]) if directory_ok else []
    pairing = {
        path: selected_label(path.name, label_dirs[0])
        for path in hazy_paths
    } if directory_ok else {}
    pairing = {hazy: clear for hazy, clear in pairing.items() if clear is not None}

    clear_info = inspect_images(clear_paths) if clear_paths else []
    hazy_info = inspect_images(hazy_paths) if hazy_paths else []
    clear_by_path = {item["path"]: item for item in clear_info}
    hazy_by_path = {item["path"]: item for item in hazy_info}
    observations: Counter[str] = Counter()
    aligned_pairs = 0
    for hazy_path, clear_path in pairing.items():
        hazy = hazy_by_path.get(hazy_path)
        clear = clear_by_path.get(clear_path)
        if not hazy or not clear or hazy["error"] or clear["error"]:
            continue
        observations[clear["digest"]] += 1
        aligned_pairs += int(
            (hazy["width"], hazy["height"]) == (clear["width"], clear["height"])
        )
    scene_digests = sorted(
        {item["digest"] for item in clear_info if item["digest"] is not None}
    )
    expected_scenes = EXPECTED[dataset]
    salt = HAZE4K_TRAIN_SALT if dataset == "HAZE4K_TRAIN" else HAZE4K_TEST_SALT
    ranked = historical_haze4k_order(scene_digests, salt)
    if dataset == "HAZE4K_TRAIN":
        development = set(ranked[:150])
        role_by_digest = {
            digest: "development_screening" if digest in development else "training"
            for digest in scene_digests
        }
        historical_lines = [
            f"{digest},{'internal_development' if digest in development else 'training'}"
            for digest in scene_digests
        ]
    else:
        confirmation = set(ranked[:100])
        role_by_digest = {
            digest: "confirmation" if digest in confirmation else "sealed_final"
            for digest in scene_digests
        }
        historical_lines = [
            f"{digest},{'development_screening' if digest in confirmation else 'candidate_confirmation'}"
            for digest in scene_digests
        ]
    rows = [
        {
            "dataset": dataset,
            "scene_id": digest,
            "canonical_digest": digest,
            "observation_count": observations[digest],
            "role": role_by_digest[digest],
            "exclusion_reason": None,
        }
        for digest in scene_digests
    ]
    checks = {
        "directory_layout": directory_ok,
        "hazy_file_count": len(hazy_paths) == expected_scenes * 4,
        "clear_file_count": len(clear_paths) == expected_scenes * 4,
        "pairing_complete": len(pairing) == expected_scenes * 4,
        "decode_complete": all(not item["error"] for item in clear_info + hazy_info),
        "dimension_alignment": aligned_pairs == expected_scenes * 4,
        "scene_count": len(scene_digests) == expected_scenes,
        "four_variants_per_scene": (
            len(observations) == expected_scenes
            and set(observations.values()) == {4}
        ),
    }
    summary = {
        "dataset": dataset,
        "root_name": root.name,
        "scene_count": len(scene_digests),
        "observation_count": sum(observations.values()),
        "membership_digest": digest_lines(scene_digests),
        "historical_partition_digest": digest_lines(historical_lines),
        "program_role_counts": dict(sorted(Counter(role_by_digest.values()).items())),
        "checks": checks,
    }
    return rows, summary


def inventory_its(reside_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specifications = (
        ("ITS_TRAIN", reside_root / "official/ITS/train/ITS_clear"),
        ("ITS_VALIDATION", reside_root / "official/ITS/val/clear"),
    )
    scene_paths: dict[str, Path] = {}
    for namespace, directory in specifications:
        for path in image_files(directory):
            scene_id = f"{namespace}:{path.stem}"
            if scene_id in scene_paths:
                raise ValueError(f"duplicate ITS scene id: {scene_id}")
            scene_paths[scene_id] = path
    info = inspect_images([scene_paths[key] for key in sorted(scene_paths)])
    by_path = {item["path"]: item for item in info}
    rows = [
        {
            "dataset": "ITS",
            "scene_id": scene_id,
            "canonical_digest": by_path[path]["digest"],
            "observation_count": 1,
            "role": None,
            "exclusion_reason": None,
        }
        for scene_id, path in sorted(scene_paths.items())
    ]
    summary = {
        "dataset": "ITS",
        "scene_count": len(rows),
        "decode_failures": sum(item["error"] is not None for item in info),
        "membership_digest": digest_lines(scene_paths),
        "checks": {
            "scene_count": len(rows) == EXPECTED["ITS"],
            "decode_complete": all(not item["error"] for item in info),
            "scene_ids_unique": len(rows) == len(scene_paths),
        },
    }
    return rows, summary


def inventory_ots(reside_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = reside_root / "official/OTS_ALPHA/clear_images"
    scene_paths = {path.stem: path for path in image_files(directory)}
    info = inspect_images([scene_paths[key] for key in sorted(scene_paths)])
    by_path = {item["path"]: item for item in info}
    rows = [
        {
            "dataset": "OTS",
            "scene_id": scene_id,
            "canonical_digest": by_path[path]["digest"],
            "observation_count": 1,
            "role": None,
            "exclusion_reason": None,
        }
        for scene_id, path in sorted(scene_paths.items())
    ]
    summary = {
        "dataset": "OTS",
        "scene_count": len(rows),
        "decode_failures": sum(item["error"] is not None for item in info),
        "membership_digest": digest_lines(scene_paths),
        "checks": {
            "scene_count": len(rows) == EXPECTED["OTS"],
            "decode_complete": all(not item["error"] for item in info),
            "scene_ids_unique": len(rows) == len(scene_paths),
        },
    }
    return rows, summary


def inventory_nh_haze(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gt_paths = sorted(root.glob("*_GT.*"))
    pairs: list[tuple[str, Path, Path]] = []
    for gt_path in gt_paths:
        scene_id = gt_path.stem[:-3]
        candidates = sorted(root.glob(f"{scene_id}_hazy.*"))
        if len(candidates) == 1:
            pairs.append((scene_id, candidates[0], gt_path))
    inspected = inspect_images([path for _, hazy, gt in pairs for path in (hazy, gt)])
    by_path = {item["path"]: item for item in inspected}
    rows = []
    aligned = 0
    for scene_id, hazy, gt in pairs:
        hazy_info = by_path[hazy]
        gt_info = by_path[gt]
        aligned += int(
            not hazy_info["error"]
            and not gt_info["error"]
            and (hazy_info["width"], hazy_info["height"])
            == (gt_info["width"], gt_info["height"])
        )
        rows.append({
            "dataset": "NH_HAZE",
            "scene_id": scene_id,
            "canonical_digest": gt_info["digest"],
            "observation_count": 1,
            "role": "sealed_final",
            "exclusion_reason": None,
        })
    summary = {
        "dataset": "NH_HAZE",
        "scene_count": len(rows),
        "pair_count": len(pairs),
        "decode_failures": sum(item["error"] is not None for item in inspected),
        "membership_digest": digest_lines(row["scene_id"] for row in rows),
        "checks": {
            "scene_count": len(rows) == EXPECTED["NH_HAZE"],
            "pairing_complete": len(pairs) == EXPECTED["NH_HAZE"],
            "decode_complete": all(not item["error"] for item in inspected),
            "dimension_alignment": aligned == EXPECTED["NH_HAZE"],
            "scene_ids_unique": len({row["scene_id"] for row in rows}) == len(rows),
        },
    }
    return rows, summary


def hamilton_counts(total: int) -> dict[str, int]:
    quotas = {role: total * weight for role, weight in ROLE_WEIGHTS.items()}
    counts = {role: math.floor(value) for role, value in quotas.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        ROLE_WEIGHTS,
        key=lambda role: (
            -(quotas[role] - counts[role]),
            HAMILTON_TIE_PRIORITY[role],
        ),
    )
    for role in order[:remainder]:
        counts[role] += 1
    return counts


def assign_program_roles(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    eligible = [row for row in rows if row["exclusion_reason"] is None]
    order = salted_order(
        (row["scene_id"] for row in eligible), PROGRAM_ROLE_SALT, dataset
    )
    by_id = {row["scene_id"]: row for row in eligible}
    target = hamilton_counts(len(order))
    cursor = 0
    for role in ("training", "development_screening", "confirmation", "sealed_final"):
        next_cursor = cursor + target[role]
        for scene_id in order[cursor:next_cursor]:
            by_id[scene_id]["role"] = role
        cursor = next_cursor
    return {
        "allocation_salt": PROGRAM_ROLE_SALT,
        "allocation_method": "SHA-256 salted scene order plus Hamilton largest remainder",
        "hamilton_tie_priority": [
            "sealed_final", "confirmation", "development_screening", "training"
        ],
        "eligible_scene_count": len(eligible),
        "role_counts": target,
        "order_digest": digest_order(order),
        "assignment_digest": digest_lines(
            f"{row['scene_id']},{row['role']}" for row in eligible
        ),
    }


def apply_exclusions_and_exact_deduplication(
    rows_by_dataset: dict[str, list[dict[str, Any]]],
    its_exclusions: set[str],
    ots_exclusions: set[str],
) -> dict[str, Any]:
    for row in rows_by_dataset["ITS"]:
        if row["scene_id"] in its_exclusions:
            row["exclusion_reason"] = "verified_cross_dataset_overlap"
    for row in rows_by_dataset["OTS"]:
        if row["scene_id"] in ots_exclusions:
            row["exclusion_reason"] = "verified_cross_dataset_overlap"

    priority = {
        "HAZE4K_TRAIN": 0,
        "HAZE4K_TEST": 1,
        "NH_HAZE": 2,
        "ITS": 3,
        "OTS": 4,
    }
    by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in rows_by_dataset.values():
        for row in rows:
            if row["exclusion_reason"] is None and row["canonical_digest"] is not None:
                by_digest[row["canonical_digest"]].append(row)

    exact_components = []
    protected_role_conflicts = []
    for digest, component in sorted(by_digest.items()):
        if len(component) < 2:
            continue
        datasets = sorted({row["dataset"] for row in component})
        if len(datasets) == 1:
            canonical = min(component, key=lambda row: row["scene_id"])
        else:
            canonical = min(
                component,
                key=lambda row: (priority[row["dataset"]], row["scene_id"]),
            )
        fixed_roles = {row["role"] for row in component if row["role"] is not None}
        if len(fixed_roles) > 1:
            protected_role_conflicts.append(digest)
        for row in component:
            if row is not canonical:
                row["exclusion_reason"] = "exact_rgb_duplicate_alias"
                row["role"] = None
        exact_components.append({
            "canonical_digest": digest,
            "datasets": datasets,
            "member_count": len(component),
            "retained_dataset": canonical["dataset"],
        })

    pair_counts: Counter[str] = Counter()
    for component in exact_components:
        datasets = component["datasets"]
        for left_index, left in enumerate(datasets):
            for right in datasets[left_index + 1:]:
                pair_counts[f"{left}__{right}"] += 1
    return {
        "known_exclusion_counts": {
            "ITS": sum(
                row["exclusion_reason"] == "verified_cross_dataset_overlap"
                for row in rows_by_dataset["ITS"]
            ),
            "OTS": sum(
                row["exclusion_reason"] == "verified_cross_dataset_overlap"
                for row in rows_by_dataset["OTS"]
            ),
        },
        "exact_duplicate_component_count": len(exact_components),
        "exact_cross_dataset_pair_matrix": dict(sorted(pair_counts.items())),
        "exact_component_digest": digest_lines(
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in exact_components
        ),
        "protected_role_conflict_count": len(protected_role_conflicts),
        "protected_role_conflict_digest": digest_lines(protected_role_conflicts),
        "claim_boundary": "Exact decoded-RGB identity plus archived verified Haze4K-to-ITS/OTS relationships; this does not prove complete capture or source provenance disjointness.",
    }


def role_summary(rows_by_dataset: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    role_digests: dict[str, set[str]] = defaultdict(set)
    for dataset, rows in rows_by_dataset.items():
        eligible = [
            row for row in rows
            if row["exclusion_reason"] is None and row["role"] is not None
        ]
        counts = Counter(row["role"] for row in eligible)
        datasets[dataset] = {
            "eligible_scene_count": len(eligible),
            "excluded_scene_count": len(rows) - len(eligible),
            "role_counts": dict(sorted(counts.items())),
            "assignment_digest": digest_lines(
                f"{row['scene_id']},{row['role']}" for row in eligible
            ),
        }
        for row in eligible:
            role_digests[row["role"]].add(row["canonical_digest"])
    roles = ("training", "development_screening", "confirmation", "sealed_final")
    intersections = {
        f"{left}__{right}": len(role_digests[left] & role_digests[right])
        for left_index, left in enumerate(roles)
        for right in roles[left_index + 1:]
    }
    aggregate = Counter()
    for item in datasets.values():
        aggregate.update(item["role_counts"])
    return {
        "datasets": datasets,
        "aggregate_role_counts": dict(sorted(aggregate.items())),
        "role_intersection_matrix": intersections,
        "all_roles_disjoint": all(value == 0 for value in intersections.values()),
    }


def parent_evidence_checks(
    context: Any, its_lines: list[str], ots_lines: list[str]
) -> dict[str, bool]:
    haze_train = read_json_asset(context, "haze4k_train_scene_summary")
    haze_test = read_json_asset(context, "haze4k_test_scene_summary")
    its_summary = read_json_asset(context, "its_overlap_summary")
    ots_summary = read_json_asset(context, "ots_overlap_summary")
    return {
        "haze4k_train_identity_only": (
            haze_train.get("scene_grouping", {}).get("canonical_scene_count") == 750
            and haze_train.get("frozen_split", {}).get("training_scene_count") == 600
            and haze_train.get("frozen_split", {}).get(
                "internal_development_scene_count"
            ) == 150
        ),
        "haze4k_test_identity_only": (
            haze_test.get("scene_grouping", {}).get("canonical_scene_count") == 250
            and haze_test.get("frozen_split", {}).get("development_scene_count") == 100
            and haze_test.get("frozen_split", {}).get(
                "candidate_confirmation_scene_count"
            ) == 150
        ),
        "its_verified_overlap_membership": (
            len(its_lines) == len(set(its_lines)) == EXPECTED_KNOWN_EXCLUSIONS["ITS"]
            and its_summary.get("quarantine_tiers", {}).get(
                "train_and_selection_exposure", {}
            ).get("authorized_exclusion_ids") == EXPECTED_KNOWN_EXCLUSIONS["ITS"]
        ),
        "ots_verified_overlap_membership": (
            len(ots_lines) == len(set(ots_lines)) == EXPECTED_KNOWN_EXCLUSIONS["OTS"]
            and ots_summary.get("exclusion_pool", {}).get(
                "deduplicated_exclusion_count"
            ) == EXPECTED_KNOWN_EXCLUSIONS["OTS"]
        ),
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    write_contract_progress(
        context,
        completed_iterations=1,
        total_iterations=1,
        stage="outcome_blind_dataset_contract",
    )
    dataset_assets_hidden = all(
        asset_id not in context.assets
        for asset_id in ("haze4k_train", "haze4k_test", "reside_root", "nh_haze_root")
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
            "dataset_directories_hidden": dataset_assets_hidden,
            "no_model_or_metric_path": True,
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
    if context.total_units != 5 or context.device != "cpu":
        raise RuntimeError("program-foundation runtime contract mismatch")
    if any(context.protected_data_permissions.values()):
        raise RuntimeError("program-foundation qualification forbids protected permissions")
    if load_completed_unit_ledger(context):
        raise RuntimeError("fresh program-foundation workload has completed units")

    its_lines, its_exclusions = read_unique_lines(
        asset_path(context, "its_verified_exclusions", kind="file")
    )
    ots_lines, ots_exclusions = read_unique_lines(
        asset_path(context, "ots_verified_exclusions", kind="file")
    )
    evidence_checks = parent_evidence_checks(context, its_lines, ots_lines)

    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    inventory_summaries: dict[str, dict[str, Any]] = {}
    haze4k_root = asset_path(context, "haze4k_root", kind="directory")
    for unit_index, (dataset, function, root) in enumerate((
        ("HAZE4K_TRAIN", inventory_haze4k, haze4k_root / "train"),
        ("HAZE4K_TEST", inventory_haze4k, haze4k_root / "test"),
    ), start=1):
        rows, summary = function(root, dataset)
        rows_by_dataset[dataset] = rows
        inventory_summaries[dataset] = summary
        relpath = f"units/{dataset.lower()}_identity.json"
        atomic_json(output_file(context, relpath), {"summary": summary, "rows": rows})
        record_completed_unit(
            context,
            unit_id=f"{dataset.lower()}_identity",
            input_sha256=unit_input_sha256(dataset, summary),
            output_relpath=relpath,
        )
        write_workload_progress(
            context, completed_units=unit_index, stage=f"{dataset.lower()}_identified"
        )

    reside_root = asset_path(context, "reside_root", kind="directory")
    its_rows, its_inventory = inventory_its(reside_root)
    ots_rows, ots_inventory = inventory_ots(reside_root)
    rows_by_dataset["ITS"] = its_rows
    rows_by_dataset["OTS"] = ots_rows
    inventory_summaries["ITS"] = its_inventory
    inventory_summaries["OTS"] = ots_inventory
    reside_payload = {"ITS": its_inventory, "OTS": ots_inventory}
    reside_relpath = "units/reside_identity.json"
    atomic_json(
        output_file(context, reside_relpath),
        {"summary": reside_payload, "rows": {"ITS": its_rows, "OTS": ots_rows}},
    )
    record_completed_unit(
        context,
        unit_id="reside_identity",
        input_sha256=unit_input_sha256("RESIDE", reside_payload),
        output_relpath=reside_relpath,
    )
    write_workload_progress(context, completed_units=3, stage="reside_identified")

    nh_rows, nh_inventory = inventory_nh_haze(
        asset_path(context, "nh_haze_root", kind="directory")
    )
    rows_by_dataset["NH_HAZE"] = nh_rows
    inventory_summaries["NH_HAZE"] = nh_inventory
    nh_relpath = "units/nh_haze_identity.json"
    atomic_json(output_file(context, nh_relpath), {"summary": nh_inventory, "rows": nh_rows})
    record_completed_unit(
        context,
        unit_id="nh_haze_identity",
        input_sha256=unit_input_sha256("NH_HAZE", nh_inventory),
        output_relpath=nh_relpath,
    )
    write_workload_progress(context, completed_units=4, stage="nh_haze_identified")

    overlap = apply_exclusions_and_exact_deduplication(
        rows_by_dataset, its_exclusions, ots_exclusions
    )
    allocations = {
        "ITS": assign_program_roles(rows_by_dataset["ITS"], "ITS"),
        "OTS": assign_program_roles(rows_by_dataset["OTS"], "OTS"),
    }
    roles = role_summary(rows_by_dataset)

    identity_checks = {
        dataset: all(summary["checks"].values())
        for dataset, summary in inventory_summaries.items()
    }
    historical_partition_checks = {
        "HAZE4K_TRAIN": (
            inventory_summaries["HAZE4K_TRAIN"]["historical_partition_digest"]
            == read_json_asset(context, "haze4k_train_scene_summary").get(
                "frozen_split", {}
            ).get("assignment_digest")
        ),
        "HAZE4K_TEST": (
            inventory_summaries["HAZE4K_TEST"]["historical_partition_digest"]
            == read_json_asset(context, "haze4k_test_scene_summary").get(
                "frozen_split", {}
            ).get("assignment_digest")
        ),
    }
    known_exclusion_ok = (
        overlap["known_exclusion_counts"] == EXPECTED_KNOWN_EXCLUSIONS
    )
    cross_dataset_ok = (
        overlap["protected_role_conflict_count"] == 0 and known_exclusion_ok
    )
    roles_ok = roles["all_roles_disjoint"] and all(
        row["role"] is not None
        for rows in rows_by_dataset.values()
        for row in rows
        if row["exclusion_reason"] is None
    )
    capacity_checks = {
        dataset: all(
            allocations[dataset]["role_counts"].get(role, 0) >= minimum
            for role, minimum in MINIMUM_ROLE_COUNTS[dataset].items()
        )
        for dataset in ("ITS", "OTS")
    }
    capacity_checks.update({
        "HAZE4K_TRAIN": (
            roles["datasets"]["HAZE4K_TRAIN"]["role_counts"]
            == {"development_screening": 150, "training": 600}
        ),
        "HAZE4K_TEST": (
            roles["datasets"]["HAZE4K_TEST"]["role_counts"]
            == {"confirmation": 100, "sealed_final": 150}
        ),
        "NH_HAZE": (
            roles["datasets"]["NH_HAZE"]["role_counts"].get("sealed_final", 0)
            >= 50
            and set(roles["datasets"]["NH_HAZE"]["role_counts"]) == {"sealed_final"}
        ),
    })
    evidence_identity_ok = all(evidence_checks.values()) and all(
        historical_partition_checks.values()
    )
    dataset_identity_ok = all(identity_checks.values())
    capacity_ok = all(capacity_checks.values())
    validity_ok = (
        evidence_identity_ok and dataset_identity_ok and cross_dataset_ok and roles_ok
    )

    ledger_rows = [
        {
            "schema_version": 1,
            "program_id": "daytime_dehazing_spatially_adaptive_restoration_v1",
            "independent_unit": "original_clear_scene",
            **row,
        }
        for dataset in sorted(rows_by_dataset)
        for row in sorted(rows_by_dataset[dataset], key=lambda item: item["scene_id"])
    ]
    ledger_path = output_file(context, RAW_LEDGER_NAME)
    write_once_jsonl(ledger_path, ledger_rows)

    identity_summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "independent_unit": "original_clear_scene",
        "inventories": inventory_summaries,
        "historical_asset_checks": evidence_checks,
        "historical_partition_membership_checks": historical_partition_checks,
        "identity_checks": identity_checks,
        "outcome_blind_structure_access": {
            "clear_targets_decoded_for_canonical_hash_only": True,
            "hazy_images_decoded_for_pair_and_dimension_checks_only": True,
            "images_rendered_or_viewed_by_researcher": False,
            "model_checkpoint_or_prediction_accessed": False,
            "restoration_metric_or_local_error_computed": False,
        },
    }
    role_result = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "program_local_prior_model_selection_exposure": False,
        "declaration_scope": "The new program does not inherit any historical dataset role, exposure label, model-selection use, terminal decision, PASS or FAIL, or authorization.",
        "reused_historical_scope": "Scene identity, grouping, deterministic partition membership, and verified overlap membership only.",
        "allocations": allocations,
        "overlap_control": overlap,
        "roles": roles,
        "capacity_checks": capacity_checks,
        "capacity_claim": "Exact census capacity to author S1 only; S1 must freeze its own effect margin, planning-SD upper bound, comparison family, and formal precision certificate.",
    }
    atomic_json(output_file(context, IDENTITY_NAME), identity_summary)
    atomic_json(output_file(context, ROLE_NAME), role_result)
    record_completed_unit(
        context,
        unit_id="scene_role_contract",
        input_sha256=unit_input_sha256(
            "SCENE_ROLE_CONTRACT",
            {
                "overlap": overlap,
                "roles": roles,
                "capacity_checks": capacity_checks,
            },
        ),
        output_relpath=ROLE_NAME,
    )

    summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "program_id": "daytime_dehazing_spatially_adaptive_restoration_v1",
        "question_answered": "Whether the initial Haze4K, ITS, OTS, and NH-Haze inventory can support a program-local, scene-grouped, known-overlap-controlled four-role contract.",
        "identity_summary_file": IDENTITY_NAME,
        "role_summary_file": ROLE_NAME,
        "evidence_identity_ok": evidence_identity_ok,
        "dataset_identity_ok": dataset_identity_ok,
        "cross_dataset_separation_ok": cross_dataset_ok,
        "role_partition_ok": roles_ok,
        "capacity_ok": capacity_ok,
        "aggregate_role_counts": roles["aggregate_role_counts"],
        "raw_cloud_ledger": {
            "filename": RAW_LEDGER_NAME,
            "rows": len(ledger_rows),
            "sha256": sha256_file(ledger_path),
            "archived_to_github": False,
        },
        "forbidden_activity_receipt": {
            "training_or_fitting_occurred": False,
            "model_checkpoint_accessed": False,
            "inference_or_prediction_accessed": False,
            "restoration_metric_computed": False,
            "candidate_or_threshold_selected": False,
            "historical_scientific_conclusion_inherited": False,
        },
        "limitations": [
            "Cross-dataset independence is certified only for decoded-RGB exact identity and the archived verified Haze4K-to-ITS/OTS overlap memberships.",
            "The contract does not claim complete capture provenance or semantic source disjointness; an unfound geometric relation is not interpreted as absence of overlap.",
            "Haze variants, crops, pixels, regions, augmentations, and resamples never increase independent scene counts.",
            "The 55-scene NH-Haze population is kept intact as sealed-final evidence and cannot alone support a high-precision small-effect claim.",
            "PASS authorizes only S1 contract authoring and is not evidence that a future mechanism effect has adequate precision.",
        ],
        "marker": "DAYTIME_DEHAZING_PROGRAM_FOUNDATION_V1_COMPLETE",
    }
    atomic_json(output_file(context, SUMMARY_NAME), summary)
    write_workload_progress(
        context, completed_units=5, stage="scene_role_contract_finalized"
    )

    write_gate_result(
        context,
        gate_outcomes={
            "evidence_identity": "pass" if evidence_identity_ok else "fail",
            "dataset_identity": "pass" if dataset_identity_ok else "fail",
            "cross_dataset_separation": "pass" if cross_dataset_ok else "fail",
            "role_partition_coverage": "pass" if roles_ok else "fail",
            "independent_capacity": (
                "favorable" if capacity_ok else "unfavorable"
            ) if validity_ok else "invalid",
            "capacity_precision": (
                "met" if capacity_ok else "unmet"
            ) if validity_ok else "invalid",
        },
        details={
            "summary_file": SUMMARY_NAME,
            "identity_summary_file": IDENTITY_NAME,
            "role_summary_file": ROLE_NAME,
            "raw_cloud_ledger_file": RAW_LEDGER_NAME,
            "raw_cloud_ledger_sha256": summary["raw_cloud_ledger"]["sha256"],
            "aggregate_role_counts": roles["aggregate_role_counts"],
            "program_local_prior_model_selection_exposure": False,
            "training_or_inference_occurred": False,
            "restoration_outcomes_accessed": False,
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
