#!/usr/bin/env python3
"""Resolve the remaining Haze4K-to-OTS source mappings conservatively."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.fft import dctn

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
RESAMPLE = Image.Resampling.LANCZOS
TOP_K = 64
MIN_CORRELATION = 0.98
MIN_SCORE_GAP = 0.005
MIN_ELIGIBLE_OTS = 6000
EXPECTED_STRICT_OTS = 458
EXPECTED_STRICT_INDOOR = 500
EXPECTED_UNRESOLVED = 42
EXPECTED_HAZE4K_GROUPS = 1000
EXPECTED_SOTS_OUTDOOR = 492
TOTAL_UNITS = 25962
VIEW_SPECS = (
    (1.00, 0.5, 0.5),
    (0.90, 0.5, 0.5),
    (0.85, 0.0, 0.0),
    (0.85, 1.0, 0.0),
    (0.85, 0.0, 1.0),
    (0.85, 1.0, 1.0),
    (0.80, 0.5, 0.5),
)
BIT_COUNTS = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"missing image directory: {directory}")
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def files(directory: Path, suffix: str) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"missing directory: {directory}")
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix == suffix)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_lines(lines: Iterable[str]) -> str:
    payload = "\n".join(sorted(lines)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "q05": None, "median": None, "q95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def packed_hash(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits.reshape(-1).astype(np.uint8))


def crop_thumbnail(gray: Image.Image, fraction: float, x_anchor: float, y_anchor: float) -> np.ndarray:
    width, height = gray.size
    crop_width = max(8, int(round(width * fraction)))
    crop_height = max(8, int(round(height * fraction)))
    left = int(round((width - crop_width) * x_anchor))
    top = int(round((height - crop_height) * y_anchor))
    crop = gray.crop((left, top, left + crop_width, top + crop_height))
    return np.asarray(crop.resize((32, 32), RESAMPLE), dtype=np.uint8)


def perceptual_hash(thumbnail: np.ndarray) -> np.ndarray:
    low = dctn(thumbnail.astype(np.float64), type=2, norm="ortho")[:8, :8]
    return packed_hash(low > np.median(low[1:]))


def fingerprint(path: Path, source: str, include_views: bool) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        gray = rgb.convert("L")
        canonical = np.asarray(rgb.resize((64, 64), RESAMPLE), dtype=np.uint8)
        direct_gray = np.asarray(gray.resize((32, 32), RESAMPLE), dtype=np.uint8)
        dhash_array = np.asarray(gray.resize((9, 8), RESAMPLE), dtype=np.int16)
        views = np.stack(
            [crop_thumbnail(gray, fraction, x_anchor, y_anchor)
             for fraction, x_anchor, y_anchor in VIEW_SPECS]
        ) if include_views else direct_gray[None, ...]
    return {
        "source": source,
        "id": path.stem,
        "canonical": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "dhash": packed_hash(dhash_array[:, 1:] > dhash_array[:, :-1]),
        "phash": perceptual_hash(direct_gray),
        "direct_gray": direct_gray.reshape(-1),
        "color_mean": canonical.reshape(-1, 3).mean(axis=0) / 255.0,
        "views": views.reshape(len(views), -1),
        "view_hashes": np.stack([perceptual_hash(view) for view in views]),
    }


def fingerprint_batch(
    *,
    paths: list[Path],
    source: str,
    include_views: bool,
    workers: int,
    offset: int,
    context: Any,
) -> list[dict[str, Any]]:
    worker = partial(fingerprint, source=source, include_views=include_views)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, item in enumerate(executor.map(worker, paths, chunksize=8), start=1):
            results.append(item)
            if index == 1 or index % 100 == 0 or index == len(paths):
                write_workload_progress(
                    context,
                    completed_units=offset + index,
                    stage=f"fingerprint_{source.lower()}",
                )
    return results


def hamming_distances(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    return BIT_COUNTS[np.bitwise_xor(reference, query)].sum(axis=-1)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float64)
    y = right.astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-12:
        return 1.0 if float(np.max(np.abs(x - y))) <= 1e-6 else 0.0
    return float(np.dot(x, y) / denominator)


def strict_matches(
    query_groups: dict[str, list[dict[str, Any]]],
    references: list[dict[str, Any]],
    *,
    progress_offset: int,
    context: Any,
) -> tuple[dict[str, dict[str, set[str]]], list[tuple[str, str, str, str, str]]]:
    reference_dhash = np.stack([item["dhash"] for item in references])
    reference_phash = np.stack([item["phash"] for item in references])
    matches: dict[str, dict[str, set[str]]] = {}
    records: set[tuple[str, str, str, str, str]] = set()
    ordered_groups = sorted(query_groups.items())
    for index, (digest, members) in enumerate(ordered_groups, start=1):
        query = members[0]
        d_distance = hamming_distances(reference_dhash, query["dhash"])
        p_distance = hamming_distances(reference_phash, query["phash"])
        candidates = np.flatnonzero((d_distance <= 2) & (p_distance <= 6))
        by_source: dict[str, set[str]] = defaultdict(set)
        for candidate_index in candidates.tolist():
            candidate = references[candidate_index]
            if correlation(query["direct_gray"], candidate["direct_gray"]) < 0.995:
                continue
            if float(np.max(np.abs(query["color_mean"] - candidate["color_mean"]))) > 0.04:
                continue
            by_source[candidate["source"]].add(candidate["id"])
            records.add((
                digest,
                query["source"],
                query["id"],
                candidate["source"],
                candidate["id"],
            ))
        matches[digest] = dict(by_source)
        if index == 1 or index % 25 == 0 or index == len(ordered_groups):
            write_workload_progress(
                context,
                completed_units=progress_offset + index,
                stage="reproduce_strict_haze4k_matches",
            )
    return matches, sorted(records)


def normalize_views(views: np.ndarray) -> np.ndarray:
    values = views.astype(np.float32)
    values -= values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def robust_scores(query_views: np.ndarray, candidate_views: np.ndarray) -> np.ndarray:
    query = normalize_views(query_views)
    candidates = np.stack([normalize_views(views) for views in candidate_views])
    pairwise = np.einsum("af,kbf->kab", query, candidates, optimize=True)
    return np.max(pairwise, axis=(1, 2))


def candidate_order(
    query: dict[str, Any],
    ots_view_hashes: np.ndarray,
    ots_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    full_to_views = hamming_distances(ots_view_hashes, query["view_hashes"][0]).min(axis=1)
    views_to_full = hamming_distances(
        ots_view_hashes[:, 0, :][:, None, :], query["view_hashes"][None, :, :]
    ).min(axis=1)
    distances = np.minimum(full_to_views, views_to_full)
    order = np.lexsort((ots_ids, distances))[:TOP_K]
    return order, distances


def robust_match(
    query: dict[str, Any],
    ots_items: list[dict[str, Any]],
    ots_view_hashes: np.ndarray,
    ots_ids: np.ndarray,
) -> dict[str, Any]:
    candidates, distances = candidate_order(query, ots_view_hashes, ots_ids)
    candidate_views = np.stack([ots_items[index]["views"] for index in candidates.tolist()])
    scores = robust_scores(query["views"], candidate_views)
    ranked_positions = sorted(
        range(len(candidates)),
        key=lambda position: (
            -float(scores[position]),
            int(distances[candidates[position]]),
            str(ots_ids[candidates[position]]),
        ),
    )
    best_position, second_position = ranked_positions[:2]
    best_index = int(candidates[best_position])
    second_index = int(candidates[second_position])
    return {
        "candidate_indices": candidates,
        "top_k_ids": [str(ots_ids[index]) for index in candidates.tolist()],
        "best_index": best_index,
        "best_id": str(ots_ids[best_index]),
        "best_score": float(scores[best_position]),
        "best_distance": int(distances[best_index]),
        "second_id": str(ots_ids[second_index]),
        "second_score": float(scores[second_position]),
        "score_gap": float(scores[best_position] - scores[second_position]),
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    prior = context.assets.get("prior_reside_qualification")
    indoor = context.assets.get("indoor_measurement_fail")
    checks = {
        "metadata_only_mode": context.engineering_contract["mode"] == "metadata_only",
        "cpu_contract": context.device == "cpu",
        "datasets_hidden_from_contract": (
            "reside_root" not in context.assets and "haze4k_root" not in context.assets
        ),
        "prior_qualification_identity_bound": (
            prior is not None and prior.contract_access is True
        ),
        "indoor_failure_identity_bound": (
            indoor is not None and indoor.contract_access is True
        ),
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
        "no_model_training_or_inference_path": True,
        "bounded_matcher_contract": True,
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
    workers = int(os.environ.get("CONVIR_ROUTE_OTS_WORKERS", "8"))
    if workers != 8:
        raise ValueError("worker count differs from the frozen value 8")
    if int(os.environ.get("CONVIR_ROUTE_OTS_MATCH_TOPK", "64")) != TOP_K:
        raise ValueError("matcher top-k differs from the frozen value 64")
    if float(os.environ.get("CONVIR_ROUTE_OTS_MIN_CORRELATION", "0.98")) != MIN_CORRELATION:
        raise ValueError("minimum correlation differs from the frozen value")
    if float(os.environ.get("CONVIR_ROUTE_OTS_MIN_SCORE_GAP", "0.005")) != MIN_SCORE_GAP:
        raise ValueError("score gap differs from the frozen value")

    reside = asset_path(context, "reside_root", kind="directory")
    haze4k = asset_path(context, "haze4k_root", kind="directory")
    qualification_path = asset_path(context, "prior_reside_qualification", kind="file")
    indoor_path = asset_path(context, "indoor_measurement_fail", kind="file")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    indoor = json.loads(indoor_path.read_text(encoding="utf-8"))
    if qualification.get("marker") != "RESIDE_MINIMAL_MEASUREMENT_QUALIFICATION_RECORDED":
        raise ValueError("prior RESIDE qualification marker is missing")
    if (
        qualification.get("decision", {}).get("ots_training_role")
        != "BLOCKED_IDENTIFICATION_PENDING_42_SOURCE_MAPPINGS"
    ):
        raise ValueError("prior qualification does not identify the 42-source OTS blocker")
    if (
        indoor.get("decision") != "ITS_LOCAL_MEASUREMENT_MAPPING_FAIL"
        or indoor.get("authorizes") != "NONE"
    ):
        raise ValueError("indoor terminal identity differs from the archived FAIL")

    identity_paths = {
        "archive_manifest_sha256": reside / "ARCHIVE_SHA256SUMS.txt",
        "pairing_report_sha256": reside / "PAIRING_VALIDATION.txt",
        "layout_record_sha256": reside / "DATASET_LAYOUT.txt",
    }
    observed_identity = {key: sha256_file(path) for key, path in identity_paths.items()}
    qualified_identity = qualification["dataset_identity"]
    identity_match = all(
        observed_identity[key] == qualified_identity[key] for key in observed_identity
    )

    its_train_paths = image_files(reside / "official/ITS/train/ITS_clear")
    its_validation_paths = image_files(reside / "official/ITS/val/clear")
    ots_paths = image_files(reside / "official/OTS_ALPHA/clear_images")
    ots_haze_paths = image_files(reside / "official/OTS_ALPHA/OTS")
    ots_depth_paths = files(reside / "official/OTS_ALPHA/depth", ".mat")
    haze4k_train_paths = image_files(haze4k / "train/gt")
    haze4k_test_paths = image_files(haze4k / "test/gt")
    sots_outdoor_paths = image_files(reside / "official/SOTS/outdoor/gt")
    observed_counts = {
        "its_train_clear": len(its_train_paths),
        "its_validation_clear": len(its_validation_paths),
        "ots_clear": len(ots_paths),
        "ots_hazy": len(ots_haze_paths),
        "ots_depth": len(ots_depth_paths),
        "haze4k_gt": len(haze4k_train_paths) + len(haze4k_test_paths),
        "sots_outdoor_gt": len(sots_outdoor_paths),
    }
    count_integrity = observed_counts == {
        "its_train_clear": 10000,
        "its_validation_clear": 1000,
        "ots_clear": 8970,
        "ots_hazy": 313950,
        "ots_depth": 8970,
        "haze4k_gt": 4000,
        "sots_outdoor_gt": 492,
    }

    offset = 0
    its_train = fingerprint_batch(
        paths=its_train_paths, source="RESIDE_ITS_TRAIN", include_views=False,
        workers=workers, offset=offset, context=context,
    )
    offset += len(its_train_paths)
    its_validation = fingerprint_batch(
        paths=its_validation_paths, source="RESIDE_ITS_VAL", include_views=False,
        workers=workers, offset=offset, context=context,
    )
    offset += len(its_validation_paths)
    ots = fingerprint_batch(
        paths=ots_paths, source="RESIDE_OTS", include_views=True,
        workers=workers, offset=offset, context=context,
    )
    offset += len(ots_paths)
    haze4k_train = fingerprint_batch(
        paths=haze4k_train_paths, source="HAZE4K_TRAIN", include_views=True,
        workers=workers, offset=offset, context=context,
    )
    offset += len(haze4k_train_paths)
    haze4k_test = fingerprint_batch(
        paths=haze4k_test_paths, source="HAZE4K_TEST", include_views=True,
        workers=workers, offset=offset, context=context,
    )
    offset += len(haze4k_test_paths)
    sots_outdoor = fingerprint_batch(
        paths=sots_outdoor_paths, source="RESIDE_SOTS_OUTDOOR", include_views=False,
        workers=workers, offset=offset, context=context,
    )
    offset += len(sots_outdoor_paths)
    if offset != 24462:
        raise ValueError(f"fingerprint workload differs from frozen size: {offset}")

    haze4k_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in haze4k_train + haze4k_test:
        haze4k_groups[item["canonical"]].append(item)
    strict, strict_records = strict_matches(
        dict(haze4k_groups), its_train + its_validation + ots,
        progress_offset=offset, context=context,
    )
    offset += len(haze4k_groups)

    indoor_digests = {
        digest for digest, match in strict.items()
        if match.get("RESIDE_ITS_TRAIN") or match.get("RESIDE_ITS_VAL")
    }
    ots_digests = {
        digest for digest, match in strict.items() if match.get("RESIDE_OTS")
    }
    ambiguous_digests = indoor_digests & ots_digests
    unresolved_digests = sorted(set(haze4k_groups) - indoor_digests - ots_digests)
    strict_ots_ids = sorted({
        scene
        for digest in ots_digests
        for scene in strict[digest].get("RESIDE_OTS", set())
    })
    strict_mapping_digest = digest_lines("|".join(record) for record in strict_records)
    strict_ots_exclusion_digest = digest_lines(
        f"RESIDE_OTS:{scene}" for scene in strict_ots_ids
    )
    prior_haze4k = qualification["scene_independence"]["haze4k"]
    prior_reproduced = (
        len(haze4k_groups) == EXPECTED_HAZE4K_GROUPS
        and len(indoor_digests) == EXPECTED_STRICT_INDOOR
        and len(ots_digests) == EXPECTED_STRICT_OTS
        and len(ambiguous_digests) == 0
        and len(unresolved_digests) == EXPECTED_UNRESOLVED
        and strict_mapping_digest == prior_haze4k["mapping_digest"]
        and strict_ots_exclusion_digest == prior_haze4k["ots_exclusion_id_set_digest"]
    )

    ots_by_canonical: dict[str, list[str]] = defaultdict(list)
    for item in ots:
        ots_by_canonical[item["canonical"]].append(item["id"])
    sots_ots_ids = sorted({
        scene for item in sots_outdoor
        for scene in ots_by_canonical.get(item["canonical"], [])
    })
    sots_exact_group_count = sum(
        bool(ots_by_canonical.get(item["canonical"])) for item in sots_outdoor
    )
    sots_exclusion_digest = digest_lines(
        f"RESIDE_OTS:{scene}" for scene in sots_ots_ids
    )
    prior_sots = qualification["scene_independence"]["sots"]
    sots_reproduced = (
        sots_exact_group_count == EXPECTED_SOTS_OUTDOOR
        and len(sots_ots_ids) == EXPECTED_SOTS_OUTDOOR
        and sots_exclusion_digest == prior_sots["ots_exclusion_id_set_digest"]
    )

    ots_view_hashes = np.stack([item["view_hashes"] for item in ots])
    ots_ids = np.asarray([item["id"] for item in ots])
    ots_index = {item["id"]: index for index, item in enumerate(ots)}
    known_digests = sorted(ots_digests - indoor_digests)
    match_digests = known_digests + unresolved_digests
    known_top_k = 0
    known_top_one = 0
    known_scores: list[float] = []
    unresolved_records: list[dict[str, Any]] = []
    unresolved_queries: list[dict[str, Any]] = []
    robust_results: dict[str, dict[str, Any]] = {}
    if prior_reproduced and len(match_digests) == 500:
        for index, digest in enumerate(match_digests, start=1):
            query = haze4k_groups[digest][0]
            result = robust_match(query, ots, ots_view_hashes, ots_ids)
            robust_results[digest] = result
            if digest in ots_digests:
                verified = strict[digest]["RESIDE_OTS"]
                known_top_k += int(bool(set(result["top_k_ids"]) & verified))
                known_top_one += int(result["best_id"] in verified)
                known_scores.append(result["best_score"])
            else:
                unresolved_queries.append(query)
                unresolved_records.append({
                    "haze4k_group_digest": digest,
                    "haze4k_representative_source": query["source"],
                    "haze4k_representative_id": query["id"],
                    "ots_scene_id": result["best_id"],
                    "candidate_hamming_distance": result["best_distance"],
                    "correlation_score": result["best_score"],
                    "score_gap": result["score_gap"],
                    "mutual_nearest": False,
                    "verified": False,
                })
            if index == 1 or index % 10 == 0 or index == len(match_digests):
                write_workload_progress(
                    context,
                    completed_units=offset + index,
                    stage="calibrated_multiview_source_retrieval",
                )
        offset += len(match_digests)
    else:
        write_workload_progress(
            context,
            completed_units=TOTAL_UNITS,
            stage="retrieval_skipped_prior_identity_drift",
        )
        offset = TOTAL_UNITS

    selected_ids = [record["ots_scene_id"] for record in unresolved_records]
    selected_counts = Counter(selected_ids)
    selected_indices = [ots_index[scene] for scene in selected_ids]
    if unresolved_queries and len(unresolved_queries) == len(selected_indices):
        score_matrix = np.stack([
            robust_scores(
                query["views"],
                np.stack([ots[index]["views"] for index in selected_indices]),
            )
            for query in unresolved_queries
        ])
        for column, record in enumerate(unresolved_records):
            ranking = sorted(
                range(len(unresolved_records)),
                key=lambda row: (-float(score_matrix[row, column]), row),
            )
            best_row, second_row = ranking[:2]
            mutual_gap = float(score_matrix[best_row, column] - score_matrix[second_row, column])
            mutual = best_row == column and mutual_gap >= MIN_SCORE_GAP
            record["mutual_nearest"] = mutual
            record["verified"] = (
                selected_counts[record["ots_scene_id"]] == 1
                and record["correlation_score"] >= MIN_CORRELATION
                and record["score_gap"] >= MIN_SCORE_GAP
                and mutual
            )

    resolved_ids = sorted({
        record["ots_scene_id"] for record in unresolved_records if record["verified"]
    })
    candidate_exclusions = sorted(set(strict_ots_ids) | set(selected_ids) | set(sots_ots_ids))
    eligible_ots = len(ots) - len(candidate_exclusions)
    exclusion_digest = digest_lines(candidate_exclusions)
    known_count = len(known_digests)
    gates = {
        "dataset_identity": identity_match and count_integrity,
        "prior_strict_mapping_reproduced": prior_reproduced,
        "sots_outdoor_overlap_reproduced": sots_reproduced,
        "known_match_top_k_recall": known_count == EXPECTED_STRICT_OTS and known_top_k == known_count,
        "known_match_top_one_recovery": known_count == EXPECTED_STRICT_OTS and known_top_one == known_count,
        "unresolved_mapping_completeness": len(resolved_ids) == EXPECTED_UNRESOLVED,
        "unresolved_mapping_uniqueness": len(set(selected_ids)) == EXPECTED_UNRESOLVED,
        "unresolved_mapping_score_floor": (
            len(unresolved_records) == EXPECTED_UNRESOLVED
            and min(record["correlation_score"] for record in unresolved_records) >= MIN_CORRELATION
        ),
        "unresolved_mapping_gap_floor": (
            len(unresolved_records) == EXPECTED_UNRESOLVED
            and min(record["score_gap"] for record in unresolved_records) >= MIN_SCORE_GAP
        ),
        "unresolved_mapping_mutual_nearest": (
            len(unresolved_records) == EXPECTED_UNRESOLVED
            and all(record["mutual_nearest"] for record in unresolved_records)
        ),
        "eligible_ots_capacity": eligible_ots >= MIN_ELIGIBLE_OTS,
    }
    identity_gates = (
        gates["dataset_identity"]
        and gates["prior_strict_mapping_reproduced"]
        and gates["sots_outdoor_overlap_reproduced"]
    )
    if not identity_gates:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "OTS_OVERLAP_RESOLUTION_INCONCLUSIVE"
        authorizes = "NONE"
        gate_reasons = [name for name in (
            "dataset_identity",
            "prior_strict_mapping_reproduced",
            "sots_outdoor_overlap_reproduced",
        ) if not gates[name]]
    elif all(gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "OTS_OVERLAP_RESOLUTION_PASS"
        authorizes = "OTS_OUTDOOR_MEASUREMENT_DESIGN"
        gate_reasons = [
            "all frozen identity, calibration, unique-mapping, mutual-nearest, and capacity gates passed"
        ]
    else:
        state = "COMPLETED_GATE_FAIL"
        decision = "OTS_OVERLAP_RESOLUTION_FAIL"
        authorizes = "NONE"
        gate_reasons = [name for name, passed in gates.items() if not passed]

    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "RESIDE OTS source-scene qualification for synthetic outdoor design",
        "dataset_identity": {
            "observed_identity": observed_identity,
            "qualified_identity_match": identity_match,
            "observed_counts": observed_counts,
            "count_integrity": count_integrity,
            "haze4k_unique_canonical_scene_count": len(haze4k_groups),
        },
        "strict_reproduction": {
            "indoor_haze4k_group_count": len(indoor_digests),
            "ots_haze4k_group_count": len(ots_digests),
            "cross_source_ambiguous_group_count": len(ambiguous_digests),
            "unresolved_haze4k_group_count": len(unresolved_digests),
            "mapping_digest": strict_mapping_digest,
            "ots_exclusion_digest": strict_ots_exclusion_digest,
            "prior_reproduced": prior_reproduced,
            "sots_outdoor_exact_group_count": sots_exact_group_count,
            "sots_ots_exclusion_digest": sots_exclusion_digest,
            "sots_prior_reproduced": sots_reproduced,
        },
        "robust_retrieval": {
            "method": (
                "Top-64 by minimum symmetric multi-view pHash distance, then maximum "
                "normalized grayscale correlation across seven fixed full/crop views."
            ),
            "top_k": TOP_K,
            "minimum_correlation": MIN_CORRELATION,
            "minimum_score_gap": MIN_SCORE_GAP,
            "known_strict_mapping_count": known_count,
            "known_top_k_recovered": known_top_k,
            "known_top_one_recovered": known_top_one,
            "known_top_one_score": quantiles(known_scores),
            "unresolved_candidate_count": len(unresolved_records),
            "verified_unresolved_count": len(resolved_ids),
            "candidate_score": quantiles([
                record["correlation_score"] for record in unresolved_records
            ]),
            "candidate_score_gap": quantiles([
                record["score_gap"] for record in unresolved_records
            ]),
            "unique_selected_ots_count": len(set(selected_ids)),
            "mutual_nearest_count": sum(
                bool(record["mutual_nearest"]) for record in unresolved_records
            ),
            "interpretation_limit": (
                "A FAIL means only that this frozen retrieval method did not close the "
                "mapping; it does not prove that an OTS source mapping is absent."
            ),
        },
        "exclusion_pool": {
            "strict_haze4k_ots_ids": len(strict_ots_ids),
            "resolved_haze4k_ots_ids": len(resolved_ids),
            "sots_outdoor_ots_ids": len(sots_ots_ids),
            "candidate_union_exclusion_count": len(candidate_exclusions),
            "candidate_union_exclusion_digest": exclusion_digest,
            "authorized_exclusion_digest": exclusion_digest if state == "COMPLETED_GATE_PASS" else None,
            "eligible_ots_scene_count": eligible_ots,
            "minimum_required_eligible_scenes": MIN_ELIGIBLE_OTS,
            "future_design_capacity": (
                "1,000 definition scenes plus 1,000 validation scenes, leaving at least "
                "4,000 eligible OTS scenes untouched when the capacity gate passes"
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
            "This operation qualifies source identity and capacity only; it does not validate an outdoor measurement field.",
            "Haze4K and SOTS-outdoor clear images are used only for source exclusion, not for model evaluation or threshold selection.",
            "No indoor q result is transported to the outdoor domain.",
            "No model, training, inference, checkpoint, hazy test outcome, canary, confirmation, or locked-test data is accessed.",
        ],
        "marker": "RESIDE_OTS_OUTDOOR_OVERLAP_RESOLUTION_COMPLETE",
    }
    output_file(context, "overlap_resolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    mapping_path = output_file(context, "unresolved_source_mapping.csv")
    fieldnames = [
        "haze4k_group_digest",
        "haze4k_representative_source",
        "haze4k_representative_id",
        "ots_scene_id",
        "candidate_hamming_distance",
        "correlation_score",
        "score_gap",
        "mutual_nearest",
        "verified",
    ]
    with mapping_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unresolved_records)
    output_file(context, "ots_exclusion_ids.txt").write_text(
        "\n".join(candidate_exclusions) + "\n", encoding="utf-8",
    )
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "overlap_resolution_summary.json",
            "strict_ots_mappings": len(ots_digests),
            "verified_unresolved_mappings": len(resolved_ids),
            "eligible_ots_scenes": eligible_ots,
            "exclusion_digest": exclusion_digest,
            "gate_reasons": gate_reasons,
            "outdoor_measurement_executed": False,
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
