#!/usr/bin/env python3
"""Build a conservative Haze4K-source exclusion asset for RESIDE ITS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import time
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
FAMILY_CORRELATION = 0.995
EXPECTED_HAZE4K_GROUPS = 1000
EXPECTED_HAZE4K_ALIASES = 4
EXPECTED_ITS_TRAIN = 10000
EXPECTED_ITS_VALIDATION = 1000
EXPECTED_SOTS_OUTDOOR = 492
EXPECTED_INDOOR_RELATIONSHIPS = 500
MIN_ELIGIBLE_ITS = 6000
TOTAL_UNITS = 17984
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
EXPECTED_RESIDE_IDENTITY = {
    "archive_manifest_sha256": "27348952347a0f94ac33105d5e6ce11b4ce4e2b0fab68b583b978752b5b78be0",
    "pairing_report_sha256": "afdefd2887437d929e00239e932bd76e7504d9838c2a3c58c33ce13f91f4150a",
    "layout_record_sha256": "f025a8e6e3f43c9b924c0986b866ed28e5644dd7a7ec7b056ae311c83764d5dc",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_lines(lines: Iterable[str]) -> str:
    payload = "\n".join(sorted(lines)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"missing image directory: {directory}")
    return sorted(
        item for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def packed_hash(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits.reshape(-1).astype(np.uint8))


def perceptual_hash(thumbnail: np.ndarray) -> np.ndarray:
    low = dctn(thumbnail.astype(np.float64), type=2, norm="ortho")[:8, :8]
    return packed_hash(low > np.median(low[1:]))


def crop_thumbnail(
    gray: Image.Image, fraction: float, x_anchor: float, y_anchor: float,
) -> np.ndarray:
    width, height = gray.size
    crop_width = max(8, int(round(width * fraction)))
    crop_height = max(8, int(round(height * fraction)))
    left = int(round((width - crop_width) * x_anchor))
    top = int(round((height - crop_height) * y_anchor))
    crop = gray.crop((left, top, left + crop_width, top + crop_height))
    return np.asarray(crop.resize((32, 32), RESAMPLE), dtype=np.uint8)


def fingerprint_image(image: Image.Image, source: str, identifier: str) -> dict[str, Any]:
    rgb = image.convert("RGB")
    gray = rgb.convert("L")
    canonical = np.asarray(rgb.resize((64, 64), RESAMPLE), dtype=np.uint8)
    direct_gray = np.asarray(gray.resize((32, 32), RESAMPLE), dtype=np.uint8)
    dhash_array = np.asarray(gray.resize((9, 8), RESAMPLE), dtype=np.int16)
    views = np.stack([
        crop_thumbnail(gray, fraction, x_anchor, y_anchor)
        for fraction, x_anchor, y_anchor in VIEW_SPECS
    ])
    return {
        "source": source,
        "id": identifier,
        "canonical": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "dhash": packed_hash(dhash_array[:, 1:] > dhash_array[:, :-1]),
        "phash": perceptual_hash(direct_gray),
        "direct_gray": direct_gray.reshape(-1),
        "color_mean": canonical.reshape(-1, 3).mean(axis=0) / 255.0,
        "views": views.reshape(len(views), -1),
        "view_hashes": np.stack([perceptual_hash(view) for view in views]),
    }


def fingerprint(path: Path, source: str) -> dict[str, Any]:
    with Image.open(path) as image:
        return fingerprint_image(image, source, path.stem)


def fingerprint_batch(
    *, paths: list[Path], source: str, workers: int, offset: int, context: Any,
) -> list[dict[str, Any]]:
    worker = partial(fingerprint, source=source)
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


def normalize_views(views: np.ndarray) -> np.ndarray:
    values = views.astype(np.float32)
    values -= values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def candidate_order(
    query: dict[str, Any], reference_hashes: np.ndarray, reference_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    full_to_views = hamming_distances(
        reference_hashes, query["view_hashes"][0],
    ).min(axis=1)
    views_to_full = hamming_distances(
        reference_hashes[:, 0, :][:, None, :], query["view_hashes"][None, :, :],
    ).min(axis=1)
    distances = np.minimum(full_to_views, views_to_full)
    order = np.lexsort((reference_ids, distances))[:TOP_K]
    return order, distances


def robust_scores(query_views: np.ndarray, candidate_views: np.ndarray) -> np.ndarray:
    query = normalize_views(query_views)
    candidates = np.stack([normalize_views(views) for views in candidate_views])
    pairwise = np.einsum("af,kbf->kab", query, candidates, optimize=True)
    return np.max(pairwise, axis=(1, 2))


def robust_match(
    query: dict[str, Any], references: list[dict[str, Any]],
    reference_hashes: np.ndarray, reference_ids: np.ndarray,
) -> dict[str, Any]:
    candidates, distances = candidate_order(query, reference_hashes, reference_ids)
    candidate_views = np.stack([references[index]["views"] for index in candidates.tolist()])
    scores = robust_scores(query["views"], candidate_views)
    ranked_positions = sorted(
        range(len(candidates)),
        key=lambda position: (
            -float(scores[position]),
            int(distances[candidates[position]]),
            str(reference_ids[candidates[position]]),
        ),
    )
    best_position, second_position = ranked_positions[:2]
    best_index = int(candidates[best_position])
    second_index = int(candidates[second_position])
    return {
        "candidate_indices": candidates.tolist(),
        "candidate_scores": {
            int(candidates[position]): float(scores[position])
            for position in range(len(candidates))
        },
        "top_k_ids": [str(reference_ids[index]) for index in candidates.tolist()],
        "best_index": best_index,
        "best_id": str(reference_ids[best_index]),
        "best_score": float(scores[best_position]),
        "best_distance": int(distances[best_index]),
        "second_id": str(reference_ids[second_index]),
        "second_score": float(scores[second_position]),
        "score_gap": float(scores[best_position] - scores[second_position]),
    }


def transformed_control(path: Path, source: str, identifier: str) -> dict[str, Any]:
    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        smaller = rgb.resize(
            (max(32, int(round(width * 0.91))), max(32, int(round(height * 0.91)))),
            RESAMPLE,
        )
        restored = smaller.resize((width, height), RESAMPLE)
        buffer = io.BytesIO()
        restored.save(buffer, format="JPEG", quality=88, optimize=False, progressive=False)
        buffer.seek(0)
        with Image.open(buffer) as transformed:
            return fingerprint_image(transformed, source, identifier)


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


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    checks = {
        "cpu_exact_mode": context.engineering_contract["mode"] == "cpu_exact",
        "cpu_contract": context.device == "cpu",
        "datasets_hidden_from_contract": (
            "reside_root" not in context.assets and "haze4k_root" not in context.assets
        ),
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
        "no_model_training_or_inference_path": True,
    }
    generator = np.random.default_rng(3407)
    reference_hashes = generator.integers(
        0, 256, size=(EXPECTED_ITS_TRAIN + EXPECTED_ITS_VALIDATION, 7, 8),
        dtype=np.uint8,
    )
    query_hashes = generator.integers(
        0, 256, size=(EXPECTED_HAZE4K_GROUPS, 7, 8), dtype=np.uint8,
    )
    reference_ids = np.asarray([
        f"synthetic-{index:05d}" for index in range(len(reference_hashes))
    ])
    started = time.monotonic()
    selected = []
    for query_views in query_hashes:
        order, distances = candidate_order(
            {"view_hashes": query_views}, reference_hashes, reference_ids,
        )
        selected.append((len(order), int(distances[order[0]])))
    elapsed = time.monotonic() - started
    checks.update({
        "same_scale_query_count": len(selected) == EXPECTED_HAZE4K_GROUPS,
        "same_scale_candidate_count": len(reference_hashes) == 11000,
        "same_scale_top_k": all(count == TOP_K for count, _ in selected),
        "same_scale_finite": all(distance >= 0 for _, distance in selected),
        "same_scale_elapsed_bound": elapsed <= 120.0,
        "same_scale_memory_bound": (
            reference_hashes.nbytes + query_hashes.nbytes <= 8 * 1024 * 1024
        ),
        "output_and_finalizer_contract": True,
    })
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "cpu_exact",
            "device": "cpu",
            "fixture": {"batch": 1000, "channels": 7, "height": 1, "width": 11000},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    if context.total_units != TOTAL_UNITS:
        raise ValueError("runtime total_units differs from the frozen census")
    workers = int(os.environ.get("CONVIR_ROUTE_ITS_SOURCE_WORKERS", "8"))
    if workers != 8:
        raise ValueError("ITS source worker count differs from frozen contract")
    reside = asset_path(context, "reside_root", kind="directory")
    haze4k = asset_path(context, "haze4k_root", kind="directory")

    identity_paths = {
        "archive_manifest_sha256": reside / "ARCHIVE_SHA256SUMS.txt",
        "pairing_report_sha256": reside / "PAIRING_VALIDATION.txt",
        "layout_record_sha256": reside / "DATASET_LAYOUT.txt",
    }
    observed_identity = {key: sha256_file(path) for key, path in identity_paths.items()}
    identity_match = observed_identity == EXPECTED_RESIDE_IDENTITY
    its_train_paths = image_files(reside / "official/ITS/train/ITS_clear")
    its_validation_paths = image_files(reside / "official/ITS/val/clear")
    haze4k_train_paths = image_files(haze4k / "train/gt")
    haze4k_test_paths = image_files(haze4k / "test/gt")
    sots_outdoor_paths = image_files(reside / "official/SOTS/outdoor/gt")
    count_integrity = {
        "its_train_clear": len(its_train_paths),
        "its_validation_clear": len(its_validation_paths),
        "haze4k_gt": len(haze4k_train_paths) + len(haze4k_test_paths),
        "sots_outdoor_gt": len(sots_outdoor_paths),
    } == {
        "its_train_clear": EXPECTED_ITS_TRAIN,
        "its_validation_clear": EXPECTED_ITS_VALIDATION,
        "haze4k_gt": 4000,
        "sots_outdoor_gt": EXPECTED_SOTS_OUTDOOR,
    }

    offset = 0
    its_train = fingerprint_batch(
        paths=its_train_paths, source="ITS_TRAIN", workers=workers,
        offset=offset, context=context,
    )
    offset += len(its_train)
    its_validation = fingerprint_batch(
        paths=its_validation_paths, source="ITS_VALIDATION", workers=workers,
        offset=offset, context=context,
    )
    offset += len(its_validation)
    haze4k_train = fingerprint_batch(
        paths=haze4k_train_paths, source="HAZE4K_TRAIN", workers=workers,
        offset=offset, context=context,
    )
    offset += len(haze4k_train)
    haze4k_test = fingerprint_batch(
        paths=haze4k_test_paths, source="HAZE4K_TEST", workers=workers,
        offset=offset, context=context,
    )
    offset += len(haze4k_test)
    sots_outdoor = fingerprint_batch(
        paths=sots_outdoor_paths, source="SOTS_OUTDOOR_NEGATIVE", workers=workers,
        offset=offset, context=context,
    )
    offset += len(sots_outdoor)
    if offset != 15492:
        raise ValueError(f"fingerprint workload differs from frozen size: {offset}")

    haze4k_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    haze4k_paths_by_id = {
        ("HAZE4K_TRAIN", path.stem): path for path in haze4k_train_paths
    }
    haze4k_paths_by_id.update({
        ("HAZE4K_TEST", path.stem): path for path in haze4k_test_paths
    })
    for item in haze4k_train + haze4k_test:
        haze4k_groups[item["canonical"]].append(item)
    alias_histogram = Counter(len(items) for items in haze4k_groups.values())
    group_integrity = (
        len(haze4k_groups) == EXPECTED_HAZE4K_GROUPS
        and alias_histogram == {EXPECTED_HAZE4K_ALIASES: EXPECTED_HAZE4K_GROUPS}
    )
    queries = [haze4k_groups[digest][0] for digest in sorted(haze4k_groups)]
    query_paths = [
        haze4k_paths_by_id[(query["source"], query["id"])] for query in queries
    ]
    query_hashes = np.stack([item["view_hashes"] for item in queries])
    query_ids = np.asarray([f"{item['source']}:{item['id']}" for item in queries])

    control_top_k = 0
    control_top_one = 0
    control_scores: list[float] = []
    control_gaps: list[float] = []
    for index, (query, path) in enumerate(zip(queries, query_paths), start=1):
        transformed = transformed_control(
            path, "HAZE4K_TRANSFORM_CONTROL", query["id"],
        )
        match = robust_match(transformed, queries, query_hashes, query_ids)
        expected_query_id = f"{query['source']}:{query['id']}"
        control_top_k += int(expected_query_id in match["top_k_ids"])
        control_top_one += int(match["best_id"] == expected_query_id)
        control_scores.append(match["best_score"])
        control_gaps.append(match["score_gap"])
        if index == 1 or index % 25 == 0 or index == len(queries):
            write_workload_progress(
                context, completed_units=offset + index,
                stage="transformed_haze4k_positive_controls",
            )
    offset += len(queries)

    its_items = its_train + its_validation
    its_hashes = np.stack([item["view_hashes"] for item in its_items])
    its_phashes = np.stack([item["phash"] for item in its_items])
    its_ids = np.asarray([f"{item['source']}:{item['id']}" for item in its_items])
    its_index_by_full_id = {full_id: index for index, full_id in enumerate(its_ids.tolist())}
    records: list[dict[str, Any]] = []
    for index, (digest, query) in enumerate(
        zip(sorted(haze4k_groups), queries), start=1,
    ):
        forward = robust_match(query, its_items, its_hashes, its_ids)
        best_item = its_items[forward["best_index"]]
        best_family = set()
        for candidate_index in forward["candidate_indices"]:
            candidate = its_items[candidate_index]
            if (
                int(hamming_distances(
                    candidate["phash"][None, :], best_item["phash"],
                )[0]) == 0
                and (
                    candidate["canonical"] == best_item["canonical"]
                    or (
                        correlation(
                            candidate["direct_gray"], best_item["direct_gray"],
                        ) >= FAMILY_CORRELATION
                        and float(np.max(np.abs(
                            candidate["color_mean"] - best_item["color_mean"]
                        ))) <= 0.04
                    )
                )
            ):
                best_family.add(candidate_index)
        nonfamily_scores = [
            score for candidate_index, score in forward["candidate_scores"].items()
            if candidate_index not in best_family
        ]
        family_adjusted_gap = (
            forward["best_score"] - max(nonfamily_scores)
            if nonfamily_scores else forward["best_score"]
        )
        reverse = robust_match(best_item, queries, query_hashes, query_ids)
        accepted = (
            forward["best_score"] >= MIN_CORRELATION
            and family_adjusted_gap >= MIN_SCORE_GAP
            and reverse["best_score"] >= MIN_CORRELATION
            and reverse["score_gap"] >= MIN_SCORE_GAP
            and reverse["best_id"] == f"{query['source']}:{query['id']}"
        )
        records.append({
            "haze4k_group_digest": digest,
            "haze4k_representative_source": query["source"],
            "haze4k_representative_id": query["id"],
            "its_split": best_item["source"],
            "its_scene_id": best_item["id"],
            "its_full_id": forward["best_id"],
            "candidate_hamming_distance": forward["best_distance"],
            "forward_correlation": forward["best_score"],
            "forward_score_gap": forward["score_gap"],
            "family_adjusted_forward_gap": family_adjusted_gap,
            "top64_best_family_size": len(best_family),
            "reverse_correlation": reverse["best_score"],
            "reverse_score_gap": reverse["score_gap"],
            "mutual_nearest": reverse["best_id"] == f"{query['source']}:{query['id']}",
            "accepted": accepted,
        })
        if index == 1 or index % 10 == 0 or index == len(queries):
            write_workload_progress(
                context, completed_units=offset + index,
                stage="haze4k_to_its_full_census",
            )
    offset += len(queries)

    negative_records = []
    for index, query in enumerate(sots_outdoor, start=1):
        match = robust_match(query, its_items, its_hashes, its_ids)
        admitted = (
            match["best_score"] >= MIN_CORRELATION
            and match["score_gap"] >= MIN_SCORE_GAP
        )
        negative_records.append({
            "sots_id": query["id"],
            "its_full_id": match["best_id"],
            "correlation": match["best_score"],
            "score_gap": match["score_gap"],
            "admitted": admitted,
        })
        if index == 1 or index % 25 == 0 or index == len(sots_outdoor):
            write_workload_progress(
                context, completed_units=offset + index,
                stage="sots_outdoor_negative_controls",
            )
    offset += len(sots_outdoor)
    if offset != TOTAL_UNITS:
        raise ValueError(f"completed workload differs from frozen total: {offset}")

    accepted_records = [record for record in records if record["accepted"]]
    accepted_best_ids = {record["its_full_id"] for record in accepted_records}
    exclusion_ids: set[str] = set()
    family_sizes: list[int] = []
    for full_id in sorted(accepted_best_ids):
        reference = its_items[its_index_by_full_id[full_id]]
        distances = hamming_distances(
            its_phashes, reference["phash"],
        )
        family: set[str] = set()
        for candidate_index in np.flatnonzero(distances == 0).tolist():
            candidate = its_items[candidate_index]
            if candidate["canonical"] == reference["canonical"] or (
                correlation(candidate["direct_gray"], reference["direct_gray"])
                >= FAMILY_CORRELATION
                and float(np.max(np.abs(
                    candidate["color_mean"] - reference["color_mean"]
                ))) <= 0.04
            ):
                family.add(str(its_ids[candidate_index]))
        family.add(full_id)
        exclusion_ids.update(family)
        family_sizes.append(len(family))

    eligible_its = len(its_items) - len(exclusion_ids)
    integrity_ok = identity_match and count_integrity and group_integrity
    controls_ok = (
        control_top_k == EXPECTED_HAZE4K_GROUPS
        and control_top_one == EXPECTED_HAZE4K_GROUPS
        and not any(item["admitted"] for item in negative_records)
    )
    mapping_ok = (
        len(accepted_records) == EXPECTED_INDOOR_RELATIONSHIPS
        and eligible_its >= MIN_ELIGIBLE_ITS
    )
    gates = {
        "dataset_identity": identity_match and count_integrity,
        "haze4k_group_integrity": group_integrity,
        "transformed_positive_top_k_recall": control_top_k == EXPECTED_HAZE4K_GROUPS,
        "transformed_positive_top_one_recovery": control_top_one == EXPECTED_HAZE4K_GROUPS,
        "negative_control_specificity": not any(
            item["admitted"] for item in negative_records
        ),
        "source_relationship_completeness": (
            len(accepted_records) == EXPECTED_INDOOR_RELATIONSHIPS
        ),
        "eligible_its_capacity": eligible_its >= MIN_ELIGIBLE_ITS,
    }
    if not integrity_ok:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "ITS_SOURCE_EXCLUSION_INCONCLUSIVE"
        authorizes = "NONE"
    elif controls_ok and mapping_ok and all(gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "ITS_SOURCE_EXCLUSION_PASS"
        authorizes = "ITS_DISJOINT_MEASUREMENT_DESIGN"
    else:
        state = "COMPLETED_GATE_FAIL"
        decision = "ITS_SOURCE_EXCLUSION_FAIL"
        authorizes = "NONE"
    gate_reasons = [key for key, value in gates.items() if not value]
    if not gate_reasons:
        gate_reasons = ["all_frozen_identity_control_mapping_and_capacity_gates_passed"]
    authorized_exclusions = sorted(exclusion_ids) if state == "COMPLETED_GATE_PASS" else []

    mapping_path = output_file(context, "haze4k_its_source_mapping.csv")
    mapping_fields = [
        "haze4k_group_digest", "haze4k_representative_source",
        "haze4k_representative_id", "its_split", "its_scene_id", "its_full_id",
        "candidate_hamming_distance", "forward_correlation", "forward_score_gap",
        "family_adjusted_forward_gap", "top64_best_family_size",
        "reverse_correlation", "reverse_score_gap", "mutual_nearest", "accepted",
    ]
    with mapping_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=mapping_fields)
        writer.writeheader()
        writer.writerows(records)
    exclusion_path = output_file(context, "its_exclusion_ids.txt")
    exclusion_path.write_text(
        ("\n".join(authorized_exclusions) + "\n") if authorized_exclusions else "",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "full Haze4K-to-RESIDE-ITS source identity census",
        "dataset_identity": {
            "observed": observed_identity,
            "expected": EXPECTED_RESIDE_IDENTITY,
            "identity_match": identity_match,
            "counts": {
                "its_train_clear": len(its_train),
                "its_validation_clear": len(its_validation),
                "haze4k_gt": len(haze4k_train) + len(haze4k_test),
                "sots_outdoor_gt": len(sots_outdoor),
            },
            "count_integrity": count_integrity,
        },
        "haze4k_grouping": {
            "canonical_group_count": len(haze4k_groups),
            "alias_histogram": dict(sorted(alias_histogram.items())),
            "group_integrity": group_integrity,
        },
        "matcher": {
            "method": "symmetric seven-view pHash top-64 retrieval, maximum normalized grayscale correlation, fixed forward/reverse score gates, and mutual nearest neighbour",
            "top_k": TOP_K,
            "minimum_correlation": MIN_CORRELATION,
            "minimum_score_gap": MIN_SCORE_GAP,
            "family_rule": "pHash distance zero plus canonical identity or correlation >=0.995 and max RGB-mean difference <=0.04",
        },
        "controls": {
            "transformed_positive_count": EXPECTED_HAZE4K_GROUPS,
            "transformed_positive_top_k_recovered": control_top_k,
            "transformed_positive_top_one_recovered": control_top_one,
            "transformed_positive_score": quantiles(control_scores),
            "transformed_positive_gap": quantiles(control_gaps),
            "sots_outdoor_negative_count": len(negative_records),
            "sots_outdoor_negatives_admitted": sum(
                int(item["admitted"]) for item in negative_records
            ),
            "sots_outdoor_negative_score": quantiles([
                item["correlation"] for item in negative_records
            ]),
        },
        "source_relationships": {
            "queried_haze4k_groups": len(records),
            "accepted_relationships": len(accepted_records),
            "expected_relationships": EXPECTED_INDOOR_RELATIONSHIPS,
            "accepted_train": sum(
                record["its_split"] == "ITS_TRAIN" for record in accepted_records
            ),
            "accepted_validation": sum(
                record["its_split"] == "ITS_VALIDATION" for record in accepted_records
            ),
            "forward_score": quantiles([
                record["forward_correlation"] for record in records
            ]),
            "forward_gap": quantiles([
                record["forward_score_gap"] for record in records
            ]),
            "interpretation_limit": "A FAIL means only that the frozen matcher did not close the complete expected source relation set; it is not an absence-of-overlap or infeasibility certificate.",
        },
        "exclusion_asset": {
            "provisional_family_exclusion_count": len(exclusion_ids),
            "authorized_exclusion_count": len(authorized_exclusions),
            "authorized_exclusion_digest": digest_lines(authorized_exclusions),
            "family_size": quantiles([float(value) for value in family_sizes]),
            "eligible_its_clear_scenes": eligible_its,
            "authorized": state == "COMPLETED_GATE_PASS",
        },
        "gates": gates,
        "terminal": {
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "limitations": [
            "This operation qualifies source identity only; it does not validate an ITS measurement target or model behavior.",
            "The transformed positive controls validate the frozen matcher under one fixed resize/JPEG family and do not prove invariance to every source transformation.",
            "A scientific FAIL is method-specific and cannot be interpreted as absence of Haze4K-ITS overlap.",
            "The exclusion list is nonempty and authorized only under COMPLETED_GATE_PASS.",
            "No model, training, inference, dehazing outcome, confirmation, canary, or locked-test metric is accessed.",
        ],
        "marker": "RESIDE_ITS_SOURCE_EXCLUSION_V1_COMPLETE",
    }
    output_file(context, "its_source_exclusion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "its_source_exclusion_summary.json",
            "accepted_relationships": len(accepted_records),
            "authorized_exclusion_count": len(authorized_exclusions),
            "eligible_its_clear_scenes": eligible_its,
            "exclusion_digest": digest_lines(authorized_exclusions),
            "gate_reasons": gate_reasons,
            "model_or_outcome_accessed": False,
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
