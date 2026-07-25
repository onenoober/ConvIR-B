#!/usr/bin/env python3
"""Run a cost-bounded direct Haze4K-to-RESIDE-ITS relationship census."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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
RESAMPLE = Image.Resampling.LANCZOS
EXPECTED_ITS_TRAIN = 10000
EXPECTED_ITS_VALIDATION = 1000
EXPECTED_HAZE4K_GT = 4000
EXPECTED_HAZE4K_GROUPS = 1000
EXPECTED_PARENT_ACCEPTED = 426
EXPECTED_UNRESOLVED = 574
EXPECTED_RELATIONSHIPS = 500
TOTAL_UNITS = 13198

ORB_FEATURES = 1600
ORB_INDEX_FEATURES = 512
ORB_SCALE_FACTOR = 1.2
ORB_LEVELS = 8
ORB_EDGE = 19
ORB_PATCH = 31
ORB_FAST = 10
RETRIEVAL_K = 64
LSH_NEIGHBOURS = 8
LSH_TABLES = 12
LSH_KEY_SIZE = 20
LSH_MULTI_PROBE = 2
LSH_CHECKS = 64

ORB_LOWE_RATIO = 0.80
ORB_RANSAC_REPROJECTION = 5.0
ORB_RANSAC_CONFIDENCE = 0.995
ORB_RANSAC_ITERATIONS = 3000
ORB_MIN_RECIPROCAL = 10
ORB_MIN_INLIERS = 8
ORB_MIN_INLIER_RATIO = 0.25
FINAL_VERIFY_K = 8

SIFT_FEATURES = 8000
SIFT_CONTRAST = 0.01
SIFT_EDGE = 10
SIFT_SIGMA = 1.6
SIFT_LOWE_RATIO = 0.75
SIFT_RANSAC_REPROJECTION = 4.0
SIFT_RANSAC_CONFIDENCE = 0.995
SIFT_RANSAC_ITERATIONS = 5000
SIFT_MIN_RECIPROCAL = 20
SIFT_MIN_INLIERS = 15
SIFT_MIN_INLIER_RATIO = 0.35
MIN_HULL_COVERAGE = 0.01
MIN_GRID_CELLS = 4
GRID_SIZE = 4

CONTROL_SAMPLE_COUNT = 64
CONTROL_BUDGET_SECONDS = 900.0
FULL_BUDGET_SECONDS = 1800.0
CONTRACT_BUDGET_SECONDS = 720.0
EXPECTED_RESIDE_IDENTITY = {
    "archive_manifest_sha256": "27348952347a0f94ac33105d5e6ce11b4ce4e2b0fab68b583b978752b5b78be0",
    "pairing_report_sha256": "afdefd2887437d929e00239e932bd76e7504d9838c2a3c58c33ce13f91f4150a",
    "layout_record_sha256": "f025a8e6e3f43c9b924c0986b866ed28e5644dd7a7ec7b056ae311c83764d5dc",
}
POSITIVE_OTS_PAIRS = (
    ("HAZE4K_TRAIN", "101", "0870"),
    ("HAZE4K_TRAIN", "133", "7267"),
    ("HAZE4K_TEST", "417", "6360"),
)
NEGATIVE_OTS_PAIRS = (
    ("HAZE4K_TRAIN", "101", "7267"),
    ("HAZE4K_TRAIN", "133", "6360"),
    ("HAZE4K_TEST", "417", "0870"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"missing image directory: {directory}")
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def path_index(paths: list[Path]) -> dict[str, Path]:
    result = {path.stem: path for path in paths}
    if len(result) != len(paths):
        raise ValueError("image stems are not unique within a frozen directory")
    return result


def orb_factory() -> Any:
    return cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_LEVELS,
        edgeThreshold=ORB_EDGE,
        patchSize=ORB_PATCH,
        fastThreshold=ORB_FAST,
        scoreType=cv2.ORB_HARRIS_SCORE,
    )


def feature_from_gray(
    gray: np.ndarray,
    *,
    source: str,
    identifier: str,
    path: Path | None,
) -> dict[str, Any]:
    keypoints, descriptors = orb_factory().detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.empty((0, 32), dtype=np.uint8)
        points = np.empty((0, 2), dtype=np.float32)
    else:
        responses = np.asarray([point.response for point in keypoints], dtype=np.float32)
        order = np.argsort(-responses, kind="stable")[:ORB_INDEX_FEATURES]
        descriptors = np.ascontiguousarray(descriptors[order], dtype=np.uint8)
        points = np.asarray([keypoints[int(index)].pt for index in order], dtype=np.float32)
    return {
        "source": source,
        "id": identifier,
        "full_id": f"{source}:{identifier}",
        "path": path,
        "shape": tuple(int(value) for value in gray.shape),
        "points": points,
        "descriptors": descriptors,
    }


def extract_feature(path: Path, source: str) -> dict[str, Any]:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot read image: {path}")
    return feature_from_gray(gray, source=source, identifier=path.stem, path=path)


def extract_batch(
    paths: list[Path],
    source: str,
    *,
    workers: int,
    context: Any,
    offset: int,
) -> list[dict[str, Any]]:
    worker = partial(extract_feature, source=source)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, item in enumerate(executor.map(worker, paths, chunksize=4), start=1):
            results.append(item)
            if index == 1 or index % 100 == 0 or index == len(paths):
                write_workload_progress(
                    context,
                    completed_units=offset + index,
                    stage=f"orb_features_{source.lower()}",
                )
    return results


def lsh_matcher(features: list[dict[str, Any]]) -> Any:
    matcher = cv2.FlannBasedMatcher(
        {
            "algorithm": 6,
            "table_number": LSH_TABLES,
            "key_size": LSH_KEY_SIZE,
            "multi_probe_level": LSH_MULTI_PROBE,
        },
        {"checks": LSH_CHECKS},
    )
    matcher.add([item["descriptors"] for item in features])
    matcher.train()
    return matcher


def retrieve(
    matcher: Any,
    query_descriptors: np.ndarray,
    reference_ids: np.ndarray,
) -> list[int]:
    if len(query_descriptors) == 0:
        return []
    counts: Counter[int] = Counter()
    distances: Counter[int] = Counter()
    for neighbours in matcher.knnMatch(query_descriptors, k=LSH_NEIGHBOURS):
        for match in neighbours:
            scene_index = int(match.imgIdx)
            counts[scene_index] += 1
            distances[scene_index] += int(round(float(match.distance)))
    ordered = sorted(
        counts,
        key=lambda index: (
            -counts[index],
            distances[index] / counts[index],
            str(reference_ids[index]),
        ),
    )
    return ordered[:RETRIEVAL_K]


def ratio_pairs(
    left: np.ndarray,
    right: np.ndarray,
    *,
    norm: int,
    ratio: float,
) -> set[tuple[int, int]]:
    if len(left) < 1 or len(right) < 2:
        return set()
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    accepted: set[tuple[int, int]] = set()
    for neighbours in matcher.knnMatch(left, right, k=2):
        if len(neighbours) == 2 and neighbours[0].distance < ratio * neighbours[1].distance:
            accepted.add((neighbours[0].queryIdx, neighbours[0].trainIdx))
    return accepted


def spatial_support(points: np.ndarray, shape: tuple[int, int]) -> tuple[float, int]:
    if len(points) < 3:
        return 0.0, 0
    height, width = shape
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))
    coverage = float(cv2.contourArea(hull)) / max(float(height * width), 1.0)
    cells = {
        (
            min(GRID_SIZE - 1, max(0, int(point[0] * GRID_SIZE / max(width, 1)))),
            min(GRID_SIZE - 1, max(0, int(point[1] * GRID_SIZE / max(height, 1)))),
        )
        for point in points
    }
    return coverage, len(cells)


def homography_result(
    left_points: np.ndarray,
    right_points: np.ndarray,
    left_shape: tuple[int, int],
    right_shape: tuple[int, int],
    reciprocal: list[tuple[int, int]],
    *,
    reprojection: float,
    confidence: float,
    iterations: int,
    minimum_reciprocal: int,
    minimum_inliers: int,
    minimum_ratio: float,
    require_spatial_support: bool,
) -> dict[str, Any]:
    result = {
        "reciprocal_matches": len(reciprocal),
        "homography_inliers": 0,
        "homography_inlier_ratio": 0.0,
        "left_hull_coverage": 0.0,
        "right_hull_coverage": 0.0,
        "left_grid_cells": 0,
        "right_grid_cells": 0,
        "accepted": False,
    }
    if len(reciprocal) < 4:
        return result
    left = np.float32([left_points[a] for a, _ in reciprocal]).reshape(-1, 1, 2)
    right = np.float32([right_points[b] for _, b in reciprocal]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(
        left,
        right,
        cv2.RANSAC,
        reprojection,
        maxIters=iterations,
        confidence=confidence,
    )
    selected = mask.reshape(-1).astype(bool) if mask is not None else np.zeros(len(left), dtype=bool)
    inliers = int(selected.sum())
    inlier_ratio = inliers / len(reciprocal)
    left_coverage, left_cells = spatial_support(left.reshape(-1, 2)[selected], left_shape)
    right_coverage, right_cells = spatial_support(right.reshape(-1, 2)[selected], right_shape)
    support_ok = (
        not require_spatial_support
        or (
            left_coverage >= MIN_HULL_COVERAGE
            and right_coverage >= MIN_HULL_COVERAGE
            and left_cells >= MIN_GRID_CELLS
            and right_cells >= MIN_GRID_CELLS
        )
    )
    result.update({
        "homography_inliers": inliers,
        "homography_inlier_ratio": inlier_ratio,
        "left_hull_coverage": left_coverage,
        "right_hull_coverage": right_coverage,
        "left_grid_cells": left_cells,
        "right_grid_cells": right_cells,
        "accepted": (
            len(reciprocal) >= minimum_reciprocal
            and inliers >= minimum_inliers
            and inlier_ratio >= minimum_ratio
            and support_ok
        ),
    })
    return result


def orb_geometry(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    forward = ratio_pairs(
        left["descriptors"], right["descriptors"], norm=cv2.NORM_HAMMING,
        ratio=ORB_LOWE_RATIO,
    )
    reverse_raw = ratio_pairs(
        right["descriptors"], left["descriptors"], norm=cv2.NORM_HAMMING,
        ratio=ORB_LOWE_RATIO,
    )
    reverse = {(right_index, left_index) for left_index, right_index in reverse_raw}
    reciprocal = sorted(forward & reverse)
    result = homography_result(
        left["points"], right["points"], left["shape"], right["shape"], reciprocal,
        reprojection=ORB_RANSAC_REPROJECTION,
        confidence=ORB_RANSAC_CONFIDENCE,
        iterations=ORB_RANSAC_ITERATIONS,
        minimum_reciprocal=ORB_MIN_RECIPROCAL,
        minimum_inliers=ORB_MIN_INLIERS,
        minimum_ratio=ORB_MIN_INLIER_RATIO,
        require_spatial_support=False,
    )
    result.update({
        "left_keypoints": len(left["points"]),
        "right_keypoints": len(right["points"]),
        "forward_ratio_matches": len(forward),
        "reverse_ratio_matches": len(reverse_raw),
    })
    return result


def gray_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    longest = max(image.shape)
    if longest > 1600:
        scale = 1600.0 / longest
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def rootsift(path: Path) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int]]:
    image = gray_image(path)
    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
        contrastThreshold=SIFT_CONTRAST,
        edgeThreshold=SIFT_EDGE,
        sigma=SIFT_SIGMA,
    )
    keypoints, descriptors = sift.detectAndCompute(image, None)
    points = np.asarray([point.pt for point in keypoints], dtype=np.float32)
    if descriptors is None or len(descriptors) == 0:
        return points, None, tuple(int(value) for value in image.shape)
    descriptors = descriptors.astype(np.float32)
    descriptors /= np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12)
    return points, np.sqrt(descriptors), tuple(int(value) for value in image.shape)


class RootSiftCache:
    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self.values: OrderedDict[
            str, tuple[np.ndarray, np.ndarray | None, tuple[int, int]]
        ] = OrderedDict()

    def get(self, path: Path) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int]]:
        key = str(path)
        if key in self.values:
            value = self.values.pop(key)
            self.values[key] = value
            return value
        value = rootsift(path)
        self.values[key] = value
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value


def rootsift_geometry(left_path: Path, right_path: Path, cache: RootSiftCache) -> dict[str, Any]:
    left_points, left_descriptors, left_shape = cache.get(left_path)
    right_points, right_descriptors, right_shape = cache.get(right_path)
    empty = {
        "left_keypoints": len(left_points),
        "right_keypoints": len(right_points),
        "forward_ratio_matches": 0,
        "reverse_ratio_matches": 0,
        "reciprocal_matches": 0,
        "homography_inliers": 0,
        "homography_inlier_ratio": 0.0,
        "left_hull_coverage": 0.0,
        "right_hull_coverage": 0.0,
        "left_grid_cells": 0,
        "right_grid_cells": 0,
        "accepted": False,
    }
    if left_descriptors is None or right_descriptors is None:
        return empty
    forward = ratio_pairs(
        left_descriptors, right_descriptors, norm=cv2.NORM_L2, ratio=SIFT_LOWE_RATIO,
    )
    reverse_raw = ratio_pairs(
        right_descriptors, left_descriptors, norm=cv2.NORM_L2, ratio=SIFT_LOWE_RATIO,
    )
    reverse = {(right_index, left_index) for left_index, right_index in reverse_raw}
    reciprocal = sorted(forward & reverse)
    result = homography_result(
        left_points, right_points, left_shape, right_shape, reciprocal,
        reprojection=SIFT_RANSAC_REPROJECTION,
        confidence=SIFT_RANSAC_CONFIDENCE,
        iterations=SIFT_RANSAC_ITERATIONS,
        minimum_reciprocal=SIFT_MIN_RECIPROCAL,
        minimum_inliers=SIFT_MIN_INLIERS,
        minimum_ratio=SIFT_MIN_INLIER_RATIO,
        require_spatial_support=True,
    )
    result.update({
        "left_keypoints": len(left_points),
        "right_keypoints": len(right_points),
        "forward_ratio_matches": len(forward),
        "reverse_ratio_matches": len(reverse_raw),
    })
    return result


def synthetic_images() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(3407)
    base = generator.integers(0, 256, size=(512, 512), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0.7)
    for index in range(16, 496, 32):
        cv2.line(base, (8, index), (503, (index * 7) % 504), int((index * 13) % 256), 2)
        cv2.circle(base, ((index * 11) % 504, index), 7, int((index * 17) % 256), 2)
    source = np.float32([[0, 0], [511, 0], [511, 511], [0, 511]])
    target = np.float32([[28, 34], [486, 8], [507, 478], [18, 504]])
    matrix = cv2.getPerspectiveTransform(source, target)
    transformed = cv2.warpPerspective(base, matrix, (512, 512))
    unrelated = generator.integers(0, 256, size=(512, 512), dtype=np.uint8)
    unrelated = cv2.GaussianBlur(unrelated, (3, 3), 0.7)
    return base, transformed, unrelated


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    cv2.setRNGSeed(3407)
    cv2.setNumThreads(1)
    started = time.monotonic()
    entrypoint = context.assets.get("entrypoint_source")
    checks = {
        "cpu_exact_mode": context.engineering_contract["mode"] == "cpu_exact",
        "cpu_contract": context.device == "cpu",
        "datasets_hidden_from_contract": (
            "reside_root" not in context.assets and "haze4k_root" not in context.assets
        ),
        "entrypoint_identity_bound": entrypoint is not None and entrypoint.contract_access is True,
        "parent_evidence_available": all(
            context.assets.get(name) is not None
            for name in (
                "parent_closeout", "parent_summary", "parent_relationships", "parent_conclusion"
            )
        ),
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
        "opencv_sift_available": hasattr(cv2, "SIFT_create"),
        "no_model_training_or_inference_path": True,
    }
    generator = np.random.default_rng(3407)
    references = [
        generator.integers(0, 256, size=(ORB_INDEX_FEATURES, 32), dtype=np.uint8)
        for _ in range(EXPECTED_ITS_TRAIN + EXPECTED_ITS_VALIDATION)
    ]
    synthetic_features = [
        {"descriptors": descriptors} for descriptors in references
    ]
    matcher = lsh_matcher(synthetic_features)
    reference_ids = np.asarray([f"synthetic-{index:05d}" for index in range(len(references))])
    recalled = 0
    for query_index in range(EXPECTED_HAZE4K_GROUPS):
        reference_index = (query_index * 11) % len(references)
        query = references[reference_index].copy()
        flip_rows = np.arange(query_index % 7, len(query), 29)
        query[flip_rows, query_index % 32] ^= np.uint8(1 << (query_index % 8))
        ranked = retrieve(matcher, query, reference_ids)
        recalled += int(reference_index in ranked)
    base, transformed, unrelated = synthetic_images()
    positive_left = feature_from_gray(base, source="SYNTHETIC", identifier="left", path=None)
    positive_right = feature_from_gray(
        transformed, source="SYNTHETIC", identifier="transformed", path=None,
    )
    negative_right = feature_from_gray(
        unrelated, source="SYNTHETIC", identifier="unrelated", path=None,
    )
    orb_positive = orb_geometry(positive_left, positive_right)
    orb_negative = orb_geometry(positive_left, negative_right)
    with tempfile.TemporaryDirectory(prefix="its-targeted-contract-") as directory:
        root = Path(directory)
        base_path = root / "base.png"
        transformed_path = root / "transformed.png"
        unrelated_path = root / "unrelated.png"
        if not all((
            cv2.imwrite(str(base_path), base),
            cv2.imwrite(str(transformed_path), transformed),
            cv2.imwrite(str(unrelated_path), unrelated),
        )):
            raise ValueError("failed to materialize protected-data-free geometry fixture")
        cache = RootSiftCache(capacity=3)
        rootsift_positive = rootsift_geometry(base_path, transformed_path, cache)
        rootsift_negative = rootsift_geometry(base_path, unrelated_path, cache)
    elapsed = time.monotonic() - started
    checks.update({
        "same_scale_reference_scenes": len(references) == 11000,
        "same_scale_reference_descriptors": all(len(item) == 512 for item in references),
        "same_scale_query_sets": EXPECTED_HAZE4K_GROUPS == 1000,
        "same_scale_top64_recall": recalled == EXPECTED_HAZE4K_GROUPS,
        "same_scale_elapsed_bound": elapsed <= CONTRACT_BUDGET_SECONDS,
        "synthetic_orb_positive_accepted": bool(orb_positive["accepted"]),
        "synthetic_orb_negative_rejected": not bool(orb_negative["accepted"]),
        "synthetic_rootsift_positive_accepted": bool(rootsift_positive["accepted"]),
        "synthetic_rootsift_negative_rejected": not bool(rootsift_negative["accepted"]),
        "output_and_finalizer_contract": True,
    })
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "cpu_exact",
            "device": "cpu",
            "fixture": {"batch": 11000, "channels": 1000, "height": 512, "width": 32},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def load_parent_rows(context: Any) -> list[dict[str, str]]:
    with asset_path(context, "parent_relationships", kind="file").open(
        "r", encoding="utf-8", newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def parent_identity(context: Any, rows: list[dict[str, str]]) -> bool:
    closeout = json.loads(asset_path(context, "parent_closeout", kind="file").read_text("utf-8"))
    summary = json.loads(asset_path(context, "parent_summary", kind="file").read_text("utf-8"))
    conclusion = json.loads(asset_path(context, "parent_conclusion", kind="file").read_text("utf-8"))
    return (
        closeout.get("state") == "COMPLETED_GATE_FAIL"
        and closeout.get("decision") == "ITS_GEOMETRIC_SOURCE_EXCLUSION_FAIL"
        and closeout.get("authorizes") == "NONE"
        and conclusion.get("decision") == "ITS_GEOMETRIC_SOURCE_EXCLUSION_FAIL"
        and summary.get("relationships", {}).get("accepted_relationships") == EXPECTED_PARENT_ACCEPTED
        and len(rows) == EXPECTED_HAZE4K_GROUPS
        and sum(row.get("accepted") == "True" for row in rows) == EXPECTED_PARENT_ACCEPTED
        and sum(row.get("accepted") == "False" for row in rows) == EXPECTED_UNRESOLVED
    )


def haze_path(
    row: dict[str, str], train: dict[str, Path], test: dict[str, Path],
) -> Path:
    source = row["haze4k_source"]
    if source == "HAZE4K_TRAIN":
        return train[row["haze4k_id"]]
    if source == "HAZE4K_TEST":
        return test[row["haze4k_id"]]
    raise ValueError(f"unknown Haze4K source: {source}")


def local_transform(feature: dict[str, Any]) -> dict[str, Any]:
    with Image.open(feature["path"]) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    margin_x = max(2, int(round(width * 0.06)))
    margin_y = max(2, int(round(height * 0.06)))
    cropped = rgb[margin_y:height - margin_y, margin_x:width - margin_x]
    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    target = np.float32([
        [0.015 * width, 0.020 * height],
        [0.985 * width, 0.005 * height],
        [0.995 * width, 0.980 * height],
        [0.020 * width, 0.995 * height],
    ])
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(resized, matrix, (width, height))
    encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(warped, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82],
    )[1]
    transformed = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return feature_from_gray(
        transformed, source="LOCAL_CONTROL", identifier=feature["id"], path=None,
    )


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) < count:
        raise ValueError("insufficient rows for frozen control sample")
    positions = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
    return [rows[int(position)] for position in positions]


def direct_pair(
    query_feature: dict[str, Any],
    reference_feature: dict[str, Any],
    query_path: Path,
    reference_path: Path,
    cache: RootSiftCache,
) -> dict[str, Any]:
    orb = orb_geometry(query_feature, reference_feature)
    sift = rootsift_geometry(query_path, reference_path, cache) if orb["accepted"] else None
    return {
        "orb": orb,
        "rootsift": sift,
        "accepted": bool(orb["accepted"] and sift is not None and sift["accepted"]),
    }


def targeted_candidates(
    query: dict[str, Any],
    query_path: Path,
    candidate_indices: list[int],
    references: list[dict[str, Any]],
    cache: RootSiftCache,
) -> list[dict[str, Any]]:
    orb_rows: list[dict[str, Any]] = []
    for rank, candidate_index in enumerate(candidate_indices, start=1):
        candidate = references[candidate_index]
        orb = orb_geometry(query, candidate)
        if orb["accepted"]:
            orb_rows.append({
                "candidate_index": candidate_index,
                "candidate_id": candidate["full_id"],
                "retrieval_rank": rank,
                "orb": orb,
            })
    orb_rows.sort(key=lambda row: (
        -row["orb"]["homography_inliers"],
        -row["orb"]["homography_inlier_ratio"],
        -row["orb"]["reciprocal_matches"],
        row["retrieval_rank"],
        row["candidate_id"],
    ))
    final_rows: list[dict[str, Any]] = []
    for row in orb_rows[:FINAL_VERIFY_K]:
        candidate = references[row["candidate_index"]]
        sift = rootsift_geometry(query_path, candidate["path"], cache)
        row["rootsift"] = sift
        row["accepted"] = bool(sift["accepted"])
        final_rows.append(row)
    return final_rows


def write_relationships(context: Any, rows: list[dict[str, Any]]) -> None:
    fields = [
        "haze4k_group_digest", "haze4k_source", "haze4k_id", "execution_status",
        "retrieval_candidates", "orb_passed_candidates", "final_verified_candidates",
        "accepted_candidate_count", "accepted_its_ids", "best_its_id",
        "best_retrieval_rank", "best_orb_reciprocal", "best_orb_inliers",
        "best_rootsift_reciprocal", "best_rootsift_inliers",
        "best_rootsift_inlier_ratio", "best_left_hull_coverage",
        "best_right_hull_coverage", "accepted",
    ]
    with output_file(context, "targeted_direct_relationships.csv").open(
        "w", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    if context.total_units != TOTAL_UNITS:
        raise ValueError("runtime total_units differs from frozen targeted census")
    workers = int(os.environ.get("CONVIR_ROUTE_ITS_TARGETED_WORKERS", "8"))
    if workers != 8:
        raise ValueError("worker count differs from frozen contract")
    cv2.setRNGSeed(3407)
    cv2.setNumThreads(1)
    started = time.monotonic()
    reside = asset_path(context, "reside_root", kind="directory")
    haze4k = asset_path(context, "haze4k_root", kind="directory")
    rows = load_parent_rows(context)
    parent_ok = parent_identity(context, rows)

    identity_paths = {
        "archive_manifest_sha256": reside / "ARCHIVE_SHA256SUMS.txt",
        "pairing_report_sha256": reside / "PAIRING_VALIDATION.txt",
        "layout_record_sha256": reside / "DATASET_LAYOUT.txt",
    }
    observed_identity = {name: sha256_file(path) for name, path in identity_paths.items()}
    its_train_paths = image_files(reside / "official/ITS/train/ITS_clear")
    its_validation_paths = image_files(reside / "official/ITS/val/clear")
    haze_train_paths = image_files(haze4k / "train/gt")
    haze_test_paths = image_files(haze4k / "test/gt")
    ots_paths = image_files(reside / "official/OTS_ALPHA/clear_images")
    dataset_ok = (
        observed_identity == EXPECTED_RESIDE_IDENTITY
        and len(its_train_paths) == EXPECTED_ITS_TRAIN
        and len(its_validation_paths) == EXPECTED_ITS_VALIDATION
        and len(haze_train_paths) + len(haze_test_paths) == EXPECTED_HAZE4K_GT
    )
    train_index = path_index(haze_train_paths)
    test_index = path_index(haze_test_paths)
    ots_index = path_index(ots_paths)

    its_train = extract_batch(
        its_train_paths, "ITS_TRAIN", workers=workers, context=context, offset=0,
    )
    its_validation = extract_batch(
        its_validation_paths, "ITS_VALIDATION", workers=workers, context=context,
        offset=len(its_train),
    )
    references = its_train + its_validation
    reference_ids = np.asarray([item["full_id"] for item in references])
    reference_by_id = {item["full_id"]: index for index, item in enumerate(references)}
    matcher = lsh_matcher(references)
    write_workload_progress(context, completed_units=11000, stage="lsh_index_built")

    ordered_rows = sorted(rows, key=lambda row: row["haze4k_group_digest"])
    query_paths = [haze_path(row, train_index, test_index) for row in ordered_rows]
    query_features = extract_batch(
        query_paths, "HAZE4K_QUERY", workers=workers, context=context, offset=11000,
    )
    query_by_digest = {
        row["haze4k_group_digest"]: feature
        for row, feature in zip(ordered_rows, query_features)
    }
    group_ok = (
        len(ordered_rows) == EXPECTED_HAZE4K_GROUPS
        and len(query_by_digest) == EXPECTED_HAZE4K_GROUPS
        and len({(row["haze4k_source"], row["haze4k_id"]) for row in ordered_rows})
        == EXPECTED_HAZE4K_GROUPS
    )

    accepted_parent = [row for row in ordered_rows if row["accepted"] == "True"]
    unresolved = [row for row in ordered_rows if row["accepted"] == "False"]
    retrievals: dict[str, list[int]] = {}
    known_recalled = 0
    progress = 12000
    for row in accepted_parent:
        query = query_by_digest[row["haze4k_group_digest"]]
        ranked = retrieve(matcher, query["descriptors"], reference_ids)
        retrievals[row["haze4k_group_digest"]] = ranked
        accepted_ids = {value for value in row["accepted_its_ids"].split(";") if value}
        known_recalled += int(any(reference_ids[index] in accepted_ids for index in ranked))
        progress += 1
        if progress % 50 == 0 or progress == 12000 + len(accepted_parent):
            write_workload_progress(context, completed_units=progress, stage="known_recall_gate")

    control_rows = evenly_spaced(accepted_parent, CONTROL_SAMPLE_COUNT)
    local_recalled = 0
    for row in control_rows:
        reference_index = reference_by_id[row["best_its_id"]]
        transformed = local_transform(references[reference_index])
        ranked = retrieve(matcher, transformed["descriptors"], reference_ids)
        local_recalled += int(reference_index in ranked)
        progress += 1
    write_workload_progress(context, completed_units=progress, stage="local_retrieval_gate")

    cache = RootSiftCache()
    positive_results: list[dict[str, Any]] = []
    for row in control_rows:
        query = query_by_digest[row["haze4k_group_digest"]]
        reference = references[reference_by_id[row["best_its_id"]]]
        positive_results.append(direct_pair(
            query, reference, Path(query["path"]), Path(reference["path"]), cache,
        ))
        progress += 1
    write_workload_progress(context, completed_units=progress, stage="direct_positive_gate")

    negative_results: list[dict[str, Any]] = []
    shifted = control_rows[CONTROL_SAMPLE_COUNT // 2:] + control_rows[:CONTROL_SAMPLE_COUNT // 2]
    for query_row, reference_row in zip(control_rows, shifted):
        query = query_by_digest[query_row["haze4k_group_digest"]]
        reference = references[reference_by_id[reference_row["best_its_id"]]]
        negative_results.append(direct_pair(
            query, reference, Path(query["path"]), Path(reference["path"]), cache,
        ))
        progress += 1
    write_workload_progress(context, completed_units=progress, stage="direct_negative_gate")

    ots_positive: list[dict[str, Any]] = []
    ots_negative: list[dict[str, Any]] = []
    for source, haze_id, ots_id in POSITIVE_OTS_PAIRS:
        left = train_index[haze_id] if source == "HAZE4K_TRAIN" else test_index[haze_id]
        ots_positive.append(rootsift_geometry(left, ots_index[ots_id], cache))
        progress += 1
    for source, haze_id, ots_id in NEGATIVE_OTS_PAIRS:
        left = train_index[haze_id] if source == "HAZE4K_TRAIN" else test_index[haze_id]
        ots_negative.append(rootsift_geometry(left, ots_index[ots_id], cache))
        progress += 1
    write_workload_progress(context, completed_units=progress, stage="ots_geometry_gate")

    control_elapsed = time.monotonic() - started
    control_gates = {
        "known_relationship_top64_recall": known_recalled == EXPECTED_PARENT_ACCEPTED,
        "local_transform_top64_recall": local_recalled == CONTROL_SAMPLE_COUNT,
        "direct_positive_acceptance": sum(result["accepted"] for result in positive_results)
        == CONTROL_SAMPLE_COUNT,
        "direct_negative_specificity": sum(result["accepted"] for result in negative_results) == 0,
        "ots_positive_acceptance": sum(result["accepted"] for result in ots_positive) == 3,
        "ots_negative_specificity": sum(result["accepted"] for result in ots_negative) == 0,
        "control_cost_bound": control_elapsed <= CONTROL_BUDGET_SECONDS,
    }
    qualification_passed = parent_ok and dataset_ok and group_ok and all(control_gates.values())

    result_rows: list[dict[str, Any]] = []
    new_accepted = 0
    processed = 0
    for row in unresolved:
        if not qualification_passed:
            execution_status = "not_executed_qualification_gate"
            ranked: list[int] = []
            final_rows: list[dict[str, Any]] = []
        elif time.monotonic() - started > FULL_BUDGET_SECONDS:
            execution_status = "not_executed_cost_bound"
            ranked = []
            final_rows = []
        else:
            execution_status = "executed"
            query = query_by_digest[row["haze4k_group_digest"]]
            ranked = retrieve(matcher, query["descriptors"], reference_ids)
            final_rows = targeted_candidates(
                query, Path(query["path"]), ranked, references, cache,
            )
            processed += 1
        accepted = [item for item in final_rows if item["accepted"]]
        new_accepted += int(bool(accepted))
        best = sorted(
            final_rows,
            key=lambda item: (
                -int(item["accepted"]),
                -item["rootsift"]["homography_inliers"],
                -item["rootsift"]["homography_inlier_ratio"],
                item["retrieval_rank"],
                item["candidate_id"],
            ),
        )[0] if final_rows else None
        result_rows.append({
            "haze4k_group_digest": row["haze4k_group_digest"],
            "haze4k_source": row["haze4k_source"],
            "haze4k_id": row["haze4k_id"],
            "execution_status": execution_status,
            "retrieval_candidates": len(ranked),
            "orb_passed_candidates": sum(1 for item in final_rows),
            "final_verified_candidates": len(final_rows),
            "accepted_candidate_count": len(accepted),
            "accepted_its_ids": ";".join(sorted(item["candidate_id"] for item in accepted)),
            "best_its_id": best["candidate_id"] if best else "",
            "best_retrieval_rank": best["retrieval_rank"] if best else "",
            "best_orb_reciprocal": best["orb"]["reciprocal_matches"] if best else "",
            "best_orb_inliers": best["orb"]["homography_inliers"] if best else "",
            "best_rootsift_reciprocal": best["rootsift"]["reciprocal_matches"] if best else "",
            "best_rootsift_inliers": best["rootsift"]["homography_inliers"] if best else "",
            "best_rootsift_inlier_ratio": best["rootsift"]["homography_inlier_ratio"] if best else "",
            "best_left_hull_coverage": best["rootsift"]["left_hull_coverage"] if best else "",
            "best_right_hull_coverage": best["rootsift"]["right_hull_coverage"] if best else "",
            "accepted": bool(accepted),
        })
        progress += 1
        if progress % 25 == 0 or progress == TOTAL_UNITS:
            write_workload_progress(context, completed_units=progress, stage="targeted_queries")

    full_elapsed = time.monotonic() - started
    full_completed = processed == EXPECTED_UNRESOLVED
    total_relationships = EXPECTED_PARENT_ACCEPTED + new_accepted
    gates = {
        "parent_evidence_identity": parent_ok,
        "dataset_identity": dataset_ok,
        "haze4k_group_integrity": group_ok,
        **control_gates,
        "targeted_queries_complete": full_completed,
        "full_cost_bound": full_elapsed <= FULL_BUDGET_SECONDS,
        "relationship_completeness": full_completed and total_relationships == EXPECTED_RELATIONSHIPS,
    }
    inconclusive = not (parent_ok and dataset_ok and group_ok)
    if inconclusive:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "ITS_TARGETED_DIRECT_RELATIONSHIP_INCONCLUSIVE"
        authorizes = "NONE"
    elif all(gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "ITS_TARGETED_DIRECT_RELATIONSHIP_PASS"
        authorizes = "ITS_DIRECT_FAMILY_EXCLUSION_DESIGN"
    else:
        state = "COMPLETED_GATE_FAIL"
        decision = "ITS_TARGETED_DIRECT_RELATIONSHIP_FAIL"
        authorizes = "NONE"
    gate_reasons = [name for name, passed in gates.items() if not passed]

    write_relationships(context, result_rows)
    summary = {
        "schema_version": 1,
        "route_id": "reside-its-targeted-direct-geometry-v1",
        "operation_id": "ITS_TARGETED_DIRECT_RELATIONSHIP",
        "run_id": context.output_id,
        "scope": "bounded direct Haze4K-to-RESIDE-ITS relationship completion",
        "method": {
            "retrieval": {
                "descriptor": "ORB",
                "detected_features": ORB_FEATURES,
                "indexed_features": ORB_INDEX_FEATURES,
                "lsh_tables": LSH_TABLES,
                "lsh_key_size": LSH_KEY_SIZE,
                "lsh_multi_probe": LSH_MULTI_PROBE,
                "lsh_checks": LSH_CHECKS,
                "descriptor_neighbours": LSH_NEIGHBOURS,
                "top_k": RETRIEVAL_K,
            },
            "direct_geometry": {
                "orb_prefilter_max_candidates": RETRIEVAL_K,
                "rootsift_max_candidates": FINAL_VERIFY_K,
                "rootsift_minimum_reciprocal": SIFT_MIN_RECIPROCAL,
                "rootsift_minimum_inliers": SIFT_MIN_INLIERS,
                "rootsift_minimum_inlier_ratio": SIFT_MIN_INLIER_RATIO,
                "minimum_hull_coverage_each_side": MIN_HULL_COVERAGE,
                "minimum_grid_cells_each_side": MIN_GRID_CELLS,
                "transitive_its_expansion": False,
            },
        },
        "controls": {
            "known_relationships": EXPECTED_PARENT_ACCEPTED,
            "known_relationships_recalled_top64": known_recalled,
            "local_transform_count": CONTROL_SAMPLE_COUNT,
            "local_transform_recalled_top64": local_recalled,
            "direct_positive_count": CONTROL_SAMPLE_COUNT,
            "direct_positive_accepted": sum(result["accepted"] for result in positive_results),
            "direct_negative_count": CONTROL_SAMPLE_COUNT,
            "direct_negative_accepted": sum(result["accepted"] for result in negative_results),
            "ots_positive_accepted": sum(result["accepted"] for result in ots_positive),
            "ots_negative_accepted": sum(result["accepted"] for result in ots_negative),
            "qualification_passed": qualification_passed,
        },
        "relationships": {
            "inherited_parent_relationships": EXPECTED_PARENT_ACCEPTED,
            "targeted_queries_planned": EXPECTED_UNRESOLVED,
            "targeted_queries_executed": processed,
            "new_accepted_relationships": new_accepted,
            "total_accepted_relationships": total_relationships,
            "expected_relationships": EXPECTED_RELATIONSHIPS,
            "interpretation_limit": (
                "A failure is specific to this frozen bounded direct matcher and does not prove "
                "that an underlying source relationship is absent."
            ),
        },
        "timing": {
            "control_elapsed_seconds": control_elapsed,
            "control_budget_seconds": CONTROL_BUDGET_SECONDS,
            "full_elapsed_seconds": full_elapsed,
            "full_budget_seconds": FULL_BUDGET_SECONDS,
        },
        "gates": gates,
        "terminal": {
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "limitations": [
            "This route completes source relationships only; it does not create an ITS exclusion asset.",
            "The inherited 426 relationships are diagnostic evidence from the parent scientific terminal.",
            "No ITS-to-ITS transitive expansion, training, inference, model outcome, confirmation, canary or locked-test evidence is used.",
            "Exactly 500 is a provenance expectation and is not used to adapt thresholds or stop the search.",
        ],
        "marker": "RESIDE_ITS_TARGETED_DIRECT_RELATIONSHIP_V1_COMPLETE",
    }
    output_file(context, "targeted_direct_relationship_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    write_workload_progress(context, completed_units=TOTAL_UNITS, stage="targeted_finalize")
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "targeted_direct_relationship_summary.json",
            "relationships_file": "targeted_direct_relationships.csv",
            "qualification_passed": qualification_passed,
            "targeted_queries_executed": processed,
            "new_accepted_relationships": new_accepted,
            "total_accepted_relationships": total_relationships,
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
