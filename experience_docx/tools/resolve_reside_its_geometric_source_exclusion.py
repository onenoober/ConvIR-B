#!/usr/bin/env python3
"""Qualify a fast local-geometric Haze4K-to-RESIDE-ITS exclusion asset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import time
from collections import Counter, OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterable

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
EXPECTED_HAZE4K_ALIASES = 4
EXPECTED_SOTS_OUTDOOR = 492
EXPECTED_RELATIONSHIPS = 500
EXPECTED_PARENT_ACCEPTED = 6
MIN_ELIGIBLE_ITS = 6000
TOTAL_UNITS = 17990

ORB_FEATURES = 1200
ORB_INDEX_FEATURES = 384
ORB_SCALE_FACTOR = 1.2
ORB_LEVELS = 8
ORB_EDGE = 19
ORB_PATCH = 31
ORB_FAST = 10
INDEX_OFFSETS = (0, 4, 8, 12, 16, 20, 24, 28)
ORB_RETRIEVAL_K = 96
PHASH_RETRIEVAL_K = 32
ORB_PREFILTER_K = 24
PHASH_PREFILTER_K = 24
FINAL_VERIFY_K = 8
ORB_LOWE_RATIO = 0.80
ORB_RANSAC_REPROJECTION = 5.0
ORB_RANSAC_CONFIDENCE = 0.995
ORB_RANSAC_ITERATIONS = 3000
ORB_MIN_RECIPROCAL = 10
ORB_MIN_INLIERS = 8
ORB_MIN_INLIER_RATIO = 0.25

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

FAMILY_MAX_MEMBERS = 4000
LOCAL_CONTROL_GEOMETRY_COUNT = 64
LOCAL_CONTROL_MIN_RECALL = 0.99
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


def path_index(paths: list[Path]) -> dict[str, Path]:
    result = {path.stem: path for path in paths}
    if len(result) != len(paths):
        raise ValueError("image stems are not unique within a frozen directory")
    return result


def packed_hash(gray: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(resized)[:8, :8]
    bits = low > np.median(low.reshape(-1)[1:])
    return np.packbits(bits.reshape(-1).astype(np.uint8))


def crop_view(gray: np.ndarray, fraction: float, x_anchor: float, y_anchor: float) -> np.ndarray:
    height, width = gray.shape
    crop_width = max(8, int(round(width * fraction)))
    crop_height = max(8, int(round(height * fraction)))
    left = int(round((width - crop_width) * x_anchor))
    top = int(round((height - crop_height) * y_anchor))
    return gray[top:top + crop_height, left:left + crop_width]


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


def feature_from_arrays(
    gray: np.ndarray,
    canonical_rgb: np.ndarray,
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
        responses = np.asarray([item.response for item in keypoints], dtype=np.float32)
        order = np.argsort(-responses, kind="stable")[:ORB_INDEX_FEATURES]
        descriptors = np.ascontiguousarray(descriptors[order], dtype=np.uint8)
        points = np.asarray([keypoints[int(index)].pt for index in order], dtype=np.float32)
    phashes = np.stack([
        packed_hash(crop_view(gray, fraction, x_anchor, y_anchor))
        for fraction, x_anchor, y_anchor in VIEW_SPECS
    ])
    return {
        "source": source,
        "id": identifier,
        "full_id": f"{source}:{identifier}",
        "path": path,
        "canonical": hashlib.sha256(canonical_rgb.tobytes()).hexdigest(),
        "phashes": phashes,
        "points": points,
        "descriptors": descriptors,
    }


def extract_feature(path: Path, source: str) -> dict[str, Any]:
    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
        canonical = np.asarray(rgb.resize((64, 64), RESAMPLE), dtype=np.uint8)
    return feature_from_arrays(
        gray, canonical, source=source, identifier=path.stem, path=path,
    )


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
                    stage=f"local_features_{source.lower()}",
                )
    return results


def descriptor_keys(descriptors: np.ndarray, offset: int) -> np.ndarray:
    if len(descriptors) == 0:
        return np.empty((0,), dtype=np.uint16)
    values = descriptors[:, offset:offset + 2].astype(np.uint16)
    return (values[:, 0] << 8) | values[:, 1]


class MultiIndex:
    """Eight sorted exact-subkey tables plus deterministic pHash fallback."""

    def __init__(self, features: list[dict[str, Any]]) -> None:
        if len(features) > np.iinfo(np.uint16).max:
            raise ValueError("multi-index scene count exceeds uint16 identity capacity")
        self.features = features
        self.ids = np.asarray([item["full_id"] for item in features])
        self.phashes = np.stack([item["phashes"] for item in features])
        self.tables: list[tuple[np.ndarray, np.ndarray]] = []
        for offset in INDEX_OFFSETS:
            key_parts: list[np.ndarray] = []
            scene_parts: list[np.ndarray] = []
            for scene_index, item in enumerate(features):
                keys = np.unique(descriptor_keys(item["descriptors"], offset))
                if len(keys):
                    key_parts.append(keys)
                    scene_parts.append(np.full(len(keys), scene_index, dtype=np.uint16))
            keys = np.concatenate(key_parts) if key_parts else np.empty((0,), dtype=np.uint16)
            scenes = (
                np.concatenate(scene_parts) if scene_parts
                else np.empty((0,), dtype=np.uint16)
            )
            order = np.argsort(keys, kind="stable")
            self.tables.append((keys[order], scenes[order]))

    def votes(self, query: dict[str, Any]) -> np.ndarray:
        votes = np.zeros(len(self.features), dtype=np.uint16)
        for offset, (keys, scenes) in zip(INDEX_OFFSETS, self.tables):
            query_keys = np.unique(descriptor_keys(query["descriptors"], offset))
            if not len(query_keys) or not len(keys):
                continue
            left = np.searchsorted(keys, query_keys, side="left")
            right = np.searchsorted(keys, query_keys, side="right")
            matched = [scenes[start:stop] for start, stop in zip(left, right) if stop > start]
            if matched:
                counts = np.bincount(
                    np.concatenate(matched).astype(np.int32), minlength=len(votes),
                )
                votes += counts.astype(np.uint16)
        return votes

    def phash_distances(self, query: dict[str, Any]) -> np.ndarray:
        full_to_views = BIT_COUNTS[
            np.bitwise_xor(self.phashes, query["phashes"][0])
        ].sum(axis=-1).min(axis=1)
        views_to_full = BIT_COUNTS[
            np.bitwise_xor(
                self.phashes[:, 0, :][:, None, :], query["phashes"][None, :, :],
            )
        ].sum(axis=-1).min(axis=1)
        return np.minimum(full_to_views, views_to_full)

    def retrieve(
        self,
        query: dict[str, Any],
        *,
        inherited: Iterable[int] = (),
    ) -> dict[str, Any]:
        votes = self.votes(query)
        distances = self.phash_distances(query)
        orb_order = np.lexsort((self.ids, distances, -votes.astype(np.int32)))
        orb_order = orb_order[:min(ORB_RETRIEVAL_K, len(orb_order))]
        phash_order = np.lexsort((self.ids, -votes.astype(np.int32), distances))
        phash_order = phash_order[:min(PHASH_RETRIEVAL_K, len(phash_order))]
        retrieval: list[int] = []
        seen: set[int] = set()
        for candidate in list(inherited) + orb_order.tolist() + phash_order.tolist():
            value = int(candidate)
            if value not in seen:
                seen.add(value)
                retrieval.append(value)
        prefilter: list[int] = []
        seen.clear()
        for candidate in (
            list(inherited)
            + orb_order[:ORB_PREFILTER_K].tolist()
            + phash_order[:PHASH_PREFILTER_K].tolist()
        ):
            value = int(candidate)
            if value not in seen:
                seen.add(value)
                prefilter.append(value)
        return {
            "votes": votes,
            "distances": distances,
            "orb_order": orb_order.tolist(),
            "phash_order": phash_order.tolist(),
            "retrieval": retrieval,
            "prefilter": prefilter,
        }


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


def homography_result(
    left_points: np.ndarray,
    right_points: np.ndarray,
    reciprocal: list[tuple[int, int]],
    *,
    reprojection: float,
    confidence: float,
    iterations: int,
    minimum_reciprocal: int,
    minimum_inliers: int,
    minimum_ratio: float,
) -> dict[str, Any]:
    result = {
        "reciprocal_matches": len(reciprocal),
        "homography_inliers": 0,
        "homography_inlier_ratio": 0.0,
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
    inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = inliers / len(reciprocal)
    result.update({
        "homography_inliers": inliers,
        "homography_inlier_ratio": inlier_ratio,
        "accepted": (
            len(reciprocal) >= minimum_reciprocal
            and inliers >= minimum_inliers
            and inlier_ratio >= minimum_ratio
        ),
    })
    return result


def orb_geometry(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    forward = ratio_pairs(
        left["descriptors"], right["descriptors"],
        norm=cv2.NORM_HAMMING, ratio=ORB_LOWE_RATIO,
    )
    reverse_raw = ratio_pairs(
        right["descriptors"], left["descriptors"],
        norm=cv2.NORM_HAMMING, ratio=ORB_LOWE_RATIO,
    )
    reverse = {(right_index, left_index) for left_index, right_index in reverse_raw}
    reciprocal = sorted(forward & reverse)
    result = homography_result(
        left["points"], right["points"], reciprocal,
        reprojection=ORB_RANSAC_REPROJECTION,
        confidence=ORB_RANSAC_CONFIDENCE,
        iterations=ORB_RANSAC_ITERATIONS,
        minimum_reciprocal=ORB_MIN_RECIPROCAL,
        minimum_inliers=ORB_MIN_INLIERS,
        minimum_ratio=ORB_MIN_INLIER_RATIO,
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


def rootsift(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    image = gray_image(path)
    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
        contrastThreshold=SIFT_CONTRAST,
        edgeThreshold=SIFT_EDGE,
        sigma=SIFT_SIGMA,
    )
    keypoints, descriptors = sift.detectAndCompute(image, None)
    points = np.asarray([item.pt for item in keypoints], dtype=np.float32)
    if descriptors is None or len(descriptors) == 0:
        return points, None
    descriptors = descriptors.astype(np.float32)
    descriptors /= np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12)
    return points, np.sqrt(descriptors)


class RootSiftCache:
    def __init__(self, capacity: int = 96) -> None:
        self.capacity = capacity
        self.values: OrderedDict[str, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()

    def get(self, path: Path) -> tuple[np.ndarray, np.ndarray | None]:
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


def rootsift_geometry(
    left_path: Path,
    right_path: Path,
    cache: RootSiftCache,
) -> dict[str, Any]:
    left_points, left_descriptors = cache.get(left_path)
    right_points, right_descriptors = cache.get(right_path)
    empty = {
        "left_keypoints": len(left_points),
        "right_keypoints": len(right_points),
        "forward_ratio_matches": 0,
        "reverse_ratio_matches": 0,
        "reciprocal_matches": 0,
        "homography_inliers": 0,
        "homography_inlier_ratio": 0.0,
        "accepted": False,
    }
    if left_descriptors is None or right_descriptors is None:
        return empty
    forward = ratio_pairs(
        left_descriptors, right_descriptors,
        norm=cv2.NORM_L2, ratio=SIFT_LOWE_RATIO,
    )
    reverse_raw = ratio_pairs(
        right_descriptors, left_descriptors,
        norm=cv2.NORM_L2, ratio=SIFT_LOWE_RATIO,
    )
    reverse = {(right_index, left_index) for left_index, right_index in reverse_raw}
    reciprocal = sorted(forward & reverse)
    result = homography_result(
        left_points,
        right_points,
        reciprocal,
        reprojection=SIFT_RANSAC_REPROJECTION,
        confidence=SIFT_RANSAC_CONFIDENCE,
        iterations=SIFT_RANSAC_ITERATIONS,
        minimum_reciprocal=SIFT_MIN_RECIPROCAL,
        minimum_inliers=SIFT_MIN_INLIERS,
        minimum_ratio=SIFT_MIN_INLIER_RATIO,
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
    encoded = cv2.imencode(".jpg", transformed, [cv2.IMWRITE_JPEG_QUALITY, 82])[1]
    transformed = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    unrelated = generator.integers(0, 256, size=(512, 512), dtype=np.uint8)
    unrelated = cv2.GaussianBlur(unrelated, (3, 3), 0.7)
    return base, transformed, unrelated


def temporary_feature(gray: np.ndarray, identifier: str) -> dict[str, Any]:
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    canonical = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    return feature_from_arrays(
        gray, canonical, source="SYNTHETIC", identifier=identifier, path=None,
    )


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    checks = {
        "cpu_exact_mode": context.engineering_contract["mode"] == "cpu_exact",
        "cpu_contract": context.device == "cpu",
        "datasets_hidden_from_contract": not any(
            key in context.assets for key in ("reside_root", "haze4k_root")
        ),
        "parent_conclusions_available": all(
            key in context.assets for key in ("prior_its_conclusion", "ots_geometry_conclusion")
        ),
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
        "opencv_orb_and_sift_available": hasattr(cv2, "ORB_create") and hasattr(cv2, "SIFT_create"),
        "no_model_training_or_inference_path": True,
    }
    prior = json.loads(asset_path(
        context, "prior_its_conclusion", kind="file",
    ).read_text(encoding="utf-8"))
    ots = json.loads(asset_path(
        context, "ots_geometry_conclusion", kind="file",
    ).read_text(encoding="utf-8"))
    checks.update({
        "prior_terminal_is_method_fail": (
            prior.get("decision") == "ITS_SOURCE_EXCLUSION_FAIL"
            and prior.get("authorizes") == "NONE"
        ),
        "ots_geometry_terminal_is_pass": ots.get("decision") == "OTS_TARGETED_GEOMETRY_PASS",
    })

    generator = np.random.default_rng(160492)
    started = time.monotonic()
    synthetic_features: list[dict[str, Any]] = []
    for scene in range(EXPECTED_ITS_TRAIN + EXPECTED_ITS_VALIDATION):
        synthetic_features.append({
            "full_id": f"SYNTHETIC:{scene:05d}",
            "descriptors": generator.integers(
                0, 256, size=(ORB_INDEX_FEATURES, 32), dtype=np.uint8,
            ),
            "phashes": generator.integers(0, 256, size=(7, 8), dtype=np.uint8),
        })
    index = MultiIndex(synthetic_features)
    retrieval_sizes = []
    for _ in range(EXPECTED_HAZE4K_GROUPS):
        query = {
            "descriptors": generator.integers(
                0, 256, size=(ORB_INDEX_FEATURES, 32), dtype=np.uint8,
            ),
            "phashes": generator.integers(0, 256, size=(7, 8), dtype=np.uint8),
        }
        result = index.retrieve(query)
        retrieval_sizes.append((len(result["retrieval"]), len(result["prefilter"])))
    elapsed = time.monotonic() - started

    base, transformed, unrelated = synthetic_images()
    base_feature = temporary_feature(base, "base")
    transformed_feature = temporary_feature(transformed, "transformed")
    unrelated_feature = temporary_feature(unrelated, "unrelated")
    orb_positive = orb_geometry(base_feature, transformed_feature)
    orb_negative = orb_geometry(base_feature, unrelated_feature)
    with tempfile.TemporaryDirectory(prefix="its-geometry-contract-") as directory:
        root = Path(directory)
        base_path = root / "base.png"
        transformed_path = root / "transformed.png"
        unrelated_path = root / "unrelated.png"
        if not all([
            cv2.imwrite(str(base_path), base),
            cv2.imwrite(str(transformed_path), transformed),
            cv2.imwrite(str(unrelated_path), unrelated),
        ]):
            raise ValueError("cannot write synthetic contract images")
        cache = RootSiftCache(capacity=4)
        sift_positive = rootsift_geometry(base_path, transformed_path, cache)
        sift_negative = rootsift_geometry(base_path, unrelated_path, cache)

    checks.update({
        "same_scale_reference_scenes": len(synthetic_features) == 11000,
        "same_scale_query_scenes": len(retrieval_sizes) == 1000,
        "same_scale_retrieval_nonempty": all(a > 0 and b > 0 for a, b in retrieval_sizes),
        "same_scale_elapsed_bound": elapsed <= 240.0,
        "same_scale_index_memory_bound": sum(
            keys.nbytes + scenes.nbytes for keys, scenes in index.tables
        ) <= 256 * 1024 * 1024,
        "synthetic_orb_positive_accepted": bool(orb_positive["accepted"]),
        "synthetic_orb_negative_rejected": not bool(orb_negative["accepted"]),
        "synthetic_rootsift_positive_accepted": bool(sift_positive["accepted"]),
        "synthetic_rootsift_negative_rejected": not bool(sift_negative["accepted"]),
        "output_and_finalizer_contract": True,
    })
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "cpu_exact",
            "device": "cpu",
            "fixture": {
                "batch": 11000,
                "channels": 8,
                "height": ORB_INDEX_FEATURES,
                "width": 32,
            },
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def resize_jpeg_control(feature: dict[str, Any]) -> dict[str, Any]:
    with Image.open(feature["path"]) as opened:
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
            transformed_rgb = transformed.convert("RGB")
            gray = np.asarray(transformed_rgb.convert("L"), dtype=np.uint8)
            canonical = np.asarray(
                transformed_rgb.resize((64, 64), RESAMPLE), dtype=np.uint8,
            )
    return feature_from_arrays(
        gray,
        canonical,
        source="RESIZE_JPEG_CONTROL",
        identifier=feature["id"],
        path=None,
    )


def local_control(feature: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
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
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    transformed_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(transformed_rgb, cv2.COLOR_RGB2GRAY)
    canonical = cv2.resize(transformed_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    return (
        feature_from_arrays(
            gray,
            canonical,
            source="LOCAL_CONTROL",
            identifier=feature["id"],
            path=None,
        ),
        gray,
    )


def candidate_geometry(
    query: dict[str, Any],
    query_path: Path,
    index: MultiIndex,
    retrieval: dict[str, Any],
    cache: RootSiftCache,
) -> list[dict[str, Any]]:
    orb_rows: list[dict[str, Any]] = []
    for candidate_index in retrieval["prefilter"]:
        candidate = index.features[candidate_index]
        orb = orb_geometry(query, candidate)
        orb_rows.append({
            "candidate_index": candidate_index,
            "candidate_id": candidate["full_id"],
            "orb_votes": int(retrieval["votes"][candidate_index]),
            "phash_distance": int(retrieval["distances"][candidate_index]),
            "orb": orb,
        })
    orb_rows.sort(key=lambda row: (
        -int(row["orb"]["accepted"]),
        -row["orb"]["homography_inliers"],
        -row["orb"]["homography_inlier_ratio"],
        -row["orb"]["reciprocal_matches"],
        -row["orb_votes"],
        row["phash_distance"],
        row["candidate_id"],
    ))
    final_rows: list[dict[str, Any]] = []
    for row in [item for item in orb_rows if item["orb"]["accepted"]][:FINAL_VERIFY_K]:
        candidate = index.features[row["candidate_index"]]
        sift = rootsift_geometry(query_path, candidate["path"], cache)
        row["rootsift"] = sift
        row["accepted"] = bool(row["orb"]["accepted"] and sift["accepted"])
        final_rows.append(row)
    return final_rows


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: (
        -int(row["accepted"]),
        -row["rootsift"]["homography_inliers"],
        -row["rootsift"]["homography_inlier_ratio"],
        -row["rootsift"]["reciprocal_matches"],
        -row["orb"]["homography_inliers"],
        -row["orb_votes"],
        row["phash_distance"],
        row["candidate_id"],
    ))[0]


def accepted_parent_pairs(context: Any) -> list[dict[str, str]]:
    with asset_path(context, "prior_its_mapping", kind="file").open(
        "r", encoding="utf-8", newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row.get("accepted") == "True"]


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


def family_expand(
    seeds: set[int],
    index: MultiIndex,
    cache: RootSiftCache,
) -> tuple[set[int], int, bool]:
    members = set(seeds)
    queue = deque(sorted(seeds))
    accepted_edges = 0
    while queue:
        source_index = queue.popleft()
        source = index.features[source_index]
        retrieval = index.retrieve(source)
        rows = candidate_geometry(source, source["path"], index, retrieval, cache)
        for row in rows:
            candidate_index = row["candidate_index"]
            if not row["accepted"] or candidate_index == source_index:
                continue
            accepted_edges += 1
            if candidate_index not in members:
                members.add(candidate_index)
                queue.append(candidate_index)
                if len(members) > FAMILY_MAX_MEMBERS:
                    return members, accepted_edges, False
    return members, accepted_edges, True


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    if context.total_units != TOTAL_UNITS:
        raise ValueError("runtime total_units differs from frozen geometric census")
    workers = int(os.environ.get("CONVIR_ROUTE_ITS_GEOMETRY_WORKERS", "8"))
    if workers != 8:
        raise ValueError("geometric worker count differs from frozen contract")
    cv2.setNumThreads(1)
    reside = asset_path(context, "reside_root", kind="directory")
    haze4k = asset_path(context, "haze4k_root", kind="directory")

    prior_conclusion = json.loads(asset_path(
        context, "prior_its_conclusion", kind="file",
    ).read_text(encoding="utf-8"))
    prior_summary = json.loads(asset_path(
        context, "prior_its_summary", kind="file",
    ).read_text(encoding="utf-8"))
    ots_conclusion = json.loads(asset_path(
        context, "ots_geometry_conclusion", kind="file",
    ).read_text(encoding="utf-8"))
    parent_rows = accepted_parent_pairs(context)
    parent_identity = (
        prior_conclusion.get("decision") == "ITS_SOURCE_EXCLUSION_FAIL"
        and prior_conclusion.get("authorizes") == "NONE"
        and prior_summary.get("source_relationships", {}).get("accepted_relationships")
        == EXPECTED_PARENT_ACCEPTED
        and len(parent_rows) == EXPECTED_PARENT_ACCEPTED
        and ots_conclusion.get("decision") == "OTS_TARGETED_GEOMETRY_PASS"
    )

    identity_paths = {
        "archive_manifest_sha256": reside / "ARCHIVE_SHA256SUMS.txt",
        "pairing_report_sha256": reside / "PAIRING_VALIDATION.txt",
        "layout_record_sha256": reside / "DATASET_LAYOUT.txt",
    }
    observed_identity = {key: sha256_file(path) for key, path in identity_paths.items()}
    its_train_paths = image_files(reside / "official/ITS/train/ITS_clear")
    its_validation_paths = image_files(reside / "official/ITS/val/clear")
    haze_train_paths = image_files(haze4k / "train/gt")
    haze_test_paths = image_files(haze4k / "test/gt")
    sots_paths = image_files(reside / "official/SOTS/outdoor/gt")
    dataset_identity = (
        observed_identity == EXPECTED_RESIDE_IDENTITY
        and len(its_train_paths) == EXPECTED_ITS_TRAIN
        and len(its_validation_paths) == EXPECTED_ITS_VALIDATION
        and len(haze_train_paths) + len(haze_test_paths) == EXPECTED_HAZE4K_GT
        and len(sots_paths) == EXPECTED_SOTS_OUTDOOR
    )

    offset = 0
    its_train = extract_batch(
        its_train_paths, "ITS_TRAIN", workers=workers, context=context, offset=offset,
    )
    offset += len(its_train)
    its_validation = extract_batch(
        its_validation_paths, "ITS_VALIDATION", workers=workers, context=context, offset=offset,
    )
    offset += len(its_validation)
    haze_train = extract_batch(
        haze_train_paths, "HAZE4K_TRAIN", workers=workers, context=context, offset=offset,
    )
    offset += len(haze_train)
    haze_test = extract_batch(
        haze_test_paths, "HAZE4K_TEST", workers=workers, context=context, offset=offset,
    )
    offset += len(haze_test)
    sots_features = extract_batch(
        sots_paths, "SOTS_OUTDOOR_NEGATIVE", workers=workers, context=context, offset=offset,
    )
    offset += len(sots_features)
    if offset != 15492:
        raise ValueError(f"feature workload differs from frozen size: {offset}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in haze_train + haze_test:
        groups[item["canonical"]].append(item)
    alias_histogram = Counter(len(items) for items in groups.values())
    group_integrity = (
        len(groups) == EXPECTED_HAZE4K_GROUPS
        and alias_histogram == {EXPECTED_HAZE4K_ALIASES: EXPECTED_HAZE4K_GROUPS}
    )
    query_digests = sorted(groups)
    queries = [groups[digest][0] for digest in query_digests]
    query_by_digest = dict(zip(query_digests, queries))
    haze_reference_index = MultiIndex(queries)
    its_features = its_train + its_validation
    its_index = MultiIndex(its_features)
    its_by_full_id = {item["full_id"]: index for index, item in enumerate(its_features)}

    inherited_by_digest: dict[str, list[int]] = defaultdict(list)
    for row in parent_rows:
        full_id = row["its_full_id"]
        digest = row["haze4k_group_digest"]
        if full_id in its_by_full_id and digest in query_by_digest:
            inherited_by_digest[digest].append(its_by_full_id[full_id])

    resize_top_one = 0
    resize_top_k = 0
    local_top_k = 0
    local_geometry_passes = 0
    local_geometry_rows: list[dict[str, Any]] = []
    for index, query in enumerate(queries, start=1):
        expected = index - 1
        resize = resize_jpeg_control(query)
        resize_retrieval = haze_reference_index.retrieve(resize)
        resize_top_one += int(resize_retrieval["orb_order"][0] == expected)
        resize_top_k += int(expected in resize_retrieval["orb_order"])
        local, local_gray = local_control(query)
        local_retrieval = haze_reference_index.retrieve(local)
        local_top_k += int(expected in local_retrieval["orb_order"])
        if expected < LOCAL_CONTROL_GEOMETRY_COUNT:
            with tempfile.TemporaryDirectory(prefix="its-local-control-") as directory:
                control_path = Path(directory) / "control.jpg"
                if not cv2.imwrite(str(control_path), local_gray):
                    raise ValueError("cannot write bounded local-control image")
                geometry = rootsift_geometry(
                    query["path"], control_path, RootSiftCache(capacity=4),
                )
            local_geometry_passes += int(geometry["accepted"])
            local_geometry_rows.append(geometry)
        if index == 1 or index % 25 == 0 or index == len(queries):
            write_workload_progress(
                context,
                completed_units=offset + index,
                stage="local_retrieval_controls",
            )
    offset += len(queries)

    cache = RootSiftCache()
    relationship_rows: list[dict[str, Any]] = []
    accepted_seed_indices: set[int] = set()
    accepted_relationships = 0
    prior_controls_recalled = 0
    prior_controls_accepted = 0
    for query_number, (digest, query) in enumerate(zip(query_digests, queries), start=1):
        inherited = inherited_by_digest.get(digest, [])
        retrieval = its_index.retrieve(query, inherited=inherited)
        prior_controls_recalled += sum(int(item in retrieval["prefilter"]) for item in inherited)
        geometry_rows = candidate_geometry(
            query, query["path"], its_index, retrieval, cache,
        )
        accepted_rows = [row for row in geometry_rows if row["accepted"]]
        accepted_ids = sorted({row["candidate_id"] for row in accepted_rows})
        accepted_indices = {row["candidate_index"] for row in accepted_rows}
        accepted_seed_indices.update(accepted_indices)
        accepted_relationships += int(bool(accepted_rows))
        prior_controls_accepted += sum(
            int(any(
                row["candidate_index"] == item and row["accepted"]
                for row in geometry_rows
            ))
            for item in inherited
        )
        best = best_row(geometry_rows)
        relationship_rows.append({
            "haze4k_group_digest": digest,
            "haze4k_source": query["source"],
            "haze4k_id": query["id"],
            "retrieval_candidates": len(retrieval["retrieval"]),
            "prefilter_candidates": len(retrieval["prefilter"]),
            "final_verified_candidates": len(geometry_rows),
            "accepted_candidate_count": len(accepted_ids),
            "accepted_its_ids": ";".join(accepted_ids),
            "best_its_id": "" if best is None else best["candidate_id"],
            "best_orb_votes": "" if best is None else best["orb_votes"],
            "best_phash_distance": "" if best is None else best["phash_distance"],
            "best_orb_reciprocal": "" if best is None else best["orb"]["reciprocal_matches"],
            "best_orb_inliers": "" if best is None else best["orb"]["homography_inliers"],
            "best_orb_inlier_ratio": "" if best is None else best["orb"]["homography_inlier_ratio"],
            "best_rootsift_reciprocal": "" if best is None else best["rootsift"]["reciprocal_matches"],
            "best_rootsift_inliers": "" if best is None else best["rootsift"]["homography_inliers"],
            "best_rootsift_inlier_ratio": "" if best is None else best["rootsift"]["homography_inlier_ratio"],
            "accepted": bool(accepted_rows),
        })
        if query_number == 1 or query_number % 10 == 0 or query_number == len(queries):
            write_workload_progress(
                context,
                completed_units=offset + query_number,
                stage="haze4k_to_its_local_geometric_census",
            )
    offset += len(queries)

    sots_admitted = 0
    sots_final_verifications = 0
    for index, query in enumerate(sots_features, start=1):
        retrieval = its_index.retrieve(query)
        rows = candidate_geometry(query, query["path"], its_index, retrieval, cache)
        sots_final_verifications += len(rows)
        sots_admitted += int(any(row["accepted"] for row in rows))
        if index == 1 or index % 25 == 0 or index == len(sots_features):
            write_workload_progress(
                context,
                completed_units=offset + index,
                stage="sots_outdoor_negative_controls",
            )
    offset += len(sots_features)

    haze_train_index = path_index(haze_train_paths)
    haze_test_index = path_index(haze_test_paths)
    ots_index = path_index(image_files(reside / "official/OTS_ALPHA/clear_images"))
    ots_positive_rows = []
    ots_negative_rows = []
    for pair_index, (source, haze_id, ots_id) in enumerate(POSITIVE_OTS_PAIRS, start=1):
        path = haze_train_index[haze_id] if source == "HAZE4K_TRAIN" else haze_test_index[haze_id]
        ots_positive_rows.append(rootsift_geometry(path, ots_index[ots_id], cache))
        write_workload_progress(
            context,
            completed_units=offset + pair_index,
            stage="ots_positive_geometry_controls",
        )
    offset += len(POSITIVE_OTS_PAIRS)
    for pair_index, (source, haze_id, ots_id) in enumerate(NEGATIVE_OTS_PAIRS, start=1):
        path = haze_train_index[haze_id] if source == "HAZE4K_TRAIN" else haze_test_index[haze_id]
        ots_negative_rows.append(rootsift_geometry(path, ots_index[ots_id], cache))
        write_workload_progress(
            context,
            completed_units=offset + pair_index,
            stage="ots_negative_geometry_controls",
        )
    offset += len(NEGATIVE_OTS_PAIRS)
    if offset != TOTAL_UNITS:
        raise ValueError(f"completed workload differs from frozen total: {offset}")

    family_members, family_edge_count, family_bounded = family_expand(
        accepted_seed_indices, its_index, cache,
    )
    eligible_its = len(its_features) - len(family_members)
    resize_control_ok = (
        resize_top_one == EXPECTED_HAZE4K_GROUPS
        and resize_top_k == EXPECTED_HAZE4K_GROUPS
    )
    local_recall_rate = local_top_k / EXPECTED_HAZE4K_GROUPS
    local_geometry_rate = local_geometry_passes / LOCAL_CONTROL_GEOMETRY_COUNT
    prior_six_ok = (
        len(parent_rows) == EXPECTED_PARENT_ACCEPTED
        and prior_controls_recalled == EXPECTED_PARENT_ACCEPTED
        and prior_controls_accepted == EXPECTED_PARENT_ACCEPTED
    )
    ots_controls_ok = (
        len(ots_positive_rows) == 3
        and all(row["accepted"] for row in ots_positive_rows)
        and len(ots_negative_rows) == 3
        and not any(row["accepted"] for row in ots_negative_rows)
    )
    gates = {
        "dataset_identity": dataset_identity,
        "parent_evidence_identity": parent_identity,
        "haze4k_group_integrity": group_integrity,
        "resize_jpeg_control_recall": resize_control_ok,
        "local_transform_control_recall": local_recall_rate >= LOCAL_CONTROL_MIN_RECALL,
        "local_transform_geometry_acceptance": local_geometry_rate >= LOCAL_CONTROL_MIN_RECALL,
        "prior_six_relationship_controls": prior_six_ok,
        "ots_geometry_controls": ots_controls_ok,
        "sots_negative_specificity": sots_admitted == 0,
        "source_relationship_completeness": accepted_relationships == EXPECTED_RELATIONSHIPS,
        "family_expansion_bound": family_bounded,
        "eligible_its_capacity": eligible_its >= MIN_ELIGIBLE_ITS,
    }
    integrity_keys = (
        "dataset_identity", "parent_evidence_identity", "haze4k_group_integrity",
    )
    if not all(gates[key] for key in integrity_keys):
        state = "COMPLETED_INCONCLUSIVE"
        decision = "ITS_GEOMETRIC_SOURCE_EXCLUSION_INCONCLUSIVE"
        authorizes = "NONE"
    elif all(gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "ITS_GEOMETRIC_SOURCE_EXCLUSION_PASS"
        authorizes = "ITS_DISJOINT_MEASUREMENT_DESIGN"
    else:
        state = "COMPLETED_GATE_FAIL"
        decision = "ITS_GEOMETRIC_SOURCE_EXCLUSION_FAIL"
        authorizes = "NONE"
    gate_reasons = [key for key, passed in gates.items() if not passed]
    if not gate_reasons:
        gate_reasons = ["all_frozen_identity_control_relationship_family_and_capacity_gates_passed"]
    authorized_exclusions = (
        sorted(its_features[index]["full_id"] for index in family_members)
        if state == "COMPLETED_GATE_PASS" else []
    )

    mapping_path = output_file(context, "haze4k_its_geometric_relationships.csv")
    fields = list(relationship_rows[0])
    with mapping_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(relationship_rows)
    output_file(context, "its_geometric_exclusion_ids.txt").write_text(
        ("\n".join(authorized_exclusions) + "\n") if authorized_exclusions else "",
        encoding="utf-8",
    )
    accepted_rows = [row for row in relationship_rows if row["accepted"]]
    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "full Haze4K-to-RESIDE-ITS local-geometric source census",
        "dataset_identity": {
            "observed": observed_identity,
            "expected": EXPECTED_RESIDE_IDENTITY,
            "matched": dataset_identity,
            "counts": {
                "its_train_clear": len(its_train),
                "its_validation_clear": len(its_validation),
                "haze4k_gt": len(haze_train) + len(haze_test),
                "sots_outdoor": len(sots_features),
            },
        },
        "parent_evidence": {
            "matched": parent_identity,
            "accepted_control_pairs": len(parent_rows),
            "prior_controls_recalled": prior_controls_recalled,
            "prior_controls_reaccepted": prior_controls_accepted,
        },
        "haze4k_grouping": {
            "canonical_groups": len(groups),
            "alias_histogram": dict(sorted(alias_histogram.items())),
            "matched": group_integrity,
        },
        "method": {
            "retrieval": {
                "descriptor": "ORB",
                "detected_features": ORB_FEATURES,
                "indexed_features": ORB_INDEX_FEATURES,
                "index_offsets": list(INDEX_OFFSETS),
                "orb_retrieval_k": ORB_RETRIEVAL_K,
                "phash_retrieval_k": PHASH_RETRIEVAL_K,
                "prefilter_max_candidates": ORB_PREFILTER_K + PHASH_PREFILTER_K,
            },
            "orb_prefilter": {
                "lowe_ratio": ORB_LOWE_RATIO,
                "minimum_reciprocal": ORB_MIN_RECIPROCAL,
                "minimum_inliers": ORB_MIN_INLIERS,
                "minimum_inlier_ratio": ORB_MIN_INLIER_RATIO,
                "ransac_reprojection": ORB_RANSAC_REPROJECTION,
            },
            "final_rootsift": {
                "lowe_ratio": SIFT_LOWE_RATIO,
                "minimum_reciprocal": SIFT_MIN_RECIPROCAL,
                "minimum_inliers": SIFT_MIN_INLIERS,
                "minimum_inlier_ratio": SIFT_MIN_INLIER_RATIO,
                "ransac_reprojection": SIFT_RANSAC_REPROJECTION,
                "maximum_verified_per_query": FINAL_VERIFY_K,
            },
        },
        "controls": {
            "resize_jpeg_top_one": resize_top_one,
            "resize_jpeg_top_96": resize_top_k,
            "local_transform_top_96": local_top_k,
            "local_transform_recall_rate": local_recall_rate,
            "local_geometry_passes": local_geometry_passes,
            "local_geometry_count": LOCAL_CONTROL_GEOMETRY_COUNT,
            "local_geometry_acceptance_rate": local_geometry_rate,
            "local_geometry_inliers": quantiles([
                float(row["homography_inliers"]) for row in local_geometry_rows
            ]),
            "ots_positive_accepted": sum(int(row["accepted"]) for row in ots_positive_rows),
            "ots_negative_accepted": sum(int(row["accepted"]) for row in ots_negative_rows),
            "sots_queries": len(sots_features),
            "sots_final_verifications": sots_final_verifications,
            "sots_admitted": sots_admitted,
        },
        "relationships": {
            "queried_haze4k_groups": len(relationship_rows),
            "accepted_relationships": accepted_relationships,
            "expected_relationships": EXPECTED_RELATIONSHIPS,
            "accepted_candidate_ids": len(accepted_seed_indices),
            "accepted_train": sum("ITS_TRAIN:" in row["accepted_its_ids"] for row in accepted_rows),
            "accepted_validation": sum(
                "ITS_VALIDATION:" in row["accepted_its_ids"] for row in accepted_rows
            ),
            "rootsift_inliers": quantiles([
                float(row["best_rootsift_inliers"]) for row in accepted_rows
                if row["best_rootsift_inliers"] != ""
            ]),
            "interpretation_limit": "A shortfall is a failure of this frozen two-stage matcher, not proof that the remaining relationships or source overlap are absent.",
        },
        "family_exclusion": {
            "seed_count": len(accepted_seed_indices),
            "provisional_family_members": len(family_members),
            "accepted_family_edges": family_edge_count,
            "bounded": family_bounded,
            "maximum_members": FAMILY_MAX_MEMBERS,
            "eligible_its_clear_scenes": eligible_its,
            "authorized_exclusion_count": len(authorized_exclusions),
            "authorized_exclusion_digest": digest_lines(authorized_exclusions),
        },
        "gates": gates,
        "terminal": {
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "limitations": [
            "This route qualifies source identity only and does not validate model behavior or an ITS measurement target.",
            "The expected 500 relationships are provenance-derived and were not used to tune thresholds or stop retrieval.",
            "A scientific FAIL is method-specific and cannot establish absence of source overlap or unseen-scene status.",
            "The exclusion asset is nonempty and reusable only under COMPLETED_GATE_PASS.",
            "No training, inference, dehazing outcome, confirmation, canary or locked-test evidence was accessed.",
        ],
        "marker": "RESIDE_ITS_GEOMETRIC_SOURCE_EXCLUSION_V1_COMPLETE",
    }
    output_file(context, "its_geometric_source_exclusion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    write_workload_progress(context, completed_units=TOTAL_UNITS, stage="geometric_finalize")
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "its_geometric_source_exclusion_summary.json",
            "accepted_relationships": accepted_relationships,
            "provisional_family_members": len(family_members),
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
