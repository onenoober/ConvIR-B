#!/usr/bin/env python3
"""Resolve three targeted Haze4K-to-OTS relationships with local geometry."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from resolve_reside_ots_outdoor_overlap import (
    candidate_order,
    digest_lines,
    fingerprint,
    fingerprint_batch,
    image_files,
    robust_scores,
    sha256_file,
)
from route_program_api import (
    asset_path,
    load_context,
    output_file,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
    write_workload_progress,
)


EXPECTED_OTS = 8970
EXPECTED_HAZE4K_GT = 4000
EXPECTED_PARENT_ROWS = 542
EXPECTED_PARENT_VERIFIED = 39
EXPECTED_PARENT_EXCLUSIONS = 959
EXPECTED_RELATIONSHIPS = 42
MIN_ELIGIBLE_OTS = 6000
SHORTLIST_SIZE = 8
TOTAL_UNITS = 8987

SIFT_FEATURES = 8000
SIFT_CONTRAST = 0.01
SIFT_EDGE = 10
SIFT_SIGMA = 1.6
LOWE_RATIO = 0.75
RANSAC_REPROJECTION = 4.0
RANSAC_CONFIDENCE = 0.995
RANSAC_ITERATIONS = 5000
MIN_RECIPROCAL = 20
MIN_INLIERS = 15
MIN_INLIER_RATIO = 0.35

POSITIVE_PAIRS = (
    ("positive_control", "HAZE4K_TRAIN", "101", "0870"),
    ("positive_control", "HAZE4K_TRAIN", "133", "7267"),
    ("positive_control", "HAZE4K_TEST", "417", "6360"),
)
NEGATIVE_PAIRS = (
    ("negative_control", "HAZE4K_TRAIN", "101", "7267"),
    ("negative_control", "HAZE4K_TRAIN", "133", "6360"),
    ("negative_control", "HAZE4K_TEST", "417", "0870"),
)
FIXED_TARGETS = (
    ("target_69", "HAZE4K_TEST", "69", "0354"),
    ("target_977", "HAZE4K_TRAIN", "977", "1964"),
)
PARENT_TARGETS = {
    ("HAZE4K_TEST", "69"): {
        "ots_scene_id": "0354",
        "candidate_hamming_distance": 0,
        "correlation_score": 0.9999719858169556,
        "score_gap": 0.10664135217666626,
        "reverse_score_gap": 0.20226562023162842,
        "mutual_nearest": True,
        "verified": False,
    },
    ("HAZE4K_TRAIN", "977"): {
        "ots_scene_id": "1964",
        "candidate_hamming_distance": 0,
        "correlation_score": 0.9999347925186157,
        "score_gap": 0.25782084465026855,
        "reverse_score_gap": 0.2406732439994812,
        "mutual_nearest": True,
        "verified": False,
    },
    ("HAZE4K_TRAIN", "937"): {
        "ots_scene_id": "0230",
        "candidate_hamming_distance": 0,
        "correlation_score": 0.9999794363975525,
        "score_gap": 0.0007998943328857422,
        "reverse_score_gap": None,
        "mutual_nearest": False,
        "verified": False,
    },
}


def _rootsift(gray: np.ndarray) -> tuple[list[Any], np.ndarray | None]:
    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
        contrastThreshold=SIFT_CONTRAST,
        edgeThreshold=SIFT_EDGE,
        sigma=SIFT_SIGMA,
    )
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(descriptors) == 0:
        return keypoints, None
    descriptors = descriptors.astype(np.float32)
    descriptors /= np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12)
    return keypoints, np.sqrt(descriptors)


def _ratio_pairs(left: np.ndarray, right: np.ndarray) -> set[tuple[int, int]]:
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    accepted: set[tuple[int, int]] = set()
    for neighbours in matcher.knnMatch(left, right, k=2):
        if len(neighbours) == 2 and neighbours[0].distance < LOWE_RATIO * neighbours[1].distance:
            accepted.add((neighbours[0].queryIdx, neighbours[0].trainIdx))
    return accepted


def geometric_pair(left_gray: np.ndarray, right_gray: np.ndarray) -> dict[str, Any]:
    left_keypoints, left_descriptors = _rootsift(left_gray)
    right_keypoints, right_descriptors = _rootsift(right_gray)
    result: dict[str, Any] = {
        "left_keypoints": len(left_keypoints),
        "right_keypoints": len(right_keypoints),
        "forward_ratio_matches": 0,
        "reverse_ratio_matches": 0,
        "reciprocal_matches": 0,
        "homography_inliers": 0,
        "homography_inlier_ratio": 0.0,
        "accepted": False,
    }
    if left_descriptors is None or right_descriptors is None:
        return result
    forward = _ratio_pairs(left_descriptors, right_descriptors)
    reverse_raw = _ratio_pairs(right_descriptors, left_descriptors)
    reverse = {(right, left) for left, right in reverse_raw}
    reciprocal = sorted(forward & reverse)
    result["forward_ratio_matches"] = len(forward)
    result["reverse_ratio_matches"] = len(reverse_raw)
    result["reciprocal_matches"] = len(reciprocal)
    if len(reciprocal) < 4:
        return result
    left_points = np.float32([left_keypoints[left].pt for left, _ in reciprocal]).reshape(-1, 1, 2)
    right_points = np.float32([right_keypoints[right].pt for _, right in reciprocal]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(
        left_points,
        right_points,
        cv2.RANSAC,
        RANSAC_REPROJECTION,
        maxIters=RANSAC_ITERATIONS,
        confidence=RANSAC_CONFIDENCE,
    )
    inliers = int(mask.sum()) if mask is not None else 0
    ratio = inliers / len(reciprocal)
    result["homography_inliers"] = inliers
    result["homography_inlier_ratio"] = ratio
    result["accepted"] = (
        len(reciprocal) >= MIN_RECIPROCAL
        and inliers >= MIN_INLIERS
        and ratio >= MIN_INLIER_RATIO
    )
    return result


def _synthetic_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(3407)
    base = generator.integers(0, 256, size=(512, 512), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0.7)
    for index in range(16, 496, 32):
        cv2.line(base, (8, index), (503, (index * 7) % 504), int((index * 13) % 256), 2)
        cv2.circle(base, ((index * 11) % 504, index), 7, int((index * 17) % 256), 2)
    source = np.float32([[0, 0], [511, 0], [511, 511], [0, 511]])
    target = np.float32([[12, 18], [494, 5], [505, 496], [20, 505]])
    matrix = cv2.getPerspectiveTransform(source, target)
    transformed = cv2.warpPerspective(base, matrix, (512, 512))
    unrelated = generator.integers(0, 256, size=(512, 512), dtype=np.uint8)
    unrelated = cv2.GaussianBlur(unrelated, (3, 3), 0.7)
    return base, transformed, unrelated


def _rank_shortlist(
    query: dict[str, Any],
    ots_items: list[dict[str, Any]],
    ots_view_hashes: np.ndarray,
    ots_ids: np.ndarray,
) -> list[dict[str, Any]]:
    candidates, distances = candidate_order(query, ots_view_hashes, ots_ids)
    views = np.stack([ots_items[index]["views"] for index in candidates.tolist()])
    scores = robust_scores(query["views"], views)
    positions = sorted(
        range(len(candidates)),
        key=lambda position: (
            -float(scores[position]),
            int(distances[candidates[position]]),
            str(ots_ids[candidates[position]]),
        ),
    )[:SHORTLIST_SIZE]
    return [
        {
            "ots_scene_id": str(ots_ids[int(candidates[position])]),
            "candidate_index": int(candidates[position]),
            "candidate_hamming_distance": int(distances[int(candidates[position])]),
            "global_correlation_score": float(scores[position]),
            "global_rank": rank,
        }
        for rank, position in enumerate(positions, start=1)
    ]


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    parent = context.assets.get("parent_closeout")
    checks = {
        "cpu_exact_mode": context.engineering_contract["mode"] == "cpu_exact",
        "cpu_contract": context.device == "cpu",
        "datasets_hidden_from_contract": (
            "reside_root" not in context.assets and "haze4k_root" not in context.assets
        ),
        "parent_identity_bound": parent is not None and parent.contract_access is True,
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
        "opencv_sift_available": hasattr(cv2, "SIFT_create"),
        "no_model_training_or_inference_path": True,
    }
    started = time.monotonic()
    base, transformed, unrelated = _synthetic_fixture()
    positive = geometric_pair(base, transformed)
    negative = geometric_pair(base, unrelated)
    generator = np.random.default_rng(937)
    ots_hashes = generator.integers(0, 256, size=(EXPECTED_OTS, 7, 8), dtype=np.uint8)
    query_hashes = generator.integers(0, 256, size=(7, 8), dtype=np.uint8)
    ots_ids = np.asarray([f"synthetic-{index:04d}" for index in range(EXPECTED_OTS)])
    order, distances = candidate_order({"view_hashes": query_hashes}, ots_hashes, ots_ids)
    elapsed = time.monotonic() - started
    checks.update({
        "synthetic_positive_accepted": bool(positive["accepted"]),
        "synthetic_negative_rejected": not bool(negative["accepted"]),
        "same_scale_candidate_count": ots_hashes.shape[0] == EXPECTED_OTS,
        "same_scale_top_k": len(order) == 64,
        "same_scale_finite": bool(np.all(distances[order] >= 0)),
        "same_scale_elapsed_bound": elapsed <= 90.0,
        "same_scale_memory_bound": ots_hashes.nbytes <= 1024 * 1024,
        "output_and_finalizer_contract": True,
    })
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "cpu_exact",
            "device": "cpu",
            "fixture": {"batch": EXPECTED_OTS, "channels": 7, "height": 32, "width": 32},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def _boolean(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"invalid parent Boolean: {value}")
    return value == "True"


def _parent_identity(context: Any) -> tuple[dict[str, Any], set[str], bool]:
    closeout = json.loads(asset_path(context, "parent_closeout", kind="file").read_text(encoding="utf-8"))
    summary = json.loads(asset_path(context, "parent_summary", kind="file").read_text(encoding="utf-8"))
    exclusions = {
        line.strip() for line in asset_path(context, "parent_exclusions", kind="file").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    with asset_path(context, "parent_mapping", kind="file").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_target = {
        (row["haze4k_representative_source"], row["haze4k_representative_id"]): row
        for row in rows
        if (row["haze4k_representative_source"], row["haze4k_representative_id"]) in PARENT_TARGETS
    }
    target_match = len(by_target) == len(PARENT_TARGETS)
    for key, expected in PARENT_TARGETS.items():
        row = by_target.get(key)
        if row is None:
            target_match = False
            continue
        observed = {
            "ots_scene_id": row["ots_scene_id"],
            "candidate_hamming_distance": int(row["candidate_hamming_distance"]),
            "correlation_score": float(row["correlation_score"]),
            "score_gap": float(row["score_gap"]),
            "reverse_score_gap": float(row["reverse_score_gap"]) if row["reverse_score_gap"] else None,
            "mutual_nearest": _boolean(row["mutual_nearest"]),
            "verified": _boolean(row["verified"]),
        }
        target_match = target_match and observed == expected
    identity = (
        closeout.get("state") == "COMPLETED_GATE_FAIL"
        and closeout.get("decision") == "OTS_FULL_RETRIEVAL_FAIL"
        and closeout.get("authorizes") == "NONE"
        and summary.get("full_query_result", {}).get("verified_new_ots_mappings") == EXPECTED_PARENT_VERIFIED
        and len(rows) == EXPECTED_PARENT_ROWS
        and sum(_boolean(row["verified"]) for row in rows) == EXPECTED_PARENT_VERIFIED
        and len(exclusions) == EXPECTED_PARENT_EXCLUSIONS
        and digest_lines(exclusions) == summary.get("exclusion_pool", {}).get("full_exclusion_digest")
        and {"0230", "0354", "1964"}.issubset(exclusions)
        and target_match
    )
    return summary, exclusions, identity


def _gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    longest = max(image.shape)
    if longest > 1600:
        scale = 1600.0 / longest
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def _path_index(paths: list[Path]) -> dict[str, Path]:
    result = {path.stem: path for path in paths}
    if len(result) != len(paths):
        raise ValueError("image stems are not unique within a frozen directory")
    return result


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    if context.total_units != TOTAL_UNITS:
        raise ValueError("runtime total_units differs from the frozen targeted workload")
    parent_summary, parent_exclusions, parent_identity = _parent_identity(context)
    qualification = json.loads(asset_path(
        context, "prior_reside_qualification", kind="file",
    ).read_text(encoding="utf-8"))
    reside = asset_path(context, "reside_root", kind="directory")
    haze4k = asset_path(context, "haze4k_root", kind="directory")
    observed_identity = {
        "archive_manifest_sha256": sha256_file(reside / "ARCHIVE_SHA256SUMS.txt"),
        "pairing_report_sha256": sha256_file(reside / "PAIRING_VALIDATION.txt"),
        "layout_record_sha256": sha256_file(reside / "DATASET_LAYOUT.txt"),
    }
    qualified_identity = qualification.get("dataset_identity", {})
    qualification_match = (
        qualification.get("marker") == "RESIDE_MINIMAL_MEASUREMENT_QUALIFICATION_RECORDED"
        and all(observed_identity[key] == qualified_identity.get(key) for key in observed_identity)
    )

    ots_paths = image_files(reside / "official/OTS_ALPHA/clear_images")
    train_paths = image_files(haze4k / "train/gt")
    test_paths = image_files(haze4k / "test/gt")
    dataset_identity = (
        qualification_match
        and len(ots_paths) == EXPECTED_OTS
        and len(train_paths) + len(test_paths) == EXPECTED_HAZE4K_GT
    )
    train_index = _path_index(train_paths)
    test_index = _path_index(test_paths)
    ots_index_paths = _path_index(ots_paths)

    ots_items = fingerprint_batch(
        paths=ots_paths,
        source="RESIDE_OTS",
        include_views=True,
        workers=8,
        offset=0,
        context=context,
    )
    ots_view_hashes = np.stack([item["view_hashes"] for item in ots_items])
    ots_ids = np.asarray([item["id"] for item in ots_items])
    query_937 = fingerprint(train_index["937"], "HAZE4K_TRAIN", include_views=True)
    shortlist = _rank_shortlist(query_937, ots_items, ots_view_hashes, ots_ids)

    pair_records: list[dict[str, Any]] = []
    pair_specs = list(POSITIVE_PAIRS + NEGATIVE_PAIRS + FIXED_TARGETS)
    pair_specs.extend(
        ("target_937_candidate", "HAZE4K_TRAIN", "937", item["ots_scene_id"])
        for item in shortlist
    )
    shortlist_by_id = {item["ots_scene_id"]: item for item in shortlist}
    for index, (role, source, haze_id, ots_id) in enumerate(pair_specs, start=1):
        haze_path = train_index[haze_id] if source == "HAZE4K_TRAIN" else test_index[haze_id]
        geometry = geometric_pair(_gray(haze_path), _gray(ots_index_paths[ots_id]))
        global_result = shortlist_by_id.get(ots_id, {}) if role == "target_937_candidate" else {}
        pair_records.append({
            "pair_role": role,
            "haze4k_source": source,
            "haze4k_id": haze_id,
            "ots_scene_id": ots_id,
            "global_rank": global_result.get("global_rank"),
            "candidate_hamming_distance": global_result.get("candidate_hamming_distance"),
            "global_correlation_score": global_result.get("global_correlation_score"),
            **geometry,
        })
        write_workload_progress(
            context,
            completed_units=EXPECTED_OTS + index,
            stage="targeted_geometric_pairs",
        )

    positive = [row for row in pair_records if row["pair_role"] == "positive_control"]
    negative = [row for row in pair_records if row["pair_role"] == "negative_control"]
    target_69 = next(row for row in pair_records if row["pair_role"] == "target_69")
    target_977 = next(row for row in pair_records if row["pair_role"] == "target_977")
    target_937_rows = [row for row in pair_records if row["pair_role"] == "target_937_candidate"]
    target_937_ids = sorted(row["ots_scene_id"] for row in target_937_rows if row["accepted"])
    target_937_resolved = bool(target_937_ids) and "0230" in shortlist_by_id
    resolved_relationships = EXPECTED_PARENT_VERIFIED + sum([
        bool(target_69["accepted"]),
        bool(target_977["accepted"]),
        target_937_resolved,
    ])
    final_exclusions = sorted(parent_exclusions | set(target_937_ids))
    eligible_ots = EXPECTED_OTS - len(final_exclusions)

    gates = {
        "parent_identity": parent_identity,
        "dataset_identity": dataset_identity,
        "positive_control_recall": len(positive) == 3 and all(row["accepted"] for row in positive),
        "negative_control_specificity": len(negative) == 3 and not any(row["accepted"] for row in negative),
        "target_69_relationship": bool(target_69["accepted"]),
        "target_977_relationship": bool(target_977["accepted"]),
        "target_937_relationship": target_937_resolved,
        "relationship_completeness": resolved_relationships == EXPECTED_RELATIONSHIPS,
        "eligible_ots_capacity": eligible_ots >= MIN_ELIGIBLE_OTS,
    }
    integrity = ["parent_identity", "dataset_identity"]
    if not all(gates[key] for key in integrity):
        state = "COMPLETED_INCONCLUSIVE"
        decision = "OTS_TARGETED_GEOMETRY_INCONCLUSIVE"
        authorizes = "NONE"
    elif all(gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "OTS_TARGETED_GEOMETRY_PASS"
        authorizes = "OTS_OUTDOOR_MEASUREMENT_DESIGN"
    else:
        state = "COMPLETED_GATE_FAIL"
        decision = "OTS_TARGETED_GEOMETRY_FAIL"
        authorizes = "NONE"
    gate_reasons = [key for key, passed in gates.items() if not passed]
    if not gate_reasons:
        gate_reasons = ["all_frozen_identity_control_relationship_and_capacity_gates_passed"]

    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "targeted geometric resolution of three Haze4K-to-OTS source relationships",
        "dataset_identity": {
            "observed_identity": observed_identity,
            "qualified_identity_match": qualification_match,
            "ots_clear_scenes": len(ots_paths),
            "haze4k_gt_images": len(train_paths) + len(test_paths),
        },
        "parent_identity": {
            "matched": parent_identity,
            "verified_relationships": parent_summary.get("full_query_result", {}).get("verified_new_ots_mappings"),
            "provisional_exclusion_count": len(parent_exclusions),
        },
        "method": {
            "descriptor": "RootSIFT",
            "sift_features": SIFT_FEATURES,
            "sift_contrast_threshold": SIFT_CONTRAST,
            "sift_edge_threshold": SIFT_EDGE,
            "sift_sigma": SIFT_SIGMA,
            "lowe_ratio": LOWE_RATIO,
            "correspondence": "bidirectional ratio-filtered reciprocal intersection",
            "ransac_reprojection_pixels": RANSAC_REPROJECTION,
            "ransac_confidence": RANSAC_CONFIDENCE,
            "ransac_iterations": RANSAC_ITERATIONS,
            "minimum_reciprocal_matches": MIN_RECIPROCAL,
            "minimum_homography_inliers": MIN_INLIERS,
            "minimum_homography_inlier_ratio": MIN_INLIER_RATIO,
            "target_937_shortlist_size": SHORTLIST_SIZE,
        },
        "controls": {
            "positive_count": len(positive),
            "positive_accepted": sum(bool(row["accepted"]) for row in positive),
            "negative_count": len(negative),
            "negative_accepted": sum(bool(row["accepted"]) for row in negative),
        },
        "target_relationships": {
            "haze4k_test_69": {"ots_ids": ["0354"], "resolved": bool(target_69["accepted"]), "already_excluded": "0354" in parent_exclusions},
            "haze4k_train_977": {"ots_ids": ["1964"], "resolved": bool(target_977["accepted"]), "already_excluded": "1964" in parent_exclusions},
            "haze4k_train_937": {
                "shortlist_ids": [row["ots_scene_id"] for row in target_937_rows],
                "accepted_source_family_ids": target_937_ids,
                "ots_0230_in_shortlist": "0230" in shortlist_by_id,
                "resolved": target_937_resolved,
            },
            "resolved_expected_relationships": resolved_relationships,
            "expected_relationships": EXPECTED_RELATIONSHIPS,
        },
        "exclusion_pool": {
            "parent_provisional_count": len(parent_exclusions),
            "accepted_target_937_family_ids": target_937_ids,
            "deduplicated_exclusion_count": len(final_exclusions),
            "deduplicated_exclusion_digest": digest_lines(final_exclusions),
            "authorized_exclusion_digest": digest_lines(final_exclusions) if state == "COMPLETED_GATE_PASS" else None,
            "eligible_ots_scenes": eligible_ots,
            "minimum_required_eligible_scenes": MIN_ELIGIBLE_OTS,
        },
        "gates": gates,
        "terminal": {"state": state, "decision": decision, "authorizes": authorizes, "gate_reasons": gate_reasons},
        "limitations": [
            "The operation qualifies source relationships only and does not validate an outdoor local measurement field.",
            "Multiple geometrically accepted train-937 candidates are treated conservatively as one source family and all excluded.",
            "A geometric FAIL is a method limitation, not proof that a source relationship is absent.",
            "No training, inference, checkpoint selection, dehazing outcome, confirmation, canary, or locked-test metric is accessed.",
        ],
        "marker": "RESIDE_OTS_TARGETED_GEOMETRY_COMPLETE",
    }
    output_file(context, "targeted_geometry_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    fields = [
        "pair_role", "haze4k_source", "haze4k_id", "ots_scene_id", "global_rank",
        "candidate_hamming_distance", "global_correlation_score", "left_keypoints",
        "right_keypoints", "forward_ratio_matches", "reverse_ratio_matches",
        "reciprocal_matches", "homography_inliers", "homography_inlier_ratio", "accepted",
    ]
    with output_file(context, "geometric_pairs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_records)
    output_file(context, "ots_exclusion_ids.txt").write_text(
        "\n".join(final_exclusions) + "\n", encoding="utf-8",
    )
    write_workload_progress(context, completed_units=TOTAL_UNITS, stage="targeted_geometry_finalize")
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "targeted_geometry_summary.json",
            "resolved_expected_relationships": resolved_relationships,
            "accepted_target_937_source_family_ids": target_937_ids,
            "deduplicated_exclusion_count": len(final_exclusions),
            "eligible_ots_scenes": eligible_ots,
            "exclusion_digest": digest_lines(final_exclusions),
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
