#!/usr/bin/env python3
"""Search all strict-unmatched Haze4K source groups against RESIDE OTS."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from resolve_reside_ots_outdoor_overlap import (
    MIN_CORRELATION,
    MIN_SCORE_GAP,
    TOP_K,
    candidate_order,
    digest_lines,
    fingerprint_batch,
    image_files,
    quantiles,
    robust_match,
    robust_scores,
    sha256_file,
    strict_matches,
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


EXPECTED_HAZE4K_GROUPS = 1000
EXPECTED_STRICT_OTS = 458
EXPECTED_STRICT_UNMATCHED = 542
EXPECTED_NEW_OTS = 42
EXPECTED_SOTS_OUTDOOR = 492
EXPECTED_SOTS_INDOOR_NEGATIVES = 50
MIN_ELIGIBLE_OTS = 6000
MAX_PREQUALIFIED = 128
TOTAL_UNITS = 15562


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    parent = context.assets.get("parent_overlap_closeout")
    checks = {
        "cpu_exact_mode": context.engineering_contract["mode"] == "cpu_exact",
        "cpu_contract": context.device == "cpu",
        "datasets_hidden_from_contract": (
            "reside_root" not in context.assets and "haze4k_root" not in context.assets
        ),
        "parent_terminal_identity_bound": (
            parent is not None and parent.contract_access is True
        ),
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
        "no_model_training_or_inference_path": True,
    }
    generator = np.random.default_rng(3407)
    ots_hashes = generator.integers(0, 256, size=(8970, 7, 8), dtype=np.uint8)
    query_hashes = generator.integers(0, 256, size=(542, 7, 8), dtype=np.uint8)
    ots_ids = np.asarray([f"synthetic-{index:04d}" for index in range(8970)])
    started = time.monotonic()
    selected = []
    for query_views in query_hashes:
        order, distances = candidate_order(
            {"view_hashes": query_views}, ots_hashes, ots_ids,
        )
        selected.append((len(order), int(distances[order[0]])))
    elapsed = time.monotonic() - started
    checks.update({
        "same_scale_query_count": len(selected) == 542,
        "same_scale_candidate_count": ots_hashes.shape[0] == 8970,
        "same_scale_top_k": all(count == TOP_K for count, _ in selected),
        "same_scale_finite": all(distance >= 0 for _, distance in selected),
        "same_scale_elapsed_bound": elapsed <= 60.0,
        "same_scale_memory_bound": (ots_hashes.nbytes + query_hashes.nbytes) <= 1024 * 1024,
        "output_and_finalizer_contract": True,
    })
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "cpu_exact",
            "device": "cpu",
            "fixture": {"batch": 542, "channels": 7, "height": 1, "width": 8970},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def _identity_assets(context: Any) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    qualification = json.loads(asset_path(
        context, "prior_reside_qualification", kind="file",
    ).read_text(encoding="utf-8"))
    parent_closeout = json.loads(asset_path(
        context, "parent_overlap_closeout", kind="file",
    ).read_text(encoding="utf-8"))
    parent_summary = json.loads(asset_path(
        context, "parent_overlap_summary", kind="file",
    ).read_text(encoding="utf-8"))
    parent_exclusions = {
        line.strip() for line in asset_path(
            context, "parent_overlap_exclusions", kind="file",
        ).read_text(encoding="utf-8").splitlines() if line.strip()
    }
    if qualification.get("marker") != "RESIDE_MINIMAL_MEASUREMENT_QUALIFICATION_RECORDED":
        raise ValueError("qualified RESIDE identity marker is missing")
    if (
        parent_closeout.get("state") != "COMPLETED_INCONCLUSIVE"
        or parent_closeout.get("decision") != "OTS_OVERLAP_RESOLUTION_INCONCLUSIVE"
        or parent_closeout.get("authorizes") != "NONE"
    ):
        raise ValueError("parent OTS overlap terminal identity differs")
    return qualification, parent_summary, parent_exclusions


def _reverse_mutual(
    unmatched_queries: list[dict[str, Any]],
    records: list[dict[str, Any]],
    ots_items: list[dict[str, Any]],
    ots_index: dict[str, int],
) -> None:
    prelim = [record for record in records if record["prequalified"]]
    unique_ids = sorted({record["ots_scene_id"] for record in prelim})
    if not prelim or len(prelim) > MAX_PREQUALIFIED:
        return
    candidate_views = np.stack([ots_items[ots_index[item]]["views"] for item in unique_ids])
    matrix = np.stack([
        robust_scores(query["views"], candidate_views) for query in unmatched_queries
    ])
    column_by_id = {item: index for index, item in enumerate(unique_ids)}
    for record in prelim:
        column = column_by_id[record["ots_scene_id"]]
        ranking = np.argsort(-matrix[:, column], kind="stable")
        best = int(ranking[0])
        second = int(ranking[1])
        reverse_gap = float(matrix[best, column] - matrix[second, column])
        record["reverse_score_gap"] = reverse_gap
        record["mutual_nearest"] = (
            best == record["query_index"] and reverse_gap >= MIN_SCORE_GAP
        )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    if context.total_units != TOTAL_UNITS:
        raise ValueError("runtime total_units differs from the frozen full-query census")
    qualification, parent_summary, parent_exclusions = _identity_assets(context)
    reside = asset_path(context, "reside_root", kind="directory")
    haze4k = asset_path(context, "haze4k_root", kind="directory")

    observed_identity = {
        "archive_manifest_sha256": sha256_file(reside / "ARCHIVE_SHA256SUMS.txt"),
        "pairing_report_sha256": sha256_file(reside / "PAIRING_VALIDATION.txt"),
        "layout_record_sha256": sha256_file(reside / "DATASET_LAYOUT.txt"),
    }
    qualified_identity = qualification["dataset_identity"]
    identity_match = all(
        observed_identity[key] == qualified_identity[key] for key in observed_identity
    )
    ots_paths = image_files(reside / "official/OTS_ALPHA/clear_images")
    haze4k_train_paths = image_files(haze4k / "train/gt")
    haze4k_test_paths = image_files(haze4k / "test/gt")
    sots_indoor_paths = image_files(reside / "official/SOTS/indoor/gt")
    sots_outdoor_paths = image_files(reside / "official/SOTS/outdoor/gt")
    count_integrity = {
        "ots_clear": len(ots_paths),
        "haze4k_gt": len(haze4k_train_paths) + len(haze4k_test_paths),
        "sots_indoor_gt": len(sots_indoor_paths),
        "sots_outdoor_gt": len(sots_outdoor_paths),
    } == {
        "ots_clear": 8970,
        "haze4k_gt": 4000,
        "sots_indoor_gt": EXPECTED_SOTS_INDOOR_NEGATIVES,
        "sots_outdoor_gt": EXPECTED_SOTS_OUTDOOR,
    }

    workers = 8
    offset = 0
    ots = fingerprint_batch(
        paths=ots_paths, source="RESIDE_OTS", include_views=True,
        workers=workers, offset=offset, context=context,
    )
    offset += len(ots)
    haze4k_train = fingerprint_batch(
        paths=haze4k_train_paths, source="HAZE4K_TRAIN", include_views=True,
        workers=workers, offset=offset, context=context,
    )
    offset += len(haze4k_train)
    haze4k_test = fingerprint_batch(
        paths=haze4k_test_paths, source="HAZE4K_TEST", include_views=True,
        workers=workers, offset=offset, context=context,
    )
    offset += len(haze4k_test)
    sots_indoor = fingerprint_batch(
        paths=sots_indoor_paths, source="SOTS_INDOOR_NEGATIVE", include_views=True,
        workers=workers, offset=offset, context=context,
    )
    offset += len(sots_indoor)
    sots_outdoor = fingerprint_batch(
        paths=sots_outdoor_paths, source="SOTS_OUTDOOR", include_views=False,
        workers=workers, offset=offset, context=context,
    )
    offset += len(sots_outdoor)
    if offset != 13512:
        raise ValueError(f"fingerprint workload differs from frozen size: {offset}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in haze4k_train + haze4k_test:
        groups[item["canonical"]].append(item)
    strict, strict_records = strict_matches(
        dict(groups), ots, progress_offset=offset, context=context,
    )
    offset += len(groups)
    ots_digests = {digest for digest, matches in strict.items() if matches.get("RESIDE_OTS")}
    unmatched_digests = sorted(set(groups) - ots_digests)
    strict_ids = sorted({
        item for digest in ots_digests for item in strict[digest].get("RESIDE_OTS", set())
    })
    strict_mapping_digest = digest_lines("|".join(record) for record in strict_records)
    strict_exclusion_digest = digest_lines(f"RESIDE_OTS:{item}" for item in strict_ids)
    prior_haze4k = qualification["scene_independence"]["haze4k"]
    strict_reproduced = (
        len(groups) == EXPECTED_HAZE4K_GROUPS
        and len(ots_digests) == EXPECTED_STRICT_OTS
        and len(strict_ids) == EXPECTED_STRICT_OTS
        and len(unmatched_digests) == EXPECTED_STRICT_UNMATCHED
        and strict_exclusion_digest == prior_haze4k["ots_exclusion_id_set_digest"]
    )

    ots_by_canonical: dict[str, list[str]] = defaultdict(list)
    for item in ots:
        ots_by_canonical[item["canonical"]].append(item["id"])
    sots_ids = sorted({
        ots_id for item in sots_outdoor
        for ots_id in ots_by_canonical.get(item["canonical"], [])
    })
    sots_digest = digest_lines(f"RESIDE_OTS:{item}" for item in sots_ids)
    prior_sots = qualification["scene_independence"]["sots"]
    sots_reproduced = (
        len(sots_ids) == EXPECTED_SOTS_OUTDOOR
        and sots_digest == prior_sots["ots_exclusion_id_set_digest"]
    )
    recomputed_parent_exclusions = set(strict_ids) | set(sots_ids)
    parent_union_reproduced = (
        recomputed_parent_exclusions == parent_exclusions
        and digest_lines(recomputed_parent_exclusions)
        == parent_summary["exclusion_pool"]["candidate_union_exclusion_digest"]
    )

    ots_hashes = np.stack([item["view_hashes"] for item in ots])
    ots_ids = np.asarray([item["id"] for item in ots])
    ots_index = {item["id"]: index for index, item in enumerate(ots)}
    known_top_k = 0
    known_top_one = 0
    known_scores: list[float] = []
    unmatched_queries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    ordered_queries = sorted(ots_digests) + unmatched_digests
    for index, digest in enumerate(ordered_queries, start=1):
        query = groups[digest][0]
        result = robust_match(query, ots, ots_hashes, ots_ids)
        if digest in ots_digests:
            verified = strict[digest]["RESIDE_OTS"]
            known_top_k += int(bool(set(result["top_k_ids"]) & verified))
            known_top_one += int(result["best_id"] in verified)
            known_scores.append(result["best_score"])
        else:
            query_index = len(unmatched_queries)
            unmatched_queries.append(query)
            records.append({
                "query_index": query_index,
                "haze4k_group_digest": digest,
                "haze4k_representative_source": query["source"],
                "haze4k_representative_id": query["id"],
                "ots_scene_id": result["best_id"],
                "candidate_hamming_distance": result["best_distance"],
                "correlation_score": result["best_score"],
                "score_gap": result["score_gap"],
                "prequalified": (
                    result["best_score"] >= MIN_CORRELATION
                    and result["score_gap"] >= MIN_SCORE_GAP
                ),
                "reverse_score_gap": None,
                "mutual_nearest": False,
                "verified": False,
            })
        if index == 1 or index % 10 == 0 or index == len(ordered_queries):
            write_workload_progress(
                context, completed_units=offset + index,
                stage="full_haze4k_ots_retrieval",
            )
    offset += len(ordered_queries)

    negative_records = []
    for index, query in enumerate(sots_indoor, start=1):
        result = robust_match(query, ots, ots_hashes, ots_ids)
        admitted = (
            result["best_score"] >= MIN_CORRELATION
            and result["score_gap"] >= MIN_SCORE_GAP
        )
        negative_records.append({
            "sots_indoor_id": query["id"],
            "ots_scene_id": result["best_id"],
            "correlation_score": result["best_score"],
            "score_gap": result["score_gap"],
            "admitted": admitted,
        })
        if index == 1 or index % 10 == 0 or index == len(sots_indoor):
            write_workload_progress(
                context, completed_units=offset + index,
                stage="sots_indoor_negative_control",
            )
    offset += len(sots_indoor)
    if offset != TOTAL_UNITS:
        raise ValueError(f"completed workload differs from frozen total: {offset}")

    _reverse_mutual(unmatched_queries, records, ots, ots_index)
    prequalified = [record for record in records if record["prequalified"]]
    selected_counts = Counter(record["ots_scene_id"] for record in prequalified)
    for record in records:
        record["verified"] = (
            record["prequalified"]
            and len(prequalified) <= MAX_PREQUALIFIED
            and selected_counts[record["ots_scene_id"]] == 1
            and record["ots_scene_id"] not in recomputed_parent_exclusions
            and record["mutual_nearest"]
        )
    verified_records = [record for record in records if record["verified"]]
    new_ids = sorted({record["ots_scene_id"] for record in verified_records})
    full_exclusions = sorted(recomputed_parent_exclusions | set(new_ids))
    eligible_ots = len(ots) - len(full_exclusions)
    full_exclusion_digest = digest_lines(full_exclusions)

    gates = {
        "dataset_identity": identity_match and count_integrity,
        "parent_terminal_identity": True,
        "strict_haze4k_ots_reproduced": strict_reproduced,
        "sots_outdoor_exclusion_reproduced": sots_reproduced,
        "parent_exclusion_union_reproduced": parent_union_reproduced,
        "known_top_k_recall": known_top_k == EXPECTED_STRICT_OTS,
        "known_top_one_recovery": known_top_one == EXPECTED_STRICT_OTS,
        "negative_control_specificity": not any(item["admitted"] for item in negative_records),
        "prequalified_bound": len(prequalified) <= MAX_PREQUALIFIED,
        "new_mapping_completeness": len(verified_records) == EXPECTED_NEW_OTS,
        "new_mapping_uniqueness": len(new_ids) == EXPECTED_NEW_OTS,
        "new_mapping_disjointness": not (set(new_ids) & recomputed_parent_exclusions),
        "new_mapping_mutual_nearest": (
            len(verified_records) == EXPECTED_NEW_OTS
            and all(record["mutual_nearest"] for record in verified_records)
        ),
        "eligible_ots_capacity": eligible_ots >= MIN_ELIGIBLE_OTS,
    }
    integrity_gate_ids = [
        "dataset_identity", "parent_terminal_identity",
        "strict_haze4k_ots_reproduced", "sots_outdoor_exclusion_reproduced",
        "parent_exclusion_union_reproduced",
    ]
    if not all(gates[item] for item in integrity_gate_ids):
        state = "COMPLETED_INCONCLUSIVE"
        decision = "OTS_FULL_RETRIEVAL_INCONCLUSIVE"
        authorizes = "NONE"
    elif all(gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "OTS_FULL_RETRIEVAL_PASS"
        authorizes = "OTS_OUTDOOR_MEASUREMENT_DESIGN"
    else:
        state = "COMPLETED_GATE_FAIL"
        decision = "OTS_FULL_RETRIEVAL_FAIL"
        authorizes = "NONE"
    gate_reasons = [key for key, passed in gates.items() if not passed]
    if not gate_reasons:
        gate_reasons = ["all_frozen_identity_control_mapping_and_capacity_gates_passed"]

    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "full 542-query Haze4K-to-RESIDE-OTS source identity audit",
        "dataset_identity": {
            "observed_identity": observed_identity,
            "qualified_identity_match": identity_match,
            "count_integrity": count_integrity,
            "haze4k_unique_canonical_groups": len(groups),
        },
        "strict_reproduction": {
            "strict_ots_groups": len(ots_digests),
            "strict_unmatched_groups": len(unmatched_digests),
            "ots_only_mapping_digest": strict_mapping_digest,
            "ots_exclusion_digest": strict_exclusion_digest,
            "sots_outdoor_exclusion_digest": sots_digest,
            "parent_union_reproduced": parent_union_reproduced,
        },
        "matcher_calibration": {
            "method": "frozen top-64 symmetric seven-view pHash retrieval with maximum multi-view normalized grayscale correlation",
            "known_positive_count": EXPECTED_STRICT_OTS,
            "known_top_k_recovered": known_top_k,
            "known_top_one_recovered": known_top_one,
            "known_top_one_score": quantiles(known_scores),
            "negative_control_count": len(negative_records),
            "negative_controls_admitted": sum(item["admitted"] for item in negative_records),
            "minimum_correlation": MIN_CORRELATION,
            "minimum_score_gap": MIN_SCORE_GAP,
        },
        "full_query_result": {
            "queried_strict_unmatched_groups": len(records),
            "prequalified_groups": len(prequalified),
            "verified_new_ots_mappings": len(verified_records),
            "unique_new_ots_ids": len(new_ids),
            "score": quantiles([item["correlation_score"] for item in records]),
            "score_gap": quantiles([item["score_gap"] for item in records]),
            "verified_reverse_gap": quantiles([
                item["reverse_score_gap"] for item in verified_records
                if item["reverse_score_gap"] is not None
            ]),
            "interpretation_limit": "A FAIL means only that this frozen full-query matcher did not identify a safe exact 42-source exclusion set; it is not an infeasibility certificate.",
        },
        "exclusion_pool": {
            "parent_exclusion_count": len(recomputed_parent_exclusions),
            "new_exclusion_count": len(new_ids),
            "full_exclusion_count": len(full_exclusions),
            "full_exclusion_digest": full_exclusion_digest,
            "authorized_exclusion_digest": (
                full_exclusion_digest if state == "COMPLETED_GATE_PASS" else None
            ),
            "eligible_ots_scenes": eligible_ots,
            "minimum_required_eligible_scenes": MIN_ELIGIBLE_OTS,
        },
        "gates": gates,
        "terminal": {
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "limitations": [
            "The operation qualifies source identity only and does not validate an outdoor measurement field.",
            "Haze4K GT and SOTS GT images are used only for source identity; no model outcome is evaluated.",
            "The frozen matcher can fail without proving that the provenance mapping is absent.",
            "No training, inference, checkpoint selection, confirmation outcome, canary, or locked-test metric is accessed.",
        ],
        "marker": "RESIDE_OTS_FULL_RETRIEVAL_COMPLETE",
    }
    output_file(context, "full_retrieval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    mapping_path = output_file(context, "full_query_mapping.csv")
    mapping_fields = [
        "haze4k_group_digest", "haze4k_representative_source",
        "haze4k_representative_id", "ots_scene_id", "candidate_hamming_distance",
        "correlation_score", "score_gap", "prequalified", "reverse_score_gap",
        "mutual_nearest", "verified",
    ]
    with mapping_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=mapping_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    negative_path = output_file(context, "negative_controls.csv")
    with negative_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(negative_records[0]))
        writer.writeheader()
        writer.writerows(negative_records)
    output_file(context, "ots_exclusion_ids.txt").write_text(
        "\n".join(full_exclusions) + "\n", encoding="utf-8",
    )
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "full_retrieval_summary.json",
            "queried_unmatched_groups": len(records),
            "verified_new_ots_mappings": len(verified_records),
            "eligible_ots_scenes": eligible_ots,
            "exclusion_digest": full_exclusion_digest,
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
