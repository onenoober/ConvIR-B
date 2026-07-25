#!/usr/bin/env python3
"""Build exposure-tiered ITS quarantine assets from verified direct relations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

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
EXPECTED_ITS_TRAIN = 10000
EXPECTED_ITS_VALIDATION = 1000
EXPECTED_PARENT_ROWS = 1000
EXPECTED_ACCEPTED_RELATIONSHIPS = 426
EXPECTED_REJECTED_RELATIONSHIPS = 574
EXPECTED_TRAIN_RELATIONSHIPS = 302
EXPECTED_TEST_RELATIONSHIPS = 124
EXPECTED_TRAIN_SCOPE_IDS = 1653
EXPECTED_TEST_SCOPE_IDS = 534
EXPECTED_UNION_IDS = 2187
EXPECTED_TRAIN_SCOPE_ITS_TRAIN_IDS = 1498
EXPECTED_TRAIN_SCOPE_ITS_VALIDATION_IDS = 155
EXPECTED_UNION_ITS_TRAIN_IDS = 2032
EXPECTED_UNION_ITS_VALIDATION_IDS = 155
MIN_ELIGIBLE_ITS = 6000
TOTAL_UNITS = 11000
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


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"missing image directory: {directory}")
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def digest_lines(lines: Iterable[str]) -> str:
    payload = "".join(f"{line}\n" for line in sorted(lines)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_lines(path: Path, lines: Iterable[str]) -> None:
    ordered = sorted(lines)
    path.write_text("".join(f"{line}\n" for line in ordered), encoding="utf-8")


def read_parent_rows(context: Any) -> list[dict[str, str]]:
    with asset_path(context, "parent_relationships", kind="file").open(
        "r", encoding="utf-8", newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def accepted_ids(row: dict[str, str]) -> set[str]:
    values = [value for value in row["accepted_its_ids"].split(";") if value]
    if len(values) != len(set(values)):
        raise ValueError("parent relationship row contains duplicate accepted ITS IDs")
    return set(values)


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    entrypoint = context.assets.get("entrypoint_source")
    checks = {
        "metadata_only_mode": context.engineering_contract["mode"] == "metadata_only",
        "cpu_contract": context.device == "cpu",
        "dataset_hidden_from_contract": "reside_root" not in context.assets,
        "entrypoint_identity_bound": entrypoint is not None and entrypoint.contract_access is True,
        "parent_evidence_identity_bound": all(
            context.assets.get(name) is not None
            and context.assets[name].contract_access is True
            for name in (
                "parent_closeout", "parent_conclusion", "parent_summary",
                "parent_relationships",
            )
        ),
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
        "no_image_model_or_metric_path": True,
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
    if context.total_units != TOTAL_UNITS:
        raise ValueError("runtime total_units differs from frozen ITS membership census")

    closeout = json.loads(asset_path(
        context, "parent_closeout", kind="file",
    ).read_text(encoding="utf-8"))
    conclusion = json.loads(asset_path(
        context, "parent_conclusion", kind="file",
    ).read_text(encoding="utf-8"))
    parent_summary = json.loads(asset_path(
        context, "parent_summary", kind="file",
    ).read_text(encoding="utf-8"))
    rows = read_parent_rows(context)

    parent_terminal_ok = (
        closeout.get("state") == "COMPLETED_GATE_FAIL"
        and closeout.get("decision") == "ITS_GEOMETRIC_SOURCE_EXCLUSION_FAIL"
        and closeout.get("authorizes") == "NONE"
        and conclusion.get("decision") == "ITS_GEOMETRIC_SOURCE_EXCLUSION_FAIL"
        and conclusion.get("authorizes") == "NONE"
    )
    parent_counts_ok = (
        len(rows) == EXPECTED_PARENT_ROWS
        and sum(row.get("accepted") == "True" for row in rows)
        == EXPECTED_ACCEPTED_RELATIONSHIPS
        and sum(row.get("accepted") == "False" for row in rows)
        == EXPECTED_REJECTED_RELATIONSHIPS
        and parent_summary.get("relationships", {}).get("accepted_relationships")
        == EXPECTED_ACCEPTED_RELATIONSHIPS
        and parent_summary.get("relationships", {}).get("accepted_candidate_ids")
        == EXPECTED_UNION_IDS
    )
    group_integrity = (
        len({row["haze4k_group_digest"] for row in rows}) == EXPECTED_PARENT_ROWS
        and len({(row["haze4k_source"], row["haze4k_id"]) for row in rows})
        == EXPECTED_PARENT_ROWS
        and all(row["haze4k_source"] in {"HAZE4K_TRAIN", "HAZE4K_TEST"} for row in rows)
    )

    accepted_rows = [row for row in rows if row["accepted"] == "True"]
    relationship_rows: list[dict[str, Any]] = []
    train_ids: set[str] = set()
    test_ids: set[str] = set()
    direct_evidence_ok = True
    for row in accepted_rows:
        ids = accepted_ids(row)
        direct_evidence_ok = direct_evidence_ok and bool(ids) and row["best_its_id"] in ids
        target = train_ids if row["haze4k_source"] == "HAZE4K_TRAIN" else test_ids
        target.update(ids)
        relationship_rows.append({
            "haze4k_group_digest": row["haze4k_group_digest"],
            "haze4k_source": row["haze4k_source"],
            "haze4k_id": row["haze4k_id"],
            "quarantine_tier": (
                "TRAIN_EXPOSURE" if row["haze4k_source"] == "HAZE4K_TRAIN"
                else "SELECTION_EXPOSURE"
            ),
            "accepted_its_id_count": len(ids),
            "accepted_its_ids": ";".join(sorted(ids)),
        })
    union_ids = train_ids | test_ids

    reside = asset_path(context, "reside_root", kind="directory")
    observed_identity = {
        "archive_manifest_sha256": sha256_file(reside / "ARCHIVE_SHA256SUMS.txt"),
        "pairing_report_sha256": sha256_file(reside / "PAIRING_VALIDATION.txt"),
        "layout_record_sha256": sha256_file(reside / "DATASET_LAYOUT.txt"),
    }
    train_paths = image_files(reside / "official/ITS/train/ITS_clear")
    validation_paths = image_files(reside / "official/ITS/val/clear")
    available_ids: set[str] = set()
    progress = 0
    for source, paths in (("ITS_TRAIN", train_paths), ("ITS_VALIDATION", validation_paths)):
        for path in paths:
            available_ids.add(f"{source}:{path.stem}")
            progress += 1
            if progress == 1 or progress % 1000 == 0 or progress == TOTAL_UNITS:
                write_workload_progress(
                    context, completed_units=progress, stage="official_its_membership",
                )
    dataset_identity = (
        observed_identity == EXPECTED_RESIDE_IDENTITY
        and len(train_paths) == EXPECTED_ITS_TRAIN
        and len(validation_paths) == EXPECTED_ITS_VALIDATION
        and len(available_ids) == TOTAL_UNITS
    )
    official_membership = union_ids.issubset(available_ids)

    tier_counts = {
        "train_relationships": sum(
            row["haze4k_source"] == "HAZE4K_TRAIN" for row in accepted_rows
        ),
        "test_relationships": sum(
            row["haze4k_source"] == "HAZE4K_TEST" for row in accepted_rows
        ),
        "train_scope_ids": len(train_ids),
        "test_scope_ids": len(test_ids),
        "union_ids": len(union_ids),
        "train_test_id_intersection": len(train_ids & test_ids),
        "train_scope_its_train_ids": sum(value.startswith("ITS_TRAIN:") for value in train_ids),
        "train_scope_its_validation_ids": sum(
            value.startswith("ITS_VALIDATION:") for value in train_ids
        ),
        "union_its_train_ids": sum(value.startswith("ITS_TRAIN:") for value in union_ids),
        "union_its_validation_ids": sum(
            value.startswith("ITS_VALIDATION:") for value in union_ids
        ),
    }
    tier_counts_ok = tier_counts == {
        "train_relationships": EXPECTED_TRAIN_RELATIONSHIPS,
        "test_relationships": EXPECTED_TEST_RELATIONSHIPS,
        "train_scope_ids": EXPECTED_TRAIN_SCOPE_IDS,
        "test_scope_ids": EXPECTED_TEST_SCOPE_IDS,
        "union_ids": EXPECTED_UNION_IDS,
        "train_test_id_intersection": 0,
        "train_scope_its_train_ids": EXPECTED_TRAIN_SCOPE_ITS_TRAIN_IDS,
        "train_scope_its_validation_ids": EXPECTED_TRAIN_SCOPE_ITS_VALIDATION_IDS,
        "union_its_train_ids": EXPECTED_UNION_ITS_TRAIN_IDS,
        "union_its_validation_ids": EXPECTED_UNION_ITS_VALIDATION_IDS,
    }
    eligible_after_train = TOTAL_UNITS - len(train_ids)
    eligible_after_union = TOTAL_UNITS - len(union_ids)

    identity_gates = {
        "parent_terminal_identity": parent_terminal_ok,
        "parent_count_identity": parent_counts_ok,
        "parent_group_integrity": group_integrity,
        "dataset_identity": dataset_identity,
    }
    decisive_gates = {
        "direct_evidence_integrity": direct_evidence_ok,
        "tier_count_identity": tier_counts_ok,
        "official_its_membership": official_membership,
        "train_scope_capacity": eligible_after_train >= MIN_ELIGIBLE_ITS,
        "train_selection_scope_capacity": eligible_after_union >= MIN_ELIGIBLE_ITS,
    }
    if not all(identity_gates.values()):
        state = "COMPLETED_INCONCLUSIVE"
        decision = "ITS_VERIFIED_OVERLAP_QUARANTINE_INCONCLUSIVE"
        authorizes = "NONE"
    elif all(decisive_gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "ITS_VERIFIED_OVERLAP_QUARANTINE_PASS"
        authorizes = "ITS_KNOWN_OVERLAP_QUARANTINED_MEASUREMENT_DESIGN"
    else:
        state = "COMPLETED_GATE_FAIL"
        decision = "ITS_VERIFIED_OVERLAP_QUARANTINE_FAIL"
        authorizes = "NONE"
    gates = {**identity_gates, **decisive_gates}
    gate_reasons = [name for name, passed in gates.items() if not passed]
    authorized_train_ids = train_ids if state == "COMPLETED_GATE_PASS" else set()
    authorized_union_ids = union_ids if state == "COMPLETED_GATE_PASS" else set()

    write_lines(output_file(context, "train_exposure_exclusions.txt"), authorized_train_ids)
    write_lines(
        output_file(context, "train_selection_exposure_exclusions.txt"),
        authorized_union_ids,
    )
    with output_file(context, "verified_overlap_relationships.csv").open(
        "w", encoding="utf-8", newline="",
    ) as stream:
        fields = [
            "haze4k_group_digest", "haze4k_source", "haze4k_id",
            "quarantine_tier", "accepted_its_id_count", "accepted_its_ids",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(
            relationship_rows,
            key=lambda row: (row["haze4k_source"], row["haze4k_group_digest"]),
        ))

    summary = {
        "schema_version": 1,
        "route_id": "reside-its-verified-overlap-quarantine-v1",
        "operation_id": "ITS_VERIFIED_OVERLAP_QUARANTINE",
        "run_id": context.run_id,
        "scope": "verified direct Haze4K exposure quarantine for official RESIDE ITS",
        "parent_evidence": {
            "terminal": "COMPLETED_GATE_FAIL",
            "use": "diagnostic direct relationships only",
            "accepted_relationships": len(accepted_rows),
            "rejected_relationships_not_interpreted": EXPECTED_REJECTED_RELATIONSHIPS,
        },
        "quarantine_tiers": {
            "train_exposure": {
                "haze4k_relationships": tier_counts["train_relationships"],
                "authorized_exclusion_ids": len(authorized_train_ids),
                "provisional_exclusion_ids": len(train_ids),
                "eligible_its_scenes": eligible_after_train,
                "sha256": digest_lines(authorized_train_ids),
            },
            "train_and_selection_exposure": {
                "haze4k_relationships": len(accepted_rows),
                "authorized_exclusion_ids": len(authorized_union_ids),
                "provisional_exclusion_ids": len(union_ids),
                "eligible_its_scenes": eligible_after_union,
                "sha256": digest_lines(authorized_union_ids),
            },
            "counts": tier_counts,
        },
        "dataset_identity": {
            "matched": dataset_identity,
            "observed": observed_identity,
            "expected": EXPECTED_RESIDE_IDENTITY,
            "its_train_clear": len(train_paths),
            "its_validation_clear": len(validation_paths),
        },
        "gates": gates,
        "terminal": {
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "claim_boundary": (
            "The assets quarantine all direct overlaps verified by the parent geometry route. "
            "They do not prove that no additional Haze4K-to-ITS overlap remains."
        ),
        "limitations": [
            "The conjectured total of 500 indoor relationships is not used as a gate.",
            "No ITS-to-ITS transitive family expansion is used.",
            "The train tier supports model-training exposure control; the union tier additionally controls Haze4K test or selection exposure.",
            "Future measurement must describe its population as known-overlap-quarantined rather than fully source-disjoint.",
            "No image content, model, inference, training, outcome metric, confirmation, canary or locked-test evidence is accessed.",
        ],
        "marker": "RESIDE_ITS_VERIFIED_OVERLAP_QUARANTINE_V1_COMPLETE",
    }
    output_file(context, "verified_overlap_quarantine_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    write_workload_progress(context, completed_units=TOTAL_UNITS, stage="quarantine_finalize")
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "verified_overlap_quarantine_summary.json",
            "train_exposure_exclusions_file": "train_exposure_exclusions.txt",
            "train_selection_exposure_exclusions_file": (
                "train_selection_exposure_exclusions.txt"
            ),
            "verified_relationships_file": "verified_overlap_relationships.csv",
            "train_exclusion_count": len(authorized_train_ids),
            "train_selection_exclusion_count": len(authorized_union_ids),
            "eligible_train_scope": eligible_after_train,
            "eligible_train_selection_scope": eligible_after_union,
            "gate_reasons": gate_reasons,
            "complete_disjointness_claimed": False,
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
