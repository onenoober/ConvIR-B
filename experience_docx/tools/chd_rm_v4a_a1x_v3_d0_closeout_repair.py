#!/usr/bin/env python3
"""Evidence-only typed closeout for the complete A1X-v3 D0 r1 result."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_accessibility_v3_20260716"
SOURCE_RUN = Path("/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v4a_a1x_accessibility_v3_20260716/a1x-v3-d0-r1/workload")
EXPECTED = {
    "v4a_a1r_bootstrap_summary.json": "6195c65c20230725ac006a599b1a6527211df1c0b0dd0274e2e7124aaaff64fe",
    "v4a_a1r_closeout.json": "d762eeba9303459d33b318041770cb594cb9be4f69638faed8b26af550d6e10a",
    "v4a_a1r_cell_operator_summary.csv": "b8590131f3e8866dee3ee3640917754683f2a4a46a3253293d953e4dc11ceb1c",
    "v4a_a1r_probe_state_manifest.json": "ce463271846fdae9a2cb2c6d6931a2bccc3fe9d801cad684c27d9c0f084b3bce",
    "v4a_a1r_fold_history.csv": "2d8dc2557574f888e96fe1c0d6aa2616d20f58d50369ebfe59eaba9caa29e7c1",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    for name, expected in EXPECTED.items():
        path = SOURCE_RUN / name
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"D0 source evidence mismatch: {name}")
    bootstrap = json.loads((SOURCE_RUN / "v4a_a1r_bootstrap_summary.json").read_text(encoding="utf-8"))
    source = json.loads((SOURCE_RUN / "v4a_a1r_closeout.json").read_text(encoding="utf-8"))
    if not (
        source["structural_valid"] is True
        and source["stage_name_count"] == 512
        and source["row_count"] == 5120
        and source["state_count"] == 20
        and source["bound_excess_max"] == 0.0
        and source["support_excess_max"] == 0.0
    ):
        raise RuntimeError("D0 complete-result structural contract failed")
    primary = bootstrap["cell_results"]["a1x_global"]
    gates = {
        "gain_lcb95_ge_0p020": float(primary["worst_operator_gain_vs_shrink_db_lcb95"]) >= 0.020,
        "retention_lcb95_ge_0p25": float(bootstrap["primary_oracle_retention_lcb95"]) >= 0.25,
        "true_minus_shuffle_lcb95_ge_0p005": float(bootstrap["primary_true_minus_shuffle_db_lcb95"]) >= 0.005,
        "global_minus_local_lcb95_gt_0": float(bootstrap["a1x_global_minus_local_db_lcb95"]) > 0.0,
        "structural_and_safety_guards": True,
    }
    passed = all(gates.values())
    if passed:
        raise RuntimeError("frozen D0 result unexpectedly passes; repair contract expected fixed fail")
    output = Path(os.environ["OUTPUT_PATH"])
    summary = {
        "schema_version": 1, "route_id": ROUTE_ID, "stage": "d0",
        "evidence_role": "development_screening", "source_run_id": "a1x-v3-d0-r1",
        "source_route_commit": "86c8c3827507f20b4dec0862ff25b4110e6e9630",
        "source_evidence_sha256": EXPECTED, "name_count": 512, "row_count": 5120,
        "state_count": 20, "gates": gates, "bootstrap": bootstrap,
        "decision": "A1X_V3_D0_GLOBAL_HEAD_CONTRACT_FAIL_STOP",
        "confirmation_images_targets_outcomes_touched": False,
        "canary_touched": False, "locked_test_touched": False,
    }
    summary_path = output / "a1x_v3_d0_summary.json"
    write_json(summary_path, summary)
    closeout = {
        "schema_version": 1, "route_id": ROUTE_ID, "run_id": os.environ["RUN_ID"],
        "route_commit": os.environ["EXPECTED_ROUTE_COMMIT"],
        "runner_sha256": os.environ["RUNNER_SHA256"], "stage": "d0",
        "state": "COMPLETED_GATE_FAIL",
        "decision": "A1X_V3_D0_GLOBAL_HEAD_CONTRACT_FAIL_STOP",
        "authorizes": "NONE", "evidence_role": "development_screening",
        "gate_type": "scientific_utility", "structural_valid": True,
        "source_run_id": "a1x-v3-d0-r1", "source_evidence_sha256": EXPECTED,
        "summary_sha256": digest(summary_path), "confirmation_images_targets_outcomes_touched": False,
        "canary_touched": False, "locked_test_touched": False,
        "candidate_selected": False, "formal_authorized": False,
    }
    write_json(output / "a1x_v3_d0_repaired_closeout.json", closeout)
    print(json.dumps(closeout, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
