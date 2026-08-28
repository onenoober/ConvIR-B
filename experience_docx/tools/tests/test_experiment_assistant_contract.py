"""Tests for the compact experiment-assistant contract and attempt model."""

import copy
import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import experiment_assistant_contract as ASSISTANT  # noqa: E402


ALL_CAPABILITIES = {
    "content_addressed_source_snapshot", "lifecycle", "automatic_result_archive",
    "experiment_record_read", "explicit_protected_data_access",
    "declared_precision_gate",
}


def contract():
    return {
        "schema_version": 1,
        "experiment_id": "dehaze-ablation-001",
        "objective": "Measure whether the current dehazing change improves validation PSNR.",
        "entrypoint": {
            "relpath": "train.py",
            "argv": ["--config", "configs/ablation.yaml"],
        },
        "datasets": [
            {"id": "reside_its_train", "role": "training"},
            {"id": "reside_sots_val", "role": "validation"},
        ],
        "budget": {
            "max_wall_seconds": 7200,
            "parameters": {"epochs": 30, "seeds": [7, 19]},
        },
        "evaluation": {
            "primary_metric": {
                "id": "psnr_db", "direction": "higher", "threshold": 30.0,
            },
            "result_files": ["results/summary.json"],
            "precision": {
                "target_half_width": 0.1,
                "available_independent_units": 100,
                "required_independent_units": 80,
            },
        },
    }


def snapshot(storage="cloud_full"):
    return {
        "sha256": "a" * 64,
        "storage": storage,
        "base_commit": "b" * 40,
        "diff_sha256": "c" * 64,
    }


def attempt(validated, number, state, *, automatic=False, storage="cloud_full"):
    result = None
    error = None
    if state.startswith("COMPLETED_"):
        result = {"primary_metric": {"id": "psnr_db", "value": 30.4}}
    if state == "FAILED_ENGINEERING":
        error = "Dataset loader could not resolve the configured path."
    return {
        "schema_version": 1,
        "experiment_id": validated["contract"]["experiment_id"],
        "attempt_number": number,
        "contract_sha256": validated["contract_sha256"],
        "source_snapshot": snapshot(storage),
        "budget": validated["contract"]["budget"],
        "state": state,
        "automatic_repair": automatic,
        "started_at": f"2026-08-28T0{number}:00:00Z",
        "ended_at": f"2026-08-28T0{number}:10:00Z",
        "error_summary": error,
        "result": result,
        "cloud_run_ref": f"runs/dehaze-ablation-001/attempt-{number}",
    }


class ExperimentAssistantContractTests(unittest.TestCase):
    def test_short_contract_has_no_research_governance_or_git_launch_fields(self):
        validated = ASSISTANT.validate_contract(contract())
        normalized = validated["contract"]
        forbidden = {
            "mode", "rules_commit", "branch", "route_commit", "program_contract",
            "route_family", "research_update_binding", "literature_basis",
            "decision_design", "capability_profile", "precision_certificate",
        }
        self.assertFalse(forbidden & set(normalized))
        self.assertEqual(64, len(validated["contract_sha256"]))
        self.assertTrue(ASSISTANT.assess_launch(validated, ALL_CAPABILITIES)["ok"])

    def test_nonessential_unknown_fields_and_missing_claim_strength_are_warnings(self):
        value = contract()
        value["display_color"] = "blue"
        value["evaluation"]["primary_metric"].pop("threshold")
        value["evaluation"].pop("precision")
        validated = ASSISTANT.validate_contract(value)
        assessment = ASSISTANT.assess_launch(validated, ALL_CAPABILITIES)
        self.assertTrue(assessment["ok"])
        self.assertTrue(any("display_color" in item for item in assessment["warnings"]))
        self.assertTrue(any("descriptive" in item for item in assessment["warnings"]))
        self.assertTrue(any("precision claim" in item for item in assessment["warnings"]))

    def test_only_required_capabilities_block_version_or_feature_drift(self):
        validated = ASSISTANT.validate_contract(contract())
        available = ALL_CAPABILITIES - {"declared_precision_gate"}
        assessment = ASSISTANT.assess_launch(validated, available)
        self.assertFalse(assessment["ok"])
        self.assertEqual(1, len(assessment["blockers"]))
        self.assertIn("declared_precision_gate", assessment["blockers"][0])

    def test_lifecycle_owned_environment_cannot_override_snapshot_or_output_identity(self):
        value = contract()
        value["entrypoint"]["environment"] = {"PYTHONPATH": "/tmp/unbound-source"}
        with self.assertRaisesRegex(ASSISTANT.ContractError, "lifecycle-owned"):
            ASSISTANT.validate_contract(value)

    def test_protected_data_defaults_to_deny_and_explicit_contract_allows(self):
        value = contract()
        value["datasets"].append({"id": "sealed_test", "role": "locked_test"})
        denied = ASSISTANT.assess_launch(
            ASSISTANT.validate_contract(value), ALL_CAPABILITIES,
        )
        self.assertFalse(denied["ok"])
        self.assertIn("explicit access", denied["blockers"][0])
        value["protected_access"] = ["locked_test"]
        allowed = ASSISTANT.assess_launch(
            ASSISTANT.validate_contract(value), ALL_CAPABILITIES,
        )
        self.assertTrue(allowed["ok"])

    def test_dataset_role_conflict_and_declared_precision_infeasibility_block(self):
        value = contract()
        value["datasets"].append({"id": "reside_its_train", "role": "test"})
        with self.assertRaisesRegex(ASSISTANT.ContractError, "cannot be used as both"):
            ASSISTANT.validate_contract(value)
        value = contract()
        value["evaluation"]["precision"]["available_independent_units"] = 20
        assessment = ASSISTANT.assess_launch(
            ASSISTANT.validate_contract(value), ALL_CAPABILITIES,
        )
        self.assertFalse(assessment["ok"])
        self.assertIn("precision is infeasible", assessment["blockers"][0])

    def test_budget_and_code_repairs_stay_in_same_experiment(self):
        original = ASSISTANT.validate_contract(contract())["contract"]
        revised_source = contract()
        revised_source["budget"]["parameters"]["epochs"] = 20
        revised_source["entrypoint"]["argv"].append("--repair-loader")
        revised = ASSISTANT.validate_contract(revised_source)["contract"]
        classification = ASSISTANT.classify_contract_revision(original, revised)
        self.assertTrue(classification["same_experiment"])
        self.assertEqual([], classification["new_experiment_reasons"])
        self.assertEqual(2, len(classification["warnings"]))

    def test_data_metric_threshold_or_precision_change_requires_new_experiment(self):
        original = ASSISTANT.validate_contract(contract())["contract"]
        revised_source = contract()
        revised_source["evaluation"]["primary_metric"]["threshold"] = 31.0
        revised = ASSISTANT.validate_contract(revised_source)["contract"]
        classification = ASSISTANT.classify_contract_revision(original, revised)
        self.assertFalse(classification["same_experiment"])
        self.assertEqual(["primary_metric"], classification["new_experiment_reasons"])

    def test_two_automatic_repairs_then_operator_confirmation(self):
        validated = ASSISTANT.validate_contract(contract())
        history = [
            attempt(validated, 1, "FAILED_ENGINEERING"),
            attempt(validated, 2, "FAILED_ENGINEERING", automatic=True),
            attempt(validated, 3, "FAILED_ENGINEERING", automatic=True),
        ]
        denied = ASSISTANT.authorize_attempt(history, automatic_repair=True)
        self.assertFalse(denied["ok"])
        self.assertEqual(2, denied["automatic_repairs_used"])
        allowed = ASSISTANT.authorize_attempt(
            history, automatic_repair=True, operator_confirmed=True,
        )
        self.assertTrue(allowed["ok"])

    def test_active_or_unknown_attempt_blocks_duplicate_launch(self):
        for state in ("PREPARED", "RUNNING", "UNKNOWN"):
            denied = ASSISTANT.authorize_attempt(
                [{"state": state, "automatic_repair": False}],
                automatic_repair=False,
            )
            self.assertFalse(denied["ok"])

    def test_pass_fail_and_inconclusive_archive_but_engineering_failure_does_not(self):
        validated = ASSISTANT.validate_contract(contract())
        for state in ("COMPLETED_PASS", "COMPLETED_FAIL", "COMPLETED_INCONCLUSIVE"):
            current = attempt(validated, 1, state)
            self.assertTrue(ASSISTANT.should_archive_attempt(current))
            archive = ASSISTANT.build_archive_record(
                validated, [current], recorded_at="2026-08-28T10:00:00Z",
            )
            self.assertEqual(state, archive["record"]["terminal"]["state"])
            self.assertEqual(64, len(archive["record_sha256"]))
        failed = attempt(validated, 1, "FAILED_ENGINEERING", storage="hash_only")
        self.assertFalse(ASSISTANT.should_archive_attempt(failed))
        with self.assertRaisesRegex(ASSISTANT.ContractError, "result-bearing"):
            ASSISTANT.build_archive_record(
                validated, [failed], recorded_at="2026-08-28T10:00:00Z",
            )

    def test_result_bearing_attempt_requires_complete_cloud_source_snapshot(self):
        validated = ASSISTANT.validate_contract(contract())
        broken = attempt(validated, 1, "COMPLETED_PASS", storage="hash_only")
        with self.assertRaisesRegex(ASSISTANT.ContractError, "recoverable cloud_full"):
            ASSISTANT.validate_attempt(broken)

    def test_public_tool_surface_hides_control_plane_identities(self):
        self.assertEqual(ASSISTANT.PUBLIC_TOOL_NAMES, tuple(ASSISTANT.PUBLIC_TOOL_SCHEMAS))
        serialized = repr(ASSISTANT.PUBLIC_TOOL_SCHEMAS)
        for forbidden in (
            "plan_token", "receipt", "catalog_sha256", "inventory_sha256",
            "snapshot_commit", "route_branch_commit", "schema_version",
        ):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
