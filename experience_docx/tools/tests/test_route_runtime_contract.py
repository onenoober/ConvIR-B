"""Unit tests for the declarative route runtime contract."""

import copy
import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import route_runtime_contract as CONTRACT  # noqa: E402


def operation():
    return {
        "runner_relpath": CONTRACT.GENERIC_RUNNER_RELPATH,
        "mode": "s0", "require_gpu": False, "output_id": "route-r1",
        "closeout_filename": "route_closeout.json", "prior_closeout_relpath": None,
        "prior_terminal_tuple": None, "allowed_terminal_tuples": [
            {"state": "COMPLETED_GATE_PASS", "decision": "PASS", "authorizes": "NEXT"},
            {"state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE"},
        ],
        "workspace_policy": "fresh_route", "output_policy": "new",
        "monitor_profile": "short", "heartbeat_timeout_seconds": 120,
        "min_free_gpu_mib": 0, "max_gpu_utilization_pct": 100,
    }


def manifest():
    return {"route_id": "route", "operations": {"S0": operation()}}


def spec():
    return {
        "schema_version": 1, "route_id": "route", "operation_id": "S0",
        "entrypoint_relpath": "experience_docx/tools/route_program.py",
        "asset_manifest_relpath": None, "timeout_seconds": 600,
        "expected_wall_seconds": 60, "total_units": 1,
        "evidence_role": "engineering_debug", "resume_policy": "none",
        "protected_data_permissions": {
            "allow_confirmation": False, "allow_canary": False, "allow_locked_test": False,
        },
        "environment": {},
        "evidence_files": [{
            "source_relpath": "workload/summary.json",
            "destination_filename": "summary.json", "required": True,
            "max_bytes": 4096,
        }],
    }


class RuntimeContractTests(unittest.TestCase):
    def test_valid_spec_passes(self):
        value = CONTRACT.validate_runtime_spec(spec(), manifest(), "S0")
        self.assertEqual("route_program.py", Path(value["entrypoint_relpath"]).name)

    def test_generic_runner_is_required(self):
        value = manifest()
        value["operations"]["S0"]["runner_relpath"] = "experience_docx/tools/run_custom.sh"
        with self.assertRaises(CONTRACT.ContractError):
            CONTRACT.validate_runtime_spec(spec(), value, "S0")

    def test_locked_test_requires_sealed_role(self):
        value = spec()
        value["protected_data_permissions"]["allow_locked_test"] = True
        with self.assertRaises(CONTRACT.ContractError):
            CONTRACT.validate_runtime_spec(value, manifest(), "S0")

    def test_output_policy_must_match_resume(self):
        value = spec()
        value["resume_policy"] = "exact_resume"
        with self.assertRaises(CONTRACT.ContractError):
            CONTRACT.validate_runtime_spec(value, manifest(), "S0")

    def test_complete_unit_recovery_uses_a_new_output(self):
        value = spec()
        value["resume_policy"] = "complete_units"
        value["asset_manifest_relpath"] = "experience_docx/route_assets/S0.json"
        self.assertEqual(
            "complete_units",
            CONTRACT.validate_runtime_spec(value, manifest(), "S0")["resume_policy"],
        )

    def test_evidence_path_cannot_escape_workload(self):
        value = spec()
        value["evidence_files"][0]["source_relpath"] = "../summary.json"
        with self.assertRaises(CONTRACT.ContractError):
            CONTRACT.validate_runtime_spec(value, manifest(), "S0")

    def test_asset_manifest_is_typed(self):
        normalized = CONTRACT.validate_runtime_spec(spec(), manifest(), "S0")
        assets = {
            "schema_version": 1, "route_id": "route", "operation_id": "S0",
            "assets": [{
                "id": "checkpoint", "kind": "file", "path": "/tmp/model.pkl",
                "sha256": "a" * 64, "access_role": "unrestricted",
                "contract_access": True,
            }],
        }
        self.assertEqual("checkpoint", CONTRACT.validate_asset_manifest(assets, normalized)["assets"][0]["id"])
        broken = copy.deepcopy(assets)
        broken["assets"][0]["sha256"] = "bad"
        with self.assertRaises(CONTRACT.ContractError):
            CONTRACT.validate_asset_manifest(broken, normalized)

    def test_asset_path_must_be_absolute_or_rooted(self):
        normalized = CONTRACT.validate_runtime_spec(spec(), manifest(), "S0")
        assets = {
            "schema_version": 1, "route_id": "route", "operation_id": "S0",
            "assets": [{
                "id": "data", "kind": "directory", "path": "relative/data",
                "access_role": "development_screening", "contract_access": False,
            }],
        }
        with self.assertRaises(CONTRACT.ContractError):
            CONTRACT.validate_asset_manifest(assets, normalized)
        assets["assets"][0]["path"] = "{RUN_ROOT}/frozen-data"
        self.assertEqual(
            "{RUN_ROOT}/frozen-data",
            CONTRACT.validate_asset_manifest(assets, normalized)["assets"][0]["path"],
        )

    def test_contract_cannot_receive_confirmation_asset(self):
        value = spec()
        value["evidence_role"] = "confirmation"
        value["protected_data_permissions"]["allow_confirmation"] = True
        normalized = CONTRACT.validate_runtime_spec(value, manifest(), "S0")
        assets = {
            "schema_version": 1, "route_id": "route", "operation_id": "S0",
            "assets": [{
                "id": "confirmation", "kind": "directory", "path": "/data/confirmation",
                "access_role": "confirmation", "contract_access": True,
            }],
        }
        with self.assertRaises(CONTRACT.ContractError):
            CONTRACT.validate_asset_manifest(assets, normalized)


if __name__ == "__main__":
    unittest.main()
