"""Unit tests for route context and atomic program results."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import route_program_api as API  # noqa: E402


def context(root: Path, phase: str):
    output = root / "runs/r1"
    phase_directory = "contract" if phase == "contract" else "workload"
    return {
        "schema_version": 1, "phase": phase, "route_id": "route",
        "operation_id": "S0", "run_id": "r1", "route_commit": "a" * 40,
        "runner_sha256": "b" * 64,
        "entrypoint_relpath": "experience_docx/tools/program.py",
        "remote_repo": str(root / "repo"), "run_root": str(root / "runs"),
        "output_path": str(output), "phase_output_path": str(output / phase_directory),
        "result_path": str(output / phase_directory / f"{phase}_result.json"),
        "status_path": str(output / "status.txt"),
        "heartbeat_path": str(output / "heartbeat.json"), "device": "cpu",
        "total_units": 1, "evidence_role": "engineering_debug",
        "resume_policy": "none", "assets": [],
        "protected_data_permissions": {
            "allow_confirmation": False, "allow_canary": False, "allow_locked_test": False,
        },
    }


class RouteProgramApiTests(unittest.TestCase):
    def load(self, root: Path, phase: str):
        path = root / "runs/r1/control" / f"{phase}_context.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(context(root, phase)), encoding="utf-8")
        return API.load_context(path, phase)

    def test_contract_result_is_atomic_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = self.load(root, "contract")
            API.prepare_phase_output(value)
            API.write_contract_result(value, checks={"finalizer": True, "paths": True})
            result = json.loads(value.result_path.read_text())
            self.assertTrue(result["ok"])
            self.assertFalse(result["locked_test_touched"])

    def test_run_result_uses_context_path_without_positional_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = self.load(root, "run")
            API.prepare_phase_output(value)
            API.write_run_result(
                value, state="COMPLETED_GATE_PASS", decision="PASS", authorizes="NEXT",
            )
            self.assertEqual("PASS", json.loads(value.result_path.read_text())["decision"])

    def test_result_path_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = context(root, "run")
            value["result_path"] = str(root / "outside.json")
            path = root / "runs/r1/control/run_context.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(API.ContractError):
                API.load_context(path, "run")

    def test_atomic_json_is_write_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            API.atomic_json(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                API.atomic_json(path, {"value": 2})
            self.assertEqual(1, json.loads(path.read_text())["value"])

    def test_verified_asset_is_available_by_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = context(root, "run")
            value["assets"] = [{
                "id": "checkpoint", "kind": "file", "path": "/assets/model.pkl",
                "access_role": "unrestricted", "contract_access": True,
                "sha256": "c" * 64,
            }]
            path = root / "runs/r1/control/run_context.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(value), encoding="utf-8")
            loaded = API.load_context(path, "run")
            self.assertEqual(
                Path("/assets/model.pkl"),
                API.asset_path(loaded, "checkpoint", kind="file"),
            )

    def test_run_result_rejects_oversized_details(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = self.load(root, "run")
            API.prepare_phase_output(value)
            with self.assertRaises(API.ContractError):
                API.write_run_result(
                    value, state="COMPLETED_GATE_PASS", decision="PASS",
                    authorizes="NEXT", details={"payload": "x" * API.MAX_RESULT_BYTES},
                )

    def test_workload_progress_uses_one_generic_status_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = self.load(root, "run")
            API.write_workload_progress(
                value, completed_units=1, stage="feature_extract",
            )
            progress = json.loads(value.status_path.read_text(encoding="utf-8"))
            self.assertEqual("workload", progress["phase"])
            self.assertEqual("workload_progress", progress["event"])
            self.assertEqual(1, progress["completed_units"])
            self.assertEqual(value.total_units, progress["total_units"])

    def test_workload_progress_rejects_contract_phase_and_out_of_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self.load(root, "contract")
            with self.assertRaises(API.ContractError):
                API.write_workload_progress(
                    contract, completed_units=1, stage="invalid",
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.load(root, "run")
            with self.assertRaises(API.ContractError):
                API.write_workload_progress(
                    run, completed_units=2, stage="invalid",
                )

    def test_contract_progress_is_control_only_and_phase_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self.load(root, "contract")
            API.write_contract_progress(
                contract, completed_iterations=4, total_iterations=8,
                stage="synthetic_probe",
            )
            progress = json.loads(contract.status_path.read_text(encoding="utf-8"))
            self.assertEqual("contract", progress["phase"])
            self.assertEqual("contract_progress", progress["event"])
            self.assertEqual(4, progress["completed_iterations"])
            self.assertEqual(8, progress["total_iterations"])
            self.assertEqual(
                {"phase", "event", "stage", "completed_iterations", "total_iterations"},
                set(progress),
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.load(root, "run")
            with self.assertRaises(API.ContractError):
                API.write_contract_progress(
                    run, completed_iterations=1, total_iterations=1, stage="invalid",
                )


if __name__ == "__main__":
    unittest.main()
