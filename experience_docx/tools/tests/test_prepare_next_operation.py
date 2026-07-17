"""Tests for deterministic next-operation amendment preparation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import prepare_next_operation as NEXT  # noqa: E402


def manifest():
    return {
        "schema_version": 4, "route_id": "route", "rules_commit": "a" * 40,
        "route_card_relpath": "experience_docx/experiment_cards/route.md",
        "operations": {"S0": {}},
    }


def closeout():
    return {
        "route_id": "route", "state": "COMPLETED_GATE_PASS",
        "decision": "S0_PASS", "authorizes": "A0",
    }


def request():
    return {
        "schema_version": 1,
        "operation_id": "A0",
        "operation": {
            "mode": "a0", "require_gpu": False, "output_id": "a0-r1",
            "closeout_filename": "a0_closeout.json",
            "allowed_terminal_tuples": [
                {"state": "COMPLETED_GATE_PASS", "decision": "A0_PASS", "authorizes": "A1"},
                NEXT.GENERIC_ENGINEERING_TERMINAL.copy(),
            ],
            "workspace_policy": "fresh_route", "output_policy": "new",
            "monitor_profile": "short", "heartbeat_timeout_seconds": 120,
            "min_free_gpu_mib": 0, "max_gpu_utilization_pct": 100,
        },
        "runtime_spec": {
            "entrypoint_relpath": "experience_docx/tools/a0.py",
            "asset_manifest_relpath": None,
            "timeout_seconds": 600, "expected_wall_seconds": 60,
            "total_units": 1, "evidence_role": "development_screening",
            "resume_policy": "none",
            "protected_data_permissions": {
                "allow_confirmation": False, "allow_canary": False,
                "allow_locked_test": False,
            },
            "environment": {},
            "evidence_files": [{
                "source_relpath": "workload/summary.json",
                "destination_filename": "a0_summary.json",
                "required": True, "max_bytes": 4096,
            }],
        },
    }


class NextOperationTests(unittest.TestCase):
    def test_authorized_closeout_builds_minimal_amendment(self):
        candidate, spec = NEXT.build_amendment(
            manifest(), closeout(), request(),
            "experience_docx/experiment_logs/route/s0_closeout.json",
        )
        self.assertEqual(["A0"], list(candidate["operations"]))
        operation = candidate["operations"]["A0"]
        self.assertEqual(NEXT.GENERIC_RUNNER_RELPATH, operation["runner_relpath"])
        self.assertEqual("S0_PASS", operation["prior_terminal_tuple"]["decision"])
        self.assertEqual("A0", spec["operation_id"])

    def test_closeout_must_authorize_exact_operation(self):
        value = closeout()
        value["authorizes"] = "OTHER"
        with self.assertRaises(NEXT.AmendmentError):
            NEXT.build_amendment(
                manifest(), value, request(),
                "experience_docx/experiment_logs/route/s0_closeout.json",
            )

    def test_scientific_identity_is_preserved(self):
        candidate, _ = NEXT.build_amendment(
            manifest(), closeout(), request(),
            "experience_docx/experiment_logs/route/s0_closeout.json",
        )
        self.assertEqual("a" * 40, candidate["rules_commit"])
        self.assertEqual(
            "experience_docx/experiment_cards/route.md",
            candidate["route_card_relpath"],
        )

    def test_apply_writes_manifest_and_spec_from_committed_closeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            manifest_path = repo / "experience_docx/route_operations.json"
            closeout_path = repo / "experience_docx/experiment_logs/route/s0_closeout.json"
            manifest_path.parent.mkdir(parents=True)
            closeout_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps(manifest()), encoding="utf-8")
            closeout_path.write_text(json.dumps(closeout()), encoding="utf-8")
            subprocess.run(["git", "add", "experience_docx"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "route"], cwd=repo, check=True)
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request()), encoding="utf-8")
            report = NEXT.prepare(
                repo, request_path,
                "experience_docx/experiment_logs/route/s0_closeout.json", apply=True,
            )
            self.assertEqual("NEXT_OPERATION_APPLIED", report["status"])
            value = json.loads(manifest_path.read_text())
            self.assertEqual(["A0"], list(value["operations"]))
            self.assertTrue((repo / report["runtime_spec_relpath"]).is_file())


if __name__ == "__main__":
    unittest.main()
