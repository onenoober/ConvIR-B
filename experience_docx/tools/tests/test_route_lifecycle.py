"""Focused tests for lifecycle result, asset, and evidence guards."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import route_lifecycle as LIFE  # noqa: E402
from test_route_runtime_contract import manifest, spec  # noqa: E402
import route_runtime_contract as CONTRACT  # noqa: E402


class LifecycleTests(unittest.TestCase):
    def normalized(self):
        value = manifest()
        return value, CONTRACT.validate_runtime_spec(spec(), value, "S0")

    def test_contract_result_requires_no_protected_data(self):
        _, runtime = self.normalized()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            value = {
                "schema_version": 1, "route_id": "route", "operation_id": "S0",
                "phase": "contract", "ok": True, "checks": {"fixture": True},
                "output_contract_checked": True, "finalizer_contract_checked": True,
                "confirmation_images_targets_outcomes_touched": False,
                "canary_touched": False, "locked_test_touched": False,
            }
            path.write_text(json.dumps(value))
            self.assertTrue(LIFE.validate_contract_result(path, runtime)["ok"])
            value["locked_test_touched"] = True
            path.write_text(json.dumps(value))
            with self.assertRaises(LIFE.LifecycleError):
                LIFE.validate_contract_result(path, runtime)

    def test_run_result_must_match_allowed_tuple(self):
        value, runtime = self.normalized()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            result = {
                "schema_version": 1, "route_id": "route", "operation_id": "S0",
                "phase": "run", "state": "COMPLETED_GATE_PASS", "decision": "PASS",
                "authorizes": "NEXT", "details": {},
                "confirmation_images_targets_outcomes_touched": False,
                "canary_touched": False, "locked_test_touched": False,
            }
            path.write_text(json.dumps(result))
            self.assertEqual("PASS", LIFE.validate_run_result(path, runtime, value["operations"]["S0"])["decision"])
            result["authorizes"] = "OTHER"
            path.write_text(json.dumps(result))
            with self.assertRaises(LIFE.LifecycleError):
                LIFE.validate_run_result(path, runtime, value["operations"]["S0"])

    def test_evidence_copy_rejects_existing_destination(self):
        _, runtime = self.normalized()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "output/workload"
            source.mkdir(parents=True)
            source.joinpath("summary.json").write_text("{}")
            evidence = root / "evidence"
            evidence.mkdir()
            evidence.joinpath("summary.json").write_text("old")
            with self.assertRaises(LIFE.LifecycleError):
                LIFE.copy_evidence(runtime, root / "output", evidence)

    def test_evidence_preflight_does_not_publish_partial_files(self):
        _, runtime = self.normalized()
        runtime["evidence_files"].append({
            "source_relpath": "workload/missing.json",
            "destination_filename": "missing.json",
            "required": True,
            "max_bytes": 4096,
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "output/workload"
            source.mkdir(parents=True)
            source.joinpath("summary.json").write_text("{}")
            evidence = root / "evidence"
            evidence.mkdir()
            with self.assertRaises(LIFE.LifecycleError):
                LIFE.copy_evidence(runtime, root / "output", evidence)
            self.assertEqual([], list(evidence.iterdir()))

    def test_contract_context_is_cpu_even_for_gpu_operation(self):
        _, runtime = self.normalized()
        env = {
            "RUN_ID": "route-r1", "EXPECTED_ROUTE_COMMIT": "a" * 40,
            "RUNNER_SHA256": "b" * 64, "REMOTE_REPO": "/repo",
            "RUN_ROOT": "/runs", "GPU": "0",
        }
        contract = LIFE.context_value(
            phase="contract", env=env, spec=runtime, output=Path("/runs/route-r1"),
            status=Path("/runs/route-r1/status.txt"),
            heartbeat=Path("/runs/route-r1/heartbeat.json"), assets=[],
        )
        run = LIFE.context_value(
            phase="run", env=env, spec=runtime, output=Path("/runs/route-r1"),
            status=Path("/runs/route-r1/status.txt"),
            heartbeat=Path("/runs/route-r1/heartbeat.json"), assets=[],
        )
        self.assertEqual("cpu", contract["device"])
        self.assertEqual("cuda", run["device"])
        self.assertEqual("contract", Path(contract["phase_output_path"]).name)
        self.assertEqual("workload", Path(run["phase_output_path"]).name)

    def test_contract_context_filters_protected_assets(self):
        _, runtime = self.normalized()
        runtime["protected_data_permissions"]["allow_confirmation"] = True
        runtime["evidence_role"] = "confirmation"
        env = {
            "RUN_ID": "route-r1", "EXPECTED_ROUTE_COMMIT": "a" * 40,
            "RUNNER_SHA256": "b" * 64, "REMOTE_REPO": "/repo",
            "RUN_ROOT": "/runs", "GPU": "",
        }
        assets = [
            {"id": "fixture", "kind": "directory", "path": "/fixture",
             "access_role": "unrestricted", "contract_access": True},
            {"id": "confirmation", "kind": "directory", "path": "/confirmation",
             "access_role": "confirmation", "contract_access": False},
        ]
        contract = LIFE.context_value(
            phase="contract", env=env, spec=runtime, output=Path("/runs/route-r1"),
            status=Path("/runs/route-r1/status.txt"),
            heartbeat=Path("/runs/route-r1/heartbeat.json"), assets=assets,
        )
        self.assertEqual(["fixture"], [item["id"] for item in contract["assets"]])
        self.assertFalse(any(contract["protected_data_permissions"].values()))

    def test_existing_closeout_never_overwrites_local_copy(self):
        value, runtime = self.normalized()
        operation = value["operations"]["S0"]
        env = {
            "RUN_ID": "route-r1", "EXPECTED_ROUTE_COMMIT": "a" * 40,
            "RUNNER_SHA256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            control = output / "control"
            control.mkdir(parents=True)
            local = control / operation["closeout_filename"]
            local.write_text("old-local")
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / operation["closeout_filename"]).write_text("old-evidence")
            result = {"state": "FAILED_ENGINEERING", "decision": None,
                      "authorizes": "NONE", "details": {}}
            with self.assertRaises(LIFE.LifecycleError):
                LIFE.write_closeout(
                    env=env, spec=runtime, operation=operation, output=output,
                    evidence_root=evidence, result=result, evidence_sha256={},
                )
            self.assertEqual("old-local", local.read_text())

    def test_verified_asset_identity_survives_failure_closeout_path(self):
        value, runtime = self.normalized()
        operation = value["operations"]["S0"]
        env = {
            "ROUTE_ID": "route", "RUN_ID": "route-r1",
            "EXPECTED_ROUTE_COMMIT": "a" * 40, "RUNNER_SHA256": "b" * 64,
        }
        assets = [{
            "id": "metadata", "kind": "file", "path": "/cloud/metadata.csv",
            "sha256": "c" * 64, "access_role": "development_screening",
            "contract_access": False,
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            (output / "control").mkdir(parents=True)
            recovered = LIFE.verified_asset_identities(assets)
            self.assertEqual("metadata", recovered[0]["id"])
            self.assertNotIn("path", recovered[0])
            evidence = root / "evidence"
            evidence.mkdir()
            result = {
                "state": "FAILED_ENGINEERING", "decision": None,
                "authorizes": "NONE", "details": {"error_type": "Timeout"},
            }
            closeout_path = LIFE.write_closeout(
                env=env, spec=runtime, operation=operation, output=output,
                evidence_root=evidence, result=result, evidence_sha256={},
                verified_assets=recovered, failure_phase="workload", returncode=1,
            )
            closeout = json.loads(closeout_path.read_text())
            self.assertEqual("metadata", closeout["verified_assets"][0]["id"])
            self.assertEqual("c" * 64, closeout["verified_assets"][0]["sha256"])

    def test_partial_asset_verification_retains_only_successful_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            first.write_text("{}")
            assets = {
                "assets": [
                    {
                        "id": "first", "kind": "file", "path": str(first),
                        "sha256": LIFE.sha256(first),
                        "access_role": "development_screening", "contract_access": False,
                    },
                    {
                        "id": "missing", "kind": "file", "path": str(root / "missing.json"),
                        "sha256": "d" * 64,
                        "access_role": "development_screening", "contract_access": False,
                    },
                ],
            }
            with self.assertRaises(LIFE.LifecycleError):
                LIFE.verify_assets(assets, repo=root, run_root=root, output=root / "output")
            self.assertEqual(["first"], [item["id"] for item in LIFE.VERIFIED_ASSETS])

    def test_program_log_tail_is_bounded_and_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.log"
            path.write_text(
                "discarded\n" * 5000
                + "api_key=do-not-expose /sda/home/private/route.py\n"
                + "ContractError: schema-2 contract requires complete engineering evidence\n",
                encoding="utf-8",
            )
            tail = LIFE.diagnostic_log_tail(path, 512)
            self.assertLessEqual(len(tail), 512)
            self.assertNotIn("do-not-expose", tail)
            self.assertNotIn("/sda/home/private", tail)
            self.assertIn("schema-2 contract requires complete engineering evidence", tail)

if __name__ == "__main__":
    unittest.main()
