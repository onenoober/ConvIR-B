"""Focused tests for lifecycle result, asset, and evidence guards."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import route_lifecycle as LIFE  # noqa: E402
import scientific_contract as SCIENCE  # noqa: E402
import capability_registry as REGISTRY  # noqa: E402
from test_route_runtime_contract import manifest, spec  # noqa: E402
from test_scientific_contract import contract as scientific_contract  # noqa: E402
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

    def test_contract_failure_exposes_only_bounded_failed_check_names(self):
        _, runtime = self.normalized()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            value = {
                "schema_version": 1, "route_id": "route", "operation_id": "S0",
                "phase": "contract", "ok": False,
                "checks": {"paths": True, "memory_bound": False},
                "output_contract_checked": True, "finalizer_contract_checked": True,
                "confirmation_images_targets_outcomes_touched": False,
                "canary_touched": False, "locked_test_touched": False,
            }
            path.write_text(json.dumps(value))
            with self.assertRaises(LIFE.LifecycleError) as raised:
                LIFE.validate_contract_result(path, runtime)
            self.assertEqual(
                {"failed_contract_checks": ["memory_bound"]},
                raised.exception.control_diagnostic,
            )
            self.assertEqual(
                {"failed_contract_checks": ["memory_bound"]},
                LIFE.safe_control_diagnostic(raised.exception.control_diagnostic),
            )

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

    def test_schema2_lifecycle_derives_fail_from_gate_outcomes(self):
        _, runtime = self.normalized()
        scientific = SCIENCE.validate_scientific_contract_v2(
            scientific_contract(), "route", "S0",
        )
        operation = {
            "allowed_terminal_tuples": [
                *SCIENCE.scientific_terminal_tuples(scientific),
                {"state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({
                "schema_version": 2, "route_id": "route", "operation_id": "S0",
                "phase": "run",
                "gate_outcomes": {
                    "materiality": "unfavorable", "precision": "unmet",
                },
                "details": {},
                "confirmation_images_targets_outcomes_touched": False,
                "canary_touched": False, "locked_test_touched": False,
            }))
            result = LIFE.validate_run_result(
                path, runtime, operation, scientific,
            )
            self.assertEqual("COMPLETED_GATE_FAIL", result["state"])
            self.assertEqual("bad_side_fails", result["decision_rule_id"])

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

    def test_operator_signal_requires_a_receipt_bound_request(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "operator_cancel_request.json"
            prior = LIFE.OPERATOR_CANCEL_REQUEST_PATH
            LIFE.OPERATOR_CANCEL_REQUEST_PATH = request
            try:
                with self.assertRaises(LIFE.LifecycleError):
                    LIFE.operator_cancel_signal(__import__("signal").SIGTERM, None)
                request.write_text("{}")
                with self.assertRaises(LIFE.OperatorCancelled):
                    LIFE.operator_cancel_signal(__import__("signal").SIGTERM, None)
            finally:
                LIFE.OPERATOR_CANCEL_REQUEST_PATH = prior

    def test_cancellation_progress_is_typed_and_result_blind(self):
        with tempfile.TemporaryDirectory() as directory:
            status = Path(directory) / "status.txt"
            status.write_text("\n".join((
                '{"R3_PROGRESS":{"stage":"extract","completed_units":7,'
                '"total_units":10,"metric":99.9,"sample_id":"secret"}}',
                '{"message":"untyped","completed_units":999,"total_units":999}',
            )))
            self.assertEqual(
                {"completed_units": 7, "total_units": 10, "stage": "extract"},
                LIFE.cancellation_progress(status),
            )

    def test_lifecycle_writes_non_scientific_operator_cancellation_closeout(self):
        value, runtime = self.normalized()
        operation = value["operations"]["S0"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            output = root / "runs/route/route-r1"
            control = output / "control"
            control.mkdir(parents=True)
            (repo / "experience_docx").mkdir(parents=True)
            (repo / "experience_docx/route_operations.json").write_text(
                json.dumps(value)
            )
            runtime_path = repo / CONTRACT.RUNTIME_SPEC_DIRECTORY / "S0.json"
            runtime_path.parent.mkdir(parents=True)
            runtime_path.write_text("{}")
            env = {
                "RUN_ID": "route-r1", "OUTPUT_ID": "route-r1",
                "EXPECTED_ROUTE_COMMIT": "a" * 40,
                "RUNNER_SHA256": "b" * 64, "MODE": "s0",
                "REMOTE_REPO": str(repo), "RUN_ROOT": str(root / "runs/route"),
                "OUTPUT_PATH": str(output), "GPU": "",
            }
            (control / "lifecycle_identity.json").write_text(
                json.dumps(LIFE.lifecycle_identity(env, runtime))
            )
            request_id = "1" * 32
            (control / "operator_cancel_request.json").write_text(json.dumps({
                "schema_version": 1, "request_id": request_id,
                "route_id": runtime["route_id"], "run_id": env["RUN_ID"],
                "route_commit": env["EXPECTED_ROUTE_COMMIT"],
                "runner_sha256": env["RUNNER_SHA256"],
                "requested_at_unix": 123, "action": "cancel",
            }))
            (output / "status.txt").write_text(
                '{"phase":"workload","event":"workload_progress",'
                '"completed_units":4,"total_units":10}\n'
            )
            prior_started = LIFE.WORKLOAD_STARTED
            LIFE.WORKLOAD_STARTED = True
            try:
                with (
                    patch.object(LIFE, "require_environment", return_value=env),
                    patch.object(
                        LIFE, "validate_lifecycle_paths",
                        return_value=(repo, root / "runs/route", output),
                    ),
                    patch.object(LIFE, "infer_operation", return_value=("S0", operation)),
                    patch.object(LIFE, "validate_runtime_spec", return_value=runtime),
                ):
                    LIFE.finalize_operator_cancellation(
                        LIFE.OperatorCancelled(__import__("signal").SIGTERM)
                    )
            finally:
                LIFE.WORKLOAD_STARTED = prior_started
            closeout = json.loads((
                repo / "experience_docx/experiment_logs" /
                runtime["route_id"] / operation["closeout_filename"]
            ).read_text())
            self.assertEqual("CANCELLED_BY_OPERATOR", closeout["state"])
            self.assertIsNone(closeout["decision"])
            self.assertEqual("NONE", closeout["authorizes"])
            self.assertEqual(4, closeout["details"]["completed_units"])
            self.assertFalse(
                closeout["details"]["scientific_result_interpretable"]
            )

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

    def test_exact_capability_registry_match_authorizes_engineering_reuse_only(self):
        identity = {
            "source_commit": "a" * 40,
            "code_path_sha256": "b" * 64,
            "checkpoint_sha256": "c" * 64,
            "runtime_environment_sha256": "d" * 64,
            "device_class": "cuda_sm89",
            "input_contract_sha256": "e" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            evidence = repo / "experience_docx/experiment_logs/q/evidence.json"
            evidence.parent.mkdir(parents=True)
            evidence.write_text("{}\n", encoding="utf-8")
            record = {
                "schema_version": 1, "qualification_id": "qualification_1",
                "identity": identity,
                "identity_sha256": REGISTRY.identity_digest(identity),
                "status": "PASSED_ENGINEERING",
                "contract_mode": "gpu_synthetic_no_data",
                "evidence_relpath": str(evidence.relative_to(repo)),
                "evidence_sha256": LIFE.sha256(evidence),
                "scientific_authorization": "NONE",
                "protected_data_touched": False,
            }
            registry = repo / REGISTRY.REGISTRY_RELPATH
            registry.parent.mkdir(parents=True, exist_ok=True)
            registry.write_text(json.dumps(record) + "\n", encoding="utf-8")
            result = LIFE.resolve_capability_reuse(
                repo, {
                    "schema_version": 2,
                    "reuse_identity": identity,
                },
            )
            self.assertTrue(result["engineering_reuse_authorized"])
            self.assertEqual("NONE", result["scientific_authorization"])

if __name__ == "__main__":
    unittest.main()
