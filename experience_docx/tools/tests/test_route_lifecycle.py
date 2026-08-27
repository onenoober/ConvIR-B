"""Focused tests for lifecycle result, asset, and evidence guards."""

import base64
import hashlib
import json
import os
import subprocess
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
from test_scientific_contract import (  # noqa: E402
    contract as scientific_contract,
    contract_v3,
    terminal_index_bytes,
)
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

    def test_schema3_lifecycle_derives_fail_from_gate_outcomes(self):
        _, runtime = self.normalized()
        terminal_index, terminal_sha256 = terminal_index_bytes()
        scientific = SCIENCE.validate_scientific_contract_v3(
            contract_v3(terminal_record_sha256=terminal_sha256), "route", "S0",
            expected_snapshot_commit="a" * 40,
            read_evidence_file=lambda _: terminal_index,
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

    def test_schema3_lifecycle_contract_uses_the_schema3_validator(self):
        scientific = LIFE.validate_lifecycle_scientific_contract(
            contract_v3(), "route", "S0",
        )
        self.assertEqual(3, scientific["schema_version"])
        self.assertIn("research_update_binding", scientific)
        self.assertTrue(LIFE.typed_scientific_contract(scientific))
        self.assertTrue(LIFE.typed_scientific_contract({"schema_version": 2}))
        self.assertFalse(LIFE.typed_scientific_contract({"schema_version": 1}))

    def test_completed_unit_ledger_is_required_only_for_resumable_runs(self):
        _, runtime = self.normalized()
        runtime["total_units"] = 3
        runtime["resume_policy"] = "none"
        self.assertFalse(LIFE.requires_completed_unit_ledger(runtime))
        runtime["resume_policy"] = "complete_units"
        self.assertTrue(LIFE.requires_completed_unit_ledger(runtime))
        runtime["total_units"] = 0
        self.assertFalse(LIFE.requires_completed_unit_ledger(runtime))

    def test_non_resumable_typed_lifecycle_does_not_read_ledger(self):
        manifest_value, runtime = self.normalized()
        operation = manifest_value["operations"]["S0"]
        manifest_value["scientific_contract_relpaths"] = {"S0": "science.json"}
        runtime["resume_policy"] = "none"
        runtime["total_units"] = 3
        route_commit = "a" * 40
        main_commit = "b" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            run_root = root / "runs/route"
            output = run_root / "route-r1"
            runner = repo / LIFE.GENERIC_RUNNER_RELPATH
            entrypoint = repo / runtime["entrypoint_relpath"]
            runner.parent.mkdir(parents=True)
            runner.write_text("runner\n", encoding="utf-8")
            entrypoint.parent.mkdir(parents=True, exist_ok=True)
            entrypoint.write_text("entrypoint\n", encoding="utf-8")
            env = {
                "EXPECTED_ROUTE_COMMIT": route_commit,
                "AUTHORITATIVE_MAIN_COMMIT": main_commit,
                "RUN_ID": "route-r1",
                "RUNNER_SHA256": LIFE.sha256(runner),
                "REMOTE_REPO": str(repo),
                "RUN_ROOT": str(run_root),
                "GPU": "",
            }

            def fake_git(_repo, *args):
                values = {
                    ("rev-parse", "HEAD"): route_commit,
                    ("rev-parse", "refs/convir-runtime/main"): main_commit,
                    ("status", "--porcelain"): "",
                }
                return values[args]

            with (
                patch.object(LIFE, "require_environment", return_value=env),
                patch.object(
                    LIFE, "validate_lifecycle_paths",
                    return_value=(repo, run_root, output),
                ),
                patch.object(LIFE, "load_json", side_effect=[
                    manifest_value, {}, {},
                ]),
                patch.object(
                    LIFE, "infer_operation", return_value=("S0", operation),
                ),
                patch.object(
                    LIFE, "validate_runtime_spec", return_value=runtime,
                ),
                patch.object(
                    LIFE, "validate_lifecycle_scientific_contract",
                    return_value={"schema_version": 2},
                ),
                patch.object(LIFE, "git", side_effect=fake_git),
                patch.object(LIFE, "verify_assets", return_value=[]),
                patch.object(
                    LIFE, "validate_receipt_contract_reuse", return_value=None,
                ),
                patch.object(LIFE, "start_sidecar"),
                patch.object(LIFE, "telemetry"),
                patch.object(LIFE, "run_program", return_value=0),
                patch.object(
                    LIFE, "validate_contract_result", return_value={"ok": True},
                ),
                patch.object(
                    LIFE, "validate_run_result",
                    return_value={
                        "state": "COMPLETED_GATE_PASS",
                        "decision": "PASS",
                        "authorizes": "NEXT",
                    },
                ),
                patch.object(LIFE, "load_completed_unit_ledger") as ledger,
                patch.object(LIFE, "copy_evidence", return_value={}),
                patch.object(
                    LIFE, "publish_raw_artifact_receipt",
                    return_value=("raw_artifact_receipt.json", "c" * 64),
                ),
                patch.object(LIFE, "write_closeout"),
            ):
                self.assertEqual(0, LIFE.lifecycle())
            ledger.assert_not_called()

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

    def test_review_facts_preflight_rejects_unbound_confidence_without_publish(self):
        _, runtime = self.normalized()
        runtime["evidence_files"].append({
            "source_relpath": "workload/route_review_facts.json",
            "destination_filename": "route_review_facts.json",
            "required": True,
            "max_bytes": 4096,
        })
        fact = {
            "fact_id": "materiality", "claim_id": "materiality",
            "metric": "materiality_gate", "unit": "gate",
            "population": "development", "grouping": "all",
            "point": None, "ci_lower": None, "ci_upper": None,
            "confidence_level": 0.95, "threshold": None,
            "threshold_operator": None, "gate_outcome": "favorable",
            "source_filename": "summary.json", "source_sha256": "c" * 64,
            "json_pointers": {
                "point": None, "ci_lower": None, "ci_upper": None,
                "confidence_level": None, "threshold": None,
                "gate_outcome": "/gate_outcomes/materiality",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "output/workload"
            source.mkdir(parents=True)
            source.joinpath("summary.json").write_text("{}")
            source.joinpath("route_review_facts.json").write_text(json.dumps({
                "schema_version": 2, "route_id": "route",
                "operation_id": "S0", "run_id": "r1", "facts": [fact],
            }))
            evidence = root / "evidence"
            evidence.mkdir()
            with self.assertRaisesRegex(LIFE.LifecycleError, "presence differs"):
                LIFE.copy_evidence(runtime, root / "output", evidence)
            self.assertEqual([], list(evidence.iterdir()))

    def test_schema2_raw_artifact_receipt_seals_stable_output_only(self):
        value, runtime = self.normalized()
        operation = value["operations"]["S0"]
        env = {
            "RUN_ID": "route-r1", "EXPECTED_ROUTE_COMMIT": "a" * 40,
            "RUNNER_SHA256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            (output / "contract").mkdir(parents=True)
            (output / "workload").mkdir()
            (output / "workload/units").mkdir()
            (output / "control").mkdir()
            (output / "contract/check.json").write_text('{"ok":true}\n')
            (output / "workload/summary.json").write_text('{"gain":1}\n')
            (output / "workload/units/unit-1.json").write_text('{"ok":true}\n')
            (output / "runtime.log").write_text("mutable\n")
            evidence = root / "evidence"
            evidence.mkdir()
            filename, digest = LIFE.publish_raw_artifact_receipt(
                output=output, evidence_root=evidence, operation=operation,
                env=env, spec=runtime,
            )
            receipt = json.loads((evidence / filename).read_text())
            manifest = (output / LIFE.RAW_ARTIFACT_MANIFEST_RELPATH).read_text()
            rows = [json.loads(line) for line in manifest.splitlines()]
            self.assertEqual(2, receipt["schema_version"])
            self.assertEqual(3, receipt["entry_count"])
            self.assertEqual(
                {"contract_output": 1, "workload_output": 2},
                receipt["category_counts"],
            )
            self.assertEqual(receipt["entry_count"], sum(receipt["category_counts"].values()))
            self.assertEqual(digest, LIFE.sha256(evidence / filename))
            self.assertEqual(
                [
                    "contract/check.json", "workload/summary.json",
                    "workload/units/unit-1.json",
                ],
                [item["relative_path"] for item in rows],
            )
            self.assertEqual(
                ["contract_output", "workload_output", "workload_output"],
                [item["artifact_class"] for item in rows],
            )
            self.assertNotIn("runtime.log", manifest)

    def test_raw_artifact_receipt_rejects_uncovered_manifest_categories(self):
        value, runtime = self.normalized()
        operation = value["operations"]["S0"]
        env = {
            "RUN_ID": "route-r1", "EXPECTED_ROUTE_COMMIT": "a" * 40,
            "RUNNER_SHA256": "b" * 64,
        }
        records = [{
            "schema_version": 2,
            "relative_path": "workload/units/unit-1.json",
            "artifact_class": "workload/units_output",
            "bytes": 2,
            "sha256": "c" * 64,
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            evidence = root / "evidence"
            output.mkdir()
            evidence.mkdir()
            with patch.object(
                LIFE, "build_raw_artifact_manifest", return_value=(b"{}\n", records),
            ):
                with self.assertRaisesRegex(
                    LIFE.LifecycleError, "category counts do not cover",
                ):
                    LIFE.publish_raw_artifact_receipt(
                        output=output, evidence_root=evidence, operation=operation,
                        env=env, spec=runtime,
                    )
            self.assertEqual([], list(evidence.iterdir()))

    def test_raw_artifact_manifest_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            (output / "workload").mkdir(parents=True)
            target = output / "target.json"
            target.write_text("{}\n")
            (output / "workload/link.json").symlink_to(target)
            with self.assertRaises(LIFE.LifecycleError):
                LIFE.build_raw_artifact_manifest(output)

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

    def test_workload_failure_message_includes_bounded_redacted_program_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.log"
            path.write_text(
                "api_key=do-not-expose /sda/home/private/route.py\n"
                "RuntimeError: first-look summary failed\n",
                encoding="utf-8",
            )
            message = LIFE.program_failure_message("run program", 1, path)
            self.assertTrue(message.startswith("run program failed rc=1; program_tail="))
            self.assertNotIn("do-not-expose", message)
            self.assertNotIn("/sda/home/private", message)
            self.assertIn("RuntimeError: first-look summary failed", message)

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
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
                 "add", "."], cwd=repo, check=True,
            )
            subprocess.run(
                ["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
                 "commit", "-qm", "registry"], cwd=repo, check=True,
            )
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                capture_output=True, check=True,
            ).stdout.strip()
            result = LIFE.resolve_capability_reuse(
                repo, {
                    "schema_version": 2,
                    "reuse_identity": identity,
                },
                commit,
            )
            self.assertTrue(result["engineering_reuse_authorized"])
            self.assertEqual("NONE", result["scientific_authorization"])

    def test_receipt_contract_reuse_revalidates_source_evidence(self):
        _, runtime = self.normalized()
        source_commit = "a" * 40
        candidate_commit = "b" * 40
        runner_sha = "c" * 64
        source_id = "source-r1"
        candidate_id = "source-r2"
        source_identity = {
            "source_commit": "d" * 40,
            "code_path_sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64,
            "runtime_environment_sha256": "3" * 64,
            "device_class": "cpu",
            "input_contract_sha256": "4" * 64,
        }
        candidate_identity = {**source_identity, "code_path_sha256": "5" * 64}
        capability = {
            "schema_version": 2,
            "reuse_identity": candidate_identity,
        }
        runtime["_validated_capability_profile"] = capability
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs"
            source_output = run_root / source_id
            (source_output / "control").mkdir(parents=True)
            (source_output / "contract").mkdir()
            source_env = {
                "RUN_ID": source_id,
                "EXPECTED_ROUTE_COMMIT": source_commit,
                "RUNNER_SHA256": runner_sha,
                "REMOTE_REPO": str(Path(directory) / "repo"),
                "RUN_ROOT": str(run_root),
                "OUTPUT_PATH": str(source_output),
                "GPU": "",
            }
            context = LIFE.context_value(
                phase="contract", env=source_env, spec=runtime,
                output=source_output, status=source_output / "status.txt",
                heartbeat=source_output / "heartbeat.json", assets=[],
            )
            lifecycle_identity = {
                "schema_version": 1,
                "route_id": runtime["route_id"],
                "operation_id": runtime["operation_id"],
                "run_id": source_id,
                "route_commit": source_commit,
                "runner_sha256": runner_sha,
            }
            result = {
                "schema_version": 1,
                "route_id": runtime["route_id"],
                "operation_id": runtime["operation_id"],
                "phase": "contract", "ok": True,
                "checks": {"fixture": True},
                "output_contract_checked": True,
                "finalizer_contract_checked": True,
                "confirmation_images_targets_outcomes_touched": False,
                "canary_touched": False,
                "locked_test_touched": False,
            }
            identity_path = source_output / "control/lifecycle_identity.json"
            context_path = source_output / "control/contract_context.json"
            result_path = source_output / "contract/contract_result.json"
            identity_path.write_text(json.dumps(lifecycle_identity))
            context_path.write_text(json.dumps(context))
            result_path.write_text(json.dumps(result))
            remote_repos = Path(directory) / "repos"
            source_seed = f"{runtime['route_id']}\0{source_id}".encode()
            source_prefix = f"{runtime['route_id'][:32]}-{source_id[:24]}"[:56]
            source_repo = remote_repos / (
                f"{source_prefix}-{hashlib.sha256(source_seed).hexdigest()[:16]}"
            )
            closeout_filename = "source_closeout.json"
            closeout_path = (
                source_repo / "experience_docx/experiment_logs"
                / runtime["route_id"] / closeout_filename
            )
            closeout_path.parent.mkdir(parents=True)
            closeout_path.write_text(json.dumps({
                "route_id": runtime["route_id"],
                "operation_id": runtime["operation_id"],
                "run_id": source_id,
                "route_commit": source_commit,
                "runner_sha256": runner_sha,
                "state": "FAILED_ENGINEERING",
                "failure_phase": "workload",
                "details": {"workload_started": True},
            }))
            profile = CONTRACT.engineering_contract_result_profile(runtime, capability)
            proof = {
                "schema_version": 1,
                "source_receipt_sha256": "6" * 64,
                "repair_parent_receipt_sha256": "6" * 64,
                "reuse_depth": 1,
                "parent_reuse_proof_sha256": None,
                "source_route_id": runtime["route_id"],
                "source_operation_id": runtime["operation_id"],
                "source_route_commit": source_commit,
                "source_output_id": source_id,
                "source_runner_sha256": runner_sha,
                "source_remote_repo": str(source_repo),
                "source_closeout_filename": closeout_filename,
                "source_closeout_path": str(closeout_path),
                "source_closeout_sha256": LIFE.sha256(closeout_path),
                "source_lifecycle_identity_sha256": LIFE.sha256(identity_path),
                "source_contract_context_sha256": LIFE.sha256(context_path),
                "source_contract_result_sha256": LIFE.sha256(result_path),
                "source_capability_identity": source_identity,
                "candidate_capability_identity": candidate_identity,
                "source_entrypoint_sha256": source_identity["code_path_sha256"],
                "candidate_entrypoint_sha256": candidate_identity["code_path_sha256"],
                "source_contract_slice_sha256": "7" * 64,
                "candidate_contract_slice_sha256": "7" * 64,
                "contract_result_profile_sha256": CONTRACT.canonical_digest(profile),
                "classification_sha256": "8" * 64,
                "candidate_route_commit": candidate_commit,
                "candidate_output_id": candidate_id,
                "source_device_class": "cpu",
                "scientific_authorization": "NONE",
            }
            encoded = base64.b64encode(json.dumps(
                proof, sort_keys=True, separators=(",", ":"),
            ).encode()).decode()
            candidate_env = {
                "EXPECTED_ROUTE_COMMIT": candidate_commit,
                "RUN_ID": candidate_id,
                "REMOTE_REPO": str(remote_repos / "candidate"),
            }
            with patch.dict(
                os.environ, {"CONVIR_RECEIPT_CONTRACT_REUSE_B64": encoded},
            ):
                observed = LIFE.validate_receipt_contract_reuse(
                    candidate_env, runtime, capability, run_root,
                )
                self.assertEqual("NONE", observed["scientific_authorization"])
                result_path.write_text(json.dumps({**result, "ok": False}))
                with self.assertRaisesRegex(
                    LIFE.LifecycleError, "source receipt contract evidence changed",
                ):
                    LIFE.validate_receipt_contract_reuse(
                        candidate_env, runtime, capability, run_root,
                    )
                result_path.write_text(json.dumps(result))
                adapter_capability = {
                    **capability,
                    "reuse_identity": {
                        **candidate_identity,
                        "code_path_sha256": "9" * 64,
                    },
                }
                finalization_env = {
                    **candidate_env,
                    "EXPECTED_ROUTE_COMMIT": "f" * 40,
                }
                observed = LIFE.validate_receipt_contract_reuse(
                    finalization_env, runtime, adapter_capability, run_root,
                    expected_candidate_commit=candidate_commit,
                    allow_adapter_code_path=True,
                )
                self.assertEqual(candidate_commit, observed["candidate_route_commit"])

if __name__ == "__main__":
    unittest.main()
