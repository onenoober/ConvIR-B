"""Focused acceptance tests for the final-slim experiment control plane."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import convir_ops_mcp as OPS  # noqa: E402
import route_runtime_contract as CONTRACT  # noqa: E402


def terminal(state, decision, authorizes):
    return {"state": state, "decision": decision, "authorizes": authorizes}


def operation(require_gpu=True):
    return {
        "runner_relpath": CONTRACT.GENERIC_RUNNER_RELPATH,
        "mode": "acceptance", "require_gpu": require_gpu,
        "output_id": "final-slim-r1",
        "closeout_filename": "final_slim_closeout.json",
        "prior_closeout_relpath": None, "prior_terminal_tuple": None,
        "allowed_terminal_tuples": [
            terminal("COMPLETED_GATE_PASS", "FINAL_SLIM_PASS", "ADOPTION"),
            terminal("COMPLETED_GATE_FAIL", "FINAL_SLIM_FAIL", "NONE"),
            terminal("COMPLETED_INCONCLUSIVE", "FINAL_SLIM_INCONCLUSIVE", "NONE"),
            terminal("FAILED_ENGINEERING", None, "NONE"),
        ],
        "workspace_policy": "fresh_route", "output_policy": "new",
        "monitor_profile": "short", "heartbeat_timeout_seconds": 120,
        "min_free_gpu_mib": 1024 if require_gpu else 0,
        "max_gpu_utilization_pct": 100,
    }


def manifest():
    return {"route_id": "final_slim", "operations": {"ACCEPT": operation()}}


def runtime(mode="gpu_synthetic_no_data", precision="formal_precision"):
    return {
        "schema_version": 2, "route_id": "final_slim",
        "operation_id": "ACCEPT",
        "entrypoint_relpath": "experience_docx/tools/final_slim_acceptance.py",
        "asset_manifest_relpath": "experience_docx/route_assets/ACCEPT.json",
        "timeout_seconds": 600, "expected_wall_seconds": 60, "total_units": 1,
        "evidence_role": "engineering_debug", "resume_policy": "none",
        "protected_data_permissions": {
            "allow_confirmation": False, "allow_canary": False,
            "allow_locked_test": False,
        },
        "environment": {},
        "evidence_files": [{
            "source_relpath": "workload/summary.json",
            "destination_filename": "summary.json", "required": True,
            "max_bytes": 65536,
        }],
        "engineering_contract": {
            "mode": mode,
            "capability_profile_relpath": None if mode == "metadata_only"
            else "experience_docx/model_capabilities/final_slim.json",
            "max_seconds": 120,
        },
        "precision_contract": {
            "mode": precision,
            "certificate_relpath": None if precision == "not_applicable"
            else "experience_docx/precision_certificates/final_slim.json",
            "rationale": "Pre-run independent-group precision feasibility check.",
        },
    }


def assets(access_role="engineering_debug", contract_access=True):
    return {
        "schema_version": 1, "route_id": "final_slim",
        "operation_id": "ACCEPT",
        "assets": [{
            "id": "source", "kind": "file", "path": "/tmp/source.py",
            "sha256": "a" * 64, "access_role": access_role,
            "contract_access": contract_access,
        }],
    }


def capability():
    return {
        "schema_version": 1, "profile_id": "final_slim_gpu",
        "contract_mode": "gpu_synthetic_no_data",
        "minimum_fixture": {"batch": 1, "channels": 3, "height": 64, "width": 64},
        "bound_assets": [{"id": "source", "identity": "a" * 64}],
        "compatibility_imports": ["compat.shim"],
        "production_path_statement": "The exact production module graph runs on a synthetic CUDA tensor.",
        "protected_data_prohibited": True,
        "scientific_output_prohibited": True,
        "scientific_training_prohibited": True,
    }


def precision(available=100, half_width=0.2, mode="normal_mean"):
    sd = 0.5
    required = 25
    return {
        "schema_version": 1, "certificate_id": "final_slim_precision",
        "estimand": "mean synthetic validation score", "method": mode,
        "confidence_level": 0.95, "target_half_width": half_width,
        "planning_sd": sd, "independent_groups_available": available,
        "independent_groups_required": required, "feasible": available >= required,
        "source_role": "engineering_debug",
        "source_reference": "Fixed analytical planning variance; no experiment outcome used.",
    }


def scientific_contract():
    return {
        "schema_version": 1, "route_id": "final_slim",
        "operation_id": "ACCEPT",
        "question": "Does the compact control plane preserve every frozen scientific boundary?",
        "population": {
            "evidence_role": "engineering_debug", "grouping_unit": "fixture",
            "independent_group_count": 100, "allow_confirmation": False,
            "allow_canary": False, "allow_locked_test": False,
        },
        "intervention": {
            "primary_variable": "control_plane_version", "reference": "v4",
            "matched_budget": "same regression corpus",
            "fixed_factors": ["tool count", "protected roles", "terminal semantics"],
        },
        "primary_estimand": {
            "id": "regression_pass", "metric": "required checks passed",
            "direction": "higher", "aggregation": "all required checks",
            "unit": "check",
        },
        "controls": ["legacy schema-1 parser", "tampered artifact rejection"],
        "uncertainty": {
            "method": "deterministic complete test corpus", "confidence_level": 0.95,
            "independent_unit": "check",
        },
        "gates": [{
            "id": "all_checks", "type": "integrity",
            "estimand": "required checks passed", "direction": "equal",
            "threshold": True, "decision_role": "decisive",
        }],
        "competing_explanation": "A smaller payload could hide missing evidence instead of removing duplication.",
        "terminal_mapping": {
            "pass": terminal("COMPLETED_GATE_PASS", "FINAL_SLIM_PASS", "ADOPTION"),
            "fail": terminal("COMPLETED_GATE_FAIL", "FINAL_SLIM_FAIL", "NONE"),
            "inconclusive": terminal("COMPLETED_INCONCLUSIVE", "FINAL_SLIM_INCONCLUSIVE", "NONE"),
        },
        "disabled_actions": ["dataset access", "model training", "historical evidence rewrite"],
    }


class FinalSlimTests(unittest.TestCase):
    def git(self, repo, *args):
        return subprocess.run(
            ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
        ).stdout.strip()

    def test_legacy_runtime_spec_remains_compatible(self):
        from test_route_runtime_contract import spec as legacy_spec, manifest as legacy_manifest
        value = CONTRACT.validate_runtime_spec(legacy_spec(), legacy_manifest(), "S0")
        self.assertEqual(1, value["schema_version"])
        self.assertTrue(value["engineering_contract"]["legacy_implicit_contract"])

    def test_gpu_synthetic_capability_is_identity_bound(self):
        spec = CONTRACT.validate_runtime_spec(runtime(), manifest(), "ACCEPT")
        asset = CONTRACT.validate_asset_manifest(assets(), spec)
        value = CONTRACT.validate_model_capability(capability(), spec, asset)
        self.assertEqual("gpu_synthetic_no_data", value["contract_mode"])
        broken = capability()
        broken["bound_assets"][0]["identity"] = "b" * 64
        with self.assertRaises(CONTRACT.ContractError):
            CONTRACT.validate_model_capability(broken, spec, asset)

    def test_gpu_synthetic_contract_rejects_scientific_data_access(self):
        spec = CONTRACT.validate_runtime_spec(runtime(), manifest(), "ACCEPT")
        asset = CONTRACT.validate_asset_manifest(
            assets(access_role="development_screening"), spec,
        )
        with self.assertRaises(CONTRACT.ContractError):
            CONTRACT.validate_model_capability(capability(), spec, asset)

    def test_formal_precision_infeasibility_fails_before_launch(self):
        spec = CONTRACT.validate_runtime_spec(runtime(), manifest(), "ACCEPT")
        value = precision(available=10)
        value["feasible"] = False
        with self.assertRaises(CONTRACT.ContractError):
            CONTRACT.validate_precision_certificate(value, spec)

    def test_descriptive_capacity_may_record_infeasible_precision(self):
        value = runtime(precision="descriptive_capacity")
        value["evidence_role"] = "development_screening"
        spec = CONTRACT.validate_runtime_spec(value, manifest(), "ACCEPT")
        certificate = precision(available=10)
        certificate["feasible"] = False
        self.assertFalse(CONTRACT.validate_precision_certificate(certificate, spec)["feasible"])

    def test_canonical_scientific_contract_binds_terminals(self):
        value = OPS.validate_scientific_contract(
            scientific_contract(), "final_slim", "ACCEPT", operation(),
        )
        self.assertEqual("ACCEPT", value["operation_id"])
        broken = scientific_contract()
        broken["terminal_mapping"]["pass"]["authorizes"] = "DEPLOYMENT"
        broken_operation = operation()
        broken_operation["allowed_terminal_tuples"].append(
            terminal("COMPLETED_GATE_PASS", "FINAL_SLIM_PASS", "DEPLOYMENT")
        )
        with self.assertRaises(OPS.ToolError):
            OPS.validate_scientific_contract(
                broken, "final_slim", "ACCEPT", broken_operation,
            )

    def test_committed_bundle_is_revalidated_at_plan_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.git(repo, "init", "-q")
            self.git(repo, "config", "user.name", "test")
            self.git(repo, "config", "user.email", "test@example.com")
            files = {
                "experience_docx/route_runtime_specs/ACCEPT.json": runtime(),
                "experience_docx/route_assets/ACCEPT.json": assets(),
                "experience_docx/model_capabilities/final_slim.json": capability(),
                "experience_docx/precision_certificates/final_slim.json": precision(),
                "experience_docx/scientific_contracts/ACCEPT.json": scientific_contract(),
            }
            for relpath, value in files.items():
                path = repo / relpath
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-qm", "bundle")
            commit = self.git(repo, "rev-parse", "HEAD")
            route_manifest = {
                "schema_version": 5, "route_id": "final_slim",
                "operations": {"ACCEPT": operation()},
                "scientific_contract_relpaths": {
                    "ACCEPT": "experience_docx/scientific_contracts/ACCEPT.json",
                },
            }
            context = {
                "route_manifest_schema_version": 5,
                "scientific_contract_relpath": "experience_docx/scientific_contracts/ACCEPT.json",
            }
            spec = OPS.validate_committed_operation_bundle(
                str(repo), commit, route_manifest, "ACCEPT", context,
            )
            self.assertEqual(2, spec["schema_version"])
            self.assertEqual("final_slim_gpu", context["capability_profile_id"])
            bad = copy.deepcopy(scientific_contract())
            bad["population"]["evidence_role"] = "development_screening"
            (repo / "experience_docx/scientific_contracts/ACCEPT.json").write_text(
                json.dumps(bad), encoding="utf-8",
            )
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-qm", "tamper role")
            tampered = self.git(repo, "rev-parse", "HEAD")
            with self.assertRaises(OPS.ToolError):
                OPS.validate_committed_operation_bundle(
                    str(repo), tampered, route_manifest, "ACCEPT", dict(context),
                )

    def test_structured_result_text_is_token_bounded(self):
        structured = {
            "operation_state": "READY", "ok": True,
            "changed_paths": [f"path-{index}" for index in range(1000)],
        }
        result = OPS.text_result(json.dumps(structured), structured=structured)
        text = result["content"][0]["text"]
        self.assertLess(len(text.encode()), 512)
        self.assertEqual(1000, len(result["structuredContent"]["changed_paths"]))

    def test_snapshot_phase_receipt_never_derives_scientific_authorization(self):
        base = {
            "branch": "codex/route", "head": "a" * 40,
            "github_main_remote": "b" * 40,
            "github_main_ref_fresh": True, "worktree_clean": True,
            "route_identity": {
                "status": "ROUTE_ID_UNRESOLVED_NO_MANIFEST",
                "route_id_confirmed": False,
            },
            "authoritative_snapshot": {
                "status": "NO_TERMINAL_RECORD", "route_id": "route",
            },
        }
        unresolved = OPS.build_snapshot_phase_receipt(base)
        self.assertEqual("NOT_DERIVED", unresolved["scientific_authorization"])
        self.assertEqual(
            "resolve_route_identity_or_snapshot", unresolved["allowed_next_action"],
        )
        self.assertEqual(64, len(unresolved["receipt_sha256"]))
        ready = OPS.build_snapshot_phase_receipt({
            **base,
            "route_identity": {
                "status": "ROUTE_ID_CONFIRMED_BY_HEAD_MANIFEST",
                "route_id_confirmed": True,
            },
            "authoritative_snapshot": {
                "status": "AUTHORITATIVE_SNAPSHOT_OK", "route_id": "route",
                "route_id_confirmed": True,
            },
        })
        self.assertEqual(
            "read_authoritative_snapshot_references", ready["allowed_next_action"],
        )
        stale = OPS.build_snapshot_phase_receipt({
            **base, "github_main_ref_fresh": False,
        })
        self.assertEqual("refresh_github_main_once", stale["allowed_next_action"])

    def test_authoritative_snapshot_returns_terminal_pointer_without_results(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            conclusion = repo / "experience_docx/experiment_logs/route/conclusion.json"
            closeout = conclusion.with_name("closeout.json")
            index = repo / "experience_docx/EXPERIMENT_TERMINAL_INDEX.jsonl"
            conclusion.parent.mkdir(parents=True)
            conclusion.write_text(json.dumps({
                "route_id": "route", "operation_id": "A0", "run_id": "r1",
                "state": "COMPLETED_GATE_PASS", "decision": "PASS",
                "authorizes": "NONE",
                "primary_result": "pass", "competing_explanation": "none",
                "limitations": ["engineering only"],
            }))
            closeout.write_text(json.dumps({
                "route_id": "route", "operation_id": "A0", "run_id": "r1",
                "state": "COMPLETED_GATE_PASS", "decision": "PASS",
                "authorizes": "NONE",
                "confirmation_images_targets_outcomes_touched": False,
                "canary_touched": False, "locked_test_touched": False,
            }))
            record = {
                "route_id": "route", "operation_id": "A0", "run_id": "r1",
                "state": "COMPLETED_GATE_PASS", "decision": "PASS",
                "authorizes": "NONE", "route_commit": "a" * 40,
                "receipt": "b" * 64, "contract_path": "contract.json",
                "closeout_path": str(closeout.relative_to(repo)),
                "conclusion_path": str(conclusion.relative_to(repo)), "result_paths": [],
            }
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text(json.dumps(record) + "\n")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "snapshot"], cwd=repo, check=True)
            snapshot = OPS.authoritative_snapshot(
                repo, "route", "HEAD", route_id_confirmed=True,
            )
            self.assertEqual("AUTHORITATIVE_SNAPSHOT_OK", snapshot["status"])
            self.assertEqual(64, len(snapshot["terminal_record_sha256"]))
            self.assertEqual(str(conclusion.relative_to(repo)), snapshot["conclusion_path"])
            self.assertNotIn("primary_result", snapshot)
            self.assertNotIn("decision", snapshot)
            self.assertEqual("NOT_DERIVED", snapshot["scientific_authorization"])

    def test_authoritative_snapshot_rejects_ambiguous_terminal_records(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo, check=True,
            )
            index = repo / "experience_docx/EXPERIMENT_TERMINAL_INDEX.jsonl"
            index.parent.mkdir(parents=True)
            records = [
                {
                    "route_id": "route", "operation_id": operation_id,
                    "run_id": run_id, "route_commit": commit,
                    "receipt": receipt, "state": "COMPLETED_GATE_PASS",
                    "decision": "PASS", "authorizes": "NONE",
                }
                for operation_id, run_id, commit, receipt in (
                    ("A0", "r0", "a" * 40, "b" * 64),
                    ("A1", "r1", "c" * 40, "d" * 64),
                )
            ]
            index.write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "ambiguous"], cwd=repo, check=True)
            snapshot = OPS.authoritative_snapshot(
                repo, "route", "HEAD", route_id_confirmed=True,
            )
            self.assertEqual("TERMINAL_RECORD_AMBIGUOUS", snapshot["status"])

    def test_unconfirmed_route_id_does_not_read_terminal_index(self):
        with patch.object(OPS.subprocess, "run") as run:
            snapshot = OPS.authoritative_snapshot(
                Path("/unused"), "requested-route", "HEAD",
                route_id_confirmed=False,
            )
        self.assertEqual("ROUTE_ID_UNRESOLVED", snapshot["status"])
        self.assertFalse(snapshot["route_id_confirmed"])
        run.assert_not_called()

    def test_manifest_route_mismatch_cannot_be_overridden_by_snapshot(self):
        receipt = OPS.build_snapshot_phase_receipt({
            "branch": "codex/actual-route", "head": "a" * 40,
            "github_main_remote": "b" * 40,
            "github_main_ref_fresh": True, "worktree_clean": True,
            "route_identity": {
                "status": "ROUTE_ID_MISMATCH_WITH_HEAD_MANIFEST",
                "route_id_confirmed": False,
            },
            "authoritative_snapshot": {
                "status": "AUTHORITATIVE_SNAPSHOT_OK",
                "route_id": "requested-route", "route_id_confirmed": True,
            },
        })
        self.assertFalse(receipt["route_id_confirmed"])
        self.assertEqual(
            "resolve_route_identity_or_snapshot", receipt["allowed_next_action"],
        )

    def test_terminal_leaf_selection_accepts_one_complete_operation_chain(self):
        root = {
            "schema_version": 2,
            "route_id": "route", "operation_id": "A0", "run_id": "r0",
            "state": "COMPLETED_GATE_PASS", "decision": "PASS",
            "authorizes": "A1", "closeout_path": "logs/a0_closeout.json",
            "prior_terminal_record": {
                "prior_closeout_path": None, "prior_terminal_tuple": None,
            },
        }
        leaf = {
            "schema_version": 2,
            "route_id": "route", "operation_id": "A1", "run_id": "r1",
            "state": "COMPLETED_GATE_FAIL", "decision": "FAIL",
            "authorizes": "NONE", "closeout_path": "logs/a1_closeout.json",
            "prior_terminal_record": {
                "prior_closeout_path": root["closeout_path"],
                "prior_terminal_tuple": {
                    "state": root["state"], "decision": root["decision"],
                    "authorizes": root["authorizes"],
                },
            },
        }
        self.assertEqual(leaf, OPS._select_terminal_leaf([root, leaf]))
        branch = dict(leaf)
        branch["operation_id"] = "A2"
        branch["run_id"] = "r2"
        branch["closeout_path"] = "logs/a2_closeout.json"
        self.assertIsNone(OPS._select_terminal_leaf([root, leaf, branch]))

if __name__ == "__main__":
    unittest.main()
