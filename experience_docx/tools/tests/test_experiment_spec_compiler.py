"""Tests for one-source deterministic experiment route compilation."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS = Path(__file__).parents[1]
TESTS = Path(__file__).parent
sys.path[:0] = [str(TOOLS), str(TESTS)]
import convir_ops_mcp as OPS  # noqa: E402
import experiment_spec_compiler as COMPILER  # noqa: E402
from test_convir_ops_v5_final_slim import (  # noqa: E402
    assets, capability, operation, precision, runtime, scientific_contract,
)
from test_research_program_contract import claim, contract  # noqa: E402
from test_scientific_contract import (  # noqa: E402
    contract as v2_scientific_contract,
    precision_certificate as v2_precision_certificate,
)


def sources(rules_commit="a" * 40):
    program = contract()
    authorization = claim()
    authorization["evidence_role"] = "engineering_debug"
    scientific = scientific_contract()
    scientific = {
        key: value for key, value in scientific.items()
        if key not in {"schema_version", "route_id", "operation_id"}
    }
    run = runtime()
    run = {
        key: value for key, value in run.items()
        if key not in {
            "schema_version", "route_id", "operation_id", "asset_manifest_relpath",
            "engineering_contract", "precision_contract",
        }
    } | {
        "engineering_contract": {
            "mode": runtime()["engineering_contract"]["mode"],
            "max_seconds": runtime()["engineering_contract"]["max_seconds"],
            "cost_contract": {
                "strategy": "same_scale_probe",
                "workload_class": "fixed_iteration_map",
                "formal_iterations": 1,
                "max_wall_seconds": 60,
                "max_peak_memory_mib": 512,
            },
        },
        "precision_contract": {
            "mode": runtime()["precision_contract"]["mode"],
            "rationale": runtime()["precision_contract"]["rationale"],
        },
    }
    spec = {
        "schema_version": 1,
        "route_id": "final_slim",
        "rules_commit": rules_commit,
        "title": "Deterministic final slim fixture",
        "rationale": "Verify one-source compilation without changing any scientific decision.",
        "first_operation": "ACCEPT",
        "program_contract_relpath": "experience_docx/research_programs/final_slim.json",
        "operations": {
            "ACCEPT": {
                "operation": operation(),
                "program_authorization": authorization,
                "scientific_contract": scientific,
                "runtime": run,
                "assets": assets()["assets"],
                "capability": {
                    key: value for key, value in capability().items() if key != "schema_version"
                },
                "precision": {
                    key: value for key, value in precision().items() if key != "schema_version"
                },
            },
        },
    }
    return program, spec


def compile_sources(program, spec):
    spec_raw = COMPILER.json_bytes(spec)
    program_raw = COMPILER.json_bytes(program)
    bundle = COMPILER.compile_bundle(
        spec_relpath="experience_docx/experiment_specs/final_slim.json",
        spec_raw=spec_raw, program_raw=program_raw,
        evidence_exists=lambda _: True,
    )
    return spec_raw, program_raw, bundle


def sources_v2(rules_commit="a" * 40):
    program, spec = sources(rules_commit)
    spec["schema_version"] = 2
    item = spec["operations"]["ACCEPT"]
    item["operation"] = {
        key: value for key, value in item["operation"].items()
        if key != "allowed_terminal_tuples"
    }
    scientific = v2_scientific_contract()
    scientific["route_id"] = "final_slim"
    scientific["operation_id"] = "ACCEPT"
    scientific["population"]["evidence_role"] = "engineering_debug"
    item["scientific_contract"] = {
        key: value for key, value in scientific.items()
        if key not in {"schema_version", "route_id", "operation_id"}
    }
    precision = v2_precision_certificate()
    precision["route_id"] = "final_slim"
    precision["operation_id"] = "ACCEPT"
    precision["source_role"] = "engineering_debug"
    item["precision"] = {
        key: value for key, value in precision.items() if key != "schema_version"
    }
    fixture = {"batch": 1, "channels": 3, "height": 64, "width": 64}
    imports = ["compat.shim"]
    cost_contract = item["runtime"]["engineering_contract"]["cost_contract"]
    item["assets"] = [
        {
            "id": "source", "kind": "git_checkout", "path": "/tmp/source",
            "commit": "a" * 40, "require_clean": True,
            "access_role": "engineering_debug", "contract_access": True,
        },
        *[
            {
                "id": identifier, "kind": "file", "path": f"/tmp/{identifier}",
                "sha256": digest, "access_role": "engineering_debug",
                "contract_access": True,
            }
            for identifier, digest in (
                ("code", "b" * 64), ("checkpoint", "c" * 64),
                ("runtime", "d" * 64),
            )
        ],
    ]
    reuse_identity = {
        "source_commit": "a" * 40,
        "code_path_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "runtime_environment_sha256": "d" * 64,
        "device_class": "cuda_sm89",
        "input_contract_sha256": COMPILER.capability_input_contract_sha256(
            contract_mode="gpu_synthetic_no_data",
            minimum_fixture=fixture,
            compatibility_imports=imports,
            cost_contract=cost_contract,
        ),
    }
    item["capability"] = {
        "profile_id": "final_slim_gpu_v2",
        "contract_mode": "gpu_synthetic_no_data",
        "minimum_fixture": fixture,
        "bound_assets": [
            {"id": "source", "identity": "a" * 40},
            {"id": "code", "identity": "b" * 64},
            {"id": "checkpoint", "identity": "c" * 64},
            {"id": "runtime", "identity": "d" * 64},
        ],
        "compatibility_imports": imports,
        "production_path_statement": (
            "The exact production module graph runs on a synthetic CUDA tensor."
        ),
        "protected_data_prohibited": True,
        "scientific_output_prohibited": True,
        "scientific_training_prohibited": True,
        "reuse_identity": reuse_identity,
    }
    return program, spec


class ExperimentSpecCompilerTests(unittest.TestCase):
    def git(self, repo, *args):
        return subprocess.run(
            ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
        ).stdout.strip()

    def test_compilation_is_deterministic_and_schema6_is_identity_bound(self):
        program, spec = sources()
        spec_raw, program_raw, first = compile_sources(program, spec)
        _, _, second = compile_sources(copy.deepcopy(program), copy.deepcopy(spec))
        self.assertEqual(first, second)
        manifest = json.loads(first[COMPILER.MANIFEST_RELPATH])
        self.assertEqual(6, manifest["schema_version"])
        self.assertEqual(COMPILER.sha256(spec_raw), manifest["experiment_spec_sha256"])
        self.assertEqual(COMPILER.sha256(program_raw), manifest["program_contract_sha256"])
        self.assertIn("experience_docx/scientific_contracts/final_slim__ACCEPT.json", first)
        self.assertIn("experience_docx/route_assets/final_slim__ACCEPT.json", first)

    def test_schema2_derives_terminals_and_preserves_source_authority(self):
        program, spec = sources_v2()
        self.assertNotIn(
            "allowed_terminal_tuples", spec["operations"]["ACCEPT"]["operation"],
        )
        _, _, bundle = compile_sources(program, spec)
        manifest = json.loads(bundle[COMPILER.MANIFEST_RELPATH])
        terminals = manifest["operations"]["ACCEPT"]["allowed_terminal_tuples"]
        self.assertEqual(4, len(terminals))
        self.assertIn(
            {"state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE"},
            terminals,
        )
        precision = json.loads(
            bundle["experience_docx/precision_certificates/final_slim__ACCEPT.json"]
        )
        route_assets = json.loads(
            bundle["experience_docx/route_assets/final_slim__ACCEPT.json"]
        )
        self.assertEqual(2, precision["schema_version"])
        self.assertEqual(2, route_assets["schema_version"])
        self.assertEqual("scene_mean_error", precision["primary_estimand_id"])

    def test_schema2_mechanically_derives_capability_input_identity(self):
        program, spec = sources_v2()
        identity = spec["operations"]["ACCEPT"]["capability"]["reuse_identity"]
        expected = identity.pop("input_contract_sha256")
        _, _, bundle = compile_sources(program, spec)
        capability_path = "experience_docx/model_capabilities/final_slim__ACCEPT.json"
        generated = json.loads(bundle[capability_path])
        self.assertEqual(expected, generated["reuse_identity"]["input_contract_sha256"])

    def test_schema2_rejects_an_explicit_wrong_capability_input_identity(self):
        program, spec = sources_v2()
        spec["operations"]["ACCEPT"]["capability"]["reuse_identity"][
            "input_contract_sha256"
        ] = "0" * 64
        with self.assertRaises(COMPILER.ExperimentSpecError):
            compile_sources(program, spec)

    def test_schema2_lint_rejects_precision_first_precedence(self):
        program, spec = sources_v2()
        rules = spec["operations"]["ACCEPT"]["scientific_contract"]["decision_table"]["rules"]
        rules[:] = [
            {
                "id": "precision_veto",
                "when": {"precision": ["unmet", "invalid"]},
                "terminal": "inconclusive",
            },
            {
                "id": "bad_precise",
                "when": {"materiality": ["unfavorable"], "precision": ["met"]},
                "terminal": "fail",
            },
            {
                "id": "good_precise",
                "when": {"materiality": ["favorable"], "precision": ["met"]},
                "terminal": "pass",
            },
            {
                "id": "unresolved_precise",
                "when": {
                    "materiality": ["indeterminate", "invalid"],
                    "precision": ["met"],
                },
                "terminal": "inconclusive",
            },
        ]
        lint = COMPILER.lint_bundle(
            spec_relpath="experience_docx/experiment_specs/final_slim.json",
            spec_raw=COMPILER.json_bytes(spec),
            program_raw=COMPILER.json_bytes(program),
            evidence_exists=lambda _: True,
        )
        self.assertEqual("EXPERIMENT_SPEC_INVALID", lint["status"])
        self.assertTrue(any(
            item["code"] == "SCIENTIFIC_CONTRACT_INVALID" for item in lint["errors"]
        ))

    def test_drift_check_reports_missing_and_modified_generated_files(self):
        program, spec = sources()
        _, _, bundle = compile_sources(program, spec)
        observed = dict(bundle)
        changed = next(path for path in bundle if path.endswith("__ACCEPT.json"))
        observed[changed] += b" "
        missing = COMPILER.MANIFEST_RELPATH
        observed.pop(missing)
        issues = COMPILER.compare_bundle(bundle, observed.__getitem__)
        self.assertIn(f"generated file drift: {changed}", issues)
        self.assertIn(f"missing generated file: {missing}", issues)

    def test_compiler_rejects_program_runtime_role_drift(self):
        program, spec = sources()
        spec["operations"]["ACCEPT"]["runtime"]["evidence_role"] = "development_screening"
        with self.assertRaises(COMPILER.ExperimentSpecError):
            compile_sources(program, spec)

    def test_compiler_does_not_invent_missing_scientific_decisions(self):
        program, spec = sources()
        del spec["operations"]["ACCEPT"]["scientific_contract"]["gates"]
        with self.assertRaises(COMPILER.ExperimentSpecError):
            compile_sources(program, spec)

    def test_lint_aggregates_independent_authoring_errors(self):
        program, spec = sources()
        first = spec["operations"]["ACCEPT"]
        second = copy.deepcopy(first)
        second["operation"]["mode"] = "second"
        second["operation"]["output_id"] = "final-slim-second"
        second["operation"]["closeout_filename"] = "final_slim_second_closeout.json"
        spec["operations"]["SECOND"] = second
        first["runtime"]["evidence_role"] = "development_screening"
        second["runtime"]["engineering_contract"]["cost_contract"]["workload_class"] = "unknown"
        result = COMPILER.lint_bundle(
            spec_relpath="experience_docx/experiment_specs/final_slim.json",
            spec_raw=COMPILER.json_bytes(spec), program_raw=COMPILER.json_bytes(program),
            evidence_exists=lambda _: True,
        )
        self.assertEqual("EXPERIMENT_SPEC_INVALID", result["status"])
        paths = {item["path"] for item in result["errors"]}
        self.assertTrue(any(path.startswith("operations.ACCEPT.") for path in paths))
        self.assertTrue(any(path.startswith("operations.SECOND.") for path in paths))
        self.assertGreaterEqual(len(paths), 2)
        self.assertTrue(all(set(item) == {"path", "code", "message"} for item in result["errors"]))

    def test_lint_aggregates_repo_asset_sha_and_role_errors(self):
        program, spec = sources()
        item = spec["operations"]["ACCEPT"]
        item["assets"][0]["path"] = "{REMOTE_REPO}/models/source.py"
        item["runtime"]["evidence_role"] = "development_screening"
        result = COMPILER.lint_bundle(
            spec_relpath="experience_docx/experiment_specs/final_slim.json",
            spec_raw=COMPILER.json_bytes(spec), program_raw=COMPILER.json_bytes(program),
            evidence_exists=lambda _: True,
            read_repo_file=lambda _: b"current source bytes",
        )
        self.assertEqual("EXPERIMENT_SPEC_INVALID", result["status"])
        codes = {item["code"] for item in result["errors"]}
        self.assertIn("REPO_ASSET_IDENTITY_MISMATCH", codes)
        self.assertTrue(
            {"ROLE_ALIGNMENT", "CONTRACT_INVALID", "RUNTIME_CONTRACT_INVALID"} & codes
        )

    def test_lint_reports_source_text_hygiene_with_other_errors(self):
        program, spec = sources()
        spec["operations"]["ACCEPT"]["runtime"]["evidence_role"] = "development_screening"
        result = COMPILER.lint_bundle(
            spec_relpath="experience_docx/experiment_specs/final_slim.json",
            spec_raw=COMPILER.json_bytes(spec) + b"\n",
            program_raw=COMPILER.json_bytes(program), evidence_exists=lambda _: True,
        )
        codes = {item["code"] for item in result["errors"]}
        self.assertIn("SOURCE_TEXT_TRAILING_BLANK_LINE", codes)
        self.assertTrue(
            {"ROLE_ALIGNMENT", "CONTRACT_INVALID", "RUNTIME_CONTRACT_INVALID"} & codes
        )

    def test_new_authoring_requires_an_explicit_cost_strategy(self):
        program, spec = sources()
        spec["operations"]["ACCEPT"]["runtime"]["engineering_contract"]["cost_contract"] = None
        lint = COMPILER.lint_bundle(
            spec_relpath="experience_docx/experiment_specs/final_slim.json",
            spec_raw=COMPILER.json_bytes(spec), program_raw=COMPILER.json_bytes(program),
            evidence_exists=lambda _: True,
        )
        self.assertTrue(lint["errors"])

    def test_finalize_is_atomic_when_aggregate_lint_fails(self):
        program, spec = sources()
        spec["operations"]["ACCEPT"]["runtime"]["evidence_role"] = "development_screening"
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            spec_path = repo / "experience_docx/experiment_specs/final_slim.json"
            program_path = repo / "experience_docx/research_programs/final_slim.json"
            spec_path.parent.mkdir(parents=True)
            program_path.parent.mkdir(parents=True)
            spec_path.write_bytes(COMPILER.json_bytes(spec))
            program_path.write_bytes(COMPILER.json_bytes(program))
            completed = subprocess.run(
                [
                    sys.executable, str(TOOLS / "experiment_spec_compiler.py"),
                    "--repo", str(repo), "--spec",
                    "experience_docx/experiment_specs/final_slim.json", "--finalize",
                ],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(2, completed.returncode)
            report = json.loads(completed.stdout)
            self.assertEqual("EXPERIMENT_SPEC_INVALID", report["status"])
            self.assertFalse((repo / COMPILER.MANIFEST_RELPATH).exists())
            self.assertFalse((repo / "experience_docx/route_runtime_specs/ACCEPT.json").exists())

    def test_schema1_is_read_only_and_cannot_write_a_new_bundle(self):
        program, spec = sources()
        for action in ("--write", "--finalize"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                repo = Path(directory)
                spec_path = repo / "experience_docx/experiment_specs/final_slim.json"
                program_path = repo / "experience_docx/research_programs/final_slim.json"
                spec_path.parent.mkdir(parents=True)
                program_path.parent.mkdir(parents=True)
                spec_path.write_bytes(COMPILER.json_bytes(spec))
                program_path.write_bytes(COMPILER.json_bytes(program))
                completed = subprocess.run(
                    [
                        sys.executable, str(TOOLS / "experiment_spec_compiler.py"),
                        "--repo", str(repo), "--spec",
                        "experience_docx/experiment_specs/final_slim.json", action,
                    ],
                    text=True, capture_output=True, check=False,
                )
                self.assertEqual(2, completed.returncode)
                report = json.loads(completed.stdout)
                self.assertIn(
                    {
                        "path": "schema_version",
                        "code": "CURRENT_SCHEMA_REQUIRED",
                        "message": (
                            "new experiment authoring requires schema_version 2; "
                            "schema_version 1 is historical read-only compatibility"
                        ),
                    },
                    report["errors"],
                )
                self.assertFalse((repo / COMPILER.MANIFEST_RELPATH).exists())
                self.assertFalse(
                    (repo / "experience_docx/route_runtime_specs/ACCEPT.json").exists()
                )

    def test_atomic_bundle_install_restores_prior_files_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            first = "experience_docx/generated/first.json"
            second = "experience_docx/generated/second.json"
            (repo / first).parent.mkdir(parents=True)
            (repo / first).write_bytes(b"old\n")
            real_replace = COMPILER.os.replace
            calls = 0

            def fail_second_install(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected install failure")
                return real_replace(source, destination)

            with patch.object(COMPILER.os, "replace", side_effect=fail_second_install):
                with self.assertRaises(COMPILER.ExperimentSpecError):
                    COMPILER.write_bundle_atomic(
                        repo, {first: b"new first\n", second: b"new second\n"},
                    )
            self.assertEqual(b"old\n", (repo / first).read_bytes())
            self.assertFalse((repo / second).exists())

    def test_repo_compilation_rejects_absolute_parent_and_symlink_sources(self):
        program, spec = sources()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            outside = root / "outside.json"
            outside.write_bytes(COMPILER.json_bytes(spec))
            with self.assertRaises(COMPILER.ExperimentSpecError):
                COMPILER.compile_from_repo(repo, str(outside))

            spec_path = repo / "experience_docx/experiment_specs/final_slim.json"
            spec_path.parent.mkdir(parents=True)
            escaped = copy.deepcopy(spec)
            escaped["program_contract_relpath"] = "../../outside.json"
            spec_path.write_bytes(COMPILER.json_bytes(escaped))
            with self.assertRaises(COMPILER.ExperimentSpecError):
                COMPILER.compile_from_repo(
                    repo, "experience_docx/experiment_specs/final_slim.json",
                )

            spec_path.unlink()
            spec_path.symlink_to(outside)
            with self.assertRaises(COMPILER.ExperimentSpecError):
                COMPILER.compile_from_repo(
                    repo, "experience_docx/experiment_specs/final_slim.json",
                )

    def test_bundle_install_rejects_symlink_parent_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            outside = root / "outside"
            repo.mkdir()
            outside.mkdir()
            (repo / "experience_docx").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(COMPILER.ExperimentSpecError):
                COMPILER.write_bundle_atomic(
                    repo, {"experience_docx/generated/result.json": b"value\n"},
                )
            self.assertFalse((outside / "generated/result.json").exists())

    def test_authoritative_evidence_is_read_from_exact_main_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.git(repo, "init", "-q")
            self.git(repo, "config", "user.name", "test")
            self.git(repo, "config", "user.email", "test@example.com")
            evidence = repo / "experience_docx/experiment_logs/closed/conclusion.json"
            evidence.parent.mkdir(parents=True)
            evidence.write_text('{"state":"COMPLETED"}\n', encoding="utf-8")
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-qm", "archive evidence")
            commit = self.git(repo, "rev-parse", "HEAD")
            self.git(repo, "update-ref", "refs/remotes/github/main", commit)
            evidence.unlink()

            resolved, exists = COMPILER._authoritative_evidence_resolver(
                repo, commit, COMPILER.DEFAULT_AUTHORITATIVE_MAIN,
            )
            self.assertEqual(commit, resolved)
            self.assertTrue(exists("experience_docx/experiment_logs/closed/conclusion.json"))
            self.assertFalse(COMPILER._repo_file_exists(
                repo, "experience_docx/experiment_logs/closed/conclusion.json",
            ))
            with self.assertRaisesRegex(
                COMPILER.ExperimentSpecError, "must equal the refreshed authoritative",
            ):
                COMPILER._authoritative_evidence_resolver(
                    repo, "0" * 40, COMPILER.DEFAULT_AUTHORITATIVE_MAIN,
                )

    def test_finalize_receipt_is_private_and_identity_bound(self):
        program, spec = sources_v2("a" * 40)
        spec_raw, program_raw, bundle = compile_sources(program, spec)
        receipt = COMPILER.build_authoring_receipt(
            spec_relpath="experience_docx/experiment_specs/final_slim.json",
            spec_raw=spec_raw,
            program_relpath="experience_docx/research_programs/final_slim.json",
            program_raw=program_raw,
            bundle=bundle,
            authoritative_main_commit="a" * 40,
        )
        self.assertEqual(
            "stage_complete_bundle_then_route_ready_once",
            receipt["allowed_next_action"],
        )
        self.assertEqual(
            COMPILER.sha256(COMPILER.json_bytes(receipt["generated_files"])),
            receipt["bundle_sha256"],
        )
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.git(repo, "init", "-q")
            path = COMPILER.write_private_receipt_atomic(
                repo, receipt["route_id"], receipt,
            )
            self.assertTrue(path.is_file())
            self.assertIn(".git", path.parts)
            self.assertEqual([], self.git(repo, "status", "--porcelain").splitlines())

    def test_finalize_uses_exact_main_and_writes_one_private_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.git(repo, "init", "-q")
            self.git(repo, "config", "user.name", "test")
            self.git(repo, "config", "user.email", "test@example.com")
            (repo / "README.md").write_text("rules anchor\n", encoding="utf-8")
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-qm", "rules main")
            rules_commit = self.git(repo, "rev-parse", "HEAD")
            self.git(repo, "update-ref", "refs/remotes/github/main", rules_commit)
            program, spec = sources_v2(rules_commit)
            spec_path = repo / "experience_docx/experiment_specs/final_slim.json"
            program_path = repo / "experience_docx/research_programs/final_slim.json"
            spec_path.parent.mkdir(parents=True)
            program_path.parent.mkdir(parents=True)
            spec_path.write_bytes(COMPILER.json_bytes(spec))
            program_path.write_bytes(COMPILER.json_bytes(program))

            completed = subprocess.run([
                sys.executable, str(TOOLS / "experiment_spec_compiler.py"),
                "--repo", str(repo), "--spec",
                "experience_docx/experiment_specs/final_slim.json", "--finalize",
            ], text=True, capture_output=True, check=False)
            self.assertEqual(0, completed.returncode, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual("EXPERIMENT_SPEC_BUNDLE_FINALIZED", report["status"])
            self.assertEqual(rules_commit, report["receipt"]["authoritative_main_commit"])
            receipt_path = COMPILER._authoring_receipt_path(repo, "final_slim")
            self.assertEqual(report["receipt"], json.loads(receipt_path.read_bytes()))
            self.assertTrue((repo / COMPILER.MANIFEST_RELPATH).is_file())

    def test_compiler_rejects_adjacent_budget_overrun_but_allows_orthogonal_escape(self):
        program, spec = sources()
        program["route_families"]["selector_family"]["adjacent_budget"] = 1
        with self.assertRaises(COMPILER.ExperimentSpecError):
            compile_sources(program, spec)
        program["route_families"]["selector_family"]["state"] = "closed"
        route = spec["operations"]["ACCEPT"]["program_authorization"]
        route.update({
            "mechanism_type": "orthogonal", "adjacent_sequence": None,
            "orthogonal_changes": [{
                "dimension": "measurement_target",
                "reason": "changes the scientific target instead of extending the adjacent selector search",
            }],
        })
        _, _, bundle = compile_sources(program, spec)
        self.assertIn(COMPILER.MANIFEST_RELPATH, bundle)

    def test_schema6_plan_boundary_recompiles_and_rejects_derived_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.git(repo, "init", "-q")
            self.git(repo, "config", "user.name", "test")
            self.git(repo, "config", "user.email", "test@example.com")
            for relpath in OPS.RULE_BUNDLE_RELPATHS:
                path = repo / relpath
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"rule bundle fixture: {relpath}\n", encoding="utf-8")
            runner = repo / "experience_docx/tools/run_route_operation.sh"
            runner.parent.mkdir(parents=True, exist_ok=True)
            runner.write_text("#!/bin/bash\nset -euo pipefail\n", encoding="utf-8")
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-qm", "rules")
            rules_commit = self.git(repo, "rev-parse", "HEAD")
            compatibility = repo / OPS.RULE_COMPATIBILITY_RELPATH
            compatibility.parent.mkdir(parents=True, exist_ok=True)
            compatibility.write_text(json.dumps({
                "schema_version": 1,
                "compatibility_id": "science-fastpath-contract-v1",
                "compatible_prior_rules_commits": [rules_commit],
            }), encoding="utf-8")
            agents = repo / "AGENTS.md"
            agents.write_text(
                agents.read_text(encoding="utf-8") + "compatible documentation update\n",
                encoding="utf-8",
            )
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-qm", "compatible rules update")
            current_main = self.git(repo, "rev-parse", "HEAD")
            program, spec = sources_v2(rules_commit)
            spec_raw, program_raw, bundle = compile_sources(program, spec)
            sources_by_path = {
                "experience_docx/research_programs/final_slim.json": program_raw,
                "experience_docx/experiment_specs/final_slim.json": spec_raw,
            }
            for relpath, raw in {**sources_by_path, **bundle}.items():
                path = repo / relpath
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-qm", "compiled route")
            route_commit = self.git(repo, "rev-parse", "HEAD")
            manifest = json.loads(bundle[COMPILER.MANIFEST_RELPATH])
            context = OPS.parse_manifest(
                manifest, "codex/schema6-fixture", route_commit, current_main,
                str(repo), "ACCEPT",
            )
            self.assertEqual(6, context["route_manifest_schema_version"])
            self.assertEqual(
                "science-fastpath-contract-v1", context["rules_compatibility_id"],
            )
            derived = repo / "experience_docx/scientific_contracts/final_slim__ACCEPT.json"
            value = json.loads(derived.read_text(encoding="utf-8"))
            value["question"] += " tampered"
            derived.write_text(json.dumps(value), encoding="utf-8")
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-qm", "tamper derived file")
            tampered_commit = self.git(repo, "rev-parse", "HEAD")
            with self.assertRaises(OPS.ToolError):
                OPS.parse_manifest(
                    manifest, "codex/schema6-fixture", tampered_commit, current_main,
                    str(repo), "ACCEPT",
                )

    def test_historical_manifest_schemas_remain_supported(self):
        self.assertEqual(4, OPS.SCHEMA_VERSION)
        self.assertTrue({4, 5, 6} <= OPS.SUPPORTED_MANIFEST_SCHEMA_VERSIONS)


if __name__ == "__main__":
    unittest.main()
