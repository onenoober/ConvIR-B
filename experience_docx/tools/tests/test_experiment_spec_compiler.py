"""Tests for one-source deterministic experiment route compilation."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
TESTS = Path(__file__).parent
sys.path[:0] = [str(TOOLS), str(TESTS)]
import convir_ops_mcp as OPS  # noqa: E402
import experiment_spec_compiler as COMPILER  # noqa: E402
from test_convir_ops_v5_final_slim import (  # noqa: E402
    assets, capability, operation, precision, runtime, scientific_contract,
)
from test_research_program_contract import claim, contract  # noqa: E402


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
        self.assertTrue({"operations.ACCEPT", "operations.SECOND"} <= paths)
        self.assertTrue(all(set(item) == {"path", "code", "message"} for item in result["errors"]))

    def test_new_authoring_requires_an_explicit_cost_strategy(self):
        program, spec = sources()
        spec["operations"]["ACCEPT"]["runtime"]["engineering_contract"]["cost_contract"] = None
        lint = COMPILER.lint_bundle(
            spec_relpath="experience_docx/experiment_specs/final_slim.json",
            spec_raw=COMPILER.json_bytes(spec), program_raw=COMPILER.json_bytes(program),
            evidence_exists=lambda _: True,
        )
        self.assertTrue(lint["errors"])

    def test_write_is_atomic_when_aggregate_lint_fails(self):
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
                    "experience_docx/experiment_specs/final_slim.json", "--write",
                ],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(2, completed.returncode)
            report = json.loads(completed.stdout)
            self.assertEqual("EXPERIMENT_SPEC_INVALID", report["status"])
            self.assertFalse((repo / COMPILER.MANIFEST_RELPATH).exists())
            self.assertFalse((repo / "experience_docx/route_runtime_specs/ACCEPT.json").exists())

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
            program, spec = sources(rules_commit)
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
                manifest, "codex/schema6-fixture", route_commit, rules_commit,
                str(repo), "ACCEPT",
            )
            self.assertEqual(6, context["route_manifest_schema_version"])
            derived = repo / "experience_docx/scientific_contracts/final_slim__ACCEPT.json"
            value = json.loads(derived.read_text(encoding="utf-8"))
            value["question"] += " tampered"
            derived.write_text(json.dumps(value), encoding="utf-8")
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-qm", "tamper derived file")
            tampered_commit = self.git(repo, "rev-parse", "HEAD")
            with self.assertRaises(OPS.ToolError):
                OPS.parse_manifest(
                    manifest, "codex/schema6-fixture", tampered_commit, rules_commit,
                    str(repo), "ACCEPT",
                )

    def test_historical_manifest_schemas_remain_supported(self):
        self.assertTrue({4, 5, 6} <= OPS.SUPPORTED_MANIFEST_SCHEMA_VERSIONS)


if __name__ == "__main__":
    unittest.main()
