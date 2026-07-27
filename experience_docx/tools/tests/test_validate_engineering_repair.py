"""Tests for conservative same-contract engineering repair classification."""

import ast
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
TESTS = Path(__file__).parent
sys.path[:0] = [str(TOOLS), str(TESTS)]
import validate_engineering_repair as REPAIR
import experiment_spec_compiler as COMPILER
from test_experiment_spec_compiler import compile_sources, sources


class EngineeringRepairTests(unittest.TestCase):
    @staticmethod
    def _git(repo: Path, *args: str) -> str:
        completed = subprocess.run(
            ["/usr/bin/git", *args], cwd=repo, text=True,
            capture_output=True, check=True,
        )
        return completed.stdout.strip()

    @staticmethod
    def _status(repo: Path) -> str:
        completed = subprocess.run(
            ["/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo, text=True, capture_output=True, check=True,
        )
        return completed.stdout.rstrip("\n")

    def _make_classifier_fixture(
        self, root: Path, *, changed_rationale: bool,
    ) -> tuple[Path, str, str]:
        operation = "REPAIR_TEST"
        repo = root / "repo"
        repo.mkdir()
        self._git(repo, "init", "--quiet")
        self._git(repo, "config", "user.name", "repair-test")
        self._git(repo, "config", "user.email", "repair-test@localhost")
        card = "experience_docx/experiment_cards/repair-test.md"
        spec = f"experience_docx/route_runtime_specs/{operation}.json"
        entrypoint = "experience_docx/tools/repair_test_entrypoint.py"
        for relpath in (card, spec, entrypoint, "experience_docx/route_operations.json"):
            (repo / relpath).parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "route_id": "repair_test_route",
            "rules_commit": "a" * 40,
            "route_card_relpath": card,
            "operations": {operation: {"output_id": "repair-test-r1"}},
        }
        (repo / "experience_docx/route_operations.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        (repo / spec).write_text(
            json.dumps({"entrypoint_relpath": entrypoint}), encoding="utf-8",
        )
        (repo / entrypoint).write_text("def run(value):\n    return value\n", encoding="utf-8")
        (repo / card).write_text(
            "Scientific rationale is frozen.\n\nTerminal authority is frozen.\n",
            encoding="utf-8",
        )
        self._git(repo, "add", ".")
        self._git(repo, "commit", "--quiet", "-m", "base")
        base = self._git(repo, "rev-parse", "HEAD")

        manifest["operations"][operation]["output_id"] = "repair-test-r2"
        (repo / "experience_docx/route_operations.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        rationale = (
            "Scientific rationale and threshold are changed."
            if changed_rationale else "Scientific rationale is frozen."
        )
        (repo / card).write_text(
            rationale
            + "\n\n- Same-contract engineering repair: finalizer binding only"
            + "\n\nTerminal authority is frozen.\n",
            encoding="utf-8",
        )
        self._git(repo, "add", ".")
        self._git(repo, "commit", "--quiet", "-m", "candidate")
        candidate = self._git(repo, "rev-parse", "HEAD")
        return repo, base, candidate

    def _make_schema6_fixture(
        self, root: Path, *, change_run: bool = False, commit_candidate: bool = True,
    ) -> tuple[Path, str, str | None, list[str]]:
        repo = root / "schema6-repo"
        repo.mkdir()
        self._git(repo, "init", "--quiet")
        self._git(repo, "config", "user.name", "repair-test")
        self._git(repo, "config", "user.email", "repair-test@localhost")
        program, spec = sources()
        operation = spec["operations"]["ACCEPT"]
        entrypoint = operation["runtime"]["entrypoint_relpath"]
        before = (
            "def contract(context_path):\n"
            "    payload = {'reference_checks': {}}\n"
            "    return payload\n"
            "def run(context_path):\n"
            "    return 1\n"
        ).encode()
        old_sha = hashlib.sha256(before).hexdigest()
        operation["operation"]["output_id"] = "final-slim-r1"
        operation["assets"][0].update({
            "path": f"{{REMOTE_REPO}}/{entrypoint}", "sha256": old_sha,
        })
        operation["capability"]["bound_assets"][0]["identity"] = old_sha
        operation["capability"]["reuse_identity"]["code_path_sha256"] = old_sha
        spec_raw, program_raw, bundle = compile_sources(program, spec)
        source_paths = {
            "experience_docx/research_programs/final_slim.json": program_raw,
            "experience_docx/experiment_specs/final_slim.json": spec_raw,
            entrypoint: before,
        }
        for relpath, raw in {**source_paths, **bundle}.items():
            path = repo / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self._git(repo, "add", ".")
        self._git(repo, "commit", "--quiet", "-m", "base")
        base = self._git(repo, "rev-parse", "HEAD")

        candidate_spec = copy.deepcopy(spec)
        candidate_operation = candidate_spec["operations"]["ACCEPT"]
        candidate_operation["operation"]["output_id"] = "final-slim-r2"
        after = (
            "def contract(context_path):\n"
            "    return {}\n"
            "def run(context_path):\n"
            f"    return {2 if change_run else 1}\n"
        ).encode()
        new_sha = hashlib.sha256(after).hexdigest()
        candidate_operation["assets"][0]["sha256"] = new_sha
        candidate_operation["capability"]["bound_assets"][0]["identity"] = new_sha
        candidate_operation["capability"]["reuse_identity"]["code_path_sha256"] = new_sha
        candidate_spec_raw, _, candidate_bundle = compile_sources(program, candidate_spec)
        candidate_files = {
            "experience_docx/experiment_specs/final_slim.json": candidate_spec_raw,
            entrypoint: after,
            **candidate_bundle,
        }
        for relpath, raw in candidate_files.items():
            path = repo / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        paths = sorted(filter(None, self._status(repo).splitlines()))
        changed_paths = sorted(line[3:] for line in paths)
        candidate = None
        if commit_candidate:
            self._git(repo, "add", ".")
            self._git(repo, "commit", "--quiet", "-m", "candidate")
            candidate = self._git(repo, "rev-parse", "HEAD")
        return repo, base, candidate, changed_paths

    def test_symbol_qualification_and_contract_fixture_are_safe(self):
        before = b'''\
import old_module
LIMIT = 3
def contract(context):
    return True
def run(value):
    return old_module.clamp(value, LIMIT)
'''
        after = b'''\
from fixed_module import clamp
LIMIT = 3
def contract(context):
    assert clamp(4, 3) == 3
    return True
def run(value):
    return clamp(value, LIMIT)
'''
        self.assertEqual(
            REPAIR.normalized_entrypoint(before),
            REPAIR.normalized_entrypoint(after),
        )

    def test_algorithm_constant_change_is_sensitive(self):
        before = b"LIMIT=3\ndef run(x):\n return min(x, LIMIT)\n"
        after = b"LIMIT=4\ndef run(x):\n return min(x, LIMIT)\n"
        self.assertNotEqual(
            REPAIR.normalized_entrypoint(before),
            REPAIR.normalized_entrypoint(after),
        )

    def test_new_algorithm_helper_is_sensitive(self):
        before = b"def run(x):\n return x\n"
        after = b"def helper(x):\n return x * 2\ndef run(x):\n return helper(x)\n"
        self.assertNotEqual(
            REPAIR.normalized_entrypoint(before),
            REPAIR.normalized_entrypoint(after),
        )

    def test_same_identity_file_path_change_is_safe(self):
        base = {"schema_version": 1, "route_id": "r", "operation_id": "A0", "assets": [{
            "id": "manifest", "kind": "file", "path": "/old", "sha256": "a" * 64,
            "access_role": "development_screening", "contract_access": False,
        }]}
        candidate = {**base, "assets": [{**base["assets"][0], "path": "/new"}]}
        self.assertEqual(["manifest"], REPAIR.validate_asset_repair(base, candidate))

    def test_directory_path_change_is_sensitive(self):
        base = {"schema_version": 1, "route_id": "r", "operation_id": "A0", "assets": [{
            "id": "data", "kind": "directory", "path": "/old",
            "access_role": "development_screening", "contract_access": False,
        }]}
        candidate = {**base, "assets": [{**base["assets"][0], "path": "/new"}]}
        with self.assertRaises(REPAIR.RepairError):
            REPAIR.validate_asset_repair(base, candidate)

    def test_card_allows_output_and_standard_repair_note_only(self):
        before = b"- output: run-r1\n- metric: fixed\n"
        after = b"- output: run-r2\n- metric: fixed\n- Same-contract engineering repair: path only\n"
        self.assertEqual(
            REPAIR.normalize_card(before, "run-r1", "run-r1"),
            REPAIR.normalize_card(after, "run-r1", "run-r2"),
        )

    def test_card_allows_paragraph_repair_note_without_blank_line_false_positive(self):
        before = b"Scientific rationale is frozen.\n\nTerminal authority is frozen.\n"
        after = (
            b"Scientific rationale is frozen.\n\n"
            b"- Same-contract engineering repair: finalizer binding only\n\n"
            b"Terminal authority is frozen.\n"
        )
        self.assertEqual(
            REPAIR.normalize_card(before, "run-r1", "run-r1"),
            REPAIR.normalize_card(after, "run-r1", "run-r2"),
        )

    def test_card_still_rejects_scientific_rationale_change_near_repair_note(self):
        before = b"Scientific rationale is frozen.\n\nTerminal authority is frozen.\n"
        after = (
            b"Scientific rationale and threshold are changed.\n\n"
            b"- Same-contract engineering repair: finalizer binding only\n\n"
            b"Terminal authority is frozen.\n"
        )
        self.assertNotEqual(
            REPAIR.normalize_card(before, "run-r1", "run-r1"),
            REPAIR.normalize_card(after, "run-r1", "run-r2"),
        )

    def test_full_classifier_allows_paragraph_repair_note(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, base, candidate = self._make_classifier_fixture(
                Path(temporary), changed_rationale=False,
            )
            report = REPAIR.validate(repo, base, candidate, "REPAIR_TEST")
        self.assertEqual("AUTO_REPAIR_ELIGIBLE", report["status"])
        self.assertTrue(report["scientific_contract_unchanged"])

    def test_full_classifier_rejects_scientific_rationale_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, base, candidate = self._make_classifier_fixture(
                Path(temporary), changed_rationale=True,
            )
            with self.assertRaisesRegex(
                REPAIR.RepairError,
                "route card changed beyond output identity and repair note",
            ):
                REPAIR.validate(repo, base, candidate, "REPAIR_TEST")

    def test_schema6_compiler_synchronized_entrypoint_repair_is_eligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, base, candidate, _ = self._make_schema6_fixture(Path(temporary))
            report = REPAIR.validate(repo, base, candidate, "ACCEPT")
        self.assertEqual("AUTO_REPAIR_ELIGIBLE", report["status"])
        self.assertTrue(report["schema6_compiler_regeneration_verified"])

    def test_schema6_run_change_remains_sensitive(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, base, candidate, _ = self._make_schema6_fixture(
                Path(temporary), change_run=True,
            )
            with self.assertRaisesRegex(REPAIR.RepairError, "algorithm/control-flow/constants"):
                REPAIR.validate(repo, base, candidate, "ACCEPT")

    def test_worktree_candidate_uses_temporary_index_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, base, _, paths = self._make_schema6_fixture(
                Path(temporary), commit_candidate=False,
            )
            self.assertEqual("", self._git(repo, "diff", "--cached", "--name-only"))
            snapshot = REPAIR.worktree_candidate_snapshot(repo, paths)
            self.assertEqual("", self._git(repo, "diff", "--cached", "--name-only"))
            report = REPAIR.validate(repo, base, snapshot, "ACCEPT")
        self.assertEqual("AUTO_REPAIR_ELIGIBLE", report["status"])

    def test_worktree_candidate_rejects_unlisted_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, _, _, paths = self._make_schema6_fixture(
                Path(temporary), commit_candidate=False,
            )
            with self.assertRaisesRegex(REPAIR.RepairError, "unlisted"):
                REPAIR.worktree_candidate_snapshot(repo, paths[:-1])


if __name__ == "__main__":
    unittest.main()
