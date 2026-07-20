"""Tests for conservative same-contract engineering repair classification."""

import ast
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import validate_engineering_repair as REPAIR


class EngineeringRepairTests(unittest.TestCase):
    @staticmethod
    def _git(repo: Path, *args: str) -> str:
        completed = subprocess.run(
            ["/usr/bin/git", *args], cwd=repo, text=True,
            capture_output=True, check=True,
        )
        return completed.stdout.strip()

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


if __name__ == "__main__":
    unittest.main()
