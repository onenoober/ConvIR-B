"""Tests for the complete minimal terminal-science archive fastpath."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import prepare_terminal_archive as ARCHIVE  # noqa: E402


class TerminalArchiveTests(unittest.TestCase):
    def git(self, repo: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
        ).stdout.strip()

    def init_repo(self, path: Path) -> Path:
        path.mkdir()
        self.git(path, "init", "-q")
        self.git(path, "config", "user.name", "test")
        self.git(path, "config", "user.email", "test@example.com")
        return path

    def source(self, root: Path, *, include_results=True, bad_hash=False,
               verdict_only=False, forbidden=False,
               incomplete_conclusion=False) -> tuple[Path, str, str, str, str]:
        repo = self.init_repo(root / "source")
        route_id = "route"
        card = "experience_docx/experiment_cards/route.md"
        card_path = repo / card
        card_path.parent.mkdir(parents=True)
        card_path.write_text("# Route\n\n- Route id: route\n", encoding="utf-8")
        self.git(repo, "add", card)
        operation_id = "A1"
        operations = {
            "schema_version": 4,
            "route_id": route_id,
            "operations": {operation_id: {}},
        }
        manifest = repo / "experience_docx/route_operations.json"
        manifest.write_text(json.dumps(operations), encoding="utf-8")
        spec = {
            "route_id": route_id,
            "operation_id": operation_id,
            "evidence_files": [{
                "destination_filename": "formal_results.csv",
                "required": True,
            }],
        }
        spec_path = repo / f"experience_docx/route_runtime_specs/{operation_id}.json"
        spec_path.parent.mkdir(parents=True)
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        self.git(repo, "add", "experience_docx/route_operations.json",
                 f"experience_docx/route_runtime_specs/{operation_id}.json")
        self.git(repo, "commit", "-qm", "contract")
        commit = self.git(repo, "rev-parse", "HEAD")
        evidence = repo / "experience_docx/experiment_logs/route"
        evidence.mkdir(parents=True)
        filename = "raw_prediction.png" if forbidden else "formal_results.csv"
        result = b"metric,point,lcb95\ngain,0.03,0.02\n"
        expected = hashlib.sha256(result).hexdigest()
        if bad_hash:
            expected = "0" * 64
        closeout = {
            "route_id": route_id,
            "operation_id": operation_id,
            "run_id": "a1-r1",
            "route_commit": commit,
            "state": "COMPLETED_GATE_PASS",
            "decision": "A1_PASS",
            "authorizes": "NONE",
            "evidence_sha256": {} if verdict_only else {filename: expected},
        }
        closeout_rel = "experience_docx/experiment_logs/route/a1_closeout.json"
        (repo / closeout_rel).write_text(json.dumps(closeout), encoding="utf-8")
        conclusion_rel = "experience_docx/experiment_logs/route/a1_conclusion.json"
        conclusion = {
            "route_id": route_id,
            "operation_id": operation_id,
            "run_id": "a1-r1",
            "decision": "A1_PASS",
            "authorizes": "NONE",
            "primary_result": "gain LCB95 passed",
            "gate_reasons": ["gain LCB95 >= threshold"],
            "competing_explanation": "matched control did not explain the gain",
            "limitations": [] if incomplete_conclusion else ["development population only"],
        }
        (repo / conclusion_rel).write_text(json.dumps(conclusion), encoding="utf-8")
        if include_results:
            (evidence / filename).write_bytes(result)
        return repo, commit, closeout_rel, card, conclusion_rel

    def destination(self, root: Path) -> Path:
        repo = self.init_repo(root / "destination")
        (repo / "README.md").write_text("main\n", encoding="utf-8")
        self.git(repo, "add", "README.md")
        self.git(repo, "commit", "-qm", "main")
        self.git(repo, "update-ref", "refs/remotes/github/main", "HEAD")
        return repo

    def audit(self, source: tuple[Path, str, str, str, str]):
        repo, commit, closeout, card, conclusion = source
        return ARCHIVE.audit_source(
            repo, commit, "route", closeout, card, conclusion, "1" * 64,
        )

    def test_complete_bundle_is_staged_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = self.destination(root)
            audit = self.audit(source)
            report = ARCHIVE.prepare_destination(destination, "refs/remotes/github/main", audit)
            self.assertEqual("TERMINAL_ARCHIVE_PREPARED", report["status"])
            self.assertEqual(1, report["preserved_result_files"])
            self.assertEqual(0, report["duplicative_document_updates"])
            self.assertEqual(["commit", "push"], report["remaining_operator_steps"])
            staged = self.git(destination, "diff", "--cached", "--name-only").splitlines()
            self.assertIn(ARCHIVE.INDEX_PATH, staged)
            self.assertIn("experience_docx/experiment_logs/route/formal_results.csv", staged)
            self.assertIn("experience_docx/experiment_logs/route/a1_conclusion.json", staged)
            self.assertNotIn("experience_docx/experiment_logs/route/README.md", staged)

    def test_missing_formal_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory), include_results=False)
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                self.audit(source)

    def test_tampered_formal_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory), bad_hash=True)
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                self.audit(source)

    def test_verdict_only_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory), verdict_only=True)
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                self.audit(source)

    def test_forbidden_binary_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory), forbidden=True)
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                self.audit(source)

    def test_incomplete_conclusion_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory), incomplete_conclusion=True)
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                self.audit(source)

    def test_dirty_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = self.destination(root)
            (destination / "README.md").write_text("dirty\n", encoding="utf-8")
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                ARCHIVE.prepare_destination(
                    destination, "refs/remotes/github/main", self.audit(source),
                )

    def test_conflicting_terminal_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = self.destination(root)
            audit = self.audit(source)
            record = ARCHIVE.index_record(audit)
            record["decision"] = "DIFFERENT"
            index = destination / ARCHIVE.INDEX_PATH
            index.parent.mkdir(parents=True)
            index.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.git(destination, "add", ARCHIVE.INDEX_PATH)
            self.git(destination, "commit", "-qm", "existing record")
            self.git(destination, "update-ref", "refs/remotes/github/main", "HEAD")
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                ARCHIVE.prepare_destination(
                    destination, "refs/remotes/github/main", audit,
                )

    def test_finalize_commits_pushes_and_verifies_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = self.destination(root)
            remote = root / "github.git"
            subprocess.run(["git", "init", "--bare", "-q", remote], check=True)
            self.git(destination, "remote", "add", "github", str(remote))
            self.git(destination, "push", "-q", "github", "HEAD:main")
            audit = self.audit(source)
            prepared = ARCHIVE.prepare_destination(
                destination, "refs/remotes/github/main", audit,
            )
            final = ARCHIVE.finalize_destination(
                destination, "route", prepared["staged_paths"],
            )
            self.assertEqual("TERMINAL_ARCHIVE_PUSHED", final["status"])
            self.assertEqual(final["evidence_commit"], final["remote_commit"])
            self.assertEqual([], final["remaining_operator_steps"])
            self.assertEqual("", self.git(destination, "status", "--porcelain"))

    def test_cli_commit_and_push_is_single_archive_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = self.destination(root)
            remote = root / "github.git"
            subprocess.run(["git", "init", "--bare", "-q", remote], check=True)
            self.git(destination, "remote", "add", "github", str(remote))
            self.git(destination, "push", "-q", "github", "HEAD:main")
            repo, commit, closeout, card, conclusion = source
            command = [
                sys.executable, str(TOOLS / "prepare_terminal_archive.py"),
                "--source-repo", str(repo), "--source-ref", commit,
                "--route-id", "route", "--closeout", closeout,
                "--contract", card, "--conclusion", conclusion,
                "--receipt", "1" * 64, "--destination-repo", str(destination),
                "--base-ref", "refs/remotes/github/main", "--commit-and-push",
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=True)
            self.assertIn("TERMINAL_ARCHIVE_OK", completed.stdout)
            self.assertEqual("", self.git(destination, "status", "--porcelain"))


if __name__ == "__main__":
    unittest.main()
