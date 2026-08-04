"""Tests for the one-pass compact evidence sync validator."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import validate_evidence_sync as SYNC  # noqa: E402


class EvidenceSyncTests(unittest.TestCase):
    def repo(self, root: Path) -> Path:
        repo = root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        (repo / "README.md").write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
        return repo

    def stage_closeout(self, repo: Path, *, state="COMPLETED_GATE_PASS",
                       engineering=False, run_id="route-r1") -> None:
        if engineering:
            directory = repo / SYNC.ENGINEERING_FAILURE_PREFIX / "route" / run_id
        else:
            directory = repo / "experience_docx/experiment_logs/route"
        directory.mkdir(parents=True)
        value = {
            "route_id": "route", "run_id": run_id, "state": state,
            "decision": None if state == "FAILED_ENGINEERING" else "PASS",
            "authorizes": "NONE",
        }
        (directory / "run_closeout.json").write_text(json.dumps(value), encoding="utf-8")
        (directory / "README.md").write_text("evidence\n", encoding="utf-8")
        subprocess.run(["git", "add", "experience_docx"], cwd=repo, check=True)

    def validate(self, repo: Path, **kwargs):
        return SYNC.validate_staged(
            repo, "route", "HEAD", allow_project_memory_update=False,
            engineering_archive=False, **kwargs,
        )

    def test_valid_compact_sync_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self.repo(Path(directory))
            self.stage_closeout(repo)
            report = self.validate(repo)
            self.assertEqual("EVIDENCE_SYNC_READY", report["status"])
            self.assertEqual(2, len(report["files"]))

    def test_code_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self.repo(Path(directory))
            self.stage_closeout(repo)
            path = repo / "experience_docx/tools/model.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("pass\n", encoding="utf-8")
            subprocess.run(["git", "add", str(path.relative_to(repo))], cwd=repo, check=True)
            with self.assertRaises(SYNC.EvidenceSyncError):
                self.validate(repo)

    def test_project_memory_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self.repo(Path(directory))
            self.stage_closeout(repo)
            path = repo / SYNC.INDEX_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("index\n", encoding="utf-8")
            subprocess.run(["git", "add", SYNC.INDEX_PATH], cwd=repo, check=True)
            with self.assertRaises(SYNC.EvidenceSyncError):
                self.validate(repo)

    def test_engineering_failure_requires_archive_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self.repo(Path(directory))
            self.stage_closeout(repo, state="FAILED_ENGINEERING")
            with self.assertRaises(SYNC.EvidenceSyncError):
                self.validate(repo)
        with tempfile.TemporaryDirectory() as directory:
            repo = self.repo(Path(directory))
            self.stage_closeout(repo, state="FAILED_ENGINEERING", engineering=True)
            report = SYNC.validate_staged(
                repo, "route", "HEAD", allow_project_memory_update=False,
                engineering_archive=True,
            )
            self.assertTrue(report["engineering_archive"])
            self.assertTrue(all(
                item["role"] == "engineering_evidence" for item in report["files"]
            ))

    def test_engineering_archive_rejects_scientific_log_path_and_run_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self.repo(Path(directory))
            self.stage_closeout(repo, state="FAILED_ENGINEERING")
            with self.assertRaisesRegex(SYNC.EvidenceSyncError, "engineering archives"):
                SYNC.validate_staged(
                    repo, "route", "HEAD", allow_project_memory_update=False,
                    engineering_archive=True,
                )
        with tempfile.TemporaryDirectory() as directory:
            repo = self.repo(Path(directory))
            self.stage_closeout(
                repo, state="FAILED_ENGINEERING", engineering=True,
                run_id="route-r1",
            )
            closeout = (
                repo / SYNC.ENGINEERING_FAILURE_PREFIX / "route" / "route-r1"
                / "run_closeout.json"
            )
            value = json.loads(closeout.read_text(encoding="utf-8"))
            value["run_id"] = "route-r2"
            closeout.write_text(json.dumps(value), encoding="utf-8")
            subprocess.run(["git", "add", str(closeout.relative_to(repo))], cwd=repo, check=True)
            with self.assertRaisesRegex(SYNC.EvidenceSyncError, "run_id mismatch"):
                SYNC.validate_staged(
                    repo, "route", "HEAD", allow_project_memory_update=False,
                    engineering_archive=True,
                )


if __name__ == "__main__":
    unittest.main()
