"""Cloud-only tests for deterministic source snapshots."""

import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import experiment_assistant_snapshot as SNAPSHOT  # noqa: E402


def git(repo, *args):
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout.rstrip()


def initialize_repo(root):
    repo = root / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Snapshot Test")
    git(repo, "config", "user.email", "snapshot@example.invalid")
    (repo / "train.py").write_text("print('base')\n", encoding="utf-8")
    (repo / "config.yaml").write_text("epochs: 1\n", encoding="utf-8")
    git(repo, "add", "train.py", "config.yaml")
    git(repo, "commit", "-qm", "base")
    return repo


class ExperimentAssistantSnapshotTests(unittest.TestCase):
    def test_dirty_tracked_and_untracked_source_are_deterministic_without_git_mutation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = initialize_repo(root)
            (repo / "train.py").write_text("print('dirty')\n", encoding="utf-8")
            (repo / "helper.py").write_text("VALUE = 3\n", encoding="utf-8")
            head_before = git(repo, "rev-parse", "HEAD")
            index_before = git(repo, "write-tree")
            first = SNAPSHOT.build_snapshot(repo, root / "first.tar")
            second = SNAPSHOT.build_snapshot(repo, root / "second.tar")
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual((root / "first.tar").read_bytes(), (root / "second.tar").read_bytes())
            with tarfile.open(root / "first.tar", "r") as archive:
                names = set(archive.getnames())
                self.assertIn("train.py", names)
                self.assertIn("helper.py", names)
                self.assertIn(SNAPSHOT.MANIFEST_NAME, names)
                self.assertEqual(b"print('dirty')\n", archive.extractfile("train.py").read())
            self.assertEqual(head_before, git(repo, "rev-parse", "HEAD"))
            self.assertEqual(index_before, git(repo, "write-tree"))
            self.assertEqual(" M train.py\n?? helper.py", git(repo, "status", "--short"))

    def test_data_weights_logs_outputs_and_binary_suffixes_are_excluded(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = initialize_repo(root)
            excluded = {
                "data/raw.txt": "secret",
                "datasets/items.csv": "id,value\n1,2\n",
                "weights/model.pth": "weight",
                "runs/runtime.txt": "log",
                "results/summary.json": "{}",
                "image.png": "not-an-image",
                "table.csv": "sample,value\n",
            }
            for relpath, content in excluded.items():
                path = repo / relpath
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            built = SNAPSHOT.build_snapshot(repo, root / "snapshot.tar")
            self.assertGreater(built["file_count"], 0)
            with tarfile.open(root / "snapshot.tar", "r") as archive:
                names = set(archive.getnames())
            for relpath in excluded:
                self.assertNotIn(relpath, names)

    def test_symlink_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = initialize_repo(root)
            outside = root / "outside.py"
            outside.write_text("SECRET = True\n", encoding="utf-8")
            os.symlink(outside, repo / "linked.py")
            with self.assertRaisesRegex(SNAPSHOT.SnapshotError, "non-symlink"):
                SNAPSHOT.build_snapshot(repo, root / "snapshot.tar")

    def test_oversized_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = initialize_repo(root)
            (repo / "large.py").write_bytes(b"12345")
            with mock.patch.object(SNAPSHOT, "MAX_FILE_BYTES", 4):
                with self.assertRaisesRegex(SNAPSHOT.SnapshotError, "exceeds 4 bytes"):
                    SNAPSHOT.build_snapshot(repo, root / "snapshot.tar")

    def test_unsafe_relative_paths_are_rejected(self):
        for value in ("../outside.py", "/absolute.py", "./local.py", "a\\b.py"):
            with self.assertRaises(SNAPSHOT.SnapshotError):
                SNAPSHOT._safe_relpath(value)


if __name__ == "__main__":
    unittest.main()
