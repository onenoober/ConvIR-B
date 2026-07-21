"""Tests for bounded exact-base research worktree initialization."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import research_workspace as WORKSPACE  # noqa: E402


class ResearchWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.git(self.source, "init", "-b", "main")
        self.git(self.source, "config", "user.email", "test@example.invalid")
        self.git(self.source, "config", "user.name", "test")
        (self.source / "tracked.txt").write_text("base\n", encoding="utf-8")
        self.git(self.source, "add", ".")
        self.git(self.source, "commit", "-m", "base")
        self.base = self.git(self.source, "rev-parse", "HEAD")
        self.project = self.root / "project.git"
        self.upstream = self.root / "upstream.git"
        subprocess.run(["/usr/bin/git", "init", "--bare", str(self.project)], check=True)
        subprocess.run(["/usr/bin/git", "init", "--bare", str(self.upstream)], check=True)
        self.git(self.source, "remote", "add", "project", str(self.project))
        self.git(self.source, "remote", "add", "upstream", str(self.upstream))
        self.git(self.source, "push", "project", "main")
        self.git(self.source, "push", "upstream", "main")
        self.git(
            self.source, "fetch", "project",
            "refs/heads/main:refs/remotes/project/main",
        )

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def git(repo, *args):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(repo), *args], text=True,
            capture_output=True, check=True,
        ).stdout.strip()

    def create(self, name="route", branch="codex/route"):
        return WORKSPACE.create_worktree(
            source_repo=str(self.source), destination=str(self.root / name),
            base_commit=self.base, branch=branch, project_remote="project",
            upstream_remote="upstream", workspace_root=self.root,
        )

    def test_dirty_source_does_not_block_or_contaminate_new_clean_worktree(self):
        source_config_before = (self.source / ".git/config").read_bytes()
        (self.source / "tracked.txt").write_text("dirty source\n", encoding="utf-8")
        result = self.create()
        self.assertEqual("WORKTREE_READY", result["status"])
        self.assertFalse(result["source_worktree_clean_required"])
        self.assertTrue(result["clean"])
        self.assertEqual(self.base, result["head"])
        self.assertEqual("project", result["push_remote"])
        self.assertEqual("project_evidence", result["project_remote_role"])
        self.assertEqual("upstream", result["upstream_remote"])
        self.assertEqual(
            "upstream_source",
            self.git(self.root / "route", "config", "--get", "remote.upstream.convirRole"),
        )
        self.assertEqual(source_config_before, (self.source / ".git/config").read_bytes())

    def test_wrong_base_existing_destination_and_non_codex_branch_are_rejected(self):
        with self.assertRaises(WORKSPACE.WorkspaceError):
            WORKSPACE.create_worktree(
                source_repo=str(self.source), destination=str(self.root / "wrong"),
                base_commit="0" * 40, branch="codex/wrong", project_remote="project",
                workspace_root=self.root,
            )
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(WORKSPACE.WorkspaceError):
            WORKSPACE.create_worktree(
                source_repo=str(self.source), destination=str(existing),
                base_commit=self.base, branch="codex/existing", project_remote="project",
                workspace_root=self.root,
            )
        with self.assertRaises(WORKSPACE.WorkspaceError):
            self.create(name="unsafe", branch="feature/unsafe")

    def test_destination_escape_nested_path_and_missing_remote_are_rejected(self):
        with self.assertRaises(WORKSPACE.WorkspaceError):
            WORKSPACE.create_worktree(
                source_repo=str(self.source), destination=str(self.root.parent / "escape"),
                base_commit=self.base, branch="codex/escape", project_remote="project",
                workspace_root=self.root,
            )
        with self.assertRaises(WORKSPACE.WorkspaceError):
            WORKSPACE.create_worktree(
                source_repo=str(self.source), destination=str(self.root / "nested" / "route"),
                base_commit=self.base, branch="codex/nested", project_remote="project",
                workspace_root=self.root,
            )
        with self.assertRaises(WORKSPACE.WorkspaceError):
            WORKSPACE.create_worktree(
                source_repo=str(self.source), destination=str(self.root / "missing"),
                base_commit=self.base, branch="codex/missing", project_remote="absent",
                workspace_root=self.root,
            )

    def test_audit_detects_remote_role_or_worktree_drift(self):
        self.create()
        route = self.root / "route"
        self.git(route, "config", "branch.codex/route.pushRemote", "upstream")
        (route / "tracked.txt").write_text("drift\n", encoding="utf-8")
        report = WORKSPACE.audit_worktree(
            route, base_commit=self.base, branch="codex/route", project_remote="project",
        )
        self.assertEqual("WORKTREE_IDENTITY_MISMATCH", report["status"])
        self.assertIn("project_remote_role", report["mismatches"])
        self.assertIn("worktree_clean", report["mismatches"])
        self.assertFalse(report["git_push_performed"])


if __name__ == "__main__":
    unittest.main()
