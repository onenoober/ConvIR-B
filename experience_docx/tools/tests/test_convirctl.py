#!/usr/bin/env python3
"""Runtime regression tests for the bounded ConvIR-B command transport."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "convirctl.py"
SPEC = importlib.util.spec_from_file_location("convirctl_under_test", MODULE_PATH)
convirctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(convirctl)
REAL_GIT = convirctl.GIT
REAL_SSH = convirctl.SSH
REAL_BASH = convirctl.BASH


class ConvirctlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="convir ctl $() | ")
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "repo with spaces"
        self.repo.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.email", "command-test@example.invalid")
        self.git("config", "user.name", "Command Transport Test")
        self.fake_ssh = self.root / "fake ssh"
        self.fake_ssh.write_text(
            """#!/bin/bash
set -euo pipefail
printf '%s\n' "$@" > "${FAKE_SSH_RECORD}.argv"
/bin/cat > "${FAKE_SSH_RECORD}.stdin"
case "${FAKE_SSH_MODE:-echo}" in
  fail) printf '%s\n' 'remote failed' >&2; exit 17 ;;
  timeout) exec /bin/sleep 5 ;;
  large) /usr/bin/head -c 71680 /dev/zero | /usr/bin/tr '\000' x ;;
  echo) printf '%s\n' 'REMOTE_TEST_OK' ;;
  *) exit 19 ;;
esac
""",
            encoding="utf-8",
        )
        self.fake_ssh.chmod(self.fake_ssh.stat().st_mode | stat.S_IXUSR)
        self.record = self.root / "ssh-record.json"
        self.patchers = [
            mock.patch.object(convirctl, "WORKSPACE_ROOT", self.root),
            mock.patch.object(convirctl, "SSH", self.fake_ssh),
            mock.patch.dict(
                os.environ,
                {"FAKE_SSH_RECORD": str(self.record), "FAKE_SSH_MODE": "echo"},
                clear=False,
            ),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp.cleanup()

    def git(self, *args, check=True):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(self.repo), *args],
            capture_output=True,
            text=True,
            check=check,
        )

    def commit_file(self, relative, raw):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        self.git("add", "--", relative)
        self.git("commit", "-m", f"add {relative}")
        return path

    def call(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = convirctl.main(list(args))
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1, output.getvalue())
        value = json.loads(lines[0])
        self.assertEqual(exit_code, value["exit_code"])
        self.assertIn("marker", value)
        return value

    def test_argument_errors_are_json_and_shell_metacharacters_are_data(self):
        invalid = self.call("sha256", "--file")
        self.assertEqual(invalid["state"], "ARGUMENTS_INVALID")
        payload = self.root / "literal $() | value.txt"
        payload.write_bytes(b"literal")
        value = self.call("sha256", "--file", str(payload))
        self.assertTrue(value["ok"])
        self.assertEqual(value["file"], str(payload))

    def test_sha256_match_mismatch_and_workspace_escape(self):
        payload = self.root / "payload.bin"
        payload.write_bytes(b"abc")
        digest = hashlib.sha256(b"abc").hexdigest()
        self.assertTrue(
            self.call("sha256", "--file", str(payload), "--expected", digest)["ok"]
        )
        mismatch = self.call(
            "sha256", "--file", str(payload), "--expected", "0" * 64
        )
        self.assertEqual(mismatch["state"], "SHA256_MISMATCH")
        outside = Path(tempfile.gettempdir()) / "convirctl-outside.txt"
        outside.write_bytes(b"outside")
        try:
            escaped = self.call("sha256", "--file", str(outside))
            self.assertEqual(escaped["state"], "REJECTED")
        finally:
            outside.unlink(missing_ok=True)

    def test_git_state_clean_dirty_identity_and_detached_head(self):
        tracked = self.commit_file("tracked.txt", b"clean")
        head = self.git("rev-parse", "HEAD").stdout.strip()
        clean = self.call(
            "git-state", "--repo", str(self.repo), "--expected-head", head,
            "--expected-branch", "main", "--require-clean",
        )
        self.assertTrue(clean["ok"])
        tracked.write_bytes(b"dirty")
        dirty = self.call(
            "git-state", "--repo", str(self.repo), "--require-clean"
        )
        self.assertEqual(dirty["mismatches"], ["worktree_clean"])
        self.git("restore", "--", "tracked.txt")
        self.git("checkout", "--detach", head)
        detached = self.call(
            "git-state", "--repo", str(self.repo), "--expected-branch", "main"
        )
        self.assertEqual(detached["mismatches"], ["branch"])

    def test_git_remote_match_missing_and_unsafe_names(self):
        self.commit_file("tracked.txt", b"clean")
        remote = self.root / "remote.git"
        subprocess.run(["/usr/bin/git", "init", "--bare", str(remote)], check=True)
        self.git("remote", "add", "origin", str(remote))
        self.git("push", "origin", "main")
        matched = self.call(
            "git-state", "--repo", str(self.repo), "--remote", "origin",
            "--ref", "refs/heads/main", "--require-remote-match",
        )
        self.assertTrue(matched["ok"])
        self.commit_file("local-only.txt", b"not pushed")
        mismatched = self.call(
            "git-state", "--repo", str(self.repo), "--remote", "origin",
            "--ref", "refs/heads/main", "--require-remote-match",
        )
        self.assertEqual(mismatched["mismatches"], ["remote_head"])
        missing = self.call(
            "git-state", "--repo", str(self.repo), "--remote", "origin",
            "--ref", "refs/heads/missing",
        )
        self.assertEqual(missing["state"], "GITHUB_REF_INVALID")
        no_remote = self.call(
            "git-state", "--repo", str(self.repo), "--require-remote-match"
        )
        self.assertEqual(no_remote["state"], "REJECTED")
        unsafe = self.call(
            "git-state", "--repo", str(self.repo), "--remote", "origin;echo",
            "--ref", "refs/heads/main",
        )
        self.assertEqual(unsafe["state"], "ARGUMENTS_INVALID")
        option_like = self.call(
            "git-state", "--repo", str(self.repo), "--remote", "--help",
            "--ref", "refs/heads/main",
        )
        self.assertEqual(option_like["state"], "ARGUMENTS_INVALID")
        bad_timeout = self.call(
            "git-state", "--repo", str(self.repo), "--timeout-seconds", "later"
        )
        self.assertEqual(bad_timeout["state"], "ARGUMENTS_INVALID")

    def test_task_context_and_literal_repo_readers(self):
        self.commit_file(
            "notes/literal name.txt",
            b"literal | $() {commit}\\nsecond line\\n",
        )
        context_ok = self.call(
            "task-context", "--repo", str(self.repo), "--cwd", str(self.repo),
        )
        self.assertTrue(context_ok["ok"])
        self.assertTrue(context_ok["cwd_matches_repo"])
        context_mismatch = self.call(
            "task-context", "--repo", str(self.repo), "--cwd", str(self.root),
        )
        self.assertEqual(context_mismatch["state"], "TASK_CONTEXT_MISMATCH")
        self.assertFalse(context_mismatch["write_allowed"])

        shown = self.call(
            "repo-show", "--repo", str(self.repo),
            "--path", "notes/literal name.txt",
        )
        self.assertTrue(shown["ok"])
        self.assertIn("literal | $() {commit}", shown["content"])

        found = self.call(
            "repo-search", "--repo", str(self.repo),
            "--term", "|", "--term", "$()", "--path", "notes",
        )
        self.assertTrue(found["ok"])
        self.assertFalse(found["zero_matches"])
        self.assertEqual(1, found["result_count"])

        absent = self.call(
            "repo-search", "--repo", str(self.repo), "--term", "does-not-exist",
        )
        self.assertTrue(absent["ok"])
        self.assertTrue(absent["zero_matches"])
        listed = self.call(
            "repo-list", "--repo", str(self.repo), "--path", "notes",
        )
        self.assertTrue(listed["ok"])
        self.assertEqual(["notes/literal name.txt"], listed["paths"])

        unsafe = self.call(
            "repo-show", "--repo", str(self.repo), "--path", "../outside.txt",
        )
        self.assertEqual("REJECTED", unsafe["state"])

    def test_remote_script_normalizes_bom_crlf_and_preserves_stdin(self):
        raw = (
            b"\xef\xbb\xbf#!/usr/bin/env bash\r\n"
            b"# literal | and $() are script data\r\n"
            b"set -euo pipefail\r\nprintf '%s\n' 'literal | $()'\r\n"
        )
        script = self.commit_file("ops/remote.sh", raw)
        value = self.call("remote-script", "--script", str(script))
        self.assertTrue(value["ok"])
        normalized = Path(f"{self.record}.stdin").read_bytes()
        self.assertFalse(normalized.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r", normalized)
        self.assertIn(b"literal | $()", normalized)
        self.assertEqual(
            Path(f"{self.record}.argv").read_text(encoding="utf-8").splitlines(),
            [
                "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
                "convir-4090", "/bin/bash", "-s", "--",
            ],
        )

    def test_remote_script_rejects_empty_missing_strict_syntax_dirty_and_untracked(self):
        cases = {
            "empty.sh": b"",
            "loose.sh": b"#!/usr/bin/env bash\necho loose\n",
            "syntax.sh": b"set -euo pipefail\nif then\n",
        }
        states = {}
        for name, raw in cases.items():
            script = self.commit_file(name, raw)
            states[name] = self.call("remote-script", "--script", str(script))["state"]
        self.assertEqual(states["empty.sh"], "REJECTED")
        self.assertEqual(states["loose.sh"], "REJECTED")
        self.assertEqual(states["syntax.sh"], "SCRIPT_SYNTAX_INVALID")
        good = self.commit_file("good.sh", b"set -euo pipefail\necho ok\n")
        good.write_bytes(b"set -euo pipefail\necho changed\n")
        self.assertEqual(
            self.call("remote-script", "--script", str(good))["state"], "REJECTED"
        )
        self.git("restore", "--", "good.sh")
        untracked = self.repo / "untracked.sh"
        untracked.write_bytes(b"set -euo pipefail\necho no\n")
        self.assertEqual(
            self.call("remote-script", "--script", str(untracked))["state"],
            "REJECTED",
        )

    def test_remote_script_rejects_binary_and_oversized_content(self):
        binary = self.commit_file(
            "binary.sh", b"set -euo pipefail\n\x00echo invalid\n"
        )
        self.assertEqual(
            self.call("remote-script", "--script", str(binary))["state"], "REJECTED"
        )
        oversized = self.commit_file(
            "oversized.sh",
            b"set -euo pipefail\n#" + b"x" * convirctl.MAX_SCRIPT_BYTES,
        )
        self.assertEqual(
            self.call("remote-script", "--script", str(oversized))["state"],
            "REJECTED",
        )

    def test_remote_failure_timeout_and_output_limit_are_typed(self):
        script = self.commit_file("remote.sh", b"set -euo pipefail\necho ok\n")
        with mock.patch.dict(os.environ, {"FAKE_SSH_MODE": "fail"}, clear=False):
            failed = self.call("remote-script", "--script", str(script))
        self.assertEqual(failed["state"], "REMOTE_SCRIPT_FAILED")
        self.assertEqual(failed["exit_code"], 17)
        with mock.patch.dict(os.environ, {"FAKE_SSH_MODE": "timeout"}, clear=False):
            timed_out = self.call(
                "remote-script", "--script", str(script), "--timeout-seconds", "1"
            )
        self.assertEqual(timed_out["state"], "REMOTE_STATE_UNKNOWN")
        self.assertEqual(timed_out["allowed_next_action"], "inspect_once")
        self.assertFalse(timed_out["blind_retry_allowed"])
        with mock.patch.dict(os.environ, {"FAKE_SSH_MODE": "large"}, clear=False):
            large = self.call("remote-script", "--script", str(script))
        self.assertTrue(large["ok"])
        self.assertTrue(large["output_truncated"])
        self.assertEqual(len(large["stdout"]), 64 * 1024)

    def test_ssh_transport_failure_is_distinct(self):
        script = self.commit_file("remote.sh", b"set -euo pipefail\necho ok\n")
        with mock.patch.dict(os.environ, {"FAKE_SSH_MODE": "fail"}, clear=False):
            original = self.fake_ssh.read_text(encoding="utf-8")
            self.fake_ssh.write_text(
                original.replace("exit 17", "exit 255"),
                encoding="utf-8",
            )
            self.fake_ssh.chmod(self.fake_ssh.stat().st_mode | stat.S_IXUSR)
            failed = self.call("remote-script", "--script", str(script))
        self.assertEqual(failed["state"], "SSH_TRANSPORT_FAILED")

    def test_run_argv_never_invokes_a_shell(self):
        probe = self.root / "argv probe"
        probe.write_text(
            "#!/usr/bin/python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n",
            encoding="utf-8",
        )
        probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
        marker = self.root / "must-not-exist"
        literal = f"| $() ; touch {marker}"
        completed = convirctl.run_argv([probe, literal])
        self.assertEqual(json.loads(completed.stdout), [literal])
        self.assertFalse(marker.exists())

    def test_interface_is_fixed_and_has_no_generic_remote_command(self):
        self.assertEqual(str(REAL_GIT), "/usr/bin/git")
        self.assertEqual(str(REAL_SSH), "/usr/bin/ssh")
        self.assertEqual(str(REAL_BASH), "/bin/bash")
        self.assertEqual(convirctl.REMOTE_HOST, "convir-4090")
        value = self.call("remote-command", "--command", "echo unsafe")
        self.assertEqual(value["state"], "ARGUMENTS_INVALID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
