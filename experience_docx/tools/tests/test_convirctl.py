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
        self.github = self.root / "github.git"
        subprocess.run(
            ["/usr/bin/git", "init", "--bare", str(self.github)],
            capture_output=True, check=True,
        )
        self.git("remote", "add", "github", str(self.github))
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
        self.git("push", "--force", "github", "HEAD:main")
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

    def assert_repo_page_contract(self, value):
        serialized = json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.assertLessEqual(len(serialized), convirctl.MAX_REPO_RESPONSE_BYTES)
        self.assertTrue(value["page_complete"])
        self.assertEqual(value["complete"], value["terminal_page"])
        self.assertEqual(value["has_more"], not value["complete"])
        self.assertEqual(value["truncated"], not value["complete"])
        self.assertFalse(value["capture_truncated"])
        self.assertEqual(value["scientific_completeness"], "not_assessed")
        self.assertEqual(value["snapshot_commit"], value["commit"])
        if value["complete"]:
            self.assertIsNone(value["next_cursor"])
        else:
            self.assertIsInstance(value["next_cursor"], str)

    def collect_repo_show(self, relative, *, page_bytes=65536):
        cursor = None
        parts = []
        pages = []
        expected_start = 0
        while True:
            argv = [
                "repo-show", "--repo", str(self.repo), "--path", relative,
                "--page-bytes", str(page_bytes),
            ]
            if cursor is not None:
                argv.extend(["--cursor", cursor])
            page = self.call(*argv)
            self.assertTrue(page["ok"])
            self.assert_repo_page_contract(page)
            self.assertEqual(page["page_start_byte"], expected_start)
            self.assertEqual(
                hashlib.sha256(page["content"].encode("utf-8")).hexdigest(),
                page["page_sha256"],
            )
            self.assertEqual(page["collection_sha256"], page["content_sha256"])
            parts.append(page["content"])
            pages.append(page)
            expected_start = page["page_end_byte"]
            if page["complete"]:
                break
            cursor = page["next_cursor"]
            self.assertLess(len(pages), 1000)
        return "".join(parts), pages

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
        self.assertTrue(shown["complete"])
        self.assertFalse(shown["truncated"])
        self.assert_repo_page_contract(shown)

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
        self.assertTrue(listed["complete"])
        self.assertEqual(listed["total_count"], 1)
        self.assert_repo_page_contract(listed)

        unsafe = self.call(
            "repo-show", "--repo", str(self.repo), "--path", "../outside.txt",
        )
        self.assertEqual("REJECTED", unsafe["state"])

    def test_repo_show_pages_long_jsonl_and_large_csv_exactly(self):
        jsonl = "".join(
            json.dumps(
                {"id": index, "text": ("x" * 240) + ("\u96fe" if index % 7 == 0 else "")},
                ensure_ascii=False, sort_keys=True,
            ) + "\n"
            for index in range(700)
        ).encode("utf-8")
        self.commit_file("evidence/index.jsonl", jsonl)
        rebuilt_jsonl, jsonl_pages = self.collect_repo_show("evidence/index.jsonl")
        self.assertEqual(rebuilt_jsonl.encode("utf-8"), jsonl)
        self.assertGreater(len(jsonl_pages), 1)
        self.assertEqual(
            {page["content_sha256"] for page in jsonl_pages},
            {hashlib.sha256(jsonl).hexdigest()},
        )

        csv_raw = (
            "id,value\n1," + ("z" * 480_000) + "\u96fe\n2,done\n"
        ).encode("utf-8")
        self.commit_file("evidence/large.csv", csv_raw)
        rebuilt_csv, csv_pages = self.collect_repo_show("evidence/large.csv")
        self.assertEqual(rebuilt_csv.encode("utf-8"), csv_raw)
        self.assertGreater(len(csv_pages), 4)
        self.assertEqual(csv_pages[-1]["page_end_byte"], len(csv_raw))

    def test_repo_list_pages_without_duplicates_or_omissions(self):
        expected = []
        for index in range(350):
            relative = f"catalog/group-{index // 25:02d}/record-{index:04d}.txt"
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"record {index}\n", encoding="utf-8")
            expected.append(relative)
        self.git("add", "--", "catalog")
        self.git("commit", "-m", "add paged catalog")
        self.git("push", "--force", "github", "HEAD:main")

        cursor = None
        observed = []
        collection_sha = None
        while True:
            argv = [
                "repo-list", "--repo", str(self.repo), "--path", "catalog",
                "--max-results", "23",
            ]
            if cursor is not None:
                argv.extend(["--cursor", cursor])
            page = self.call(*argv)
            self.assertTrue(page["ok"])
            self.assert_repo_page_contract(page)
            collection_sha = collection_sha or page["collection_sha256"]
            self.assertEqual(page["collection_sha256"], collection_sha)
            self.assertEqual(page["page_start"], len(observed))
            observed.extend(page["paths"])
            if page["complete"]:
                self.assertEqual(page["total_count"], len(observed))
                break
            cursor = page["next_cursor"]
        self.assertEqual(observed, sorted(expected))
        self.assertEqual(len(observed), len(set(observed)))

    def test_repo_path_filters_are_literal_not_pathspec_patterns(self):
        literal = self.repo / "magic" / "*.txt"
        literal.parent.mkdir(parents=True, exist_ok=True)
        literal.write_text("literal target\n", encoding="utf-8")
        (literal.parent / "other.txt").write_text("other target\n", encoding="utf-8")
        self.git("--literal-pathspecs", "add", "--", "magic/*.txt", "magic/other.txt")
        self.git("commit", "-m", "add literal pathspec fixtures")

        listed = self.call(
            "repo-list", "--repo", str(self.repo), "--path", "magic/*.txt",
        )
        self.assertTrue(listed["ok"])
        self.assertEqual(listed["paths"], ["magic/*.txt"])
        searched = self.call(
            "repo-search", "--repo", str(self.repo), "--term", "target",
            "--path", "magic/*.txt",
        )
        self.assertTrue(searched["ok"])
        self.assertEqual(
            [record["path"] for record in searched["match_records"]],
            ["magic/*.txt"],
        )

    def test_repo_search_pages_and_bounds_oversized_lines(self):
        lines = [f"target short {index}" for index in range(450)]
        long_line = "target " + ("q" * 70_000) + "\u96fe"
        lines.extend([long_line, "target final"])
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        self.commit_file("search/matches.txt", raw)

        cursor = None
        records = []
        while True:
            argv = [
                "repo-search", "--repo", str(self.repo), "--path", "search",
                "--term", "target", "--max-results", "19",
            ]
            if cursor is not None:
                argv.extend(["--cursor", cursor])
            page = self.call(*argv)
            self.assertTrue(page["ok"])
            self.assert_repo_page_contract(page)
            self.assertEqual(page["page_start"], len(records))
            self.assertEqual(len(page["matches"]), len(page["match_records"]))
            records.extend(page["match_records"])
            if page["complete"]:
                self.assertEqual(page["total_count"], len(records))
                break
            cursor = page["next_cursor"]
        self.assertEqual(len(records), 452)
        self.assertEqual(len({(item["path"], item["line"]) for item in records}), 452)
        oversized = records[450]
        self.assertTrue(oversized["line_truncated"])
        self.assertEqual(oversized["line_bytes"], len(long_line.encode("utf-8")))
        self.assertEqual(
            oversized["line_sha256"], hashlib.sha256(long_line.encode("utf-8")).hexdigest(),
        )

    def test_repo_cursor_is_query_operation_and_snapshot_bound(self):
        old_raw = ("old-value-" * 12_000).encode("utf-8")
        tracked = self.repo / "paged.txt"
        tracked.write_bytes(old_raw)
        (self.repo / "other.txt").write_text("other\n", encoding="utf-8")
        self.git("add", "--", "paged.txt", "other.txt")
        self.git("commit", "-m", "add cursor fixture")
        self.git("push", "--force", "github", "HEAD:main")
        old_commit = self.git("rev-parse", "HEAD").stdout.strip()

        first = self.call(
            "repo-show", "--repo", str(self.repo), "--path", "paged.txt",
            "--page-bytes", "4096",
        )
        cursor = first["next_cursor"]
        self.assertIsInstance(cursor, str)
        wrong_query = self.call(
            "repo-show", "--repo", str(self.repo), "--path", "other.txt",
            "--cursor", cursor,
        )
        self.assertEqual(wrong_query["state"], "REPO_CURSOR_IDENTITY_MISMATCH")
        wrong_operation = self.call(
            "repo-list", "--repo", str(self.repo), "--cursor", cursor,
        )
        self.assertEqual(wrong_operation["state"], "REPO_CURSOR_IDENTITY_MISMATCH")
        tamper_at = len(cursor) // 2
        replacement = "A" if cursor[tamper_at] != "A" else "B"
        tampered_cursor = cursor[:tamper_at] + replacement + cursor[tamper_at + 1:]
        malformed = self.call(
            "repo-show", "--repo", str(self.repo), "--path", "paged.txt",
            "--cursor", tampered_cursor,
        )
        self.assertEqual(malformed["state"], "REPO_CURSOR_INVALID")

        tracked.write_bytes(("new-value-" * 12_000).encode("utf-8"))
        self.git("add", "--", "paged.txt")
        self.git("commit", "-m", "advance cursor fixture")
        self.git("push", "--force", "github", "HEAD:main")
        new_commit = self.git("rev-parse", "HEAD").stdout.strip()
        continued = self.call(
            "repo-show", "--repo", str(self.repo), "--path", "paged.txt",
            "--cursor", cursor, "--page-bytes", "8192",
        )
        self.assertTrue(continued["ok"])
        self.assertTrue(continued["snapshot_drifted"])
        self.assertEqual(continued["commit"], old_commit)
        self.assertEqual(continued["ref_commit"], new_commit)
        start = continued["page_start_byte"]
        end = continued["page_end_byte"]
        self.assertEqual(continued["content"].encode("utf-8"), old_raw[start:end])
        replayed = self.call(
            "repo-show", "--repo", str(self.repo), "--path", "paged.txt",
            "--cursor", cursor, "--page-bytes", "8192",
        )
        self.assertEqual(continued, replayed)
        explicit_mismatch = self.call(
            "repo-show", "--repo", str(self.repo), "--ref", new_commit,
            "--path", "paged.txt", "--cursor", cursor,
        )
        self.assertEqual(
            explicit_mismatch["state"], "REPO_CURSOR_IDENTITY_MISMATCH"
        )

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

    def test_remote_script_rejects_unpushed_head_and_unrelated_dirty_state(self):
        script = self.commit_file("remote.sh", b"set -euo pipefail\necho ok\n")
        unrelated = self.repo / "unrelated.txt"
        unrelated.write_text("dirty\n", encoding="utf-8")
        self.assertEqual(
            "REJECTED",
            self.call("remote-script", "--script", str(script))["state"],
        )
        unrelated.unlink()
        self.commit_file("local-only.txt", b"not yet remote\n")
        self.git("reset", "--soft", "HEAD~1")
        self.git("commit", "-m", "different unpushed head")
        self.assertEqual(
            "REJECTED",
            self.call("remote-script", "--script", str(script))["state"],
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
