"""Transport and lifecycle tests for the minimal convir-ops schema-v4 bridge."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


MODULE_PATH = Path(__file__).parents[1] / "convir_ops_mcp.py"
SPEC = importlib.util.spec_from_file_location("convir_ops_mcp", MODULE_PATH)
OPS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPS)


def payload(result):
    return result["structuredContent"]


def terminal(decision="PASS", authorizes="formal"):
    return {"state": "COMPLETED_GATE_PASS", "decision": decision, "authorizes": authorizes}


def engineering_terminal():
    return {"state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE"}


def cancelled_terminal():
    return {"state": "CANCELLED_BY_OPERATOR", "decision": None, "authorizes": "NONE"}


def closeout_binding(sha="1" * 64):
    return {
        "identity": {}, "terminal_tuple": terminal(),
        "closeout_sha256": sha, "closeout_filename": "s0_closeout.json",
    }


def operation(**overrides):
    value = {
        "runner_relpath": "experience_docx/tools/run_a1x.sh",
        "mode": "s0",
        "require_gpu": False,
        "output_id": "a1x-s0-r1",
        "closeout_filename": "s0_closeout.json",
        "prior_closeout_relpath": None,
        "prior_terminal_tuple": None,
        "allowed_terminal_tuples": [terminal()],
        "workspace_policy": "fresh_route",
        "output_policy": "new",
        "monitor_profile": "short",
        "heartbeat_timeout_seconds": 120,
        "min_free_gpu_mib": 0,
        "max_gpu_utilization_pct": 100,
    }
    value.update(overrides)
    return value


def manifest(op=None, operation_id="S0"):
    return {
        "schema_version": 4,
        "route_id": "a1x",
        "rules_commit": "b" * 40,
        "route_card_relpath": "experience_docx/experiment_cards/a1x.md",
        "operations": {operation_id: op or operation()},
    }


def context(require_gpu=False):
    return {
        "schema_version": 4,
        "branch": "codex/a1x",
        "route_branch_commit": "a" * 40,
        "current_rules_commit": "b" * 40,
        "route_id": "a1x",
        "operation_id": "S0",
        "remote_repo": "/remote/a1x",
        "run_root": "/runs/a1x",
        "route_card_relpath": "experience_docx/experiment_cards/a1x.md",
        "route_card_blob": "c" * 40,
        "rules_commit": "b" * 40,
        "rules_bundle_digest": "d" * 64,
        "runner_relpath": "experience_docx/tools/run_a1x.sh",
        "runner_sha256": "e" * 64,
        "mode": "s0",
        "require_gpu": require_gpu,
        "output_id": "a1x-s0-r1",
        "output_path": "/runs/a1x/a1x-s0-r1",
        "closeout_filename": "s0_closeout.json",
        "closeout_path": "/remote/a1x/experience_docx/experiment_logs/a1x/s0_closeout.json",
        "prior_closeout_relpath": None,
        "prior_terminal_tuple": None,
        "allowed_terminal_tuples": [terminal(), engineering_terminal()],
        "workspace_policy": "fresh_route",
        "output_policy": "new",
        "monitor_profile": "short",
        "heartbeat_timeout_seconds": 120,
        "min_free_gpu_mib": 12000 if require_gpu else 0,
        "max_gpu_utilization_pct": 10 if require_gpu else 100,
        "session": "convir-a1x-s0",
        "engineering_max_seconds": 120,
        "expected_wall_seconds": 60,
    }


class ConvirOpsV4Tests(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.TemporaryDirectory()
        OPS.STATE_DIR = Path(self.state.name)

    def tearDown(self):
        self.state.cleanup()

    def run_fake_gpu_probe(self, fake_body=None, *, gpu_index=None):
        with tempfile.TemporaryDirectory() as root:
            binary = Path(root) / "nvidia-smi"
            if fake_body is not None:
                binary.write_text(
                    "#!/usr/bin/env bash\nset -euo pipefail\n" + fake_body + "\n",
                    encoding="utf-8",
                )
                binary.chmod(0o755)
            with (
                patch.object(OPS, "NVIDIA_SMI", str(binary)),
                patch.object(OPS, "GPU_PROBE_RETRY_DELAY_SECONDS", 0),
            ):
                body = OPS.gpu_probe_body(context(require_gpu=True), gpu_index)
            return subprocess.run(
                ["/bin/bash", "-c", "set -euo pipefail\n" + body],
                text=True, capture_output=True, timeout=10,
            )

    def parse(self, value=None, operation_id="S0"):
        with (
            patch.object(OPS, "git_show", side_effect=lambda _repo, _commit, path: "- First operation: S0\n" if path.endswith("a1x.md") else "runner"),
            patch.object(OPS, "git_show_bytes", return_value=b"runner\n"),
            patch.object(OPS, "blob_sha", return_value="c" * 40),
            patch.object(OPS, "rule_bundle_digest", return_value="d" * 64),
        ):
            return OPS.parse_manifest(value or manifest(), "codex/a1x", "a" * 40, "b" * 40, "/tmp/repo", operation_id)

    def test_first_operation_is_authorized_by_frozen_card(self):
        value = self.parse()
        self.assertIsNone(value["prior_closeout_relpath"])
        self.assertEqual("d" * 64, value["rules_bundle_digest"])
        self.assertEqual(__import__("hashlib").sha256(b"runner\n").hexdigest(), value["runner_sha256"])

    def test_later_operation_requires_exact_prior_closeout(self):
        prior = terminal("S0_PASS", "FORMAL")
        op = operation(
            mode="formal", prior_closeout_relpath="experience_docx/experiment_logs/a1x/s0_closeout.json",
            prior_terminal_tuple=prior, output_id="a1x-formal-r1",
        )
        value = manifest(op, "FORMAL")
        with (
            patch.object(OPS, "git_show", side_effect=lambda _repo, _commit, path: json.dumps({"route_id": "a1x", **prior}) if path.endswith("s0_closeout.json") else ("- First operation: S0\n" if path.endswith("a1x.md") else "runner")),
            patch.object(OPS, "git_show_bytes", return_value=b"runner\n"),
            patch.object(OPS, "blob_sha", return_value="c" * 40),
            patch.object(OPS, "rule_bundle_digest", return_value="d" * 64),
        ):
            parsed = OPS.parse_manifest(value, "codex/a1x", "a" * 40, "b" * 40, "/tmp/repo", "FORMAL")
        self.assertEqual(prior, parsed["prior_terminal_tuple"])

    def test_later_operation_rejects_ambiguous_authorization(self):
        prior = terminal("S0_PASS", "OTHER")
        value = manifest(operation(
            mode="formal",
            prior_closeout_relpath="experience_docx/experiment_logs/a1x/s0_closeout.json",
            prior_terminal_tuple=prior,
        ), "FORMAL")
        with (
            patch.object(OPS, "git_show", return_value=json.dumps({"route_id": "a1x", **prior})),
            patch.object(OPS, "git_show_bytes", return_value=b"runner\n"),
            patch.object(OPS, "blob_sha", return_value="c" * 40),
            patch.object(OPS, "rule_bundle_digest", return_value="d" * 64),
        ):
            with self.assertRaises(OPS.ToolError):
                OPS.parse_manifest(value, "codex/a1x", "a" * 40, "b" * 40, "/tmp/repo", "FORMAL")

    def test_later_operation_requires_pass_state(self):
        prior = {"state": "FAILED_ENGINEERING", "decision": "STOP", "authorizes": "FORMAL"}
        value = manifest(operation(
            mode="formal",
            prior_closeout_relpath="experience_docx/experiment_logs/a1x/s0_closeout.json",
            prior_terminal_tuple=prior,
        ), "FORMAL")
        with (
            patch.object(OPS, "git_show", return_value=json.dumps({"route_id": "a1x", **prior})),
            patch.object(OPS, "git_show_bytes", return_value=b"runner\n"),
            patch.object(OPS, "blob_sha", return_value="c" * 40),
            patch.object(OPS, "rule_bundle_digest", return_value="d" * 64),
        ):
            with self.assertRaises(OPS.ToolError):
                OPS.parse_manifest(value, "codex/a1x", "a" * 40, "b" * 40, "/tmp/repo", "FORMAL")

    def test_current_rule_bundle_must_match_recorded_digest(self):
        with (
            patch.object(OPS, "git_show", side_effect=lambda _repo, _commit, path: "- First operation: S0\n" if path.endswith("a1x.md") else "runner"),
            patch.object(OPS, "git_show_bytes", return_value=b"runner\n"),
            patch.object(OPS, "blob_sha", return_value="c" * 40),
            patch.object(OPS, "rule_bundle_digest", side_effect=["d" * 64, "f" * 64]),
        ):
            with self.assertRaises(OPS.ToolError):
                OPS.parse_manifest(manifest(), "codex/a1x", "a" * 40, "b" * 40, "/tmp/repo", "S0")

    def test_explicit_compatibility_profile_accepts_prior_rules_commit(self):
        with (
            patch.object(OPS, "git_show", side_effect=lambda _repo, _commit, path: "- First operation: S0\n" if path.endswith("a1x.md") else "runner"),
            patch.object(OPS, "git_show_bytes", return_value=b"runner\n"),
            patch.object(OPS, "blob_sha", return_value="c" * 40),
            patch.object(OPS, "rule_bundle_digest", side_effect=["d" * 64, "f" * 64]),
            patch.object(OPS, "git_object_exists", return_value=True),
            patch.object(OPS, "rule_compatibility_profile", return_value={
                "schema_version": 1,
                "compatibility_id": "science-fastpath-contract-v1",
                "compatible_prior_rules_commits": ["b" * 40],
            }),
        ):
            parsed = OPS.parse_manifest(
                manifest(), "codex/a1x", "a" * 40, "b" * 40, "/tmp/repo", "S0",
            )
        self.assertEqual(
            "science-fastpath-contract-v1", parsed["rules_compatibility_id"],
        )
        self.assertEqual("f" * 64, parsed["rules_bundle_digest"])

    def test_monitor_profiles_are_bounded_to_sixty_seconds(self):
        for profile in OPS.MONITOR_PROFILES.values():
            self.assertLessEqual(profile["max_polls"] * profile["interval_seconds"], 60)

    def test_unknown_start_recovery_bodies_have_valid_bash_syntax(self):
        for body in (
            OPS.atomic_start_body(context(), None),
            OPS.gpu_probe_body(context(require_gpu=True)),
            OPS.gpu_probe_body(context(require_gpu=True), 0),
            OPS.atomic_start_body(context(require_gpu=True), 0),
            OPS.unknown_start_inspection_body(context()),
            OPS.abandoned_start_cleanup_body(context()),
        ):
            completed = subprocess.run(
                ["/bin/bash", "-n"], input=body, text=True,
                capture_output=True, timeout=10,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("clone --quiet --shared --no-checkout", OPS.atomic_start_body(context(), None))

    def test_gpu_probe_selects_gpu_zero_and_preserves_summary(self):
        completed = self.run_fake_gpu_probe(
            "printf '0, 21312, 0\n1, 12848, 47\n'"
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        value = OPS.parse_gpu(completed.stdout)
        self.assertEqual(0, value["index"])
        self.assertEqual(2, value["total_gpu_count"])
        self.assertEqual(21312, value["rows"][0]["free_mib"])

    def test_gpu_probe_selects_first_device_that_satisfies_both_gates(self):
        completed = self.run_fake_gpu_probe(
            "printf '0, 11000, 0\n1, 15000, 9\n2, 16000, 5\n'"
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(1, OPS.parse_gpu(completed.stdout)["index"])

    def test_gpu_probe_true_resource_wait_is_typed_and_retryable(self):
        completed = self.run_fake_gpu_probe(
            "printf '0, 11000, 0\n1, 15000, 11\n'"
        )
        self.assertEqual(75, completed.returncode, completed.stderr)
        error = OPS.ToolError(
            completed.stdout, failure_phase="resource_preflight",
            failure_class="command_infra",
        )
        value = payload(OPS.gpu_probe_failure(error))
        self.assertEqual("RESOURCE_WAIT_REQUIRED", value["operation_state"])
        self.assertFalse(value["observed"]["runner_started"])
        self.assertEqual(2, value["observed"]["gpu_summary"]["total_gpu_count"])
        self.assertEqual(["convir_route_start"], value["allowed_next_actions"])

    def test_gpu_probe_missing_binary_is_not_resource_wait(self):
        completed = self.run_fake_gpu_probe()
        self.assertEqual(76, completed.returncode, completed.stderr)
        self.assertIn("CONVIR_OPS_GPU_QUERY_FAILED reason=binary_missing", completed.stdout)
        value = payload(OPS.gpu_probe_failure(OPS.ToolError(completed.stdout)))
        self.assertEqual("GPU_RESOURCE_PROBE_FAILED", value["operation_state"])
        self.assertEqual(["engineering_review_once"], value["allowed_next_actions"])

    def test_gpu_probe_nonzero_query_preserves_bounded_failure_identity(self):
        completed = self.run_fake_gpu_probe("echo 'driver query failed' >&2\nexit 9")
        self.assertEqual(76, completed.returncode, completed.stderr)
        self.assertRegex(
            completed.stdout,
            r"^CONVIR_OPS_GPU_QUERY_FAILED rc=9 stderr_bytes=20 stderr_sha256=[0-9a-f]{64} stderr_text=driver query failed\n$",
        )
        self.assertLessEqual(len(completed.stdout.encode()), 1024)

    def test_gpu_probe_malformed_query_is_not_resource_wait(self):
        completed = self.run_fake_gpu_probe("printf '0, N/A, 0\n'")
        self.assertEqual(77, completed.returncode, completed.stderr)
        self.assertEqual("CONVIR_OPS_GPU_QUERY_UNPARSEABLE\n", completed.stdout)
        value = payload(OPS.gpu_probe_failure(OPS.ToolError(completed.stdout)))
        self.assertEqual("GPU_RESOURCE_PROBE_FAILED", value["operation_state"])

    def test_gpu_probe_summary_is_bounded(self):
        rows = "".join(f"printf '{index}, 20000, 0\\n'\n" for index in range(10))
        completed = self.run_fake_gpu_probe(rows)
        self.assertEqual(0, completed.returncode, completed.stderr)
        value = OPS.parse_gpu(completed.stdout)
        self.assertEqual(10, value["total_gpu_count"])
        self.assertEqual(OPS.GPU_SUMMARY_LIMIT, len(value["rows"]))
        self.assertTrue(value["summary_truncated"])

    def test_atomic_gpu_recheck_precedes_workspace_creation_and_uses_fixed_binary(self):
        body = OPS.atomic_start_body(context(require_gpu=True), 0)
        self.assertLess(body.index("NVIDIA_SMI=/usr/bin/nvidia-smi"), body.index("REMOTE_REPO="))
        self.assertNotIn("nvidia-smi --query-gpu", body)

    def test_first_operation_requires_exact_card_field(self):
        self.assertEqual("S0", OPS.first_operation_from_card("- First operation: S0\n"))
        with self.assertRaises(OPS.ToolError):
            OPS.first_operation_from_card("S0 appears elsewhere")

    def test_verified_ref_fetch_uses_one_bounded_network_call(self):
        branch = "a" * 40
        main = "b" * 40
        observed = []

        def fake_run(command, **_kwargs):
            observed.append(command)
            if command[-1] == "refs/convir-verify/route":
                return branch
            if command[-1] == "refs/convir-verify/main":
                return main
            return ""

        with patch.object(OPS, "run_local", side_effect=fake_run):
            OPS.fetch_verified_refs(
                "/tmp/repo.git", "refs/heads/codex/a1x", branch, main
            )
        fetches = [command for command in observed if "fetch" in command]
        self.assertEqual(1, len(fetches))
        self.assertIn("+refs/heads/codex/a1x:refs/convir-verify/route", fetches[0])
        self.assertIn("+refs/heads/main:refs/convir-verify/main", fetches[0])

    def test_remote_transport_uses_fixed_argv_and_complete_stdin(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            fake_ssh = root / "fake-ssh"
            argv_path = root / "argv.json"
            stdin_path = root / "stdin.bin"
            fake_ssh.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['CONVIR_TEST_ARGV']).write_text(json.dumps(sys.argv[1:]))\n"
                "Path(os.environ['CONVIR_TEST_STDIN']).write_bytes(sys.stdin.buffer.read())\n"
                "sys.stdout.write('REMOTE_BOUNDARY_OK\\n')\n",
                encoding="utf-8",
            )
            fake_ssh.chmod(0o700)
            with (
                patch.object(OPS, "SSH", str(fake_ssh)),
                patch.dict(os.environ, {
                    "CONVIR_TEST_ARGV": str(argv_path),
                    "CONVIR_TEST_STDIN": str(stdin_path),
                }),
            ):
                output = OPS.run_remote("printf 'PAYLOAD_OK\\n'", timeout=5)
            self.assertEqual("REMOTE_BOUNDARY_OK", output)
            self.assertEqual([
                "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                "convir-4090", "/bin/bash", "-s", "--",
            ], json.loads(argv_path.read_text(encoding="utf-8")))
            self.assertEqual(
                b"#!/usr/bin/env bash\nset -euo pipefail\nprintf 'PAYLOAD_OK\\n'\n",
                stdin_path.read_bytes(),
            )

    def test_remote_transport_drains_but_rejects_oversized_output(self):
        with tempfile.TemporaryDirectory() as root:
            fake_ssh = Path(root) / "fake-ssh"
            fake_ssh.write_text(
                f"#!{sys.executable}\n"
                "import sys\n"
                "sys.stdin.buffer.read()\n"
                f"sys.stdout.write('x' * {OPS.MAX_REMOTE_CAPTURE_BYTES + 1})\n",
                encoding="utf-8",
            )
            fake_ssh.chmod(0o700)
            with patch.object(OPS, "SSH", str(fake_ssh)):
                with self.assertRaisesRegex(OPS.ToolError, "output exceeded"):
                    OPS.run_remote("true", timeout=5)

    def test_remote_transport_timeout_is_unknown_and_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            fake_ssh = Path(root) / "fake-ssh"
            fake_ssh.write_text(
                f"#!{sys.executable}\n"
                "import sys, time\n"
                "sys.stdin.buffer.read()\n"
                "time.sleep(10)\n",
                encoding="utf-8",
            )
            fake_ssh.chmod(0o700)
            started = time.monotonic()
            with patch.object(OPS, "SSH", str(fake_ssh)):
                with self.assertRaisesRegex(OPS.ToolError, "remote state is unknown"):
                    OPS.run_remote("true", timeout=0.1)
            self.assertLess(time.monotonic() - started, 3.0)

    def test_live_rules_check_does_not_mutate_signed_plan(self):
        ctx = context()
        before = json.dumps(ctx, sort_keys=True)
        refs = {
            "refs/heads/codex/a1x": "a" * 40,
            "refs/heads/main": "f" * 40,
        }
        with (
            patch.object(OPS, "github_refs", return_value=refs),
            patch.object(OPS, "prepare_seeded_bare"),
            patch.object(OPS, "ensure_commit"),
            patch.object(OPS, "rule_bundle_digest", return_value="d" * 64),
        ):
            OPS.verify_live_context(ctx)
        self.assertEqual(before, json.dumps(ctx, sort_keys=True))

    def test_launch_timeout_recovers_receipt_from_bound_output(self):
        plan = {"context": context(), "issued_at": int(time.time()), "expires_at": int(time.time()) + 60, "nonce": "n"}
        token = OPS.write_new_record("plan", plan, {"receipt": None})
        error = OPS.ToolError("timeout", failure_phase="launch_command", failure_class="command_infra")
        inspection = (
            "CONVIR_OPS_START_INSPECTION repo=exact runner=exact active=false "
            "output=present identity=valid closeout=valid dirty=1\n"
        )
        monitor = (
            "CONVIR_OPS_MONITOR polls=1 active=true terminal=false stale=false "
            "heartbeat_age=1 heartbeat_source=heartbeat\n"
            "CONVIR_OPS_STATUS_BEGIN\n"
            '{"phase":"workload","completed":1,"total":2}\n'
            "CONVIR_OPS_STATUS_END\n"
        )
        with patch.object(OPS, "verify_live_context"), patch.object(
            OPS, "run_remote", side_effect=[error, inspection, monitor]
        ):
            first = payload(OPS.tool_start({"plan_token": token}))
            second = payload(OPS.tool_start({"plan_token": token}))
            third = payload(OPS.tool_start({"plan_token": token}))
        self.assertEqual("START_STATE_UNKNOWN", first["operation_state"])
        self.assertEqual("RUNNING_VERIFIED", second["operation_state"])
        self.assertEqual("LAUNCH_IDEMPOTENT", third["operation_state"])
        self.assertEqual(second["receipt"], third["receipt"])

    def test_abandoned_exact_workspace_is_cleaned_before_retry(self):
        plan = {"context": context(), "issued_at": int(time.time()), "expires_at": int(time.time()) + 60, "nonce": "n"}
        token = OPS.write_new_record("plan", plan, {"receipt": None})
        error = OPS.ToolError("timeout", failure_phase="launch_command", failure_class="command_infra")
        inspection = (
            "CONVIR_OPS_START_INSPECTION repo=exact runner=exact active=false "
            "output=absent identity=absent closeout=absent dirty=0\n"
        )
        with patch.object(OPS, "verify_live_context"), patch.object(
            OPS, "run_remote",
            side_effect=[error, inspection, "CONVIR_OPS_ABANDONED_START_CLEANUP_OK\n"],
        ):
            first = payload(OPS.tool_start({"plan_token": token}))
            second = payload(OPS.tool_start({"plan_token": token}))
        self.assertEqual("START_STATE_UNKNOWN", first["operation_state"])
        self.assertEqual("START_RETRY_READY", second["operation_state"])
        self.assertEqual(["convir_route_start"], second["allowed_next_actions"])

    def test_unknown_start_recovery_is_single_shot(self):
        plan = {"context": context(), "issued_at": int(time.time()), "expires_at": int(time.time()) + 60, "nonce": "n"}
        token = OPS.write_new_record("plan", plan, {"receipt": None})
        error = OPS.ToolError("timeout", failure_phase="launch_command", failure_class="command_infra")
        remote = Mock(side_effect=[error, error])
        with patch.object(OPS, "verify_live_context"), patch.object(OPS, "run_remote", remote):
            payload(OPS.tool_start({"plan_token": token}))
            second = payload(OPS.tool_start({"plan_token": token}))
            third = payload(OPS.tool_start({"plan_token": token}))
        self.assertEqual("START_STATE_UNKNOWN", second["operation_state"])
        self.assertEqual("START_STATE_UNKNOWN", third["operation_state"])
        self.assertEqual(2, remote.call_count)

    def test_resource_wait_is_retryable_before_launch_attempt(self):
        plan = {"context": context(require_gpu=True), "issued_at": int(time.time()), "expires_at": int(time.time()) + 60, "nonce": "n"}
        token = OPS.write_new_record("plan", plan, {"receipt": None})
        error = OPS.ToolError(
            "CONVIR_OPS_GPU_SUMMARY rows=1 total=1 data=0:1000:99\n"
            "CONVIR_OPS_RESOURCE_WAIT_REQUIRED",
            failure_phase="resource_preflight", failure_class="command_infra",
        )
        with patch.object(OPS, "verify_live_context"), patch.object(OPS, "run_remote", side_effect=error):
            first = payload(OPS.tool_start({"plan_token": token}))
            second = payload(OPS.tool_start({"plan_token": token}))
        self.assertEqual("RESOURCE_WAIT_REQUIRED", first["operation_state"])
        self.assertEqual("RESOURCE_WAIT_REQUIRED", second["operation_state"])
        self.assertFalse(first["observed"]["runner_started"])

    def test_gpu_query_failure_is_not_misclassified_as_resource_wait(self):
        plan = {"context": context(require_gpu=True), "issued_at": int(time.time()), "expires_at": int(time.time()) + 60, "nonce": "n"}
        token = OPS.write_new_record("plan", plan, {"receipt": None})
        error = OPS.ToolError(
            "resource_preflight failed rc=76: CONVIR_OPS_GPU_QUERY_FAILED rc=9 stderr_bytes=20 stderr_sha256=" + "a" * 64,
            failure_phase="resource_preflight", failure_class="command_infra",
        )
        with patch.object(OPS, "verify_live_context"), patch.object(OPS, "run_remote", side_effect=error):
            result = payload(OPS.tool_start({"plan_token": token}))
        self.assertEqual("GPU_RESOURCE_PROBE_FAILED", result["operation_state"])
        self.assertFalse(result["observed"]["runner_started"])
        self.assertEqual(["engineering_review_once"], result["allowed_next_actions"])

    def test_dead_session_closes_finish_and_cannot_be_polled_again(self):
        receipt_payload = {
            "context": context(), "gpu_index": None,
            "launch_digest": "f" * 64, "issued_at": 1,
        }
        receipt = OPS.write_new_record(
            "receipt", receipt_payload,
            {"launched": True, "finish_calls": 0, "finish_closed": None},
        )
        output = (
            "CONVIR_OPS_MONITOR polls=1 active=false terminal=false "
            "stale=false heartbeat_age=-1\n"
            "CONVIR_OPS_STATUS_BEGIN\nCONVIR_OPS_STATUS_END\n"
        )
        with patch.object(OPS, "run_remote", return_value=output):
            first = payload(OPS.tool_finish({"receipt": receipt}))
            second = payload(OPS.tool_finish({"receipt": receipt}))
        self.assertEqual("CLOSEOUT_MISSING", first["operation_state"])
        self.assertEqual("FINISH_REJECTED", second["operation_state"])

    def test_start_returns_running_verified_only_after_positive_progress(self):
        plan = {
            "context": context(), "issued_at": int(time.time()),
            "expires_at": int(time.time()) + 60, "nonce": "n",
        }
        token = OPS.write_new_record("plan", plan, {"receipt": None})
        monitor = (
            "CONVIR_OPS_MONITOR polls=1 active=true terminal=false stale=false "
            "heartbeat_age=1 heartbeat_source=heartbeat\n"
            "CONVIR_OPS_STATUS_BEGIN\n"
            '{"phase":"contract","event":"contract_pass","completed":1,"total":1}\n'
            '{"phase":"workload","event":"workload_progress","completed":3,"total":10}\n'
            "CONVIR_OPS_STATUS_END\n"
        )
        with patch.object(OPS, "verify_live_context"), patch.object(
            OPS, "run_remote", side_effect=["CONVIR_OPS_LAUNCHED\n", monitor],
        ):
            result = payload(OPS.tool_start({"plan_token": token}))
        self.assertEqual("RUNNING_VERIFIED", result["operation_state"])
        self.assertEqual(3, result["observed"]["workload_progress"]["completed_units"])
        self.assertTrue(result["workload_verified"])
        self.assertIn("receipt", result)

    def test_start_does_not_claim_running_at_workload_zero(self):
        plan = {
            "context": context(), "issued_at": int(time.time()),
            "expires_at": int(time.time()) + 60, "nonce": "n",
        }
        token = OPS.write_new_record("plan", plan, {"receipt": None})
        monitor = (
            "CONVIR_OPS_MONITOR polls=1 active=true terminal=false stale=false "
            "heartbeat_age=1 heartbeat_source=heartbeat\n"
            "CONVIR_OPS_STATUS_BEGIN\n"
            '{"phase":"contract","event":"contract_pass","completed":1,"total":1}\n'
            '{"phase":"workload","event":"workload_start","completed":0,"total":10}\n'
            "CONVIR_OPS_STATUS_END\n"
        )
        with patch.object(OPS, "verify_live_context"), patch.object(
            OPS, "run_remote", side_effect=["CONVIR_OPS_LAUNCHED\n", monitor],
        ):
            result = payload(OPS.tool_start({"plan_token": token}))
        self.assertEqual("LAUNCHED_PENDING_VERIFICATION", result["operation_state"])
        self.assertFalse(result["workload_verified"])
        self.assertEqual(
            {"completed_units": 0, "total_units": 10},
            result["observed"]["workload_progress"],
        )
        self.assertEqual(30, result["retry_after_seconds"])
        self.assertIn("not_before_unix", result)
        self.assertIn("expected_phase_end_unix", result)

    def test_finish_throttle_returns_cached_result_without_remote_or_budget_use(self):
        ctx = context()
        receipt_payload = {
            "context": ctx, "gpu_index": None,
            "launch_digest": "f" * 64, "issued_at": int(time.time()),
        }
        cached = OPS.typed_result(
            True, "LAUNCHED_PENDING_VERIFICATION", receipt="r",
            retry_after_seconds=30, not_before_unix=int(time.time()) + 30,
        )
        receipt = OPS.write_new_record(
            "receipt", receipt_payload,
            {
                "launched": True, "finish_calls": 1, "finish_closed": None,
                "finish_not_before_unix": int(time.time()) + 30,
                "pending_finish_response": cached,
            },
        )
        with patch.object(OPS, "run_remote") as remote:
            result = payload(OPS.tool_finish({"receipt": receipt}))
        self.assertEqual("LAUNCHED_PENDING_VERIFICATION", result["operation_state"])
        remote.assert_not_called()
        with OPS.locked_record("receipt", receipt) as record:
            self.assertEqual(1, record["finish_calls"])

    def test_progress_only_bypasses_eta_and_redacts_scientific_status(self):
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": context(), "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": int(time.time()) - 10,
            },
            {
                "launched": True, "finish_calls": 1, "finish_closed": None,
                "finish_not_before_unix": int(time.time()) + 3600,
                "pending_finish_response": OPS.typed_result(
                    True, "RUNNING_VERIFIED", receipt="sealed",
                ),
            },
        )
        remote_output = (
            f"CONVIR_OPS_OPERATOR_OBSERVATION snapshot_at={int(time.time())} "
            "active=true terminal=false heartbeat_age=2 heartbeat_source=heartbeat\n"
            "CONVIR_OPS_STATUS_BEGIN\n"
            '{"R3_PROGRESS":{"stage":"outcome_blind_scene_extraction",'
            '"completed_units":31,"total_units":851,"metric":99.9,'
            '"sample_id":"secret"}}\n'
            "CONVIR_OPS_STATUS_END\n"
        )
        with patch.object(OPS, "validate_operator_context"), patch.object(
            OPS, "run_remote", return_value=remote_output,
        ) as remote:
            result = payload(OPS.tool_finish({
                "receipt": receipt, "observation_mode": "progress_only",
            }))
        self.assertEqual("PROGRESS_REFRESHED", result["operation_state"])
        self.assertEqual(31, result["observed"]["completed_units"])
        self.assertEqual("outcome_blind_scene_extraction", result["observed"]["stage"])
        self.assertFalse(result["observed"]["cached"])
        self.assertNotIn("metric", json.dumps(result))
        self.assertNotIn("secret", json.dumps(result))
        remote.assert_called_once()
        with OPS.locked_record("receipt", receipt) as record:
            self.assertEqual(1, record["finish_calls"])
            self.assertEqual(1, record["operator_observation_calls"])

    def test_progress_only_rate_limit_returns_explicit_cached_snapshot(self):
        snapshot = {
            "snapshot_at_unix": int(time.time()), "active": True,
            "terminal": False, "heartbeat_age_seconds": 3,
            "heartbeat_source": "heartbeat", "stage": "extract",
            "workload_progress": {"completed_units": 4, "total_units": 10},
        }
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": context(), "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": int(time.time()) - 10,
            },
            {
                "launched": True, "finish_calls": 0, "finish_closed": None,
                "operator_observation_calls": 1,
                "operator_observation_not_before_unix": int(time.time()) + 10,
                "operator_observation_cache": snapshot,
            },
        )
        with patch.object(OPS, "run_remote") as remote:
            result = payload(OPS.tool_finish({
                "receipt": receipt, "observation_mode": "progress_only",
            }))
        self.assertEqual("PROGRESS_REFRESH_CACHED", result["operation_state"])
        self.assertTrue(result["observed"]["cached"])
        self.assertFalse(result["observed"]["current_health_claimed"])
        remote.assert_not_called()

    def test_progress_terminal_probe_unlocks_formal_finish_before_eta(self):
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": context(), "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": int(time.time()) - 10,
            },
            {
                "launched": True, "finish_calls": 1, "finish_closed": None,
                "finish_not_before_unix": int(time.time()) + 3600,
                "pending_finish_response": OPS.typed_result(
                    True, "RUNNING_VERIFIED", receipt="sealed",
                ),
            },
        )
        probe = (
            f"CONVIR_OPS_OPERATOR_OBSERVATION snapshot_at={int(time.time())} "
            "active=false terminal=true heartbeat_age=1 heartbeat_source=status\n"
            "CONVIR_OPS_STATUS_BEGIN\nCONVIR_OPS_STATUS_END\n"
        )
        with patch.object(OPS, "validate_operator_context"), patch.object(
            OPS, "run_remote", return_value=probe,
        ):
            result = payload(OPS.tool_finish({
                "receipt": receipt, "observation_mode": "progress_only",
            }))
        self.assertEqual("TERMINAL_DETECTED", result["operation_state"])
        self.assertNotIn("decision", json.dumps(result["observed"]))
        with OPS.locked_record("receipt", receipt) as record:
            self.assertEqual(0, record["finish_not_before_unix"])
            self.assertIsNone(record["pending_finish_response"])
            self.assertTrue(record["operator_terminal_detected"])
        raw = json.dumps({
            "route_id": "a1x", "run_id": "a1x-s0-r1",
            "route_commit": "a" * 40, "runner_sha256": "e" * 64,
            **terminal(),
        }, separators=(",", ":")).encode()
        finish_output = (
            "CONVIR_OPS_MONITOR polls=1 active=false terminal=true stale=false "
            "heartbeat_age=1 heartbeat_source=status\n"
            "CONVIR_OPS_STATUS_BEGIN\nCONVIR_OPS_STATUS_END\n"
            "CONVIR_OPS_CLOSEOUT_SHA256="
            + __import__("hashlib").sha256(raw).hexdigest()
            + "\nCONVIR_OPS_CLOSEOUT_BEGIN\n" + raw.decode()
            + "\nCONVIR_OPS_CLOSEOUT_END\n"
        )
        with patch.object(OPS, "run_remote", return_value=finish_output):
            finished = payload(OPS.tool_finish({"receipt": receipt}))
        self.assertEqual("CLOSEOUT_VALIDATED", finished["operation_state"])

    def test_progress_only_detects_dead_session_without_waiting_for_eta(self):
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": context(), "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": int(time.time()) - 10,
            },
            {"launched": True, "finish_calls": 0, "finish_closed": None},
        )
        probe = (
            f"CONVIR_OPS_OPERATOR_OBSERVATION snapshot_at={int(time.time())} "
            "active=false terminal=false heartbeat_age=5 heartbeat_source=status\n"
            "CONVIR_OPS_STATUS_BEGIN\nCONVIR_OPS_STATUS_END\n"
        )
        with patch.object(OPS, "validate_operator_context"), patch.object(
            OPS, "run_remote", return_value=probe,
        ):
            result = payload(OPS.tool_finish({
                "receipt": receipt, "observation_mode": "progress_only",
            }))
        self.assertEqual("CLOSEOUT_MISSING", result["operation_state"])
        with OPS.locked_record("receipt", receipt) as record:
            self.assertEqual("CLOSEOUT_MISSING", record["finish_closed"])

    def test_contract_progress_parser_rejects_scientific_or_untyped_payloads(self):
        status = "\n".join((
            '{"phase":"contract","event":"contract_progress","stage":"probe","completed_iterations":4,"total_iterations":8}',
            '{"phase":"contract","event":"contract_progress","stage":"bad stage","completed_iterations":8,"total_iterations":8,"metric":0.1}',
        ))
        self.assertEqual(
            {"stage": "probe", "completed_iterations": 4, "total_iterations": 8},
            OPS.contract_progress(status),
        )

    def test_start_accepts_a1_machine_readable_progress_envelope(self):
        plan = {
            "context": context(), "issued_at": int(time.time()),
            "expires_at": int(time.time()) + 60, "nonce": "n",
        }
        token = OPS.write_new_record("plan", plan, {"receipt": None})
        monitor = (
            "CONVIR_OPS_MONITOR polls=1 active=true terminal=false stale=false "
            "heartbeat_age=1 heartbeat_source=heartbeat\n"
            "CONVIR_OPS_STATUS_BEGIN\n"
            '{"phase":"workload","event":"workload_start","completed":0,"total":1536}\n'
            '{"R3_A1_PROGRESS":{"stage":"feature_extract","completed_units":8,"total_units":1536}}\n'
            "CONVIR_OPS_STATUS_END\n"
        )
        with patch.object(OPS, "verify_live_context"), patch.object(
            OPS, "run_remote", side_effect=["CONVIR_OPS_LAUNCHED\n", monitor],
        ):
            result = payload(OPS.tool_start({"plan_token": token}))
        self.assertEqual("RUNNING_VERIFIED", result["operation_state"])
        self.assertEqual(
            {"completed_units": 8, "total_units": 1536},
            result["observed"]["workload_progress"],
        )

    def test_progress_parser_rejects_untyped_and_preserves_zero_capacity(self):
        status = "\n".join((
            '{"message":"completed_units","completed_units":99,"total_units":100}',
            '{"route_progress":{"completed_units":50,"total_units":100}}',
            '{"R3_A2_PROGRESS":{"completed_units":0,"total_units":10}}',
        ))
        self.assertEqual(
            {"completed_units": 0, "total_units": 10}, OPS.workload_progress(status)
        )

    def test_progress_stage_prefers_workload_after_contract_completion(self):
        status = "\n".join((
            '{"phase":"contract","event":"contract_progress","stage":"probe",'
            '"completed_iterations":100,"total_iterations":100}',
            '{"phase":"workload","event":"workload_start",'
            '"completed_units":0,"total_units":851}',
        ))
        self.assertEqual("workload", OPS.progress_stage(status))
        status += (
            '\n{"phase":"workload","event":"workload_progress",'
            '"stage":"scene_extract","completed_units":1,"total_units":851}'
        )
        self.assertEqual("scene_extract", OPS.progress_stage(status))

    def test_start_surfaces_early_engineering_failure_and_auto_authorizes_repair(self):
        plan = {
            "context": context(), "issued_at": int(time.time()),
            "expires_at": int(time.time()) + 60, "nonce": "n",
        }
        token = OPS.write_new_record("plan", plan, {"receipt": None})
        raw = json.dumps({
            "route_id": "a1x", "run_id": "a1x-s0-r1",
            "route_commit": "a" * 40, "runner_sha256": "e" * 64,
            **engineering_terminal(), "failure_phase": "asset_preflight", "returncode": 1,
            "details": {"error_type": "LifecycleError", "error_message": "asset mismatch"},
        }, separators=(",", ":")).encode()
        monitor = (
            "CONVIR_OPS_MONITOR polls=1 active=false terminal=true stale=false "
            "heartbeat_age=1 heartbeat_source=heartbeat\n"
            "CONVIR_OPS_STATUS_BEGIN\nfailed\nCONVIR_OPS_STATUS_END\n"
            "CONVIR_OPS_CLOSEOUT_SHA256=" + __import__("hashlib").sha256(raw).hexdigest()
            + "\nCONVIR_OPS_CLOSEOUT_BEGIN\n" + raw.decode() + "\nCONVIR_OPS_CLOSEOUT_END\n"
        )
        with patch.object(OPS, "verify_live_context"), patch.object(
            OPS, "run_remote", side_effect=["CONVIR_OPS_LAUNCHED\n", monitor],
        ):
            result = payload(OPS.tool_start({"plan_token": token}))
        self.assertEqual("ENGINEERING_AUTO_REPAIR_AUTHORIZED", result["operation_state"])
        self.assertFalse(result["ok"])
        self.assertEqual("asset_preflight", result["failure_phase"])
        self.assertIn("receipt", result)

    def test_stale_heartbeat_does_not_block_later_closeout_validation(self):
        ctx = context()
        receipt_payload = {
            "context": ctx, "gpu_index": None,
            "launch_digest": "f" * 64, "issued_at": int(time.time()) - 300,
        }
        receipt = OPS.write_new_record(
            "receipt", receipt_payload,
            {
                "launched": True, "finish_calls": 0, "finish_closed": None,
                "monitor_stale_count": 0,
            },
        )
        stale = (
            "CONVIR_OPS_MONITOR polls=1 active=true terminal=false stale=true "
            "heartbeat_age=300 heartbeat_source=heartbeat\n"
            "CONVIR_OPS_STATUS_BEGIN\nwork\nCONVIR_OPS_STATUS_END\n"
        )
        raw = json.dumps({
            "route_id": "a1x", "run_id": "a1x-s0-r1",
            "route_commit": "a" * 40, "runner_sha256": "e" * 64,
            **terminal(),
        }, separators=(",", ":")).encode()
        complete = (
            "CONVIR_OPS_MONITOR polls=1 active=false terminal=true stale=false "
            "heartbeat_age=301 heartbeat_source=heartbeat\n"
            "CONVIR_OPS_STATUS_BEGIN\ndone\nCONVIR_OPS_STATUS_END\n"
            + "CONVIR_OPS_CLOSEOUT_SHA256=" + __import__("hashlib").sha256(raw).hexdigest()
            + "\nCONVIR_OPS_CLOSEOUT_BEGIN\n" + raw.decode() + "\nCONVIR_OPS_CLOSEOUT_END\n"
        )
        with patch.object(OPS, "run_remote", side_effect=[stale, complete]):
            first = payload(OPS.tool_finish({"receipt": receipt}))
            second = payload(OPS.tool_finish({"receipt": receipt}))
        self.assertEqual("MONITOR_STALE", first["operation_state"])
        self.assertTrue(first["receipt_remains_open"])
        self.assertEqual("CLOSEOUT_VALIDATED", second["operation_state"])

    def test_engineering_closeout_requires_explicit_resolution_before_evidence(self):
        ctx = context()
        receipt_payload = {
            "context": ctx, "gpu_index": None,
            "launch_digest": "f" * 64, "issued_at": int(time.time()),
        }
        receipt = OPS.write_new_record(
            "receipt", receipt_payload,
            {
                "launched": True, "finish_calls": 0, "finish_closed": None,
                "monitor_stale_count": 0, "terminal_closeout": None,
                "engineering_failure_resolution": None,
            },
        )
        raw = json.dumps({
            "route_id": "a1x", "run_id": "a1x-s0-r1",
            "route_commit": "a" * 40, "runner_sha256": "e" * 64,
            **engineering_terminal(), "failure_phase": "workload", "returncode": 1,
            "verified_assets": [{
                "id": "metadata", "kind": "file", "sha256": "1" * 64,
                "path": "/must/not/be/returned",
            }],
            "details": {"error_type": "LifecycleError", "error_message": "run program failed rc=124"},
        }, separators=(",", ":")).encode()
        complete = (
            "CONVIR_OPS_MONITOR polls=1 active=false terminal=true stale=false "
            "heartbeat_age=0 heartbeat_source=heartbeat\n"
            "CONVIR_OPS_STATUS_BEGIN\nfailed\nCONVIR_OPS_STATUS_END\n"
            + "CONVIR_OPS_CLOSEOUT_SHA256=" + __import__("hashlib").sha256(raw).hexdigest()
            + "\nCONVIR_OPS_CLOSEOUT_BEGIN\n" + raw.decode() + "\nCONVIR_OPS_CLOSEOUT_END\n"
        )
        with patch.object(OPS, "run_remote", return_value=complete):
            stopped = payload(OPS.tool_finish({"receipt": receipt}))
        self.assertEqual("ENGINEERING_AUTO_REPAIR_AUTHORIZED", stopped["operation_state"])
        self.assertFalse(stopped["ok"])
        self.assertFalse(stopped["archive_authorized"])
        self.assertFalse(stopped["relaunch_authorized"])
        self.assertEqual("workload", stopped["failure_phase"])
        self.assertEqual(
            "metadata",
            stopped["observed"]["closeout"]["engineering_diagnostic"]["verified_assets"][0]["id"],
        )
        self.assertNotIn(
            "path",
            stopped["observed"]["closeout"]["engineering_diagnostic"]["verified_assets"][0],
        )
        blocked = payload(OPS.tool_evidence_manifest({"receipt": receipt}))
        self.assertEqual("EVIDENCE_MANIFEST_FAILED", blocked["operation_state"])
        self.assertEqual("engineering_runtime", blocked["failure_class"])

        repair = payload(OPS.tool_finish({
            "receipt": receipt, "engineering_failure_resolution": "repair",
        }))
        self.assertEqual("ENGINEERING_AUTO_REPAIR_AUTHORIZED", repair["operation_state"])
        self.assertFalse(repair["archive_authorized"])
        blocked_after_repair = payload(OPS.tool_evidence_manifest({"receipt": receipt}))
        self.assertEqual("EVIDENCE_MANIFEST_FAILED", blocked_after_repair["operation_state"])

    def test_engineering_archive_resolution_unlocks_compact_evidence(self):
        receipt_payload = {
            "context": context(), "gpu_index": None,
            "launch_digest": "f" * 64, "issued_at": 1,
        }
        closeout = {
            "identity": {}, "terminal_tuple": engineering_terminal(),
            "closeout_sha256": "1" * 64, "closeout_filename": "s0_closeout.json",
            "engineering_diagnostic": {"failure_phase": "workload"},
        }
        receipt = OPS.write_new_record(
            "receipt", receipt_payload,
            {
                "launched": True, "finish_calls": 1,
                "finish_closed": "ENGINEERING_REVIEW_REQUIRED",
                "monitor_stale_count": 0, "terminal_closeout": closeout,
                "engineering_failure_resolution": None,
            },
        )
        archived = payload(OPS.tool_finish({
            "receipt": receipt, "engineering_failure_resolution": "archive",
        }))
        self.assertEqual("ENGINEERING_ARCHIVE_AUTHORIZED", archived["operation_state"])
        self.assertTrue(archived["archive_authorized"])
        remote = "README.md\t12\t" + "a" * 64 + "\nCONVIR_OPS_EVIDENCE_MANIFEST_OK\nCONVIR_REMOTE_SCRIPT_OK"
        with patch.object(OPS, "run_remote", return_value=remote):
            evidence = OPS.tool_evidence_manifest({"receipt": receipt})["structuredContent"]
        self.assertEqual("README.md", evidence["files"][0]["name"])

    def test_finalization_repair_is_single_use_and_never_reexecutes_workload(self):
        source = context()
        closeout = {
            "identity": {
                "route_id": source["route_id"], "run_id": source["output_id"],
                "route_commit": source["route_branch_commit"],
                "runner_sha256": source["runner_sha256"],
            },
            "terminal_tuple": engineering_terminal(),
            "closeout_sha256": "1" * 64,
            "closeout_filename": source["closeout_filename"],
            "engineering_diagnostic": {
                "failure_phase": "evidence", "workload_started": True,
            },
        }
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": source, "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": 1,
            },
            {
                "launched": True, "finish_calls": 1,
                "finish_closed": "ENGINEERING_AUTO_REPAIR_AUTHORIZED",
                "terminal_closeout": closeout,
                "engineering_failure_resolution": "repair",
                "finalization_repair_attempted": False,
            },
        )
        final = {**source, "route_branch_commit": "f" * 40}
        raw = json.dumps({
            "route_id": source["route_id"], "run_id": source["output_id"],
            "route_commit": "f" * 40, "runner_sha256": source["runner_sha256"],
            **terminal(),
        }, separators=(",", ":")).encode()
        output = (
            "CONVIR_OPS_FINALIZATION_REPAIR rc=0\n"
            "CONVIR_OPS_CLOSEOUT_SHA256="
            + OPS.hashlib.sha256(raw).hexdigest()
            + "\nCONVIR_OPS_CLOSEOUT_BEGIN\n" + raw.decode()
            + "\nCONVIR_OPS_CLOSEOUT_END\n"
        )
        classification = {
            "status": "FINALIZATION_REPAIR_ELIGIBLE",
            "scientific_kernel_unchanged": True,
            "terminal_adapter_only": True,
        }
        with patch.object(
            OPS, "prepare_finalization_repair",
            return_value=(final, {"evidence_files": []}, classification),
        ), patch.object(OPS, "finalization_repair_body", return_value="body"), \
                patch.object(OPS, "run_remote", return_value=output):
            result = payload(OPS.tool_finish({
                "receipt": receipt,
                "engineering_failure_resolution": "finalize",
                "finalization_repair_commit": "f" * 40,
            }))
        self.assertEqual("CLOSEOUT_VALIDATED", result["operation_state"])
        self.assertFalse(result["observed"]["workload_reexecuted"])
        with OPS.locked_record("receipt", receipt) as record:
            self.assertEqual(
                source["route_branch_commit"],
                record["payload"]["context"]["route_branch_commit"],
            )
            self.assertEqual(
                final["route_branch_commit"],
                record["finalization_context"]["route_branch_commit"],
            )
        repeated = payload(OPS.tool_finish({
            "receipt": receipt,
            "engineering_failure_resolution": "finalize",
            "finalization_repair_commit": "f" * 40,
        }))
        self.assertEqual("FINISH_REJECTED", repeated["operation_state"])

    def test_finalization_repair_rejects_nonfinalization_failure(self):
        receipt = self.discardable_receipt(
            failure_phase="workload", workload_started=True,
        )
        with patch.object(OPS, "prepare_finalization_repair") as prepare:
            result = payload(OPS.tool_finish({
                "receipt": receipt,
                "engineering_failure_resolution": "finalize",
                "finalization_repair_commit": "f" * 40,
            }))
        self.assertEqual("FINISH_REJECTED", result["operation_state"])
        prepare.assert_not_called()

    def test_finalization_body_restores_original_evidence_without_new_closeout(self):
        source = context()
        final = {
            **source,
            "route_branch_commit": "f" * 40,
            "route_manifest_schema_version": 6,
        }
        spec = {
            "evidence_files": [{
                "destination_filename": "s0_review_facts.json",
            }],
            "engineering_contract": {"capability_profile_relpath": None},
        }
        body = OPS.finalization_repair_body(
            source, final, spec,
            {"closeout_sha256": "1" * 64},
            {"status": "FINALIZATION_REPAIR_ELIGIBLE"},
        )
        self.assertIn("finalization_repair_backup", body)
        self.assertIn("CONVIR_OPS_FINALIZATION_ORIGINAL_EVIDENCE_RESTORED", body)
        self.assertIn('test ! -f "$CLOSEOUT"', body)

    def test_finalization_candidate_rejection_does_not_consume_execution_slot(self):
        receipt = self.discardable_receipt(
            failure_phase="finalize", workload_started=True,
        )
        with patch.object(
            OPS, "prepare_finalization_repair",
            side_effect=OPS.ToolError("candidate rejected"),
        ):
            result = payload(OPS.tool_finish({
                "receipt": receipt,
                "engineering_failure_resolution": "finalize",
                "finalization_repair_commit": "f" * 40,
            }))
        self.assertEqual("FINISH_REJECTED", result["operation_state"])
        with OPS.locked_record("receipt", receipt) as record:
            self.assertIsNot(record.get("finalization_repair_attempted"), True)
            self.assertIsNone(record.get("engineering_failure_resolution"))

    def discardable_receipt(self, **diagnostic_overrides):
        ctx = context()
        ctx.update({
            "remote_repo": OPS.derive_remote_repo("a1x", "a1x-s0-r1"),
            "run_root": f"{OPS.REMOTE_RUNS}/a1x",
            "output_path": f"{OPS.REMOTE_RUNS}/a1x/a1x-s0-r1",
        })
        ctx["closeout_path"] = (
            f"{ctx['remote_repo']}/experience_docx/experiment_logs/a1x/s0_closeout.json"
        )
        diagnostic = {
            "failure_phase": "asset_preflight",
            "workload_started": False,
            "scientific_data_touched": False,
            "protected_data_touched": False,
        }
        diagnostic.update(diagnostic_overrides)
        closeout = {
            "identity": {
                "route_id": "a1x", "run_id": "a1x-s0-r1",
                "route_commit": "a" * 40, "runner_sha256": "e" * 64,
            },
            "terminal_tuple": engineering_terminal(),
            "closeout_sha256": "1" * 64,
            "closeout_filename": "s0_closeout.json",
            "engineering_diagnostic": diagnostic,
        }
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": ctx, "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": time.time_ns(),
            },
            {
                "launched": True, "finish_calls": 1,
                "finish_closed": "ENGINEERING_REVIEW_REQUIRED",
                "monitor_stale_count": 0, "terminal_closeout": closeout,
                "engineering_failure_resolution": None,
            },
        )
        return receipt

    def test_receipt_bound_engineering_discard_requires_verified_no_data_touch(self):
        receipt = self.discardable_receipt()
        with patch.object(OPS, "run_remote", return_value="CONVIR_OPS_ENGINEERING_DISCARD_OK") as remote:
            result = payload(OPS.tool_finish({
                "receipt": receipt, "engineering_failure_resolution": "discard",
            }))
        self.assertEqual("ENGINEERING_DISCARDED", result["operation_state"])
        self.assertTrue(result["observed"]["receipt_bound"])
        self.assertTrue(result["observed"]["postcheck"]["remote_route_workspace_absent"])
        self.assertIn("EXPECTED_CLOSEOUT_SHA", remote.call_args.args[0])

        for field in ("scientific_data_touched", "protected_data_touched"):
            receipt = self.discardable_receipt(**{field: True})
            with patch.object(OPS, "run_remote") as remote:
                result = payload(OPS.tool_finish({
                    "receipt": receipt, "engineering_failure_resolution": "discard",
                }))
            self.assertEqual("FINISH_REJECTED", result["operation_state"])
            remote.assert_not_called()

    def test_engineering_discard_rejects_unknown_touch_state_and_path_tamper(self):
        receipt = self.discardable_receipt(scientific_data_touched=None)
        with patch.object(OPS, "run_remote") as remote:
            result = payload(OPS.tool_finish({
                "receipt": receipt, "engineering_failure_resolution": "discard",
            }))
        self.assertEqual("FINISH_REJECTED", result["operation_state"])
        remote.assert_not_called()

        receipt = self.discardable_receipt()
        with OPS.locked_record("receipt", receipt) as record:
            record["payload"]["context"]["remote_repo"] = OPS.CLOUD_GIT_SEED
        with patch.object(OPS, "run_remote") as remote:
            result = payload(OPS.tool_finish({
                "receipt": receipt, "engineering_failure_resolution": "discard",
            }))
        self.assertEqual("FINISH_REJECTED", result["operation_state"])
        remote.assert_not_called()

    def test_scientific_terminal_cannot_be_discarded(self):
        receipt = self.discardable_receipt()
        with OPS.locked_record("receipt", receipt) as record:
            record["terminal_closeout"]["terminal_tuple"] = terminal()
        with patch.object(OPS, "run_remote") as remote:
            result = payload(OPS.tool_finish({
                "receipt": receipt, "engineering_failure_resolution": "discard",
            }))
        self.assertEqual("FINISH_REJECTED", result["operation_state"])
        remote.assert_not_called()

    def test_engineering_diagnostic_is_bounded_redacted_and_control_only(self):
        raw = json.dumps({
            "route_id": "a1x", "run_id": "a1x-s0-r1",
            "route_commit": "a" * 40, "runner_sha256": "e" * 64,
            **engineering_terminal(), "failure_phase": "workload", "returncode": 1,
            "verified_assets": [],
            "details": {
                "error_type": "LifecycleError",
                "error_message": "/sda/home/private/model.py token=supersecret failed",
                "traceback_tail": ("/home/private/file.py password=hunter2\n" * 400),
                "workload_started": True,
                "scientific_data_touched": False,
                "protected_data_touched": False,
            },
        }, separators=(",", ":")).encode()
        output = (
            "CONVIR_OPS_CLOSEOUT_SHA256=" + __import__("hashlib").sha256(raw).hexdigest()
            + "\nCONVIR_OPS_CLOSEOUT_BEGIN\n" + raw.decode() + "\nCONVIR_OPS_CLOSEOUT_END\n"
        )
        parsed = OPS.parse_closeout(context(), output)["engineering_diagnostic"]
        rendered = json.dumps(parsed)
        self.assertNotIn("supersecret", rendered)
        self.assertNotIn("hunter2", rendered)
        self.assertNotIn("/sda/home", rendered)
        self.assertNotIn("/home/private", rendered)
        self.assertLessEqual(len(parsed["traceback_tail"].encode()), 4096)
        status = OPS.safe_status_summary(
            '{"phase":"workload","event":"workload_progress","completed_units":1,'
            '"metric":99.9,"image":"secret.png"}'
        )
        self.assertIn("completed_units", status)
        self.assertNotIn("metric", status)
        self.assertNotIn("secret.png", status)

    def test_auto_migrated_engineering_archive_can_be_reopened_for_repair(self):
        closeout = {
            "identity": {}, "terminal_tuple": engineering_terminal(),
            "closeout_sha256": "1" * 64, "closeout_filename": "s0_closeout.json",
            "engineering_diagnostic": {"failure_phase": "workload"},
        }
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": context(), "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": 1,
            },
            {
                "launched": True, "finish_calls": 1,
                "finish_closed": "ENGINEERING_ARCHIVE_AUTHORIZED",
                "monitor_stale_count": 0, "terminal_closeout": closeout,
                "engineering_failure_resolution": "archive",
                "v43_migrated_at": 123,
            },
        )
        reopened = payload(OPS.tool_finish({
            "receipt": receipt, "engineering_failure_resolution": "repair",
        }))
        self.assertEqual("ENGINEERING_REPAIR_AUTHORIZED", reopened["operation_state"])
        self.assertTrue(reopened["observed"]["migrated_archive_reopened"])
        self.assertFalse(reopened["archive_authorized"])
        blocked = payload(OPS.tool_evidence_manifest({"receipt": receipt}))
        self.assertEqual("EVIDENCE_MANIFEST_FAILED", blocked["operation_state"])

    def test_explicit_engineering_archive_cannot_be_reopened_for_repair(self):
        closeout = {
            "identity": {}, "terminal_tuple": engineering_terminal(),
            "closeout_sha256": "1" * 64, "closeout_filename": "s0_closeout.json",
            "engineering_diagnostic": {"failure_phase": "workload"},
        }
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": context(), "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": 1,
            },
            {
                "launched": True, "finish_calls": 1,
                "finish_closed": "ENGINEERING_ARCHIVE_AUTHORIZED",
                "monitor_stale_count": 0, "terminal_closeout": closeout,
                "engineering_failure_resolution": "archive",
            },
        )
        rejected = payload(OPS.tool_finish({
            "receipt": receipt, "engineering_failure_resolution": "repair",
        }))
        self.assertEqual("FINISH_REJECTED", rejected["operation_state"])

    def test_monitor_prefers_heartbeat_then_status_then_launch_age(self):
        body = OPS.monitor_body({**context(), "_receipt_issued_at": 123}, {"max_polls": 1, "interval_seconds": 0})
        self.assertIn('test -f "$HEARTBEAT"', body)
        self.assertIn('elif test -f "$STATUS"', body)
        self.assertIn("LAUNCHED_AT=123", body)
        parsed = OPS.parse_monitor(
            "CONVIR_OPS_MONITOR polls=1 active=true terminal=false stale=false "
            "heartbeat_age=4 heartbeat_source=heartbeat\n"
            "CONVIR_OPS_STATUS_BEGIN\nok\nCONVIR_OPS_STATUS_END\n"
        )
        self.assertEqual("heartbeat", parsed["heartbeat_source"])

    def test_closeout_requires_receipt_identity_and_allowed_tuple(self):
        ctx = context()
        raw = json.dumps({
            "route_id": "a1x", "run_id": "a1x-s0-r1", "route_commit": "a" * 40,
            "runner_sha256": "e" * 64, **terminal(),
        }, separators=(",", ":")).encode()
        output = "CONVIR_OPS_CLOSEOUT_SHA256=" + __import__("hashlib").sha256(raw).hexdigest() + "\nCONVIR_OPS_CLOSEOUT_BEGIN\n" + raw.decode() + "\nCONVIR_OPS_CLOSEOUT_END"
        self.assertEqual(terminal(), OPS.parse_closeout(ctx, output)["terminal_tuple"])
        with self.assertRaises(OPS.ToolError):
            OPS.parse_closeout(ctx, output.replace('"run_id":"a1x-s0-r1"', '"run_id":"other"'))

    def test_operator_cancellation_is_receipt_bound_typed_and_idempotent(self):
        request_id = "1" * 32
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": context(), "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": int(time.time()) - 10,
            },
            {
                "launched": True, "finish_calls": 1, "finish_closed": None,
                "operator_cancel_attempts": 0,
                "operator_cancel_request_id": request_id,
            },
        )
        raw = json.dumps({
            "route_id": "a1x", "run_id": "a1x-s0-r1",
            "route_commit": "a" * 40, "runner_sha256": "e" * 64,
            **cancelled_terminal(),
            "details": {
                "request_id": request_id, "requested_at_unix": 123,
                "completed_units": 37, "total_units": 851,
                "stage": "outcome_blind_scene_extraction",
                "termination_mode": "graceful",
                "scientific_result_interpretable": False,
            },
        }, separators=(",", ":")).encode()
        output = (
            f"CONVIR_OPS_CANCEL snapshot_at={int(time.time())} active=false "
            "terminal=true mode=graceful\n"
            "CONVIR_OPS_CLOSEOUT_SHA256="
            + __import__("hashlib").sha256(raw).hexdigest()
            + "\nCONVIR_OPS_CLOSEOUT_BEGIN\n" + raw.decode()
            + "\nCONVIR_OPS_CLOSEOUT_END\n"
        )
        with patch.object(OPS, "validate_operator_context"), patch.object(
            OPS, "run_remote", return_value=output,
        ) as remote:
            first = payload(OPS.tool_finish({
                "receipt": receipt, "operator_action": "cancel",
            }))
            second = payload(OPS.tool_finish({
                "receipt": receipt, "operator_action": "cancel",
            }))
        self.assertEqual("CANCELLED_BY_OPERATOR", first["operation_state"])
        self.assertEqual(37, first["observed"]["completed_units"])
        self.assertFalse(first["observed"]["scientific_result_interpretable"])
        self.assertTrue(second["observed"]["cached"])
        self.assertEqual("NONE", first["scientific_authorization"])
        remote.assert_called_once()
        blocked = payload(OPS.tool_evidence_manifest({"receipt": receipt}))
        self.assertEqual("EVIDENCE_MANIFEST_FAILED", blocked["operation_state"])

    def test_operator_cancellation_rejects_request_or_receipt_mismatch(self):
        ctx = context()
        raw = json.dumps({
            "route_id": "a1x", "run_id": "other",
            "route_commit": "a" * 40, "runner_sha256": "e" * 64,
            **cancelled_terminal(),
            "details": {"request_id": "1" * 32},
        }, separators=(",", ":")).encode()
        output = (
            "CONVIR_OPS_CLOSEOUT_SHA256="
            + __import__("hashlib").sha256(raw).hexdigest()
            + "\nCONVIR_OPS_CLOSEOUT_BEGIN\n" + raw.decode()
            + "\nCONVIR_OPS_CLOSEOUT_END\n"
        )
        with self.assertRaises(OPS.ToolError):
            OPS.parse_closeout(ctx, output)

        raw = json.dumps({
            "route_id": "a1x", "run_id": "a1x-s0-r1",
            "route_commit": "a" * 40, "runner_sha256": "e" * 64,
            **cancelled_terminal(),
            "details": {"request_id": "2" * 32},
        }, separators=(",", ":")).encode()
        output = (
            f"CONVIR_OPS_CANCEL snapshot_at={int(time.time())} active=false "
            "terminal=true mode=forced\nCONVIR_OPS_CLOSEOUT_SHA256="
            + __import__("hashlib").sha256(raw).hexdigest()
            + "\nCONVIR_OPS_CLOSEOUT_BEGIN\n" + raw.decode()
            + "\nCONVIR_OPS_CLOSEOUT_END\n"
        )
        receipt = OPS.write_new_record(
            "receipt",
            {"context": ctx, "gpu_index": None, "launch_digest": "f" * 64, "issued_at": 1},
            {
                "launched": True, "finish_closed": None,
                "operator_cancel_request_id": "1" * 32,
                "operator_cancel_attempts": 0,
            },
        )
        with patch.object(OPS, "validate_operator_context"), patch.object(
            OPS, "run_remote", return_value=output,
        ):
            result = payload(OPS.tool_finish({
                "receipt": receipt, "operator_action": "cancel",
            }))
        self.assertEqual("FINISH_REJECTED", result["operation_state"])
        self.assertEqual("evidence", result["failure_class"])

    def test_operator_control_has_no_pid_input_and_actions_are_exclusive(self):
        body_context = context()
        with patch.object(OPS, "validate_operator_context"):
            body = OPS.operator_cancel_body(body_context, "1" * 32)
        self.assertIn("lifecycle_identity.json", body)
        self.assertIn("EXPECTED_ROUTE_COMMIT", body)
        self.assertIn("RUNNER_SHA256", body)
        self.assertIn("operator_cancel_request.json", body)
        self.assertIn("SIGTERM", body)
        self.assertIn("SIGKILL", body)

        receipt = OPS.write_new_record(
            "receipt",
            {"context": context(), "gpu_index": None, "launch_digest": "f" * 64, "issued_at": 1},
            {"launched": True, "finish_closed": None},
        )
        result = payload(OPS.tool_finish({
            "receipt": receipt, "operator_action": "cancel",
            "observation_mode": "progress_only",
        }))
        self.assertEqual("FINISH_REJECTED", result["operation_state"])

    def test_engineering_closeout_allows_null_decision_but_cannot_authorize_next_stage(self):
        failure = {"state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE"}
        self.assertEqual(failure, OPS.require_terminal_tuples([failure])[0])
        with self.assertRaises(OPS.ToolError):
            OPS.require_terminal_tuple(failure, "prior_terminal_tuple")

    def test_engineering_failure_class_tracks_phase_without_changing_review_gate(self):
        self.assertEqual("preflight_resource", OPS.engineering_failure_class("asset_preflight"))
        self.assertEqual("engineering_runtime", OPS.engineering_failure_class("workload"))
        self.assertEqual("evidence_closeout", OPS.engineering_failure_class("evidence"))

    def test_evidence_tools_resolve_workspace_only_from_receipt(self):
        receipt_payload = {"context": context(), "gpu_index": None, "launch_digest": "f" * 64, "issued_at": 1}
        receipt = OPS.write_new_record(
            "receipt", receipt_payload,
            {
                "launched": True, "finish_closed": "CLOSEOUT_VALIDATED",
                "terminal_closeout": closeout_binding(),
            },
        )
        remote = "README.md\t12\t" + "a" * 64 + "\nCONVIR_OPS_EVIDENCE_MANIFEST_OK\nCONVIR_REMOTE_SCRIPT_OK"
        with patch.object(OPS, "run_remote", return_value=remote):
            result = OPS.tool_evidence_manifest({"receipt": receipt})["structuredContent"]
        self.assertEqual("README.md", result["files"][0]["name"])

    def test_schema6_scientific_evidence_returns_canonical_archive_contract(self):
        scientific_context = context()
        scientific_context["route_manifest_schema_version"] = 6
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": scientific_context, "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": 1,
            },
            {
                "launched": True, "finish_closed": "CLOSEOUT_VALIDATED",
                "terminal_closeout": closeout_binding(),
            },
        )
        remote = "README.md\t12\t" + "a" * 64 + "\nCONVIR_OPS_EVIDENCE_MANIFEST_OK\nCONVIR_REMOTE_SCRIPT_OK"
        with patch.object(OPS, "run_remote", return_value=remote):
            result = OPS.tool_evidence_manifest({"receipt": receipt})["structuredContent"]
        contract = result["archive_contract"]
        self.assertEqual(
            "experience_docx/experiment_logs/a1x/s0_conclusion.json",
            contract["conclusion_path"],
        )
        self.assertEqual(3, contract["conclusion_schema_version"])
        self.assertIn("primary_result", contract["required_conclusion_fields"])
        self.assertIn("gate_fact_ids", contract["required_conclusion_fields"])

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            destination = repo / "experience_docx/experiment_logs/a1x"
            destination.mkdir(parents=True)
            evidence = b"review data\n"
            (destination / "README.md").write_bytes(evidence)
            remote = (
                f"README.md\t{len(evidence)}\t"
                f"{OPS.hashlib.sha256(evidence).hexdigest()}\n"
                "CONVIR_OPS_EVIDENCE_MANIFEST_OK\nCONVIR_REMOTE_SCRIPT_OK"
            )
            with (
                patch.object(OPS, "run_remote", return_value=remote),
                patch.object(OPS, "validate_local_repo", return_value=repo),
            ):
                fetched = OPS.tool_evidence_fetch({
                    "receipt": receipt, "local_repo": str(repo),
                    "files": ["README.md"],
                })["structuredContent"]
            self.assertEqual(contract, fetched["archive_contract"])

    def test_schema6_terminal_returns_ordered_decide_archive_handoff(self):
        scientific_context = context()
        scientific_context["route_manifest_schema_version"] = 6
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": scientific_context, "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": 1,
            },
            {"launched": True, "finish_closed": None},
        )
        closeout = closeout_binding()
        result = payload(OPS.validated_scientific_result(
            receipt, closeout, {"closeout": closeout},
        ))
        expected = [
            "convir_evidence_list",
            "convir_evidence_fetch",
            "author_scientific_conclusion",
            "prepare_terminal_archive",
        ]
        self.assertEqual(expected, result["allowed_next_actions"])
        self.assertEqual(expected, result["required_action_sequence"])
        self.assertFalse(result["archive_ready"])
        self.assertEqual(
            "experience_docx/experiment_logs/a1x/s0_conclusion.json",
            result["archive_contract"]["conclusion_path"],
        )
        self.assertEqual("CLOSEOUT_VALIDATED", result["operation_state"])

    def test_engineering_archive_evidence_has_no_scientific_archive_contract(self):
        engineering_context = context()
        engineering_context["route_manifest_schema_version"] = 6
        binding = closeout_binding()
        binding["terminal_tuple"] = engineering_terminal()
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": engineering_context, "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": 1,
            },
            {
                "launched": True,
                "finish_closed": "ENGINEERING_ARCHIVE_AUTHORIZED",
                "terminal_closeout": binding,
            },
        )
        remote = "README.md\t12\t" + "a" * 64 + "\nCONVIR_OPS_EVIDENCE_MANIFEST_OK\nCONVIR_REMOTE_SCRIPT_OK"
        with patch.object(OPS, "run_remote", return_value=remote):
            result = OPS.tool_evidence_manifest({"receipt": receipt})["structuredContent"]
        self.assertNotIn("archive_contract", result)

    def test_record_tampering_is_rejected(self):
        plan = {"context": context(), "issued_at": 1, "expires_at": 2, "nonce": "n"}
        token = OPS.write_new_record("plan", plan, {"receipt": None})
        path = OPS.record_path("plan", token)
        value = json.loads(path.read_text())
        value["payload"]["nonce"] = "changed"
        path.write_text(json.dumps(value))
        with self.assertRaises(OPS.ToolError):
            with OPS.locked_record("plan", token):
                pass

    def test_mutable_record_state_tampering_is_rejected(self):
        token = OPS.write_new_record(
            "receipt",
            {"context": context(), "issued_at": 1, "launch_digest": "f" * 64},
            {"launched": True, "finish_closed": None},
        )
        path = OPS.record_path("receipt", token)
        value = json.loads(path.read_text())
        value["finish_closed"] = "CLOSEOUT_VALIDATED"
        path.write_text(json.dumps(value))
        path.chmod(0o600)
        with self.assertRaises(OPS.ToolError):
            with OPS.locked_record("receipt", token):
                pass

    def test_state_directory_mode_and_key_symlink_fail_closed(self):
        Path(self.state.name).chmod(0o755)
        with self.assertRaises(OPS.ToolError):
            OPS.state_secret()
        Path(self.state.name).chmod(0o700)
        target = Path(self.state.name).parent / "external-key"
        target.write_bytes(b"x" * 32)
        target.chmod(0o600)
        key = OPS.STATE_DIR / "hmac.key"
        key.symlink_to(target)
        try:
            with self.assertRaises(OPS.ToolError):
                OPS.state_secret()
        finally:
            target.unlink(missing_ok=True)

    def test_evidence_manifest_rejects_symlinks_and_binds_closeout_sha(self):
        body = OPS.evidence_manifest_body({
            **context(),
            "evidence_dir": "/remote/a1x/experience_docx/experiment_logs/a1x",
            "validated_closeout_filename": "s0_closeout.json",
            "validated_closeout_sha256": "1" * 64,
        })
        self.assertIn('test ! -L "$EVIDENCE_DIR"', body)
        self.assertIn('test ! -L "$VALIDATED_CLOSEOUT"', body)
        self.assertIn("1" * 64, body)

    def test_inline_evidence_delivery_has_no_local_filesystem_side_effect(self):
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": context(), "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": 1,
            },
            {
                "launched": True, "finish_closed": "CLOSEOUT_VALIDATED",
                "terminal_closeout": closeout_binding(),
            },
        )
        raw = b"review data\n"
        digest = OPS.hashlib.sha256(raw).hexdigest()
        manifest = (
            f"README.md\t{len(raw)}\t{digest}\n"
            "CONVIR_OPS_EVIDENCE_MANIFEST_OK\n"
        )
        page = json.dumps({
            "name": "README.md", "content": raw.decode(), "offset": 0,
            "next_offset": len(raw), "size_bytes": len(raw),
            "file_sha256": digest, "content_sha256": digest, "complete": True,
        }, separators=(",", ":"))
        inline = (
            "CONVIR_OPS_EVIDENCE_INLINE_BEGIN\n" + page
            + "\nCONVIR_OPS_EVIDENCE_INLINE_END\n"
        )
        receipt_path = OPS.record_path("receipt", receipt)
        receipt_before = receipt_path.read_bytes()
        with patch.object(OPS, "run_remote", side_effect=[manifest, inline]), \
                patch.object(OPS, "validate_local_repo") as local_repo:
            result = OPS.tool_evidence_fetch({
                "receipt": receipt, "files": ["README.md"],
                "delivery": "inline",
            })["structuredContent"]
        local_repo.assert_not_called()
        self.assertEqual("review data\n", result["files"][0]["content"])
        self.assertTrue(result["complete"])
        self.assertFalse(result["filesystem_mutations_performed"])
        self.assertEqual(receipt_before, receipt_path.read_bytes())
        self.assertEqual([], list(OPS.STATE_DIR.glob("evidence-cursor-*.json")))

    def test_inline_continuation_is_stateless_and_idempotent(self):
        receipt = OPS.write_new_record(
            "receipt",
            {
                "context": context(), "gpu_index": None,
                "launch_digest": "f" * 64, "issued_at": 1,
            },
            {
                "launched": True, "finish_closed": "CLOSEOUT_VALIDATED",
                "terminal_closeout": closeout_binding(),
            },
        )
        raw = b"abc"
        digest = OPS.hashlib.sha256(raw).hexdigest()
        manifest = (
            f"README.md\t{len(raw)}\t{digest}\n"
            "CONVIR_OPS_EVIDENCE_MANIFEST_OK\n"
        )

        def page(content, offset, complete):
            encoded = content.encode()
            value = json.dumps({
                "name": "README.md", "content": content, "offset": offset,
                "next_offset": offset + len(encoded), "size_bytes": len(raw),
                "file_sha256": digest,
                "content_sha256": OPS.hashlib.sha256(encoded).hexdigest(),
                "complete": complete,
            }, separators=(",", ":"))
            return (
                "CONVIR_OPS_EVIDENCE_INLINE_BEGIN\n" + value
                + "\nCONVIR_OPS_EVIDENCE_INLINE_END\n"
            )

        with patch.object(
            OPS, "run_remote",
            side_effect=[
                manifest, page("a", 0, False),
                manifest, page("bc", 1, True),
                manifest, page("bc", 1, True),
            ],
        ):
            first = OPS.tool_evidence_fetch({
                "receipt": receipt, "files": ["README.md"], "delivery": "inline",
            })["structuredContent"]
            request = {
                "receipt": receipt, "files": ["README.md"], "delivery": "inline",
                "continuation": first["continuation"],
            }
            second = OPS.tool_evidence_fetch(request)["structuredContent"]
            replay = OPS.tool_evidence_fetch(request)["structuredContent"]
        self.assertFalse(first["complete"])
        self.assertEqual(second, replay)
        self.assertEqual("bc", second["files"][0]["content"])
        self.assertTrue(second["complete"])
        self.assertEqual([], list(OPS.STATE_DIR.glob("evidence-cursor-*.json")))

    def test_inline_evidence_rejects_ambiguous_marker_framing(self):
        with self.assertRaisesRegex(OPS.ToolError, "framing"):
            OPS.parse_inline_evidence(
                "CONVIR_OPS_EVIDENCE_INLINE_BEGIN\n{}\n"
                "CONVIR_OPS_EVIDENCE_INLINE_END\n"
                "CONVIR_OPS_EVIDENCE_INLINE_END\n"
            )

    def test_local_evidence_destination_rejects_symlink_chain(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
            outside = Path(root) / "outside"
            repo.mkdir()
            outside.mkdir()
            (repo / "experience_docx").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(OPS.ToolError):
                OPS.ensure_real_directory_chain(
                    repo, Path("experience_docx") / "experiment_logs" / "a1x",
                )

    def test_stdio_exposes_exact_six_schema_v4_tools(self):
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ]
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH)], input="".join(json.dumps(item) + "\n" for item in requests),
            text=True, capture_output=True, check=True, timeout=10,
        )
        responses = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual("5.8.0", responses[0]["result"]["serverInfo"]["version"])
        tools = responses[1]["result"]["tools"]
        self.assertEqual(6, len(tools))
        evidence = next(item for item in tools if item["name"] == "convir_evidence_list")
        self.assertEqual(["receipt"], evidence["inputSchema"]["required"])
        plan = next(item for item in tools if item["name"] == "convir_route_plan")
        self.assertEqual(4, plan["inputSchema"]["properties"]["schema_version"]["const"])
        finish = next(item for item in tools if item["name"] == "convir_route_finish")
        self.assertEqual(
            ["repair", "archive", "discard", "finalize"],
            finish["inputSchema"]["properties"]["engineering_failure_resolution"]["enum"],
        )
        self.assertEqual(
            ["observe", "cancel"],
            finish["inputSchema"]["properties"]["operator_action"]["enum"],
        )
        self.assertEqual(
            ["sealed", "progress_only"],
            finish["inputSchema"]["properties"]["observation_mode"]["enum"],
        )
        self.assertNotIn("pid", finish["inputSchema"]["properties"])
        fetch = next(item for item in tools if item["name"] == "convir_evidence_fetch")
        self.assertEqual(["receipt", "files"], fetch["inputSchema"]["required"])
        self.assertEqual(
            ["inline", "materialize"],
            fetch["inputSchema"]["properties"]["delivery"]["enum"],
        )
        status = next(item for item in tools if item["name"] == "convir_git_status")
        self.assertEqual([], status["inputSchema"]["required"])
        self.assertEqual(
            ["project", "route"],
            status["inputSchema"]["properties"]["scope"]["enum"],
        )


if __name__ == "__main__":
    unittest.main()
