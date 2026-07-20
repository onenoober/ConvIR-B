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
    }


class ConvirOpsV4Tests(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.TemporaryDirectory()
        OPS.STATE_DIR = Path(self.state.name)

    def tearDown(self):
        self.state.cleanup()

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

    def test_monitor_profiles_are_bounded_to_sixty_seconds(self):
        for profile in OPS.MONITOR_PROFILES.values():
            self.assertLessEqual(profile["max_polls"] * profile["interval_seconds"], 60)

    def test_unknown_start_recovery_bodies_have_valid_bash_syntax(self):
        for body in (
            OPS.atomic_start_body(context(), None),
            OPS.unknown_start_inspection_body(context()),
            OPS.abandoned_start_cleanup_body(context()),
        ):
            completed = subprocess.run(
                ["/bin/bash", "-n"], input=body, text=True,
                capture_output=True, timeout=10,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("clone --quiet --shared --no-checkout", OPS.atomic_start_body(context(), None))

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
        error = OPS.ToolError("no gpu", failure_phase="resource_preflight", failure_class="command_infra")
        with patch.object(OPS, "verify_live_context"), patch.object(OPS, "run_remote", side_effect=error):
            first = payload(OPS.tool_start({"plan_token": token}))
            second = payload(OPS.tool_start({"plan_token": token}))
        self.assertEqual("RESOURCE_WAIT_REQUIRED", first["operation_state"])
        self.assertEqual("RESOURCE_WAIT_REQUIRED", second["operation_state"])

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

    def test_progress_parser_rejects_untyped_or_zero_progress(self):
        status = "\n".join((
            '{"message":"completed_units","completed_units":99,"total_units":100}',
            '{"route_progress":{"completed_units":50,"total_units":100}}',
            '{"R3_A2_PROGRESS":{"completed_units":0,"total_units":10}}',
        ))
        self.assertEqual(
            {"completed_units": 0, "total_units": 0}, OPS.workload_progress(status)
        )

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
            {"launched": True, "finish_closed": "CLOSEOUT_VALIDATED"},
        )
        remote = "README.md\t12\t" + "a" * 64 + "\nCONVIR_OPS_EVIDENCE_MANIFEST_OK\nCONVIR_REMOTE_SCRIPT_OK"
        with patch.object(OPS, "run_remote", return_value=remote):
            result = OPS.tool_evidence_manifest({"receipt": receipt})["structuredContent"]
        self.assertEqual("README.md", result["files"][0]["name"])

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
        self.assertEqual("5.0.0", responses[0]["result"]["serverInfo"]["version"])
        tools = responses[1]["result"]["tools"]
        self.assertEqual(6, len(tools))
        evidence = next(item for item in tools if item["name"] == "convir_evidence_list")
        self.assertEqual(["receipt"], evidence["inputSchema"]["required"])
        plan = next(item for item in tools if item["name"] == "convir_route_plan")
        self.assertEqual(4, plan["inputSchema"]["properties"]["schema_version"]["const"])
        finish = next(item for item in tools if item["name"] == "convir_route_finish")
        self.assertEqual(
            ["repair", "archive"],
            finish["inputSchema"]["properties"]["engineering_failure_resolution"]["enum"],
        )


if __name__ == "__main__":
    unittest.main()
