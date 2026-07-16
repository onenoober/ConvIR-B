"""Mocked-transport tests for the minimal convir-ops schema-v3 lifecycle."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "convir_ops_mcp.py"
SPEC = importlib.util.spec_from_file_location("convir_ops_mcp", MODULE_PATH)
OPS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPS)


def payload(result):
    return result["structuredContent"]


def terminal(decision="PASS", authorizes="formal"):
    return {"state": "COMPLETED_GATE_PASS", "decision": decision, "authorizes": authorizes}


def operation(**overrides):
    value = {
        "runner_relpath": "experience_docx/tools/run_a1x.sh",
        "mode": "s0",
        "require_gpu": False,
        "stage_state": "PLANNED",
        "decision": "START",
        "authorizes": "s0",
        "locked_test_policy": "blocked",
        "forbidden_continuations": ["locked_test"],
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


def manifest(op=None):
    return {
        "schema_version": 3,
        "route_id": "a1x",
        "repo_name": "convir-a1x",
        "workspace_id": "a1x-v2",
        "rules_commit": "b" * 40,
        "rules_digest": "d" * 64,
        "route_card_relpath": "experience_docx/experiment_cards/a1x.md",
        "route_card_blob": "c" * 40,
        "operations": {"S0": op or operation()},
    }


def context(require_gpu=False):
    return {
        "schema_version": 3,
        "branch": "codex/a1x",
        "route_branch_commit": "a" * 40,
        "current_rules_commit": "b" * 40,
        "route_id": "a1x",
        "repo_name": "convir-a1x",
        "workspace_id": "a1x-v2",
        "remote_repo": "/remote/a1x",
        "run_root": "/runs/a1x",
        "route_card_relpath": "experience_docx/experiment_cards/a1x.md",
        "route_card_blob": "c" * 40,
        "rules_commit": "b" * 40,
        "rules_digest": "d" * 64,
        "runner_relpath": "experience_docx/tools/run_a1x.sh",
        "runner_sha256": "e" * 64,
        "mode": "s0",
        "require_gpu": require_gpu,
        "stage_state": "PLANNED",
        "decision": "START",
        "authorizes": "s0",
        "locked_test_policy": "blocked",
        "forbidden_continuations": ["locked_test"],
        "output_id": "a1x-s0-r1",
        "output_path": "/runs/a1x/a1x-s0-r1",
        "closeout_filename": "s0_closeout.json",
        "closeout_path": "/remote/a1x/experience_docx/experiment_logs/a1x/s0_closeout.json",
        "prior_closeout_relpath": None,
        "prior_terminal_tuple": None,
        "allowed_terminal_tuples": [terminal()],
        "workspace_policy": "fresh_route",
        "output_policy": "new",
        "monitor_profile": "short",
        "heartbeat_timeout_seconds": 120,
        "min_free_gpu_mib": 12000 if require_gpu else 0,
        "max_gpu_utilization_pct": 10 if require_gpu else 100,
        "session": "convir-a1x-s0",
    }


class ConvirOpsV3Tests(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.TemporaryDirectory()
        OPS.STATE_DIR = Path(self.state.name)

    def tearDown(self):
        self.state.cleanup()

    def parse(self, value=None, operation_id="S0"):
        with (
            patch.object(OPS, "git_show", side_effect=lambda _repo, _commit, path: "First authorized stage: S0\n" if path.endswith("a1x.md") else "runner"),
            patch.object(OPS, "git_show_bytes", return_value=b"runner\n"),
            patch.object(OPS, "blob_sha", return_value="c" * 40),
            patch.object(OPS, "rule_bundle_digest", return_value="d" * 64),
        ):
            return OPS.parse_manifest(value or manifest(), "codex/a1x", "a" * 40, "b" * 40, "/tmp/repo", operation_id)

    def test_first_operation_is_authorized_by_frozen_card(self):
        value = self.parse()
        self.assertIsNone(value["prior_closeout_relpath"])
        self.assertEqual("d" * 64, value["rules_digest"])
        self.assertEqual(__import__("hashlib").sha256(b"runner\n").hexdigest(), value["runner_sha256"])

    def test_later_operation_requires_exact_prior_closeout(self):
        prior = terminal("S0_PASS", "formal")
        op = operation(
            mode="formal", prior_closeout_relpath="experience_docx/experiment_logs/a1x/s0_closeout.json",
            prior_terminal_tuple=prior, stage_state=prior["state"], decision=prior["decision"],
            authorizes=prior["authorizes"], output_id="a1x-formal-r1",
        )
        value = manifest(op)
        with (
            patch.object(OPS, "git_show", side_effect=lambda _repo, _commit, path: json.dumps({"route_id": "a1x", **prior}) if path.endswith("s0_closeout.json") else ("First authorized stage: S0\n" if path.endswith("a1x.md") else "runner")),
            patch.object(OPS, "git_show_bytes", return_value=b"runner\n"),
            patch.object(OPS, "blob_sha", return_value="c" * 40),
            patch.object(OPS, "rule_bundle_digest", return_value="d" * 64),
        ):
            parsed = OPS.parse_manifest(value, "codex/a1x", "a" * 40, "b" * 40, "/tmp/repo", "S0")
        self.assertEqual(prior, parsed["prior_terminal_tuple"])

    def test_current_rule_bundle_must_match_recorded_digest(self):
        with (
            patch.object(OPS, "git_show", side_effect=lambda _repo, _commit, path: "First authorized stage: S0\n" if path.endswith("a1x.md") else "runner"),
            patch.object(OPS, "git_show_bytes", return_value=b"runner\n"),
            patch.object(OPS, "blob_sha", return_value="c" * 40),
            patch.object(OPS, "rule_bundle_digest", side_effect=["d" * 64, "f" * 64]),
        ):
            with self.assertRaises(OPS.ToolError):
                OPS.parse_manifest(manifest(), "codex/a1x", "a" * 40, "b" * 40, "/tmp/repo", "S0")

    def test_monitor_profiles_are_bounded_to_sixty_seconds(self):
        for profile in OPS.MONITOR_PROFILES.values():
            self.assertLessEqual(profile["max_polls"] * profile["interval_seconds"], 60)

    def test_launch_timeout_opens_unknown_state_without_blind_retry(self):
        plan = {"context": context(), "issued_at": int(time.time()), "expires_at": int(time.time()) + 60, "nonce": "n"}
        token = OPS.write_new_record("plan", plan, {"receipt": None})
        error = OPS.ToolError("timeout", failure_phase="launch_command", failure_class="command_infra")
        with patch.object(OPS, "verify_live_context"), patch.object(OPS, "run_remote", side_effect=error):
            first = payload(OPS.tool_start_authorized({"plan_token": token}))
            second = payload(OPS.tool_start_authorized({"plan_token": token}))
        self.assertEqual("START_STATE_UNKNOWN", first["operation_state"])
        self.assertEqual("START_STATE_UNKNOWN", second["operation_state"])

    def test_resource_wait_is_retryable_before_launch_attempt(self):
        plan = {"context": context(require_gpu=True), "issued_at": int(time.time()), "expires_at": int(time.time()) + 60, "nonce": "n"}
        token = OPS.write_new_record("plan", plan, {"receipt": None})
        error = OPS.ToolError("no gpu", failure_phase="resource_preflight", failure_class="command_infra")
        with patch.object(OPS, "verify_live_context"), patch.object(OPS, "run_remote", side_effect=error):
            first = payload(OPS.tool_start_authorized({"plan_token": token}))
            second = payload(OPS.tool_start_authorized({"plan_token": token}))
        self.assertEqual("RESOURCE_WAIT_REQUIRED", first["operation_state"])
        self.assertEqual("RESOURCE_WAIT_REQUIRED", second["operation_state"])

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

    def test_evidence_tools_resolve_workspace_only_from_receipt(self):
        receipt_payload = {"context": context(), "gpu_index": None, "launch_digest": "f" * 64, "issued_at": 1}
        receipt = OPS.write_new_record("receipt", receipt_payload, {"launched": True})
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

    def test_stdio_exposes_exact_six_schema_v3_tools(self):
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ]
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH)], input="".join(json.dumps(item) + "\n" for item in requests),
            text=True, capture_output=True, check=True, timeout=10,
        )
        responses = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual("3.0.0", responses[0]["result"]["serverInfo"]["version"])
        tools = responses[1]["result"]["tools"]
        self.assertEqual(6, len(tools))
        evidence = next(item for item in tools if item["name"] == "convir_evidence_manifest")
        self.assertEqual(["receipt"], evidence["inputSchema"]["required"])
        plan = next(item for item in tools if item["name"] == "convir_route_plan_manifest")
        self.assertEqual(3, plan["inputSchema"]["properties"]["schema_version"]["const"])


if __name__ == "__main__":
    unittest.main()
