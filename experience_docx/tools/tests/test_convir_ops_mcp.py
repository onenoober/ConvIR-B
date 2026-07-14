"""Deterministic mocked-transport coverage for the convir-ops P0 lifecycle."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "convir_ops_mcp.py"
SPEC = importlib.util.spec_from_file_location("convir_ops_mcp", MODULE_PATH)
OPS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPS)

A0R_TERMINAL_TUPLE = {
    "state": "COMPLETED_GATE_PASS",
    "decision": "V4A_A0R_REPRODUCTION_PASS_AUTHORIZE_A0D_AND_A0P",
    "authorizes": "A0D_AND_A0P_ONLY",
}


def payload(result):
    return json.loads(result["content"][0]["text"])


def load_fresh_ops_module():
    spec = importlib.util.spec_from_file_location("convir_ops_mcp_reload", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConvirOpsLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.receipts = tempfile.TemporaryDirectory()
        OPS.RECEIPT_DIR = Path(self.receipts.name)
        self.args = {"schema_version": 2, "route_id": "a0r", "repo_name": "convir", "branch": "codex/a0r", "route_branch_commit": "a" * 40, "rules_commit": "b" * 40, "runner_relpath": "experience_docx/tools/run_a0r.sh", "mode": "audit", "stage_state": "COMPLETED_GATE_PASS", "decision": "V4A_A0R_REPRODUCTION_PASS_AUTHORIZE_A0D_AND_A0P", "authorizes": "A0D_AND_A0P_ONLY", "locked_test_policy": "blocked", "forbidden_continuations": ["locked_test"], "output_id": "a0r-r2", "closeout_filename": "a0r_closeout.json", "collision_policy": "must_not_exist", "authorization_relpath": "experience_docx/experiment_logs/a0r/prior.json", "prior_terminal_tuple": A0R_TERMINAL_TUPLE, "allowed_terminal_tuples": [A0R_TERMINAL_TUPLE]}

    def tearDown(self):
        self.receipts.cleanup()

    def prepare(self, transport=None):
        def default_transport(body, **_kwargs):
            if "CONVIR_OPS_PLAN_GITHUB_OK" in body:
                return "CONVIR_OPS_PLAN_GITHUB_OK"
            return "a" * 64 + "  experience_docx/tools/run_a0r.sh\nCONVIR_OPS_PREFLIGHT_OK"
        transport = transport or default_transport
        with patch.object(OPS, "run_remote_body", side_effect=transport):
            plan = payload(OPS.tool_prepare_authorized({**self.args, "phase": "plan"}))
            return payload(OPS.tool_prepare_authorized({**self.args, "phase": "apply", "plan_hash": plan["expected"]["plan_hash"]}))

    def test_a0r_command_infra_failure_requires_fresh_corrected_receipt(self):
        failed = self.prepare(lambda body, **_kwargs: "CONVIR_OPS_PLAN_GITHUB_OK" if "CONVIR_OPS_PLAN_GITHUB_OK" in body else (_ for _ in ()).throw(subprocess.TimeoutExpired("mock", 1)))
        corrected = self.prepare()
        with patch.object(OPS, "run_remote_body", return_value="noise\nCONVIR_OPS_CLOSEOUT_SHA256=" + "a" * 64 + "\nCONVIR_OPS_CLOSEOUT_JSON_BEGIN\n" + json.dumps({"route_id": "a0r", **A0R_TERMINAL_TUPLE}) + "\nCONVIR_OPS_CLOSEOUT_JSON_END\nwrapper"):
            validated = payload(OPS.tool_closeout_validate({"receipt": corrected["receipt"], "terminal_tuple": A0R_TERMINAL_TUPLE}))
        self.assertEqual("command_infra", failed["failure_class"])
        self.assertTrue(corrected["ok"])
        self.assertNotEqual(failed.get("receipt"), corrected["receipt"])
        self.assertTrue(validated["ok"])

    def test_receipt_tamper_and_reuse_are_rejected(self):
        prepared = self.prepare()
        tampered = payload(OPS.tool_receipt_launch({"receipt": "0" * 64, "idempotency_key": "launch-1"}))
        with patch.object(OPS, "run_remote_body", return_value="CONVIR_OPS_LAUNCH_OK") as remote:
            launched = payload(OPS.tool_receipt_launch({"receipt": prepared["receipt"], "idempotency_key": "launch-1"}))
            reused = payload(OPS.tool_receipt_launch({"receipt": prepared["receipt"], "idempotency_key": "launch-2"}))
        self.assertFalse(tampered["ok"])
        self.assertTrue(launched["ok"])
        self.assertFalse(reused["ok"])
        self.assertIn("RUN_ID=a0r-r2", remote.call_args.args[0])
        self.assertIn("OUTPUT_ID=a0r-r2", remote.call_args.args[0])

    def test_tuple_mismatch_is_not_a_closeout_pass(self):
        prepared = self.prepare()
        closeout = {"state": "COMPLETED_GATE_PASS", "decision": "wrong", "authorizes": "A0D_AND_A0P_ONLY"}
        with patch.object(OPS, "run_remote_body", return_value="CONVIR_OPS_CLOSEOUT_JSON_BEGIN\n" + json.dumps({"route_id": "a0r", **closeout}) + "\nCONVIR_OPS_CLOSEOUT_JSON_END"):
            result = payload(OPS.tool_closeout_validate({"receipt": prepared["receipt"], "terminal_tuple": A0R_TERMINAL_TUPLE}))
        self.assertEqual("evidence", result["failure_class"])

    def test_receipt_reloads_across_process_memory_and_derives_session(self):
        prepared = self.prepare()
        self.assertTrue((OPS.RECEIPT_DIR / (prepared["receipt"] + ".json")).is_file())
        self.assertNotIn("session", self.args)
        reloaded = load_fresh_ops_module()
        reloaded.RECEIPT_DIR = OPS.RECEIPT_DIR
        with patch.object(reloaded, "run_remote_body", return_value="CONVIR_OPS_LAUNCH_OK"):
            result = payload(reloaded.tool_receipt_launch({"receipt": prepared["receipt"], "idempotency_key": "reload-1"}))
        self.assertEqual(OPS.derive_session("a0r", "audit", "a" * 40, "a0r-r2"), result["observed"]["session"])
        self.assertLessEqual(len(result["observed"]["session"]), 64)
        self.assertIn("structuredContent", reloaded.typed_result(True, "CHECK"))

    def test_safe_arbitrary_mode_and_bounded_session_are_sealed(self):
        args = {**self.args, "route_id": "r" * 80, "mode": "repair.v2"}
        with patch.object(OPS, "run_remote_body", return_value="CONVIR_OPS_PLAN_GITHUB_OK"):
            planned = payload(OPS.tool_prepare_authorized({**args, "phase": "plan"}))
            next_stage = payload(OPS.tool_prepare_authorized({**args, "output_id": "a0d", "phase": "plan"}))
        self.assertTrue(planned["ok"])
        context = planned["observed"]["authorization_tuple"]
        self.assertEqual("repair.v2", context["mode"])
        self.assertLessEqual(len(context["session"]), 64)
        self.assertRegex(context["session"], r"^convir-")
        self.assertNotEqual(context["session"], next_stage["observed"]["authorization_tuple"]["session"])
        self.assertNotEqual(context["remote_repo"], next_stage["observed"]["authorization_tuple"]["remote_repo"])

    def test_secret_first_use_is_race_safe(self):
        secrets, failures = [], []
        def create():
            try:
                secrets.append(OPS.receipt_secret())
            except Exception as exc:
                failures.append(exc)
        workers = [threading.Thread(target=create) for _ in range(8)]
        for worker in workers: worker.start()
        for worker in workers: worker.join()
        self.assertFalse(failures)
        self.assertEqual(1, len(set(secrets)))

    def test_apply_failure_cleans_only_the_new_fresh_workspace(self):
        with patch.object(OPS, "run_remote_body", return_value="CONVIR_OPS_PLAN_GITHUB_OK"):
            plan = payload(OPS.tool_prepare_authorized({**self.args, "phase": "plan"}))
        calls = []
        def transport(body, **_kwargs):
            calls.append(body)
            if "git clone --origin github" in body:
                raise subprocess.TimeoutExpired("mock", 1)
            return "CONVIR_OPS_FRESH_WORKSPACE_CLEANED"
        with patch.object(OPS, "run_remote_body", side_effect=transport):
            result = payload(OPS.tool_prepare_authorized({**self.args, "phase": "apply", "plan_hash": plan["expected"]["plan_hash"]}))
        self.assertEqual("PREPARE_RECOVERY_REQUIRED", result["operation_state"])
        self.assertEqual("trap_after_path_reservation", result["observed"]["fresh_workspace_cleanup"])
        self.assertEqual(1, len(calls))
        self.assertIn("rm -rf -- \"$REMOTE_REPO\"", calls[0])

    def test_plan_checks_github_and_receipt_binds_fresh_repo(self):
        with patch.object(OPS, "run_remote_body", return_value="CONVIR_OPS_PLAN_GITHUB_OK"):
            plan = payload(OPS.tool_prepare_authorized({**self.args, "phase": "plan"}))
        self.assertIn("github_checks", plan["observed"])
        with patch.object(OPS, "run_remote_body", return_value="a" * 64 + "  runner\nCONVIR_OPS_PREFLIGHT_OK"):
            prepared = payload(OPS.tool_prepare_authorized({**self.args, "phase": "apply", "plan_hash": plan["expected"]["plan_hash"]}))
        receipt = json.loads((OPS.RECEIPT_DIR / (prepared["receipt"] + ".json")).read_text())["payload"]
        self.assertEqual(plan["observed"]["authorization_tuple"]["remote_repo"], receipt["remote_repo"])

    def test_compact_monitor_uses_sealed_terminal_tuples_and_budgeted_timeout(self):
        prepared = self.prepare()
        response = "CONVIR_OPS_MONITOR_META polls=3 active=false terminal=true\nCONVIR_OPS_MONITOR_STATUS_BEGIN\nstate=COMPLETED_GATE_PASS decision=V4A_A0R_REPRODUCTION_PASS_AUTHORIZE_A0D_AND_A0P authorizes=A0D_AND_A0P_ONLY\nCONVIR_OPS_MONITOR_STATUS_END\nCONVIR_REMOTE_SCRIPT_OK"
        with patch.object(OPS, "run_remote_body", return_value=response) as remote:
            result = payload(OPS.tool_receipt_monitor({"receipt": prepared["receipt"], "monitor_mode": "until_terminal", "max_polls": 3, "interval_seconds": 2}))
        self.assertEqual("MONITOR_TERMINAL", result["operation_state"])
        self.assertEqual(3, result["observed"]["poll_count"])
        self.assertNotIn("CONVIR_OPS_MONITOR_OK", result["observed"]["status"])
        self.assertNotIn("CONVIR_REMOTE_SCRIPT_OK", result["observed"]["status"])
        self.assertGreater(remote.call_args.kwargs["timeout"], 6)

    def test_launched_receipt_remains_valid_after_launch_window(self):
        prepared = self.prepare()
        receipt_path = OPS.RECEIPT_DIR / (prepared["receipt"] + ".json")
        record = json.loads(receipt_path.read_text())
        record["payload"]["expires_at"] = 0
        record["launched"] = True
        record["launch_key"] = "launch-1"
        receipt_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))
        # Re-sign the intentionally modified fixture payload.
        old_path = receipt_path
        new_token = OPS.receipt_token(record["payload"])
        new_path = OPS.RECEIPT_DIR / (new_token + ".json")
        old_path.replace(new_path)
        response = "CONVIR_OPS_MONITOR_META polls=1 active=false terminal=false\nCONVIR_OPS_MONITOR_STATUS_BEGIN\nrunning\nCONVIR_OPS_MONITOR_STATUS_END"
        with patch.object(OPS, "run_remote_body", return_value=response):
            result = payload(OPS.tool_receipt_monitor({"receipt": new_token}))
        self.assertTrue(result["ok"])
        self.assertEqual("running", result["observed"]["status"])

    def test_closeout_manifest_hashes_raw_file_bytes(self):
        prepared = self.prepare()
        raw = b'{"route_id":"a0r","state":"COMPLETED_GATE_PASS","decision":"V4A_A0R_REPRODUCTION_PASS_AUTHORIZE_A0D_AND_A0P","authorizes":"A0D_AND_A0P_ONLY"}\n'
        expected_hash = __import__("hashlib").sha256(raw).hexdigest()
        output = "CONVIR_OPS_CLOSEOUT_SHA256=" + expected_hash + "\nCONVIR_OPS_CLOSEOUT_JSON_BEGIN\n" + raw.decode() + "CONVIR_OPS_CLOSEOUT_JSON_END"
        with patch.object(OPS, "run_remote_body", return_value=output):
            result = payload(OPS.tool_closeout_validate({"receipt": prepared["receipt"], "terminal_tuple": A0R_TERMINAL_TUPLE}))
        self.assertEqual(expected_hash, result["manifest"]["closeout_sha256"])

    def test_composed_start_and_finish_preserve_receipt_boundaries(self):
        with patch.object(OPS, "run_remote_body", return_value="CONVIR_OPS_PLAN_GITHUB_OK"):
            plan = payload(OPS.tool_prepare_authorized({**self.args, "phase": "plan"}))
        with patch.object(OPS, "run_remote_body", side_effect=[
            "a" * 64 + "  experience_docx/tools/run_a0r.sh\nCONVIR_OPS_PREFLIGHT_OK",
            "CONVIR_OPS_LAUNCH_OK",
        ]) as remote:
            started = payload(OPS.tool_start_authorized({
                **self.args, "plan_hash": plan["expected"]["plan_hash"], "idempotency_key": "start-1",
            }))
        self.assertTrue(started["ok"])
        self.assertEqual(2, remote.call_count)
        closeout = {"route_id": "a0r", **A0R_TERMINAL_TUPLE}
        finish_outputs = [
            "CONVIR_OPS_MONITOR_META polls=2 active=false terminal=false\nCONVIR_OPS_MONITOR_STATUS_BEGIN\nV4A_A0R_OK\nCONVIR_OPS_MONITOR_STATUS_END",
            "CONVIR_OPS_CLOSEOUT_SHA256=" + "b" * 64 + "\nCONVIR_OPS_CLOSEOUT_JSON_BEGIN\n" + json.dumps(closeout) + "\nCONVIR_OPS_CLOSEOUT_JSON_END",
        ]
        with patch.object(OPS, "run_remote_body", side_effect=finish_outputs) as remote:
            finished = payload(OPS.tool_finish({"receipt": started["receipt"], "terminal_tuple": A0R_TERMINAL_TUPLE}))
        self.assertTrue(finished["ok"])
        self.assertEqual("CLOSEOUT_VALIDATED", finished["operation_state"])
        self.assertEqual(2, remote.call_count)
        self.assertEqual(2, finished["observed"]["monitor"]["poll_count"])

    def test_stale_plan_and_unsealed_tuple_are_rejected(self):
        with patch.object(OPS, "run_remote_body", return_value="CONVIR_OPS_PLAN_GITHUB_OK"):
            plan = payload(OPS.tool_prepare_authorized({**self.args, "phase": "plan"}))
        stale = payload(OPS.tool_prepare_authorized({**self.args, "phase": "apply", "plan_hash": "0" * 64}))
        prepared = self.prepare()
        rejected = payload(OPS.tool_closeout_validate({"receipt": prepared["receipt"], "terminal_tuple": {"state": "COMPLETED_GATE_PASS", "decision": "other", "authorizes": "A0D_AND_A0P_ONLY"}}))
        self.assertEqual(64, len(plan["expected"]["plan_hash"]))
        self.assertFalse(stale["ok"])
        self.assertEqual("authorization", rejected["failure_class"])

    def test_collision_timeout_and_forbidden_input_are_typed_failures(self):
        collision = self.prepare(lambda body, **_kwargs: "CONVIR_OPS_PLAN_GITHUB_OK" if "CONVIR_OPS_PLAN_GITHUB_OK" in body else (_ for _ in ()).throw(OPS.ToolError("CONVIR_OPS_SESSION_CONFLICT")))
        timeout = self.prepare(lambda body, **_kwargs: "CONVIR_OPS_PLAN_GITHUB_OK" if "CONVIR_OPS_PLAN_GITHUB_OK" in body else (_ for _ in ()).throw(subprocess.TimeoutExpired("mock", 1)))
        forbidden = payload(OPS.tool_prepare_authorized({**self.args, "phase": "plan", "runner_relpath": "../arbitrary.sh"}))
        self.assertEqual("collision", collision["failure_class"])
        self.assertEqual("command_infra", timeout["failure_class"])
        self.assertEqual("authorization", forbidden["failure_class"])

    def test_mcp_stdio_exposes_only_schema_v2_lifecycle_and_evidence_tools(self):
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ]
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH)],
            input="".join(json.dumps(item) + "\n" for item in requests),
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        responses = [json.loads(line) for line in completed.stdout.splitlines()]
        names = {tool["name"] for tool in responses[1]["result"]["tools"]}
        self.assertEqual({
            "convir_route_prepare_authorized", "convir_route_launch",
            "convir_route_monitor", "convir_route_closeout_validate",
            "convir_route_start_authorized", "convir_route_finish",
            "convir_evidence_manifest", "convir_evidence_fetch",
            "convir_git_evidence_status",
        }, names)
        prepare = next(tool for tool in responses[1]["result"]["tools"] if tool["name"] == "convir_route_prepare_authorized")
        self.assertIn("closeout_filename", prepare["inputSchema"]["required"])


if __name__ == "__main__":
    unittest.main()
