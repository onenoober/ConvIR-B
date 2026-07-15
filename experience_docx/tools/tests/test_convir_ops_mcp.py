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


def closeout_payload(**overrides):
    value = {
        "route_id": "a0r",
        "run_id": "a0r-r2",
        "route_commit": "a" * 40,
        "runner_sha256": "a" * 64,
        **A0R_TERMINAL_TUPLE,
    }
    value.update(overrides)
    return value


def load_fresh_ops_module():
    spec = importlib.util.spec_from_file_location("convir_ops_mcp_reload", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConvirOpsLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.receipts = tempfile.TemporaryDirectory()
        OPS.RECEIPT_DIR = Path(self.receipts.name)
        self.args = {"schema_version": 2, "route_id": "a0r", "repo_name": "convir", "workspace_id": "a0r-workspace", "workspace_policy": "fresh_route", "branch": "codex/a0r", "route_branch_commit": "a" * 40, "rules_commit": "b" * 40, "runner_relpath": "experience_docx/tools/run_a0r.sh", "mode": "audit", "require_gpu": False, "min_free_gpu_mib": 0, "max_gpu_utilization_pct": 100, "monitor_profile": "short", "heartbeat_timeout_seconds": 120, "stage_state": "COMPLETED_GATE_PASS", "decision": "V4A_A0R_REPRODUCTION_PASS_AUTHORIZE_A0D_AND_A0P", "authorizes": "A0D_AND_A0P_ONLY", "locked_test_policy": "blocked", "forbidden_continuations": ["locked_test"], "output_id": "a0r-r2", "closeout_filename": "a0r_closeout.json", "collision_policy": "must_not_exist", "authorization_relpath": "experience_docx/experiment_logs/a0r/prior.json", "prior_terminal_tuple": A0R_TERMINAL_TUPLE, "allowed_terminal_tuples": [A0R_TERMINAL_TUPLE]}

    def tearDown(self):
        self.receipts.cleanup()

    def prepare(self, transport=None):
        def default_transport(body, **_kwargs):
            return "a" * 64 + "  experience_docx/tools/run_a0r.sh\nCONVIR_OPS_PREFLIGHT_OK"
        transport = transport or default_transport
        checks = {"branch": "codex/a0r", "route_commit": "a" * 40, "rules_commit": "b" * 40, "runner_relpath": self.args["runner_relpath"]}
        with patch.object(OPS, "verify_github_context", return_value=checks), patch.object(OPS, "run_remote_body", side_effect=transport):
            plan = payload(OPS.tool_prepare_authorized({**self.args, "phase": "plan"}))
            return payload(OPS.tool_prepare_authorized({**self.args, "phase": "apply", "plan_hash": plan["expected"]["plan_hash"]}))

    def plan_full(self, arguments=None):
        checks = {"branch": "codex/a0r", "route_commit": "a" * 40, "rules_commit": "b" * 40, "runner_relpath": self.args["runner_relpath"]}
        with patch.object(OPS, "verify_github_context", return_value=checks):
            return payload(OPS.tool_prepare_authorized({**(arguments or self.args), "phase": "plan"}))

    def test_a0r_command_infra_failure_requires_fresh_corrected_receipt(self):
        failed = self.prepare(lambda body, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("mock", 1)))
        corrected = self.prepare()
        with patch.object(OPS, "run_remote_body", return_value="noise\nCONVIR_OPS_CLOSEOUT_SHA256=" + "a" * 64 + "\nCONVIR_OPS_CLOSEOUT_JSON_BEGIN\n" + json.dumps(closeout_payload()) + "\nCONVIR_OPS_CLOSEOUT_JSON_END\nwrapper"):
            validated = payload(OPS.tool_closeout_validate({"receipt": corrected["receipt"]}))
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
        self.assertIn("a0r_closeout.json", remote.call_args.args[0])

    def test_tuple_mismatch_is_not_a_closeout_pass(self):
        prepared = self.prepare()
        closeout = {"state": "COMPLETED_GATE_PASS", "decision": "wrong", "authorizes": "A0D_AND_A0P_ONLY"}
        with patch.object(OPS, "run_remote_body", return_value="CONVIR_OPS_CLOSEOUT_SHA256=" + "a" * 64 + "\nCONVIR_OPS_CLOSEOUT_JSON_BEGIN\n" + json.dumps(closeout_payload(**closeout)) + "\nCONVIR_OPS_CLOSEOUT_JSON_END"):
            result = payload(OPS.tool_closeout_validate({"receipt": prepared["receipt"]}))
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

    def test_plan_token_reloads_across_process_before_composed_start(self):
        plan = self.plan_full()
        reloaded = load_fresh_ops_module()
        reloaded.RECEIPT_DIR = OPS.RECEIPT_DIR
        with patch.object(reloaded, "run_remote_body", return_value="a" * 64 + "  experience_docx/tools/run_a0r.sh\nCONVIR_OPS_PREFLIGHT_OK\nCONVIR_OPS_LAUNCH_OK"):
            started = payload(reloaded.tool_start_authorized({
                "plan_token": plan["plan_token"],
            }))
        self.assertTrue(started["ok"])
        self.assertEqual("LAUNCHED", started["operation_state"])

    def test_manifest_plan_loads_exact_operation_with_short_request(self):
        operation_fields = {
            "runner_relpath", "mode", "require_gpu", "stage_state", "decision", "authorizes",
            "locked_test_policy", "forbidden_continuations", "output_id", "closeout_filename",
            "collision_policy", "authorization_relpath", "prior_terminal_tuple", "allowed_terminal_tuples",
            "workspace_policy", "monitor_profile", "heartbeat_timeout_seconds", "min_free_gpu_mib",
            "max_gpu_utilization_pct",
        }
        operation = {key: self.args.get(key, True) for key in operation_fields}
        manifest = {
            "schema_version": 2, "route_id": self.args["route_id"], "repo_name": self.args["repo_name"],
            "workspace_id": self.args["workspace_id"], "rules_commit": self.args["rules_commit"], "operations": {"A0R": operation},
        }
        short = {
            "schema_version": 2, "branch": self.args["branch"],
            "route_branch_commit": self.args["route_branch_commit"],
            "operation_id": "A0R",
        }
        def local_git(args, **_kwargs):
            return json.dumps(manifest) if "show" in args else ""
        refs = {
            "refs/heads/main": self.args["rules_commit"],
            "refs/heads/codex/a0r": self.args["route_branch_commit"],
        }
        with patch.object(OPS, "_run_local", side_effect=local_git), patch.object(OPS, "github_ref_shas", return_value=refs) as remote_refs, patch.object(OPS, "run_remote_body") as remote:
            planned = payload(OPS.tool_plan_manifest(short))
        self.assertTrue(planned["ok"])
        remote.assert_not_called()
        remote_refs.assert_called_once_with(["refs/heads/codex/a0r", "refs/heads/main"])
        self.assertEqual("A0R", planned["observed"]["operation_id"])
        self.assertLess(len(json.dumps(short)), len(json.dumps({**self.args, "phase": "plan"})))

    def test_safe_arbitrary_mode_and_bounded_session_are_sealed(self):
        args = {**self.args, "route_id": "r" * 80, "mode": "repair.v2"}
        planned = self.plan_full(args)
        next_stage = self.plan_full({**args, "output_id": "a0d"})
        self.assertTrue(planned["ok"])
        context = planned["observed"]
        self.assertEqual("repair.v2", context["mode"])
        self.assertLessEqual(len(context["session"]), 64)
        self.assertRegex(context["session"], r"^convir-")
        self.assertNotEqual(context["session"], next_stage["observed"]["session"])
        self.assertEqual(context["remote_repo"], next_stage["observed"]["remote_repo"])

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
        plan = self.plan_full()
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
        plan = self.plan_full()
        self.assertIn("github_checks", plan["observed"])
        with patch.object(OPS, "run_remote_body", return_value="a" * 64 + "  runner\nCONVIR_OPS_PREFLIGHT_OK"):
            prepared = payload(OPS.tool_prepare_authorized({**self.args, "phase": "apply", "plan_hash": plan["expected"]["plan_hash"]}))
        receipt = json.loads((OPS.RECEIPT_DIR / (prepared["receipt"] + ".json")).read_text())["payload"]
        self.assertEqual(plan["observed"]["remote_repo"], receipt["remote_repo"])

    def test_compact_monitor_uses_sealed_terminal_tuples_and_budgeted_timeout(self):
        prepared = self.prepare()
        response = "CONVIR_OPS_MONITOR_META polls=3 active=false terminal=true stale=false heartbeat_age=1\nCONVIR_OPS_MONITOR_STATUS_BEGIN\nstate=COMPLETED_GATE_PASS decision=V4A_A0R_REPRODUCTION_PASS_AUTHORIZE_A0D_AND_A0P authorizes=A0D_AND_A0P_ONLY\nCONVIR_OPS_MONITOR_STATUS_END\nCONVIR_REMOTE_SCRIPT_OK"
        with patch.object(OPS, "run_remote_body", return_value=response) as remote:
            result = payload(OPS.tool_receipt_monitor({"receipt": prepared["receipt"], "monitor_mode": "until_terminal", "max_polls": 3, "interval_seconds": 2}))
        self.assertEqual("MONITOR_TERMINAL", result["operation_state"])
        self.assertEqual(3, result["observed"]["poll_count"])
        self.assertNotIn("CONVIR_OPS_MONITOR_OK", result["observed"]["status"])
        self.assertNotIn("CONVIR_REMOTE_SCRIPT_OK", result["observed"]["status"])
        self.assertGreater(remote.call_args.kwargs["timeout"], 6)

    def test_finish_rejects_stale_or_missing_closeout_instead_of_polling_forever(self):
        prepared = self.prepare()
        stale_output = "CONVIR_OPS_MONITOR_META polls=2 active=true terminal=false stale=true heartbeat_age=121\nCONVIR_OPS_MONITOR_STATUS_BEGIN\nrunning\nCONVIR_OPS_MONITOR_STATUS_END"
        with patch.object(OPS, "run_remote_body", return_value=stale_output):
            stale = payload(OPS.tool_finish({"receipt": prepared["receipt"]}))
        self.assertEqual("MONITOR_STALE", stale["operation_state"])
        self.assertEqual("command_infra", stale["failure_class"])

        missing_output = "CONVIR_OPS_MONITOR_META polls=1 active=false terminal=false stale=false heartbeat_age=2\nCONVIR_OPS_MONITOR_STATUS_BEGIN\nfailed\nCONVIR_OPS_MONITOR_STATUS_END"
        with patch.object(OPS, "run_remote_body", return_value=missing_output):
            missing = payload(OPS.tool_finish({"receipt": prepared["receipt"]}))
        self.assertEqual("CLOSEOUT_MISSING", missing["operation_state"])
        self.assertEqual("evaluation", missing["failure_class"])
        self.assertIn("session_created", OPS.monitor_body(json.loads((OPS.RECEIPT_DIR / (prepared["receipt"] + ".json")).read_text())["payload"], 1, 0))

    def test_gpu_thresholds_are_sealed_and_rechecked_on_the_same_gpu(self):
        gpu_args = {**self.args, "require_gpu": True, "min_free_gpu_mib": 12000, "max_gpu_utilization_pct": 5}
        plan = self.plan_full(gpu_args)
        preflight = "a" * 64 + "  experience_docx/tools/run_a0r.sh\nCONVIR_OPS_GPU_OK index=2 min_free_mib=12000 max_util_pct=5\nCONVIR_OPS_PREFLIGHT_OK"
        with patch.object(OPS, "run_remote_body", side_effect=["CONVIR_OPS_GPU_OK index=2 min_free_mib=12000 max_util_pct=5\nCONVIR_OPS_RESOURCE_OK", preflight + "\nCONVIR_OPS_LAUNCH_OK"]) as remote:
            started = payload(OPS.tool_start_authorized({"plan_token": plan["plan_token"]}))
        self.assertTrue(started["ok"])
        self.assertIn('nvidia-smi -i "$GPU_INDEX"', remote.call_args_list[1].args[0])
        self.assertIn("GPU_ATTEMPTS=2", remote.call_args_list[1].args[0])

    def test_exact_continuation_never_clones_or_cleans_existing_workspace(self):
        continuation = {**self.args, "workspace_policy": "exact_continuation"}
        body = OPS.preflight_body(OPS.authorization_context(continuation), False, create_clone=True)
        self.assertIn('test -d "$REMOTE_REPO/.git"', body)
        self.assertIn("merge --quiet --ff-only", body)
        self.assertIn("refs/remotes/github/main", body)
        self.assertNotIn("git clone", body)
        self.assertNotIn("rm -rf", body)

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
        response = "CONVIR_OPS_MONITOR_META polls=1 active=false terminal=false stale=false heartbeat_age=1\nCONVIR_OPS_MONITOR_STATUS_BEGIN\nrunning\nCONVIR_OPS_MONITOR_STATUS_END"
        with patch.object(OPS, "run_remote_body", return_value=response):
            result = payload(OPS.tool_receipt_monitor({"receipt": new_token}))
        self.assertTrue(result["ok"])
        self.assertEqual("running", result["observed"]["status"])

    def test_closeout_manifest_hashes_raw_file_bytes(self):
        prepared = self.prepare()
        raw = (json.dumps(closeout_payload(), separators=(",", ":")) + "\n").encode()
        expected_hash = __import__("hashlib").sha256(raw).hexdigest()
        output = "CONVIR_OPS_CLOSEOUT_SHA256=" + expected_hash + "\nCONVIR_OPS_CLOSEOUT_JSON_BEGIN\n" + raw.decode() + "CONVIR_OPS_CLOSEOUT_JSON_END"
        with patch.object(OPS, "run_remote_body", return_value=output):
            result = payload(OPS.tool_closeout_validate({"receipt": prepared["receipt"]}))
        self.assertEqual(expected_hash, result["manifest"]["closeout_sha256"])

    def test_closeout_provenance_mismatch_is_evidence_failure(self):
        prepared = self.prepare()
        wrong = closeout_payload(route_commit="c" * 40, runner_sha256="d" * 64)
        output = "CONVIR_OPS_CLOSEOUT_SHA256=" + "a" * 64 + "\nCONVIR_OPS_CLOSEOUT_JSON_BEGIN\n" + json.dumps(wrong) + "\nCONVIR_OPS_CLOSEOUT_JSON_END"
        with patch.object(OPS, "run_remote_body", return_value=output):
            rejected = payload(OPS.tool_closeout_validate({"receipt": prepared["receipt"]}))
        self.assertEqual("CLOSEOUT_INVALID", rejected["operation_state"])
        self.assertEqual("evidence", rejected["failure_class"])
        self.assertNotEqual(rejected["observed"], rejected["expected"])

    def test_composed_start_and_finish_preserve_receipt_boundaries(self):
        plan = self.plan_full()
        with patch.object(OPS, "run_remote_body", return_value="a" * 64 + "  experience_docx/tools/run_a0r.sh\nCONVIR_OPS_PREFLIGHT_OK\nCONVIR_OPS_LAUNCH_OK") as remote:
            started = payload(OPS.tool_start_authorized({
                "plan_token": plan["plan_token"],
            }))
            repeated = payload(OPS.tool_start_authorized({"plan_token": plan["plan_token"]}))
        self.assertTrue(started["ok"])
        self.assertEqual(1, remote.call_count)
        self.assertEqual("LAUNCH_IDEMPOTENT", repeated["operation_state"])
        closeout = closeout_payload()
        finish_output = "CONVIR_OPS_MONITOR_META polls=2 active=false terminal=true stale=false heartbeat_age=1\nCONVIR_OPS_MONITOR_STATUS_BEGIN\nV4A_A0R_OK\nCONVIR_OPS_MONITOR_STATUS_END\nCONVIR_OPS_CLOSEOUT_SHA256=" + "b" * 64 + "\nCONVIR_OPS_CLOSEOUT_JSON_BEGIN\n" + json.dumps(closeout) + "\nCONVIR_OPS_CLOSEOUT_JSON_END"
        with patch.object(OPS, "run_remote_body", return_value=finish_output) as remote:
            finished = payload(OPS.tool_finish({"receipt": started["receipt"]}))
        self.assertTrue(finished["ok"])
        self.assertEqual("CLOSEOUT_VALIDATED", finished["operation_state"])
        self.assertEqual(1, remote.call_count)
        self.assertEqual(2, finished["observed"]["monitor"]["poll_count"])

    def test_stale_plan_and_unsealed_tuple_are_rejected(self):
        plan = self.plan_full()
        stale = payload(OPS.tool_prepare_authorized({**self.args, "phase": "apply", "plan_hash": "0" * 64}))
        prepared = self.prepare()
        wrong = closeout_payload(decision="other")
        output = "CONVIR_OPS_CLOSEOUT_SHA256=" + "a" * 64 + "\nCONVIR_OPS_CLOSEOUT_JSON_BEGIN\n" + json.dumps(wrong) + "\nCONVIR_OPS_CLOSEOUT_JSON_END"
        with patch.object(OPS, "run_remote_body", return_value=output):
            rejected = payload(OPS.tool_closeout_validate({"receipt": prepared["receipt"]}))
        self.assertEqual(64, len(plan["expected"]["plan_hash"]))
        self.assertFalse(stale["ok"])
        self.assertEqual("evidence", rejected["failure_class"])

    def test_collision_timeout_and_forbidden_input_are_typed_failures(self):
        collision = self.prepare(lambda body, **_kwargs: (_ for _ in ()).throw(OPS.ToolError("CONVIR_OPS_SESSION_CONFLICT")))
        timeout = self.prepare(lambda body, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("mock", 1)))
        forbidden = self.plan_full({**self.args, "runner_relpath": "../arbitrary.sh"})
        self.assertEqual("collision", collision["failure_class"])
        self.assertEqual("command_infra", timeout["failure_class"])
        self.assertEqual("authorization", forbidden["failure_class"])

    def test_failure_phase_is_preserved_for_resource_and_transport_errors(self):
        self.assertEqual("resource_preflight", OPS.failure_phase_for_error(OPS.ToolError("phase=resource_preflight failed")))
        error = OPS.ToolError("resource unavailable", failure_phase="resource_preflight", failure_class="command_infra")
        self.assertEqual("command_infra", OPS.failure_class_for_error(error))
        self.assertEqual("resource_preflight", OPS.failure_phase_for_error(error))

    def test_monitor_body_avoids_nested_python_command_substitution(self):
        prepared = self.prepare()
        record = json.loads((OPS.RECEIPT_DIR / (prepared["receipt"] + ".json")).read_text())
        body = OPS.monitor_body(record["payload"], 1, 0)
        self.assertIn('monitor_tmp=$(mktemp -d)', body)
        self.assertIn('<<\'PY\'', body)
        self.assertNotIn('allowed=json.loads(sys.argv[1]); raw=sys.argv[2]', body)

    def test_manifest_parser_requires_marker_and_accepts_empty_manifest(self):
        self.assertEqual({}, OPS.parse_manifest("CONVIR_OPS_EVIDENCE_MANIFEST_OK\nCONVIR_REMOTE_SCRIPT_OK\n"))
        with self.assertRaises(OPS.ToolError):
            OPS.parse_manifest("")
        with self.assertRaises(OPS.ToolError):
            OPS.parse_manifest("bad\nCONVIR_OPS_EVIDENCE_MANIFEST_OK\n")

    def test_manifest_body_filters_raw_and_cloud_only_files(self):
        context = OPS.route_context({"route_id": "a0r", "repo_name": "convir"})
        body = OPS.manifest_body(context)
        self.assertIn('test -f "$path" || continue', body)
        self.assertIn('case "${name,,}" in *cloud_only*) continue ;; esac', body)
        self.assertIn("CONVIR_OPS_EVIDENCE_MANIFEST_OK", body)

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
            "convir_route_start_authorized", "convir_route_finish", "convir_route_plan_manifest",
            "convir_evidence_manifest", "convir_evidence_fetch",
            "convir_git_evidence_status",
        }, names)
        finish = next(tool for tool in responses[1]["result"]["tools"] if tool["name"] == "convir_route_finish")
        self.assertEqual(["receipt"], finish["inputSchema"]["required"])
        self.assertEqual({"receipt"}, set(finish["inputSchema"]["properties"]))


if __name__ == "__main__":
    unittest.main()
