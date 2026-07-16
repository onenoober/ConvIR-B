#!/usr/bin/env python3
"""Run one receipt-bound CPU-only validation through the registered six tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import subprocess
import sys
from pathlib import Path


SHA40 = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_TOOLS = {
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
EVIDENCE = {
    "generic_run_monitoring_e2e_closeout.json",
    "generic_run_monitoring_e2e_summary.json",
}
ROUTE_ID = "generic_run_monitoring_validation_20260716"
OUTPUT_ID = "generic-monitor-e2e-r1"
MODE = "synthetic_e2e"
RUNNER = "experience_docx/tools/run_generic_run_monitoring_e2e_validation.sh"
REMOTE_BASE = "/sda/home/wangyuxin/ConvIR-B"


class ClientError(RuntimeError):
    pass


class RpcClient:
    def __init__(self, executable: Path, state_dir: Path):
        if not executable.is_file():
            raise ClientError("registered server executable is missing")
        if state_dir.exists():
            raise ClientError("isolated state directory already exists")
        environment = os.environ.copy()
        environment["CONVIR_OPS_STATE_DIR"] = str(state_dir)
        self.process = subprocess.Popen(
            ["/usr/bin/python3", str(executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.next_id = 1

    def call(self, method: str, params: dict, timeout: int) -> dict:
        request_id = self.next_id
        self.next_id += 1
        request = {
            "jsonrpc": "2.0", "id": request_id,
            "method": method, "params": params,
        }
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        ready = self.selector.select(timeout)
        if not ready:
            raise ClientError(f"RPC timeout: {method}")
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read(4096)
            raise ClientError(f"server closed during {method}: {stderr}")
        response = json.loads(line)
        if response.get("id") != request_id:
            raise ClientError(f"RPC id mismatch for {method}")
        if "error" in response:
            raise ClientError(f"RPC error for {method}: {response['error']}")
        return response["result"]

    def tool(self, name: str, arguments: dict, timeout: int = 360) -> dict:
        result = self.call(
            "tools/call", {"name": name, "arguments": arguments}, timeout,
        )
        if result.get("isError"):
            structured = result.get("structuredContent") or {}
            raise ClientError(
                f"{name} failed state={structured.get('operation_state')} "
                f"class={structured.get('failure_class')}"
            )
        structured = result.get("structuredContent")
        if not isinstance(structured, dict):
            raise ClientError(f"{name} returned no structured content")
        return structured

    def close(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        return_code = self.process.wait(timeout=10)
        if return_code != 0:
            stderr = self.process.stderr.read(4096)
            raise ClientError(f"server exited rc={return_code}: {stderr}")


def digest_token(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def git_output(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *arguments],
        text=True, capture_output=True, timeout=30,
    )
    if completed.returncode:
        raise ClientError(f"Git preflight failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def expected_plan_identity(route_commit: str) -> dict:
    repo_digest = hashlib.sha256(f"{ROUTE_ID}\0{OUTPUT_ID}".encode()).hexdigest()[:16]
    repo_prefix = f"{ROUTE_ID[:32]}-{OUTPUT_ID[:24]}"[:56]
    remote_repo = f"{REMOTE_BASE}/repos/{repo_prefix}-{repo_digest}"
    session_digest = hashlib.sha256(
        f"{ROUTE_ID}\0{MODE}\0{route_commit}\0{OUTPUT_ID}".encode()
    ).hexdigest()[:12]
    session = f"convir-{ROUTE_ID[:18]}-{MODE[:10]}-{OUTPUT_ID[:10]}-{session_digest}"[:64]
    return {
        "route_id": ROUTE_ID,
        "remote_repo": remote_repo,
        "output_path": f"{REMOTE_BASE}/runs/{ROUTE_ID}/{OUTPUT_ID}",
        "session": session,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--route-commit", required=True)
    parser.add_argument("--local-repo", type=Path, required=True)
    args = parser.parse_args()
    if not SHA40.fullmatch(args.route_commit):
        raise ClientError("route commit must be 40 lowercase hex")
    if not (args.local_repo / ".git").exists():
        raise ClientError("local repository is not a worktree")
    local_repo = args.local_repo.resolve()
    if git_output(local_repo, "rev-parse", "HEAD") != args.route_commit:
        raise ClientError("local route HEAD does not match the requested commit")
    if git_output(local_repo, "status", "--porcelain"):
        raise ClientError("local route worktree must be clean before launch")
    runner_path = local_repo / RUNNER
    if not runner_path.is_file():
        raise ClientError("tracked E2E runner is missing")
    runner_sha256 = hashlib.sha256(runner_path.read_bytes()).hexdigest()

    client = RpcClient(args.server.resolve(), args.state_dir.resolve())
    report = {}
    try:
        initialized = client.call(
            "initialize", {"protocolVersion": "2024-11-05"}, 10,
        )
        info = initialized.get("serverInfo") or {}
        if info.get("name") != "convir-ops" or info.get("version") != "4.1.0":
            raise ClientError(f"unexpected server info: {info}")
        source_hash = hashlib.sha256(args.server.read_bytes()).hexdigest()
        if info.get("sourceSha256") != source_hash:
            raise ClientError("server source SHA-256 mismatch")

        listed = client.call("tools/list", {}, 10).get("tools") or []
        names = {item.get("name") for item in listed}
        if names != EXPECTED_TOOLS or len(listed) != 6:
            raise ClientError(f"unexpected tool surface: {sorted(names)}")
        plan_schema = next(
            item["inputSchema"] for item in listed
            if item["name"] == "convir_route_plan"
        )
        if plan_schema["properties"]["schema_version"].get("const") != 4:
            raise ClientError("route plan is not schema v4")

        planned = client.tool("convir_route_plan", {
            "schema_version": 4,
            "branch": "codex/generic-run-monitoring-20260716",
            "route_branch_commit": args.route_commit,
            "operation_id": "MAIN_INTEGRATION_REVIEW_ONLY",
        })
        if planned.get("operation_state") != "PLAN_READY":
            raise ClientError(f"unexpected plan state: {planned.get('operation_state')}")
        identity = expected_plan_identity(args.route_commit)
        if planned.get("observed") != {
            **identity,
            "operation_id": "MAIN_INTEGRATION_REVIEW_ONLY",
            "manifest_digest": planned.get("observed", {}).get("manifest_digest"),
            "rules_bundle_digest": planned.get("observed", {}).get("rules_bundle_digest"),
        }:
            raise ClientError(f"planned cloud identity mismatch: {planned.get('observed')}")
        if planned.get("expected", {}).get("route_commit") != args.route_commit:
            raise ClientError("planned route commit mismatch")
        if planned.get("expected", {}).get("runner_sha256") != runner_sha256:
            raise ClientError("planned runner SHA-256 mismatch")
        plan_token = planned["plan_token"]

        started = client.tool("convir_route_start", {"plan_token": plan_token})
        if started.get("operation_state") not in {"LAUNCHED", "LAUNCH_IDEMPOTENT"}:
            raise ClientError(f"unexpected start state: {started.get('operation_state')}")
        receipt = started["receipt"]

        finished = client.tool("convir_route_finish", {"receipt": receipt}, 90)
        if finished.get("operation_state") != "CLOSEOUT_VALIDATED":
            raise ClientError(f"unexpected finish state: {finished.get('operation_state')}")
        terminal = (
            finished.get("observed", {}).get("closeout", {}).get("terminal_tuple")
        )
        if terminal != {
            "state": "COMPLETED_GATE_PASS",
            "decision": "GENERIC_RUN_MONITORING_E2E_PASS",
            "authorizes": "GENERIC_RUN_MONITORING_ADOPTION",
        }:
            raise ClientError(f"unexpected terminal tuple: {terminal}")

        manifest = client.tool("convir_evidence_list", {"receipt": receipt})
        files = {item["name"] for item in manifest.get("files", [])}
        if not EVIDENCE <= files:
            raise ClientError(f"required evidence is absent: {sorted(EVIDENCE - files)}")

        fetched = client.tool("convir_evidence_fetch", {
            "receipt": receipt,
            "local_repo": str(local_repo),
            "files": sorted(EVIDENCE),
        })
        transferred = set(fetched.get("fetched", [])) | set(
            fetched.get("already_verified", [])
        )
        if transferred != EVIDENCE or fetched.get("git_mutations_performed") is not False:
            raise ClientError("receipt-bound evidence fetch did not match the allowlist")

        git_status = client.tool("convir_git_status", {
            "local_repo": str(local_repo),
            "route_id": ROUTE_ID,
        })
        if not git_status.get("github_main_ref_fresh"):
            raise ClientError("local GitHub main ref is stale")

        evidence_dir = local_repo / "experience_docx" / "experiment_logs" / ROUTE_ID
        closeout = json.loads(
            (evidence_dir / "generic_run_monitoring_e2e_closeout.json").read_text()
        )
        summary = json.loads(
            (evidence_dir / "generic_run_monitoring_e2e_summary.json").read_text()
        )
        expected_closeout = {
            "route_id": ROUTE_ID,
            "run_id": OUTPUT_ID,
            "route_commit": args.route_commit,
            "runner_sha256": runner_sha256,
            "state": "COMPLETED_GATE_PASS",
            "decision": "GENERIC_RUN_MONITORING_E2E_PASS",
            "authorizes": "GENERIC_RUN_MONITORING_ADOPTION",
            "validation_pass": True,
            "model_calls": 0,
            "gpu_used": False,
            "dataset_touched": False,
            "checkpoint_touched": False,
            "canary_touched": False,
            "locked_test_touched": False,
        }
        if {key: closeout.get(key) for key in expected_closeout} != expected_closeout:
            raise ClientError("fetched E2E closeout safety identity mismatch")
        expected_summary = {
            "pass": True,
            "files_created": ["heartbeat.json"],
            "forbidden_control_constructs": [],
            "reads_scientific_outputs": False,
            "sends_process_signals": False,
            "queries_gpu": False,
            "source_audit": "python_ast_v1",
        }
        if {key: summary.get(key) for key in expected_summary} != expected_summary:
            raise ClientError("fetched E2E summary safety contract mismatch")
        if not isinstance(summary.get("pulse_cpu_seconds"), (int, float)) or summary["pulse_cpu_seconds"] >= 5:
            raise ClientError("fetched E2E summary exceeds the CPU gate")

        report = {
            "schema_version": 1,
            "server_name": info["name"],
            "server_version": info["version"],
            "server_source_sha256": source_hash,
            "tool_count": len(listed),
            "tool_names": sorted(names),
            "route_schema_version": 4,
            "route_commit": args.route_commit,
            "planned_identity": identity,
            "runner_sha256": runner_sha256,
            "plan_state": planned["operation_state"],
            "plan_token_sha256": digest_token(plan_token),
            "start_state": started["operation_state"],
            "receipt_sha256": digest_token(receipt),
            "finish_state": finished["operation_state"],
            "terminal_tuple": terminal,
            "evidence_files": sorted(EVIDENCE),
            "evidence_fetch_git_mutations": fetched["git_mutations_performed"],
            "github_main_ref_fresh": git_status["github_main_ref_fresh"],
            "closeout_safety_identity_valid": True,
            "summary_safety_contract_valid": True,
            "pulse_cpu_seconds": summary["pulse_cpu_seconds"],
            "pass": True,
        }
    finally:
        client.close()
    print(json.dumps(report, sort_keys=True))
    print("GENERIC_MONITORING_E2E_CLIENT_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClientError as exc:
        print(f"GENERIC_MONITORING_E2E_CLIENT_FAILED {exc}", file=sys.stderr)
        raise SystemExit(1)
