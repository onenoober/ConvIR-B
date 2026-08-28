"""Cloud-only lifecycle and MCP tests for the experiment assistant candidate."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import convir_experiment_assistant_mcp as MCP  # noqa: E402
import experiment_assistant_contract as CONTRACT  # noqa: E402
import experiment_assistant_runner as RUNNER  # noqa: E402


ENTRYPOINT = """\
import json
import os
import sys
import time
from pathlib import Path

output = Path(os.environ["CONVIR_EXPERIMENT_OUTPUT"])
output.mkdir(parents=True, exist_ok=True)
if "--sleep" in sys.argv:
    time.sleep(30)
if "--fail" in sys.argv:
    raise RuntimeError("synthetic loader failure")
metric_id = "wrong_metric" if "--bad-result" in sys.argv else "score"
(output / "summary.json").write_text(
    json.dumps({"primary_metric": {"id": metric_id, "value": 3.5}}) + "\\n",
    encoding="utf-8",
)
"""


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
    return result.stdout.strip()


def make_repo(root):
    repo = root / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Runner Test")
    git(repo, "config", "user.email", "runner@example.invalid")
    (repo / "experiment.py").write_text(ENTRYPOINT, encoding="utf-8")
    git(repo, "add", "experiment.py")
    git(repo, "commit", "-qm", "synthetic entrypoint")
    return repo


def make_contract(experiment_id="synthetic-001", argv=None, threshold=3.0):
    metric = {"id": "score", "direction": "higher"}
    if threshold is not None:
        metric["threshold"] = threshold
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "objective": "Measure one synthetic result without scientific data or GPU access.",
        "entrypoint": {"relpath": "experiment.py", "argv": argv or []},
        "datasets": [{"id": "synthetic_fixture", "role": "training"}],
        "budget": {"max_wall_seconds": 10, "parameters": {"units": 1}},
        "evaluation": {
            "primary_metric": metric,
            "result_files": ["summary.json"],
        },
    }


def wait_terminal(backend, experiment_id, timeout=10):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = backend.status(experiment_id)
        if latest["state"] not in {"PREPARED", "RUNNING"}:
            return latest
        time.sleep(0.05)
    raise AssertionError(f"synthetic attempt did not terminate: {latest}")


class ExperimentAssistantRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = make_repo(self.root)
        self.backend = RUNNER.ExperimentBackend(
            self.root / "assistant", runtime_enabled=True,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_pass_is_snapshot_bound_and_cloud_recorded_once(self):
        started = self.backend.start(str(self.repo), make_contract())
        self.assertEqual("RUNNING", started["state"])
        terminal = wait_terminal(self.backend, "synthetic-001")
        self.assertEqual("COMPLETED_PASS", terminal["state"])
        self.assertEqual(3.5, terminal["latest_attempt"]["result"]["primary_metric"]["value"])
        self.assertEqual("CLOUD_RECORDED_GITHUB_PENDING", terminal["archive"]["state"])
        snapshot_sha = terminal["latest_attempt"]["source_snapshot"]["sha256"]
        self.assertTrue((self.root / "assistant" / "snapshots" / f"{snapshot_sha}.tar").is_file())
        record = json.loads(
            (self.root / "assistant" / "records" / "synthetic-001.json").read_text()
        )
        self.assertEqual("COMPLETED_PASS", record["terminal"]["state"])
        with self.assertRaisesRegex(RUNNER.BackendError, "complete result"):
            self.backend.start(str(self.repo), make_contract())

    def test_missing_threshold_completes_inconclusive_and_is_recorded(self):
        value = make_contract("synthetic-inconclusive", threshold=None)
        started = self.backend.start(str(self.repo), value)
        self.assertIn("descriptive", " ".join(started["warnings"]))
        terminal = wait_terminal(self.backend, "synthetic-inconclusive")
        self.assertEqual("COMPLETED_INCONCLUSIVE", terminal["state"])
        self.assertEqual(
            "descriptive_no_threshold",
            terminal["latest_attempt"]["result"]["decision_basis"],
        )

    def test_failed_attempt_stays_cloud_only_then_same_experiment_repair_succeeds(self):
        failed_contract = make_contract("synthetic-repair", argv=["--fail"])
        self.backend.start(str(self.repo), failed_contract)
        failed = wait_terminal(self.backend, "synthetic-repair")
        self.assertEqual("FAILED_ENGINEERING", failed["state"])
        self.assertEqual("CLOUD_ONLY_ENGINEERING_FAILURE", failed["archive"]["state"])
        self.assertFalse((self.root / "assistant" / "records" / "synthetic-repair.json").exists())
        repaired_contract = make_contract("synthetic-repair")
        self.backend.repair("synthetic-repair", contract=repaired_contract)
        terminal = wait_terminal(self.backend, "synthetic-repair")
        self.assertEqual("COMPLETED_PASS", terminal["state"])
        self.assertEqual(2, terminal["attempt_count"])
        self.assertEqual(1, terminal["automatic_repairs_used"])
        full = self.backend.get("synthetic-repair", view="full")
        self.assertEqual(64, len(full["attempts"][0]["source_snapshot"]["sha256"]))
        self.assertEqual(64, len(full["attempts"][1]["source_snapshot"]["sha256"]))

    def test_result_identity_mismatch_is_engineering_failure(self):
        value = make_contract("synthetic-bad-result", argv=["--bad-result"])
        self.backend.start(str(self.repo), value)
        terminal = wait_terminal(self.backend, "synthetic-bad-result")
        self.assertEqual("FAILED_ENGINEERING", terminal["state"])
        self.assertIn("does not match", terminal["latest_attempt"]["error_summary"])

    def test_two_automatic_repairs_then_confirmation_is_required(self):
        value = make_contract("synthetic-limit", argv=["--fail"])
        self.backend.start(str(self.repo), value)
        wait_terminal(self.backend, "synthetic-limit")
        for _ in range(2):
            self.backend.repair("synthetic-limit")
            wait_terminal(self.backend, "synthetic-limit")
        with self.assertRaisesRegex(RUNNER.BackendError, "operator confirmation"):
            self.backend.repair("synthetic-limit")
        self.backend.repair("synthetic-limit", operator_confirmed=True)
        terminal = wait_terminal(self.backend, "synthetic-limit")
        self.assertEqual("FAILED_ENGINEERING", terminal["state"])
        self.assertEqual(2, terminal["automatic_repairs_used"])
        self.assertEqual(4, terminal["attempt_count"])

    def test_scientific_kernel_change_requires_new_experiment(self):
        value = make_contract("synthetic-new-id", argv=["--fail"])
        self.backend.start(str(self.repo), value)
        wait_terminal(self.backend, "synthetic-new-id")
        changed = copy.deepcopy(value)
        changed["evaluation"]["primary_metric"]["threshold"] = 4.0
        with self.assertRaisesRegex(RUNNER.BackendError, "new experiment_id"):
            self.backend.repair("synthetic-new-id", contract=changed)

    def test_duplicate_active_launch_blocks_and_exact_cancel_is_cloud_only(self):
        value = make_contract("synthetic-cancel", argv=["--sleep"])
        self.backend.start(str(self.repo), value)
        with self.assertRaisesRegex(RUNNER.BackendError, "already exists"):
            self.backend.start(str(self.repo), value)
        cancelled = self.backend.cancel("synthetic-cancel")
        if cancelled["state"] == "RUNNING":
            cancelled = wait_terminal(self.backend, "synthetic-cancel")
        self.assertEqual("CANCELLED", cancelled["state"])
        self.assertEqual("CLOUD_ONLY_CANCELLED", cancelled["archive"]["state"])
        self.assertFalse((self.root / "assistant" / "records" / "synthetic-cancel.json").exists())

    def test_protected_data_is_unavailable_until_backend_can_enforce_it(self):
        value = make_contract("synthetic-protected")
        value["datasets"].append({"id": "sealed", "role": "locked_test"})
        value["protected_access"] = ["locked_test"]
        with self.assertRaisesRegex(RUNNER.BackendError, "explicit_protected_data_access"):
            self.backend.start(str(self.repo), value)

    def test_search_and_compare_read_compact_records(self):
        for experiment_id, threshold in (("compare-pass", 3.0), ("compare-fail", 4.0)):
            self.backend.start(str(self.repo), make_contract(experiment_id, threshold=threshold))
            wait_terminal(self.backend, experiment_id)
        result = self.backend.search(
            query="compare", compare_experiment_ids=["compare-pass", "compare-fail"],
        )
        self.assertEqual(2, result["count"])
        self.assertEqual(
            ["COMPLETED_PASS", "COMPLETED_FAIL"],
            [item["state"] for item in result["comparison"]],
        )

    def test_runtime_mutations_are_disabled_without_explicit_cloud_candidate_gate(self):
        disabled = RUNNER.ExperimentBackend(self.root / "disabled")
        with self.assertRaisesRegex(RUNNER.BackendError, "runtime is disabled"):
            disabled.start(str(self.repo), make_contract("disabled-runtime"))


class ExperimentAssistantMcpTests(unittest.TestCase):
    def test_exact_six_tool_surface_has_no_control_plane_fields(self):
        response = MCP.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tools = response["result"]["tools"]
        self.assertEqual(CONTRACT.PUBLIC_TOOL_NAMES, tuple(item["name"] for item in tools))
        serialized = json.dumps(tools, sort_keys=True)
        for forbidden in (
            "plan_token", "receipt", "catalog_sha256", "inventory_sha256",
            "route_commit", "rules_commit", "terminal_record_sha256",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_initialization_reports_candidate_not_active_runtime(self):
        response = MCP.handle_request({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })
        self.assertEqual("0.2.0-candidate", response["result"]["serverInfo"]["version"])
        self.assertIn("not yet enabled", response["result"]["instructions"])


if __name__ == "__main__":
    unittest.main()
