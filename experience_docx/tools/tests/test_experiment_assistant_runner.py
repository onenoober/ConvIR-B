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
from experiment_assistant_archive import ArchiveError, GitArchiveStore  # noqa: E402
from experiment_assistant_datasets import DatasetRegistry  # noqa: E402
import experiment_assistant_runner as RUNNER  # noqa: E402
import experiment_assistant_transport as TRANSPORT  # noqa: E402


ENTRYPOINT = """\
import json
import os
import sys
import time
from pathlib import Path

output = Path(os.environ["CONVIR_EXPERIMENT_OUTPUT"])
output.mkdir(parents=True, exist_ok=True)
if "--read-dataset" in sys.argv:
    bindings = json.loads(Path(os.environ["CONVIR_EXPERIMENT_DATASETS"]).read_text())
    protected = [item for item in bindings["datasets"] if item["protected"]]
    assert len(protected) == 1
    assert (Path(protected[0]["path"]) / "fixture.txt").read_text() == "sealed fixture\\n"
if "--sleep" in sys.argv:
    time.sleep(30)
if "--brief-sleep" in sys.argv:
    time.sleep(1)
if "--fail" in sys.argv:
    raise RuntimeError("synthetic loader failure")
metric_id = "wrong_metric" if "--bad-result" in sys.argv else "score"
(output / "summary.json").write_text(
    json.dumps({"primary_metric": {"id": metric_id, "value": 3.5}}) + "\\n",
    encoding="utf-8",
)
"""


class UnavailableArchiveStore:
    def get(self, _experiment_id):
        raise ArchiveError("synthetic GitHub outage")


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


def make_registry(root):
    training = root / "synthetic-training"
    sealed = root / "synthetic-sealed"
    training.mkdir()
    sealed.mkdir()
    (training / "fixture.txt").write_text("training fixture\n", encoding="utf-8")
    (sealed / "fixture.txt").write_text("sealed fixture\n", encoding="utf-8")
    path = root / "dataset-registry.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "datasets": [
            {
                "id": "synthetic_fixture", "role": "training",
                "path": str(training), "identity_sha256": "1" * 64,
                "protected": False,
            },
            {
                "id": "sealed", "role": "locked_test",
                "path": str(sealed), "identity_sha256": "2" * 64,
                "protected": True,
            },
        ],
    }) + "\n", encoding="utf-8")
    return DatasetRegistry(path)


def make_remote(root):
    seed = root / "seed"
    remote = root / "remote.git"
    seed.mkdir()
    git(seed, "init", "-q")
    git(seed, "config", "user.name", "Archive Seed")
    git(seed, "config", "user.email", "archive@example.invalid")
    (seed / "README.md").write_text("# Synthetic archive\n", encoding="utf-8")
    git(seed, "add", "README.md")
    git(seed, "commit", "-qm", "seed")
    git(seed, "branch", "-M", "main")
    subprocess.run(["/usr/bin/git", "init", "--bare", "-q", str(remote)], check=True)
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "-q", "origin", "main")
    subprocess.run(
        ["/usr/bin/git", "--git-dir", str(remote), "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True,
    )
    return remote


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
        self.registry = make_registry(self.root)
        self.remote = make_remote(self.root)
        self.archive_store = GitArchiveStore(
            str(self.remote), self.root / "archive-tmp", allow_test_remote=True,
        )
        self.backend = RUNNER.ExperimentBackend(
            self.root / "assistant",
            runtime_enabled=True,
            dataset_registry=self.registry,
            archive_store=self.archive_store,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_pass_is_snapshot_bound_and_cloud_recorded_once(self):
        started = self.backend.start(str(self.repo), make_contract())
        self.assertEqual("RUNNING", started["state"])
        terminal = wait_terminal(self.backend, "synthetic-001")
        self.assertEqual("COMPLETED_PASS", terminal["state"])
        self.assertEqual(3.5, terminal["latest_attempt"]["result"]["primary_metric"]["value"])
        self.assertEqual("GITHUB_ARCHIVED", terminal["archive"]["state"])
        self.assertEqual("github", terminal["record_source"])
        snapshot_sha = terminal["latest_attempt"]["source_snapshot"]["sha256"]
        self.assertTrue((self.root / "assistant" / "snapshots" / f"{snapshot_sha}.tar").is_file())
        record = json.loads(
            (self.root / "assistant" / "records" / "synthetic-001.json").read_text()
        )
        self.assertEqual("COMPLETED_PASS", record["terminal"]["state"])
        self.assertEqual("1" * 64, record["datasets"][0]["identity_sha256"])
        self.assertNotIn("path", record["datasets"][0])
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
        self.assertEqual([], self.archive_store.records())
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
        self.assertEqual([], self.archive_store.records())

    def test_protected_data_defaults_to_deny_then_explicit_contract_is_delivered(self):
        value = make_contract("synthetic-protected")
        value["datasets"].append({"id": "sealed", "role": "locked_test"})
        with self.assertRaisesRegex(RUNNER.BackendError, "explicit access"):
            self.backend.start(str(self.repo), value)
        value["protected_access"] = ["locked_test"]
        value["entrypoint"]["argv"] = ["--read-dataset"]
        self.backend.start(str(self.repo), value)
        terminal = wait_terminal(self.backend, "synthetic-protected")
        self.assertEqual("COMPLETED_PASS", terminal["state"])
        bindings = terminal["datasets"]
        self.assertEqual(
            {"sealed": True, "synthetic_fixture": False},
            {item["id"]: item["protected"] for item in bindings},
        )

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
        self.assertTrue(all(item["record_source"] == "github" for item in result["experiments"]))

    def test_github_only_reader_needs_no_cloud_attempt_state(self):
        self.backend.start(str(self.repo), make_contract("github-only"))
        wait_terminal(self.backend, "github-only")
        reader = RUNNER.ExperimentBackend(
            self.root / "empty-reader",
            archive_store=self.archive_store,
        )
        record = reader.get("github-only", view="full")
        self.assertEqual("github", record["record_source"])
        self.assertEqual("COMPLETED_PASS", record["state"])
        self.assertEqual(1, reader.search(query="github-only")["count"])

    def test_github_record_blocks_duplicate_id_from_a_fresh_cloud_state(self):
        self.backend.start(str(self.repo), make_contract("github-duplicate"))
        wait_terminal(self.backend, "github-duplicate")
        fresh = RUNNER.ExperimentBackend(
            self.root / "fresh-assistant",
            runtime_enabled=True,
            dataset_registry=self.registry,
            archive_store=self.archive_store,
        )
        with self.assertRaisesRegex(RUNNER.BackendError, "GitHub already contains"):
            fresh.start(str(self.repo), make_contract("github-duplicate"))

    def test_completed_cloud_result_remains_readable_during_github_outage(self):
        self.backend.start(str(self.repo), make_contract("github-outage"))
        wait_terminal(self.backend, "github-outage")
        self.backend.archive_store = UnavailableArchiveStore()
        status = self.backend.status("github-outage")
        full = self.backend.get("github-outage", view="full")
        self.assertEqual("cloud", status["record_source"])
        self.assertEqual("COMPLETED_PASS", status["state"])
        self.assertIn("temporarily unavailable", " ".join(status["warnings"]))
        self.assertEqual("COMPLETED_PASS", full["attempts"][-1]["state"])

    def test_archive_push_failure_keeps_complete_cloud_result(self):
        self.backend.start(
            str(self.repo),
            make_contract("archive-push-failure", argv=["--brief-sleep"]),
        )
        hook = self.remote / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        hook.chmod(0o700)
        terminal = wait_terminal(self.backend, "archive-push-failure")
        self.assertEqual("COMPLETED_PASS", terminal["state"])
        self.assertEqual("cloud", terminal["record_source"])
        self.assertEqual(
            "CLOUD_RECORDED_GITHUB_ARCHIVE_FAILED",
            terminal["archive"]["state"],
        )
        self.assertTrue(
            (
                self.root / "assistant" / "records" / "archive-push-failure.json"
            ).is_file()
        )
        self.assertEqual([], self.archive_store.records())

    def test_archive_is_idempotent_and_never_overwrites_same_experiment_id(self):
        self.backend.start(str(self.repo), make_contract("immutable-record"))
        wait_terminal(self.backend, "immutable-record")
        record = json.loads(
            (self.root / "assistant" / "records" / "immutable-record.json").read_text()
        )
        repeated = self.archive_store.archive(record, CONTRACT.canonical_sha256(record))
        self.assertTrue(repeated["idempotent"])
        changed = copy.deepcopy(record)
        changed["result"]["primary_metric"]["value"] = 3.6
        changed["attempts"][-1]["result"]["primary_metric"]["value"] = 3.6
        with self.assertRaisesRegex(ArchiveError, "different record"):
            self.archive_store.archive(changed, CONTRACT.canonical_sha256(changed))

    def test_runtime_mutations_are_disabled_without_explicit_cloud_candidate_gate(self):
        disabled = RUNNER.ExperimentBackend(self.root / "disabled")
        with self.assertRaisesRegex(RUNNER.BackendError, "runtime is disabled"):
            disabled.start(str(self.repo), make_contract("disabled-runtime"))


class ExperimentAssistantTransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = make_repo(self.root)
        self.registry = make_registry(self.root)
        self.remote = make_remote(self.root)
        self.state = self.root / "remote-state"
        self.client = TRANSPORT.CloudExperimentClient(
            remote_argv=[
                sys.executable,
                str(TOOLS / "experiment_assistant_runner.py"),
                "_remote",
                "--root",
                str(self.state),
                "--dataset-registry",
                str(self.root / "dataset-registry.json"),
                "--archive-remote",
                str(self.remote),
                "--archive-test-remote",
            ],
            timeout=30,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def wait_terminal(self, experiment_id, timeout=10):
        deadline = time.monotonic() + timeout
        latest = None
        while time.monotonic() < deadline:
            latest = self.client.status(experiment_id)
            if latest["state"] not in {"PREPARED", "RUNNING"}:
                return latest
            time.sleep(0.05)
        raise AssertionError(f"transport attempt did not terminate: {latest}")

    def test_dirty_local_source_is_uploaded_and_archived_by_cloud_backend(self):
        changed = ENTRYPOINT.replace('"value": 3.5', '"value": 4.5')
        (self.repo / "experiment.py").write_text(changed, encoding="utf-8")
        value = make_contract("transport-dirty", threshold=4.0)
        started = self.client.start(str(self.repo), value)
        self.assertEqual("RUNNING", started["state"])
        terminal = self.wait_terminal("transport-dirty")
        self.assertEqual("COMPLETED_PASS", terminal["state"])
        self.assertEqual(
            4.5, terminal["latest_attempt"]["result"]["primary_metric"]["value"],
        )
        self.assertEqual("GITHUB_ARCHIVED", terminal["archive"]["state"])
        snapshot = terminal["latest_attempt"]["source_snapshot"]
        self.assertTrue((self.state / "snapshots" / f"{snapshot['sha256']}.tar").is_file())
        (self.root / "dataset-registry.json").unlink()
        self.assertEqual("COMPLETED_PASS", self.client.status("transport-dirty")["state"])

    def test_repair_fetches_context_and_uploads_a_new_snapshot(self):
        value = make_contract("transport-repair", argv=["--fail"])
        self.client.start(str(self.repo), value)
        failed = self.wait_terminal("transport-repair")
        self.assertEqual("FAILED_ENGINEERING", failed["state"])
        repaired = make_contract("transport-repair")
        changed = ENTRYPOINT.replace('"value": 3.5', '"value": 3.75')
        (self.repo / "experiment.py").write_text(changed, encoding="utf-8")
        self.client.repair("transport-repair", contract=repaired)
        terminal = self.wait_terminal("transport-repair")
        self.assertEqual("COMPLETED_PASS", terminal["state"])
        full = self.client.get("transport-repair", view="full")
        self.assertEqual(2, len(full["attempts"]))
        self.assertNotEqual(
            full["attempts"][0]["source_snapshot"]["sha256"],
            full["attempts"][1]["source_snapshot"]["sha256"],
        )
        self.assertEqual(1, full["automatic_repairs_used"])


class ExperimentAssistantMcpTests(unittest.TestCase):
    def test_production_transport_is_one_fixed_cloud_argv(self):
        client = TRANSPORT.CloudExperimentClient()
        self.assertEqual("/usr/bin/ssh", client.remote_argv[0])
        self.assertIn("convir-4090", client.remote_argv)
        self.assertIn(TRANSPORT.REMOTE_PYTHON, client.remote_argv)
        self.assertIn(TRANSPORT.REMOTE_RUNNER, client.remote_argv)
        self.assertNotIn("-c", client.remote_argv)

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
        self.assertEqual("0.4.0-candidate", response["result"]["serverInfo"]["version"])
        self.assertIn("cloud-only", response["result"]["instructions"])


if __name__ == "__main__":
    unittest.main()
