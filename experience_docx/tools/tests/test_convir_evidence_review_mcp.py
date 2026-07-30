#!/usr/bin/env python3
"""Tests for the GitHub-only convir-evidence-review MCP facade."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import convir_evidence_review_mcp as review


def terminal_record(route_id="route-a", group="route-a"):
    prefix = f"experience_docx/experiment_logs/{group}"
    return {
        "schema_version": 1,
        "route_id": route_id,
        "operation_id": "A0",
        "run_id": "run-a",
        "state": "COMPLETED_GATE_PASS",
        "decision": "A0_PASS",
        "authorizes": "NEXT_STAGE",
        "receipt": "a" * 64,
        "route_commit": "b" * 40,
        "contract_path": f"experience_docx/experiment_cards/{route_id}.md",
        "closeout_path": f"{prefix}/route_closeout.json",
        "conclusion_path": f"{prefix}/route_conclusion.json",
        "result_paths": [f"{prefix}/route_summary.json"],
    }


def evidence_files(record):
    return {
        record["closeout_path"]: b"not read",
        record["conclusion_path"]: b"not read",
        record["result_paths"][0]: b"not read",
    }


def cloud_binding(commit, catalog_sha256, *, raw_inventory_authorized=True):
    terminal_sha256 = "d" * 64
    return {
        "eligible": True,
        "state": "TERMINAL_BINDING_VERIFIED",
        "snapshot_commit": commit,
        "catalog_sha256": catalog_sha256,
        "terminal_index_sha256": "e" * 64,
        "terminal_record_sha256": terminal_sha256,
        "route_id": "route-a",
        "operation_id": "A0",
        "run_id": "run-a",
        "output_id": "run-a",
        "mode": "synthetic",
        "session": "convir-route-a-synthetic-run-a-0123456789ab",
        "route_commit": "b" * 40,
        "manifest_sha256": "1" * 64,
        "runtime_spec_sha256": "2" * 64,
        "closeout_sha256": "3" * 64,
        "runner_sha256": "4" * 64,
        "run_root": f"{review.inventory.REMOTE_RUNS}/route-a/run-a",
        "evidence_role": "development_screening",
        "raw_inventory_authorized": raw_inventory_authorized,
        "raw_inventory_exclusion_reason": (
            None if raw_inventory_authorized else
            "protected_or_unknown_role_permission_or_touch"
        ),
        "expected_lifecycle_identity": {
            "schema_version": 1,
            "route_id": "route-a",
            "operation_id": "A0",
            "run_id": "run-a",
            "route_commit": "b" * 40,
            "runner_sha256": "4" * 64,
        },
        "expected_evidence": [{
            "source_relpath": "workload/metric.json",
            "destination_filename": "metric.json",
            "github_path": "experience_docx/experiment_logs/route-a/metric.json",
            "bytes": 12,
            "sha256": "5" * 64,
            "max_bytes": 4096,
            "required": True,
        }],
        "optional_evidence": [],
        "unmapped_results": [],
    }


def remote_summary(binding, inventory_sha256="6" * 64):
    return {
        "schema_version": 1,
        "ok": True,
        "operation": "cloud-inventory",
        "state": "INVENTORY_READY",
        "exit_code": 0,
        "identity": review._binding_identity(binding),
        "declared_run_root": binding["run_root"],
        "scope": "bound_run_root",
        "root_binding_enforced": True,
        "discovery_completeness": "complete",
        "scientific_completeness": "not_assessed",
        "inventory_sha256": inventory_sha256,
        "entry_count": 3,
    }


class EvidenceReviewMcpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="convir evidence review ")
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.previous_root = os.environ.get(review.WORKSPACE_ROOT_ENV)
        os.environ[review.WORKSPACE_ROOT_ENV] = str(self.root)
        self.git("init", "-b", "main")
        self.git("config", "user.email", "evidence-review@example.invalid")
        self.git("config", "user.name", "Evidence Review Test")
        self.git("remote", "add", review.TRUSTED_REMOTE_NAME, review.TRUSTED_REMOTE_URLS[0])

    def tearDown(self):
        if self.previous_root is None:
            os.environ.pop(review.WORKSPACE_ROOT_ENV, None)
        else:
            os.environ[review.WORKSPACE_ROOT_ENV] = self.previous_root
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(self.repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )

    def commit_snapshot(
        self, record, extra_files=None, message="snapshot", *, publish_main=True
    ):
        index = self.repo / review.catalog.INDEX_PATH
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        files = {**evidence_files(record), **(extra_files or {})}
        for relative, raw in files.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.git("add", "--", "experience_docx")
        self.git("commit", "-m", message)
        commit = self.git("rev-parse", "HEAD").stdout.strip()
        if publish_main:
            self.git("update-ref", review.TRUSTED_MAIN_REF, commit)
        return commit

    def call(self, name, arguments):
        return review.handle({
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })

    def test_server_exposes_exact_seven_read_only_tools(self):
        initialized = review.handle({
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })
        self.assertEqual("convir-evidence-review", initialized["serverInfo"]["name"])
        self.assertEqual("1.3.0", initialized["serverInfo"]["version"])
        listed = review.handle({"method": "tools/list", "params": {}})
        self.assertEqual(
            [
                "convir_evidence_catalog_summary",
                "convir_evidence_completeness_receipt",
                "convir_evidence_catalog_query",
                "convir_evidence_bundle",
                "convir_evidence_cloud_inventory_summary",
                "convir_evidence_cloud_inventory_query",
                "convir_evidence_cloud_text_read",
            ],
            [tool["name"] for tool in listed["tools"]],
        )
        self.assertTrue(all("outputSchema" in tool for tool in listed["tools"]))
        forbidden = {
            "host", "command", "remote_path", "run_root", "cloud_available",
            "active_session", "scan_limits",
        }
        cloud_tools = [
            tool for tool in listed["tools"]
            if "cloud_" in tool["name"]
        ]
        self.assertTrue(all(
            forbidden.isdisjoint(tool["inputSchema"]["properties"])
            for tool in cloud_tools
        ))

    def test_evidence_bundle_resolves_sha_bound_leaf_and_pages_all_files(self):
        record = terminal_record()
        prefix = "experience_docx/experiment_logs/route-a"
        contract_raw = b"contract\n"
        closeout_raw = b"not read"
        conclusion_raw = b"not read"
        result_raw = b"not read"
        manifest_raw = b"{}\n"
        manifest_path = f"{prefix}/launch_contract/A0/manifest.json"
        record.update({
            "schema_version": 2,
            "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
            "closeout_sha256": hashlib.sha256(closeout_raw).hexdigest(),
            "conclusion_sha256": hashlib.sha256(conclusion_raw).hexdigest(),
            "contract_bundle": [{
                "path": manifest_path,
                "source_path": "experience_docx/route_operations.json",
                "bytes": len(manifest_raw),
                "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            }],
            "result_files": [{
                "path": record["result_paths"][0],
                "bytes": len(result_raw),
                "sha256": hashlib.sha256(result_raw).hexdigest(),
            }],
            "prior_terminal_record": {
                "prior_closeout_path": None,
                "prior_terminal_tuple": None,
            },
        })
        commit = self.commit_snapshot(record, {
            record["contract_path"]: contract_raw,
            record["closeout_path"]: closeout_raw,
            record["conclusion_path"]: conclusion_raw,
            record["result_paths"][0]: result_raw,
            manifest_path: manifest_raw,
        })
        loaded = review.catalog.load_catalog(self.repo, commit)
        _, records, _ = review.catalog.load_terminal_records(self.repo, commit)
        record_sha256 = records[0]["record_sha256"]
        binding = cloud_binding(commit, loaded["catalog_sha256"])
        binding.update({
            "terminal_record_sha256": record_sha256,
            "closeout_sha256": record["closeout_sha256"],
            "expected_evidence": [{
                "source_relpath": "workload/route_summary.json",
                "destination_filename": "route_summary.json",
                "github_path": record["result_paths"][0],
                "bytes": len(result_raw),
                "sha256": hashlib.sha256(result_raw).hexdigest(),
                "max_bytes": 4096,
                "required": True,
            }],
        })
        arguments = {
            "local_repo": str(self.repo),
            "snapshot_commit": commit,
            "catalog_sha256": loaded["catalog_sha256"],
            "terminal_record_sha256": record_sha256,
            "limit": 2,
        }
        files = []
        with mock.patch.object(
            review.inventory, "prepare_terminal_binding", return_value=binding
        ), mock.patch.object(review, "_run_fixed_remote") as transport:
            while True:
                result = self.call("convir_evidence_bundle", arguments)
                self.assertFalse(result["isError"])
                value = result["structuredContent"]
                self.assertEqual(2, value["schema_version"])
                self.assertEqual("complete", value["bundle_completeness"])
                files.extend(value["files"])
                if not value["has_more"]:
                    break
                arguments["cursor"] = value["next_cursor"]
        transport.assert_not_called()
        self.assertEqual(value["total_count"], len(files))
        self.assertEqual(len(files), len({item["path"] for item in files}))
        self.assertTrue(any(
            "scientific_conclusion" in item["roles"] for item in files
        ))
        self.assertTrue(any(
            "formal_result" in item["roles"] for item in files
        ))
        self.assertTrue(all(item["content_returned"] is False for item in files))

    def test_completeness_receipt_is_main_bound_stable_and_transport_free(self):
        record = terminal_record()
        first = self.commit_snapshot(
            record,
            {"experience_docx/experiment_logs/legacy/summary.json": b"not read"},
        )
        with mock.patch.object(review, "_run_fixed_remote") as transport:
            latest = self.call(
                "convir_evidence_completeness_receipt",
                {"local_repo": str(self.repo)},
            )
        transport.assert_not_called()
        self.assertFalse(latest["isError"])
        value = latest["structuredContent"]
        self.assertEqual(2, value["schema_version"])
        self.assertEqual(first, value["snapshot_commit"])
        self.assertEqual("incomplete", value["review_completeness"])
        self.assertEqual(2, value["entry_partition"]["catalog_entries"])
        self.assertEqual(1, value["entry_partition"]["unindexed_entries"])
        self.assertEqual(
            1, value["unresolved_counts"]["path_only_legacy_terminal_records"]
        )
        self.assertIn("result_contents", value["excluded_sources"])
        self.assertFalse(value["git_mutations_performed"])
        self.assertLessEqual(
            len(review.canonical_bytes(latest)), review.MAX_TOOL_RESULT_BYTES
        )
        self.assertEqual(value, json.loads(latest["content"][0]["text"]))

        second = self.commit_snapshot(
            record,
            {"experience_docx/experiment_logs/new-history/summary.json": b"new"},
            "moved ref",
        )
        self.assertNotEqual(first, second)
        pinned = self.call(
            "convir_evidence_completeness_receipt",
            {"local_repo": str(self.repo), "snapshot_commit": first},
        )
        self.assertFalse(pinned["isError"])
        self.assertEqual(value["receipt_sha256"], pinned["structuredContent"]["receipt_sha256"])
        self.assertEqual(first, pinned["structuredContent"]["snapshot_commit"])

        route_only = self.commit_snapshot(
            record,
            {"experience_docx/experiment_logs/route-only/summary.json": b"local"},
            "unpublished route commit",
            publish_main=False,
        )
        rejected = self.call(
            "convir_evidence_completeness_receipt",
            {"local_repo": str(self.repo), "snapshot_commit": route_only},
        )
        self.assertTrue(rejected["isError"])
        self.assertEqual(
            "SNAPSHOT_OUTSIDE_GITHUB_MAIN",
            rejected["structuredContent"]["state"],
        )

    def test_summary_freezes_symbolic_ref_before_repository_moves(self):
        record = terminal_record()
        first = self.commit_snapshot(record)
        summary = self.call(
            "convir_evidence_catalog_summary",
            {"local_repo": str(self.repo)},
        )
        self.assertFalse(summary["isError"])
        self.assertEqual(first, summary["structuredContent"]["header"]["snapshot_commit"])
        self.assertEqual(
            "not_assessed", summary["structuredContent"]["ref_freshness"]
        )
        self.assertEqual(
            summary["structuredContent"], json.loads(summary["content"][0]["text"])
        )

        second = self.commit_snapshot(
            record,
            {"experience_docx/experiment_logs/new-history/summary.json": b"new"},
            "moved ref",
        )
        self.assertNotEqual(first, second)
        old_query = self.call(
            "convir_evidence_catalog_query",
            {
                "local_repo": str(self.repo),
                "snapshot_commit": first,
                "coverage": "all",
                "terms": ["new-history"],
            },
        )
        self.assertFalse(old_query["isError"])
        self.assertEqual(0, old_query["structuredContent"]["total_count"])

    def test_query_cursor_reassembles_without_duplicates_and_rejects_drift(self):
        record = terminal_record()
        extras = {
            f"experience_docx/experiment_logs/legacy-{index:03d}/summary.json": b"x"
            for index in range(45)
        }
        commit = self.commit_snapshot(record, extras)
        observed = []
        cursor = None
        first_cursor = None
        while True:
            arguments = {
                "local_repo": str(self.repo),
                "snapshot_commit": commit,
                "coverage": "unindexed",
                "limit": 7,
            }
            if cursor is not None:
                arguments["cursor"] = cursor
            result = self.call("convir_evidence_catalog_query", arguments)
            self.assertFalse(result["isError"])
            self.assertLessEqual(
                len(review.canonical_bytes(result)), review.MAX_TOOL_RESULT_BYTES
            )
            value = result["structuredContent"]
            observed.extend(entry["directory_name"] for entry in value["entries"])
            if value["complete"]:
                break
            cursor = value["next_cursor"]
            first_cursor = first_cursor or cursor
        self.assertEqual(45, len(observed))
        self.assertEqual(len(observed), len(set(observed)))

        drift = self.call(
            "convir_evidence_catalog_query",
            {
                "local_repo": str(self.repo),
                "snapshot_commit": commit,
                "coverage": "all",
                "cursor": first_cursor,
                "limit": 7,
            },
        )
        self.assertTrue(drift["isError"])
        self.assertEqual(
            "REPO_CURSOR_IDENTITY_MISMATCH",
            drift["structuredContent"]["state"],
        )

    def test_loose_file_and_zero_match_remain_non_scientific(self):
        record = terminal_record()
        commit = self.commit_snapshot(
            record,
            {"experience_docx/experiment_logs/root_helper.sh": b"not read"},
        )
        found = self.call(
            "convir_evidence_catalog_query",
            {
                "local_repo": str(self.repo),
                "snapshot_commit": commit,
                "coverage": "unindexed",
                "terms": ["root_helper.sh"],
            },
        )
        self.assertFalse(found["isError"])
        entry = found["structuredContent"]["entries"][0]
        self.assertEqual("loose_file", entry["record_kind"])
        self.assertEqual("NOT_ASSESSED", entry["terminal_assessment"])

        empty = self.call(
            "convir_evidence_catalog_query",
            {
                "local_repo": str(self.repo),
                "snapshot_commit": commit,
                "terms": ["does-not-exist"],
            },
        )
        self.assertFalse(empty["isError"])
        self.assertEqual("all", empty["structuredContent"]["coverage"])
        self.assertEqual(0, empty["structuredContent"]["total_count"])

    def test_invalid_repo_commit_and_cursor_fail_closed(self):
        record = terminal_record()
        commit = self.commit_snapshot(record)
        outside = self.call(
            "convir_evidence_catalog_summary",
            {"local_repo": "/"},
        )
        self.assertTrue(outside["isError"])
        self.assertEqual("ARGUMENTS_INVALID", outside["structuredContent"]["state"])

        bad_commit = self.call(
            "convir_evidence_catalog_query",
            {"local_repo": str(self.repo), "snapshot_commit": "bad"},
        )
        self.assertTrue(bad_commit["isError"])
        self.assertEqual("ARGUMENTS_INVALID", bad_commit["structuredContent"]["state"])

        too_many_terms = self.call(
            "convir_evidence_catalog_query",
            {
                "local_repo": str(self.repo),
                "snapshot_commit": commit,
                "terms": ["term"] * 9,
            },
        )
        self.assertTrue(too_many_terms["isError"])
        self.assertEqual(
            "ARGUMENTS_INVALID", too_many_terms["structuredContent"]["state"]
        )

        route_only = self.commit_snapshot(
            record,
            {"experience_docx/experiment_logs/route-only/summary.json": b"local"},
            "unpublished route commit",
            publish_main=False,
        )
        rejected = self.call(
            "convir_evidence_catalog_query",
            {"local_repo": str(self.repo), "snapshot_commit": route_only},
        )
        self.assertTrue(rejected["isError"])
        self.assertEqual(
            "SNAPSHOT_OUTSIDE_GITHUB_MAIN",
            rejected["structuredContent"]["state"],
        )

        bad_cursor = self.call(
            "convir_evidence_catalog_query",
            {
                "local_repo": str(self.repo),
                "snapshot_commit": commit,
                "cursor": "not-a-valid-cursor",
            },
        )
        self.assertTrue(bad_cursor["isError"])
        self.assertEqual("REPO_CURSOR_INVALID", bad_cursor["structuredContent"]["state"])

        self.git("remote", "set-url", review.TRUSTED_REMOTE_NAME, "https://example.invalid/repo")
        untrusted = self.call(
            "convir_evidence_catalog_summary", {"local_repo": str(self.repo)}
        )
        self.assertTrue(untrusted["isError"])
        self.assertEqual("GITHUB_REMOTE_UNTRUSTED", untrusted["structuredContent"]["state"])

    def test_cloud_summary_query_rescan_cursor_and_drift_are_identity_bound(self):
        record = terminal_record()
        commit = self.commit_snapshot(record)
        catalog_sha256 = review.catalog.load_catalog(
            self.repo, commit
        )["catalog_sha256"]
        binding = cloud_binding(commit, catalog_sha256)
        inventory_sha256 = "6" * 64
        entries = [
            {
                "scope": "raw_output",
                "relative_path": f"workload/item-{index}-\"\\\x01.json",
                "artifact_class": "raw_artifact",
                "extension": ".json",
                "file_type": "file",
                "bytes": index,
                "github_path": None,
                "github_bytes": None,
                "github_sha256": None,
                "cloud_sha256": None,
                "identity_basis": "metadata_only",
                "reconciliation_state": "CLOUD_ONLY",
                "policy_assessment": "expected_cloud_only_raw_artifact",
            }
            for index in range(3)
        ]

        def fixed_remote(request):
            if request["operation"] == "summary":
                return remote_summary(binding, inventory_sha256)
            query = request["query"]
            if query["inventory_sha256"] != inventory_sha256:
                return {
                    "ok": False,
                    "operation": "cloud-inventory-query",
                    "state": "INVENTORY_DRIFT",
                    "exit_code": 3,
                    "scientific_completeness": "not_assessed",
                }
            query_sha256 = review.inventory.inventory_query_sha256(
                review._binding_identity(binding),
                inventory_sha256,
                query["reconciliation_states"],
                query["terms"],
            )
            selected = entries[
                query["offset"]:query["offset"] + query["limit"]
            ]
            end = query["offset"] + len(selected)
            return {
                "schema_version": 1,
                "ok": True,
                "operation": "cloud-inventory-query",
                "state": "INVENTORY_ENTRIES_OK",
                "exit_code": 0,
                "snapshot_commit": commit,
                "terminal_record_sha256": binding["terminal_record_sha256"],
                "inventory_sha256": inventory_sha256,
                "query_sha256": query_sha256,
                "reconciliation_states": query["reconciliation_states"],
                "terms": query["terms"],
                "offset": query["offset"],
                "returned_count": len(selected),
                "total_count": len(entries),
                "entries": selected,
                "page_sha256": review.inventory.canonical_sha256(selected),
                "complete": end == len(entries),
                "has_more": end != len(entries),
                "next_offset": None if end == len(entries) else end,
                "discovery_completeness": "complete",
                "scientific_completeness": "not_assessed",
            }

        base_args = {
            "local_repo": str(self.repo),
            "snapshot_commit": commit,
            "catalog_sha256": catalog_sha256,
            "terminal_record_sha256": binding["terminal_record_sha256"],
        }
        with mock.patch.object(
            review.inventory, "prepare_terminal_binding", return_value=binding
        ), mock.patch.object(review, "_run_fixed_remote", side_effect=fixed_remote) as transport:
            summary = self.call(
                "convir_evidence_cloud_inventory_summary", base_args
            )
            self.assertFalse(summary["isError"])
            self.assertEqual(
                inventory_sha256, summary["structuredContent"]["inventory_sha256"]
            )

            first = self.call(
                "convir_evidence_cloud_inventory_query",
                {**base_args, "inventory_sha256": inventory_sha256, "limit": 2},
            )
            self.assertFalse(first["isError"])
            self.assertEqual(2, first["structuredContent"]["returned_count"])
            cursor = first["structuredContent"]["next_cursor"]
            self.assertIsNotNone(cursor)
            second = self.call(
                "convir_evidence_cloud_inventory_query",
                {
                    **base_args,
                    "inventory_sha256": inventory_sha256,
                    "cursor": cursor,
                    "limit": 2,
                },
            )
            self.assertFalse(second["isError"])
            self.assertEqual(2, second["structuredContent"]["offset"])
            self.assertTrue(second["structuredContent"]["complete"])

            drift = self.call(
                "convir_evidence_cloud_inventory_query",
                {**base_args, "inventory_sha256": "7" * 64},
            )
            self.assertTrue(drift["isError"])
            self.assertEqual("INVENTORY_DRIFT", drift["structuredContent"]["state"])
            self.assertEqual(4, transport.call_count)

    def test_protected_binding_returns_without_cloud_transport(self):
        record = terminal_record()
        commit = self.commit_snapshot(record)
        catalog_sha256 = review.catalog.load_catalog(
            self.repo, commit
        )["catalog_sha256"]
        binding = cloud_binding(
            commit, catalog_sha256, raw_inventory_authorized=False
        )
        args = {
            "local_repo": str(self.repo),
            "snapshot_commit": commit,
            "catalog_sha256": catalog_sha256,
            "terminal_record_sha256": binding["terminal_record_sha256"],
        }
        with mock.patch.object(
            review.inventory, "prepare_terminal_binding", return_value=binding
        ), mock.patch.object(review, "_run_fixed_remote") as transport:
            summary = self.call(
                "convir_evidence_cloud_inventory_summary", args
            )
            self.assertFalse(summary["isError"])
            self.assertEqual("NOT_INVENTORIED", summary["structuredContent"]["state"])
            query = self.call(
                "convir_evidence_cloud_inventory_query",
                {
                    **args,
                    "inventory_sha256": summary["structuredContent"]["inventory_sha256"],
                },
            )
            self.assertFalse(query["isError"])
            self.assertTrue(all(
                item["reconciliation_state"] == "NOT_INVENTORIED"
                for item in query["structuredContent"]["entries"]
            ))
            transport.assert_not_called()

    def test_cloud_text_read_pages_raw_utf8_and_binds_continuation_sha(self):
        record = terminal_record()
        commit = self.commit_snapshot(record)
        catalog_sha256 = review.catalog.load_catalog(
            self.repo, commit
        )["catalog_sha256"]
        binding = cloud_binding(commit, catalog_sha256)
        inventory_sha256 = "6" * 64
        relative_path = "workload/per_scene.csv"
        raw = ("scene,value\n" + "alpha,\u96fe\\\"\n" * 80).encode("utf-8")
        file_sha256 = hashlib.sha256(raw).hexdigest()

        def fixed_remote(request):
            self.assertEqual(2, request["schema_version"])
            self.assertEqual("text_read", request["operation"])
            query = request["query"]
            self.assertEqual(inventory_sha256, query["inventory_sha256"])
            self.assertEqual(relative_path, query["relative_path"])
            if query["offset"]:
                self.assertEqual(file_sha256, query["expected_file_sha256"])
            start = query["offset"]
            end = min(len(raw), start + query["page_bytes"])
            while end > start:
                try:
                    content = raw[start:end].decode("utf-8")
                    break
                except UnicodeDecodeError:
                    end -= 1
            else:
                content = ""
            page_raw = raw[start:end]
            return {
                "schema_version": 2,
                "ok": True,
                "operation": "cloud-text-read",
                "state": "CLOUD_TEXT_PAGE_OK",
                "exit_code": 0,
                "snapshot_commit": commit,
                "catalog_sha256": catalog_sha256,
                "terminal_record_sha256": binding["terminal_record_sha256"],
                "inventory_sha256": inventory_sha256,
                "relative_path": relative_path,
                "artifact_class": "raw_artifact",
                "policy_assessment": "expected_cloud_only_raw_artifact",
                "evidence_status": "unmapped_raw_text",
                "reconciliation_state": "CLOUD_ONLY",
                "bytes": len(raw),
                "file_sha256": file_sha256,
                "identity_basis": "content_sha256",
                "encoding": "utf-8",
                "page_start_byte": start,
                "page_end_byte": end,
                "page_bytes": len(page_raw),
                "page_sha256": hashlib.sha256(page_raw).hexdigest(),
                "content": content,
                "page_complete": True,
                "terminal_page": end == len(raw),
                "complete": end == len(raw),
                "has_more": end != len(raw),
                "scientific_completeness": "not_assessed",
            }

        base_args = {
            "local_repo": str(self.repo),
            "snapshot_commit": commit,
            "catalog_sha256": catalog_sha256,
            "terminal_record_sha256": binding["terminal_record_sha256"],
            "inventory_sha256": inventory_sha256,
            "relative_path": relative_path,
            "page_bytes": 256,
        }
        observed = ""
        arguments = dict(base_args)
        with mock.patch.object(
            review.inventory, "prepare_terminal_binding", return_value=binding
        ), mock.patch.object(
            review, "_run_fixed_remote", side_effect=fixed_remote
        ) as transport:
            first = self.call("convir_evidence_cloud_text_read", arguments)
            self.assertFalse(first["isError"])
            first_value = first["structuredContent"]
            self.assertEqual(2, first_value["schema_version"])
            self.assertEqual("unmapped_raw_text", first_value["evidence_status"])
            self.assertEqual(file_sha256, first_value["file_sha256"])
            self.assertTrue(first_value["has_more"])

            missing_sha = self.call(
                "convir_evidence_cloud_text_read",
                {**base_args, "cursor": first_value["next_cursor"]},
            )
            self.assertTrue(missing_sha["isError"])
            self.assertEqual(
                "ARGUMENTS_INVALID", missing_sha["structuredContent"]["state"]
            )

            value = first_value
            while True:
                observed += value["content"]
                if value["terminal_page"]:
                    break
                arguments = {
                    **base_args,
                    "file_sha256": file_sha256,
                    "cursor": value["next_cursor"],
                }
                page = self.call("convir_evidence_cloud_text_read", arguments)
                self.assertFalse(page["isError"])
                value = page["structuredContent"]
        self.assertEqual(raw.decode("utf-8"), observed)
        self.assertGreater(transport.call_count, 1)

    def test_cloud_text_read_rejects_protected_scope_without_transport(self):
        record = terminal_record()
        commit = self.commit_snapshot(record)
        catalog_sha256 = review.catalog.load_catalog(
            self.repo, commit
        )["catalog_sha256"]
        binding = cloud_binding(
            commit, catalog_sha256, raw_inventory_authorized=False
        )
        with mock.patch.object(
            review.inventory, "prepare_terminal_binding", return_value=binding
        ), mock.patch.object(review, "_run_fixed_remote") as transport:
            result = self.call("convir_evidence_cloud_text_read", {
                "local_repo": str(self.repo),
                "snapshot_commit": commit,
                "catalog_sha256": catalog_sha256,
                "terminal_record_sha256": binding["terminal_record_sha256"],
                "inventory_sha256": "6" * 64,
                "relative_path": "workload/details.csv",
            })
        self.assertTrue(result["isError"])
        self.assertEqual(
            "CLOUD_TEXT_PROTECTED_SCOPE", result["structuredContent"]["state"]
        )
        transport.assert_not_called()

    def test_full_jsonrpc_budget_shrinks_cloud_text_without_skipping_bytes(self):
        binding = cloud_binding("a" * 40, "b" * 64)
        raw = ("\\\"\x01" * 3000).encode("utf-8")
        file_sha256 = hashlib.sha256(raw).hexdigest()
        value = {
            "schema_version": 2,
            "ok": True,
            "operation": "cloud-text-read",
            "state": "CLOUD_TEXT_PAGE_OK",
            "exit_code": 0,
            "snapshot_commit": binding["snapshot_commit"],
            "catalog_sha256": binding["catalog_sha256"],
            "terminal_record_sha256": binding["terminal_record_sha256"],
            "inventory_sha256": "c" * 64,
            "relative_path": "workload/escaped.txt",
            "artifact_class": "raw_artifact",
            "policy_assessment": "expected_cloud_only_raw_artifact",
            "evidence_status": "unmapped_raw_text",
            "reconciliation_state": "CLOUD_ONLY",
            "bytes": len(raw),
            "file_sha256": file_sha256,
            "identity_basis": "content_sha256",
            "encoding": "utf-8",
            "page_start_byte": 0,
            "page_end_byte": len(raw),
            "page_bytes": len(raw),
            "page_sha256": file_sha256,
            "content": raw.decode("utf-8"),
            "page_complete": True,
            "terminal_page": True,
            "complete": True,
            "has_more": False,
            "scientific_completeness": "not_assessed",
        }
        result = review._bounded_cloud_text(
            value,
            binding,
            review.TRUSTED_REMOTE_URLS[0],
            binding["snapshot_commit"],
        )
        page = result["structuredContent"]
        self.assertLess(page["page_bytes"], len(raw))
        self.assertEqual(page["page_end_byte"], page["page_bytes"])
        self.assertEqual(
            raw[:page["page_end_byte"]].decode("utf-8"), page["content"]
        )
        self.assertTrue(page["has_more"])
        envelope = {
            "jsonrpc": "2.0",
            "id": review.MAX_REQUEST_ID_PLACEHOLDER,
            "result": result,
        }
        self.assertLessEqual(
            len(review.canonical_bytes(envelope)) + 1,
            review.MAX_JSONRPC_RESPONSE_BYTES,
        )

    def test_full_jsonrpc_budget_shrinks_escaped_cloud_entries(self):
        binding = cloud_binding("a" * 40, "b" * 64)
        inventory_sha256 = "c" * 64
        states = list(review.inventory.RECONCILIATION_STATES)
        terms = ["\\\"\x01"]
        query_sha256 = review.inventory.inventory_query_sha256(
            review._binding_identity(binding), inventory_sha256, states, terms
        )
        entries = [
            {
                "scope": "raw_output",
                "relative_path": ("\\\"\x01" * 80) + f"-{index}",
                "reconciliation_state": "CLOUD_ONLY",
            }
            for index in range(100)
        ]
        page = {
            "schema_version": 1,
            "ok": True,
            "operation": "cloud-inventory-query",
            "state": "INVENTORY_ENTRIES_OK",
            "exit_code": 0,
            "snapshot_commit": binding["snapshot_commit"],
            "terminal_record_sha256": binding["terminal_record_sha256"],
            "inventory_sha256": inventory_sha256,
            "query_sha256": query_sha256,
            "reconciliation_states": states,
            "terms": terms,
            "offset": 0,
            "returned_count": len(entries),
            "total_count": len(entries),
            "entries": entries,
            "next_offset": None,
            "discovery_completeness": "complete",
            "scientific_completeness": "not_assessed",
        }
        result = review._bounded_cloud_query(
            page,
            binding,
            query_sha256,
            review.TRUSTED_REMOTE_URLS[0],
            binding["snapshot_commit"],
        )
        self.assertLess(result["structuredContent"]["returned_count"], 100)
        envelope = {
            "jsonrpc": "2.0",
            "id": review.MAX_REQUEST_ID_PLACEHOLDER,
            "result": result,
        }
        self.assertLessEqual(
            len(review.canonical_bytes(envelope)) + 1,
            review.MAX_JSONRPC_RESPONSE_BYTES,
        )

    def test_mcp_result_budget_and_fresh_stdio_handshake(self):
        record = terminal_record()
        expected_names = [
            f"legacy-{index:03d}-{'x' * 80}" for index in range(100)
        ]
        extras = {
            f"experience_docx/experiment_logs/{name}/summary.json": b"x"
            for name in expected_names
        }
        commit = self.commit_snapshot(record, extras)
        observed = []
        cursor = None
        page_count = 0
        while True:
            arguments = {
                "local_repo": str(self.repo),
                "snapshot_commit": commit,
                "coverage": "unindexed",
                "limit": 100,
            }
            if cursor is not None:
                arguments["cursor"] = cursor
            bounded = self.call("convir_evidence_catalog_query", arguments)
            self.assertFalse(bounded["isError"])
            self.assertLessEqual(
                len(review.canonical_bytes(bounded)), review.MAX_TOOL_RESULT_BYTES
            )
            self.assertEqual(
                bounded["structuredContent"],
                json.loads(bounded["content"][0]["text"]),
            )
            page = bounded["structuredContent"]
            self.assertEqual(len(observed), page["offset"])
            observed.extend(entry["directory_name"] for entry in page["entries"])
            page_count += 1
            if page["complete"]:
                break
            cursor = page["next_cursor"]
        self.assertGreater(page_count, 1)
        self.assertEqual(expected_names, observed)
        self.assertEqual(len(observed), len(set(observed)))

        long_id = "\x01" * 20
        self.assertLessEqual(
            len(review.canonical_bytes(long_id)), review.MAX_REQUEST_ID_BYTES
        )
        oversized_id = "z" * review.MAX_JSONRPC_RESPONSE_BYTES
        requests = [
            {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            },
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {
                    "name": "convir_evidence_catalog_summary",
                    "arguments": {"local_repo": str(self.repo)},
                },
            },
            {
                "jsonrpc": "2.0", "id": long_id, "method": "tools/call",
                "params": {
                    "name": "convir_evidence_catalog_query",
                    "arguments": {
                        "local_repo": str(self.repo),
                        "snapshot_commit": commit,
                        "coverage": "unindexed",
                        "limit": 100,
                    },
                },
            },
            {"jsonrpc": "2.0", "id": oversized_id, "method": "ping", "params": {}},
        ]
        environment = os.environ.copy()
        environment[review.WORKSPACE_ROOT_ENV] = str(self.root)
        completed = subprocess.run(
            [sys.executable, str(review.SELF_PATH)],
            input="".join(json.dumps(item) + "\n" for item in requests),
            text=True,
            capture_output=True,
            check=True,
            timeout=15,
            env=environment,
        )
        lines = completed.stdout.splitlines()
        self.assertTrue(all(
            len(line.encode("utf-8")) + 1 <= review.MAX_JSONRPC_RESPONSE_BYTES
            for line in lines
        ))
        responses = [json.loads(line) for line in lines]
        self.assertEqual([1, 2, 3, long_id, None], [item["id"] for item in responses])
        self.assertFalse(responses[2]["result"]["isError"])
        self.assertFalse(responses[3]["result"]["isError"])
        self.assertEqual(
            responses[3]["result"]["structuredContent"],
            json.loads(responses[3]["result"]["content"][0]["text"]),
        )
        self.assertIn("error", responses[4])


if __name__ == "__main__":
    unittest.main()
