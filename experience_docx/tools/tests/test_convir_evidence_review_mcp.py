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


def context_terminal_record(
    trigger_route_id="prior-route", trigger_sha="f" * 64,
    snapshot_commit="a" * 40, route_id="route-a", program_id="program-a",
):
    record = terminal_record(route_id=route_id, group=route_id)
    prefix = f"experience_docx/experiment_logs/{route_id}/launch_contract/A0"
    program_source = f"experience_docx/research_programs/{program_id}.json"
    spec_source = f"experience_docx/experiment_specs/{route_id}.json"
    program_raw = json.dumps(
        {"program_id": program_id}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    spec_raw = json.dumps({
        "schema_version": 3,
        "route_id": route_id,
        "program_contract_relpath": program_source,
        "operations": {
            "A0": {
                "program_authorization": {
                    "program_id": program_id,
                    "family_id": "family-a",
                    "stage_id": "screening",
                    "mechanism_type": "orthogonal",
                },
                "scientific_contract": {
                    "research_update_binding": {
                        "snapshot_commit": snapshot_commit,
                        "trigger_type": "post_terminal",
                        "trigger_terminals": [{
                            "route_id": trigger_route_id,
                            "terminal_record_sha256": trigger_sha,
                        }],
                        "bottleneck_class": "scientific_hypothesis",
                        "bottleneck_statement": (
                            "A valid terminal left the preferred mechanism unsupported."
                        ),
                        "literature_basis": [{
                            "identifier": "doi:10.0000/example",
                            "source_status": "peer_reviewed",
                            "task": "image restoration",
                            "transferable_claim": (
                                "Conditional processing may separate heterogeneous effects."
                            ),
                            "applicability_limit": (
                                "The publication does not validate this project result."
                            ),
                        }],
                        "hypotheses": [{"id": "conditional_capacity"}, {
                            "id": "measurement_mismatch",
                        }],
                        "design_selection": {"strategy": "multi_arm"},
                    },
                },
            },
        },
    }, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    program_path = f"{prefix}/program_contract.json"
    spec_path = f"{prefix}/experiment_spec.json"
    result_raw = b"not read"
    contract_raw = b"contract\n"
    closeout_raw = b"not read"
    conclusion_raw = b"not read"
    record.update({
        "schema_version": 2,
        "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
        "closeout_sha256": hashlib.sha256(closeout_raw).hexdigest(),
        "conclusion_sha256": hashlib.sha256(conclusion_raw).hexdigest(),
        "contract_bundle": [
            {
                "path": program_path, "source_path": program_source,
                "bytes": len(program_raw),
                "sha256": hashlib.sha256(program_raw).hexdigest(),
            },
            {
                "path": spec_path, "source_path": spec_source,
                "bytes": len(spec_raw),
                "sha256": hashlib.sha256(spec_raw).hexdigest(),
            },
        ],
        "result_files": [{
            "path": record["result_paths"][0], "bytes": len(result_raw),
            "sha256": hashlib.sha256(result_raw).hexdigest(),
        }],
        "prior_terminal_record": {
            "prior_closeout_path": None, "prior_terminal_tuple": None,
        },
    })
    files = {
        **evidence_files(record),
        record["contract_path"]: contract_raw,
        record["closeout_path"]: closeout_raw,
        record["conclusion_path"]: conclusion_raw,
        record["result_paths"][0]: result_raw,
        program_path: program_raw,
        spec_path: spec_raw,
    }
    return record, files


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
        with review._CATALOG_CACHE_LOCK:
            review._CATALOG_CACHE.clear()
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
        records = record if isinstance(record, list) else [record]
        index = self.repo / review.catalog.INDEX_PATH
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(
            "".join(
                json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                for item in records
            ),
            encoding="utf-8",
        )
        files = {}
        for item in records:
            files.update(evidence_files(item))
        files.update(extra_files or {})
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

    def test_binding_identity_matches_inventory_contract(self):
        binding = cloud_binding("a" * 40, "b" * 64)
        binding["raw_artifact_receipt_sha256"] = "7" * 64
        binding["raw_artifact_receipt"] = {"manifest_sha256": "8" * 64}
        generated = review.inventory._inventory(
            binding,
            state="INVENTORY_READY",
            discovery_completeness="complete",
            entries=[],
            scan=None,
            limits=review.inventory._limits(None),
            issues=[],
        )
        self.assertEqual(review._binding_identity(binding), generated["identity"])

    def test_server_exposes_exact_six_read_only_tools(self):
        initialized = review.handle({
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })
        self.assertEqual("convir-evidence-review", initialized["serverInfo"]["name"])
        self.assertEqual("2.2.0", initialized["serverInfo"]["version"])
        listed = review.handle({"method": "tools/list", "params": {}})
        self.assertEqual(
            [
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

    def test_catalog_cache_reuses_one_commit_and_isolates_another(self):
        first = {"catalog_sha256": "a" * 64}
        second = {"catalog_sha256": "b" * 64}
        with mock.patch.object(
            review.catalog, "load_catalog", side_effect=[first, second],
        ) as load_catalog:
            first_read = review.load_catalog_cached(self.repo, "1" * 40)
            repeated = review.load_catalog_cached(self.repo, "1" * 40)
            other_commit = review.load_catalog_cached(self.repo, "2" * 40)
        self.assertIs(first, first_read)
        self.assertIs(first_read, repeated)
        self.assertIs(second, other_commit)
        self.assertEqual(2, load_catalog.call_count)

    def test_catalog_relationship_filters_use_terminal_bound_launch_contracts(self):
        prior = terminal_record(route_id="prior-route", group="prior-route")
        prior["receipt"] = "c" * 64
        prior_raw = json.dumps(
            prior, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        prior_sha = hashlib.sha256(prior_raw).hexdigest()
        prior_commit = self.commit_snapshot(prior, message="prior terminal")
        record, files = context_terminal_record(
            trigger_sha=prior_sha, snapshot_commit=prior_commit,
        )
        commit = self.commit_snapshot([prior, record], files)
        result = self.call("convir_evidence_catalog_query", {
            "local_repo": str(self.repo),
            "snapshot_commit": commit,
            "program_ids": ["program-a"],
            "family_ids": ["family-a"],
            "mechanism_types": ["orthogonal"],
            "trigger_route_ids": ["prior-route"],
            "trigger_terminal_record_sha256s": [prior_sha],
        })
        self.assertFalse(result["isError"])
        value = result["structuredContent"]
        self.assertEqual(1, value["total_count"])
        context = value["entries"][0]["routes"][0]["research_context"]
        self.assertEqual("modeled", context["relationship_status"])
        self.assertEqual("program-a", context["program_id"])
        self.assertEqual("family-a", context["family_id"])
        self.assertEqual(["prior-route"], context["trigger_route_ids"])
        self.assertEqual("multi_arm", context["design_strategy"])
        self.assertIn("research_context_collection_sha256", value)

    def test_research_trigger_must_exist_in_the_frozen_snapshot(self):
        frozen = terminal_record(route_id="frozen-route", group="frozen-route")
        frozen["receipt"] = "b" * 64
        frozen_commit = self.commit_snapshot(frozen, message="frozen research snapshot")
        prior = terminal_record(route_id="prior-route", group="prior-route")
        prior["receipt"] = "c" * 64
        prior_raw = json.dumps(
            prior, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        record, files = context_terminal_record(
            trigger_sha=hashlib.sha256(prior_raw).hexdigest(),
            snapshot_commit=frozen_commit,
        )
        commit = self.commit_snapshot([frozen, prior, record], files)
        result = self.call("convir_evidence_catalog_query", {
            "local_repo": str(self.repo),
            "snapshot_commit": commit,
            "route_ids": ["route-a"],
        })
        self.assertFalse(result["isError"])
        context = result["structuredContent"]["entries"][0]["routes"][0][
            "research_context"
        ]
        self.assertEqual("identity_conflict", context["relationship_status"])
        self.assertIn("absent from the frozen snapshot", context["reason"])

    def test_malformed_relationship_is_a_local_identity_conflict(self):
        prior = terminal_record(route_id="prior-route", group="prior-route")
        prior["receipt"] = "c" * 64
        prior_raw = json.dumps(
            prior, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        prior_commit = self.commit_snapshot(prior, message="prior terminal")
        record, files = context_terminal_record(
            trigger_sha=hashlib.sha256(prior_raw).hexdigest(),
            snapshot_commit=prior_commit,
        )
        binding = next(
            item for item in record["contract_bundle"]
            if item["source_path"].startswith("experience_docx/experiment_specs/")
        )
        spec = json.loads(files[binding["path"]])
        spec["operations"] = []
        raw = json.dumps(
            spec, sort_keys=True, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        files[binding["path"]] = raw
        binding["bytes"] = len(raw)
        binding["sha256"] = hashlib.sha256(raw).hexdigest()
        commit = self.commit_snapshot([prior, record], files)
        result = self.call("convir_evidence_catalog_query", {
            "local_repo": str(self.repo),
            "snapshot_commit": commit,
            "route_ids": ["route-a"],
        })
        self.assertFalse(result["isError"])
        context = result["structuredContent"]["entries"][0]["routes"][0][
            "research_context"
        ]
        self.assertEqual("identity_conflict", context["relationship_status"])
        self.assertIn("operations are not an object", context["reason"])

    def test_program_foundation_relationship_has_no_invented_trigger(self):
        prior = terminal_record(route_id="unrelated", group="unrelated")
        prior["receipt"] = "c" * 64
        prior_commit = self.commit_snapshot(prior, message="main before foundation")
        record, files = context_terminal_record(snapshot_commit=prior_commit)
        binding = next(
            item for item in record["contract_bundle"]
            if item["source_path"].startswith("experience_docx/experiment_specs/")
        )
        spec = json.loads(files[binding["path"]])
        claim = spec["operations"]["A0"]["program_authorization"]
        claim["mechanism_type"] = "adjacent"
        claim["adjacent_sequence"] = 1
        update = spec["operations"]["A0"]["scientific_contract"][
            "research_update_binding"
        ]
        update["trigger_type"] = "program_foundation"
        update["trigger_terminals"] = []
        raw = json.dumps(
            spec, sort_keys=True, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        files[binding["path"]] = raw
        binding["bytes"] = len(raw)
        binding["sha256"] = hashlib.sha256(raw).hexdigest()
        program_binding = next(
            item for item in record["contract_bundle"]
            if item["source_path"].startswith("experience_docx/research_programs/")
        )
        program = {
            "program_id": "program-a",
            "route_families": {
                "family-a": {"state": "open", "attempts_used": 0},
            },
        }
        program_raw = json.dumps(
            program, sort_keys=True, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        files[program_binding["path"]] = program_raw
        program_binding["bytes"] = len(program_raw)
        program_binding["sha256"] = hashlib.sha256(program_raw).hexdigest()
        commit = self.commit_snapshot(record, files, message="foundation terminal")
        result = self.call("convir_evidence_catalog_query", {
            "local_repo": str(self.repo),
            "snapshot_commit": commit,
            "route_ids": ["route-a"],
        })
        self.assertFalse(result["isError"])
        context = result["structuredContent"]["entries"][0]["routes"][0][
            "research_context"
        ]
        self.assertEqual("program_foundation", context["research_trigger_type"])
        self.assertEqual([], context["trigger_route_ids"])

    def test_program_foundation_rejects_an_already_archived_program(self):
        prior = terminal_record(route_id="prior-route", group="prior-route")
        prior["receipt"] = "c" * 64
        prior_raw = json.dumps(
            prior, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        prior_sha = hashlib.sha256(prior_raw).hexdigest()
        prior_commit = self.commit_snapshot(prior, message="prior terminal")
        previous, previous_files = context_terminal_record(
            trigger_sha=prior_sha, snapshot_commit=prior_commit,
            route_id="previous-route",
        )
        previous["receipt"] = "d" * 64
        previous_commit = self.commit_snapshot(
            [prior, previous], previous_files, message="existing program terminal",
        )
        foundation, files = context_terminal_record(
            snapshot_commit=previous_commit,
        )
        foundation["receipt"] = "e" * 64
        spec_binding = next(
            item for item in foundation["contract_bundle"]
            if item["source_path"].startswith("experience_docx/experiment_specs/")
        )
        spec = json.loads(files[spec_binding["path"]])
        claim = spec["operations"]["A0"]["program_authorization"]
        claim["mechanism_type"] = "adjacent"
        claim["adjacent_sequence"] = 1
        update = spec["operations"]["A0"]["scientific_contract"][
            "research_update_binding"
        ]
        update["trigger_type"] = "program_foundation"
        update["trigger_terminals"] = []
        spec_raw = json.dumps(
            spec, sort_keys=True, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        files[spec_binding["path"]] = spec_raw
        spec_binding["bytes"] = len(spec_raw)
        spec_binding["sha256"] = hashlib.sha256(spec_raw).hexdigest()
        program_binding = next(
            item for item in foundation["contract_bundle"]
            if item["source_path"].startswith("experience_docx/research_programs/")
        )
        program_raw = json.dumps({
            "program_id": "program-a",
            "route_families": {
                "family-a": {"state": "open", "attempts_used": 0},
            },
        }, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        files[program_binding["path"]] = program_raw
        program_binding["bytes"] = len(program_raw)
        program_binding["sha256"] = hashlib.sha256(program_raw).hexdigest()
        commit = self.commit_snapshot(
            [prior, previous, foundation], files, message="invalid foundation terminal",
        )
        result = self.call("convir_evidence_catalog_query", {
            "local_repo": str(self.repo),
            "snapshot_commit": commit,
            "route_ids": ["route-a"],
        })
        context = result["structuredContent"]["entries"][0]["routes"][0][
            "research_context"
        ]
        self.assertEqual("identity_conflict", context["relationship_status"])
        self.assertIn("not a first-route program binding", context["reason"])

    def test_legacy_relationship_is_not_guessed_from_route_or_directory_name(self):
        record = terminal_record(route_id="program-a-family-a")
        commit = self.commit_snapshot(record)
        by_route = self.call("convir_evidence_catalog_query", {
            "local_repo": str(self.repo),
            "snapshot_commit": commit,
            "route_ids": ["program-a-family-a"],
        })["structuredContent"]
        context = by_route["entries"][0]["routes"][0]["research_context"]
        self.assertEqual("not_modeled", context["relationship_status"])
        by_program = self.call("convir_evidence_catalog_query", {
            "local_repo": str(self.repo),
            "snapshot_commit": commit,
            "program_ids": ["program-a"],
        })["structuredContent"]
        self.assertEqual(0, by_program["total_count"])

    def test_evidence_bundle_resolves_sha_bound_leaf_and_pages_all_files(self):
        record = terminal_record()
        prefix = "experience_docx/experiment_logs/route-a"
        contract_raw = b"contract\n"
        closeout_raw = b"not read"
        conclusion_raw = b"not read"
        result_raw = b"not read"
        receipt_raw = b'{"schema_version":2}\n'
        facts_raw = b'{"schema_version":2,"facts":[]}\n'
        recovery_raw = b'{"status":"REVIEW_FACTS_RECOVERED"}\n'
        manifest_raw = b"{}\n"
        manifest_path = f"{prefix}/launch_contract/A0/manifest.json"
        receipt_path = f"{prefix}/route_raw_artifact_receipt.json"
        facts_path = f"{prefix}/route_review_facts.json"
        recovery_path = f"{prefix}/route_review_facts_recovery.json"
        record["result_paths"].append(receipt_path)
        record["result_paths"].append(facts_path)
        record["result_paths"].append(recovery_path)
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
            }, {
                "path": receipt_path,
                "bytes": len(receipt_raw),
                "sha256": hashlib.sha256(receipt_raw).hexdigest(),
            }, {
                "path": facts_path,
                "bytes": len(facts_raw),
                "sha256": hashlib.sha256(facts_raw).hexdigest(),
            }, {
                "path": recovery_path,
                "bytes": len(recovery_raw),
                "sha256": hashlib.sha256(recovery_raw).hexdigest(),
            }],
            "prior_terminal_record": {
                "prior_closeout_path": None,
                "prior_terminal_tuple": None,
            },
            "review_facts_recovery": {
                "status": "REVIEW_FACTS_RECOVERED",
                "recovery_type": review.catalog.REVIEW_FACTS_RECOVERY_TYPE,
                "proof_path": recovery_path,
                "proof_bytes": len(recovery_raw),
                "proof_sha256": hashlib.sha256(recovery_raw).hexdigest(),
                "original_path": facts_path,
                "original_sha256": hashlib.sha256(facts_raw).hexdigest(),
                "recovered_review_facts_sha256": "9" * 64,
            },
        })
        commit = self.commit_snapshot(record, {
            record["contract_path"]: contract_raw,
            record["closeout_path"]: closeout_raw,
            record["conclusion_path"]: conclusion_raw,
            record["result_paths"][0]: result_raw,
            receipt_path: receipt_raw,
            facts_path: facts_raw,
            recovery_path: recovery_raw,
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
            "raw_artifact_receipt_github_path": receipt_path,
            "unmapped_results": [{"github_path": recovery_path}],
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
        self.assertTrue(any(
            "raw_artifact_receipt" in item["roles"] for item in files
        ))
        self.assertTrue(any(
            "review_facts" in item["roles"] for item in files
        ))
        self.assertTrue(any(
            "review_facts_recovery" in item["roles"] for item in files
        ))
        self.assertEqual(
            "REVIEW_FACTS_RECOVERED",
            value["lineage"][0]["review_facts_recovery_status"],
        )
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

    def test_completeness_freezes_symbolic_ref_before_repository_moves(self):
        record = terminal_record()
        first = self.commit_snapshot(record)
        summary = self.call(
            "convir_evidence_completeness_receipt",
            {"local_repo": str(self.repo)},
        )
        self.assertFalse(summary["isError"])
        self.assertEqual(first, summary["structuredContent"]["snapshot_commit"])
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
            "convir_evidence_completeness_receipt",
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
            "convir_evidence_completeness_receipt", {"local_repo": str(self.repo)}
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
                    "name": "convir_evidence_completeness_receipt",
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
