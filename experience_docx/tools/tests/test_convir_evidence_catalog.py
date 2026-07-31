#!/usr/bin/env python3
"""Regression tests for the compact experiment evidence catalog."""

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import convir_evidence_catalog as catalog


def common_record(route_id, operation_id, run_id, group, token):
    prefix = f"experience_docx/experiment_logs/{group}"
    return {
        "schema_version": 1,
        "route_id": route_id,
        "operation_id": operation_id,
        "run_id": run_id,
        "state": "COMPLETED_GATE_PASS",
        "decision": f"{operation_id}_PASS",
        "authorizes": "NEXT_STAGE",
        "receipt": token * 64,
        "route_commit": token * 40,
        "contract_path": f"experience_docx/experiment_cards/{route_id}.md",
        "closeout_path": f"{prefix}/{operation_id.lower()}_closeout.json",
        "conclusion_path": f"{prefix}/{operation_id.lower()}_conclusion.json",
        "result_paths": [f"{prefix}/{operation_id.lower()}_summary.json"],
    }


def schema2_record(route_id, operation_id, run_id, group, token, prior=None):
    value = common_record(route_id, operation_id, run_id, group, token)
    value["schema_version"] = 2
    result_path = value["result_paths"][0]
    value.update({
        "contract_bundle": [{
            "path": f"experience_docx/experiment_logs/{group}/launch_contract/{operation_id}/manifest.json",
            "source_path": "experience_docx/route_operations.json",
            "bytes": 10,
            "sha256": token * 64,
        }],
        "prior_terminal_record": prior or {
            "prior_closeout_path": None, "prior_terminal_tuple": None,
        },
        "result_files": [{"path": result_path, "bytes": 20, "sha256": token * 64}],
        "contract_sha256": token * 64,
        "closeout_sha256": token * 64,
        "conclusion_sha256": token * 64,
    })
    return value


def evidence_files(*records):
    files = {}
    for record in records:
        paths = [
            record["closeout_path"], record["conclusion_path"],
            *record["result_paths"],
            *(item["path"] for item in record.get("contract_bundle", [])),
        ]
        files.update({path: b"unread evidence" for path in paths})
    return files


class EvidenceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="convir evidence catalog ")
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.email", "catalog-test@example.invalid")
        self.git("config", "user.name", "Evidence Catalog Test")

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(self.repo), *args],
            capture_output=True, text=True, check=True,
        )

    def commit_snapshot(self, records, files, message="catalog snapshot"):
        index = self.repo / catalog.INDEX_PATH
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        for relative, raw in files.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.git("add", "--", "experience_docx")
        self.git("commit", "-m", message)
        return self.git("rev-parse", "HEAD").stdout.strip()

    def call(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = catalog.main(list(args))
        value = json.loads(output.getvalue())
        self.assertEqual(exit_code, value["exit_code"])
        return value

    def test_catalog_classifies_indexed_and_unindexed_without_result_reads(self):
        record = common_record("route-a", "A0", "run-a", "route-a", "a")
        commit = self.commit_snapshot(
            [record],
            {
                record["closeout_path"]: b"not JSON and intentionally unread",
                record["conclusion_path"]: b"not JSON",
                record["result_paths"][0]: b"not JSON",
                "experience_docx/experiment_logs/legacy/legacy_closeout.json": b"old",
                "experience_docx/experiment_logs/legacy/metric_summary.csv": b"x",
                "experience_docx/experiment_logs/legacy/notes.txt": b"note",
                "experience_docx/experiment_logs/root_helper.sh": b"unread",
            },
        )
        value = catalog.load_catalog(self.repo, commit)
        header = value["header"]
        self.assertEqual(header["terminal_index"]["record_count"], 1)
        self.assertEqual(header["experiment_log_tree"]["directory_count"], 2)
        self.assertEqual(header["experiment_log_tree"]["catalog_entry_count"], 3)
        self.assertEqual(header["experiment_log_tree"]["indexed_directory_count"], 1)
        self.assertEqual(header["experiment_log_tree"]["unindexed_directory_count"], 1)
        self.assertEqual(header["experiment_log_tree"]["loose_file_count"], 1)
        self.assertEqual(header["scientific_completeness"], "not_assessed")
        legacy = next(
            item for item in value["entries"]
            if item.get("directory_name") == "legacy"
        )
        self.assertEqual(legacy["terminal_assessment"], "NOT_ASSESSED")
        self.assertEqual(
            legacy["marker_counts"],
            {"closeout_named": 1, "conclusion_named": 0, "summary_named": 1},
        )
        loose = next(
            item for item in value["entries"]
            if item["record_kind"] == "loose_file"
        )
        self.assertEqual(loose["file_name"], "root_helper.sh")
        self.assertEqual(loose["terminal_assessment"], "NOT_ASSESSED")

        receipt = catalog.completeness_receipt(value)
        self.assertEqual(2, receipt["schema_version"])
        self.assertEqual("incomplete", receipt["review_completeness"])
        self.assertEqual(
            {
                "catalog_entries": 3,
                "indexed_entries": 1,
                "unindexed_entries": 2,
                "evidence_directories": 2,
                "indexed_directories": 1,
                "unindexed_directories": 1,
                "loose_files": 1,
                "partition_complete": True,
            },
            receipt["entry_partition"],
        )
        self.assertEqual(
            2, receipt["unresolved_counts"]["unclassified_unindexed_entries"]
        )
        self.assertEqual(
            1, receipt["unresolved_counts"]["path_only_legacy_terminal_records"]
        )
        self.assertEqual(
            receipt["receipt_sha256"],
            catalog.completeness_receipt(value)["receipt_sha256"],
        )
        unhashed = dict(receipt)
        del unhashed["receipt_sha256"]
        self.assertEqual(receipt["receipt_sha256"], catalog.canonical_sha256(unhashed))

    def test_terminal_resolution_preserves_valid_chain_and_legacy_ambiguity(self):
        root = schema2_record("route-chain", "A0", "run-a0", "route-chain", "a")
        prior = {
            "prior_closeout_path": root["closeout_path"],
            "prior_terminal_tuple": {
                "state": root["state"], "decision": root["decision"],
                "authorizes": root["authorizes"],
            },
        }
        leaf = schema2_record(
            "route-chain", "A1", "run-a1", "route-chain", "b", prior=prior
        )
        legacy_a = common_record("route-legacy", "B0", "run-b0", "route-legacy", "c")
        legacy_b = common_record("route-legacy", "B1", "run-b1", "route-legacy", "d")
        records = [root, leaf, legacy_a, legacy_b]
        files = evidence_files(*records)
        commit = self.commit_snapshot(records, files)
        value = catalog.load_catalog(self.repo, commit)
        self.assertEqual(value["header"]["terminal_index"]["unmodeled_record_count"], 0)
        routes = {
            route["route_id"]: route
            for entry in value["entries"] for route in entry["routes"]
        }
        self.assertEqual(routes["route-chain"]["terminal_resolution"], "VALID_CHAIN")
        self.assertEqual(routes["route-chain"]["selected_operation_id"], "A1")
        self.assertEqual(
            routes["route-legacy"]["terminal_resolution"], "AMBIGUOUS_LEGACY"
        )
        self.assertIsNone(routes["route-legacy"]["selected_operation_id"])
        self.assertEqual(len(routes["route-legacy"]["terminals"]), 2)

        receipt = catalog.completeness_receipt(value)
        self.assertEqual(
            1, receipt["unresolved_counts"]["ambiguous_legacy_routes"]
        )
        self.assertEqual(1, receipt["terminal_partition"]["unresolved_routes"])
        self.assertEqual(
            {"path_only_legacy": 2, "sha256_manifest": 2},
            receipt["terminal_partition"]["binding_counts"],
        )

        duplicate_run = schema2_record(
            "route-chain", "A2", "run-a1", "route-chain", "e",
            prior={
                "prior_closeout_path": leaf["closeout_path"],
                "prior_terminal_tuple": {
                    "state": leaf["state"], "decision": leaf["decision"],
                    "authorizes": leaf["authorizes"],
                },
            },
        )
        parsed = catalog.parse_terminal_index(
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in (root, leaf, duplicate_run)
            ).encode()
        )
        self.assertEqual(
            catalog.select_terminal_leaf(parsed)[1], "INVALID_CHAIN"
        )

    def test_schema2_only_catalog_has_a_complete_stable_cli_receipt(self):
        record = schema2_record("route-a", "A0", "run-a", "route-a", "a")
        commit = self.commit_snapshot([record], evidence_files(record))
        receipt = self.call(
            "--repo", str(self.repo), "--commit", commit, "receipt"
        )
        self.assertTrue(receipt["ok"])
        self.assertEqual("complete", receipt["review_completeness"])
        self.assertEqual(0, receipt["terminal_partition"]["unresolved_routes"])
        self.assertFalse(any(receipt["unresolved_counts"].values()))
        self.assertEqual(
            receipt["source_identities"]["catalog_sha256"],
            catalog.load_catalog(self.repo, commit)["catalog_sha256"],
        )
        self.assertLessEqual(
            len(catalog.canonical_bytes(receipt)) + 1, catalog.MAX_RESPONSE_BYTES
        )

    def test_schema2_review_facts_recovery_is_modeled_and_identity_bound(self):
        record = schema2_record("route-a", "A0", "run-a", "route-a", "a")
        prefix = "experience_docx/experiment_logs/route-a"
        original_path = f"{prefix}/a0_review_facts.json"
        proof_path = f"{prefix}/a0_review_facts_recovery.json"
        record["result_paths"] = [original_path, proof_path]
        record["result_files"] = [
            {"path": original_path, "bytes": 20, "sha256": "b" * 64},
            {"path": proof_path, "bytes": 40, "sha256": "c" * 64},
        ]
        record["review_facts_recovery"] = {
            "status": "REVIEW_FACTS_RECOVERED",
            "recovery_type": catalog.REVIEW_FACTS_RECOVERY_TYPE,
            "proof_path": proof_path,
            "proof_bytes": 40,
            "proof_sha256": "c" * 64,
            "original_path": original_path,
            "original_sha256": "b" * 64,
            "recovered_review_facts_sha256": "d" * 64,
        }
        parsed = catalog.parse_terminal_index(
            (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
        self.assertEqual([], parsed[0]["unmodeled_fields"])
        self.assertEqual(
            "REVIEW_FACTS_RECOVERED",
            parsed[0]["review_facts_recovery"]["status"],
        )
        self.assertEqual(
            "REVIEW_FACTS_RECOVERED",
            catalog.terminal_summary(parsed[0])["review_facts_recovery_status"],
        )

        changed = json.loads(json.dumps(record))
        changed["review_facts_recovery"]["proof_sha256"] = "e" * 64
        with self.assertRaisesRegex(catalog.CatalogError, "result identity differs"):
            catalog.parse_terminal_index(
                (json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )

    def test_completeness_receipt_rejects_partition_drift(self):
        record = common_record("route-a", "A0", "run-a", "route-a", "a")
        commit = self.commit_snapshot([record], evidence_files(record))
        value = catalog.load_catalog(self.repo, commit)
        value["entries"][0]["index_coverage"] = "UNKNOWN"
        with self.assertRaisesRegex(catalog.CatalogError, "unknown index partition"):
            catalog.completeness_receipt(value)

    def test_schema2_single_record_must_still_have_a_valid_root(self):
        record = schema2_record("route-a", "A0", "run-a", "route-a", "a")
        record["prior_terminal_record"] = {
            "prior_closeout_path": (
                "experience_docx/experiment_logs/route-a/missing_closeout.json"
            ),
            "prior_terminal_tuple": {
                "state": "COMPLETED_GATE_PASS",
                "decision": "MISSING_PASS",
                "authorizes": "NEXT_STAGE",
            },
        }
        commit = self.commit_snapshot([record], evidence_files(record))
        value = catalog.load_catalog(self.repo, commit)
        route = value["entries"][0]["routes"][0]
        self.assertEqual(route["terminal_resolution"], "INVALID_CHAIN")
        self.assertIsNone(route["selected_operation_id"])

    def test_catalog_rejects_route_split_across_directories(self):
        first = common_record("route-a", "A0", "run-a0", "group-a", "a")
        second = common_record("route-a", "A1", "run-a1", "group-b", "b")
        commit = self.commit_snapshot(
            [first, second], evidence_files(first, second)
        )
        with self.assertRaisesRegex(catalog.CatalogError, "multiple evidence directories"):
            catalog.load_catalog(self.repo, commit)

    def test_catalog_rejects_missing_index_reference(self):
        record = common_record("route-a", "A0", "run-a", "route-a", "a")
        files = evidence_files(record)
        del files[record["closeout_path"]]
        commit = self.commit_snapshot([record], files)
        with self.assertRaisesRegex(catalog.CatalogError, "reference is missing"):
            catalog.load_catalog(self.repo, commit)

    def test_snapshot_is_stable_and_invalid_index_fails_closed(self):
        first = common_record("route-a", "A0", "run-a", "route-a", "a")
        first_commit = self.commit_snapshot(
            [first], evidence_files(first), "first snapshot"
        )
        second = common_record("route-b", "B0", "run-b", "route-b", "b")
        second_commit = self.commit_snapshot(
            [first, second],
            evidence_files(first, second),
            "second snapshot",
        )
        self.assertEqual(
            catalog.load_catalog(self.repo, first_commit)["header"]["terminal_index"]["record_count"],
            1,
        )
        self.assertEqual(
            catalog.load_catalog(self.repo, second_commit)["header"]["terminal_index"]["record_count"],
            2,
        )

        broken = schema2_record("route-c", "C0", "run-c", "route-c", "c")
        broken["result_files"][0]["path"] = "experience_docx/experiment_logs/route-c/other.json"
        with self.assertRaisesRegex(catalog.CatalogError, "result_paths and result_files"):
            catalog.parse_terminal_index(
                (json.dumps(broken, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )

    def test_catalog_entry_pages_are_bounded_and_reassemble_exactly(self):
        record = common_record("route-a", "A0", "run-a", "route-a", "a")
        files = evidence_files(record)
        for index in range(45):
            files[f"experience_docx/experiment_logs/legacy-{index:03d}/summary.json"] = b"x"
        commit = self.commit_snapshot([record], files)
        expected = [
            entry["directory_name"]
            for entry in catalog.load_catalog(self.repo, commit)["entries"]
            if entry["index_coverage"] == "UNINDEXED"
        ]
        observed = []
        cursor = None
        while True:
            args = [
                "--repo", str(self.repo), "--commit", commit, "entries",
                "--coverage", "unindexed", "--limit", "7",
            ]
            if cursor is not None:
                args.extend(["--cursor", cursor])
            page = self.call(*args)
            self.assertTrue(page["ok"])
            self.assertLessEqual(
                len(catalog.canonical_bytes(page)) + 1, catalog.MAX_RESPONSE_BYTES
            )
            self.assertEqual(page["offset"], len(observed))
            observed.extend(entry["directory_name"] for entry in page["entries"])
            if page["complete"]:
                self.assertIsNone(page["next_cursor"])
                break
            cursor = page["next_cursor"]
            self.assertIsInstance(cursor, str)
        self.assertEqual(observed, expected)
        self.assertEqual(len(observed), len(set(observed)))

    def test_catalog_cursor_rejects_filter_drift(self):
        record = common_record("route-a", "A0", "run-a", "route-a", "a")
        commit = self.commit_snapshot(
            [record],
            {
                **evidence_files(record),
                "experience_docx/experiment_logs/legacy-a/summary.json": b"x",
                "experience_docx/experiment_logs/legacy-b/summary.json": b"x",
            },
        )
        first = self.call(
            "--repo", str(self.repo), "--commit", commit, "entries",
            "--coverage", "unindexed", "--limit", "1",
        )
        self.assertFalse(first["complete"])
        drifted = self.call(
            "--repo", str(self.repo), "--commit", commit, "entries",
            "--coverage", "all", "--limit", "1", "--cursor", first["next_cursor"],
        )
        self.assertFalse(drifted["ok"])
        self.assertEqual(drifted["state"], "REPO_CURSOR_IDENTITY_MISMATCH")


if __name__ == "__main__":
    unittest.main()
