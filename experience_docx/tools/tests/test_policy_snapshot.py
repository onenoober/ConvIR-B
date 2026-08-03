"""Tests for the compact deterministic policy read index."""

import copy
import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import policy_snapshot as SNAPSHOT  # noqa: E402


class PolicySnapshotTests(unittest.TestCase):
    def sources(self):
        return {
            relpath: f"authoritative rule source: {relpath}\n".encode()
            for relpath in SNAPSHOT.POLICY_SOURCES
        }

    def test_snapshot_is_deterministic_and_routes_reads_to_full_authority(self):
        sources = self.sources()
        first = SNAPSHOT.build_snapshot(
            rules_commit="a" * 40, read_bytes=sources.__getitem__,
        )
        second = SNAPSHOT.build_snapshot(
            rules_commit="a" * 40, read_bytes=copy.deepcopy(sources).__getitem__,
        )
        self.assertEqual(first, second)
        self.assertEqual("deterministic_read_index_not_policy_authority", first["snapshot_role"])
        self.assertFalse(first["change_routes"]["route_authoring"]["snapshot_is_authority"])
        self.assertIn("AGENTS.md", first["change_routes"]["governance_change"]["read_full"])
        self.assertIn(
            "experience_docx/EXPERIMENT_GOVERNANCE_PROTOCOL.md",
            first["change_routes"]["read_only_audit"]["read_full"],
        )
        self.assertIn("orthogonal", first["route_mechanisms"])
        self.assertFalse(first["local_runtime_allowed"])

    def test_any_authoritative_rule_change_invalidates_snapshot(self):
        sources = self.sources()
        value = SNAPSHOT.build_snapshot(
            rules_commit="a" * 40, read_bytes=sources.__getitem__,
        )
        changed = dict(sources)
        changed["AGENTS.md"] += b"changed\n"
        with self.assertRaises(SNAPSHOT.PolicySnapshotError):
            SNAPSHOT.verify_snapshot(value, read_bytes=changed.__getitem__)

    def test_missing_source_and_invalid_commit_fail_closed(self):
        sources = self.sources()
        sources.pop("AGENTS.md")
        with self.assertRaises(SNAPSHOT.PolicySnapshotError):
            SNAPSHOT.build_snapshot(
                rules_commit="a" * 40, read_bytes=sources.__getitem__,
            )
        with self.assertRaises(SNAPSHOT.PolicySnapshotError):
            SNAPSHOT.build_snapshot(
                rules_commit="main", read_bytes=self.sources().__getitem__,
            )

    def test_snapshot_tamper_is_rejected_even_when_sources_are_unchanged(self):
        sources = self.sources()
        value = SNAPSHOT.build_snapshot(
            rules_commit="a" * 40, read_bytes=sources.__getitem__,
        )
        value["local_runtime_allowed"] = True
        with self.assertRaises(SNAPSHOT.PolicySnapshotError):
            SNAPSHOT.verify_snapshot(value, read_bytes=sources.__getitem__)


if __name__ == "__main__":
    unittest.main()
