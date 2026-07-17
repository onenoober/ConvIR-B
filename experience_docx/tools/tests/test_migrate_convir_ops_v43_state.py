"""Tests for the signed v4.3 convir-ops state migration."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import convir_ops_mcp as OPS  # noqa: E402
import migrate_convir_ops_v43_state as MIGRATE  # noqa: E402
from test_convir_ops_mcp import context, engineering_terminal, terminal  # noqa: E402


class StateMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = self.root / "state"
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / ".git").mkdir()
        OPS.STATE_DIR = self.state

    def tearDown(self):
        self.temporary.cleanup()

    def write_closeout(self, ctx, terminal_tuple):
        path = (
            self.repo / "experience_docx" / "experiment_logs" / ctx["route_id"]
            / ctx["closeout_filename"]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "route_id": ctx["route_id"], "run_id": ctx["output_id"],
            "route_commit": ctx["route_branch_commit"],
            "runner_sha256": ctx["runner_sha256"],
            **terminal_tuple, "failure_phase": "workload", "returncode": 1,
            "verified_assets": [], "details": {},
        }))

    def receipt(self, ctx):
        payload = {
            "context": ctx, "gpu_index": None,
            "launch_digest": "f" * 64, "issued_at": 1,
        }
        return OPS.write_new_record(
            "receipt", payload,
            {"launched": True, "finish_calls": 1, "finish_closed": "CLOSEOUT_VALIDATED"},
        )

    def test_migration_verifies_and_preserves_receipts_while_removing_expired_plans(self):
        failed_context = context()
        failed_context["output_id"] = "engineering-r1"
        failed_context["closeout_filename"] = "engineering_closeout.json"
        failed_context["allowed_terminal_tuples"] = [engineering_terminal()]
        scientific_context = context()
        scientific_context["output_id"] = "scientific-r1"
        scientific_context["closeout_filename"] = "scientific_closeout.json"
        scientific_context["allowed_terminal_tuples"] = [terminal()]
        failed = self.receipt(failed_context)
        scientific = self.receipt(scientific_context)
        self.write_closeout(failed_context, engineering_terminal())
        self.write_closeout(scientific_context, terminal())
        now = int(time.time())
        expired = OPS.write_new_record(
            "plan", {"expires_at": now - 1, "nonce": "expired"}, {"receipt": None},
        )
        live = OPS.write_new_record(
            "plan", {"expires_at": now + 60, "nonce": "live"}, {"receipt": None},
        )

        report = MIGRATE.migrate(self.state, self.repo, apply=True, now=now)
        self.assertFalse(OPS.record_path("plan", expired).exists())
        self.assertTrue(OPS.record_path("plan", live).exists())
        self.assertEqual(0, report["receipt_records_deleted"])
        with OPS.locked_record("receipt", failed) as record:
            self.assertEqual("ENGINEERING_ARCHIVE_AUTHORIZED", record["finish_closed"])
            self.assertEqual("archive", record["engineering_failure_resolution"])
        with OPS.locked_record("receipt", scientific) as record:
            self.assertEqual("CLOSEOUT_VALIDATED", record["finish_closed"])
            self.assertIsNone(record["engineering_failure_resolution"])
            self.assertEqual("COMPLETED_GATE_PASS", record["terminal_closeout"]["terminal_tuple"]["state"])


if __name__ == "__main__":
    unittest.main()
