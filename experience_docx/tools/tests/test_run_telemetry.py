"""Runtime tests for metadata-only, fail-open route telemetry."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "run_telemetry.py"
SPEC = importlib.util.spec_from_file_location("run_telemetry", MODULE_PATH)
TELEMETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TELEMETRY)


class RunTelemetryTests(unittest.TestCase):
    def test_pulse_is_atomic_and_identity_bound(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "heartbeat.json"
            self.assertEqual(0, TELEMETRY.main([
                "pulse", "--route-id", "route-a", "--run-id", "run-1",
                "--phase", "unit", "--completed", "2", "--total", "5",
                "--heartbeat", str(path), "--sequence", "7",
            ]))
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(("route-a", "run-1"), (value["route_id"], value["run_id"]))
            self.assertEqual((2, 5, 7), (value["completed"], value["total"], value["sequence"]))
            self.assertEqual([], list(Path(root).glob("*.tmp")))

    def test_event_appends_fixed_metadata_only(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "status.txt"
            for event in ("stage_start", "unit_complete"):
                self.assertEqual(0, TELEMETRY.main([
                    "event", "--route-id", "route-a", "--run-id", "run-1",
                    "--phase", "unit", "--status", str(path), "--event", event,
                    "--completed", "1", "--total", "2",
                ]))
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(["stage_start", "unit_complete"], [row["event"] for row in rows])
            forbidden = {"metric", "result", "checkpoint", "image", "gpu"}
            self.assertFalse(forbidden & set().union(*(row.keys() for row in rows)))

    def test_sidecar_does_not_control_parent_workload(self):
        with tempfile.TemporaryDirectory() as root:
            heartbeat = Path(root) / "heartbeat.json"
            workload = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.4); print('WORKLOAD_OK')"], stdout=subprocess.PIPE, text=True)
            sidecar = subprocess.run([
                sys.executable, str(MODULE_PATH), "sidecar",
                "--route-id", "route-a", "--run-id", "run-1", "--phase", "work",
                "--heartbeat", str(heartbeat), "--parent-pid", str(workload.pid),
                "--interval-seconds", "0.03", "--max-pulses", "3",
            ], check=False, capture_output=True, text=True, timeout=5)
            output, _ = workload.communicate(timeout=5)
            self.assertEqual(0, sidecar.returncode)
            self.assertEqual(0, workload.returncode)
            self.assertIn("WORKLOAD_OK", output)
            self.assertTrue(heartbeat.is_file())

    def test_unwritable_telemetry_is_fail_open(self):
        started = time.monotonic()
        result = subprocess.run([
            sys.executable, str(MODULE_PATH), "sidecar",
            "--route-id", "route-a", "--run-id", "run-1", "--phase", "work",
            "--heartbeat", "/proc/1/heartbeat.json", "--parent-pid", str(__import__("os").getpid()),
            "--interval-seconds", "0.02", "--max-pulses", "1",
        ], check=False, capture_output=True, text=True, timeout=5)
        self.assertEqual(0, result.returncode)
        self.assertIn("RUN_TELEMETRY_DEGRADED", result.stderr)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_destination_name_and_symlink_are_rejected_fail_open(self):
        with tempfile.TemporaryDirectory() as root:
            wrong = Path(root) / "result.json"
            self.assertEqual(0, TELEMETRY.main([
                "pulse", "--route-id", "route-a", "--run-id", "run-1",
                "--phase", "work", "--heartbeat", str(wrong),
            ]))
            self.assertFalse(wrong.exists())
            target = Path(root) / "target.txt"
            target.write_text("KEEP", encoding="utf-8")
            link = Path(root) / "heartbeat.json"
            link.symlink_to(target)
            self.assertEqual(0, TELEMETRY.main([
                "pulse", "--route-id", "route-a", "--run-id", "run-1",
                "--phase", "work", "--heartbeat", str(link),
            ]))
            self.assertEqual("KEEP", target.read_text(encoding="utf-8"))

    def test_source_contains_no_process_or_gpu_control(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for forbidden in ("os.kill", "signal.", "terminate(", "nvidia-smi", "CUDA"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
