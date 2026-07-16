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
AUDIT_PATH = Path(__file__).parents[1] / "audit_run_telemetry.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("audit_run_telemetry", AUDIT_PATH)
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)


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

    def test_semantic_source_audit_rejects_control_and_non_proc_reads(self):
        self.assertEqual([], AUDIT.audit_path(MODULE_PATH))
        safe_prose = '''
"""Without sending any signal. No nvidia-smi or CUDA control is used."""
from pathlib import Path
def process_start_ticks(pid):
    return (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
'''
        self.assertEqual([], AUDIT.audit_source(safe_prose))
        unsafe_cases = {
            "signal_import": "import signal\ndef process_start_ticks(pid): return '/proc'\n",
            "kill_call": "import os\ndef process_start_ticks(pid): return '/proc'\nos.kill(1, 9)\n",
            "gpu_command": "import os\ndef process_start_ticks(pid): return '/proc'\nos.system('nvidia-smi')\n",
            "data_read": "from pathlib import Path\ndef process_start_ticks(pid): return '/proc'\nPath('/data/result').read_text()\n",
            "read_hidden_in_proc_function": "from pathlib import Path\ndef process_start_ticks(pid):\n    marker='/proc'\n    return Path('/data/result').read_text(encoding='utf-8')\n",
            "aliased_control": "import os as x\ndef process_start_ticks(pid): return '/proc'\nx.system('true')\n",
            "dynamic_control": "import os\ndef process_start_ticks(pid): return '/proc'\ngetattr(os, 'kill')(1, 9)\n",
        }
        for name, source in unsafe_cases.items():
            with self.subTest(name=name):
                self.assertTrue(AUDIT.audit_source(source))


if __name__ == "__main__":
    unittest.main()
