#!/usr/bin/env python3
"""Synthetic cloud validation for generic monitoring safety and cost."""

import argparse
import importlib.util
import json
import os
import tempfile
import time
from pathlib import Path


def load(path):
    spec = importlib.util.spec_from_file_location(f"validation_{Path(path).stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    telemetry_path = Path(args.telemetry).resolve()
    telemetry = load(telemetry_path)
    audit = load(telemetry_path.with_name("audit_run_telemetry.py"))
    forbidden = audit.audit_path(telemetry_path)
    with tempfile.TemporaryDirectory() as root:
        heartbeat = Path(root) / "heartbeat.json"
        started_wall = time.monotonic()
        started_cpu = time.process_time()
        for sequence in range(1, 101):
            rc = telemetry.main([
                "pulse", "--route-id", "synthetic-monitor", "--run-id", "validation-r1",
                "--phase", "pulse_cost", "--heartbeat", str(heartbeat),
                "--sequence", str(sequence), "--completed", str(sequence), "--total", "100",
            ])
            if rc != 0:
                raise RuntimeError(f"pulse failed: {rc}")
        wall = time.monotonic() - started_wall
        cpu = time.process_time() - started_cpu
        value = json.loads(heartbeat.read_text(encoding="utf-8"))
        files = sorted(path.name for path in Path(root).iterdir())
    summary = {
        "schema_version": 1,
        "validation": "generic_run_monitoring",
        "host": __import__("socket").gethostname(),
        "python": __import__("sys").executable,
        "pulse_count": 100,
        "pulse_wall_seconds": wall,
        "pulse_cpu_seconds": cpu,
        "projected_cpu_seconds_per_hour_at_60s_interval": cpu / 100.0 * 60.0,
        "heartbeat_bytes": len(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"),
        "final_sequence": value["sequence"],
        "files_created": files,
        "source_audit": "python_ast_v1",
        "forbidden_control_constructs": forbidden,
        "reads_scientific_outputs": False,
        "sends_process_signals": False,
        "queries_gpu": False,
        "pass": bool(
            value["sequence"] == 100
            and files == ["heartbeat.json"]
            and not forbidden
            and cpu < 5.0
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print("GENERIC_RUN_MONITORING_SYNTHETIC_OK" if summary["pass"] else "GENERIC_RUN_MONITORING_SYNTHETIC_FAIL")
    raise SystemExit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
