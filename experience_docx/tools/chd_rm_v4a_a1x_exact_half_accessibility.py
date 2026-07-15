#!/usr/bin/env python3
"""Guarded S0-only entrypoint for the frozen A1X accessibility diagnostic.

This module contains no data discovery or training implementation.  A future
tracked runner may call it only after the exact initial authorization guard.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
SOURCE_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
REQUIRED = {"state": "PLANNED", "decision": "V4A_A1X_S0_AUTHORIZED_INITIAL_ONLY", "authorizes": "A1X_S0_ONLY"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authorization(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("route_id") != ROUTE_ID or payload.get("source_commit") != SOURCE_COMMIT:
        raise SystemExit("A1X runner refused: route identity mismatch")
    for key, value in REQUIRED.items():
        if payload.get(key) != value:
            raise SystemExit(f"A1X runner refused: authorization {key} mismatch")
    return payload


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--authorization-json", required=True)
    parser.add_argument("--status-json")
    parser.add_argument("--heartbeat-json")
    parser.add_argument("--learned-state-manifest-json")
    parser.add_argument("--closeout-json")
    args = parser.parse_args()
    if args.stage != "s0":
        raise SystemExit("A1X runner refused: formal mode is not enabled")
    authorization(args.authorization_json)
    payload = {"route_id": ROUTE_ID, "stage": "s0", "authorization_sha256": sha256(args.authorization_json), "runtime_started": False}
    for destination in (args.status_json, args.heartbeat_json, args.learned_state_manifest_json, args.closeout_json):
        if destination:
            write_json(destination, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
