#!/usr/bin/env python3
"""Guarded future runner for the frozen A1X_ACCESS diagnostic.

This file deliberately does not discover data or launch work during R2 setup.
An R1 authorization artifact must be supplied before a future cloud execution
implementation may be enabled.
"""

import argparse
import hashlib
import json
from pathlib import Path


ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
SOURCE_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
AUTHORIZED_STAGES = {"s0", "formal"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(AUTHORIZED_STAGES), required=True)
    parser.add_argument("--authorization-json", required=True)
    parser.add_argument("--emit-plan", action="store_true")
    args = parser.parse_args()
    authorization = json.loads(Path(args.authorization_json).read_text(encoding="utf-8"))
    if authorization.get("route_id") != ROUTE_ID:
        raise SystemExit("A1X runner refused: route_id mismatch")
    if authorization.get("source_commit") != SOURCE_COMMIT:
        raise SystemExit("A1X runner refused: source_commit mismatch")
    if authorization.get("authorizes") != "A1X_FORMAL_CONFIRMATION_ONLY":
        raise SystemExit("A1X runner refused: missing committed R1 authorization")
    if not args.emit_plan:
        raise SystemExit("A1X runner is planning-only until a separately authorized execution implementation")
    print(json.dumps({"route_id": ROUTE_ID, "stage": args.stage, "authorization_sha256": sha256(args.authorization_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
