#!/usr/bin/env python3
"""One-time bridge for a v4.1 plan after the v4.2 recovery code is validated."""

import argparse
import json
import re

import convir_ops_mcp as ops


TOKEN = re.compile(r"^[0-9a-f]{64}$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-token", required=True)
    args = parser.parse_args()
    if not TOKEN.fullmatch(args.plan_token):
        raise SystemExit("invalid plan token")
    if ops.SERVER_VERSION != "4.2.0" or ops.SCHEMA_VERSION != 4 or len(ops.TOOLS) != 6:
        raise SystemExit("candidate MCP identity mismatch")
    result = ops.tool_start({"plan_token": args.plan_token})
    value = result.get("structuredContent", {})
    print(json.dumps(value, sort_keys=True))
    if not value.get("ok") or value.get("operation_state") != "LAUNCH_RECOVERED":
        raise SystemExit(1)
    print(f"CONVIR_OPS_START_RECOVERY_OK receipt={value['receipt']}")


if __name__ == "__main__":
    main()
