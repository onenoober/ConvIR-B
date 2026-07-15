#!/usr/bin/env python3
"""Static validator for the frozen A1X S0 operational contract."""
import argparse, ast, hashlib, json
from pathlib import Path

ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
SOURCE_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
CARD_SHA = "c723e59a3cb06de63b1ae4a72eabcd64ca0a3d08ba1346f3a53e0a52cce452da"
TUPLE = {"state": "PLANNED", "decision": "V4A_A1X_S0_AUTHORIZED_INITIAL_ONLY", "authorizes": "A1X_S0_ONLY"}

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok, text):
    if not ok: raise ValueError(text)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", required=True); parser.add_argument("--output-json", required=True); args = parser.parse_args()
    repo = Path(args.repo).resolve(); card = repo / "experience_docx/experiment_cards/2026-07-15-haze4k-v5-v4a-a1x-exact-half-deployable-accessibility.md"
    handoff = repo / "experience_docx/experiment_logs" / ROUTE_ID / "v4a_a1x_r3_design_handoff.json"; head = repo / "Dehazing/ITS/models/A1XAccess.py"
    entry = repo / "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py"; runner = repo / "experience_docx/tools/run_chd_rm_v4a_a1x_exact_half_accessibility.sh"; operations = repo / "experience_docx/route_operations.json"
    for path in (card, handoff, head, entry, runner, operations): require(path.is_file(), f"missing required file: {path}")
    for path in (head, entry): ast.parse(path.read_text(encoding="utf-8"))
    require(digest(card) == CARD_SHA, "frozen route-card hash mismatch")
    typed = json.loads(handoff.read_text(encoding="utf-8")); require(typed["route_id"] == ROUTE_ID and typed["source_commit"] == SOURCE_COMMIT, "handoff identity mismatch")
    manifest = json.loads(operations.read_text(encoding="utf-8")); items = manifest.get("operations", [])
    require(len(items) == 1 and items[0].get("operation_id") == "A1X_S0", "exactly one A1X_S0 operation required")
    item = items[0]; require(item.get("route_id") == ROUTE_ID and item.get("source_commit") == SOURCE_COMMIT, "manifest identity mismatch")
    require(all(item.get(k) == v for k, v in TUPLE.items()), "manifest authorization tuple mismatch")
    require(item.get("mode") == "s0" and item.get("formal_mode_enabled") is False, "formal mode must be disabled")
    require(item.get("authorization_relpath", "").endswith("/initial_authorization.json"), "authorization path mismatch")
    require(len(item.get("allowed_terminal_tuples", [])) == 3, "terminal tuple contract mismatch")
    runner_text = runner.read_text(encoding="utf-8"); entry_text = entry.read_text(encoding="utf-8")
    for token in (ROUTE_ID, "V4A_A1X_S0_AUTHORIZED_INITIAL_ONLY", "A1X_S0_ONLY", "formal mode", "authorization-json"):
        require(token in runner_text or token in entry_text, f"guard token missing: {token}")
    payload = {"route_id": ROUTE_ID, "route_card_sha256": digest(card), "handoff_sha256": digest(handoff), "head_sha256": digest(head), "entrypoint_sha256": digest(entry), "runner_sha256": digest(runner), "route_operations_sha256": digest(operations), "runtime_started": False, "a1x_data_touched": False, "canary_touched": False, "locked_test_touched": False, "validator": "PASS"}
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print("A1X_ROUTE_STATIC_VALIDATOR_OK")
if __name__ == "__main__": main()
