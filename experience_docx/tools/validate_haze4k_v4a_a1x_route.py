#!/usr/bin/env python3
"""Source-only proof suite for the frozen A1X S0 runner contract."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
SOURCE_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
FROZEN = {"experience_docx/experiment_cards/2026-07-15-haze4k-v5-v4a-a1x-exact-half-deployable-accessibility.md": "c723e59a3cb06de63b1ae4a72eabcd64ca0a3d08ba1346f3a53e0a52cce452da", "experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715/v4a_a1x_r3_design_handoff.json": "630b81edb07fd6a4f4243c529998e5c97fbeb834926d3fd9cde5ce26bca6340a", "Dehazing/ITS/models/A1XAccess.py": "625b60368dde9316df3c506cd339b94650e05243edc7cb8090f0b5b555b6df33", "experience_docx/route_operations.json": "a5cbe7d8e0559ee75c1b53b152561f10f754befb4e289963da3c465520e3f22f"}
EXPECTED = ("1594_0.71_0.5.png", "1595_0.99_1.84.png", "1597_0.69_1.45.png", "1598_0.67_1.4.png", "159_0.6_1.46.png", "1600_0.78_1.77.png", "1603_0.54_0.74.png", "1607_0.91_0.88.png", "160_0.63_1.04.png", "1613_0.56_1.31.png", "1614_0.81_0.78.png", "1615_0.91_1.25.png", "1616_0.76_0.88.png", "1617_0.56_1.97.png", "1619_0.94_1.08.png", "1622_0.98_1.75.png", "1623_0.78_1.81.png", "1627_0.94_0.52.png", "1628_0.8_1.49.png", "1633_0.73_1.49.png", "1634_0.75_1.81.png", "1639_0.69_1.12.png", "1640_0.53_0.59.png", "1646_0.55_1.55.png", "1649_0.8_0.86.png", "1650_0.76_1.07.png", "1652_0.62_1.35.png", "1653_0.9_1.01.png", "1654_0.66_1.9.png", "1656_0.64_1.45.png", "1658_0.96_1.72.png", "1660_0.83_0.67.png")
MARKERS = ("STATIC_EXACT_FIRST32_LITERAL_MATCH_PASS", "STATIC_EXACT_FIRST32_COUNT_UNIQUE_ORDER_PASS", "STATIC_NO_FILESYSTEM_ENUMERATION_PASS", "STATIC_IMAGE_OPEN_DERIVES_ONLY_FROM_FIRST32_PASS", "STATIC_NO_A1X_CONFIRMATION_CANARY_LOCKED_OPEN_PASS", "STATIC_FORMAL_MODE_REJECTED_PASS", "STATIC_EXACT_AUTHORIZATION_GUARD_PRECEDES_OPENS_PASS", "STATIC_MODEL_FREEZE_AND_TRAINABLE_PREFIX_PASS", "STATIC_FIVE_INPUT_PROVENANCE_PASS", "STATIC_TRUE_SHUFFLED_TWO_UPDATE_CONTRACT_PASS", "STATIC_INTEGRATED_S0_CHECK_SURFACES_PASS", "STATIC_DISTINCT_LIFECYCLE_PATHS_AND_WRITERS_PASS", "STATIC_TYPED_CLOSEOUT_FIELDS_AND_TERMINAL_TUPLES_PASS", "STATIC_RUNNER_EXPLICIT_PYTHON_ROOTS_LOG_AND_EXIT_PASS", "STATIC_FROZEN_FILES_HASHES_PASS")

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def require(value, message):
    if not value: raise ValueError(message)
def literal_tuple(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "S0_FIRST32_NAMES" for t in node.targets): return ast.literal_eval(node.value)
    raise ValueError("S0_FIRST32_NAMES must be a literal assignment")
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", required=True); parser.add_argument("--output-json", required=True); args = parser.parse_args()
    repo = Path(args.repo).resolve(); entry = repo / "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py"; runner = repo / "experience_docx/tools/run_chd_rm_v4a_a1x_exact_half_accessibility.sh"
    entry_text = entry.read_text(encoding="utf-8"); runner_text = runner.read_text(encoding="utf-8"); tree = ast.parse(entry_text)
    names = tuple(literal_tuple(tree)); require(names == EXPECTED, "literal first32 mismatch"); require(len(names) == 32 and len(set(names)) == 32, "first32 count/unique failure")
    forbidden = {"glob", "iglob", "rglob", "iterdir", "listdir", "scandir", "walk"}; calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]; require(not any(isinstance(node.func, ast.Attribute) and node.func.attr in forbidden for node in calls), "filesystem enumeration forbidden")
    require("def image_path(root, name)" in entry_text and "if name not in S0_FIRST32_NAMES" in entry_text and "Image.open(path)" in entry_text, "image open must derive from literal names")
    require(not any(word in entry_text for word in ("confirmation_root", "canary_root", "locked_test_root", "a1x_root")), "out-of-scope data path present")
    require('if args.stage != "s0"' in entry_text and "formal mode is not enabled" in entry_text, "formal rejection missing")
    require(entry_text.index("authorization(args.authorization_json)") < entry_text.index("import torch"), "authorization must precede model/data opens")
    require("name.startswith(\"A1X_ACCESS_\")" in entry_text and "parameter.requires_grad" in entry_text, "freeze/trainable prefix surface missing")
    require("torch.cat(inputs, dim=1)" in entry_text and "deployable.shape[1] != 15" in entry_text, "five-input provenance missing")
    require("A1X_ACCESS_TRUE" in entry_text and "A1X_ACCESS_SHUFFLED" in entry_text and "for update in range(2)" in entry_text and "any(left == right" in entry_text, "true/shuffled two-update contract missing")
    require(all(item in entry_text for item in ("INTEGRATED_S0_CHECKS", "added parameter cost limit", "nonfinite", "first gradient")), "integrated check surfaces missing")
    require(all(item in entry_text for item in ("status_json", "heartbeat_json", "learned_state_manifest_json", "closeout_json", "write_json")), "distinct lifecycle writers missing")
    operations = json.loads((repo / "experience_docx/route_operations.json").read_text(encoding="utf-8")); item = operations["operations"][0]; tuples = item["allowed_terminal_tuples"]
    require(item["route_id"] == ROUTE_ID and item["source_commit"] == SOURCE_COMMIT and item["formal_mode_enabled"] is False and len(tuples) == 3, "terminal tuple source mismatch")
    required_fields = ("schema_version", "route_id", "route_commit", "source_commit", "run_id", "stage", "decision", "authorizes", "failure_class", "failure_phase", "locked_test_touched")
    require(all(field in entry_text for field in required_fields), "typed closeout fields missing")
    require(all(token in runner_text for token in ("EXPLICIT_CLOUD_PYTHON", "RUN_ROOT", "RUNTIME_LOG_PATH", "PIPESTATUS[0]", "A1X_S0_OK", "A1X_S0_FAILED")), "runner lifecycle/exit surface missing")
    frozen_hashes = {rel: digest(repo / rel) for rel in FROZEN}; require(frozen_hashes == FROZEN, "frozen file hash mismatch")
    payload = {"schema_version": 1, "route_id": ROUTE_ID, "source_commit": SOURCE_COMMIT, "entrypoint_sha256": digest(entry), "runner_sha256": digest(runner), "frozen_hashes": frozen_hashes, "proof_markers": list(MARKERS), "runtime_started": False, "a1x_data_touched": False, "confirmation_touched": False, "canary_touched": False, "locked_test_touched": False, "formal_authorized": False, "validator": "A1X_S0_RUNNER_COMPLETION_STATIC_VALIDATOR_OK"}
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("A1X_S0_RUNNER_COMPLETION_STATIC_VALIDATOR_OK")
if __name__ == "__main__": main()
