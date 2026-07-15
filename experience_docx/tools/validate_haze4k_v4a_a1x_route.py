#!/usr/bin/env python3
"""Source-only semantic proof suite for the frozen A1X S0 runner contract."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
BASE_COMMIT = "597cbb1fa82c9cf31c93ed8328f9d80e47f2198f"
FROZEN = {"experience_docx/experiment_cards/2026-07-15-haze4k-v5-v4a-a1x-exact-half-deployable-accessibility.md": "c723e59a3cb06de63b1ae4a72eabcd64ca0a3d08ba1346f3a53e0a52cce452da", "experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715/v4a_a1x_r3_design_handoff.json": "630b81edb07fd6a4f4243c529998e5c97fbeb834926d3fd9cde5ce26bca6340a", "Dehazing/ITS/models/A1XAccess.py": "625b60368dde9316df3c506cd339b94650e05243edc7cb8090f0b5b555b6df33", "experience_docx/route_operations.json": "a5cbe7d8e0559ee75c1b53b152561f10f754befb4e289963da3c465520e3f22f"}
EXPECTED = ("1594_0.71_0.5.png", "1595_0.99_1.84.png", "1597_0.69_1.45.png", "1598_0.67_1.4.png", "159_0.6_1.46.png", "1600_0.78_1.77.png", "1603_0.54_0.74.png", "1607_0.91_0.88.png", "160_0.63_1.04.png", "1613_0.56_1.31.png", "1614_0.81_0.78.png", "1615_0.91_1.25.png", "1616_0.76_0.88.png", "1617_0.56_1.97.png", "1619_0.94_1.08.png", "1622_0.98_1.75.png", "1623_0.78_1.81.png", "1627_0.94_0.52.png", "1628_0.8_1.49.png", "1633_0.73_1.49.png", "1634_0.75_1.81.png", "1639_0.69_1.12.png", "1640_0.53_0.59.png", "1646_0.55_1.55.png", "1649_0.8_0.86.png", "1650_0.78_1.77.png", "1652_0.62_1.35.png", "1653_0.9_1.01.png", "1654_0.66_1.9.png", "1656_0.64_1.45.png", "1658_0.96_1.72.png", "1660_0.83_0.67.png")
CODES = ("A1X_HEAD_PARAMETERS_ARE_ALL_FROZEN", "OFFICIAL_MODEL_LOAD_AND_FREEZE_AUDIT_MISSING", "SHUFFLED_TARGET_CONTROL_IS_REVERSED", "FROZEN_LOSS_AND_OPERATOR_CONTRACT_NOT_IMPLEMENTED", "INTEGRATED_S0_CHECKS_ARE_DECLARED_BUT_NOT_MEASURED", "DURABLE_LIFECYCLE_AND_CLOSEOUT_CONTRACT_INCOMPLETE", "REMOTE_REPO_DOES_NOT_MATCH_FROZEN_FRESH_ROUTE_WORKSPACE", "FINAL_MARKER_CONTRACT_DUPLICATED", "STATIC_VALIDATOR_PROVES_TOKENS_NOT_REQUIRED_SEMANTICS")
MARKERS = ("SEMANTIC_A1X_HEAD_TRAINABLE_SCOPE_PASS", "SEMANTIC_OFFICIAL_MODEL_LOAD_FREEZE_EVAL_PASS", "SEMANTIC_TARGET_ONLY_OPERATOR_SHAPE_SHUFFLE_PASS", "SEMANTIC_ACTIVE_SUPPORT_OPERATOR_NATIVE_BATCH_LOSS_PASS", "SEMANTIC_MEASURED_STRUCTURAL_COST_CHECKS_PASS", "SEMANTIC_COMPLETE_LIFECYCLE_CLOSEOUT_PASS", "SEMANTIC_FROZEN_REMOTE_REPO_PASS", "SEMANTIC_EXACTLY_ONE_TERMINAL_MARKER_PASS", "SEMANTIC_VALIDATOR_NEGATIVE_FIXTURE_PASS", "STATIC_FROZEN_SCIENTIFIC_HASHES_PASS")


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value, message):
    if not value: raise ValueError(message)
def functions(tree): return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
def calls(node): return [item for item in ast.walk(node) if isinstance(item, ast.Call)]
def name_of(node):
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Attribute): return name_of(node.value) + "." + node.attr
    return ""
def call_named(node, suffix): return [item for item in calls(node) if name_of(item.func).endswith(suffix)]
def true_call(node, suffix): return any(item.args and isinstance(item.args[0], ast.Constant) and item.args[0].value is True for item in call_named(node, suffix))
def source_of(node): return ast.unparse(node)
def subscript_constant(node): return node.slice.value if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) else None
def official_freeze_ast_proof(official):
    for loop in [node for node in ast.walk(official) if isinstance(node, ast.For) and isinstance(node.target, ast.Name)]:
        if not isinstance(loop.iter, ast.Call) or name_of(loop.iter.func) != "official_model.parameters": continue
        if any(name_of(call.func).endswith("requires_grad_") and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == loop.target.id and call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value is False for call in calls(loop)):
            return True
    return False
def target_shuffle_ast_proof(shuffle, forward):
    paired_assignment = any(isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript) and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == "paired" and isinstance(node.targets[0].slice, ast.Tuple) and len(node.targets[0].slice.elts) == 2 and isinstance(node.targets[0].slice.elts[0], ast.Subscript) and subscript_constant(node.targets[0].slice.elts[0]) == "operator" and isinstance(node.targets[0].slice.elts[1], ast.Subscript) and subscript_constant(node.targets[0].slice.elts[1]) == "name" and "(index + 1) % len(ordered)" in source_of(node.value) for node in ast.walk(shuffle))
    target_lookup = any(isinstance(node, ast.Subscript) and subscript_constant(node) == "TARGET_DELTA_U" and isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name) and node.value.value.id == "target_map" and isinstance(node.value.slice, ast.Tuple) and len(node.value.slice.elts) == 2 and isinstance(node.value.slice.elts[0], ast.Subscript) and subscript_constant(node.value.slice.elts[0]) == "operator" for node in ast.walk(forward))
    deployable_source = any(isinstance(node, ast.Subscript) and isinstance(node.value, ast.Subscript) and subscript_constant(node.value) == "source" for node in ast.walk(forward))
    current_delta = any(isinstance(node, ast.Constant) and node.value == "CURRENT_DELTA_U" for node in ast.walk(forward))
    return paired_assignment and target_lookup and deployable_source and current_delta
def emit_record(record): print(json.dumps(record, sort_keys=True))
def replace_once(text, old, new):
    require(text.count(old) == 1, "fixture mutation anchor is not unique")
    return text.replace(old, new, 1)
def neg09_assignment_span(validator_text):
    tree = ast.parse(validator_text); fixtures = functions(tree).get("isolated_fixtures")
    candidates = [node for node in ast.walk(fixtures) if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "detected" and isinstance(node.value, ast.Call) and name_of(node.value.func) == "semantic_findings" and [name_of(arg) for arg in node.value.args] == ["mutated_entry", "mutated_runner", "mutated_validator"]] if fixtures else []
    require(len(candidates) == 1, "NEG_09 structural fixture mutation target is not unique")
    lines = validator_text.splitlines(keepends=True); starts = [0]
    for line in lines: starts.append(starts[-1] + len(line))
    node = candidates[0]
    return starts[node.lineno - 1] + node.col_offset, starts[node.end_lineno - 1] + node.end_col_offset
def mutate_neg09_fixture_analysis(validator_text):
    start, end = neg09_assignment_span(validator_text)
    return validator_text[:start] + "detected = []" + validator_text[end:]
def isolated_fixture_diagnostics_span(validator_text):
    tree = ast.parse(validator_text)
    function_candidates = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "isolated_fixtures"]
    require(len(function_candidates) == 1, "isolated-fixtures function target is not unique")
    fixture_function = function_candidates[0]
    loop_candidates = [node for node in ast.walk(fixture_function) if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple) and tuple(item.id for item in node.target.elts if isinstance(item, ast.Name)) == ("fixture_id", "expected", "path", "mutate") and isinstance(node.iter, ast.Name) and node.iter.id == "fixtures"]
    require(len(loop_candidates) == 1, "isolated-fixtures loop target is not unique")
    loop = loop_candidates[0]
    diagnostic_candidates = [(index, node) for index, node in enumerate(loop.body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and name_of(node.value.func) == "emit_record" and len(node.value.args) == 1 and isinstance(node.value.args[0], ast.Name) and node.value.args[0].id == "record"]
    require(len(diagnostic_candidates) == 1, "isolated-fixtures diagnostic target is not unique")
    diagnostic_index, diagnostic = diagnostic_candidates[0]
    record_assignments = [node for node in loop.body[:diagnostic_index] if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "record" and isinstance(node.value, ast.Dict) and any(isinstance(key, ast.Constant) and key.value == "expected_finding_code" and isinstance(value, ast.Name) and value.id == "expected" for key, value in zip(node.value.keys, node.value.values))]
    require(len(record_assignments) == 1, "isolated-fixtures record payload target is not unique")
    require(diagnostic_index + 1 < len(loop.body), "isolated-fixtures records append is missing")
    append_statement = loop.body[diagnostic_index + 1]
    require(isinstance(append_statement, ast.Expr) and isinstance(append_statement.value, ast.Call) and name_of(append_statement.value.func) == "records.append" and len(append_statement.value.args) == 1 and isinstance(append_statement.value.args[0], ast.Name) and append_statement.value.args[0].id == "record", "isolated-fixtures diagnostic does not immediately precede records append")
    loop_index = fixture_function.body.index(loop)
    generic_gates = [node for node in fixture_function.body[loop_index + 1:] if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and name_of(node.value.func) == "require" and any(isinstance(item, ast.Name) and item.id == "records" for item in ast.walk(node.value))]
    require(len(generic_gates) == 1, "isolated-fixtures generic records gate is not unique")
    lines = validator_text.splitlines(keepends=True); starts = [0]
    for line in lines: starts.append(starts[-1] + len(line))
    return starts[diagnostic.lineno - 1] + diagnostic.col_offset, starts[diagnostic.end_lineno - 1] + diagnostic.end_col_offset
def mutate_omitted_pre_failure_diagnostics(validator_text):
    start, end = isolated_fixture_diagnostics_span(validator_text)
    return validator_text[:start] + "pass" + validator_text[end:]
def diagnostics_before_gate_ast_proof(validator_text):
    try: isolated_fixture_diagnostics_span(validator_text)
    except ValueError: return False
    return True
def freeze_polarity_comparator_span(validator_text):
    tree = ast.parse(validator_text); proof = functions(tree).get("official_freeze_ast_proof")
    candidates = []
    for node in ast.walk(proof) if proof else []:
        left = node.left if isinstance(node, ast.Compare) else None
        argument_value = isinstance(left, ast.Attribute) and left.attr == "value" and isinstance(left.value, ast.Subscript) and subscript_constant(left.value) == 0 and isinstance(left.value.value, ast.Attribute) and left.value.value.attr == "args" and isinstance(left.value.value.value, ast.Name)
        is_false_polarity = isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Is) and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is False
        requires_grad_proof = any(isinstance(parent, ast.BoolOp) and any(child is node for child in ast.walk(parent)) and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "endswith" and child.args and isinstance(child.args[0], ast.Constant) and child.args[0].value == "requires_grad_" for child in ast.walk(parent)) for parent in ast.walk(proof))
        if argument_value and is_false_polarity and requires_grad_proof: candidates.append(node.comparators[0])
    require(len(candidates) == 1, "freeze-polarity structural mutation target is not unique")
    lines = validator_text.splitlines(keepends=True); starts = [0]
    for line in lines: starts.append(starts[-1] + len(line))
    node = candidates[0]
    return starts[node.lineno - 1] + node.col_offset, starts[node.end_lineno - 1] + node.end_col_offset
def mutate_freeze_polarity_self_proof(validator_text):
    start, end = freeze_polarity_comparator_span(validator_text)
    require(validator_text[start:end] == "False", "freeze-polarity structural target is not the False comparator")
    return validator_text[:start] + "True" + validator_text[end:]
def target_shuffle_return_span(validator_text):
    tree = ast.parse(validator_text); proof = functions(tree).get("target_shuffle_ast_proof")
    expected_terms = ("paired_assignment", "target_lookup", "deployable_source", "current_delta")
    candidates = [node for node in ast.walk(proof) if isinstance(node, ast.Return) and isinstance(node.value, ast.BoolOp) and isinstance(node.value.op, ast.And) and len(node.value.values) == len(expected_terms) and tuple(item.id for item in node.value.values if isinstance(item, ast.Name)) == expected_terms] if proof else []
    require(len(candidates) == 1, "target-shuffle structural mutation target is not unique")
    lines = validator_text.splitlines(keepends=True); starts = [0]
    for line in lines: starts.append(starts[-1] + len(line))
    node = candidates[0]
    return starts[node.lineno - 1] + node.col_offset, starts[node.end_lineno - 1] + node.end_col_offset
def mutate_quote_spelling_only_target_acceptance(validator_text):
    start, end = target_shuffle_return_span(validator_text)
    return validator_text[:start] + "return 'target_map[(example[\\\"operator\\\"], name)][\\\"TARGET_DELTA_U\\\"]' in source_of(forward)" + validator_text[end:]
def official_freeze_validator_ast_proof(validator_text):
    tree = ast.parse(validator_text); proof = functions(tree).get("official_freeze_ast_proof")
    if not proof: return False
    for node in ast.walk(proof):
        left = node.left if isinstance(node, ast.Compare) else None
        argument_value = isinstance(left, ast.Attribute) and left.attr == "value" and isinstance(left.value, ast.Subscript) and subscript_constant(left.value) == 0 and isinstance(left.value.value, ast.Attribute) and left.value.value.attr == "args" and isinstance(left.value.value.value, ast.Name)
        if argument_value and len(node.ops) == 1 and isinstance(node.ops[0], ast.Is) and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is False:
            return True
    return False
def target_shuffle_validator_ast_proof(validator_text):
    try: target_shuffle_return_span(validator_text)
    except ValueError: return False
    return True


def semantic_findings(entry_text, runner_text, validator_text):
    tree = ast.parse(entry_text); fn = functions(tree); findings = []
    head = fn.get("trainable_head")
    if not head or not true_call(head, "requires_grad_") or not call_named(head, "parameters") or not call_named(head, "AdamW"):
        findings.append(CODES[0])
    official = fn.get("load_official_model")
    if not official or not call_named(official, "torch.load") or not call_named(official, "load_state_dict") or not call_named(official, "eval") or not official_freeze_ast_proof(official) or not official_freeze_validator_ast_proof(validator_text) or "strict=True" not in source_of(official):
        findings.append(CODES[1])
    shuffle, forward = fn.get("shuffled_target_map"), fn.get("forward_batch")
    if not shuffle or not forward or not target_shuffle_ast_proof(shuffle, forward) or not target_shuffle_validator_ast_proof(validator_text):
        findings.append(CODES[2])
    batches, loss, cell = fn.get("native_shape_batches"), fn.get("normalized_active_support_endpoint_mse"), fn.get("run_cell")
    if not batches or not loss or not cell or "index + 4" not in source_of(batches) or "error.sum() / denominator" not in source_of(loss) or "range(2)" not in source_of(cell):
        findings.append(CODES[3])
    measure = fn.get("measure_checks")
    if not measure or not fn.get("profile_measurement") or len(call_named(measure, "profile_measurement")) < 2 or "<= 300000" not in source_of(measure) or "<= 10.0" not in source_of(measure) or "<= 15.0" not in source_of(measure):
        findings.append(CODES[4])
    main = fn.get("main")
    lifecycle = ("append_status", "Heartbeat", "learned_state_manifest_json", "runner_sha256", "route_card_sha256", "phase_timings", "runtime_started")
    if not main or not all(item in source_of(main) for item in lifecycle): findings.append(CODES[5])
    expected_remote = 'REMOTE_REPO="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-a1x-exact-half-accessibility-20260715"'
    if expected_remote not in runner_text or "ENTRYPOINT=\"${REMOTE_REPO}/experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py\"" not in runner_text:
        findings.append(CODES[6])
    if "A1X_S0_OK" in entry_text or "A1X_S0_FAILED" in entry_text or runner_text.count("A1X_S0_OK") != 1 or runner_text.count("A1X_S0_FAILED") != 1 or runner_text.find('[ ! -f "${CLOSEOUT_PATH}" ]') > runner_text.find("A1X_S0_OK") or "PIPESTATUS[0]" not in runner_text:
        findings.append(CODES[7])
    validator_tree = ast.parse(validator_text); validator_fn = functions(validator_tree); fixtures = validator_fn.get("isolated_fixtures")
    if not fixtures or not call_named(fixtures, "semantic_findings") or not call_named(fixtures, "replace_once") or "ast.parse" not in validator_text or not diagnostics_before_gate_ast_proof(validator_text) or not validator_fn.get("validator_self_proof"):
        findings.append(CODES[8])
    return sorted(set(findings))


def isolated_fixtures(entry_text, runner_text, validator_text):
    fixtures = (
        ("NEG_01_HEAD_TRAINABLE_SCOPE", CODES[0], "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py", lambda e, r, v: (replace_once(e, "parameter.requires_grad_(True)", "parameter.requires_grad_(False)"), r, v)),
        ("NEG_02_OFFICIAL_MODEL_LOAD_FREEZE_AUDIT", CODES[1], "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py", lambda e, r, v: (replace_once(e, "official_model.eval()", "official_model.train()"), r, v)),
        ("NEG_03_TARGET_ONLY_SHUFFLE", CODES[2], "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py", lambda e, r, v: (replace_once(e, "(index + 1) % len(ordered)", "index % len(ordered)"), r, v)),
        ("NEG_04_OPERATOR_NATIVE_ACTIVE_SUPPORT_LOSS", CODES[3], "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py", lambda e, r, v: (replace_once(e, "error.sum() / denominator", "error.mean()"), r, v)),
        ("NEG_05_MEASURED_STRUCTURAL_COST_CHECKS", CODES[4], "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py", lambda e, r, v: (replace_once(e, "baseline = profile_measurement", "baseline = {} # declared PASS control"), r, v)),
        ("NEG_06_DURABLE_LIFECYCLE_CLOSEOUT", CODES[5], "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py", lambda e, r, v: (replace_once(e, "Heartbeat(args.heartbeat_json, args)", "None"), r, v)),
        ("NEG_07_FROZEN_REMOTE_REPO", CODES[6], "experience_docx/tools/run_chd_rm_v4a_a1x_exact_half_accessibility.sh", lambda e, r, v: (e, replace_once(r, "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-a1x-exact-half-accessibility-20260715", "/sda/home/wangyuxin/ConvIR-B"), v)),
        ("NEG_08_SINGLE_TERMINAL_MARKER_OWNER", CODES[7], "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py", lambda e, r, v: (replace_once(e, "write_json(args.closeout_json, closeout); append_status", "print(\"A1X_S0_OK\"); write_json(args.closeout_json, closeout); append_status"), r, v)),
        ("NEG_09_VALIDATOR_SEMANTIC_PROOF", CODES[8], "experience_docx/tools/validate_haze4k_v4a_a1x_route.py", lambda e, r, v: (e, r, mutate_neg09_fixture_analysis(v))),
    )
    records = []
    for fixture_id, expected, path, mutate in fixtures:
        mutated_entry, mutated_runner, mutated_validator = mutate(entry_text, runner_text, validator_text)
        detected = semantic_findings(mutated_entry, mutated_runner, mutated_validator)
        before = {"experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py": entry_text, "experience_docx/tools/run_chd_rm_v4a_a1x_exact_half_accessibility.sh": runner_text, "experience_docx/tools/validate_haze4k_v4a_a1x_route.py": validator_text}[path]
        after = {"experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py": mutated_entry, "experience_docx/tools/run_chd_rm_v4a_a1x_exact_half_accessibility.sh": mutated_runner, "experience_docx/tools/validate_haze4k_v4a_a1x_route.py": mutated_validator}[path]
        record = {"record_type": "ISOLATED_FIXTURE_FINDINGS", "fixture_id": fixture_id, "expected_finding_code": expected, "mutated_source_path": path, "mutation_description": "one in-memory structural mutation", "source_before_sha256": hashlib.sha256(before.encode()).hexdigest(), "source_after_sha256": hashlib.sha256(after.encode()).hexdigest(), "detected_finding_codes": detected, "expected_code_detected": expected in detected, "target_code_executed_false": True, "data_or_checkpoint_opened_false": True}
        emit_record(record); records.append(record)
    require(len(records) == len(CODES) and all(item["expected_code_detected"] for item in records), "isolated fixture did not detect its expected semantic defect")
    return records


def base_source(repo, relpath):
    return subprocess.run(["git", "-C", str(repo), "show", BASE_COMMIT + ":" + relpath], check=True, text=True, capture_output=True).stdout


def validator_self_proof(entry_text, runner_text, validator_text):
    fixtures = (
        ("INVERTED_OFFICIAL_FREEZE_POLARITY", CODES[1]),
        ("QUOTE_SPELLING_ONLY_TARGET_ACCEPTANCE", CODES[2]),
        ("OMITTED_PRE_FAILURE_DIAGNOSTICS", CODES[8]),
        ("BYPASSED_ISOLATED_FIXTURE_ANALYSIS", CODES[8]),
    )
    records = []
    for fixture_id, expected in fixtures:
        mutated_validator = mutate_neg09_fixture_analysis(validator_text) if fixture_id == "BYPASSED_ISOLATED_FIXTURE_ANALYSIS" else mutate_freeze_polarity_self_proof(validator_text) if fixture_id == "INVERTED_OFFICIAL_FREEZE_POLARITY" else mutate_quote_spelling_only_target_acceptance(validator_text) if fixture_id == "QUOTE_SPELLING_ONLY_TARGET_ACCEPTANCE" else mutate_omitted_pre_failure_diagnostics(validator_text)
        detected = semantic_findings(entry_text, runner_text, mutated_validator)
        record = {"record_type": "VALIDATOR_SELF_PROOF_FINDINGS", "fixture_id": fixture_id, "expected_finding_code": expected, "detected_finding_codes": detected, "expected_code_detected": expected in detected, "validator_executed_false": True}
        emit_record(record); records.append(record)
    require(all(record["expected_code_detected"] for record in records), "validator self-proof did not reject a direct validator defect")
    return records


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", required=True); parser.add_argument("--output-json", required=True); args = parser.parse_args()
    repo = Path(args.repo).resolve(); entry = repo / "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py"; runner = repo / "experience_docx/tools/run_chd_rm_v4a_a1x_exact_half_accessibility.sh"; validator = Path(__file__)
    entry_text, runner_text, validator_text = entry.read_text(encoding="utf-8"), runner.read_text(encoding="utf-8"), validator.read_text(encoding="utf-8")
    corrected_findings = semantic_findings(entry_text, runner_text, validator_text)
    emit_record({"record_type": "CORRECTED_SOURCE_FINDINGS", "detected_finding_codes": corrected_findings})
    require(not corrected_findings, "corrected source violates semantic contract: " + json.dumps(corrected_findings))
    fixtures = isolated_fixtures(entry_text, runner_text, validator_text)
    base_findings = semantic_findings(base_source(repo, "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py"), base_source(repo, "experience_docx/tools/run_chd_rm_v4a_a1x_exact_half_accessibility.sh"), base_source(repo, "experience_docx/tools/validate_haze4k_v4a_a1x_route.py"))
    emit_record({"record_type": "AUTHORIZATION_BASE_FINDINGS", "base_commit": BASE_COMMIT, "detected_finding_codes": base_findings})
    require(set(CODES).issubset(base_findings), "authorization base does not report all nine findings: " + json.dumps(base_findings))
    self_proof = validator_self_proof(entry_text, runner_text, validator_text)
    frozen_hashes = {rel: digest(repo / rel) for rel in FROZEN}; require(frozen_hashes == FROZEN, "frozen scientific hash mismatch")
    payload = {"schema_version": 2, "route_id": ROUTE_ID, "entrypoint_sha256": digest(entry), "runner_sha256": digest(runner), "validator_sha256": digest(validator), "frozen_hashes": frozen_hashes, "isolated_negative_fixtures": fixtures, "authorization_base_detected_finding_codes": base_findings, "corrected_source_detected_finding_codes": corrected_findings, "validator_self_proof": self_proof, "proof_markers": list(MARKERS), "runtime_started": False, "data_accessed": False, "checkpoint_opened": False, "cloud_transport": False, "formal_authorized": False, "canary_touched": False, "locked_test_touched": False, "validator": "A1X_S0_RUNNER_CORRECTNESS_STATIC_VALIDATOR_OK"}
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("A1X_S0_RUNNER_CORRECTNESS_STATIC_VALIDATOR_OK")


if __name__ == "__main__": main()
