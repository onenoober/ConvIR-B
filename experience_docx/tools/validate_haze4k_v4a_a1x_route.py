#!/usr/bin/env python3
"""AST-only validation for the seven authorized S0 semantic corrections."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
BASE_COMMIT = "a9283820e0a73ad376fd00e62b5b1847ff66456f"
EXPECTED = ("1594_0.71_0.5.png", "1595_0.99_1.84.png", "1597_0.69_1.45.png", "1598_0.67_1.4.png", "159_0.6_1.46.png", "1600_0.78_1.77.png", "1603_0.54_0.74.png", "1607_0.91_0.88.png", "160_0.63_1.04.png", "1613_0.56_1.31.png", "1614_0.81_0.78.png", "1615_0.91_1.25.png", "1616_0.76_0.88.png", "1617_0.56_1.97.png", "1619_0.94_1.08.png", "1622_0.98_1.75.png", "1623_0.78_1.81.png", "1627_0.94_0.52.png", "1628_0.8_1.49.png", "1633_0.73_1.49.png", "1634_0.75_1.81.png", "1639_0.69_1.12.png", "1640_0.53_0.59.png", "1646_0.55_1.55.png", "1649_0.8_0.86.png", "1650_0.76_1.07.png", "1652_0.62_1.35.png", "1653_0.9_1.01.png", "1654_0.66_1.9.png", "1656_0.64_1.45.png", "1658_0.96_1.72.png", "1660_0.83_0.67.png")
CODES = ("TRUE_AND_SHUFFLED_CELLS_SHARE_LEARNED_STATE", "FROZEN_OPERATOR_CONTRACT_IS_LABEL_ONLY", "FROZEN_ACTIVE_SUPPORT_IS_REPLACED_BY_ALL_ONES", "ZERO_INITIALIZATION_NOOP_GATE_IS_MEASURED_AFTER_OPTIMIZER_UPDATES", "MATCHED_CUDA_COST_AND_MEMORY_GATE_IS_NOT_IMPLEMENTED", "LEARNED_STATE_RETENTION_AND_PHASE_LIFECYCLE_ARE_INCOMPLETE", "S0_TERMINAL_TUPLE_CANNOT_REPORT_THE_FROZEN_PASS_GATE")
REQUIRED = {CODES[0]: ("independent_cells", "independent_cell_bundle", "copy.deepcopy", "AdamW", "true_cell[\"optimizer\"] is shuffled_cell[\"optimizer\"]"), CODES[1]: ("render_d_ref", "render_d_rep", "operator_render", "renderer.render_reference", "renderer.render_representation"), CODES[2]: ("measured_active_support", "for_operator_and_shape", "active_count", "support.sum", "normalized_active_support_endpoint_mse"), CODES[3]: ("verify_zero_noop", "maximum != 0.0", "update_cell", "optimizer"), CODES[4]: ("torch.cuda.is_available", "torch.device(\"cuda\")", "torch.cuda.synchronize", "reset_peak_memory_stats", "max_memory_allocated", "len(native_shapes) != 2", "structural_s0_gate"), CODES[5]: ("retain_learned_states", "torch.save", "absolute_state_path", "relative_state_path", "sha256", "update_count"), CODES[6]: ("TERMINALS", "COMPLETED_GATE_PASS", "COMPLETED_GATE_FAIL", "COMPLETED_GATE_INCONCLUSIVE", "terminal_closeout")}
FROZEN = {"Dehazing/ITS/models/A1XAccess.py": "625b60368dde9316df3c506cd339b94650e05243edc7cb8090f0b5b555b6df33", "experience_docx/route_operations.json": "a5cbe7d8e0559ee75c1b53b152561f10f754befb4e289963da3c465520e3f22f"}
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def name_of(node):
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Attribute): return name_of(node.value) + "." + node.attr
    return ""


def subscript_key(node):
    return node.slice.value if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) else None


def direct_assignment(function, target):
    matches = [node for node in function.body if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == target]
    return matches[0] if len(matches) == 1 else None


def call_is(value, suffix):
    return isinstance(value, ast.Call) and name_of(value.func).endswith(suffix)


def has_name(node, expected):
    return any(isinstance(item, ast.Name) and item.id == expected for item in ast.walk(node))


def bundle_return_proof(function):
    returns = [node.value for node in function.body if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)]
    if len(returns) != 1: return False
    fields = {key.value: value for key, value in zip(returns[0].keys, returns[0].values) if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    return all(isinstance(fields.get(field), ast.Name) and fields[field].id == field for field in ("model", "head", "optimizer")) and isinstance(fields.get("initial_head_state"), ast.Name) and fields["initial_head_state"].id == "head_state" and isinstance(fields.get("updates"), ast.Constant) and fields["updates"].value == 0


def bundle_construction_proof(function):
    model = direct_assignment(function, "model")
    head = direct_assignment(function, "head")
    optimizer = direct_assignment(function, "optimizer")
    head_state = direct_assignment(function, "head_state")
    if not all((model, head, optimizer, head_state)) or not bundle_return_proof(function): return False
    model_copy = call_is(model.value, "copy.deepcopy") and len(model.value.args) == 1 and isinstance(model.value.args[0], ast.Name) and model.value.args[0].id == "official_model"
    fresh_head = call_is(head.value, "A1X_ACCESS_Head")
    head_only_optimizer = call_is(optimizer.value, "AdamW") and optimizer.value.args and has_name(optimizer.value.args[0], "head") and not has_name(optimizer.value.args[0], "model")
    copied_initial_state = call_is(head_state.value, "copy.deepcopy") and len(head_state.value.args) == 1 and call_is(head_state.value.args[0], "state_dict") and name_of(head_state.value.args[0].func.value) == "head"
    return model_copy and fresh_head and head_only_optimizer and copied_initial_state


def bundle_call(node):
    return isinstance(node, ast.Call) and name_of(node.func) == "independent_cell_bundle" and len(node.args) == 3 and isinstance(node.args[0], ast.Name) and node.args[0].id == "official_model" and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, int) and isinstance(node.args[2], ast.Name) and node.args[2].id == "torch"


def is_cell_slot(node, cell, slot):
    return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == cell and subscript_key(node) == slot


def identity_guard_proof(function):
    comparisons = [node for node in ast.walk(function) if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Is) and len(node.comparators) == 1]
    return all(any(is_cell_slot(node.left, "true_cell", field) and is_cell_slot(node.comparators[0], "shuffled_cell", field) for node in comparisons) for field in ("model", "head", "optimizer"))


def learned_state_identity_proof(text):
    tree = ast.parse(text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    bundle, cells = functions.get("independent_cell_bundle"), functions.get("independent_cells")
    if not bundle or not cells or not bundle_construction_proof(bundle) or not identity_guard_proof(cells): return False
    true_cell, shuffled_cell = direct_assignment(cells, "true_cell"), direct_assignment(cells, "shuffled_cell")
    if not true_cell or not shuffled_cell or not bundle_call(true_cell.value) or not bundle_call(shuffled_cell.value): return False
    return true_cell.value.args[1].value == shuffled_cell.value.args[1].value


def call_name_is(node, expected):
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and name_of(node.value.func) == expected


def range_count(node):
    return isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and name_of(node.iter.func) == "range" and len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Constant) and isinstance(node.iter.args[0].value, int) and node.iter.args[0].value > 0


def cuda_measurement_nodes(tree):
    profile = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cuda_profile_pair"), None)
    if not profile:
        return None
    measure = next((node for node in profile.body if isinstance(node, ast.FunctionDef) and node.name == "measure"), None)
    timed_loops = [node for node in measure.body if range_count(node)] if measure else []
    timed = next((node for node in timed_loops if any(call_name_is(item, "torch.cuda.reset_peak_memory_stats") for item in node.body)), None)
    measured_calls = [(index, item) for index, item in enumerate(timed.body) if call_name_is(item, "fn")] if timed else []
    post_sync = next((item for index, item in enumerate(timed.body) if call_name_is(item, "torch.cuda.synchronize") and measured_calls and index > measured_calls[0][0]), None) if timed else None
    return profile, measure, timed, measured_calls[0][1] if len(measured_calls) == 1 else None, post_sync


def constant(node, value):
    return isinstance(node, ast.Constant) and node.value == value


def subscript_of(node, owner, key):
    return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == owner and subscript_key(node) == key


def comparison(node, left, operator, right):
    return isinstance(node, ast.Compare) and node.left is left and len(node.ops) == 1 and isinstance(node.ops[0], operator) and len(node.comparators) == 1 and node.comparators[0] is right


def all_thresholds(value, key, limit):
    if not call_is(value, "all") or len(value.args) != 1 or value.keywords or not isinstance(value.args[0], ast.GeneratorExp): return False
    expression = value.args[0]
    if len(expression.generators) != 1 or expression.generators[0].ifs: return False
    generator = expression.generators[0]
    if not isinstance(generator.target, ast.Name) or generator.target.id != "cost" or not call_is(generator.iter, "values") or name_of(generator.iter.func.value) != "overheads": return False
    return isinstance(expression.elt, ast.Compare) and subscript_of(expression.elt.left, "cost", key) and len(expression.elt.ops) == 1 and isinstance(expression.elt.ops[0], ast.LtE) and len(expression.elt.comparators) == 1 and constant(expression.elt.comparators[0], limit)


def positive_finite_baseline_guard(node):
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.UnaryOp) or not isinstance(node.test.op, ast.Not) or not isinstance(node.test.operand, ast.Call) or name_of(node.test.operand.func) != "all" or node.test.operand.keywords or not node.body or not any(isinstance(item, ast.Raise) for item in node.body): return False
    values = node.test.operand.args
    if len(values) != 1 or not isinstance(values[0], ast.GeneratorExp): return False
    expression = values[0]
    if len(expression.generators) != 1 or expression.generators[0].ifs: return False
    generator = expression.generators[0]
    if not isinstance(generator.target, ast.Name) or generator.target.id != "value" or not isinstance(generator.iter, ast.Call) or name_of(generator.iter.func) != "baseline_cost.values" or generator.iter.args or generator.iter.keywords: return False
    if not isinstance(expression.elt, ast.BoolOp) or not isinstance(expression.elt.op, ast.And) or len(expression.elt.values) != 2: return False
    lower, upper = expression.elt.values
    return isinstance(lower, ast.Compare) and isinstance(lower.left, ast.Name) and lower.left.id == "value" and len(lower.ops) == 1 and isinstance(lower.ops[0], ast.Gt) and len(lower.comparators) == 1 and constant(lower.comparators[0], 0.0) and isinstance(upper, ast.Compare) and isinstance(upper.left, ast.Name) and upper.left.id == "value" and len(upper.ops) == 1 and isinstance(upper.ops[0], ast.Lt) and len(upper.comparators) == 1 and isinstance(upper.comparators[0], ast.Call) and name_of(upper.comparators[0].func) == "float" and len(upper.comparators[0].args) == 1 and not upper.comparators[0].keywords and constant(upper.comparators[0].args[0], "inf")


def overhead_assignment(node):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Subscript) or not isinstance(node.targets[0].value, ast.Name) or node.targets[0].value.id != "overheads" or not isinstance(node.targets[0].slice, ast.Name) or node.targets[0].slice.id != "shape": return False
    if not isinstance(node.value, ast.DictComp) or len(node.value.generators) != 1: return False
    generator = node.value.generators[0]
    if not isinstance(node.value.key, ast.Name) or node.value.key.id != "metric" or not isinstance(generator.target, ast.Name) or generator.target.id != "metric" or generator.ifs or not isinstance(generator.iter, ast.Tuple) or len(generator.iter.elts) != 3 or [item.value for item in generator.iter.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)] != ["macs", "median_seconds", "peak_memory_bytes"]: return False
    def metric_subscript(value, owner): return isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and value.value.id == owner and isinstance(value.slice, ast.Name) and value.slice.id == "metric"
    formula = node.value.value
    return isinstance(formula, ast.BinOp) and isinstance(formula.op, ast.Div) and metric_subscript(formula.right, "baseline_cost") and isinstance(formula.left, ast.BinOp) and isinstance(formula.left.op, ast.Mult) and constant(formula.left.left, 100.0) and isinstance(formula.left.right, ast.BinOp) and isinstance(formula.left.right.op, ast.Sub) and metric_subscript(formula.left.right.left, "augmented_cost") and metric_subscript(formula.left.right.right, "baseline_cost")


def structural_gate_def_use_proof(tree):
    owner = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "structural_s0_gate"), None)
    if not owner or [argument.arg for argument in owner.args.args] != ["head", "baseline", "augmented", "native_shapes", "torch"]: return False
    added, limit, cuda_cost, cuda_results = (direct_assignment(owner, name) for name in ("added_parameters", "parameter_limit", "cuda_cost", "cuda_results"))
    if not all((added, limit, cuda_cost, cuda_results)): return False
    parameter_count = call_is(added.value, "sum") and len(added.value.args) == 1 and isinstance(added.value.args[0], ast.GeneratorExp) and len(added.value.args[0].generators) == 1 and isinstance(added.value.args[0].generators[0].target, ast.Name) and added.value.args[0].generators[0].target.id == "parameter" and call_is(added.value.args[0].elt, "numel") and isinstance(added.value.args[0].elt.func.value, ast.Name) and added.value.args[0].elt.func.value.id == "parameter" and call_is(added.value.args[0].generators[0].iter, "parameters") and isinstance(added.value.args[0].generators[0].iter.func.value, ast.Name) and added.value.args[0].generators[0].iter.func.value.id == "head"
    parameter_limit = isinstance(limit.value, ast.Compare) and isinstance(limit.value.left, ast.Name) and limit.value.left.id == "added_parameters" and len(limit.value.ops) == 1 and isinstance(limit.value.ops[0], ast.LtE) and len(limit.value.comparators) == 1 and constant(limit.value.comparators[0], 300000)
    cuda_call = call_is(cuda_cost.value, "cuda_cost_contract") and [name_of(argument) for argument in cuda_cost.value.args] == ["baseline", "augmented", "native_shapes", "torch"] and not cuda_cost.value.keywords
    result_use = subscript_of(cuda_results.value, "cuda_cost", "results")
    gate = next((node for node in owner.body if isinstance(node, ast.If) and any(isinstance(item, ast.Raise) for item in node.body)), None)
    gate_uses = gate and isinstance(gate.test, ast.BoolOp) and isinstance(gate.test.op, ast.Or) and len(gate.test.values) == 2 and all(isinstance(item, ast.UnaryOp) and isinstance(item.op, ast.Not) for item in gate.test.values) and isinstance(gate.test.values[0].operand, ast.Name) and gate.test.values[0].operand.id == "parameter_limit" and call_is(gate.test.values[1].operand, "all") and len(gate.test.values[1].operand.args) == 1 and call_is(gate.test.values[1].operand.args[0], "values") and isinstance(gate.test.values[1].operand.args[0].func.value, ast.Name) and gate.test.values[1].operand.args[0].func.value.id == "cuda_results"
    returned = next((node.value for node in owner.body if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)), None)
    fields = {key.value: value for key, value in zip(returned.keys, returned.values) if isinstance(key, ast.Constant) and isinstance(key.value, str)} if returned else {}
    result_fields = fields.get("results")
    retained = isinstance(fields.get("added_parameters"), ast.Name) and fields["added_parameters"].id == "added_parameters" and isinstance(fields.get("cuda_cost"), ast.Name) and fields["cuda_cost"].id == "cuda_cost" and isinstance(result_fields, ast.Dict) and any(isinstance(key, ast.Constant) and key.value == "added_parameter_limit" and isinstance(value, ast.Name) and value.id == "parameter_limit" for key, value in zip(result_fields.keys, result_fields.values)) and any(key is None and isinstance(value, ast.Name) and value.id == "cuda_results" for key, value in zip(result_fields.keys, result_fields.values))
    return bool(parameter_count and parameter_limit and cuda_call and result_use and gate_uses and retained)


def cuda_cost_measurement_proof(text):
    tree = ast.parse(text); found = cuda_measurement_nodes(tree)
    if not found: return False
    profile, measure, timed, measured, post_sync = found
    unavailable = any(isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not) and isinstance(node.test.operand, ast.Call) and name_of(node.test.operand.func) == "torch.cuda.is_available" and any(isinstance(item, ast.Raise) for item in node.body) for node in profile.body)
    device = direct_assignment(profile, "device")
    probe = direct_assignment(profile, "probe")
    explicit_cuda = device and call_is(device.value, "torch.device") and len(device.value.args) == 1 and isinstance(device.value.args[0], ast.Constant) and device.value.args[0].value == "cuda"
    cuda_probe = probe and call_is(probe.value, "torch.zeros") and any(keyword.arg == "device" and isinstance(keyword.value, ast.Name) and keyword.value.id == "device" for keyword in probe.value.keywords)
    warmups = [node for node in measure.body if range_count(node) and node is not timed]
    matched_warmup = len(warmups) == 1 and len(warmups[0].body) == 1 and call_name_is(warmups[0].body[0], "fn")
    reset = [item for item in timed.body if call_name_is(item, "torch.cuda.reset_peak_memory_stats")]
    measured_index = next((index for index, item in enumerate(timed.body) if item is measured), None)
    syncs = [(index, item) for index, item in enumerate(timed.body) if call_name_is(item, "torch.cuda.synchronize")]
    pre_sync = [item for index, item in syncs if measured_index is not None and index < measured_index]
    peak_reads = [item for item in ast.walk(timed) if isinstance(item, ast.Call) and name_of(item.func) == "torch.cuda.max_memory_allocated" and len(item.args) == 1 and isinstance(item.args[0], ast.Name) and item.args[0].id == "device"]
    measured_call = call_name_is(measured, "fn") and len(measured.value.args) == 1 and isinstance(measured.value.args[0], ast.Name) and measured.value.args[0].id == "probe"
    mac_measure = next((node for node in profile.body if isinstance(node, ast.FunctionDef) and node.name == "measured_macs"), None)
    profiler = next((node for node in ast.walk(mac_measure) if isinstance(node, ast.With) and any(call_is(item.context_expr, "torch.profiler.profile") for item in node.items)), None) if mac_measure else None
    profiler_cuda = profiler and any(any(keyword.arg == "activities" and "ProfilerActivity.CUDA" in ast.unparse(keyword.value) for keyword in item.context_expr.keywords) and any(keyword.arg == "with_flops" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in item.context_expr.keywords) for item in profiler.items if call_is(item.context_expr, "torch.profiler.profile"))
    flops = direct_assignment(mac_measure, "flops") if mac_measure else None
    macs = direct_assignment(mac_measure, "macs") if mac_measure else None
    converted_macs = flops and macs and isinstance(macs.value, ast.BinOp) and isinstance(macs.value.op, ast.Div) and isinstance(macs.value.left, ast.Name) and macs.value.left.id == "flops" and isinstance(macs.value.right, ast.Constant) and macs.value.right.value == 2.0 and "event.flops" in ast.unparse(flops.value)
    contract = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cuda_cost_contract"), None)
    shape_guard = contract and any(isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Call) and name_of(node.test.left.func) == "len" and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.NotEq) and isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == 2 and any(isinstance(item, ast.Raise) for item in node.body) for node in contract.body)
    matched_paths = contract and any(isinstance(node, ast.DictComp) and isinstance(node.value, ast.Call) and name_of(node.value.func) == "cuda_profile_pair" and [name_of(arg) for arg in node.value.args] == ["baseline", "augmented", "shape", "torch"] and isinstance(node.generators[0].iter, ast.Name) and node.generators[0].iter.id == "native_shapes" for node in ast.walk(contract))
    per_shape_loops = [node for node in contract.body if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and name_of(node.iter.func) == "measurements.items"] if contract else []
    denominators = len(per_shape_loops) == 1 and any(positive_finite_baseline_guard(node) for node in per_shape_loops[0].body)
    overhead_formula = len(per_shape_loops) == 1 and any(overhead_assignment(node) for node in per_shape_loops[0].body)
    guard_index = next((index for index, node in enumerate(per_shape_loops[0].body) if positive_finite_baseline_guard(node)), None) if len(per_shape_loops) == 1 else None
    overhead_index = next((index for index, node in enumerate(per_shape_loops[0].body) if overhead_assignment(node)), None) if len(per_shape_loops) == 1 else None
    ordered_denominator = len(per_shape_loops) == 1 and denominators and overhead_formula and guard_index is not None and overhead_index is not None and guard_index < overhead_index
    results = direct_assignment(contract, "results") if contract else None
    frozen_thresholds = results and isinstance(results.value, ast.Dict) and {key.value: value for key, value in zip(results.value.keys, results.value.values) if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    threshold_fields = frozen_thresholds if isinstance(frozen_thresholds, dict) else {}
    frozen_thresholds = all_thresholds(threshold_fields.get("mac_overhead"), "macs", 10.0) and all_thresholds(threshold_fields.get("matched_median_cuda_latency_overhead"), "median_seconds", 15.0) and all_thresholds(threshold_fields.get("cuda_peak_memory_overhead"), "peak_memory_bytes", 15.0)
    all_metrics_gate = contract and any(isinstance(node, ast.If) and "all(results.values())" in ast.unparse(node.test) and any(isinstance(item, ast.Raise) for item in node.body) for node in contract.body)
    return bool(unavailable and explicit_cuda and cuda_probe and matched_warmup and len(reset) == 1 and len(pre_sync) == 1 and post_sync and measured_call and len(peak_reads) == 1 and profiler_cuda and converted_macs and shape_guard and matched_paths and denominators and overhead_formula and ordered_denominator and frozen_thresholds and all_metrics_gate and structural_gate_def_use_proof(tree))


def source_findings(text):
    ast.parse(text)
    findings = [] if learned_state_identity_proof(text) else [CODES[0]]
    findings.extend(code for code, markers in REQUIRED.items() if code not in (CODES[0], CODES[4]) and not all(marker in text for marker in markers))
    if not cuda_cost_measurement_proof(text): findings.append(CODES[4])
    return findings
def replace_once(text, old, new):
    if text.count(old) != 1: raise ValueError("fixture mutation anchor is not unique")
    return text.replace(old, new, 1)


def mutate_neg_sem_01(text):
    tree = ast.parse(text)
    bundle = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "independent_cell_bundle"), None)
    model = direct_assignment(bundle, "model") if bundle else None
    if not model or not call_is(model.value, "copy.deepcopy"): raise ValueError("NEG_SEM_01 model assignment anchor is not unique")
    lines = text.splitlines(keepends=True); starts = [0]
    for line in lines: starts.append(starts[-1] + len(line))
    start = starts[model.value.lineno - 1] + model.value.col_offset
    end = starts[model.value.end_lineno - 1] + model.value.end_col_offset
    return text[:start] + "official_model" + text[end:]


def mutate_neg_sem_05(text):
    tree = ast.parse(text); found = cuda_measurement_nodes(tree)
    if not found or not found[4]: raise ValueError("NEG_SEM_05 post-call CUDA synchronization anchor is not structurally unique")
    post_sync = found[4]; lines = text.splitlines(keepends=True); starts = [0]
    for line in lines: starts.append(starts[-1] + len(line))
    start = starts[post_sync.value.lineno - 1] + post_sync.value.col_offset
    end = starts[post_sync.value.end_lineno - 1] + post_sync.value.end_col_offset
    mutated = text[:start] + "time.sleep()" + text[end:]
    return mutated, {"function": "cuda_profile_pair.measure", "role": "post_measured_call_synchronization", "call_target": "torch.cuda.synchronize", "line": post_sync.value.lineno, "column": post_sync.value.col_offset}


def isolated_fixtures(text):
    mutations = ((CODES[0], mutate_neg_sem_01), (CODES[1], lambda source: replace_once(source, "renderer.render_representation", "renderer.render_reference")), (CODES[2], lambda source: replace_once(source, "for_operator_and_shape", "ones_like")), (CODES[3], lambda source: replace_once(source, "def verify_zero_noop", "def removed_noop")), (CODES[4], mutate_neg_sem_05), (CODES[5], lambda source: replace_once(source, "torch.save", "write_json")), (CODES[6], lambda source: replace_once(source, "COMPLETED_GATE_PASS", "COMPLETED_GATE_BLOCKED")))
    records = []
    for index, (expected, mutate) in enumerate(mutations, 1):
        result = mutate(text); mutated, anchor = result if isinstance(result, tuple) else (result, None); detected = source_findings(mutated)
        if detected != [expected]: raise ValueError("isolated negative failed: " + expected + ":" + json.dumps(detected))
        record = {"id": "NEG_SEM_%02d" % index, "expected_code": expected, "detected_finding_codes": detected, "isolated": True, "target_code_executed_false": True}
        if anchor: record["ast_anchor_identity"] = anchor
        records.append(record)
    return records


def aggregate_fixture(text):
    mutations = (mutate_neg_sem_01, lambda source: replace_once(source, "renderer.render_representation", "renderer.render_reference"), lambda source: replace_once(source, "for_operator_and_shape", "ones_like"), lambda source: replace_once(source, "def verify_zero_noop", "def removed_noop"), mutate_neg_sem_05, lambda source: replace_once(source, "torch.save", "write_json"), lambda source: replace_once(source, "COMPLETED_GATE_PASS", "COMPLETED_GATE_BLOCKED"))
    mutated = text
    for mutate in mutations:
        result = mutate(mutated); mutated = result[0] if isinstance(result, tuple) else result
    detected = sorted(source_findings(mutated))
    if detected != sorted(CODES): raise ValueError("aggregate negative fixture failed: " + json.dumps(detected))
    return detected
def committed_first32(repo):
    text = subprocess.run(["git", "-C", str(repo), "show", BASE_COMMIT + ":experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py"], check=True, text=True, capture_output=True).stdout
    tree = ast.parse(text); values = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "S0_FIRST32_NAMES" for t in node.targets)]
    return tuple(ast.literal_eval(values[0]))
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", required=True); parser.add_argument("--output-json", required=True); args = parser.parse_args(); repo = Path(args.repo)
    entry = repo / "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py"; text = entry.read_text(encoding="utf-8")
    findings = source_findings(text)
    if findings: raise ValueError("corrected source findings: " + json.dumps(findings))
    if EXPECTED != committed_first32(repo) or "1650_0.76_1.07.png" not in EXPECTED: raise ValueError("authoritative first32 equality failed")
    fixtures = isolated_fixtures(text); aggregate = aggregate_fixture(text)
    frozen = {path: digest(repo / path) for path in FROZEN}
    if frozen != FROZEN: raise ValueError("frozen scientific source hash mismatch")
    payload = {"schema_version": 2, "route_id": ROUTE_ID, "entrypoint_sha256": digest(entry), "validator_sha256": digest(Path(__file__)), "corrected_source_findings": findings, "isolated_negative_controls": fixtures, "aggregate_negative_codes": aggregate, "aggregate_fixture_analyzed": True, "authoritative_first32_equal": True, "frozen_hashes": frozen, "runtime_started": False, "target_code_executed": False, "target_code_imported": False, "data_accessed": False, "checkpoint_opened": False, "cloud_transport": False, "s0_authorized": False, "formal_authorized": False, "canary_touched": False, "locked_test_touched": False, "validator": "A1X_S0_SEMANTICS_REPAIR_STATIC_VALIDATOR_OK"}
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("A1X_S0_SEMANTICS_REPAIR_STATIC_VALIDATOR_OK")
if __name__ == "__main__": main()
