#!/usr/bin/env python3
"""Source-only proof and mutation validation for the guarded A1X S0 route."""
import argparse
import ast
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
SEMANTIC_BASE_COMMIT = "53e28f9672495f2291d08b1f02567329fa66e7b1"
IMPLEMENTATION_PARENT = "493acb91dcf5da4e9d14840f1e0cefcb14c37273"
AUTHORIZATION_PATH = "experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715/v4a_a1x_r3_semantic_preserving_runtime_asset_repair_authorization.json"
CANONICAL_CONTRACT_COMMIT = "2681b6737515f24062b287a60b83c2b41dabc98c"
CANONICAL_CONTRACT_PATH = "experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715/v4a_a1x_r3_runtime_asset_contract_r2_repair_authorization.json"
ROUTE_CARD_COMMIT = "a8acf26f9dfdab4df3ea4737c53cbbeb82d87f99"
ROUTE_CARD_PATH = "experience_docx/experiment_cards/2026-07-15-haze4k-v5-v4a-a1x-exact-half-deployable-accessibility.md"
DESIGN_HANDOFF_PATH = "experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715/v4a_a1x_r3_design_handoff.json"
EXPECTED = ("1594_0.71_0.5.png", "1595_0.99_1.84.png", "1597_0.69_1.45.png", "1598_0.67_1.4.png", "159_0.6_1.46.png", "1600_0.78_1.77.png", "1603_0.54_0.74.png", "1607_0.91_0.88.png", "160_0.63_1.04.png", "1613_0.56_1.31.png", "1614_0.81_0.78.png", "1615_0.91_1.25.png", "1616_0.76_0.88.png", "1617_0.56_1.97.png", "1619_0.94_1.08.png", "1622_0.98_1.75.png", "1623_0.78_1.81.png", "1627_0.94_0.52.png", "1628_0.8_1.49.png", "1633_0.73_1.49.png", "1634_0.75_1.81.png", "1639_0.69_1.12.png", "1640_0.53_0.59.png", "1646_0.55_1.55.png", "1649_0.8_0.86.png", "1650_0.76_1.07.png", "1652_0.62_1.35.png", "1653_0.9_1.01.png", "1654_0.66_1.9.png", "1656_0.64_1.45.png", "1658_0.96_1.72.png", "1660_0.83_0.67.png")
CLI_OPTIONS = ("--stage", "--authorization-json", "--route-commit", "--run-id", "--run-root", "--runtime-asset-manifest-json", "--runtime-asset-manifest-sha256", "--status-json", "--heartbeat-json", "--learned-state-manifest-json", "--closeout-json")
CONTRACT_CODES = ("RUNNER_ENTRYPOINT_CLI_ARGUMENT_SET_MISMATCH", "INITIAL_AUTHORIZATION_TUPLE_OR_DOMINANCE_INVALID", "S0_LIFECYCLE_UNREACHABLE_OR_INCOMPLETE", "RUNTIME_ASSET_MANIFEST_IDENTITY_UNGUARDED", "RUNTIME_ASSET_OPERATOR_PROVENANCE_INVALID", "RUNTIME_ASSET_SUPPORT_PROVENANCE_INVALID", "RUNTIME_ASSET_FIRST32_OR_ENUMERATION_INVALID", "RUNTIME_ASSET_PATH_CONTENT_VALIDATION_INVALID", "RUNTIME_ASSET_NATIVE_SHAPE_TRANSPORT_INVALID")
SEMANTIC_CODES = ("TRUE_AND_SHUFFLED_CELLS_SHARE_LEARNED_STATE", "FROZEN_OPERATOR_CONTRACT_IS_LABEL_ONLY", "FROZEN_ACTIVE_SUPPORT_IS_REPLACED_BY_ALL_ONES", "ZERO_INITIALIZATION_NOOP_GATE_IS_MEASURED_AFTER_OPTIMIZER_UPDATES", "MATCHED_CUDA_COST_AND_MEMORY_GATE_IS_NOT_IMPLEMENTED", "LEARNED_STATE_RETENTION_AND_PHASE_LIFECYCLE_ARE_INCOMPLETE", "S0_TERMINAL_TUPLE_CANNOT_REPORT_THE_FROZEN_PASS_GATE")
ALL_CODES = CONTRACT_CODES + SEMANTIC_CODES
NEGATIVE_IDS = ("NEG_CONTRACT_CLI_01", "NEG_CONTRACT_AUTH_01", "NEG_CONTRACT_LIFECYCLE_01", "NEG_ASSET_MANIFEST_01", "NEG_ASSET_OPERATOR_01", "NEG_ASSET_SUPPORT_01", "NEG_ASSET_FIRST32_01", "NEG_ASSET_PATH_CONTENT_01", "NEG_ASSET_SHAPE_01", "NEG_SEM_01", "NEG_SEM_02", "NEG_SEM_03", "NEG_SEM_04", "NEG_SEM_05", "NEG_SEM_06", "NEG_SEM_07")
POSITIVE_IDS = ("POS_CLI_01", "POS_AUTH_01", "POS_AUTH_02", "POS_MANIFEST_01", "POS_ASSET_01", "POS_ASSET_02", "POS_ASSET_03", "POS_ASSET_04", "POS_DATA_01", "POS_DATA_02", "POS_SEM_01", "POS_SEM_02", "POS_SEM_03", "POS_SEM_04", "POS_SEM_05", "POS_SEM_06", "POS_SEM_07", "POS_RUNNER_01", "POS_FROZEN_01")
FROZEN_HASHES = {"Dehazing/ITS/models/A1XAccess.py": "625b60368dde9316df3c506cd339b94650e05243edc7cb8090f0b5b555b6df33", "route_card": "c723e59a3cb06de63b1ae4a72eabcd64ca0a3d08ba1346f3a53e0a52cce452da", "design_handoff": "630b81edb07fd6a4f4243c529998e5c97fbeb834926d3fd9cde5ce26bca6340a"}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    return digest_bytes(Path(path).read_bytes())


def git_show(repo, commit, path):
    return subprocess.run(["git", "-C", str(repo), "show", commit + ":" + path], check=True, capture_output=True).stdout


def name_of(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return name_of(node.value) + "." + node.attr
    return ""


def function_map(tree):
    return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def direct_assignment(function, target):
    matches = [node for node in function.body if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == target]
    return matches[0] if len(matches) == 1 else None


def literal_assignment(tree, target):
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(item, ast.Name) and item.id == target for item in node.targets)]
    if len(matches) != 1:
        return None
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError):
        return None


def calls(node, target):
    return [item for item in ast.walk(node) if isinstance(item, ast.Call) and name_of(item.func) == target]


def call_suffix(node, suffix):
    return isinstance(node, ast.Call) and name_of(node.func).endswith(suffix)


def subscript_key(node):
    return node.slice.value if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) else None


def bundle_return_proof(function):
    returns = [node.value for node in function.body if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)]
    if len(returns) != 1:
        return False
    fields = {key.value: value for key, value in zip(returns[0].keys, returns[0].values) if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    return all(isinstance(fields.get(field), ast.Name) and fields[field].id == field for field in ("model", "head", "optimizer")) and isinstance(fields.get("initial_head_state"), ast.Name) and fields["initial_head_state"].id == "head_state" and isinstance(fields.get("updates"), ast.Constant) and fields["updates"].value == 0


def learned_state_identity_proof(tree):
    functions = function_map(tree)
    bundle, cells = functions.get("independent_cell_bundle"), functions.get("independent_cells")
    if not bundle or not cells or not bundle_return_proof(bundle):
        return False
    model, head, optimizer, head_state = (direct_assignment(bundle, name) for name in ("model", "head", "optimizer", "head_state"))
    if not all((model, head, optimizer, head_state)):
        return False
    construction = call_suffix(model.value, "copy.deepcopy") and call_suffix(head.value, "A1X_ACCESS_Head") and call_suffix(optimizer.value, "AdamW") and call_suffix(head_state.value, "copy.deepcopy")
    true_cell, shuffled_cell = direct_assignment(cells, "true_cell"), direct_assignment(cells, "shuffled_cell")
    calls_match = true_cell and shuffled_cell and call_suffix(true_cell.value, "independent_cell_bundle") and call_suffix(shuffled_cell.value, "independent_cell_bundle") and ast.unparse(true_cell.value.args[1]) == "3407" and ast.unparse(shuffled_cell.value.args[1]) == "3407"
    guards = [node for node in ast.walk(cells) if isinstance(node, ast.Compare) and any(isinstance(operator, ast.Is) for operator in node.ops)]
    guard_text = "\n".join(ast.unparse(node) for node in guards)
    identities = all(f"true_cell['{slot}'] is shuffled_cell['{slot}']" in guard_text for slot in ("model", "head", "optimizer"))
    return bool(construction and calls_match and identities)


def operator_semantic_proof(tree):
    functions = function_map(tree)
    reference, representation, dispatch = (functions.get(name) for name in ("render_d_ref", "render_d_rep", "operator_render"))
    if not all((reference, representation, dispatch)):
        return False
    return len(calls(reference, "renderer.render_reference")) == 1 and len(calls(representation, "renderer.render_representation")) == 1 and "render_d_ref" in ast.unparse(dispatch) and "render_d_rep" in ast.unparse(dispatch)


def support_semantic_proof(tree):
    functions = function_map(tree)
    support, loss = functions.get("measured_active_support"), functions.get("normalized_active_support_endpoint_mse")
    if not support or not loss:
        return False
    assignment = direct_assignment(support, "support")
    return bool(assignment and call_suffix(assignment.value, "for_operator_and_shape") and "support.sum()" in ast.unparse(support) and "active_count" in ast.unparse(loss) and "support.to(predicted.dtype)" in ast.unparse(loss))


def noop_semantic_proof(tree):
    functions = function_map(tree)
    noop, lifecycle = functions.get("verify_zero_noop"), functions.get("runtime_lifecycle")
    if not noop or not lifecycle or "maximum != 0.0" not in ast.unparse(noop):
        return False
    noop_calls = calls(lifecycle, "verify_zero_noop")
    update_calls = calls(lifecycle, "update_cell")
    return bool(noop_calls and update_calls and max(node.lineno for node in noop_calls) < min(node.lineno for node in update_calls))


def cuda_semantic_proof(tree):
    functions = function_map(tree)
    profile, contract, structural = (functions.get(name) for name in ("cuda_profile_pair", "cuda_cost_contract", "structural_s0_gate"))
    if not all((profile, contract, structural)):
        return False
    measure = next((node for node in profile.body if isinstance(node, ast.FunctionDef) and node.name == "measure"), None)
    measured_macs = next((node for node in profile.body if isinstance(node, ast.FunctionDef) and node.name == "measured_macs"), None)
    if not measure or not measured_macs:
        return False
    loops = [node for node in measure.body if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and name_of(node.iter.func) == "range"]
    counts = [ast.literal_eval(node.iter.args[0]) for node in loops if len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Constant)]
    timed = next((node for node in loops if ast.literal_eval(node.iter.args[0]) == 11), None)
    timed_calls = [(index, name_of(item.value.func)) for index, item in enumerate(timed.body) if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)] if timed else []
    fn_index = next((index for index, name in timed_calls if name == "fn"), None)
    post_sync = any(name == "torch.cuda.synchronize" and fn_index is not None and index > fn_index for index, name in timed_calls)
    profile_text, contract_text, structural_text = ast.unparse(profile), ast.unparse(contract), ast.unparse(structural)
    return bool(sorted(counts) == [5, 11] and post_sync and "torch.profiler.ProfilerActivity.CUDA" in ast.unparse(measured_macs) and "flops / 2.0" in ast.unparse(measured_macs) and "len(native_shapes) != 2" in contract_text and "baseline_cost.values()" in contract_text and "<= 10.0" in contract_text and contract_text.count("<= 15.0") == 2 and "torch.cuda.is_available()" in profile_text and "added_parameters <= 300000" in structural_text and calls(structural, "cuda_cost_contract"))


def retention_semantic_proof(tree):
    functions = function_map(tree)
    retain, lifecycle = functions.get("retain_learned_states"), functions.get("runtime_lifecycle")
    return bool(retain and lifecycle and calls(retain, "torch.save") and calls(retain, "sha256") and calls(lifecycle, "retain_learned_states") and "absolute_state_path" in ast.unparse(retain) and "relative_state_path" in ast.unparse(retain) and "update_count" in ast.unparse(retain))


def terminal_semantic_proof(tree):
    terminals = literal_assignment(tree, "TERMINALS")
    expected = {"PASS": ("COMPLETED_GATE_PASS", "V4A_A1X_S0_PASS_AUTHORIZE_FORMAL_ONLY", "A1X_FORMAL_CONFIRMATION_ONLY"), "FAIL": ("COMPLETED_GATE_FAIL", "V4A_A1X_S0_ENGINEERING_GATE_FAIL_STOP", "NONE"), "INCONCLUSIVE": ("COMPLETED_GATE_INCONCLUSIVE", "V4A_A1X_S0_ENGINEERING_GATE_INCONCLUSIVE_STOP", "NONE")}
    functions = function_map(tree)
    lifecycle_text = ast.unparse(functions["runtime_lifecycle"])
    inconclusive_text = ast.unparse(functions["write_runtime_inconclusive"])
    reachable = "terminal_kind = 'PASS'" in lifecycle_text and "terminal_kind = 'FAIL'" in lifecycle_text and "terminal_closeout(terminal_kind" in lifecycle_text and "terminal_closeout('INCONCLUSIVE'" in inconclusive_text
    return terminals == expected and functions.get("terminal_closeout") is not None and reachable


def semantic_results(tree):
    return {
        SEMANTIC_CODES[0]: learned_state_identity_proof(tree),
        SEMANTIC_CODES[1]: operator_semantic_proof(tree),
        SEMANTIC_CODES[2]: support_semantic_proof(tree),
        SEMANTIC_CODES[3]: noop_semantic_proof(tree),
        SEMANTIC_CODES[4]: cuda_semantic_proof(tree),
        SEMANTIC_CODES[5]: retention_semantic_proof(tree),
        SEMANTIC_CODES[6]: terminal_semantic_proof(tree),
    }


def parser_options(tree):
    parser_function = function_map(tree).get("parser")
    loops = [node for node in parser_function.body if isinstance(node, ast.For)] if parser_function else []
    if len(loops) != 1:
        return ()
    try:
        options = tuple(ast.literal_eval(loops[0].iter))
    except (ValueError, TypeError):
        return ()
    required_calls = [node for node in ast.walk(loops[0]) if isinstance(node, ast.Call) and name_of(node.func).endswith("add_argument") and any(keyword.arg == "required" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in node.keywords)]
    return options if len(required_calls) == 1 else ()


def runner_options(shell):
    invocation = next((line for line in shell.splitlines() if '"${EXPLICIT_CLOUD_PYTHON}" "${ENTRYPOINT}"' in line), "")
    return tuple(re.findall(r"--[a-z0-9-]+", invocation)), invocation


def authorization_proof(tree):
    functions = function_map(tree)
    guard, main = functions.get("authorization_guard"), functions.get("main")
    expected = direct_assignment(guard, "expected") if guard else None
    required = direct_assignment(guard, "required") if guard else None
    try:
        expected_value = ast.literal_eval(expected.value)
    except (AttributeError, ValueError, TypeError):
        expected_value = None
    required_keys = {key.value for key in required.value.keys if isinstance(key, ast.Constant)} if required and isinstance(required.value, ast.Dict) else set()
    exact_tuple = expected_value == {"route_id": ROUTE_ID, "state": "PLANNED", "decision": "V4A_A1X_S0_AUTHORIZED_INITIAL_ONLY", "authorizes": "A1X_S0_ONLY"}
    exact_fields = required_keys == {"source_commit", "route_commit", "route_card_sha256", "locked_test_policy", "contract_id", "runtime_asset_manifest_json", "runtime_asset_manifest_sha256"}
    top_imports = {alias.name.split(".")[0] for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    protected_absent = not top_imports.intersection({"numpy", "torch", "PIL", "Dehazing"})
    main_text = ast.unparse(main) if main else ""
    ordered = all(token in main_text for token in ("authorization_guard(args)", "validate_manifest(args, authorization)", "runtime_lifecycle(args, manifest, records)")) and main_text.index("authorization_guard(args)") < main_text.index("validate_manifest(args, authorization)") < main_text.index("runtime_lifecycle(args, manifest, records)")
    first_open = guard and calls(guard, "read_json") and not calls(main, "read_json")
    return bool(exact_tuple and exact_fields and protected_absent and ordered and first_open)


def lifecycle_proof(tree):
    functions = function_map(tree)
    lifecycle = functions.get("runtime_lifecycle")
    if not lifecycle:
        return False
    text = ast.unparse(lifecycle)
    phases = {node.args[1].value for node in calls(lifecycle, "append_status") if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)}
    required_phases = {"assets_verified", "cells_initialized", "cell_progress", "cost_gate", "states_written", "terminal"}
    required_calls = {"build_operator_batches", "load_frozen_official_model", "independent_cells", "verify_zero_noop", "update_cell", "structural_s0_gate", "retain_learned_states", "terminal_closeout"}
    observed_calls = {name_of(node.func) for node in ast.walk(lifecycle) if isinstance(node, ast.Call)}
    main = functions.get("main")
    return phases == required_phases and required_calls.issubset(observed_calls) and main is not None and calls(main, "runtime_lifecycle") and "return" not in text.split("terminal_closeout", 1)[0]


def manifest_identity_proof(tree):
    function = function_map(tree).get("validate_manifest")
    if not function:
        return False
    hash_calls = calls(function, "sha256")
    parse_calls = calls(function, "read_json")
    return len(hash_calls) == 1 and len(parse_calls) == 1 and hash_calls[0].lineno < parse_calls[0].lineno and "authorization['runtime_asset_manifest_json']" in ast.unparse(function) and "authorization['runtime_asset_manifest_sha256']" in ast.unparse(function)


def asset_operator_proof(tree):
    assets = literal_assignment(tree, "PINNED_OPERATOR_ASSETS")
    if not isinstance(assets, dict) or set(assets) != {"manifest", "D_ref", "D_rep"}:
        return False
    distinct = all(assets["D_ref"][key] != assets["D_rep"][key] for key in ("seed", "path", "sha256", "bytes"))
    functions = function_map(tree)
    validate = functions.get("validate_static_assets")
    return distinct and validate is not None and "PINNED_OPERATOR_ASSETS" in ast.unparse(validate) and "ordered_operators" in ast.unparse(validate)


def asset_support_proof(tree):
    functions = function_map(tree)
    load, build = functions.get("load_validated_array"), functions.get("build_operator_batches")
    if not load or not build:
        return False
    load_text, build_text = ast.unparse(load), ast.unparse(build)
    return "np.bool_ if role == 'measured_active_support_native'" in load_text and "not value.any()" in load_text and "support=role == 'measured_active_support_native'" in build_text and not calls(build, "torch.ones_like")


def first32_proof(tree):
    names = literal_assignment(tree, "S0_FIRST32_NAMES")
    forbidden = {"glob", "rglob", "iterdir", "listdir", "scandir", "walk"}
    call_names = {name_of(node.func).split(".")[-1] for node in ast.walk(tree) if isinstance(node, ast.Call)}
    return names == EXPECTED and len(set(names or ())) == 32 and not call_names.intersection(forbidden)


def path_content_proof(tree):
    descriptor_fields = literal_assignment(tree, "DESCRIPTOR_FIELDS")
    functions = function_map(tree)
    canonical, identity, descriptor, loader = (functions.get(name) for name in ("canonical_regular_file", "validate_file_identity", "validate_descriptor", "load_validated_array"))
    if not all((canonical, identity, descriptor, loader)):
        return False
    combined = "\n".join(ast.unparse(node) for node in (canonical, identity, descriptor, loader))
    required_calls = {"is_absolute", "is_symlink", "is_file", "resolve", "relative_to", "stat", "sha256"}
    observed = {name_of(node.func).split(".")[-1] for node in ast.walk(ast.Module(body=[canonical, identity, descriptor, loader], type_ignores=[])) if isinstance(node, ast.Call)}
    return descriptor_fields == ("path", "sha256", "bytes", "encoding", "dtype", "shape", "producer_role") and required_calls.issubset(observed) and all(token in combined for token in ("npy_allow_pickle_false", "float32", "bool", "np.isfinite(value).all()", "value.any()"))


def shape_transport_proof(tree):
    pairs = literal_assignment(tree, "SHAPE_PAIRS")
    function = function_map(tree).get("exact_half")
    if not function:
        return False
    text = ast.unparse(function)
    return pairs == (((400, 400), (208, 208)), ((480, 640), (240, 320))) and "mode='bilinear'" in text and "align_corners=False" in text and "antialias=False" in text and "transported > 0.0 if support else transported" in text


def projection_proof(operations, canonical_contract):
    if operations.get("schema_version") != 2 or len(operations.get("operations", [])) != 1:
        return False
    operation = operations["operations"][0]
    contract = operation.get("runtime_asset_contract", {})
    canonical_equal = all(key in contract and contract[key] == value for key, value in canonical_contract.items())
    capture = contract.get("lifecycle_capture", {})
    capture_ok = capture.get("status_phases") == ["authorized", "assets_verified", "cells_initialized", "cell_progress", "cost_gate", "states_written", "terminal"] and capture.get("runner_combined_stdout_stderr") is True and capture.get("runner_pipe_status_required") is True and capture.get("terminal_marker_owner") == "runner_only"
    return canonical_equal and contract.get("manifest_path_env") == "A1X_RUNTIME_ASSET_MANIFEST_JSON" and contract.get("manifest_sha256_env") == "A1X_RUNTIME_ASSET_MANIFEST_SHA256" and capture_ok


def runner_capture_proof(shell):
    _, invocation = runner_options(shell)
    return bool(invocation and '2>&1 | tee "${RUNTIME_LOG_PATH}"' in invocation and 'exit_code="${PIPESTATUS[0]}"' in shell and 'if [ ! -f "${CLOSEOUT_PATH}" ]; then exit_code=2; fi' in shell and shell.count('"A1X_S0_OK"') == 1 and shell.count('"A1X_S0_FAILED"') == 1 and "RUNTIME_LOG_PATH=" in shell)


def contract_results(bundle, canonical_contract):
    tree = ast.parse(bundle["entry"])
    options, invocation = runner_options(bundle["runner"])
    operations = json.loads(bundle["operations"])
    cli_ok = parser_options(tree) == CLI_OPTIONS and options == CLI_OPTIONS and "--stage s0" in invocation
    auth_ok = authorization_proof(tree)
    lifecycle_ok = lifecycle_proof(tree)
    manifest_ok = manifest_identity_proof(tree)
    operator_ok = asset_operator_proof(tree) and projection_proof(operations, canonical_contract)
    support_ok = asset_support_proof(tree)
    names_ok = first32_proof(tree)
    path_ok = path_content_proof(tree)
    shape_ok = shape_transport_proof(tree)
    return dict(zip(CONTRACT_CODES, (cli_ok, auth_ok, lifecycle_ok, manifest_ok, operator_ok, support_ok, names_ok, path_ok, shape_ok)))


def analyze_bundle(bundle, canonical_contract):
    tree = ast.parse(bundle["entry"])
    results = {**contract_results(bundle, canonical_contract), **semantic_results(tree)}
    return [code for code in ALL_CODES if not results[code]], results


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("fixture mutation anchor is not unique: " + old)
    return text.replace(old, new, 1)


def replace_ast_node(text, node, replacement):
    lines = text.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    start = starts[node.lineno - 1] + node.col_offset
    end = starts[node.end_lineno - 1] + node.end_col_offset
    return text[:start] + replacement + text[end:]


def mutate_sem_01(bundle):
    result = copy.deepcopy(bundle)
    tree = ast.parse(result["entry"])
    model = direct_assignment(function_map(tree)["independent_cell_bundle"], "model")
    result["entry"] = replace_ast_node(result["entry"], model.value, "official_model")
    return result


def mutate_sem_03(bundle):
    result = copy.deepcopy(bundle)
    old = 'support = frozen_support_artifact.for_operator_and_shape(operator_batch["operator"], operator_batch["native_shape"], operator_batch)'
    result["entry"] = replace_once(result["entry"], old, 'support = torch.ones_like(operator_batch["measured_active_support_exact"], dtype=torch.bool)')
    return result


def mutate_sem_05(bundle):
    result = copy.deepcopy(bundle)
    tree = ast.parse(result["entry"])
    measure = next(node for node in function_map(tree)["cuda_profile_pair"].body if isinstance(node, ast.FunctionDef) and node.name == "measure")
    timed = next(node for node in measure.body if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and ast.literal_eval(node.iter.args[0]) == 11)
    fn_index = next(index for index, node in enumerate(timed.body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and name_of(node.value.func) == "fn")
    post_sync = next(node for index, node in enumerate(timed.body) if index > fn_index and isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and name_of(node.value.func) == "torch.cuda.synchronize")
    result["entry"] = replace_ast_node(result["entry"], post_sync.value, "time.sleep()")
    return result


def text_mutation(target, old, new):
    def mutate(bundle):
        result = copy.deepcopy(bundle)
        result[target] = replace_once(result[target], old, new)
        return result
    return mutate


MUTATIONS = (
    text_mutation("runner", '--status-json "${STATUS_PATH}" ', ""),
    text_mutation("entry", '"state": "PLANNED"', '"state": "PLANNED_WEAK"'),
    text_mutation("entry", 'append_status(args, "states_written"', 'append_status(args, "states_disconnected"'),
    text_mutation("entry", "sha256(manifest_path)", '("0" * 64)'),
    text_mutation("entry", '"seed": 3408', '"seed": 3407'),
    text_mutation("entry", 'support=role == "measured_active_support_native"', "support=False"),
    text_mutation("entry", '"1594_0.71_0.5.png"', '"1594_MUTATED.png"'),
    text_mutation("entry", '"sha256", "bytes", "encoding"', '"sha256", "encoding"'),
    text_mutation("entry", "((480, 640), (240, 320))", "((480, 640), (240, 321))"),
    mutate_sem_01,
    text_mutation("entry", "renderer.render_representation", "renderer.render_reference"),
    mutate_sem_03,
    text_mutation("entry", "def verify_zero_noop", "def removed_noop"),
    mutate_sem_05,
    text_mutation("entry", "torch.save", "write_json"),
    text_mutation("entry", "COMPLETED_GATE_PASS", "COMPLETED_GATE_BLOCKED"),
)


def isolated_fixtures(bundle, canonical_contract):
    records = []
    for identifier, expected, mutate in zip(NEGATIVE_IDS, ALL_CODES, MUTATIONS):
        mutated = mutate(bundle)
        detected, _ = analyze_bundle(mutated, canonical_contract)
        if detected != [expected]:
            raise ValueError("isolated negative failed: " + identifier + ":" + json.dumps(detected))
        records.append({"id": identifier, "expected_code": expected, "detected_finding_codes": detected, "isolated": True, "authorizes": "NONE", "target_code_executed_false": True})
    return records


def aggregate_fixture(bundle, canonical_contract):
    mutated = copy.deepcopy(bundle)
    for mutate in MUTATIONS:
        mutated = mutate(mutated)
    detected, _ = analyze_bundle(mutated, canonical_contract)
    if detected != list(ALL_CODES):
        raise ValueError("aggregate negative fixture failed: " + json.dumps(detected))
    return detected


def committed_first32(repo):
    text = git_show(repo, SEMANTIC_BASE_COMMIT, "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py").decode("utf-8")
    return literal_assignment(ast.parse(text), "S0_FIRST32_NAMES")


def frozen_source_proof(repo, operations, canonical_contract):
    baseline_operations = json.loads(git_show(repo, SEMANTIC_BASE_COMMIT, "experience_docx/route_operations.json"))
    current_operation = copy.deepcopy(operations["operations"][0])
    current_operation.pop("runtime_asset_contract", None)
    source_hashes = {
        "Dehazing/ITS/models/A1XAccess.py": digest(repo / "Dehazing/ITS/models/A1XAccess.py"),
        "route_card": digest_bytes(git_show(repo, ROUTE_CARD_COMMIT, ROUTE_CARD_PATH)),
        "design_handoff": digest_bytes(git_show(repo, ROUTE_CARD_COMMIT, DESIGN_HANDOFF_PATH)),
    }
    return source_hashes == FROZEN_HASHES and current_operation == baseline_operations["operations"][0] and projection_proof(operations, canonical_contract), source_hashes


def positive_proofs(repo, bundle, canonical_contract, results):
    tree = ast.parse(bundle["entry"])
    operations = json.loads(bundle["operations"])
    frozen_ok, frozen_hashes = frozen_source_proof(repo, operations, canonical_contract)
    semantic = semantic_results(tree)
    details = {
        "POS_CLI_01": {"parser_options": list(parser_options(tree)), "runner_options": list(runner_options(bundle["runner"])[0]), "stage": "s0"},
        "POS_AUTH_01": {"exact_future_initial_tuple": results[CONTRACT_CODES[1]], "route_id": ROUTE_ID},
        "POS_AUTH_02": {"guard_before_manifest_and_runtime": results[CONTRACT_CODES[1]], "top_level_target_imports": []},
        "POS_MANIFEST_01": {"path_and_byte_hash_before_parse": results[CONTRACT_CODES[3]]},
        "POS_ASSET_01": {"canonical_contract_projection": projection_proof(operations, canonical_contract), "source_checkpoint_operator_pins": asset_operator_proof(tree)},
        "POS_ASSET_02": {"record_count": canonical_contract["record_contract"]["record_count"], "first32_operator_cross_product": results[CONTRACT_CODES[6]]},
        "POS_ASSET_03": {"descriptor_fields": list(literal_assignment(tree, "DESCRIPTOR_FIELDS")), "path_content_predicate": results[CONTRACT_CODES[7]]},
        "POS_ASSET_04": {"per_name_shared_and_operator_distinct": results[CONTRACT_CODES[4]]},
        "POS_DATA_01": {"first32_count": len(EXPECTED), "committed_equal": committed_first32(repo) == EXPECTED},
        "POS_DATA_02": {"target_only_cyclic_shuffle": "shuffled_target_delta_u_native_exact" in ast.unparse(function_map(tree)["build_operator_batches"])},
        "POS_SEM_01": {"predicate": semantic[SEMANTIC_CODES[0]]},
        "POS_SEM_02": {"predicate": semantic[SEMANTIC_CODES[1]]},
        "POS_SEM_03": {"predicate": semantic[SEMANTIC_CODES[2]]},
        "POS_SEM_04": {"predicate": semantic[SEMANTIC_CODES[3]]},
        "POS_SEM_05": {"predicate": semantic[SEMANTIC_CODES[4]]},
        "POS_SEM_06": {"predicate": semantic[SEMANTIC_CODES[5]]},
        "POS_SEM_07": {"predicate": semantic[SEMANTIC_CODES[6]]},
        "POS_RUNNER_01": {"combined_capture_pipe_status_closeout_markers": runner_capture_proof(bundle["runner"])},
        "POS_FROZEN_01": {"frozen": frozen_ok, "hashes": frozen_hashes, "semantic_base_commit": SEMANTIC_BASE_COMMIT},
    }
    proofs = []
    for identifier in POSITIVE_IDS:
        passed = all(value is not False for value in details[identifier].values())
        proofs.append({"id": identifier, "passed": passed, "details": details[identifier]})
    if not all(record["passed"] for record in proofs):
        raise ValueError("positive proof failed: " + json.dumps([record["id"] for record in proofs if not record["passed"]]))
    return proofs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    validator_path = Path(__file__).resolve()
    entry = repo / "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py"
    runner = repo / "experience_docx/tools/run_chd_rm_v4a_a1x_exact_half_accessibility.sh"
    operations_path = repo / "experience_docx/route_operations.json"
    authorization_path = repo / AUTHORIZATION_PATH
    bundle = {"entry": entry.read_text(encoding="utf-8"), "runner": runner.read_text(encoding="utf-8"), "operations": operations_path.read_text(encoding="utf-8")}
    canonical_record = json.loads(git_show(repo, CANONICAL_CONTRACT_COMMIT, CANONICAL_CONTRACT_PATH))
    canonical_contract = canonical_record["runtime_asset_transport_contract"]
    findings, results = analyze_bundle(bundle, canonical_contract)
    if findings:
        raise ValueError("corrected source findings: " + json.dumps(findings))
    if committed_first32(repo) != EXPECTED or literal_assignment(ast.parse(bundle["entry"]), "S0_FIRST32_NAMES") != EXPECTED:
        raise ValueError("authoritative first32 equality failed")
    proofs = positive_proofs(repo, bundle, canonical_contract, results)
    fixtures = isolated_fixtures(bundle, canonical_contract)
    aggregate = aggregate_fixture(bundle, canonical_contract)
    if subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip() != IMPLEMENTATION_PARENT:
        raise ValueError("implementation parent commit mismatch")
    source_paths = (operations_path, runner, entry, validator_path)
    payload = {
        "schema_version": 2,
        "route_id": ROUTE_ID,
        "authorization_base_commit": "a17a2bbc44602ae19e16dbe21ef150d6e57387ae",
        "authorization_sha256": digest(authorization_path),
        "implementation_parent_commit": IMPLEMENTATION_PARENT,
        "post_repair_hashes": {str(path.relative_to(repo)): digest(path) for path in source_paths},
        "corrected_source_findings": findings,
        "positive_proofs": proofs,
        "independent_negative_controls": fixtures,
        "aggregate_negative_codes": aggregate,
        "aggregate_fixture_analyzed": True,
        "authoritative_first32_equal": True,
        "runtime_started": False,
        "runner_executed": False,
        "entrypoint_imported": False,
        "entrypoint_executed": False,
        "data_accessed": False,
        "checkpoint_opened": False,
        "runtime_manifest_opened": False,
        "cloud_transport": False,
        "initial_authorization_created": False,
        "s0_authorized": False,
        "formal_authorized": False,
        "canary_touched": False,
        "locked_test_touched": False,
        "validator": "A1X_S0_SEMANTIC_PRESERVING_RUNTIME_ASSET_REPAIR_STATIC_VALIDATOR_OK",
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("A1X_S0_SEMANTIC_PRESERVING_RUNTIME_ASSET_REPAIR_STATIC_VALIDATOR_OK")


if __name__ == "__main__":
    main()
