#!/usr/bin/env python3
"""Guarded future A1X S0 runner with immutable runtime-asset transport."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
SOURCE_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
ROUTE_CARD_SHA256 = "c723e59a3cb06de63b1ae4a72eabcd64ca0a3d08ba1346f3a53e0a52cce452da"
CONTRACT_ID = "A1X_S0_RUNTIME_ASSET_TRANSPORT_V1"
S0_FIRST32_NAMES = ("1594_0.71_0.5.png", "1595_0.99_1.84.png", "1597_0.69_1.45.png", "1598_0.67_1.4.png", "159_0.6_1.46.png", "1600_0.78_1.77.png", "1603_0.54_0.74.png", "1607_0.91_0.88.png", "160_0.63_1.04.png", "1613_0.56_1.31.png", "1614_0.81_0.78.png", "1615_0.91_1.25.png", "1616_0.76_0.88.png", "1617_0.56_1.97.png", "1619_0.94_1.08.png", "1622_0.98_1.75.png", "1623_0.78_1.81.png", "1627_0.94_0.52.png", "1628_0.8_1.49.png", "1633_0.73_1.49.png", "1634_0.75_1.81.png", "1639_0.69_1.12.png", "1640_0.53_0.59.png", "1646_0.55_1.55.png", "1649_0.8_0.86.png", "1650_0.76_1.07.png", "1652_0.62_1.35.png", "1653_0.9_1.01.png", "1654_0.66_1.9.png", "1656_0.64_1.45.png", "1658_0.96_1.72.png", "1660_0.83_0.67.png")
TERMINALS = {"PASS": ("COMPLETED_GATE_PASS", "V4A_A1X_S0_PASS_AUTHORIZE_FORMAL_ONLY", "A1X_FORMAL_CONFIRMATION_ONLY"), "FAIL": ("COMPLETED_GATE_FAIL", "V4A_A1X_S0_ENGINEERING_GATE_FAIL_STOP", "NONE"), "INCONCLUSIVE": ("COMPLETED_GATE_INCONCLUSIVE", "V4A_A1X_S0_ENGINEERING_GATE_INCONCLUSIVE_STOP", "NONE")}
OPERATORS = ("D_ref", "D_rep")
SHAPE_PAIRS = (((400, 400), (208, 208)), ((480, 640), (240, 320)))
DELTA_BOUND = (0.00919640064239502, 0.009401530027389526, 0.009474039077758789)
RECORD_ORDER = "native_image_hw ascending by the frozen pair order, then name lexicographically, then operator in [D_ref,D_rep] order"
MANIFEST_FIELDS = ("schema_version", "contract_id", "route_id", "stage", "evidence_role", "locked_test_policy", "asset_set_id", "asset_root", "source_chain", "checkpoint_and_state_assets", "operator_assets", "native_shape_pairs", "first32_names", "record_order", "records")
RECORD_ROLES = ("hazy_rgb_native", "frozen_base_rgb_native", "old_0p125_rgb_native", "old_0p25_rgb_native", "current_delta_u_native", "target_delta_u_native", "measured_active_support_native")
FORWARD_ROLES = RECORD_ROLES[:5]
SHARED_ROLES = RECORD_ROLES[:2]
PER_OPERATOR_ROLES = RECORD_ROLES[2:]
DESCRIPTOR_FIELDS = ("path", "sha256", "bytes", "encoding", "dtype", "shape", "producer_role")

PINNED_SOURCE_CHAIN = {
    "a1x_anchor": {"branch": "github/codex/haze4k-official-arch-anchor", "commit": SOURCE_COMMIT},
    "a1r_smoke": {"route_id": "haze4k_v5_chd_rm_v4a_a1r_representation_sufficiency_20260714", "route_commit": "d79fdff7f2bd923d84969ac242f214d6736b2d7f", "run_id": "v4a_a1r_s0_smoke_r2", "source_manifest_runtime_path": "/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v4a_a1r_representation_sufficiency_20260714/v4a_a1r_s0_smoke_r2/v4a_a1r_source_manifest.json", "source_manifest_github_ref": "github:fe08ba7c0fde4d6086083490430246ea39fbf766:experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1r_representation_sufficiency_20260714/v4a_a1r_smoke_source_manifest.json", "source_manifest_git_blob_sha1": "c477612a456df86f465ce305436022305fb64118", "source_manifest_sha256": "900dcb1dfb4e2e25ff283e7753b8288b1df0a23cbdd66547131b8a1045dfe474"},
    "a1c_exact_half": {"route_id": "haze4k_v5_chd_rm_v4a_a1c_safe_action_interface_ceiling_20260715", "route_commit": "b62dd75ea30123e2fca94ecb1d4a57c02dbc8f64", "run_id": "v4a-a1c-s0-r1", "source_manifest_runtime_path": "/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v4a_a1c_safe_action_interface_ceiling_20260715/v4a-a1c-s0-r1/v4a_a1c_source_manifest.json", "source_manifest_github_ref": "github:fe08ba7c0fde4d6086083490430246ea39fbf766:experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1c_safe_action_interface_ceiling_20260715/v4a_a1c_s0_source_manifest.json", "source_manifest_git_blob_sha1": "f39c7fb6f696420223f5e273f418beeaa5d9b711", "source_manifest_sha256": "6cad07413599d5ad163d3cbe9cf39f09efa2f7de249ffeca0db92b0ad5258284"},
    "parent_source_checkouts": (("A1F_PARENT", "d4f8d0936869c822ae19b5c21172efc2eb973dd8"), ("V3Z_SOURCE", "3caddcc5265732e5be77e3404119a28cb28c11e6"), ("V3S_SOURCE", "2860f580bb25cc75ec9ade56378af6d77f5c8d8b"), ("V3P_SOURCE", "555fd008e29f02128564f2fad41d0095ee44f5ea")),
    "required_parent_evidence_hashes": {"a1f_r3_review_sha256": "a8b9064308710ac5fc890b9de0158c1faddb4d51f7d298d4991e9ddfb3616e1d", "a0r_trace_manifest_sha256": "370f3f46b949391a8200c50b893561d78608adfef3c59f9ba2f76dd84a155780"},
}
PINNED_CHECKPOINT_ASSETS = {
    "official_a0_checkpoint": {"path": "/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl", "bytes": 34797069, "sha256": "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"},
    "control_checkpoint": {"path": "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl", "bytes": 34798010, "sha256": "08207119a5cf9e5c439dd2cb81b99029ade1861f2739d31e75f2f9f78d57c0f2"},
    "a0r_final_model_state": {"path": "/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714/a0r_r2/r1/trace/states/epoch16_update512_final.pt", "sha256": "ed0832f220996af3fd8e617b7d04d643dc6ca052a3603adee99d59e78fd1e125"},
    "density_artifact": {"path": "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt", "bytes": 59658, "sha256": "1ffce13dccb41d96a47c2b5275f87bf2fdb73c226a190cfa240e5c71c1ec326f"},
    "d7c_artifact": {"path": "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt", "bytes": 288048, "sha256": "09f449232024395cf64db15a2a0efa0f12d3e0e049e1da3d67229a3dc5729361"},
    "fresh_split_manifest": {"path": "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711/fresh_route_confirm_split_manifest.json", "bytes": 66646, "sha256": "c8c00fefc965ded3389b6311fc67ea521e1f3174f27793688544abe09dc420e7"},
    "v3j_action_bounds": {"path": "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711/bounded_action_space_bounds.json", "bytes": 374, "sha256": "485ea12ff14c33b87105a50b6d118a9937c7e7f1b113062fe03d91eef3c9cc21"},
    "reference_oof_rows": {"path": "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711/cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv", "bytes": 2671809, "sha256": "b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1"},
}
PINNED_OPERATOR_ASSETS = {
    "manifest": {"path": "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711/v3l_a0_canonical_operator_artifact_manifest.json", "bytes": 1021, "git_blob_sha1": "43e9400c5a0bb5f4eb6195a610e11e80872c0fc8", "sha256": "1d2ffa499128ad08a272d67c5439583900afe8ef87fb3256193ad5fe21c3af84"},
    "D_ref": {"seed": 3407, "path": "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711/cloud_only_artifacts/v3l_a0_D_ref_context_seed3407.pt", "bytes": 335232, "sha256": "3c5ed807cc0cb25170720fe2b8f34cbe81b7bbc3857f7a3f1423f73d77cd6692"},
    "D_rep": {"seed": 3408, "path": "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711/cloud_only_artifacts/v3l_a0_D_rep_context_seed3408.pt", "bytes": 335104, "sha256": "3843471c6f4c130495ee6f6d62193c4235d9c89061621288ecb8d7455944ae62"},
}


class PreflightError(RuntimeError):
    pass


class GateFailure(RuntimeError):
    def __init__(self, payload):
        super().__init__("frozen structural S0 gate failed")
        self.payload = payload


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PreflightError("duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicate_keys)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def append_status(args, phase, progress, **details):
    payload = dict(identity(args), phase=phase, progress=progress, timestamp=time.time(), **details)
    path = Path(args.status_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def heartbeat(args, phase, progress):
    write_json(args.heartbeat_json, dict(identity(args), phase=phase, progress=progress, timestamp=time.time()))


def identity(args):
    return {"route_id": ROUTE_ID, "route_commit": args.route_commit, "run_id": args.run_id, "contract_id": CONTRACT_ID, "runtime_asset_manifest_json": args.runtime_asset_manifest_json, "runtime_asset_manifest_sha256": args.runtime_asset_manifest_sha256}


def exact_names(names):
    if tuple(names) != S0_FIRST32_NAMES or len(set(names)) != 32:
        raise PreflightError("exact committed first32 order required")


def canonical_regular_file(path, root=None):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise PreflightError("absolute canonical regular nonsymlink file required")
    if root is not None:
        try:
            path.relative_to(root)
        except ValueError as error:
            raise PreflightError("asset path escapes exact asset root") from error
    return path


def validate_file_identity(descriptor, root=None):
    path = canonical_regular_file(descriptor["path"], root)
    if "bytes" in descriptor and (not isinstance(descriptor["bytes"], int) or descriptor["bytes"] <= 0 or path.stat().st_size != descriptor["bytes"]):
        raise PreflightError("asset byte count mismatch")
    if sha256(path) != descriptor["sha256"]:
        raise PreflightError("asset SHA-256 mismatch")
    return path


def parser():
    result = argparse.ArgumentParser()
    for name in ("--stage", "--authorization-json", "--route-commit", "--run-id", "--run-root", "--runtime-asset-manifest-json", "--runtime-asset-manifest-sha256", "--status-json", "--heartbeat-json", "--learned-state-manifest-json", "--closeout-json"):
        result.add_argument(name, required=True)
    return result


def authorization_guard(args):
    authorization_path = canonical_regular_file(args.authorization_json)
    authorization = read_json(authorization_path)
    expected = {"route_id": "haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715", "state": "PLANNED", "decision": "V4A_A1X_S0_AUTHORIZED_INITIAL_ONLY", "authorizes": "A1X_S0_ONLY"}
    required = {"source_commit": SOURCE_COMMIT, "route_commit": args.route_commit, "route_card_sha256": ROUTE_CARD_SHA256, "locked_test_policy": "blocked", "contract_id": CONTRACT_ID, "runtime_asset_manifest_json": args.runtime_asset_manifest_json, "runtime_asset_manifest_sha256": args.runtime_asset_manifest_sha256}
    if any(authorization.get(key) != value for key, value in {**expected, **required}.items()):
        raise PreflightError("INITIAL_AUTHORIZATION_TUPLE_OR_DOMINANCE_INVALID")
    manifest_path = Path(args.runtime_asset_manifest_json)
    manifest_hash = args.runtime_asset_manifest_sha256
    if not manifest_path.is_absolute() or len(manifest_hash) != 64 or any(character not in "0123456789abcdef" for character in manifest_hash):
        raise PreflightError("RUNTIME_ASSET_MANIFEST_IDENTITY_UNGUARDED")
    return authorization


def validate_source_chain(source_chain):
    if set(source_chain) != set(PINNED_SOURCE_CHAIN):
        raise PreflightError("source-chain key mismatch")
    for key in ("a1x_anchor", "a1r_smoke", "a1c_exact_half", "required_parent_evidence_hashes"):
        if source_chain.get(key) != PINNED_SOURCE_CHAIN[key]:
            raise PreflightError("pinned source-chain identity mismatch")
    parents = source_chain["parent_source_checkouts"]
    if len(parents) != len(PINNED_SOURCE_CHAIN["parent_source_checkouts"]):
        raise PreflightError("parent checkout count mismatch")
    for record, (role, commit) in zip(parents, PINNED_SOURCE_CHAIN["parent_source_checkouts"]):
        if set(record) != {"role", "commit", "root"} or (record["role"], record["commit"]) != (role, commit):
            raise PreflightError("parent checkout identity mismatch")
        root = Path(record["root"])
        if not root.is_absolute() or root.is_symlink() or not root.is_dir() or root.resolve() != root:
            raise PreflightError("parent checkout root invalid")
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True, text=True, capture_output=True).stdout
        if head != commit or dirty:
            raise PreflightError("parent checkout commit or cleanliness mismatch")
    a1r_path = validate_file_identity({"path": source_chain["a1r_smoke"]["source_manifest_runtime_path"], "sha256": source_chain["a1r_smoke"]["source_manifest_sha256"]})
    a1c_path = validate_file_identity({"path": source_chain["a1c_exact_half"]["source_manifest_runtime_path"], "sha256": source_chain["a1c_exact_half"]["source_manifest_sha256"]})
    a1r, a1c = read_json(a1r_path), read_json(a1c_path)
    if tuple(a1r.get("fresh_names", ())[:32]) != S0_FIRST32_NAMES or a1r.get("a1f_root_commit") != "d4f8d0936869c822ae19b5c21172efc2eb973dd8" or a1r.get("final_state_sha256") != PINNED_CHECKPOINT_ASSETS["a0r_final_model_state"]["sha256"]:
        raise PreflightError("A1R source manifest mismatch")
    if (a1c.get("route_id"), a1c.get("route_commit"), a1c.get("run_id"), a1c.get("stage")) != (source_chain["a1c_exact_half"]["route_id"], source_chain["a1c_exact_half"]["route_commit"], source_chain["a1c_exact_half"]["run_id"], "s0"):
        raise PreflightError("A1C source manifest mismatch")
    if a1c.get("a1r_transport_pairs") != {"400x400": "208x208", "480x640": "240x320"}:
        raise PreflightError("A1C exact-half transport mismatch")


def validate_static_assets(manifest):
    if manifest["checkpoint_and_state_assets"] != PINNED_CHECKPOINT_ASSETS:
        raise PreflightError("checkpoint/state asset pins mismatch")
    operator_assets = manifest["operator_assets"]
    if set(operator_assets) != {"manifest", "ordered_operators", "D_ref", "D_rep"} or tuple(operator_assets["ordered_operators"]) != OPERATORS:
        raise PreflightError("operator asset key/order mismatch")
    for key in ("manifest", "D_ref", "D_rep"):
        if operator_assets[key] != PINNED_OPERATOR_ASSETS[key]:
            raise PreflightError("operator asset pin mismatch")
    if operator_assets["D_ref"]["path"] == operator_assets["D_rep"]["path"] or operator_assets["D_ref"]["seed"] == operator_assets["D_rep"]["seed"]:
        raise PreflightError("RUNTIME_ASSET_OPERATOR_PROVENANCE_INVALID")
    for descriptor in PINNED_CHECKPOINT_ASSETS.values():
        validate_file_identity(descriptor)
    for key in ("manifest", "D_ref", "D_rep"):
        validate_file_identity(PINNED_OPERATOR_ASSETS[key])


def expected_record_order(records):
    shape_order = {native: index for index, (native, _) in enumerate(SHAPE_PAIRS)}
    return sorted(records, key=lambda record: (shape_order[tuple(record["native_image_hw"])], record["name"], OPERATORS.index(record["operator"])))


def validate_descriptor(descriptor, role, record, asset_root):
    if set(descriptor) != set(DESCRIPTOR_FIELDS) or descriptor["encoding"] != "npy_allow_pickle_false":
        raise PreflightError("RUNTIME_ASSET_PATH_CONTENT_VALIDATION_INVALID")
    native = tuple(record["native_image_hw"])
    channels = 1 if role == "measured_active_support_native" else 3
    dtype = "bool" if channels == 1 else "float32"
    if descriptor["dtype"] != dtype or descriptor["shape"] != [1, channels, *native]:
        raise PreflightError("RUNTIME_ASSET_PATH_CONTENT_VALIDATION_INVALID")
    expected_producer = role if role in SHARED_ROLES else record["operator"] + ":" + role
    if descriptor["producer_role"] != expected_producer:
        raise PreflightError("RUNTIME_ASSET_PATH_CONTENT_VALIDATION_INVALID")
    validate_file_identity(descriptor, asset_root)


def validate_records(manifest):
    records = manifest["records"]
    expected_keys = {(name, operator) for name in S0_FIRST32_NAMES for operator in OPERATORS}
    if len(records) != 64 or {(record.get("name"), record.get("operator")) for record in records} != expected_keys or records != expected_record_order(records):
        raise PreflightError("RUNTIME_ASSET_FIRST32_OR_ENUMERATION_INVALID")
    record_fields = {"name", "operator", "native_image_hw", "exact_a1r_context_hw", *RECORD_ROLES}
    asset_root = Path(manifest["asset_root"])
    if not asset_root.is_absolute() or asset_root.is_symlink() or not asset_root.is_dir() or asset_root.resolve() != asset_root:
        raise PreflightError("runtime asset root invalid")
    observed_pairs = set()
    by_key = {}
    for record in records:
        if set(record) != record_fields:
            raise PreflightError("record field mismatch")
        pair = (tuple(record["native_image_hw"]), tuple(record["exact_a1r_context_hw"]))
        if pair not in SHAPE_PAIRS:
            raise PreflightError("RUNTIME_ASSET_NATIVE_SHAPE_TRANSPORT_INVALID")
        observed_pairs.add(pair)
        for role in RECORD_ROLES:
            validate_descriptor(record[role], role, record, asset_root)
        by_key[(record["name"], record["operator"])] = record
    if observed_pairs != set(SHAPE_PAIRS):
        raise PreflightError("RUNTIME_ASSET_NATIVE_SHAPE_TRANSPORT_INVALID")
    for name in S0_FIRST32_NAMES:
        reference, representation = by_key[(name, "D_ref")], by_key[(name, "D_rep")]
        if tuple(reference["native_image_hw"]) != tuple(representation["native_image_hw"]):
            raise PreflightError("operator shape mismatch")
        for role in SHARED_ROLES:
            if reference[role] != representation[role]:
                raise PreflightError("per-name shared descriptor mismatch")
        for role in PER_OPERATOR_ROLES:
            if reference[role]["path"] == representation[role]["path"] or reference[role]["producer_role"] == representation[role]["producer_role"]:
                raise PreflightError("RUNTIME_ASSET_OPERATOR_PROVENANCE_INVALID")
    return records


def validate_manifest(args, authorization):
    manifest_path = canonical_regular_file(args.runtime_asset_manifest_json)
    if str(manifest_path) != authorization["runtime_asset_manifest_json"] or sha256(manifest_path) != args.runtime_asset_manifest_sha256 or args.runtime_asset_manifest_sha256 != authorization["runtime_asset_manifest_sha256"]:
        raise PreflightError("RUNTIME_ASSET_MANIFEST_IDENTITY_UNGUARDED")
    manifest = read_json(manifest_path)
    if tuple(manifest) != MANIFEST_FIELDS:
        raise PreflightError("runtime manifest section order/key mismatch")
    expected_identity = (1, CONTRACT_ID, ROUTE_ID, "s0", "engineering_debug", "blocked")
    observed_identity = tuple(manifest[key] for key in ("schema_version", "contract_id", "route_id", "stage", "evidence_role", "locked_test_policy"))
    if observed_identity != expected_identity or not isinstance(manifest["asset_set_id"], str) or not manifest["asset_set_id"]:
        raise PreflightError("runtime manifest identity mismatch")
    exact_names(manifest["first32_names"])
    if manifest["native_shape_pairs"] != [{"native_image_hw": list(native), "exact_a1r_context_hw": list(context)} for native, context in SHAPE_PAIRS] or manifest["record_order"] != RECORD_ORDER:
        raise PreflightError("RUNTIME_ASSET_NATIVE_SHAPE_TRANSPORT_INVALID")
    validate_source_chain(manifest["source_chain"])
    validate_static_assets(manifest)
    return manifest, validate_records(manifest)


def render_d_ref(frozen_artifact, renderer):
    return renderer.render_reference(frozen_artifact)


def render_d_rep(frozen_artifact, renderer):
    return renderer.render_representation(frozen_artifact)


def operator_render(operator, frozen_artifact, renderer):
    if operator == "D_ref":
        return render_d_ref(frozen_artifact, renderer)
    if operator == "D_rep":
        return render_d_rep(frozen_artifact, renderer)
    raise RuntimeError("unknown frozen operator")


class AssetBackedRenderer:
    def __init__(self, operator_assets):
        self.operator_assets = operator_assets

    def render_reference(self, record):
        if record["operator"] != "D_ref" or self.operator_assets["D_ref"] != PINNED_OPERATOR_ASSETS["D_ref"]:
            raise RuntimeError("D_ref asset provenance mismatch")
        return record

    def render_representation(self, record):
        if record["operator"] != "D_rep" or self.operator_assets["D_rep"] != PINNED_OPERATOR_ASSETS["D_rep"]:
            raise RuntimeError("D_rep asset provenance mismatch")
        return record


class AssetBackedSupport:
    def for_operator_and_shape(self, operator, native_shape, operator_batch=None):
        if operator_batch is None or operator_batch["operator"] != operator or operator_batch["native_shape"] != native_shape:
            raise RuntimeError("measured support lookup provenance mismatch")
        return operator_batch["measured_active_support_exact"]


def measured_active_support(operator_batch, frozen_support_artifact, torch):
    support = frozen_support_artifact.for_operator_and_shape(operator_batch["operator"], operator_batch["native_shape"], operator_batch)
    if not torch.isfinite(support).all() or support.dtype != torch.bool or int(support.sum().item()) == 0:
        raise RuntimeError("frozen measured support invalid")
    return support, int(support.sum().item())


def normalized_active_support_endpoint_mse(predicted, target, support, active_count, torch):
    if active_count <= 0 or not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise RuntimeError("invalid active support loss")
    return ((predicted - target).square() * support.to(predicted.dtype)).sum() / active_count


def state_dict_equal(left, right, torch):
    return tuple(left) == tuple(right) and all(torch.equal(left[key], right[key]) for key in left)


def independent_cell_bundle(official_model, seed, torch):
    random.seed(seed)
    torch.manual_seed(seed)
    from Dehazing.ITS.models.A1XAccess import A1X_ACCESS_Head
    head = A1X_ACCESS_Head()
    head.to(next(official_model.parameters()).device)
    head_state = copy.deepcopy(head.state_dict())
    model = copy.deepcopy(official_model)
    model.load_state_dict(copy.deepcopy(official_model.state_dict()), strict=True)
    optimizer = torch.optim.AdamW(tuple(head.parameters()), lr=0.0005, weight_decay=0.00001)
    return {"model": model, "head": head, "optimizer": optimizer, "initial_head_state": head_state, "updates": 0}


def independent_cells(official_model, torch):
    true_cell = independent_cell_bundle(official_model, 3407, torch)
    shuffled_cell = independent_cell_bundle(official_model, 3407, torch)
    if true_cell["model"] is shuffled_cell["model"] or true_cell["head"] is shuffled_cell["head"] or true_cell["optimizer"] is shuffled_cell["optimizer"]:
        raise RuntimeError("true/shuffled learned state must be disjoint")
    if not state_dict_equal(true_cell["initial_head_state"], shuffled_cell["initial_head_state"], torch):
        raise RuntimeError("identical pre-update initialization required")
    return {"A1X_ACCESS_TRUE": true_cell, "A1X_ACCESS_SHUFFLED": shuffled_cell}


def exact_half(value, context_shape, torch, support=False):
    transported = torch.nn.functional.interpolate(value.to(torch.float32), size=context_shape, mode="bilinear", align_corners=False, antialias=False)
    return transported > 0.0 if support else transported


def channelwise_clamp(value, bound, torch):
    bound_tensor = torch.tensor(bound, device=value.device, dtype=value.dtype).view(1, 3, 1, 1)
    return torch.maximum(torch.minimum(value, bound_tensor), -bound_tensor)


def forward_endpoint(model, head, operator_batch, torch):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("official model must remain frozen")
    head_input = torch.cat(tuple(operator_batch[role + "_exact"] for role in FORWARD_ROLES), dim=1)
    if head_input.shape[1] != 15:
        raise RuntimeError("five-input forward must have 15 channels")
    current = operator_batch["current_delta_u_native_exact"]
    support = operator_batch["measured_active_support_exact"].to(current.dtype)
    correction = head(head_input)
    return channelwise_clamp(current + support * torch.tensor(DELTA_BOUND, device=current.device, dtype=current.dtype).view(1, 3, 1, 1) * correction, DELTA_BOUND, torch)


def verify_zero_noop(cell, operator_batches, forward, torch):
    discrepancies = [float((forward(cell["model"], cell["head"], batch, torch) - batch["current_delta_u_native_exact"]).abs().max()) for batch in operator_batches]
    maximum = max(discrepancies)
    if maximum != 0.0:
        raise RuntimeError("zero final projection no-op must be exactly zero before update")
    return maximum


def true_target(batch):
    return batch["target_delta_u_native_exact"]


def shuffled_target(batch):
    return batch["shuffled_target_delta_u_native_exact"]


def update_cell(cell, batches, forward, target_selector, support_artifact, torch):
    gradient_max = 0.0
    for _ in range(2):
        cell["optimizer"].zero_grad(set_to_none=True)
        for batch in batches:
            target = target_selector(batch)
            support, active_count = measured_active_support(batch, support_artifact, torch)
            predicted = forward(cell["model"], cell["head"], batch, torch)
            loss = normalized_active_support_endpoint_mse(predicted, target, support, active_count, torch) / len(batches)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite active-support loss")
            loss.backward()
        gradients = [float(parameter.grad.abs().max()) for parameter in cell["head"].parameters() if parameter.grad is not None]
        if not gradients or not all(math.isfinite(value) for value in gradients):
            raise RuntimeError("finite nonzero head gradients required")
        gradient_max = max(gradient_max, max(gradients))
        torch.nn.utils.clip_grad_norm_(cell["head"].parameters(), 0.1)
        cell["optimizer"].step()
        cell["updates"] += 1
    if cell["updates"] != 2 or gradient_max <= 0.0:
        raise RuntimeError("exactly two finite updates required")
    return gradient_max


def cuda_profile_pair(baseline, augmented, shape, torch):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE_INCONCLUSIVE")
    device = torch.device("cuda")
    probe = torch.zeros((1, 15, *shape), device=device, dtype=torch.float32)

    def measured_macs(fn):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], with_flops=True) as profiler:
            fn(probe)
        torch.cuda.synchronize(device)
        flops = sum(event.flops for event in profiler.key_averages() if event.flops)
        macs = flops / 2.0
        if not macs > 0.0:
            raise RuntimeError("nonpositive measured CUDA MACs")
        return macs

    def measure(fn):
        for _ in range(5):
            fn(probe)
        samples = []
        for _ in range(11):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            fn(probe)
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - started, torch.cuda.max_memory_allocated(device)))
        samples.sort()
        return {"macs": measured_macs(fn), "median_seconds": samples[len(samples) // 2][0], "peak_memory_bytes": max(item[1] for item in samples)}

    return measure(baseline), measure(augmented)


def cuda_cost_contract(baseline, augmented, native_shapes, torch):
    if len(native_shapes) != 2:
        raise RuntimeError("both frozen native shapes required")
    measurements = {str(shape): cuda_profile_pair(baseline, augmented, shape, torch) for shape in native_shapes}
    overheads = {}
    for shape, (baseline_cost, augmented_cost) in measurements.items():
        if not all(value > 0.0 and value < float("inf") for value in baseline_cost.values()):
            raise RuntimeError("invalid baseline CUDA cost denominator")
        overheads[shape] = {metric: 100.0 * (augmented_cost[metric] - baseline_cost[metric]) / baseline_cost[metric] for metric in ("macs", "median_seconds", "peak_memory_bytes")}
    results = {"mac_overhead": all(cost["macs"] <= 10.0 for cost in overheads.values()), "matched_median_cuda_latency_overhead": all(cost["median_seconds"] <= 15.0 for cost in overheads.values()), "cuda_peak_memory_overhead": all(cost["peak_memory_bytes"] <= 15.0 for cost in overheads.values())}
    payload = {"measurements": measurements, "overhead_percent_by_shape": overheads, "results": results}
    if not all(results.values()):
        raise GateFailure(payload)
    return payload


def structural_s0_gate(head, baseline, augmented, native_shapes, torch):
    added_parameters = sum(parameter.numel() for parameter in head.parameters())
    parameter_limit = added_parameters <= 300000
    try:
        cuda_cost = cuda_cost_contract(baseline, augmented, native_shapes, torch)
    except GateFailure as error:
        cuda_cost = error.payload
    cuda_results = cuda_cost["results"]
    payload = {"added_parameters": added_parameters, "cuda_cost": cuda_cost, "results": {"added_parameter_limit": parameter_limit, **cuda_results}}
    if not parameter_limit or not all(cuda_results.values()):
        raise GateFailure(payload)
    return payload


def retain_learned_states(cells, run_root, base):
    records = []
    for name, cell in cells.items():
        path = Path(run_root) / (name.lower() + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save({"model": cell["model"].state_dict(), "head": cell["head"].state_dict()}, path)
        records.append(dict(base, cell=name, absolute_state_path=str(path.resolve()), relative_state_path=str(path.relative_to(run_root)), sha256=sha256(path), seed=3407, update_count=cell["updates"], no_resume=True))
    if len({record["sha256"] for record in records}) != 2 or len({record["absolute_state_path"] for record in records}) != 2:
        raise RuntimeError("distinct written learned states required")
    return records


def terminal_closeout(kind, base, closeout_path):
    state, decision, authorizes = TERMINALS[kind]
    payload = dict(base, state=state, decision=decision, authorizes=authorizes)
    write_json(closeout_path, payload)
    return payload


def load_validated_array(descriptor, record, role, np, torch, device):
    value = np.load(descriptor["path"], allow_pickle=False)
    native = tuple(record["native_image_hw"])
    expected_shape = (1, 1 if role == "measured_active_support_native" else 3, *native)
    expected_dtype = np.bool_ if role == "measured_active_support_native" else np.float32
    if value.shape != expected_shape or value.dtype != expected_dtype or (role != "measured_active_support_native" and not np.isfinite(value).all()) or (role == "measured_active_support_native" and not value.any()):
        raise PreflightError("RUNTIME_ASSET_PATH_CONTENT_VALIDATION_INVALID")
    return torch.from_numpy(value).to(device)


def build_operator_batches(manifest, records, np, torch, device):
    renderer = AssetBackedRenderer(manifest["operator_assets"])
    batches = []
    for raw_record in records:
        record = operator_render(raw_record["operator"], raw_record, renderer)
        context_shape = tuple(record["exact_a1r_context_hw"])
        batch = {"name": record["name"], "operator": record["operator"], "native_shape": tuple(record["native_image_hw"]), "context_shape": context_shape}
        for role in RECORD_ROLES:
            native = load_validated_array(record[role], record, role, np, torch, device)
            batch[role + "_exact"] = exact_half(native, context_shape, torch, support=role == "measured_active_support_native")
        batches.append(batch)
    by_block = {}
    for batch in batches:
        by_block.setdefault((batch["operator"], batch["native_shape"]), []).append(batch)
    for block in by_block.values():
        block.sort(key=lambda item: item["name"])
        if len(block) < 2:
            raise PreflightError("target shuffle block requires two names")
        for index, batch in enumerate(block):
            peer = block[(index + 1) % len(block)]
            if peer["name"] == batch["name"]:
                raise PreflightError("target shuffle self-pair forbidden")
            batch["shuffled_target_delta_u_native_exact"] = peer["target_delta_u_native_exact"]
    return batches


def load_frozen_official_model(torch, device):
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from Dehazing.ITS.models.ConvIR import build_net
    model = build_net("base", "Haze4K").to(device)
    state = torch.load(PINNED_CHECKPOINT_ASSETS["official_a0_checkpoint"]["path"], map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def matched_cost_paths(model, head):
    def baseline(probe):
        return model(probe[:, :3])[-1]

    def augmented(probe):
        return model(probe[:, :3])[-1] + head(probe)

    return baseline, augmented


def runtime_lifecycle(args, manifest, records):
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE_INCONCLUSIVE")
    device = torch.device("cuda")
    started = time.monotonic()
    batches = build_operator_batches(manifest, records, np, torch, device)
    append_status(args, "assets_verified", 1.0, record_count=len(records))
    heartbeat(args, "assets_verified", 1.0)
    official_model = load_frozen_official_model(torch, device)
    cells = independent_cells(official_model, torch)
    append_status(args, "cells_initialized", 0.0, cells=list(cells))
    heartbeat(args, "cells_initialized", 0.0)
    zero_noop = {name: verify_zero_noop(cell, batches, forward_endpoint, torch) for name, cell in cells.items()}
    support_artifact = AssetBackedSupport()
    gradients = {}
    gradients["A1X_ACCESS_TRUE"] = update_cell(cells["A1X_ACCESS_TRUE"], batches, forward_endpoint, true_target, support_artifact, torch)
    append_status(args, "cell_progress", 0.5, cell="A1X_ACCESS_TRUE", updates=2)
    heartbeat(args, "cell_progress", 0.5)
    gradients["A1X_ACCESS_SHUFFLED"] = update_cell(cells["A1X_ACCESS_SHUFFLED"], batches, forward_endpoint, shuffled_target, support_artifact, torch)
    append_status(args, "cell_progress", 1.0, cell="A1X_ACCESS_SHUFFLED", updates=2)
    heartbeat(args, "cell_progress", 1.0)
    baseline, augmented = matched_cost_paths(cells["A1X_ACCESS_TRUE"]["model"], cells["A1X_ACCESS_TRUE"]["head"])
    terminal_kind = "PASS"
    try:
        gate = structural_s0_gate(cells["A1X_ACCESS_TRUE"]["head"], baseline, augmented, [native for native, _ in SHAPE_PAIRS], torch)
    except GateFailure as error:
        gate = error.payload
        terminal_kind = "FAIL"
    append_status(args, "cost_gate", 1.0, results=gate["results"])
    heartbeat(args, "cost_gate", 1.0)
    base = dict(identity(args), source_commit=SOURCE_COMMIT, seed=3407, update_count=2, no_resume=True)
    states = retain_learned_states(cells, args.run_root, base)
    timings = {"setup": 0.0, "cache": 0.0, "train": time.monotonic() - started, "eval_or_audit": 0.0, "summary": 0.0, "closeout": 0.0}
    state_manifest = dict(base, schema_version=1, first32_source_manifest_sha256=PINNED_SOURCE_CHAIN["a1r_smoke"]["source_manifest_sha256"], first32_names=list(S0_FIRST32_NAMES), operators=list(OPERATORS), native_shape_pairs=[{"native_image_hw": list(native), "exact_a1r_context_hw": list(context)} for native, context in SHAPE_PAIRS], cells=states, timings=timings)
    write_json(args.learned_state_manifest_json, state_manifest)
    state_manifest_sha256 = sha256(args.learned_state_manifest_json)
    append_status(args, "states_written", 1.0, state_manifest_sha256=state_manifest_sha256)
    heartbeat(args, "states_written", 1.0)
    closeout_base = dict(base, authorization_verified=True, runtime_asset_manifest_verified=True, source_chain_verified=True, checkpoint_and_state_assets_verified=True, operator_assets_verified=True, record_count=64, first32_count=32, operators=list(OPERATORS), native_shape_pairs=[list(native) + list(context) for native, context in SHAPE_PAIRS], zero_noop=zero_noop, gradient_max=gradients, structural_gate=gate, learned_state_manifest_sha256=state_manifest_sha256, a1x_data_accessed=False, confirmation_data_accessed=False, canary_touched=False, locked_test_touched=False)
    terminal_closeout(terminal_kind, closeout_base, args.closeout_json)
    append_status(args, "terminal", 1.0, terminal=terminal_kind)
    heartbeat(args, "terminal", 1.0)


def write_preflight_failure(args, reason):
    payload = dict(identity(args), state="FAILED_PREFLIGHT", decision="V4A_A1X_S0_RUNTIME_ASSET_PREFLIGHT_FAILED_STOP", authorizes="NONE", failure_class="RESOURCE_PREFLIGHT", reason=str(reason), runtime_started=False, scientific_gate_evaluated=False, a1x_data_accessed=False, confirmation_data_accessed=False, canary_touched=False, locked_test_touched=False)
    write_json(args.closeout_json, payload)


def write_runtime_inconclusive(args, reason):
    base = dict(identity(args), source_commit=SOURCE_COMMIT, failure_class="RUNTIME_EVALUATION", reason=str(reason), runtime_started=True, scientific_gate_evaluated=False, a1x_data_accessed=False, confirmation_data_accessed=False, canary_touched=False, locked_test_touched=False)
    terminal_closeout("INCONCLUSIVE", base, args.closeout_json)
    append_status(args, "terminal", 1.0, terminal="INCONCLUSIVE")
    heartbeat(args, "terminal", 1.0)


def main():
    args = parser().parse_args()
    if args.stage != "s0":
        write_preflight_failure(args, "formal mode blocked")
        raise PreflightError("formal mode blocked")
    try:
        authorization = authorization_guard(args)
        append_status(args, "authorized", 1.0)
        heartbeat(args, "authorized", 1.0)
        manifest, records = validate_manifest(args, authorization)
    except (PreflightError, OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
        write_preflight_failure(args, error)
        raise
    try:
        runtime_lifecycle(args, manifest, records)
    except RuntimeError as error:
        write_runtime_inconclusive(args, error)


if __name__ == "__main__":
    main()
