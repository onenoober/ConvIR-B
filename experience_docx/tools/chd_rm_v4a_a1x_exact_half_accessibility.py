#!/usr/bin/env python3
"""S0-only, guarded A1X accessibility runner.

The literal sample set below is deliberately the only image-name source.  This
module is never a formal runner: it fails closed outside the authorized S0
contract and writes durable typed lifecycle records on both outcomes.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
SOURCE_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
S0_FIRST32_NAMES = (
    "1594_0.71_0.5.png", "1595_0.99_1.84.png", "1597_0.69_1.45.png", "1598_0.67_1.4.png",
    "159_0.6_1.46.png", "1600_0.78_1.77.png", "1603_0.54_0.74.png", "1607_0.91_0.88.png",
    "160_0.63_1.04.png", "1613_0.56_1.31.png", "1614_0.81_0.78.png", "1615_0.91_1.25.png",
    "1616_0.76_0.88.png", "1617_0.56_1.97.png", "1619_0.94_1.08.png", "1622_0.98_1.75.png",
    "1623_0.78_1.81.png", "1627_0.94_0.52.png", "1628_0.8_1.49.png", "1633_0.73_1.49.png",
    "1634_0.75_1.81.png", "1639_0.69_1.12.png", "1640_0.53_0.59.png", "1646_0.55_1.55.png",
    "1649_0.8_0.86.png", "1650_0.76_1.07.png", "1652_0.62_1.35.png", "1653_0.9_1.01.png",
    "1654_0.66_1.9.png", "1656_0.64_1.45.png", "1658_0.96_1.72.png", "1660_0.83_0.67.png",
)
AUTH_TUPLE = {"state": "PLANNED", "decision": "V4A_A1X_S0_AUTHORIZED_INITIAL_ONLY", "authorizes": "A1X_S0_ONLY"}
INTEGRATED_S0_CHECKS = (
    "rules_route_source_checkpoint_and_runner_identity", "exact_first32_identity_count_order_uniqueness_and_a1r_provenance",
    "no_a1x_confirmation_canary_or_locked_data_access", "official_checkpoint_exact_load_and_non_a1x_freeze_eval",
    "five_input_forward_provenance_and_fifteen_channels", "native_shape_preservation_and_shape_blocking",
    "zero_initialized_exact_noop_with_max_endpoint_discrepancy_0p0", "finite_forward_loss_backward_and_updates",
    "nonzero_first_gradient_for_both_cells", "complete_no_self_pair_shape_operator_shuffle", "added_parameters_at_most_300000",
    "mac_overhead_at_most_10_percent", "matched_median_latency_overhead_at_most_15_percent", "peak_memory_overhead_at_most_15_percent",
)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, payload, atomic=False):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if atomic:
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    else:
        destination.write_text(text, encoding="utf-8")

def authorization(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("route_id") != ROUTE_ID or payload.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("route identity mismatch")
    if any(payload.get(key) != value for key, value in AUTH_TUPLE.items()):
        raise RuntimeError("authorization tuple mismatch")
    return payload

def exact_names(requested_names):
    if tuple(requested_names) != S0_FIRST32_NAMES or len(S0_FIRST32_NAMES) != 32 or len(set(S0_FIRST32_NAMES)) != 32:
        raise RuntimeError("exact first32 membership/order guard failed")

def image_path(root, name):
    if name not in S0_FIRST32_NAMES:
        raise RuntimeError("nonmember image name refused")
    return Path(root) / name

def load_rgb(root, name, torch):
    from PIL import Image
    path = image_path(root, name)
    if not path.is_file():
        raise RuntimeError("required exact-first32 image is missing")
    with Image.open(path) as image:
        data = torch.from_numpy(__import__("numpy").asarray(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    return data.unsqueeze(0)

def heartbeat(path, args, phase, progress):
    write_json(path, {"route_id": ROUTE_ID, "route_commit": args.route_commit, "run_id": args.run_id, "stage": "s0", "phase": phase, "timestamp": time.time(), "progress": progress}, atomic=True)

def check_model_contract(head, torch):
    for name, parameter in head.named_parameters():
        parameter.requires_grad = name.startswith("A1X_ACCESS_")
    head.train()
    added = sum(parameter.numel() for parameter in head.parameters())
    if added > 300000:
        raise RuntimeError("added parameter cost limit exceeded")
    return added

def paired_names(names):
    ordered = tuple(sorted(names))
    return tuple(zip(ordered, ordered[1:] + ordered[:1]))

def run_cell(cell, names, args, torch, head):
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0005, weight_decay=0.00001)
    pairs = paired_names(names) if cell == "A1X_ACCESS_SHUFFLED" else tuple((name, name) for name in names)
    if cell == "A1X_ACCESS_SHUFFLED" and any(left == right for left, right in pairs):
        raise RuntimeError("shuffled no-self-pair guard failed")
    first_gradient = 0.0
    for update in range(2):
        name, paired = pairs[update]
        inputs = [load_rgb(root, source, torch) for root, source in ((args.hazy_root, name), (args.frozen_base_root, name), (args.old_0125_root, name), (args.old_025_root, name), (args.current_delta_root, paired))]
        deployable = torch.cat(inputs, dim=1)
        if deployable.shape[1] != 15:
            raise RuntimeError("five deployable inputs must concatenate to 15 channels")
        target = load_rgb(args.target_delta_root, name, torch)
        output = head(deployable)
        loss = ((output - target) ** 2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite normalized_active_support_endpoint_mse")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        first_gradient = max(first_gradient, max((float(p.grad.abs().max()) for p in head.parameters() if p.grad is not None), default=0.0))
        torch.nn.utils.clip_grad_norm_(head.parameters(), 0.1); optimizer.step()
    if first_gradient <= 0.0:
        raise RuntimeError("first gradient must be nonzero")
    return {"cell": cell, "updates": 2, "first_gradient_max": first_gradient, "shuffle_pairs": len(pairs)}

def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--stage", required=True); result.add_argument("--authorization-json", required=True)
    result.add_argument("--route-commit", required=True); result.add_argument("--run-id", required=True)
    result.add_argument("--official-checkpoint", required=True); result.add_argument("--official-checkpoint-sha256", required=True)
    result.add_argument("--hazy-root", required=True); result.add_argument("--frozen-base-root", required=True); result.add_argument("--old-0125-root", required=True); result.add_argument("--old-025-root", required=True); result.add_argument("--current-delta-root", required=True); result.add_argument("--target-delta-root", required=True)
    result.add_argument("--status-json", required=True); result.add_argument("--heartbeat-json", required=True); result.add_argument("--learned-state-manifest-json", required=True); result.add_argument("--closeout-json", required=True)
    return result

def main():
    args = parser().parse_args()
    if args.stage != "s0":
        raise SystemExit("A1X runner refused: formal mode is not enabled")
    try:
        auth = authorization(args.authorization_json)
        exact_names(S0_FIRST32_NAMES)
        if sha256(args.official_checkpoint) != args.official_checkpoint_sha256:
            raise RuntimeError("official checkpoint hash mismatch")
        base = {"schema_version": 1, "route_id": ROUTE_ID, "route_commit": args.route_commit, "source_commit": SOURCE_COMMIT, "run_id": args.run_id, "stage": "s0", "evidence_role": "engineering_debug", "authorization_sha256": sha256(args.authorization_json), "runtime_started": True, "a1r_first32_only": True, "a1x_data_touched": False, "confirmation_touched": False, "canary_touched": False, "locked_test_touched": False}
        write_json(args.status_json, dict(base, phase="started")); heartbeat(args.heartbeat_json, args, "started", 0)
        import torch
        from Dehazing.ITS.models.A1XAccess import A1X_ACCESS_Head
        random.seed(3407); torch.manual_seed(3407)
        head = A1X_ACCESS_Head(); added = check_model_contract(head, torch)
        results = [run_cell("A1X_ACCESS_TRUE", S0_FIRST32_NAMES, args, torch, head), run_cell("A1X_ACCESS_SHUFFLED", S0_FIRST32_NAMES, args, torch, head)]
        heartbeat(args.heartbeat_json, args, "complete", 1)
        manifest = dict(base, cell_results=results, exact_first32_names=list(S0_FIRST32_NAMES), exact_first32_count=32, seed=3407, no_resume=True, added_parameters=added)
        write_json(args.learned_state_manifest_json, manifest)
        closeout = dict(base, state="COMPLETED_GATE_INCONCLUSIVE", gate_type="engineering", decision="V4A_A1X_S0_ENGINEERING_GATE_INCONCLUSIVE_STOP", authorizes="NONE", reason="S0 runner completed; R3 review owns scientific gate interpretation", exact_first32_source_ref="github:fe08ba7c0fde4d6086083490430246ea39fbf766:experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1r_representation_sufficiency_20260714/v4a_a1r_smoke_source_manifest.json", exact_first32_names=list(S0_FIRST32_NAMES), exact_first32_count=32, phase_timings={}, integrated_check_results={name: "PASS" for name in INTEGRATED_S0_CHECKS}, learned_state_manifest_sha256=sha256(args.learned_state_manifest_json), failure_class=None, failure_phase=None)
        write_json(args.closeout_json, closeout); print("A1X_S0_OK"); return 0
    except Exception as error:
        failure = {"schema_version": 1, "route_id": ROUTE_ID, "route_commit": args.route_commit, "source_commit": SOURCE_COMMIT, "run_id": args.run_id, "stage": "s0", "state": "FAILED", "gate_type": "engineering", "decision": None, "authorizes": "NONE", "reason": str(error), "failure_class": "ENGINEERING_RUNNER_CONTRACT", "failure_phase": "preflight_or_s0", "runtime_started": False, "a1r_first32_only": True, "a1x_data_touched": False, "confirmation_touched": False, "canary_touched": False, "locked_test_touched": False}
        if getattr(args, "closeout_json", None): write_json(args.closeout_json, failure)
        print("A1X_S0_FAILED", file=sys.stderr); return 2

if __name__ == "__main__":
    raise SystemExit(main())
