#!/usr/bin/env python3
"""Future S0 runner. It is inert until separately authorized on convir-4090."""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
S0_FIRST32_NAMES = ("1594_0.71_0.5.png", "1595_0.99_1.84.png", "1597_0.69_1.45.png", "1598_0.67_1.4.png", "159_0.6_1.46.png", "1600_0.78_1.77.png", "1603_0.54_0.74.png", "1607_0.91_0.88.png", "160_0.63_1.04.png", "1613_0.56_1.31.png", "1614_0.81_0.78.png", "1615_0.91_1.25.png", "1616_0.76_0.88.png", "1617_0.56_1.97.png", "1619_0.94_1.08.png", "1622_0.98_1.75.png", "1623_0.78_1.81.png", "1627_0.94_0.52.png", "1628_0.8_1.49.png", "1633_0.73_1.49.png", "1634_0.75_1.81.png", "1639_0.69_1.12.png", "1640_0.53_0.59.png", "1646_0.55_1.55.png", "1649_0.8_0.86.png", "1650_0.76_1.07.png", "1652_0.62_1.35.png", "1653_0.9_1.01.png", "1654_0.66_1.9.png", "1656_0.64_1.45.png", "1658_0.96_1.72.png", "1660_0.83_0.67.png")
TERMINALS = {"PASS": ("COMPLETED_GATE_PASS", "V4A_A1X_S0_PASS_AUTHORIZE_FORMAL_ONLY", "A1X_FORMAL_CONFIRMATION_ONLY"), "FAIL": ("COMPLETED_GATE_FAIL", "V4A_A1X_S0_ENGINEERING_GATE_FAIL_STOP", "NONE"), "INCONCLUSIVE": ("COMPLETED_GATE_INCONCLUSIVE", "V4A_A1X_S0_ENGINEERING_GATE_INCONCLUSIVE_STOP", "NONE")}

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path, payload): Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def exact_names(names):
    if tuple(names) != S0_FIRST32_NAMES: raise RuntimeError("exact committed first32 order required")

def render_d_ref(frozen_artifact, renderer): return renderer.render_reference(frozen_artifact)
def render_d_rep(frozen_artifact, renderer): return renderer.render_representation(frozen_artifact)
def operator_render(operator, frozen_artifact, renderer):
    if operator == "D_ref": return render_d_ref(frozen_artifact, renderer)
    if operator == "D_rep": return render_d_rep(frozen_artifact, renderer)
    raise RuntimeError("unknown frozen operator")

def measured_active_support(operator_batch, frozen_support_artifact, torch):
    support = frozen_support_artifact.for_operator_and_shape(operator_batch["operator"], operator_batch["native_shape"])
    if not torch.isfinite(support).all() or support.dtype != torch.bool or int(support.sum().item()) == 0: raise RuntimeError("frozen measured support invalid")
    return support, int(support.sum().item())
def normalized_active_support_endpoint_mse(predicted, target, support, active_count, torch):
    if active_count <= 0 or not torch.isfinite(predicted).all(): raise RuntimeError("invalid active support loss")
    return ((predicted - target).square() * support.to(predicted.dtype)).sum() / active_count

def independent_cell_bundle(official_model, seed, torch):
    random.seed(seed); torch.manual_seed(seed)
    from Dehazing.ITS.models.A1XAccess import A1X_ACCESS_Head
    head = A1X_ACCESS_Head(); head_state = copy.deepcopy(head.state_dict())
    model = copy.deepcopy(official_model); model.load_state_dict(copy.deepcopy(official_model.state_dict()), strict=True)
    optimizer = torch.optim.AdamW(tuple(head.parameters()), lr=0.0005, weight_decay=0.00001)
    return {"model": model, "head": head, "optimizer": optimizer, "initial_head_state": head_state, "updates": 0}
def independent_cells(official_model, torch):
    true_cell = independent_cell_bundle(official_model, 3407, torch)
    shuffled_cell = independent_cell_bundle(official_model, 3407, torch)
    if true_cell["model"] is shuffled_cell["model"] or true_cell["head"] is shuffled_cell["head"] or true_cell["optimizer"] is shuffled_cell["optimizer"]: raise RuntimeError("true/shuffled learned state must be disjoint")
    if true_cell["initial_head_state"] != shuffled_cell["initial_head_state"]: raise RuntimeError("identical pre-update initialization required")
    return {"A1X_ACCESS_TRUE": true_cell, "A1X_ACCESS_SHUFFLED": shuffled_cell}
def verify_zero_noop(cell, operator_batches, forward, torch):
    discrepancies = [float(forward(cell["model"], cell["head"], batch).abs().max()) for batch in operator_batches]
    maximum = max(discrepancies)
    if maximum != 0.0: raise RuntimeError("zero final projection no-op must be exactly zero before update")
    return maximum
def update_cell(cell, batches, forward, target_selector, support_artifact, torch):
    gradient_max = 0.0
    for batch in batches[:2]:
        target = target_selector(batch); support, active_count = measured_active_support(batch, support_artifact, torch)
        loss = normalized_active_support_endpoint_mse(forward(cell["model"], cell["head"], batch), target, support, active_count, torch)
        cell["optimizer"].zero_grad(set_to_none=True); loss.backward(); gradient_max = max(gradient_max, max(float(p.grad.abs().max()) for p in cell["head"].parameters() if p.grad is not None)); torch.nn.utils.clip_grad_norm_(cell["head"].parameters(), 0.1); cell["optimizer"].step(); cell["updates"] += 1
    if cell["updates"] != 2 or gradient_max <= 0.0: raise RuntimeError("exactly two finite updates required")
    return gradient_max

def cuda_profile_pair(baseline, augmented, shape, torch):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_UNAVAILABLE_INCONCLUSIVE")
    device = torch.device("cuda"); probe = torch.zeros((1, 15, *shape), device=device, dtype=torch.float32)
    def measured_macs(fn):
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], with_flops=True) as profiler: fn(probe)
        torch.cuda.synchronize(device); flops = sum(event.flops for event in profiler.key_averages() if event.flops)
        macs = flops / 2.0
        if not macs > 0.0: raise RuntimeError("nonpositive measured CUDA MACs")
        return macs
    def measure(fn):
        for _ in range(5): fn(probe)
        samples = []
        for _ in range(11):
            torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); started = time.perf_counter(); fn(probe); torch.cuda.synchronize(device); samples.append((time.perf_counter() - started, torch.cuda.max_memory_allocated(device)))
        samples.sort(); return {"macs": measured_macs(fn), "median_seconds": samples[len(samples)//2][0], "peak_memory_bytes": max(item[1] for item in samples)}
    return measure(baseline), measure(augmented)
def cuda_cost_contract(baseline, augmented, native_shapes, torch):
    if len(native_shapes) != 2: raise RuntimeError("both frozen native shapes required")
    measurements = {str(shape): cuda_profile_pair(baseline, augmented, shape, torch) for shape in native_shapes}
    overheads = {}
    for shape, (baseline_cost, augmented_cost) in measurements.items():
        if not all(value > 0.0 and value < float("inf") for value in baseline_cost.values()): raise RuntimeError("invalid baseline CUDA cost denominator")
        overheads[shape] = {metric: 100.0 * (augmented_cost[metric] - baseline_cost[metric]) / baseline_cost[metric] for metric in ("macs", "median_seconds", "peak_memory_bytes")}
    results = {"mac_overhead": all(cost["macs"] <= 10.0 for cost in overheads.values()), "matched_median_cuda_latency_overhead": all(cost["median_seconds"] <= 15.0 for cost in overheads.values()), "cuda_peak_memory_overhead": all(cost["peak_memory_bytes"] <= 15.0 for cost in overheads.values())}
    if not all(results.values()): raise RuntimeError("frozen CUDA cost gate failed")
    return {"measurements": measurements, "overhead_percent_by_shape": overheads, "results": results}

def structural_s0_gate(head, baseline, augmented, native_shapes, torch):
    added_parameters = sum(parameter.numel() for parameter in head.parameters())
    parameter_limit = added_parameters <= 300000
    cuda_cost = cuda_cost_contract(baseline, augmented, native_shapes, torch)
    cuda_results = cuda_cost["results"]
    if not parameter_limit or not all(cuda_results.values()): raise RuntimeError("frozen structural S0 gate failed")
    return {"added_parameters": added_parameters, "cuda_cost": cuda_cost, "results": {"added_parameter_limit": parameter_limit, **cuda_results}}

def retain_learned_states(cells, run_root, base):
    records = []
    for name, cell in cells.items():
        path = Path(run_root) / (name.lower() + ".pt"); path.parent.mkdir(parents=True, exist_ok=True)
        import torch; torch.save({"model": cell["model"].state_dict(), "head": cell["head"].state_dict()}, path)
        records.append(dict(base, cell=name, absolute_state_path=str(path.resolve()), relative_state_path=str(path.relative_to(run_root)), sha256=sha256(path), seed=3407, update_count=cell["updates"], no_resume=True))
    if len({record["sha256"] for record in records}) != 2 or len({record["absolute_state_path"] for record in records}) != 2: raise RuntimeError("distinct written learned states required")
    return records
def terminal_closeout(kind, base, closeout_path):
    state, decision, authorizes = TERMINALS[kind]; payload = dict(base, state=state, decision=decision, authorizes=authorizes); write_json(closeout_path, payload); return payload

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True); parser.add_argument("--run-root", required=True); parser.add_argument("--learned-state-manifest-json", required=True); parser.add_argument("--closeout-json", required=True); args = parser.parse_args()
    if args.stage != "s0": raise RuntimeError("formal mode blocked")
    # Runtime wiring is deliberately reachable only from the separately authorized cloud runner.
    raise RuntimeError("S0 runner requires separately authorized cloud assets")
if __name__ == "__main__": main()
