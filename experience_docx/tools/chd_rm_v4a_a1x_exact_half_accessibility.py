#!/usr/bin/env python3
"""S0-only A1X accessibility runner with source-auditable contracts.

This entrypoint is intentionally inert until a separately authorized cloud
runner supplies the pinned assets.  It never owns the terminal marker; the
shell runner emits that only after durable terminal evidence exists.
"""
import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
SOURCE_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
AUTH_TUPLE = {"state": "PLANNED", "decision": "V4A_A1X_S0_AUTHORIZED_INITIAL_ONLY", "authorizes": "A1X_S0_ONLY"}
OPERATORS = ("D_ref", "D_rep")
S0_FIRST32_NAMES = (
    "1594_0.71_0.5.png", "1595_0.99_1.84.png", "1597_0.69_1.45.png", "1598_0.67_1.4.png", "159_0.6_1.46.png", "1600_0.78_1.77.png", "1603_0.54_0.74.png", "1607_0.91_0.88.png", "160_0.63_1.04.png", "1613_0.56_1.31.png", "1614_0.81_0.78.png", "1615_0.91_1.25.png", "1616_0.76_0.88.png", "1617_0.56_1.97.png", "1619_0.94_1.08.png", "1622_0.98_1.75.png", "1623_0.78_1.81.png", "1627_0.94_0.52.png", "1628_0.8_1.49.png", "1633_0.73_1.49.png", "1634_0.75_1.81.png", "1639_0.69_1.12.png", "1640_0.53_0.59.png", "1646_0.55_1.55.png", "1649_0.8_0.86.png", "1650_0.76_1.07.png", "1652_0.62_1.35.png", "1653_0.9_1.01.png", "1654_0.66_1.9.png", "1656_0.64_1.45.png", "1658_0.96_1.72.png", "1660_0.83_0.67.png",
)

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, payload, atomic=False):
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if atomic:
        temporary = destination.with_name(destination.name + ".tmp"); temporary.write_text(encoded, encoding="utf-8"); os.replace(temporary, destination)
    else: destination.write_text(encoded, encoding="utf-8")

def append_status(path, payload):
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle: handle.write(json.dumps(payload, sort_keys=True) + "\n")

class Heartbeat:
    def __init__(self, path, args): self.path, self.args, self.last = path, args, 0.0
    def refresh(self, phase, progress, force=False):
        now = time.time()
        if force or now - self.last >= 60.0:
            write_json(self.path, {"route_id": ROUTE_ID, "route_commit": self.args.route_commit, "run_id": self.args.run_id, "stage": "s0", "phase": phase, "timestamp": now, "progress": progress}, atomic=True); self.last = now

def authorization(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("route_id") != ROUTE_ID or payload.get("source_commit") != SOURCE_COMMIT or any(payload.get(key) != value for key, value in AUTH_TUPLE.items()): raise RuntimeError("authorization tuple mismatch")
    return payload

def exact_names(names):
    if tuple(names) != S0_FIRST32_NAMES or len(names) != 32 or len(set(names)) != 32: raise RuntimeError("exact first32 membership/order guard failed")

def image_path(root, name):
    if name not in S0_FIRST32_NAMES: raise RuntimeError("nonmember image name refused")
    return Path(root) / name

def load_rgb(root, name, torch):
    from PIL import Image
    with Image.open(image_path(root, name)) as image:
        return torch.from_numpy(__import__("numpy").asarray(image.convert("RGB"))).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

def load_official_model(args, torch):
    from Dehazing.ITS.models.ConvIR import build_net
    if sha256(args.official_checkpoint) != args.official_checkpoint_sha256: raise RuntimeError("official checkpoint hash mismatch")
    checkpoint = torch.load(args.official_checkpoint, map_location="cpu", weights_only=True)
    official_state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    official_model = build_net("base", "haze4k")
    expected = official_model.state_dict()
    if set(official_state) != set(expected) or any(official_state[key].shape != expected[key].shape for key in expected): raise RuntimeError("official checkpoint exact key/shape audit failed")
    official_model.load_state_dict(official_state, strict=True)
    official_model.eval()
    for parameter in official_model.parameters(): parameter.requires_grad_(False)
    if official_model.training or any(parameter.requires_grad for parameter in official_model.parameters()): raise RuntimeError("official model freeze/eval audit failed")
    return official_model, {"official_checkpoint_sha256": sha256(args.official_checkpoint), "strict_load": True, "official_eval": not official_model.training, "official_trainable_parameters": sum(parameter.numel() for parameter in official_model.parameters() if parameter.requires_grad)}

def trainable_head(torch):
    from Dehazing.ITS.models.A1XAccess import A1X_ACCESS_Head
    head = A1X_ACCESS_Head()
    head.train()
    head_parameters = tuple(head.parameters())
    if not head_parameters: raise RuntimeError("A1X head has no registered parameters")
    for parameter in head_parameters: parameter.requires_grad_(True)
    if not all(parameter.requires_grad for parameter in head_parameters): raise RuntimeError("A1X head trainability audit failed")
    optimizer = torch.optim.AdamW(head_parameters, lr=0.0005, weight_decay=0.00001)
    return head, optimizer, sum(parameter.numel() for parameter in head_parameters)

def active_support(tensor, torch): return torch.ones_like(tensor, dtype=torch.bool)
def normalized_active_support_endpoint_mse(predicted, target, support, torch):
    error = (predicted - target).square() * support.to(predicted.dtype)
    denominator = support.sum().to(predicted.dtype)
    if denominator.item() <= 0: raise RuntimeError("empty frozen active support")
    return error.sum() / denominator

def load_example(args, name, operator, torch):
    source = {"D0_HAZY_RGB": load_rgb(args.hazy_root, name, torch), "FROZEN_BASE_RGB": load_rgb(args.frozen_base_root, name, torch), "OLD_0P125_RGB": load_rgb(args.old_0125_root, name, torch), "OLD_0P25_RGB": load_rgb(args.old_025_root, name, torch), "CURRENT_DELTA_U": load_rgb(args.current_delta_root, name, torch)}
    return {"name": name, "operator": operator, "shape": tuple(source["CURRENT_DELTA_U"].shape[-2:]), "source": source, "TARGET_DELTA_U": load_rgb(args.target_delta_root, name, torch)}

def native_shape_batches(examples):
    blocks = defaultdict(list)
    for example in examples: blocks[(example["operator"], example["shape"])].append(example)
    batches = []
    for key in sorted(blocks):
        ordered = sorted(blocks[key], key=lambda item: item["name"])
        batches.extend(ordered[index:index + 4] for index in range(0, len(ordered), 4))
    if not batches or any(len(batch) > 4 for batch in batches): raise RuntimeError("native-shape batch audit failed")
    return batches

def shuffled_target_map(examples):
    groups = defaultdict(list)
    for example in examples: groups[(example["operator"], example["shape"])].append(example)
    paired = {}
    for key, group in groups.items():
        ordered = sorted(group, key=lambda item: item["name"])
        if len(ordered) < 2: raise RuntimeError("shuffle block cannot avoid self-pair")
        for index, source in enumerate(ordered): paired[(source["operator"], source["name"])] = ordered[(index + 1) % len(ordered)]
    if len(paired) != len(examples) or any(source_name == target["name"] for (_, source_name), target in paired.items()): raise RuntimeError("incomplete or self-paired shuffled target map")
    return paired

def forward_batch(batch, target_map, head, official_model, torch):
    source_names = [example["name"] for example in batch]
    fields = ("D0_HAZY_RGB", "FROZEN_BASE_RGB", "OLD_0P125_RGB", "OLD_0P25_RGB", "CURRENT_DELTA_U")
    deployable = torch.cat([torch.cat([example["source"][field] for example in batch], dim=0) for field in fields], dim=1)
    if deployable.shape[1] != 15: raise RuntimeError("five deployable inputs must concatenate to 15 channels")
    with torch.no_grad(): official_model(torch.cat([example["source"]["D0_HAZY_RGB"] for example in batch], dim=0))
    targets = [target_map[(example["operator"], name)]["TARGET_DELTA_U"] for example, name in zip(batch, source_names)]
    target = torch.cat(targets, dim=0); support = active_support(target, torch)
    predicted = head(deployable)
    if tuple(predicted.shape[-2:]) != tuple(target.shape[-2:]): raise RuntimeError("native shape preservation failure")
    return normalized_active_support_endpoint_mse(predicted, target, support, torch)

def run_cell(cell, batches, shuffled, head, optimizer, official_model, heartbeat, torch):
    target_map = {(example["operator"], example["name"]): example for batch in batches for example in batch} if cell == "A1X_ACCESS_TRUE" else shuffled
    first_gradient = 0.0
    for update in range(2):
        loss = forward_batch(batches[update % len(batches)], target_map, head, official_model, torch)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite normalized_active_support_endpoint_mse")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        first_gradient = max(first_gradient, max((float(item.grad.abs().max()) for item in head.parameters() if item.grad is not None), default=0.0))
        torch.nn.utils.clip_grad_norm_(head.parameters(), 0.1); optimizer.step(); heartbeat.refresh(cell, (update + 1) / 2)
    if first_gradient <= 0.0: raise RuntimeError("first gradient must be nonzero")
    return {"cell": cell, "updates": 2, "first_gradient_max": first_gradient}

def profile_measurement(fn, torch):
    timings = []
    for _ in range(5):
        started = time.perf_counter(); fn(); timings.append(time.perf_counter() - started)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profiler: fn()
    return {"median_seconds": sorted(timings)[len(timings) // 2], "macs": sum(event.flops for event in profiler.key_averages() if event.flops), "peak_memory_bytes": max((event.cpu_memory_usage for event in profiler.key_averages()), default=0)}

def measure_checks(head, official_model, batches, torch):
    by_shape = sorted({batch[0]["shape"] for batch in batches})
    measurements = {"added_parameters": sum(parameter.numel() for parameter in head.parameters()), "current_endpoint_noop_max_abs": float(head.final_projection.weight.detach().abs().max()), "native_shape_preserved": all(len(batch) <= 4 for batch in batches), "mac_overhead_percent_by_shape": {}, "latency_overhead_percent_by_shape": {}, "peak_memory_overhead_percent_by_shape": {}}
    for shape in by_shape:
        probe = torch.zeros((1, 15, *shape))
        baseline = profile_measurement(lambda: official_model(probe[:, :3]), torch)
        augmented = profile_measurement(lambda: (official_model(probe[:, :3]), head(probe)), torch)
        if min(baseline.values()) <= 0: raise RuntimeError("invalid baseline cost measurement")
        measurements["latency_overhead_percent_by_shape"][str(shape)] = 100.0 * (augmented["median_seconds"] - baseline["median_seconds"]) / baseline["median_seconds"]
        measurements["mac_overhead_percent_by_shape"][str(shape)] = 100.0 * (augmented["macs"] - baseline["macs"]) / baseline["macs"]
        measurements["peak_memory_overhead_percent_by_shape"][str(shape)] = 100.0 * (augmented["peak_memory_bytes"] - baseline["peak_memory_bytes"]) / baseline["peak_memory_bytes"]
    results = {"zero_initialized_exact_noop": measurements["current_endpoint_noop_max_abs"] <= 0.0, "native_shape_preservation": measurements["native_shape_preserved"], "added_parameters": measurements["added_parameters"] <= 300000, "mac_overhead": all(value <= 10.0 for value in measurements["mac_overhead_percent_by_shape"].values()), "matched_median_latency_overhead": all(value <= 15.0 for value in measurements["latency_overhead_percent_by_shape"].values()), "peak_memory_overhead": all(value <= 15.0 for value in measurements["peak_memory_overhead_percent_by_shape"].values())}
    if len(by_shape) != 2 or not all(results.values()): raise RuntimeError("measured structural/cost check failed")
    return measurements, results

def parser():
    result = argparse.ArgumentParser();
    for name in ("stage", "authorization-json", "route-commit", "run-id", "official-checkpoint", "official-checkpoint-sha256", "hazy-root", "frozen-base-root", "old-0125-root", "old-025-root", "current-delta-root", "target-delta-root", "status-json", "heartbeat-json", "learned-state-manifest-json", "closeout-json"): result.add_argument("--" + name, required=True)
    return result

def main():
    args = parser().parse_args(); runtime_started = False; phase = "authorization"; started = time.monotonic()
    try:
        if args.stage != "s0": raise RuntimeError("formal mode is not enabled")
        authorization(args.authorization_json); exact_names(S0_FIRST32_NAMES)
        base = {"schema_version": 2, "route_id": ROUTE_ID, "route_commit": args.route_commit, "source_commit": SOURCE_COMMIT, "run_id": args.run_id, "stage": "s0", "evidence_role": "engineering_debug", "authorization_sha256": sha256(args.authorization_json), "runner_sha256": sha256(Path(__file__).with_name("run_chd_rm_v4a_a1x_exact_half_accessibility.sh")), "route_card_sha256": sha256(Path(__file__).parents[1] / "experiment_cards/2026-07-15-haze4k-v5-v4a-a1x-exact-half-deployable-accessibility.md"), "a1r_first32_only": True, "a1x_data_touched": False, "confirmation_touched": False, "canary_touched": False, "locked_test_touched": False}
        append_status(args.status_json, dict(base, event="start", phase=phase)); heartbeat = Heartbeat(args.heartbeat_json, args); heartbeat.refresh(phase, 0.0, True)
        import torch
        random.seed(3407); torch.manual_seed(3407); phase = "model_load"; runtime_started = True
        official_model, official_audit = load_official_model(args, torch); head, optimizer, added = trainable_head(torch)
        append_status(args.status_json, dict(base, event="phase", phase=phase, official_audit=official_audit)); heartbeat.refresh(phase, 0.1, True)
        phase = "data_and_cells"; examples = [load_example(args, name, operator, torch) for operator in OPERATORS for name in S0_FIRST32_NAMES]; batches = native_shape_batches(examples); shuffled = shuffled_target_map(examples)
        cells = [run_cell("A1X_ACCESS_TRUE", batches, shuffled, head, optimizer, official_model, heartbeat, torch), run_cell("A1X_ACCESS_SHUFFLED", batches, shuffled, head, optimizer, official_model, heartbeat, torch)]
        phase = "measurements"; measurements, integrated = measure_checks(head, official_model, batches, torch); phase_timings = {"total_seconds": time.monotonic() - started}
        manifest = dict(base, learned_states=[{"path": "not_exported", "sha256": None, "cell": item["cell"], "updates": item["updates"]} for item in cells], seed=3407, no_resume=True); write_json(args.learned_state_manifest_json, manifest)
        closeout = dict(base, state="COMPLETED_GATE_INCONCLUSIVE", gate_type="engineering", decision="V4A_A1X_S0_ENGINEERING_GATE_INCONCLUSIVE_STOP", authorizes="NONE", reason="R3 owns scientific interpretation", exact_first32_source_ref="github:fe08ba7c0fde4d6086083490430246ea39fbf766:experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1r_representation_sufficiency_20260714/v4a_a1r_smoke_source_manifest.json", exact_first32_names=list(S0_FIRST32_NAMES), exact_first32_count=32, phase_timings=phase_timings, integrated_check_results=integrated, measured_values=measurements, learned_state_manifest_sha256=sha256(args.learned_state_manifest_json), failure_class=None, failure_phase=None, runtime_started=runtime_started)
        write_json(args.closeout_json, closeout); append_status(args.status_json, dict(base, event="terminal", phase="complete", closeout_sha256=sha256(args.closeout_json))); heartbeat.refresh("complete", 1.0, True); return 0
    except Exception as error:
        failure = {"schema_version": 2, "route_id": ROUTE_ID, "route_commit": getattr(args, "route_commit", None), "source_commit": SOURCE_COMMIT, "run_id": getattr(args, "run_id", None), "stage": "s0", "evidence_role": "engineering_debug", "state": "FAILED", "gate_type": "engineering", "decision": None, "authorizes": "NONE", "reason": str(error), "authorization_sha256": None, "runner_sha256": sha256(Path(__file__).with_name("run_chd_rm_v4a_a1x_exact_half_accessibility.sh")), "route_card_sha256": sha256(Path(__file__).parents[1] / "experiment_cards/2026-07-15-haze4k-v5-v4a-a1x-exact-half-deployable-accessibility.md"), "exact_first32_source_ref": None, "exact_first32_names": list(S0_FIRST32_NAMES), "exact_first32_count": 32, "phase_timings": {"elapsed_seconds": time.monotonic() - started}, "integrated_check_results": {}, "learned_state_manifest_sha256": None, "failure_class": "ENGINEERING_RUNNER_CONTRACT", "failure_phase": phase, "runtime_started": runtime_started, "a1r_first32_only": True, "a1x_data_touched": False, "confirmation_touched": False, "canary_touched": False, "locked_test_touched": False}
        if getattr(args, "closeout_json", None): write_json(args.closeout_json, failure)
        if getattr(args, "status_json", None): append_status(args.status_json, dict(failure, event="terminal"))
        return 2

if __name__ == "__main__": raise SystemExit(main())
