#!/usr/bin/env python3
"""Train and audit a zero-init Delta-u repair of the frozen v3l operator.

This route intentionally reuses the frozen, cloud-only v3l direct operators as
inputs.  It does not train a selector, confidence head, threshold, or policy.
The only trainable parameters are ``DIRR_*`` in ``DIRR_DeltaU``.
"""

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


ROUTE_ID = "haze4k_v5_chd_rm_v3s_delta_u_direction_repair_20260713"
OPERATORS = ("D_ref", "D_rep")
ALPHA_LOW = 0.125
ALPHA_HIGH = 0.25
BLOCK_SIZE = 16
CANONICAL_ATOL_SSE = 1e-12
CANONICAL_RTOL = 1e-12
SEVERE_DB = -0.2


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_output(repo, *arguments):
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clean_reference_group(name):
    return Path(name).stem.split("_", 1)[0]


def import_route_module():
    route_root = Path(__file__).resolve().parents[2]
    model_dir = route_root / "Dehazing" / "ITS" / "models"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    return importlib.import_module("direction_repair")


def import_legacy_modules(legacy_repo):
    """Import the pinned v3p implementation without using it as governance."""
    legacy_repo = Path(legacy_repo).resolve()
    tools_dir = legacy_repo / "experience_docx" / "tools"
    its_dir = legacy_repo / "Dehazing" / "ITS"
    if not tools_dir.is_dir() or not its_dir.is_dir():
        raise FileNotFoundError(f"invalid legacy v3p repository: {legacy_repo}")
    for path in (str(its_dir), str(tools_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("chd_rm_v3p_a0_canonical_signed_gain")


def asset_manifest(args):
    expected = {
        "a0_checkpoint": args.expected_a0_checkpoint_sha256,
        "control_checkpoint": args.expected_control_checkpoint_sha256,
        "density_artifact": args.expected_density_artifact_sha256,
        "d7c_artifact": args.expected_d7c_artifact_sha256,
        "fresh_split_manifest": args.expected_fresh_split_manifest_sha256,
        "operator_artifact_manifest": args.expected_operator_manifest_sha256,
        "reference_oof_rows": args.expected_reference_oof_rows_sha256,
        "v3j_a_bounds": args.expected_v3j_a_bounds_sha256,
    }
    observed = {}
    for key, expected_hash in expected.items():
        path = Path(getattr(args, key))
        if not path.is_file():
            raise FileNotFoundError(f"required asset missing: {key}={path}")
        digest = sha256_file(path)
        if digest != expected_hash:
            raise RuntimeError(f"asset hash mismatch for {key}: {digest} != {expected_hash}")
        observed[key] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    return observed


def build_frozen_operator(args, legacy, device):
    """Strictly load the frozen base, control, gate, and two direct heads."""
    gate_args = SimpleNamespace(
        version="base",
        data="Haze4K",
        fam_mode="fam2_d7c_noop",
        d7c_gate_mode="d7c_fixed",
        d7c_base_checkpoint=args.a0_checkpoint,
        d7c_density_artifact=args.density_artifact,
        d7c_need_artifact=args.d7c_artifact,
        d7c_threshold=float(args.d7c_threshold),
        proj_channels=int(args.proj_channels),
    )
    base = legacy.build_model("original", args.a0_checkpoint, device)
    control = legacy.build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = legacy.build_gate_producer(gate_args, device)
    for module in (base, control):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    bound = read_json(args.v3j_a_bounds)["channel_bounds_rgb"]
    if len(bound) != 3 or any(float(value) <= 0.0 for value in bound):
        raise ValueError("v3j channel bounds are invalid")
    artifact_entries = read_json(args.operator_artifact_manifest)
    if not isinstance(artifact_entries, list) or len(artifact_entries) != 2:
        raise ValueError("operator artifact manifest must contain exactly D_ref and D_rep")
    caches = {}
    artifact_summary = []
    for entry in artifact_entries:
        label = entry.get("operator_label")
        if label not in OPERATORS or label in caches:
            raise ValueError(f"unexpected operator manifest label: {label}")
        artifact_path = Path(entry["artifact_path"])
        if not artifact_path.is_file():
            raise FileNotFoundError(f"missing frozen operator artifact: {artifact_path}")
        digest = sha256_file(artifact_path)
        if digest != entry.get("artifact_sha256"):
            raise RuntimeError(f"operator artifact hash mismatch: {label}")
        artifact = torch.load(artifact_path, map_location=device)
        if artifact.get("operator_label") != label:
            raise RuntimeError(f"operator label mismatch in artifact: {label}")
        if int(artifact.get("config", {}).get("proj_channels", -1)) != int(args.proj_channels):
            raise RuntimeError(f"operator projection width mismatch: {label}")
        caches[label] = legacy.v3l_a0.build_model_cache(artifact, gate_args, device)
        artifact_summary.append(
            {
                "operator_label": label,
                "path": str(artifact_path),
                "sha256": digest,
                "seed": int(artifact["seed"]),
            }
        )
    if set(caches) != set(OPERATORS):
        raise RuntimeError("frozen operator pair is incomplete")
    return {
        "base": base,
        "control": control,
        "gate_producer": gate_producer,
        "caches": caches,
        "bound": [float(value) for value in bound],
        "artifacts": sorted(artifact_summary, key=lambda row: row["operator_label"]),
    }


def load_names_and_folds(args, legacy):
    manifest = legacy.read_json(args.fresh_split_manifest)
    names = legacy.names_from_manifest(manifest, args.train_key, args.formal_sample_count)
    if len(names) != args.formal_sample_count or len(set(names)) != len(names):
        raise RuntimeError("frozen OOF train-name contract is invalid")
    folds, _ = legacy.v3l_a1.v3j_b.fold_assignments(names, args.fold_count)
    if sorted(set(int(value) for value in folds.tolist())) != list(range(args.fold_count)):
        raise RuntimeError("frozen OOF folds are incomplete")
    return names, {name: int(fold) for name, fold in zip(names, folds.tolist())}


def frozen_sample(args, legacy, frozen, name, fold, device):
    """Return the old D_ref/D_rep steps from the exact frozen v3p pathway."""
    input_img, label = legacy.load_pair(args.data_dir, args.source_split, name)
    input_img = input_img.unsqueeze(0).to(device)
    label = label.unsqueeze(0).to(device)
    with torch.no_grad():
        padded, height, width = legacy.pad_to_factor(input_img)
        label = label[:, :, :height, :width]
        hazy = padded[:, :, :height, :width]
        base_prediction = legacy.forward_final(frozen["base"], padded, height, width)
        gate_full, _, _ = frozen["gate_producer"](padded)
        hard_gate = legacy.action_gate_from_full(
            gate_full,
            legacy.action_shape_for_input(padded),
        ).to(device)
        support = legacy.output_gate_from_action_gate(hard_gate, base_prediction.shape[-2:])
        fmap, _, _, _ = legacy.full_context_maps(frozen["control"], frozen["gate_producer"], padded)
        steps = {}
        for operator in OPERATORS:
            head, mean, std = legacy.v3l_a0.model_pack_from_cache(
                frozen["caches"][operator], "OOF", fold
            )
            pred_low = legacy.v3l_a0.v3j_b.score_map(
                "context",
                head,
                fmap,
                mean,
                std,
                frozen["bound"],
            )
            steps[operator] = support * F.interpolate(
                pred_low,
                size=base_prediction.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
    return {
        "hazy": hazy.detach(),
        "label": label.detach(),
        "base": base_prediction.detach(),
        "support": support.detach(),
        "steps": {key: value.detach() for key, value in steps.items()},
    }


def candidate_predictions(base, step, delta):
    old_low = torch.clamp(base + ALPHA_LOW * step, 0.0, 1.0)
    old_high = torch.clamp(base + ALPHA_HIGH * step, 0.0, 1.0)
    new_low = torch.clamp(base + ALPHA_LOW * (step + delta), 0.0, 1.0)
    new_high = torch.clamp(base + ALPHA_HIGH * (step + delta), 0.0, 1.0)
    return old_low, old_high, new_low, new_high


def per_image_mse(prediction, label):
    return (prediction - label).square().mean(dim=(1, 2, 3))


def active_block_negative_margin_loss(low_prediction, high_prediction, label, support):
    error_low = (low_prediction - label).square().mean(dim=1, keepdim=True)
    error_high = (high_prediction - label).square().mean(dim=1, keepdim=True)
    low_blocks = F.avg_pool2d(error_low, BLOCK_SIZE, BLOCK_SIZE, ceil_mode=True, count_include_pad=False)
    high_blocks = F.avg_pool2d(error_high, BLOCK_SIZE, BLOCK_SIZE, ceil_mode=True, count_include_pad=False)
    active = F.max_pool2d(support, BLOCK_SIZE, BLOCK_SIZE, ceil_mode=True) > 0.5
    if not bool(active.any()):
        return low_blocks.new_zeros(())
    return F.relu(-(low_blocks - high_blocks)[active]).mean()


def loss_bundle(args, delta, sample, operator):
    step = sample["steps"][operator]
    old_low, _, new_low, new_high = candidate_predictions(sample["base"], step, delta)
    old_low_mse = per_image_mse(old_low, sample["label"])
    new_low_mse = per_image_mse(new_low, sample["label"])
    new_high_mse = per_image_mse(new_high, sample["label"])
    anchor_harm = F.relu(new_low_mse - old_low_mse)
    high_vs_anchor_harm = F.relu(new_high_mse - old_low_mse)
    block_margin = active_block_negative_margin_loss(
        new_low,
        new_high,
        sample["label"],
        sample["support"],
    )
    repair_fraction = torch.mean(torch.abs(delta) / (delta.new_tensor(args.delta_bound).view(1, 3, 1, 1)))
    return {
        "render": new_high_mse.mean(),
        "anchor": anchor_harm.mean(),
        "margin": block_margin,
        "harm": high_vs_anchor_harm.mean(),
        "repair": repair_fraction,
        "delta_abs": torch.mean(torch.abs(delta)),
    }


def train_delta_model(args, legacy, frozen, names, folds, device, checkpoint_path, epochs, run_label):
    direction = import_route_module()
    model = direction.DIRR_DeltaU(
        channels=args.dirr_channels,
        delta_bound=args.delta_bound,
    ).to(device)
    if any(parameter.requires_grad for parameter in frozen["base"].parameters()):
        raise RuntimeError("base model is unexpectedly trainable")
    if any(parameter.requires_grad for parameter in frozen["control"].parameters()):
        raise RuntimeError("control model is unexpectedly trainable")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    history = []
    initial_render = None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_totals = defaultdict(float)
        update_count = 0
        for start in range(0, len(names), args.risk_window):
            window = names[start:start + args.risk_window]
            optimizer.zero_grad(set_to_none=True)
            terms = defaultdict(list)
            for name in window:
                sample = frozen_sample(args, legacy, frozen, name, folds[name], device)
                for operator in OPERATORS:
                    delta = model(sample["hazy"], sample["base"], sample["steps"][operator], sample["support"])
                    bundle = loss_bundle(args, delta, sample, operator)
                    for key, value in bundle.items():
                        terms[key].append(value)
            if not terms["render"]:
                raise RuntimeError("empty optimization window")
            means = {key: torch.stack(values).mean() for key, values in terms.items()}
            harms = torch.stack(terms["harm"])
            cvar_count = max(1, int(math.ceil(args.cvar_fraction * harms.numel())))
            cvar = torch.topk(harms, cvar_count).values.mean()
            total = (
                means["render"]
                + args.anchor_weight * means["anchor"]
                + args.margin_weight * means["margin"]
                + args.harm_weight * means["harm"]
                + args.cvar_weight * cvar
                + args.repair_weight * means["repair"]
            )
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("non-finite training loss")
            total.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm).item())
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("non-finite direction-repair gradient")
            optimizer.step()
            epoch_totals["total"] += float(total.detach().item())
            epoch_totals["cvar"] += float(cvar.detach().item())
            epoch_totals["gradient_norm"] += gradient_norm
            for key, value in means.items():
                epoch_totals[key] += float(value.detach().item())
            update_count += 1
        row = {
            "run_label": run_label,
            "epoch": epoch,
            "updates": update_count,
            **{key: value / update_count for key, value in epoch_totals.items()},
        }
        if initial_render is None:
            initial_render = row["render"]
        history.append(row)
        print(json.dumps({"TRAIN_PROGRESS": row}, sort_keys=True), flush=True)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "route_id": ROUTE_ID,
            "run_label": run_label,
            "state_dict": model.state_dict(),
            "dirr_channels": int(args.dirr_channels),
            "delta_bound": list(args.delta_bound),
            "epochs": int(epochs),
            "loss_contract": {
                "render": "real rendered alpha=.25 MSE",
                "anchor": "ReLU(new .125 MSE - frozen old .125 MSE)",
                "margin": "active block ReLU(-(new G1))",
                "harm": "ReLU(new .25 MSE - frozen old .125 MSE)",
                "cvar": float(args.cvar_fraction),
                "repair": "mean abs(Delta-u / bound)",
            },
        },
        checkpoint_path,
    )
    return model, history, initial_render


def load_direction_checkpoint(path, device):
    direction = import_route_module()
    state = torch.load(path, map_location=device)
    if state.get("route_id") != ROUTE_ID:
        raise RuntimeError("checkpoint route identity mismatch")
    model = direction.DIRR_DeltaU(
        channels=int(state["dirr_channels"]),
        delta_bound=tuple(state["delta_bound"]),
    ).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    return model, state


def psnr_from_mse(mse):
    return 10.0 * math.log10(1.0 / max(float(mse), 1e-30))


def canonical_block_metrics(base, old_step, delta, old_low, old_high, new_low, new_high, label, support):
    """Regenerate float64 G1 labels with exact block coverage accounting."""
    height, width = label.shape[-2:]
    pad_h = (BLOCK_SIZE - height % BLOCK_SIZE) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - width % BLOCK_SIZE) % BLOCK_SIZE
    valid = torch.ones((1, 1, height, width), device=label.device, dtype=torch.float64)
    valid = F.pad(valid, (0, pad_w, 0, pad_h))
    counts = F.avg_pool2d(valid, BLOCK_SIZE, BLOCK_SIZE, stride=BLOCK_SIZE) * (BLOCK_SIZE * BLOCK_SIZE)
    support_blocks = F.max_pool2d(
        F.pad(support.to(torch.float64), (0, pad_w, 0, pad_h)),
        BLOCK_SIZE,
        BLOCK_SIZE,
        stride=BLOCK_SIZE,
    ) > 0.5

    def block_sse(prediction):
        error = (prediction.to(torch.float64) - label.to(torch.float64)).square().sum(dim=1, keepdim=True)
        error = F.pad(error, (0, pad_w, 0, pad_h))
        return F.avg_pool2d(error, BLOCK_SIZE, BLOCK_SIZE, stride=BLOCK_SIZE) * (BLOCK_SIZE * BLOCK_SIZE)

    old_low_sse = block_sse(old_low)
    old_high_sse = block_sse(old_high)
    new_low_sse = block_sse(new_low)
    new_high_sse = block_sse(new_high)
    old_gain = old_low_sse - old_high_sse
    new_gain = new_low_sse - new_high_sse
    old_epsilon = 2.0 * (
        CANONICAL_ATOL_SSE
        + CANONICAL_RTOL * torch.maximum(old_low_sse.abs(), old_high_sse.abs())
    )
    new_epsilon = 2.0 * (
        CANONICAL_ATOL_SSE
        + CANONICAL_RTOL * torch.maximum(new_low_sse.abs(), new_high_sse.abs())
    )
    direction_old = ((label.to(torch.float64) - base.to(torch.float64)) * old_step.to(torch.float64)).sum(
        dim=1,
        keepdim=True,
    )
    direction_new = (
        (label.to(torch.float64) - base.to(torch.float64))
        * (old_step.to(torch.float64) + delta.to(torch.float64))
    ).sum(dim=1, keepdim=True)
    direction_old = F.avg_pool2d(F.pad(direction_old, (0, pad_w, 0, pad_h)), BLOCK_SIZE, BLOCK_SIZE, stride=BLOCK_SIZE)
    direction_new = F.avg_pool2d(F.pad(direction_new, (0, pad_w, 0, pad_h)), BLOCK_SIZE, BLOCK_SIZE, stride=BLOCK_SIZE)
    active = support_blocks & (counts > 0.0)
    if not bool(active.any()):
        return {
            "active_blocks": 0,
            "old_wrong_blocks": 0,
            "new_wrong_blocks": 0,
            "repaired_wrong_blocks": 0,
            "mean_g1_improvement_per_pixel": 0.0,
            "aggregation_abs_error": 0.0,
        }
    old_active = old_gain[active]
    new_active = new_gain[active]
    old_epsilon_active = old_epsilon[active]
    new_epsilon_active = new_epsilon[active]
    count_active = counts[active]
    old_wrong = direction_old[active] <= 0.0
    new_positive = direction_new[active] > 0.0
    old_total = float(torch.sum((old_low.to(torch.float64) - label.to(torch.float64)).square()).item())
    old_blocks = math.fsum(float(value) for value in old_low_sse.reshape(-1).detach().cpu().tolist())
    return {
        "active_blocks": int(active.sum().item()),
        "old_wrong_blocks": int(old_wrong.sum().item()),
        "new_wrong_blocks": int((direction_new[active] <= 0.0).sum().item()),
        "repaired_wrong_blocks": int((old_wrong & new_positive).sum().item()),
        "mean_g1_improvement_per_pixel": float(torch.mean((new_active - old_active) / count_active).item()),
        "old_harmful_g1_blocks": int((old_active < -old_epsilon_active).sum().item()),
        "new_beneficial_g1_blocks": int((new_active > new_epsilon_active).sum().item()),
        "aggregation_abs_error": abs(old_total - old_blocks),
    }


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def cvar_lower(values, fraction):
    if not values:
        return float("nan")
    count = max(1, int(math.ceil(fraction * len(values))))
    return float(np.mean(sorted(float(value) for value in values)[:count]))


def bootstrap_interval(rows, field, q, seed, upper=False, samples=4000):
    groups = defaultdict(list)
    for row in rows:
        groups[row["clean_reference_group"]].append(float(row[field]))
    values = np.asarray([np.mean(group_values) for _, group_values in sorted(groups.items())], dtype=np.float64)
    if values.size == 0:
        raise ValueError("bootstrap received no groups")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draws[index] = float(np.mean(values[rng.integers(0, values.size, size=values.size)]))
    return float(np.quantile(draws, 1.0 - q if upper else q)), int(values.size)


def no_op_smoke(args, legacy, frozen, names, folds, device, output_dir):
    direction = import_route_module()
    model = direction.DIRR_DeltaU(args.dirr_channels, args.delta_bound).to(device).eval()
    reference = legacy.v3m_a1.load_fixed_reference(args.reference_oof_rows)
    max_delta = 0.0
    max_prediction_diff = 0.0
    max_reference_diff = 0.0
    counts = defaultdict(int)
    with torch.no_grad():
        for name in names[:args.smoke_sample_count]:
            sample = frozen_sample(args, legacy, frozen, name, folds[name], device)
            _, base_psnr = legacy.metric_pair(sample["base"], sample["label"])
            for operator in OPERATORS:
                delta = model(sample["hazy"], sample["base"], sample["steps"][operator], sample["support"])
                old_low, _, new_low, _ = candidate_predictions(sample["base"], sample["steps"][operator], delta)
                _, new_psnr = legacy.metric_pair(new_low, sample["label"])
                max_delta = max(max_delta, float(torch.max(torch.abs(delta)).item()))
                max_prediction_diff = max(
                    max_prediction_diff,
                    float(torch.max(torch.abs(new_low - old_low)).item()),
                )
                max_reference_diff = max(
                    max_reference_diff,
                    abs((new_psnr - base_psnr) - float(reference[(operator, name)])),
                )
                counts[operator] += 1
    passed = (
        all(counts[operator] == args.smoke_sample_count for operator in OPERATORS)
        and max_delta == 0.0
        and max_prediction_diff == 0.0
        and max_reference_diff <= args.replay_tolerance_db
    )
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": "v3s-S0-exact-noop-smoke",
        "state": "COMPLETED_GATE_PASS" if passed else "COMPLETED_GATE_FAIL",
        "gate_type": "structural_integrity",
        "decision": (
            "V3S_S0_EXACT_NOOP_PASS_AUTHORIZE_SCOUT_ONLY"
            if passed
            else "V3S_S0_EXACT_NOOP_FAIL_STOP"
        ),
        "metric_contract": "zero DIRR head must exactly reproduce frozen v3p .125 replay",
        "authorizes": "v3s-S1 fixed-32 trainability scout only" if passed else "none",
        "reason": "zero-init Delta-u branch checked on frozen OOF names",
        "counts": dict(counts),
        "max_abs_delta": max_delta,
        "max_abs_prediction_diff": max_prediction_diff,
        "max_abs_reference_psnr_delta_diff_db": max_reference_diff,
        "replay_tolerance_db": args.replay_tolerance_db,
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    write_json(Path(output_dir) / f"{args.run_tag}_closeout.json", closeout)
    return closeout


def scout_train(args, legacy, frozen, names, folds, device, output_dir):
    scout_names = names[:args.scout_sample_count]
    checkpoint = Path(output_dir) / f"{args.run_tag}_checkpoint_cloud_only.pt"
    model, history, initial_render = train_delta_model(
        args,
        legacy,
        frozen,
        scout_names,
        folds,
        device,
        checkpoint,
        args.scout_steps,
        args.run_tag,
    )
    final = history[-1]
    delta_abs = final["delta_abs"]
    passed = (
        math.isfinite(final["total"])
        and math.isfinite(final["gradient_norm"])
        and delta_abs > args.scout_min_delta_abs
        and final["render"] < initial_render
    )
    write_rows(Path(output_dir) / f"{args.run_tag}_history.csv", history)
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": "v3s-S1-fixed32-trainability-scout",
        "state": "COMPLETED_GATE_PASS" if passed else "COMPLETED_GATE_FAIL",
        "gate_type": "structural_integrity",
        "decision": (
            "V3S_S1_TRAINABILITY_PASS_AUTHORIZE_FORMAL_TRAIN_ONLY"
            if passed
            else "V3S_S1_TRAINABILITY_FAIL_STOP"
        ),
        "metric_contract": "finite real-rendered loss and nonzero bounded Delta-u on fixed 32 OOF names",
        "authorizes": "v3s-S2 frozen five-fold training only" if passed else "none",
        "reason": "scout is a numerical activity check, not a utility claim",
        "initial_render_loss": initial_render,
        "final": final,
        "checkpoint_cloud_only": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": True,
    }
    write_json(Path(output_dir) / f"{args.run_tag}_closeout.json", closeout)
    return closeout


def formal_train(args, legacy, frozen, names, folds, device, output_dir):
    root = Path(output_dir)
    all_history = []
    checkpoints = []
    for held_out_fold in range(args.fold_count):
        train_names = [name for name in names if folds[name] != held_out_fold]
        if len(train_names) != args.formal_train_per_fold:
            raise RuntimeError("formal train-fold size changed from the route contract")
        label = f"fold{held_out_fold}"
        checkpoint = root / "checkpoints_cloud_only" / f"{args.run_tag}_{label}.pt"
        _, history, _ = train_delta_model(
            args,
            legacy,
            frozen,
            train_names,
            folds,
            device,
            checkpoint,
            args.formal_epochs,
            label,
        )
        all_history.extend(history)
        checkpoints.append(
            {
                "fold": held_out_fold,
                "checkpoint_cloud_only": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "train_images": len(train_names),
            }
        )
    write_rows(root / f"{args.run_tag}_history.csv", all_history)
    manifest = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": "v3s-S2-five-fold-train",
        "state": "COMPLETED_GATE_PASS",
        "gate_type": "structural_integrity",
        "decision": "V3S_S2_TRAIN_COMPLETE_AUTHORIZE_FORMAL_OOF_EVAL_ONLY",
        "authorizes": "v3s-S3 canonical float64 OOF evaluation only",
        "fold_count": args.fold_count,
        "formal_train_per_fold": args.formal_train_per_fold,
        "formal_epochs": args.formal_epochs,
        "checkpoints": checkpoints,
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": True,
    }
    write_json(root / f"{args.run_tag}_closeout.json", manifest)
    return manifest


def formal_eval(args, legacy, frozen, names, folds, device, output_dir):
    root = Path(output_dir)
    checkpoint_manifest = read_json(args.train_closeout)
    if checkpoint_manifest.get("decision") != "V3S_S2_TRAIN_COMPLETE_AUTHORIZE_FORMAL_OOF_EVAL_ONLY":
        raise RuntimeError("formal evaluation requires the typed five-fold training closeout")
    checkpoints = {int(row["fold"]): row for row in checkpoint_manifest["checkpoints"]}
    if set(checkpoints) != set(range(args.fold_count)):
        raise RuntimeError("formal training checkpoint set is incomplete")
    models = {}
    for fold, entry in checkpoints.items():
        path = Path(entry["checkpoint_cloud_only"])
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"formal checkpoint hash mismatch for fold {fold}")
        models[fold], _ = load_direction_checkpoint(path, device)

    rows = []
    block_rows = []
    with torch.no_grad():
        for index, name in enumerate(names):
            fold = folds[name]
            sample = frozen_sample(args, legacy, frozen, name, fold, device)
            model = models[fold]
            for operator in OPERATORS:
                delta = model(sample["hazy"], sample["base"], sample["steps"][operator], sample["support"])
                old_low, old_high, new_low, new_high = candidate_predictions(
                    sample["base"], sample["steps"][operator], delta
                )
                old_low_mse = float(per_image_mse(old_low, sample["label"]).item())
                old_high_mse = float(per_image_mse(old_high, sample["label"]).item())
                new_high_mse = float(per_image_mse(new_high, sample["label"]).item())
                block = canonical_block_metrics(
                    sample["base"],
                    sample["steps"][operator],
                    delta,
                    old_low,
                    old_high,
                    new_low,
                    new_high,
                    sample["label"],
                    sample["support"],
                )
                row = {
                    "operator_label": operator,
                    "fold": fold,
                    "index": index,
                    "name": name,
                    "clean_reference_group": clean_reference_group(name),
                    "old_low_psnr": psnr_from_mse(old_low_mse),
                    "old_high_psnr": psnr_from_mse(old_high_mse),
                    "new_high_psnr": psnr_from_mse(new_high_mse),
                    "lift_vs_old_low_db": psnr_from_mse(new_high_mse) - psnr_from_mse(old_low_mse),
                    "lift_vs_old_high_db": psnr_from_mse(new_high_mse) - psnr_from_mse(old_high_mse),
                    "harm_mse_vs_old_low": max(new_high_mse - old_low_mse, 0.0),
                    "old_step_energy": float(torch.mean(sample["steps"][operator].square()).item()),
                    "delta_abs": float(torch.mean(torch.abs(delta)).item()),
                    **block,
                }
                rows.append(row)
                block_rows.append(
                    {
                        "operator_label": operator,
                        "fold": fold,
                        "name": name,
                        "active_blocks": block["active_blocks"],
                        "old_wrong_blocks": block["old_wrong_blocks"],
                        "new_wrong_blocks": block["new_wrong_blocks"],
                        "repaired_wrong_blocks": block["repaired_wrong_blocks"],
                        "mean_g1_improvement_per_pixel": block["mean_g1_improvement_per_pixel"],
                        "aggregation_abs_error": block["aggregation_abs_error"],
                    }
                )
            if (index + 1) % args.progress_every == 0:
                print(f"{args.run_tag}_eval_{index + 1}/{len(names)}", flush=True)

    write_rows(root / f"{args.run_tag}_image_rows_cloud_only.csv", rows)
    write_rows(root / f"{args.run_tag}_block_summary_cloud_only.csv", block_rows)
    summaries = []
    decisions = []
    for operator_index, operator in enumerate(OPERATORS):
        selected = [row for row in rows if row["operator_label"] == operator]
        if len(selected) != args.formal_sample_count:
            raise RuntimeError(f"formal OOF coverage mismatch for {operator}")
        lcb_high, group_count = bootstrap_interval(
            selected, "lift_vs_old_high_db", 0.05, args.seed + operator_index
        )
        lcb_low, _ = bootstrap_interval(
            selected, "lift_vs_old_low_db", 0.05, args.seed + 20 + operator_index
        )
        harm_ucb, _ = bootstrap_interval(
            selected, "harm_mse_vs_old_low", 0.05, args.seed + 40 + operator_index, upper=True
        )
        low_energy_threshold = float(np.quantile([row["old_step_energy"] for row in selected], 0.25))
        low_action = [row for row in selected if row["old_step_energy"] <= low_energy_threshold]
        low_action_lcb, _ = bootstrap_interval(
            low_action, "lift_vs_old_low_db", 0.05, args.seed + 60 + operator_index
        )
        old_wrong = sum(row["old_wrong_blocks"] for row in selected)
        repaired = sum(row["repaired_wrong_blocks"] for row in selected)
        severe = sum(row["lift_vs_old_low_db"] <= SEVERE_DB for row in selected)
        summary = {
            "operator_label": operator,
            "image_count": len(selected),
            "group_count": group_count,
            "mean_lift_vs_old_high_db": float(np.mean([row["lift_vs_old_high_db"] for row in selected])),
            "lcb95_lift_vs_old_high_db": lcb_high,
            "mean_lift_vs_old_low_db": float(np.mean([row["lift_vs_old_low_db"] for row in selected])),
            "lcb95_lift_vs_old_low_db": lcb_low,
            "p05_lift_vs_old_low_db": percentile([row["lift_vs_old_low_db"] for row in selected], 0.05),
            "cvar5_lift_vs_old_low_db": cvar_lower([row["lift_vs_old_low_db"] for row in selected], 0.05),
            "severe_vs_old_low_count": severe,
            "harm_mse_mean": float(np.mean([row["harm_mse_vs_old_low"] for row in selected])),
            "harm_mse_ucb95": harm_ucb,
            "low_action_energy_threshold": low_energy_threshold,
            "low_action_image_count": len(low_action),
            "low_action_lcb95_lift_vs_old_low_db": low_action_lcb,
            "old_wrong_blocks": old_wrong,
            "repaired_wrong_blocks": repaired,
            "repaired_wrong_fraction": repaired / old_wrong if old_wrong else 0.0,
            "new_wrong_blocks": sum(row["new_wrong_blocks"] for row in selected),
            "mean_g1_improvement_per_pixel": float(
                np.mean([row["mean_g1_improvement_per_pixel"] for row in selected])
            ),
            "max_aggregation_abs_error": max(row["aggregation_abs_error"] for row in selected),
            "mean_delta_abs": float(np.mean([row["delta_abs"] for row in selected])),
        }
        summaries.append(summary)
        decisions.append(
            summary["lcb95_lift_vs_old_high_db"] >= args.utility_sesoi_db
            and summary["lcb95_lift_vs_old_low_db"] >= -args.anchor_lcb_floor_db
            and summary["low_action_lcb95_lift_vs_old_low_db"] >= -args.anchor_lcb_floor_db
            and summary["severe_vs_old_low_count"] == 0
            and summary["repaired_wrong_fraction"] >= args.min_wrong_direction_repair_fraction
            and summary["mean_g1_improvement_per_pixel"] > 0.0
            and summary["max_aggregation_abs_error"] <= args.canonical_aggregation_tolerance
        )
    write_rows(root / f"{args.run_tag}_operator_summary.csv", summaries)
    canonical_contract = {
        "route_id": ROUTE_ID,
        "stage": "v3s-S3-canonical-float64-G1",
        "block_size": BLOCK_SIZE,
        "atol_sse": CANONICAL_ATOL_SSE,
        "rtol": CANONICAL_RTOL,
        "epsilon_g": "2 * (atol_sse + rtol * max(abs(L_.125), abs(L_.25)))",
        "loss_dtype": "float64",
        "analysis": "new frozen operator only; v3p/v3r labels are not reused",
        "operator_summaries": summaries,
    }
    write_json(root / f"{args.run_tag}_canonical_g1_contract.json", canonical_contract)
    passed = all(decisions)
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": "v3s-S3-canonical-float64-OOF-evaluation",
        "state": "COMPLETED_GATE_PASS" if passed else "COMPLETED_GATE_FAIL",
        "gate_type": "scientific_utility_and_safety",
        "decision": (
            "V3S_S3_DIRECTION_REPAIR_PASS_AUTHORIZE_NEW_OPERATOR_CONFIRMATION_DESIGN_ONLY"
            if passed
            else "V3S_S3_DIRECTION_REPAIR_FAIL_STOP_THIS_LOW_CAPACITY_CONTRACT"
        ),
        "metric_contract": "five-fold OOF fixed alpha=.25 Delta-u operator versus frozen old .125/.25; canonical float64 G1",
        "authorizes": "new fixed-operator confirmation contract design only" if passed else "none",
        "reason": "dual-operator fixed-action utility, signed-direction repair, and low-action protection gates",
        "gates": {
            "utility_sesoi_db": args.utility_sesoi_db,
            "anchor_lcb_floor_db": -args.anchor_lcb_floor_db,
            "severe_db": SEVERE_DB,
            "min_wrong_direction_repair_fraction": args.min_wrong_direction_repair_fraction,
            "canonical_aggregation_tolerance": args.canonical_aggregation_tolerance,
        },
        "operator_summaries": summaries,
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": True,
    }
    write_json(root / f"{args.run_tag}_closeout.json", closeout)
    return closeout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("noop_smoke", "scout_train", "formal_train", "formal_eval"))
    parser.add_argument("--legacy_repo", required=True)
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fresh_split_manifest", required=True)
    parser.add_argument("--v3j_a_bounds", required=True)
    parser.add_argument("--operator_artifact_manifest", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--reference_oof_rows", required=True)
    parser.add_argument("--expected_a0_checkpoint_sha256", required=True)
    parser.add_argument("--expected_control_checkpoint_sha256", required=True)
    parser.add_argument("--expected_density_artifact_sha256", required=True)
    parser.add_argument("--expected_d7c_artifact_sha256", required=True)
    parser.add_argument("--expected_fresh_split_manifest_sha256", required=True)
    parser.add_argument("--expected_operator_manifest_sha256", required=True)
    parser.add_argument("--expected_reference_oof_rows_sha256", required=True)
    parser.add_argument("--expected_v3j_a_bounds_sha256", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--train_closeout", default="")
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--train_key", default="v3j_controller_train")
    parser.add_argument("--formal_sample_count", type=int, default=1200)
    parser.add_argument("--smoke_sample_count", type=int, default=32)
    parser.add_argument("--scout_sample_count", type=int, default=32)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--formal_train_per_fold", type=int, default=960)
    parser.add_argument("--proj_channels", type=int, default=24)
    parser.add_argument("--dirr_channels", type=int, default=24)
    parser.add_argument("--d7c_threshold", type=float, default=0.5773006677627563)
    parser.add_argument("--delta_bound_multiplier", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--grad_clip_norm", type=float, default=0.1)
    parser.add_argument("--risk_window", type=int, default=4)
    parser.add_argument("--formal_epochs", type=int, default=6)
    parser.add_argument("--scout_steps", type=int, default=8)
    parser.add_argument("--anchor_weight", type=float, default=30.0)
    parser.add_argument("--margin_weight", type=float, default=5.0)
    parser.add_argument("--harm_weight", type=float, default=20.0)
    parser.add_argument("--cvar_weight", type=float, default=40.0)
    parser.add_argument("--repair_weight", type=float, default=0.02)
    parser.add_argument("--cvar_fraction", type=float, default=0.25)
    parser.add_argument("--scout_min_delta_abs", type=float, default=0.000001)
    parser.add_argument("--utility_sesoi_db", type=float, default=0.02)
    parser.add_argument("--anchor_lcb_floor_db", type=float, default=0.005)
    parser.add_argument("--min_wrong_direction_repair_fraction", type=float, default=0.20)
    parser.add_argument("--canonical_aggregation_tolerance", type=float, default=1e-8)
    parser.add_argument("--replay_tolerance_db", type=float, default=1e-6)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected_legacy_commit", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source_split.lower() != "train":
        raise ValueError("v3s is train-derived OOF only")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v3s requires the authorized CUDA runtime")
    if not 0.0 < args.cvar_fraction <= 1.0:
        raise ValueError("cvar_fraction must be in (0, 1]")
    if args.risk_window <= 0 or args.scout_steps <= 0 or args.formal_epochs <= 0:
        raise ValueError("training budgets must be positive")
    legacy_repo = Path(args.legacy_repo)
    legacy_commit = git_output(legacy_repo, "rev-parse", "HEAD")
    if legacy_commit != args.expected_legacy_commit:
        raise RuntimeError(f"legacy source commit mismatch: {legacy_commit}")
    if git_output(legacy_repo, "status", "--porcelain"):
        raise RuntimeError("legacy source checkout must be clean")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    set_determinism(args.seed)
    legacy = import_legacy_modules(legacy_repo)
    assets = asset_manifest(args)
    device = torch.device(args.device)
    frozen = build_frozen_operator(args, legacy, device)
    args.delta_bound = tuple(args.delta_bound_multiplier * value for value in frozen["bound"])
    names, folds = load_names_and_folds(args, legacy)
    source = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "mode": args.mode,
        "legacy_repo": str(legacy_repo),
        "legacy_commit": legacy_commit,
        "assets": assets,
        "frozen_operator_artifacts": frozen["artifacts"],
        "delta_bound": list(args.delta_bound),
        "names": len(names),
        "fold_counts": {str(fold): sum(value == fold for value in folds.values()) for fold in range(args.fold_count)},
        "locked_test_touched": False,
        "canary_touched": False,
    }
    write_json(output_dir / f"{args.run_tag}_source_manifest.json", source)
    if args.mode == "noop_smoke":
        closeout = no_op_smoke(args, legacy, frozen, names, folds, device, output_dir)
    elif args.mode == "scout_train":
        closeout = scout_train(args, legacy, frozen, names, folds, device, output_dir)
    elif args.mode == "formal_train":
        closeout = formal_train(args, legacy, frozen, names, folds, device, output_dir)
    else:
        if not args.train_closeout:
            raise ValueError("formal_eval requires --train_closeout")
        closeout = formal_eval(args, legacy, frozen, names, folds, device, output_dir)
    print(json.dumps(closeout, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
