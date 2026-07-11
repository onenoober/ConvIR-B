#!/usr/bin/env python3
"""v3j-B direct bounded residual OOF diagnostic.

This diagnostic asks whether tiny heads can directly predict a bounded output
residual supervised by the v3j-A primary safe teacher. It trains no backbone,
does not update W_U, and does not save probe weights.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chd_rm_v3i_a_teacher_compressibility_audit import (  # noqa: E402
    D7C_THRESHOLD,
    action_gate_from_full,
    action_shape_for_input,
    build_gate_producer,
    build_model,
    forward_final,
    load_pair,
    metric_pair,
    pad_to_factor,
    percentile,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)
from chd_rm_v3i_b_full_context_probe import full_context_maps  # noqa: E402
from chd_rm_v3j_a_bounded_action_audit import (  # noqa: E402
    FIXED_25PCT_ORACLE_GAP_LINE,
    FIXED_UNGATED_MEAN_LINE,
    FIXED_V3I_HARD_P10_LINE,
    ROUTE_ID,
    bootstrap_delta,
    names_from_manifest,
    output_gate_from_action_gate,
    summarize_policy,
    teacher_policy_outputs,
)


def clean_id(name):
    return Path(name).stem.split("_")[0]


def fold_assignments(names, fold_count):
    ids = sorted({clean_id(name) for name in names})
    id_to_fold = {cid: idx % fold_count for idx, cid in enumerate(ids)}
    return np.asarray([id_to_fold[clean_id(name)] for name in names], dtype=np.int64), id_to_fold


def bounded_residual(raw, bound):
    bound_t = torch.as_tensor(bound, device=raw.device, dtype=raw.dtype).view(1, 3, 1, 1)
    return bound_t * torch.tanh(raw / bound_t.clamp_min(1e-12))


def bounded_residual_samples(raw, bound):
    bound_t = torch.as_tensor(bound, device=raw.device, dtype=raw.dtype).view(1, 3)
    return bound_t * torch.tanh(raw / bound_t.clamp_min(1e-12))


def target_low_from_teacher(base_pred, teacher_pred, hard_gate, bound):
    target = teacher_pred - base_pred
    target_low = F.interpolate(target, size=hard_gate.shape[-2:], mode="bilinear", align_corners=False)
    bound_t = torch.as_tensor(bound, device=target_low.device, dtype=target_low.dtype).view(1, 3, 1, 1)
    target_low = torch.maximum(torch.minimum(target_low, bound_t), -bound_t)
    return target_low * (hard_gate > 0.5).to(target_low.dtype)


def replay_prediction(base_pred, hard_gate, pred_low):
    up = F.interpolate(pred_low, size=base_pred.shape[-2:], mode="bilinear", align_corners=False)
    gate = output_gate_from_action_gate(hard_gate, base_pred.shape[-2:])
    return torch.clamp(base_pred + gate * up, 0, 1)


def extract_samples(fmap, hard_gate, target_low, active_count, inactive_count, rng):
    height, width = hard_gate.shape[-2:]
    active = torch.nonzero((hard_gate > 0.5).reshape(-1), as_tuple=False).reshape(-1)
    inactive = torch.nonzero((hard_gate <= 0.5).reshape(-1), as_tuple=False).reshape(-1)
    chosen = []
    active_flags = []
    if int(active.numel()) > 0 and active_count > 0:
        pick = rng.choice(active.detach().cpu().numpy(), size=min(active_count, int(active.numel())), replace=False)
        chosen.extend(int(v) for v in pick)
        active_flags.extend([1] * len(pick))
    if int(inactive.numel()) > 0 and inactive_count > 0:
        pick = rng.choice(inactive.detach().cpu().numpy(), size=min(inactive_count, int(inactive.numel())), replace=False)
        chosen.extend(int(v) for v in pick)
        active_flags.extend([0] * len(pick))
    if not chosen:
        return None
    chosen_t = torch.as_tensor(chosen, device=fmap.device, dtype=torch.long)
    ys = torch.div(chosen_t, width, rounding_mode="floor")
    xs = chosen_t % width
    padded_feature = F.pad(fmap, (1, 1, 1, 1), mode="replicate")
    patches = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        patches.append(padded_feature[0, :, y : y + 3, x : x + 3].detach().cpu().numpy())
    patch_np = np.stack(patches).astype(np.float16)
    center_np = patch_np[:, :, 1, 1].astype(np.float32)
    target = target_low.reshape(3, -1).transpose(0, 1)[chosen_t].detach().cpu().numpy().astype(np.float32)
    return patch_np, center_np, target, np.asarray(active_flags, dtype=np.float32)


class LinearResidualHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Linear(channels, 3)

    def forward(self, x):
        return self.linear(x)

    def forward_map(self, fmap):
        weight = self.linear.weight[:, :, None, None]
        return F.conv2d(fmap, weight, self.linear.bias)


class ContextResidualHead(nn.Module):
    def __init__(self, channels, proj_channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, proj_channels, 1)
        self.depthwise = nn.Conv2d(proj_channels, proj_channels, 3, padding=1, groups=proj_channels)
        self.pointwise = nn.Conv2d(proj_channels, 3, 1)

    def forward(self, patch):
        out = self.pointwise(self.depthwise(self.proj(patch)))
        return out[:, :, 1, 1]

    def forward_map(self, fmap):
        return self.pointwise(self.depthwise(self.proj(fmap)))


def normalize_center(x, mean, std):
    return (x - mean) / std


def normalize_patch(x, mean, std):
    return (x - mean[:, None, None]) / std[:, None, None]


def train_head(kind, arrays, train_mask, args, device, fold_label):
    rng = np.random.default_rng(args.seed + int(fold_label) * 1009 + {"linear": 11, "context": 13}[kind])
    center = arrays["center"]
    mean = center[train_mask].mean(axis=0).astype(np.float32)
    std = center[train_mask].std(axis=0).astype(np.float32) + 1e-6
    channels = center.shape[1]
    if kind == "linear":
        x_all = center
        model = LinearResidualHead(channels).to(device)
        steps = args.linear_steps
    elif kind == "context":
        x_all = arrays["patch"]
        model = ContextResidualHead(channels, args.proj_channels).to(device)
        steps = args.context_steps
    else:
        raise ValueError(kind)
    y_all = arrays["target"]
    active_all = arrays["active"]
    train_indices = np.flatnonzero(train_mask)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bound = json.loads(Path(args.v3j_a_bounds).read_text(encoding="utf-8"))["channel_bounds_rgb"]
    model.train()
    history = []
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    for step in range(steps):
        batch_idx = rng.choice(train_indices, size=min(args.batch_size, len(train_indices)), replace=len(train_indices) < args.batch_size)
        y = torch.from_numpy(y_all[batch_idx]).to(device=device, dtype=torch.float32)
        active = torch.from_numpy(active_all[batch_idx]).to(device=device, dtype=torch.float32)
        if kind == "context":
            xb = torch.from_numpy(x_all[batch_idx].astype(np.float32)).to(device=device)
            xb = normalize_patch(xb, mean_t, std_t)
        else:
            xb = torch.from_numpy(x_all[batch_idx].astype(np.float32)).to(device=device)
            xb = normalize_center(xb, mean_t, std_t)
        pred = bounded_residual_samples(model(xb), bound)
        per = F.smooth_l1_loss(pred, y, reduction="none").mean(dim=1)
        weights = torch.where(active > 0.5, torch.ones_like(active), torch.full_like(active, args.lambda_noop))
        loss = (per * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        if (step + 1) % max(1, args.log_every) == 0 or step == steps - 1:
            history.append({"fold": fold_label, "head": kind, "step": step + 1, "loss": float(loss.item())})
    return model.eval(), mean, std, history


def score_map(kind, model, fmap, mean, std, bound):
    mean_t = torch.from_numpy(mean).to(device=fmap.device, dtype=fmap.dtype)
    std_t = torch.from_numpy(std).to(device=fmap.device, dtype=fmap.dtype)
    with torch.no_grad():
        x = normalize_patch(fmap, mean_t, std_t)
        return bounded_residual(model.forward_map(x), bound)


def make_arrays(args, names, folds_by_image, base, action_model, gate_producer, device):
    rng = np.random.default_rng(args.seed + 17)
    patches = []
    centers = []
    targets = []
    actives = []
    image_indices = []
    folds = []
    bound = json.loads(Path(args.v3j_a_bounds).read_text(encoding="utf-8"))["channel_bounds_rgb"]
    for index, name in enumerate(names):
        input_img, label = load_pair(args.data_dir, args.source_split, name)
        input_img = input_img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        padded, height, width = pad_to_factor(input_img)
        label = label[:, :, :height, :width]
        action_shape = action_shape_for_input(padded)
        with torch.no_grad():
            base_pred = forward_final(base, padded, height, width)
            gate_full, _, _ = gate_producer(padded)
            hard_gate = action_gate_from_full(gate_full, action_shape).to(device)
            fmap, _, _, _ = full_context_maps(action_model, gate_producer, padded)
        alphas, _ = teacher_policy_outputs(action_model, padded, label, hard_gate, height, width, args)
        with torch.no_grad():
            teacher_pred = forward_final(action_model, padded, height, width, d7c_gate=hard_gate * alphas[args.teacher_policy])
            target_low = target_low_from_teacher(base_pred, teacher_pred, hard_gate, bound)
        sampled = extract_samples(fmap, hard_gate, target_low, args.active_sample_per_image, args.inactive_sample_per_image, rng)
        if sampled is not None:
            patch_np, center_np, target_np, active_np = sampled
            patches.append(patch_np)
            centers.append(center_np)
            targets.append(target_np)
            actives.append(active_np)
            image_indices.append(np.full(len(target_np), index, dtype=np.int32))
            folds.append(np.full(len(target_np), folds_by_image[index], dtype=np.int16))
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"v3j_b_extract_progress {index + 1}/{len(names)}", flush=True)
        del input_img, label, padded, base_pred, hard_gate, fmap, alphas
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "patch": np.concatenate(patches, axis=0),
        "center": np.concatenate(centers, axis=0),
        "target": np.concatenate(targets, axis=0),
        "active": np.concatenate(actives, axis=0),
        "image_index": np.concatenate(image_indices, axis=0),
        "fold": np.concatenate(folds, axis=0),
    }


def replay_names(args, names, policy_prefix, fold_models, folds_by_image, base, action_model, gate_producer, device):
    rows = []
    bound = json.loads(Path(args.v3j_a_bounds).read_text(encoding="utf-8"))["channel_bounds_rgb"]
    for index, name in enumerate(names):
        input_img, label = load_pair(args.data_dir, args.source_split, name)
        input_img = input_img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        padded, height, width = pad_to_factor(input_img)
        label = label[:, :, :height, :width]
        action_shape = action_shape_for_input(padded)
        with torch.no_grad():
            base_pred = forward_final(base, padded, height, width)
            base_mse, base_psnr = metric_pair(base_pred, label)
            gate_full, _, _ = gate_producer(padded)
            hard_gate = action_gate_from_full(gate_full, action_shape).to(device)
            hard_pred = forward_final(action_model, padded, height, width, d7c_gate=hard_gate)
            _, hard_psnr = metric_pair(hard_pred, label)
            fmap, _, _, _ = full_context_maps(action_model, gate_producer, padded)
            if policy_prefix == "OOF":
                model_pack = fold_models[folds_by_image[index]]
            else:
                model_pack = fold_models["final"]
            rows.append({"split": policy_prefix, "fold": folds_by_image[index], "index": index, "name": name, "policy": f"{policy_prefix}_A0", "psnr_delta": 0.0})
            rows.append(
                {
                    "split": policy_prefix,
                    "fold": folds_by_image[index],
                    "index": index,
                    "name": name,
                    "policy": f"{policy_prefix}_HARD_D7C_ALPHA1",
                    "psnr_delta": hard_psnr - base_psnr,
                }
            )
            for kind, (model, mean, std) in model_pack.items():
                pred_low = score_map(kind, model, fmap, mean, std, bound)
                pred = replay_prediction(base_pred, hard_gate, pred_low)
                _, psnr = metric_pair(pred, label)
                rows.append(
                    {
                        "split": policy_prefix,
                        "fold": folds_by_image[index],
                        "index": index,
                        "name": name,
                        "policy": f"{policy_prefix}_DIRECT_{kind.upper()}",
                        "psnr_delta": psnr - base_psnr,
                    }
                )
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"v3j_b_replay_{policy_prefix.lower()}_progress {index + 1}/{len(names)}", flush=True)
        del input_img, label, padded, base_pred, hard_gate, fmap
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def summarize_and_gate(rows, prefix, args):
    summary = summarize_policy(rows)
    by_policy = {row["policy"]: row for row in summary}
    hard_key = f"{prefix}_HARD_D7C_ALPHA1"
    hard = by_policy[hard_key]
    hard_by_name = {row["name"]: float(row["psnr_delta"]) for row in rows if row["policy"] == hard_key}
    bootstrap_rows = []
    pass_policies = []
    strong_policies = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    for policy in sorted(grouped):
        if not policy.startswith(f"{prefix}_DIRECT_"):
            continue
        boot = bootstrap_delta(grouped[policy], hard_by_name, args.seed + 707, args.bootstrap_draws)
        boot["policy"] = policy
        bootstrap_rows.append(boot)
        row = by_policy[policy]
        min_pass = (
            boot["ci95_low"] > 0
            and row["regression_le_0p2_count"] <= hard["regression_le_0p2_count"]
            and row["p10_psnr_delta"] >= hard["p10_psnr_delta"]
            and row["p10_psnr_delta"] >= FIXED_V3I_HARD_P10_LINE
        )
        strong = (
            min_pass
            and row["mean_psnr_delta"] > FIXED_UNGATED_MEAN_LINE
            and row["mean_psnr_delta"] >= FIXED_25PCT_ORACLE_GAP_LINE
        )
        if min_pass:
            pass_policies.append(policy)
        if strong:
            strong_policies.append(policy)
    return summary, bootstrap_rows, pass_policies, strong_policies


def matching_direct_heads(oof_policies, confirm_policies):
    def heads(prefix, policies):
        marker = f"{prefix}_DIRECT_"
        return {policy.split(marker, 1)[1] for policy in policies if policy.startswith(marker)}

    return sorted(heads("OOF", oof_policies) & heads("CONFIRM", confirm_policies))


def validate_stage_authorization(args):
    if args.source_split.lower() != "train":
        raise ValueError(f"v3j-B is train-derived only; got source_split={args.source_split!r}")
    for key in (args.train_key, args.confirm_key):
        if "test" in key.lower():
            raise ValueError(f"locked-test-like split key is forbidden for v3j-B: {key!r}")
    summary = read_json(args.v3j_a_summary)
    decision = summary.get("decision")
    expected = "V3J_BOUNDED_ACTION_SPACE_PASS_AUTHORIZE_DIRECT_CORRECTION_OOF_ONLY"
    if decision != expected:
        raise RuntimeError(f"v3j-A did not authorize v3j-B: expected {expected}, got {decision}")
    return summary


def audit(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "direct_correction_probe_summary.json"
    if summary_path.exists() and not args.allow_overwrite:
        raise FileExistsError(f"refusing to overwrite existing v3j-B summary: {summary_path}")
    v3j_a_summary = validate_stage_authorization(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    manifest = read_json(args.fresh_split_manifest)
    train_names = names_from_manifest(manifest, args.train_key, args.max_train_samples)
    confirm_names = names_from_manifest(manifest, args.confirm_key, args.max_confirm_samples)
    train_folds, id_to_fold = fold_assignments(train_names, args.fold_count)
    confirm_folds = np.zeros(len(confirm_names), dtype=np.int64)
    base = build_model("original", args.a0_checkpoint, device)
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    arrays = make_arrays(args, train_names, train_folds, base, action_model, gate_producer, device)
    histories = []
    fold_models = {}
    for fold in range(args.fold_count):
        train_mask = arrays["fold"] != fold
        fold_models[fold] = {}
        for kind in ("linear", "context"):
            model, mean, std, history = train_head(kind, arrays, train_mask, args, device, fold)
            fold_models[fold][kind] = (model, mean, std)
            histories.extend(history)
        print(f"v3j_b_fold_done {fold}", flush=True)
    final_models = {"final": {}}
    all_mask = np.ones(arrays["target"].shape[0], dtype=bool)
    for kind in ("linear", "context"):
        model, mean, std, history = train_head(kind, arrays, all_mask, args, device, 99)
        final_models["final"][kind] = (model, mean, std)
        histories.extend(history)
    oof_rows = replay_names(args, train_names, "OOF", fold_models, train_folds, base, action_model, gate_producer, device)
    confirm_rows = replay_names(args, confirm_names, "CONFIRM", final_models, confirm_folds, base, action_model, gate_producer, device)
    oof_summary, oof_boot, oof_pass, oof_strong = summarize_and_gate(oof_rows, "OOF", args)
    confirm_summary, confirm_boot, confirm_pass, confirm_strong = summarize_and_gate(confirm_rows, "CONFIRM", args)
    dual_pass_heads = matching_direct_heads(oof_pass, confirm_pass)
    dual_strong_heads = matching_direct_heads(oof_strong, confirm_strong)
    decision = (
        "V3J_DIRECT_SAFE_CORRECTION_OOF_PASS_AUTHORIZE_NOOP_ARCH_EQUIVALENCE_ONLY"
        if dual_pass_heads
        else "V3J_DIRECT_SAFE_CORRECTION_OOF_FAIL_REQUIRE_NEW_INFORMATION_NO_INTERNAL_ROUTER"
    )
    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3j-B direct bounded residual OOF diagnostic",
        "decision": decision,
        "training_authorized": False,
        "saved_probe_weights": False,
        "locked_test_touched": False,
        "direct_confirm_pass_policies": confirm_pass,
        "direct_confirm_strong_policies": confirm_strong,
        "direct_oof_pass_policies": oof_pass,
        "direct_oof_strong_policies": oof_strong,
        "direct_dual_pass_heads": dual_pass_heads,
        "direct_dual_strong_heads": dual_strong_heads,
        "sample_counts": {
            "controller_train_images": len(train_names),
            "route_confirm_images": len(confirm_names),
            "sampled_training_sites": int(arrays["target"].shape[0]),
            "active_sample_fraction": float(np.mean(arrays["active"] > 0.5)),
            "clean_reference_count": len(id_to_fold),
        },
        "feature_manifest": {
            "feature_channels": int(arrays["center"].shape[1]),
            "head_linear": "1x1 linear projection to 3-channel bounded residual",
            "head_context": "1x1 projection -> depthwise 3x3 -> pointwise 1x1 -> 3-channel bounded residual",
            "raw_feature_tensors_saved": False,
            "probe_weights_saved": False,
        },
        "metric_contract": {
            "target": f"bounded low-resolution residual from {args.teacher_policy} teacher",
            "replay": "A0 + upsampled bounded residual under D7c output veto",
            "minimum_pass": "paired mean delta vs hard D7c CI95 low > 0, severe regressions <= hard, p10 >= hard and fixed v3i hard p10 line",
            "strong_pass": "minimum pass plus mean > fixed ungated line and >= fixed 25% oracle-gap line",
            "fixed_ungated_mean_line": FIXED_UNGATED_MEAN_LINE,
            "fixed_25pct_oracle_gap_line": FIXED_25PCT_ORACLE_GAP_LINE,
        },
        "oof_policy_summary": oof_summary,
        "oof_bootstrap_vs_hard": oof_boot,
        "confirm_policy_summary": confirm_summary,
        "confirm_bootstrap_vs_hard": confirm_boot,
        "a0_checkpoint": args.a0_checkpoint,
        "a0_checkpoint_sha256": sha256_file(args.a0_checkpoint),
        "control_checkpoint": args.control_checkpoint,
        "control_checkpoint_sha256": sha256_file(args.control_checkpoint),
        "v3j_a_summary": args.v3j_a_summary,
        "v3j_a_summary_sha256": sha256_file(args.v3j_a_summary),
        "v3j_a_decision": v3j_a_summary.get("decision"),
    }
    write_csv(output_dir / "direct_correction_probe_training_history.csv", histories)
    write_csv(output_dir / "direct_correction_oof_policy_replay.csv", oof_rows)
    write_csv(output_dir / "direct_correction_oof_policy_summary.csv", oof_summary)
    write_csv(output_dir / "direct_correction_oof_bootstrap_vs_hard.csv", oof_boot)
    write_csv(output_dir / "direct_correction_route_confirm_policy_replay.csv", confirm_rows)
    write_csv(output_dir / "direct_correction_route_confirm_policy_summary.csv", confirm_summary)
    write_csv(output_dir / "direct_correction_route_confirm_bootstrap_vs_hard.csv", confirm_boot)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fresh_split_manifest", required=True)
    parser.add_argument("--v3j_a_summary", required=True)
    parser.add_argument("--v3j_a_bounds", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--train_key", default="v3j_controller_train")
    parser.add_argument("--confirm_key", default="v3j_route_confirm")
    parser.add_argument("--max_train_samples", type=int, default=1200)
    parser.add_argument("--max_confirm_samples", type=int, default=600)
    parser.add_argument("--teacher_policy", default="CC_MIN4_FROM_OPEN_TOP_0.5")
    parser.add_argument("--top_fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--block_sizes", type=int, nargs="*", default=[])
    parser.add_argument("--smooth_kernels", type=int, nargs="*", default=[])
    parser.add_argument("--min_component_sizes", type=int, nargs="+", default=[4])
    parser.add_argument("--interior_kernels", type=int, nargs="*", default=[])
    parser.add_argument("--uniform_alphas", type=float, nargs="*", default=[])
    parser.add_argument("--score_eps", type=float, default=0.0)
    parser.add_argument("--denom_eps", type=float, default=1e-12)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--active_sample_per_image", type=int, default=96)
    parser.add_argument("--inactive_sample_per_image", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--linear_steps", type=int, default=360)
    parser.add_argument("--context_steps", type=int, default=480)
    parser.add_argument("--proj_channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_norm", type=float, default=5.0)
    parser.add_argument("--lambda_noop", type=float, default=1.0)
    parser.add_argument("--bootstrap_draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--d7c_threshold", type=float, default=D7C_THRESHOLD)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--log_every", type=int, default=120)
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
