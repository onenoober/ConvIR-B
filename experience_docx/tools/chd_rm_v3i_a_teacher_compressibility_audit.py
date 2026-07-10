#!/usr/bin/env python3
"""v3i-A no-training FAM2 open-value teacher compressibility audit."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
if str(ITS_ROOT) not in sys.path:
    sys.path.insert(0, str(ITS_ROOT))

from d7c_gate import build_d7c_gate_producer, load_checkpoint_state  # noqa: E402
from models.ConvIR import build_net  # noqa: E402


ROUTE_ID = "haze4k_v5_chd_rm_v3i_fam2_open_value_distillability_20260711"
D7C_THRESHOLD = 0.5773006677627563


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, pct):
    values = sorted(float(v) for v in values)
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def summarize(values):
    values = [float(v) for v in values]
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else float("nan"),
        "std": float(np.std(values)) if values else float("nan"),
        "min": min(values) if values else float("nan"),
        "p05": percentile(values, 5),
        "p10": percentile(values, 10),
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values) if values else float("nan"),
    }


def label_path_for_hazy(data_dir, source_split, hazy_name):
    stem, ext = os.path.splitext(hazy_name)
    candidates = [hazy_name, f"{stem.split('_')[0]}{ext}", f"{stem.split('_')[0]}.png"]
    label_dir = Path(data_dir) / source_split / "gt"
    for candidate in candidates:
        path = label_dir / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"No GT match for {hazy_name}; tried {candidates} under {label_dir}")


def load_pair(data_dir, source_split, hazy_name):
    hazy_path = Path(data_dir) / source_split / "haze" / hazy_name
    label_path = label_path_for_hazy(data_dir, source_split, hazy_name)
    if not hazy_path.is_file():
        raise FileNotFoundError(f"Hazy image missing: {hazy_path}")
    image = Image.open(hazy_path).convert("RGB")
    label = Image.open(label_path).convert("RGB")
    image_t = torch.from_numpy(np.asarray(image).transpose(2, 0, 1)).float() / 255.0
    label_t = torch.from_numpy(np.asarray(label).transpose(2, 0, 1)).float() / 255.0
    return image_t, label_t


def load_names(split_json, split_key, max_samples):
    split = read_json(split_json)
    names = sorted(split["splits"][split_key])
    if max_samples > 0:
        names = names[:max_samples]
    if not names:
        raise ValueError(f"No names in split {split_key}")
    return names


def pad_to_factor(input_img, factor=32):
    height, width = input_img.shape[2], input_img.shape[3]
    pad_h = (factor - height % factor) % factor
    pad_w = (factor - width % factor) % factor
    if pad_h or pad_w:
        input_img = F.pad(input_img, (0, pad_w, 0, pad_h), "reflect")
    return input_img, height, width


def mse_value(pred, label):
    return F.mse_loss(torch.clamp(pred, 0, 1), torch.clamp(label, 0, 1)).item()


def psnr_from_mse(mse):
    return float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def metric_pair(pred, label):
    mse = mse_value(pred, label)
    return mse, psnr_from_mse(mse)


class GateArgs:
    pass


def build_gate_producer(args, device):
    g = GateArgs()
    g.version = "base"
    g.data = "Haze4K"
    g.fam_mode = "fam2_d7c_noop"
    g.d7c_gate_mode = "d7c_fixed"
    g.d7c_base_checkpoint = args.a0_checkpoint
    g.d7c_density_artifact = args.density_artifact
    g.d7c_need_artifact = args.d7c_artifact
    g.d7c_threshold = args.d7c_threshold
    return build_d7c_gate_producer(g, device)


def build_model(fam_mode, checkpoint, device):
    model = build_net("base", "Haze4K", fam_mode).to(device).eval()
    model.load_state_dict(load_checkpoint_state(checkpoint, device), strict=True)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def forward_final(model, padded, height, width, d7c_gate=None):
    if d7c_gate is None:
        out = model(padded)[2]
    else:
        out = model(padded, d7c_gate=d7c_gate)[2]
    return torch.clamp(out[:, :, :height, :width], 0, 1)


def action_shape_for_input(padded):
    return (padded.shape[-2] // 2, padded.shape[-1] // 2)


def action_gate_from_full(gate_full, shape):
    return F.interpolate(gate_full, size=shape, mode="nearest")


def gradient_at_alpha(model, padded, label, hard_action_gate, alpha_value, height, width):
    alpha = torch.full_like(hard_action_gate, float(alpha_value), requires_grad=True)
    model.zero_grad(set_to_none=True)
    pred = forward_final(model, padded, height, width, d7c_gate=hard_action_gate * alpha)
    loss = F.mse_loss(pred, label)
    loss.backward()
    grad = alpha.grad.detach()
    pred_detached = pred.detach()
    loss_value = loss.item()
    del pred, loss, alpha
    return grad, pred_detached, loss_value


def top_fraction_alpha(score, hard_gate, fraction, largest=True, eligible=None):
    alpha = torch.zeros_like(hard_gate)
    active = hard_gate > 0.5
    if eligible is not None:
        active = active & eligible
    count = int(active.sum().item())
    if count == 0 or fraction <= 0:
        return alpha
    keep = max(1, int(round(count * fraction)))
    active_scores = score[active]
    if keep >= count:
        alpha[active] = 1.0
        return alpha
    threshold = torch.topk(active_scores.reshape(-1), keep, largest=largest).values.min()
    if largest:
        alpha[(score >= threshold) & active] = 1.0
    else:
        alpha[(score <= threshold) & active] = 1.0
    return alpha


def normalized_box_mean(value, mask, kernel):
    mask_f = mask.float()
    numerator = F.avg_pool2d(value * mask_f, kernel_size=kernel, stride=1, padding=kernel // 2)
    denominator = F.avg_pool2d(mask_f, kernel_size=kernel, stride=1, padding=kernel // 2)
    return numerator / denominator.clamp_min(1e-12)


def smooth_top_alpha(score, hard_gate, fraction, kernel):
    active = hard_gate > 0.5
    smoothed = normalized_box_mean(score, active, kernel)
    return top_fraction_alpha(smoothed, hard_gate, fraction, largest=True)


def majority_smooth_alpha(alpha, hard_gate, kernel, threshold=0.5):
    active = hard_gate > 0.5
    density = F.avg_pool2d(alpha.float(), kernel_size=kernel, stride=1, padding=kernel // 2)
    return ((density >= threshold) & active).to(alpha.dtype)


def block_top_alpha(score, hard_gate, fraction, block_size):
    active = (hard_gate > 0.5).float()
    total_active = int(active.sum().item())
    alpha = torch.zeros_like(hard_gate)
    if total_active == 0 or fraction <= 0:
        return alpha
    _, _, height, width = score.shape
    crop_h = (height // block_size) * block_size
    crop_w = (width // block_size) * block_size
    if crop_h == 0 or crop_w == 0:
        return top_fraction_alpha(score, hard_gate, fraction, largest=True)
    score_crop = score[..., :crop_h, :crop_w]
    active_crop = active[..., :crop_h, :crop_w]
    score_sum = F.avg_pool2d(score_crop * active_crop, block_size, stride=block_size) * (block_size * block_size)
    active_sum = F.avg_pool2d(active_crop, block_size, stride=block_size) * (block_size * block_size)
    block_score = score_sum / active_sum.clamp_min(1.0)
    flat_active = active_sum.reshape(-1) > 0.5
    if not bool(flat_active.any()):
        return alpha
    flat_score = block_score.reshape(-1)
    flat_mass = active_sum.reshape(-1)
    idx = torch.nonzero(flat_active, as_tuple=False).reshape(-1)
    order = torch.argsort(flat_score[idx], descending=True)
    target_mass = max(1, int(round(total_active * fraction)))
    selected = []
    mass = 0.0
    grid_w = block_score.shape[-1]
    for local in order.tolist():
        flat_idx = int(idx[local].item())
        by = flat_idx // grid_w
        bx = flat_idx % grid_w
        selected.append((by, bx))
        mass += float(flat_mass[flat_idx].item())
        if mass >= target_mass:
            break
    for by, bx in selected:
        y0 = by * block_size
        x0 = bx * block_size
        region = alpha[..., y0 : y0 + block_size, x0 : x0 + block_size]
        region_active = hard_gate[..., y0 : y0 + block_size, x0 : x0 + block_size] > 0.5
        region[region_active] = 1.0
    return alpha


def remove_small_components_alpha(alpha, hard_gate, min_size):
    active = ((alpha > 0.5) & (hard_gate > 0.5))[0, 0].detach().cpu().numpy().astype(bool)
    height, width = active.shape
    keep = np.zeros_like(active, dtype=bool)
    visited = np.zeros_like(active, dtype=bool)
    for y in range(height):
        for x in range(width):
            if not active[y, x] or visited[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            comp = []
            while queue:
                cy, cx = queue.popleft()
                comp.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and active[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(comp) >= min_size:
                for cy, cx in comp:
                    keep[cy, cx] = True
    out = torch.zeros_like(alpha)
    out[0, 0] = torch.from_numpy(keep).to(device=alpha.device, dtype=alpha.dtype)
    return out


def interior_mask(hard_gate, kernel):
    active = (hard_gate > 0.5).float()
    density = F.avg_pool2d(active, kernel_size=kernel, stride=1, padding=kernel // 2)
    return density >= (1.0 - 1e-6)


def alpha_secant_teacher(q0, q1, hard_gate, score_eps, denom_eps):
    active = hard_gate > 0.5
    alpha = torch.zeros_like(q0)
    denom = q0 - q1
    full = active & (q1 >= score_eps)
    mid = active & (q0 > score_eps) & (q1 < score_eps) & (denom.abs() > denom_eps)
    alpha[full] = 1.0
    alpha[mid] = torch.clamp(q0[mid] / denom[mid], 0.0, 1.0)
    ignored = active & (q0 > score_eps) & (q1 < score_eps) & (denom.abs() <= denom_eps)
    near_zero = active & (q0.abs() <= score_eps) & (q1.abs() <= score_eps)
    return alpha, ignored, near_zero


def quantize_alpha(alpha, levels):
    if levels == 3:
        out = torch.zeros_like(alpha)
        out[alpha >= 0.75] = 1.0
        out[(alpha >= 0.25) & (alpha < 0.75)] = 0.5
        return out
    if levels == 5:
        return torch.round(alpha * 4.0) / 4.0
    raise ValueError(f"Unsupported alpha levels: {levels}")


def binary_shape_stats(alpha, hard_gate):
    selected = ((alpha > 0.5) & (hard_gate > 0.5))[0, 0]
    selected_count = int(selected.sum().item())
    if selected_count == 0:
        return {
            "component_count": 0,
            "largest_component": 0,
            "mean_component": 0.0,
            "boundary_edges": 0,
            "boundary_ratio": 0.0,
        }
    arr = selected.detach().cpu().numpy().astype(bool)
    height, width = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    sizes = []
    boundary_edges = 0
    for y in range(height):
        for x in range(width):
            if arr[y, x]:
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width or not arr[ny, nx]:
                        boundary_edges += 1
            if not arr[y, x] or visited[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            size = 0
            while queue:
                cy, cx = queue.popleft()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and arr[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            sizes.append(size)
    return {
        "component_count": len(sizes),
        "largest_component": int(max(sizes)) if sizes else 0,
        "mean_component": float(np.mean(sizes)) if sizes else 0.0,
        "boundary_edges": int(boundary_edges),
        "boundary_ratio": float(boundary_edges / selected_count),
    }


def alpha_stats(policy, alpha, hard_gate):
    active = hard_gate > 0.5
    active_count = int(active.sum().item())
    if active_count == 0:
        base = {
            "policy": policy,
            "active_count": 0,
            "alpha_mass_fraction": 0.0,
            "binary_active_fraction": 0.0,
            "selected_count": 0,
        }
        base.update(binary_shape_stats(alpha, hard_gate))
        return base
    selected = (alpha > 0.5) & active
    base = {
        "policy": policy,
        "active_count": active_count,
        "alpha_mass_fraction": float(alpha[active].mean().item()),
        "binary_active_fraction": float(selected.float().sum().item() / active_count),
        "selected_count": int(selected.sum().item()),
    }
    base.update(binary_shape_stats(alpha, hard_gate))
    return base


def make_policy_alphas(open_score, q1, hard_gate, args):
    policies = {}
    raw_top50 = top_fraction_alpha(open_score, hard_gate, 0.5, largest=True)
    for fraction in args.top_fractions:
        policies[f"OPEN_TOP_{fraction:g}"] = top_fraction_alpha(open_score, hard_gate, fraction, largest=True)
    for block in args.block_sizes:
        policies[f"BLOCK{block}_OPEN_TOP_0.5"] = block_top_alpha(open_score, hard_gate, 0.5, block)
    for kernel in args.smooth_kernels:
        policies[f"SMOOTH{kernel}_OPEN_TOP_0.5"] = smooth_top_alpha(open_score, hard_gate, 0.5, kernel)
        policies[f"MAJORITY{kernel}_FROM_OPEN_TOP_0.5"] = majority_smooth_alpha(raw_top50, hard_gate, kernel)
    for min_size in args.min_component_sizes:
        policies[f"CC_MIN{min_size}_FROM_OPEN_TOP_0.5"] = remove_small_components_alpha(raw_top50, hard_gate, min_size)
    for kernel in args.interior_kernels:
        policies[f"INTERIOR{kernel}_OPEN_TOP_0.5"] = top_fraction_alpha(
            open_score, hard_gate, 0.5, largest=True, eligible=interior_mask(hard_gate, kernel)
        )
    for value in args.uniform_alphas:
        policies[f"UNIFORM_ALPHA_{value:g}"] = torch.full_like(hard_gate, float(value)) * (
            hard_gate > 0.5
        ).to(hard_gate.dtype)
    secant, ignored, near_zero = alpha_secant_teacher(open_score, q1, hard_gate, args.score_eps, args.denom_eps)
    policies["ALPHA_SECANT_CONT"] = secant
    policies["ALPHA_SECANT_Q3"] = quantize_alpha(secant, 3)
    policies["ALPHA_SECANT_Q5"] = quantize_alpha(secant, 5)
    return policies, {
        "secant_ignored_count": int(ignored.sum().item()),
        "secant_near_zero_count": int(near_zero.sum().item()),
    }


def summarize_policy(policy_rows, alpha_rows):
    by_policy = defaultdict(list)
    alpha_by_policy = defaultdict(list)
    for row in policy_rows:
        by_policy[row["policy"]].append(row)
    for row in alpha_rows:
        alpha_by_policy[row["policy"]].append(row)
    summary = []
    for policy in sorted(by_policy):
        rows = by_policy[policy]
        vals = [float(row["psnr_delta"]) for row in rows]
        mses = [float(row["mse_gain"]) for row in rows]
        arows = alpha_by_policy.get(policy, [])

        def mean_alpha(key):
            return float(np.mean([float(row[key]) for row in arows])) if arows else float("nan")

        summary.append(
            {
                "policy": policy,
                "n": len(vals),
                "mean_psnr_delta": float(np.mean(vals)),
                "median_psnr_delta": percentile(vals, 50),
                "p10_psnr_delta": percentile(vals, 10),
                "p05_psnr_delta": percentile(vals, 5),
                "worst_psnr_delta": min(vals),
                "best_psnr_delta": max(vals),
                "positive_ratio": float(np.mean(np.asarray(vals) > 0)),
                "regression_le_0p2_count": int(np.sum(np.asarray(vals) <= -0.2)),
                "regression_le_0p5_count": int(np.sum(np.asarray(vals) <= -0.5)),
                "mean_mse_gain": float(np.mean(mses)),
                "mean_alpha_mass_fraction": mean_alpha("alpha_mass_fraction"),
                "mean_binary_active_fraction": mean_alpha("binary_active_fraction"),
                "mean_boundary_ratio": mean_alpha("boundary_ratio"),
                "mean_component_count": mean_alpha("component_count"),
                "mean_largest_component": mean_alpha("largest_component"),
                "mean_selected_count": mean_alpha("selected_count"),
            }
        )
    return summary


def audit(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base = build_model("original", args.a0_checkpoint, device)
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    names = load_names(args.split_json, args.split_key, args.max_samples)

    policy_rows = []
    alpha_rows = []
    target_rows = []
    secant_ignored_total = 0
    secant_near_zero_total = 0

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

        grad_open_raw, pred_zero, mse_zero = gradient_at_alpha(
            action_model, padded, label, hard_gate, 0.0, height, width
        )
        grad_close_raw, pred_hard, mse_hard = gradient_at_alpha(
            action_model, padded, label, hard_gate, 1.0, height, width
        )
        open_score = -grad_open_raw * hard_gate
        q1 = -grad_close_raw * hard_gate
        active = hard_gate > 0.5
        active_count = int(active.sum().item())
        if active_count:
            active_open = open_score[active]
            active_q1 = q1[active]
            target_rows.append(
                {
                    "index": index,
                    "name": name,
                    "active_count": active_count,
                    "action_coverage": float(hard_gate.mean().item()),
                    "open_positive_fraction": float((active_open > args.score_eps).float().mean().item()),
                    "q1_positive_fraction": float((active_q1 > args.score_eps).float().mean().item()),
                    "open_score_mean": float(active_open.mean().item()),
                    "open_score_p10": float(torch.quantile(active_open, 0.10).item()),
                    "open_score_median": float(torch.quantile(active_open, 0.50).item()),
                    "open_score_p90": float(torch.quantile(active_open, 0.90).item()),
                    "q1_score_mean": float(active_q1.mean().item()),
                    "q1_score_p10": float(torch.quantile(active_q1, 0.10).item()),
                    "q1_score_median": float(torch.quantile(active_q1, 0.50).item()),
                    "q1_score_p90": float(torch.quantile(active_q1, 0.90).item()),
                    "mse_alpha0": mse_zero,
                    "mse_alpha1": mse_hard,
                }
            )
        else:
            target_rows.append(
                {
                    "index": index,
                    "name": name,
                    "active_count": 0,
                    "action_coverage": 0.0,
                    "open_positive_fraction": 0.0,
                    "q1_positive_fraction": 0.0,
                    "open_score_mean": 0.0,
                    "open_score_p10": 0.0,
                    "open_score_median": 0.0,
                    "open_score_p90": 0.0,
                    "q1_score_mean": 0.0,
                    "q1_score_p10": 0.0,
                    "q1_score_median": 0.0,
                    "q1_score_p90": 0.0,
                    "mse_alpha0": mse_zero,
                    "mse_alpha1": mse_hard,
                }
            )

        alphas, teacher_stats = make_policy_alphas(open_score, q1, hard_gate, args)
        secant_ignored_total += teacher_stats["secant_ignored_count"]
        secant_near_zero_total += teacher_stats["secant_near_zero_count"]
        alphas["A0"] = torch.zeros_like(hard_gate)
        alphas["HARD_D7C_ALPHA1"] = (hard_gate > 0.5).to(hard_gate.dtype)

        for policy, alpha in sorted(alphas.items()):
            if policy == "A0":
                pred = base_pred
            elif policy in {"HARD_D7C_ALPHA1", "OPEN_TOP_1"}:
                pred = pred_hard
            else:
                pred = forward_final(action_model, padded, height, width, d7c_gate=hard_gate * alpha)
            mse, psnr = metric_pair(pred, label)
            policy_rows.append(
                {
                    "index": index,
                    "name": name,
                    "policy": policy,
                    "base_mse": base_mse,
                    "base_psnr": base_psnr,
                    "mse": mse,
                    "psnr": psnr,
                    "mse_gain": base_mse - mse,
                    "psnr_delta": psnr - base_psnr,
                }
            )
            alpha_detail = alpha_stats(policy, alpha, hard_gate)
            alpha_detail.update({"index": index, "name": name})
            alpha_rows.append(alpha_detail)

        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"v3i_a_progress {index + 1}/{len(names)}", flush=True)

        del input_img, label, padded, base_pred, pred_zero, pred_hard
        del grad_open_raw, grad_close_raw, open_score, q1, hard_gate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    policy_summary = summarize_policy(policy_rows, alpha_rows)
    by_policy = {row["policy"]: row for row in policy_summary}
    hard = by_policy["HARD_D7C_ALPHA1"]
    oracle = by_policy["OPEN_TOP_0.5"]
    gate_mean_line = hard["mean_psnr_delta"] + args.min_oracle_gap_retention * (
        oracle["mean_psnr_delta"] - hard["mean_psnr_delta"]
    )
    excluded = {"A0", "HARD_D7C_ALPHA1", "OPEN_TOP_0.25", "OPEN_TOP_0.5", "OPEN_TOP_0.75", "OPEN_TOP_1"}
    compressed = [row for row in policy_summary if row["policy"] not in excluded]
    compressed_sorted = sorted(
        compressed,
        key=lambda row: (row["mean_psnr_delta"], -row["regression_le_0p2_count"]),
        reverse=True,
    )
    best_compressed = compressed_sorted[0] if compressed_sorted else None
    passing = [
        row
        for row in compressed
        if row["mean_psnr_delta"] >= gate_mean_line
        and row["regression_le_0p2_count"] <= hard["regression_le_0p2_count"]
    ]
    decision = (
        "V3I_A_ORACLE_COMPRESSIBLE_AUTHORIZE_FULL_CONTEXT_OOF_PROBE_ONLY"
        if passing
        else "V3I_A_OPEN_VALUE_TEACHER_UNSTABLE_STOP_CONTROLLER_DISTILLATION"
    )
    next_action = (
        "Proceed to v3i-B tiny full-context OOF probes; no controller canary or locked test."
        if passing
        else "Stop deployable controller distillation for this target; review new target/source before training."
    )
    target_summary = {
        "sample_count": len(names),
        "active_image_count": int(sum(1 for row in target_rows if row["active_count"] > 0)),
        "active_count": summarize([row["active_count"] for row in target_rows]),
        "action_coverage": summarize([row["action_coverage"] for row in target_rows]),
        "open_positive_fraction": summarize([row["open_positive_fraction"] for row in target_rows]),
        "q1_positive_fraction": summarize([row["q1_positive_fraction"] for row in target_rows]),
        "open_score_mean": summarize([row["open_score_mean"] for row in target_rows]),
        "q1_score_mean": summarize([row["q1_score_mean"] for row in target_rows]),
        "secant_ignored_total": secant_ignored_total,
        "secant_near_zero_total": secant_near_zero_total,
    }
    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3i-A no-training open-value teacher compressibility audit",
        "decision": decision,
        "next_action": next_action,
        "locked_test_touched": False,
        "training_authorized": False,
        "sample_count": len(names),
        "a0_checkpoint": args.a0_checkpoint,
        "a0_checkpoint_sha256": sha256_file(args.a0_checkpoint),
        "control_checkpoint": args.control_checkpoint,
        "control_checkpoint_sha256": sha256_file(args.control_checkpoint),
        "metric_contract": {
            "baseline": "A0 PSNR on internal train-derived val_inner split",
            "primary_target": "open_score = -dL/dalpha at alpha=0 inside D7c hard action sites",
            "primary_policy": "per-image D7c-active open_score top-50 replay",
            "gate": "compressed policy retains at least min_oracle_gap_retention of OPEN_TOP_0.5 over HARD_D7C and has <= hard severe regressions",
            "min_oracle_gap_retention": args.min_oracle_gap_retention,
            "gate_mean_line": gate_mean_line,
        },
        "target_summary": target_summary,
        "hard_policy": hard,
        "open_top50_oracle_policy": oracle,
        "best_compressed_policy": best_compressed,
        "passing_compressed_policies": passing,
        "policy_summary": policy_summary,
    }
    write_csv(output_dir / "v3i_a_policy_per_image.csv", policy_rows)
    write_csv(output_dir / "v3i_a_policy_summary.csv", policy_summary)
    write_csv(output_dir / "v3i_a_alpha_spatial_summary.csv", alpha_rows)
    write_csv(output_dir / "v3i_a_target_stats_by_image.csv", target_rows)
    write_json(output_dir / "v3i_a_teacher_compressibility_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--split_key", default="val_inner")
    parser.add_argument("--max_samples", type=int, default=600)
    parser.add_argument("--top_fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--block_sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--smooth_kernels", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--min_component_sizes", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--interior_kernels", type=int, nargs="+", default=[3])
    parser.add_argument("--uniform_alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--score_eps", type=float, default=0.0)
    parser.add_argument("--denom_eps", type=float, default=1e-12)
    parser.add_argument("--min_oracle_gap_retention", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--d7c_threshold", type=float, default=D7C_THRESHOLD)
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
