#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
sys.path.insert(0, str(ITS_ROOT))

from d7c_gate import build_d7c_gate_producer  # noqa: E402
from models.ConvIR import build_net  # noqa: E402


ROUTE_ID = "haze4k_v5_chd_rm_v3g_fam2_action_space_correctability_20260710"
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
    values = sorted(values)
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
    values = list(values)
    return {
        "n": len(values),
        "mean": sum(values) / len(values) if values else float("nan"),
        "min": min(values) if values else float("nan"),
        "p05": percentile(values, 5),
        "p10": percentile(values, 10),
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values) if values else float("nan"),
    }


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def load_checkpoint_state(path, map_location):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


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
    return TF.to_tensor(image), TF.to_tensor(label)


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
    padded_height = ((height + factor) // factor) * factor
    padded_width = ((width + factor) // factor) * factor
    pad_height = padded_height - height if height % factor != 0 else 0
    pad_width = padded_width - width if width % factor != 0 else 0
    if pad_height or pad_width:
        input_img = F.pad(input_img, (0, pad_width, 0, pad_height), "reflect")
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


def masked_replay(base_pred, control_pred, mask):
    return torch.clamp(base_pred + (control_pred - base_pred) * mask, 0, 1)


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


def top_fraction_alpha(score, hard_gate, fraction, largest=True):
    alpha = torch.zeros_like(hard_gate)
    active = hard_gate > 0.5
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


def block_grid_mean(tensor, block_size):
    height = (tensor.shape[-2] // block_size) * block_size
    width = (tensor.shape[-1] // block_size) * block_size
    cropped = tensor[..., :height, :width]
    return F.avg_pool2d(cropped, kernel_size=block_size, stride=block_size)


def add_block(alpha, by, bx, block_size, value):
    alpha = alpha.clone()
    y0 = by * block_size
    x0 = bx * block_size
    alpha[..., y0 : y0 + block_size, x0 : x0 + block_size] = value
    return alpha


def choose_blocks(score_grid, active_grid, count, largest, rng):
    flat_score = score_grid.reshape(-1)
    flat_active = active_grid.reshape(-1) > 0.25
    active_indices = torch.nonzero(flat_active, as_tuple=False).reshape(-1).detach().cpu().numpy()
    if len(active_indices) == 0:
        return []
    count = min(count, len(active_indices))
    scores = flat_score[flat_active].detach().cpu().numpy()
    order = np.argsort(scores)
    if largest:
        order = order[::-1]
    picked = active_indices[order[:count]].tolist()
    if count > 0 and len(active_indices) > count:
        random_count = min(count, len(active_indices))
        random_pick = rng.choice(active_indices, size=random_count, replace=False).tolist()
        picked.extend(random_pick)
    unique = []
    seen = set()
    grid_w = score_grid.shape[-1]
    for idx in picked:
        by, bx = divmod(int(idx), int(grid_w))
        key = (by, bx)
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def audit(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base = build_model("original", args.a0_checkpoint, device)
    # Load W_U into d7c_noop mode so alpha is the actual FAM2 action variable.
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    names = load_names(args.split_json, args.split_key, args.max_samples)

    fractions = args.top_fractions
    policy_names = [
        "A0",
        "W_U_G_1_ACTION_ALL",
        "W_U_G_D_ACTION_HARD",
        "OUTPUT_D7C_GAIN_ORACLE_POS",
        "OUTPUT_GLOBAL_GAIN_ORACLE_POS",
        "ACTION_OPEN_POSITIVE_GRAD",
        "ACTION_CLOSE_FILTER_POSITIVE_GRAD",
    ]
    for frac in fractions:
        policy_names.append(f"ACTION_OPEN_TOP_{frac:g}")
        policy_names.append(f"ACTION_CLOSE_KEEP_{frac:g}")

    policy_rows_by_image = []
    fd_rows = []
    gradient_pixel_rows = []

    for index, name in enumerate(names):
        input_img, label = load_pair(args.data_dir, args.source_split, name)
        input_img = input_img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        padded, height, width = pad_to_factor(input_img)
        label = label[:, :, :height, :width]
        action_shape = action_shape_for_input(padded)

        with torch.no_grad():
            base_pred = forward_final(base, padded, height, width)
            gate_full, score_full, _ = gate_producer(padded)
            hard_action_gate = action_gate_from_full(gate_full, action_shape).to(device)
            all_action_gate = torch.ones_like(hard_action_gate)
            full_control_pred = forward_final(action_model, padded, height, width, d7c_gate=all_action_gate)
            output_gate = F.interpolate(gate_full, size=(height, width), mode="nearest")
            base_err = (base_pred - label).abs().mean(dim=1, keepdim=True)
            control_err = (full_control_pred - label).abs().mean(dim=1, keepdim=True)
            output_gain = base_err - control_err
            output_d7c_oracle = masked_replay(
                base_pred,
                full_control_pred,
                ((output_gain > args.output_gain_eps) & (output_gate > 0.5)).to(base_pred.dtype),
            )
            output_global_oracle = masked_replay(
                base_pred,
                full_control_pred,
                (output_gain > args.output_gain_eps).to(base_pred.dtype),
            )

        grad_open_raw, pred_zero, mse_zero = gradient_at_alpha(
            action_model, padded, label, hard_action_gate, 0.0, height, width
        )
        grad_close_raw, pred_hard, mse_hard = gradient_at_alpha(
            action_model, padded, label, hard_action_gate, 1.0, height, width
        )
        open_score = -grad_open_raw * hard_action_gate
        close_score = grad_close_raw * hard_action_gate

        hard_active = hard_action_gate > 0.5
        if hard_active.any():
            gradient_pixel_rows.append(
                {
                    "index": index,
                    "name": name,
                    "action_coverage": hard_action_gate.mean().item(),
                    "open_positive_fraction": ((open_score > 0) & hard_active).float().sum().item()
                    / hard_active.float().sum().item(),
                    "close_positive_fraction": ((close_score > 0) & hard_active).float().sum().item()
                    / hard_active.float().sum().item(),
                    "open_score_mean_active": open_score[hard_active].mean().item(),
                    "close_score_mean_active": close_score[hard_active].mean().item(),
                    "open_score_p90_active": torch.quantile(open_score[hard_active], 0.9).item(),
                    "close_score_p90_active": torch.quantile(close_score[hard_active], 0.9).item(),
                    "action_zero_mse": mse_zero,
                    "action_hard_mse": mse_hard,
                }
            )
        else:
            gradient_pixel_rows.append(
                {
                    "index": index,
                    "name": name,
                    "action_coverage": 0.0,
                    "open_positive_fraction": 0.0,
                    "close_positive_fraction": 0.0,
                    "open_score_mean_active": 0.0,
                    "close_score_mean_active": 0.0,
                    "open_score_p90_active": 0.0,
                    "close_score_p90_active": 0.0,
                    "action_zero_mse": mse_zero,
                    "action_hard_mse": mse_hard,
                }
            )

        policy_preds = {
            "A0": base_pred,
            "W_U_G_1_ACTION_ALL": full_control_pred,
            "W_U_G_D_ACTION_HARD": pred_hard,
            "OUTPUT_D7C_GAIN_ORACLE_POS": output_d7c_oracle,
            "OUTPUT_GLOBAL_GAIN_ORACLE_POS": output_global_oracle,
        }
        alpha_open_pos = ((open_score > 0) & hard_active).to(hard_action_gate.dtype)
        alpha_close_filter = ((close_score <= 0) & hard_active).to(hard_action_gate.dtype)
        policy_preds["ACTION_OPEN_POSITIVE_GRAD"] = forward_final(
            action_model, padded, height, width, d7c_gate=hard_action_gate * alpha_open_pos
        )
        policy_preds["ACTION_CLOSE_FILTER_POSITIVE_GRAD"] = forward_final(
            action_model, padded, height, width, d7c_gate=hard_action_gate * alpha_close_filter
        )
        for frac in fractions:
            alpha_open = top_fraction_alpha(open_score, hard_action_gate, frac, largest=True)
            alpha_close_keep = top_fraction_alpha(close_score, hard_action_gate, frac, largest=False)
            policy_preds[f"ACTION_OPEN_TOP_{frac:g}"] = forward_final(
                action_model, padded, height, width, d7c_gate=hard_action_gate * alpha_open
            )
            policy_preds[f"ACTION_CLOSE_KEEP_{frac:g}"] = forward_final(
                action_model, padded, height, width, d7c_gate=hard_action_gate * alpha_close_keep
            )

        base_mse, base_psnr = metric_pair(base_pred, label)
        row = {"index": index, "name": name, "base_mse": base_mse, "base_psnr": base_psnr}
        for policy, pred in policy_preds.items():
            mse, psnr = metric_pair(pred, label)
            row[f"{policy}_mse_delta"] = base_mse - mse
            row[f"{policy}_psnr_delta"] = psnr - base_psnr
        policy_rows_by_image.append(row)

        if index < args.fd_max_images:
            for block_size in args.fd_block_sizes:
                active_grid = block_grid_mean(hard_action_gate, block_size)[0, 0]
                open_grid = block_grid_mean(open_score, block_size)[0, 0]
                close_grid = block_grid_mean(close_score, block_size)[0, 0]
                open_blocks = choose_blocks(open_grid, active_grid, args.fd_blocks_per_strategy, True, rng)
                close_blocks = choose_blocks(close_grid, active_grid, args.fd_blocks_per_strategy, True, rng)
                for strategy, blocks in [("open", open_blocks), ("close", close_blocks)]:
                    for by, bx in blocks:
                        if strategy == "open":
                            alpha = torch.zeros_like(hard_action_gate)
                            alpha = add_block(alpha, by, bx, block_size, 1.0)
                            pred = forward_final(
                                action_model, padded, height, width, d7c_gate=hard_action_gate * alpha
                            )
                            mse, psnr = metric_pair(pred, label)
                            base_for_fd_mse = mse_zero
                            base_for_fd_psnr = psnr_from_mse(mse_zero)
                            fd_gain_mse = base_for_fd_mse - mse
                            fd_gain_psnr = psnr - base_for_fd_psnr
                            pred_score = open_grid[by, bx].item()
                        else:
                            alpha = torch.ones_like(hard_action_gate)
                            alpha = add_block(alpha, by, bx, block_size, 0.0)
                            pred = forward_final(
                                action_model, padded, height, width, d7c_gate=hard_action_gate * alpha
                            )
                            mse, psnr = metric_pair(pred, label)
                            base_for_fd_mse = mse_hard
                            base_for_fd_psnr = psnr_from_mse(mse_hard)
                            fd_gain_mse = base_for_fd_mse - mse
                            fd_gain_psnr = psnr - base_for_fd_psnr
                            pred_score = close_grid[by, bx].item()
                        fd_rows.append(
                            {
                                "index": index,
                                "name": name,
                                "strategy": strategy,
                                "block_size": block_size,
                                "block_y": by,
                                "block_x": bx,
                                "active_mean": active_grid[by, bx].item(),
                                "pred_score": pred_score,
                                "fd_gain_mse": fd_gain_mse,
                                "fd_gain_psnr": fd_gain_psnr,
                                "score_positive": pred_score > 0,
                                "fd_positive": fd_gain_mse > args.fd_gain_eps,
                                "sign_agree": (pred_score > 0) == (fd_gain_mse > args.fd_gain_eps),
                                "near_zero_fd": abs(fd_gain_mse) <= args.fd_gain_eps,
                            }
                        )

        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"v3g_action_progress {index + 1}/{len(names)}", flush=True)

        del (
            input_img,
            label,
            padded,
            base_pred,
            full_control_pred,
            pred_zero,
            pred_hard,
            grad_open_raw,
            grad_close_raw,
            open_score,
            close_score,
            hard_action_gate,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    policy_summary = []
    for policy in policy_names:
        psnr_deltas = [row[f"{policy}_psnr_delta"] for row in policy_rows_by_image]
        mse_gains = [row[f"{policy}_mse_delta"] for row in policy_rows_by_image]
        policy_summary.append(
            {
                "policy": policy,
                "mean_psnr_delta": sum(psnr_deltas) / len(psnr_deltas),
                "median_psnr_delta": percentile(psnr_deltas, 50),
                "p10_psnr_delta": percentile(psnr_deltas, 10),
                "worst_psnr_delta": min(psnr_deltas),
                "positive_ratio": sum(v > 0 for v in psnr_deltas) / len(psnr_deltas),
                "regression_le_0p2_count": sum(v <= -0.2 for v in psnr_deltas),
                "regression_le_0p5_count": sum(v <= -0.5 for v in psnr_deltas),
                "mean_mse_gain": sum(mse_gains) / len(mse_gains),
            }
        )

    fd_valid = [row for row in fd_rows if not row["near_zero_fd"]]
    fd_summary = {
        "rows": len(fd_rows),
        "non_near_zero_rows": len(fd_valid),
        "overall_sign_agreement": sum(row["sign_agree"] for row in fd_valid) / len(fd_valid)
        if fd_valid
        else float("nan"),
        "overall_spearman": spearman([row["pred_score"] for row in fd_rows], [row["fd_gain_mse"] for row in fd_rows])
        if fd_rows
        else float("nan"),
        "by_strategy": {},
    }
    for strategy in sorted({row["strategy"] for row in fd_rows}):
        rows = [row for row in fd_rows if row["strategy"] == strategy]
        valid = [row for row in rows if not row["near_zero_fd"]]
        fd_summary["by_strategy"][strategy] = {
            "rows": len(rows),
            "non_near_zero_rows": len(valid),
            "sign_agreement": sum(row["sign_agree"] for row in valid) / len(valid) if valid else float("nan"),
            "spearman": spearman([row["pred_score"] for row in rows], [row["fd_gain_mse"] for row in rows])
            if rows
            else float("nan"),
            "fd_gain_mse": summarize([row["fd_gain_mse"] for row in rows]),
            "fd_gain_psnr": summarize([row["fd_gain_psnr"] for row in rows]),
        }

    policy_by_name = {row["policy"]: row for row in policy_summary}
    hard = policy_by_name["W_U_G_D_ACTION_HARD"]
    ungated = policy_by_name["W_U_G_1_ACTION_ALL"]
    output_oracle = policy_by_name["OUTPUT_D7C_GAIN_ORACLE_POS"]
    action_candidates = [
        row
        for row in policy_summary
        if row["policy"].startswith("ACTION_")
    ]
    best_action = max(action_candidates, key=lambda row: (row["mean_psnr_delta"], -row["regression_le_0p2_count"]))
    output_gap = output_oracle["mean_psnr_delta"] - hard["mean_psnr_delta"]
    strong_mean_line = hard["mean_psnr_delta"] + 0.5 * output_gap
    action_oracle_strong = (
        best_action["mean_psnr_delta"] > max(ungated["mean_psnr_delta"], strong_mean_line)
        and best_action["regression_le_0p2_count"] <= hard["regression_le_0p2_count"]
    )
    fd_pass = (
        fd_summary["non_near_zero_rows"] > 0
        and fd_summary["overall_sign_agreement"] >= args.min_fd_sign_agreement
        and fd_summary["overall_spearman"] >= args.min_fd_spearman
    )

    if not fd_pass:
        decision = "V3G_ACTION_TARGET_UNSTABLE_REQUIRE_MARGIN_OR_COARSER_BLOCK_NO_ROUTER"
        next_action = "Do not train a router; action-space gradient does not reliably match finite differences."
    elif action_oracle_strong:
        decision = "V3G_ACTION_ORACLE_STRONG_FEATURES_WEAK_REQUIRE_OPERATOR_CONTEXT_NO_TRAINING"
        next_action = "Authorize a separate operator-site context feature audit; no training yet."
    else:
        decision = "V3G_FAM2_ACTION_ORACLE_WEAK_STOP_INTERNAL_RARM_MOVE_TO_OUTPUT_BLEND"
        next_action = "Stop internal FAM2 routing and design output-space bounded residual selection preflight."

    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3g-A/B/C no-training FAM2 action-space correctability audit",
        "sample_count": len(names),
        "fd_max_images": args.fd_max_images,
        "locked_test_touched": False,
        "training_authorized": False,
        "a0_checkpoint": args.a0_checkpoint,
        "a0_checkpoint_sha256": sha256_file(args.a0_checkpoint),
        "control_checkpoint": args.control_checkpoint,
        "control_checkpoint_sha256": sha256_file(args.control_checkpoint),
        "metric_contract": {
            "primary": "image-level MSE/PSNR delta against A0 on internal val_inner",
            "output_oracle": "post-output L1 positive pixel replay upper bound, not deployable FAM2 action",
            "action_variable": "FAM2-scale alpha in d7c_gate = D7c_hard_gate * alpha, W_U modulator weights frozen",
        },
        "policy_summary": policy_summary,
        "fd_summary": fd_summary,
        "hard_action_policy": hard,
        "ungated_action_policy": ungated,
        "output_d7c_oracle_policy": output_oracle,
        "best_action_space_policy": best_action,
        "strong_action_mean_line": strong_mean_line,
        "fd_pass": fd_pass,
        "action_oracle_strong": action_oracle_strong,
        "decision": decision,
        "next_action": next_action,
    }

    write_csv(output_dir / "v3g_action_policy_per_image.csv", policy_rows_by_image)
    write_csv(output_dir / "v3g_action_policy_summary.csv", policy_summary)
    write_csv(output_dir / "v3g_gradient_fd_alignment.csv", fd_rows)
    write_csv(output_dir / "v3g_gradient_action_stats_by_image.csv", gradient_pixel_rows)
    write_json(output_dir / "v3g_action_space_audit_summary.json", summary)
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
    parser.add_argument("--fd_max_images", type=int, default=120)
    parser.add_argument("--fd_blocks_per_strategy", type=int, default=4)
    parser.add_argument("--fd_block_sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--top_fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--d7c_threshold", type=float, default=D7C_THRESHOLD)
    parser.add_argument("--output_gain_eps", type=float, default=0.0)
    parser.add_argument("--fd_gain_eps", type=float, default=1e-10)
    parser.add_argument("--min_fd_sign_agreement", type=float, default=0.75)
    parser.add_argument("--min_fd_spearman", type=float, default=0.5)
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
