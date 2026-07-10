#!/usr/bin/env python3
"""v3h no-training operator-site context feature audit.

This audit tests whether inference-time feature maps at or near the true FAM2
action grid can recover the v3g label-derived alpha target. It does not train
or save any model weights.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ITS_DIR = Path(__file__).resolve().parents[2] / "Dehazing" / "ITS"
if str(ITS_DIR) not in sys.path:
    sys.path.insert(0, str(ITS_DIR))

from d7c_gate import build_d7c_gate_producer, density_pred_from_head, load_checkpoint_state  # noqa: E402
from models.ConvIR import build_net  # noqa: E402


GRAY_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def label_path_for_hazy(data_dir, source_split, hazy_name):
    stem, ext = os.path.splitext(hazy_name)
    candidates = [hazy_name, f"{stem.split('_')[0]}{ext}", f"{stem.split('_')[0]}.png"]
    label_dir = Path(data_dir) / source_split / "gt"
    for candidate in candidates:
        path = label_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No GT found for {hazy_name} under {label_dir}")


def load_pair(data_dir, source_split, hazy_name):
    hazy_path = Path(data_dir) / source_split / "haze" / hazy_name
    label_path = label_path_for_hazy(data_dir, source_split, hazy_name)
    hazy = Image.open(hazy_path).convert("RGB")
    label = Image.open(label_path).convert("RGB")
    hazy_t = torch.from_numpy(np.asarray(hazy).transpose(2, 0, 1)).float() / 255.0
    label_t = torch.from_numpy(np.asarray(label).transpose(2, 0, 1)).float() / 255.0
    return hazy_t, label_t


def load_names(split_json, split_key, max_samples):
    split = read_json(split_json)
    names = sorted(split["splits"][split_key])
    if max_samples is not None:
        names = names[: max_samples]
    if not names:
        raise ValueError(f"No names in split {split_key}")
    return names


def pad_to_factor(input_img, factor=32):
    _, _, height, width = input_img.shape
    pad_h = (factor - height % factor) % factor
    pad_w = (factor - width % factor) % factor
    if pad_h == 0 and pad_w == 0:
        return input_img, height, width
    padded = F.pad(input_img, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, height, width


def mse_to_psnr(mse):
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def metric_pair(pred, label):
    mse = F.mse_loss(pred, label).item()
    return mse, mse_to_psnr(mse)


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


def forward_full(model, padded, d7c_gate=None):
    if d7c_gate is None:
        out = model(padded)[2]
    else:
        out = model(padded, d7c_gate=d7c_gate)[2]
    return torch.clamp(out, 0, 1)


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


def tensor_norm(tensor):
    return tensor.pow(2).mean(dim=1, keepdim=True).sqrt()


def tensor_abs_mean(tensor):
    return tensor.abs().mean(dim=1, keepdim=True)


def resize_map(tensor, shape, mode="bilinear"):
    if tensor.shape[-2:] == shape:
        return tensor
    if mode == "nearest":
        return F.interpolate(tensor, size=shape, mode=mode)
    return F.interpolate(tensor, size=shape, mode=mode, align_corners=False)


def local_mean_std(single_channel, shape):
    mean = resize_map(single_channel, shape)
    sq_mean = resize_map(single_channel * single_channel, shape)
    var = torch.clamp(sq_mean - mean * mean, min=0.0)
    return mean, var.sqrt()


def action_site_maps(action_model, gate_producer, padded, base_full):
    shape = action_shape_for_input(padded)
    with torch.no_grad():
        x_2 = F.interpolate(padded, scale_factor=0.5, mode="bilinear", align_corners=False)
        z2 = action_model.SCM2(x_2)
        x0 = action_model.feat_extract[0](padded)
        res1 = action_model.Encoder[0](x0)
        z = action_model.feat_extract[1](res1)
        fused = action_model.FAM2.merge(torch.cat([z, z2], dim=1))
        gamma, beta = action_model.FAM2.modulator(z2).chunk(2, dim=1)
        action_delta = fused * gamma + beta

        gate_full, score_full, logits_full = gate_producer(padded)
        gate = action_gate_from_full(gate_full, shape)
        score = resize_map(score_full, shape)
        logits = resize_map(logits_full, shape)
        density_full = density_pred_from_head(gate_producer.density_head, res1)
        density = resize_map(density_full, shape)

        gray_weights = GRAY_WEIGHTS.to(device=padded.device, dtype=padded.dtype)
        input_luma_full = (padded * gray_weights).sum(dim=1, keepdim=True)
        a0_luma_full = (base_full * gray_weights).sum(dim=1, keepdim=True)
        input_luma, input_std = local_mean_std(input_luma_full, shape)
        a0_luma = resize_map(a0_luma_full, shape)
        residual_abs = resize_map((padded - base_full).abs().mean(dim=1, keepdim=True), shape)
        input_dark = resize_map(padded.min(dim=1, keepdim=True).values, shape)
        input_saturation = resize_map(
            padded.max(dim=1, keepdim=True).values - padded.min(dim=1, keepdim=True).values, shape
        )

        height, width = shape
        yy = torch.linspace(0, 1, height, device=padded.device, dtype=padded.dtype).view(1, 1, height, 1)
        xx = torch.linspace(0, 1, width, device=padded.device, dtype=padded.dtype).view(1, 1, 1, width)
        y_map = yy.expand(1, 1, height, width)
        x_map = xx.expand(1, 1, height, width)
        border_dist = torch.minimum(torch.minimum(y_map, 1 - y_map), torch.minimum(x_map, 1 - x_map))

        delta_abs = tensor_abs_mean(action_delta)
        gamma_abs = tensor_abs_mean(gamma)
        beta_abs = tensor_abs_mean(beta)
        fused_norm = tensor_norm(fused)

        features = {
            "d7c_score": score,
            "d7c_margin": score - float(gate_producer.threshold),
            "d7c_logit_mean": logits.mean(dim=1, keepdim=True),
            "d7c_logit_max": logits.max(dim=1, keepdim=True).values,
            "density": density,
            "input_luma": input_luma,
            "input_dark": input_dark,
            "input_std": input_std,
            "input_saturation": input_saturation,
            "a0_luma": a0_luma,
            "residual_abs": residual_abs,
            "x1_norm": tensor_norm(z),
            "x2_norm": tensor_norm(z2),
            "fused_norm": fused_norm,
            "gamma_abs": gamma_abs,
            "beta_abs": beta_abs,
            "action_delta_abs": delta_abs,
            "action_delta_signed": action_delta.mean(dim=1, keepdim=True),
            "score_x_delta_abs": score * delta_abs,
            "margin_x_delta_abs": (score - float(gate_producer.threshold)) * delta_abs,
            "density_x_delta_abs": density * delta_abs,
            "residual_x_delta_abs": residual_abs * delta_abs,
            "x_pos": x_map,
            "y_pos": y_map,
            "border_dist": border_dist,
        }
    return gate, score, features


def ranks_average(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def auroc_score(labels, scores):
    labels = labels.astype(np.int32)
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = ranks_average(scores)
    pos_rank_sum = float(ranks[labels == 1].sum())
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def average_precision(labels, scores):
    labels = labels.astype(np.int32)
    pos = int(labels.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(labels)) + 1.0)
    return float((precision * sorted_labels).sum() / pos)


def spearman_corr(a, b):
    if len(a) < 2:
        return float("nan")
    ra = ranks_average(a)
    rb = ranks_average(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def split_name_for_index(index):
    return "calib_even" if index % 2 == 0 else "holdout_odd"


def summarize_policy_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["policy"])].append(row["psnr_delta"])
    out = []
    for (split, policy), vals in sorted(grouped.items()):
        vals = [float(v) for v in vals]
        out.append(
            {
                "split": split,
                "policy": policy,
                "n": len(vals),
                "mean_psnr_delta": float(np.mean(vals)),
                "median_psnr_delta": float(np.median(vals)),
                "p10_psnr_delta": percentile(vals, 10),
                "worst_psnr_delta": float(np.min(vals)),
                "positive_ratio": float(np.mean(np.asarray(vals) > 0)),
                "regression_le_0p2_count": int(np.sum(np.asarray(vals) <= -0.2)),
                "regression_le_0p5_count": int(np.sum(np.asarray(vals) <= -0.5)),
            }
        )
    return out


def summarize_all_split(policy_summary):
    by_policy = defaultdict(list)
    for row in policy_summary:
        by_policy[row["policy"]].append(row)
    return by_policy


def top_fraction_alpha(feature, hard_gate, fraction, largest=True):
    alpha = torch.zeros_like(hard_gate)
    active = hard_gate > 0.5
    count = int(active.sum().item())
    if count == 0 or fraction <= 0:
        return alpha
    keep = max(1, int(round(count * fraction)))
    active_scores = feature[active]
    if keep >= count:
        alpha[active] = 1.0
        return alpha
    threshold = torch.topk(active_scores.reshape(-1), keep, largest=largest).values.min()
    if largest:
        alpha[(feature >= threshold) & active] = 1.0
    else:
        alpha[(feature <= threshold) & active] = 1.0
    return alpha


def add_policy_record(records, index, name, policy, split, pred, label, base_psnr):
    _, psnr = metric_pair(pred, label)
    records.append(
        {
            "index": index,
            "name": name,
            "split": split,
            "policy": policy,
            "psnr_delta": psnr - base_psnr,
        }
    )


def feature_metrics_for_split(feature_values, keep_positive, open_positive, keep_score, open_score, mask):
    rows = []
    keep_rate = float(np.mean(keep_positive[mask])) if int(mask.sum()) else float("nan")
    open_rate = float(np.mean(open_positive[mask])) if int(mask.sum()) else float("nan")
    for feature, values in sorted(feature_values.items()):
        scores = values[mask]
        kp = keep_positive[mask]
        op = open_positive[mask]
        ks = keep_score[mask]
        oscore = open_score[mask]
        keep_auc = auroc_score(kp, scores)
        open_auc = auroc_score(op, scores)
        keep_dir_auc = max(keep_auc, 1.0 - keep_auc) if not math.isnan(keep_auc) else float("nan")
        open_dir_auc = max(open_auc, 1.0 - open_auc) if not math.isnan(open_auc) else float("nan")
        rows.append(
            {
                "feature": feature,
                "keep_auroc": keep_auc,
                "keep_dir_auroc": keep_dir_auc,
                "keep_direction": "high" if (math.isnan(keep_auc) or keep_auc >= 0.5) else "low",
                "keep_ap": average_precision(kp, scores),
                "keep_positive_rate": keep_rate,
                "keep_spearman": spearman_corr(scores, ks),
                "open_auroc": open_auc,
                "open_dir_auroc": open_dir_auc,
                "open_direction": "high" if (math.isnan(open_auc) or open_auc >= 0.5) else "low",
                "open_ap": average_precision(op, scores),
                "open_positive_rate": open_rate,
                "open_spearman": spearman_corr(scores, oscore),
            }
        )
    return rows


def select_replay_candidates(feature_values, image_indices, keep_score, calib_mask, fractions, max_candidates):
    candidates = []
    unique_images = sorted(set(int(i) for i in image_indices[calib_mask]))
    for feature, values in sorted(feature_values.items()):
        for direction in ("high", "low"):
            largest = direction == "high"
            for fraction in fractions:
                selected = np.zeros_like(calib_mask, dtype=bool)
                for image_index in unique_images:
                    local = calib_mask & (image_indices == image_index)
                    if not np.any(local):
                        continue
                    local_values = values[local]
                    keep = max(1, int(round(len(local_values) * fraction)))
                    if keep >= len(local_values):
                        local_selected = np.ones(len(local_values), dtype=bool)
                    else:
                        if largest:
                            threshold = np.partition(local_values, len(local_values) - keep)[len(local_values) - keep]
                            local_selected = local_values >= threshold
                        else:
                            threshold = np.partition(local_values, keep - 1)[keep - 1]
                            local_selected = local_values <= threshold
                    local_indices = np.flatnonzero(local)
                    selected[local_indices[local_selected]] = True
                if not np.any(selected):
                    continue
                candidates.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "fraction": float(fraction),
                        "calib_sample_mean_keep_score": float(np.mean(keep_score[selected])),
                        "calib_sample_positive_ratio": float(np.mean(keep_score[selected] >= 0)),
                        "calib_sample_count": int(np.sum(selected)),
                    }
                )
    candidates.sort(
        key=lambda row: (
            row["calib_sample_mean_keep_score"],
            row["calib_sample_positive_ratio"],
            -row["fraction"],
            row["feature"],
        ),
        reverse=True,
    )
    deduped = []
    seen_features = set()
    for candidate in candidates:
        key = candidate["feature"]
        if key in seen_features and len(deduped) >= max_candidates // 2:
            continue
        deduped.append(candidate)
        seen_features.add(key)
        if len(deduped) >= max_candidates:
            break
    return deduped


def policy_summary_lookup(policy_summary, split, policy):
    for row in policy_summary:
        if row["split"] == split and row["policy"] == policy:
            return row
    return None


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
    parser.add_argument("--sample_per_image", type=int, default=1024)
    parser.add_argument("--replay_max_candidates", type=int, default=8)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--d7c_threshold", type=float, default=0.5773006677627563)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    base = build_model("original", args.a0_checkpoint, device)
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    names = load_names(args.split_json, args.split_key, args.max_samples)

    feature_chunks = defaultdict(list)
    image_index_chunks = []
    keep_score_chunks = []
    open_score_chunks = []
    keep_positive_chunks = []
    open_positive_chunks = []
    policy_records = []
    feature_names = None

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    for index, name in enumerate(names):
        input_img, label = load_pair(args.data_dir, args.source_split, name)
        input_img = input_img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        padded, height, width = pad_to_factor(input_img)
        label = label[:, :, :height, :width]
        split = split_name_for_index(index)

        with torch.no_grad():
            base_full = forward_full(base, padded)
            base_pred = base_full[:, :, :height, :width]
            base_mse, base_psnr = metric_pair(base_pred, label)
            hard_gate, _, features = action_site_maps(action_model, gate_producer, padded, base_full)
            pred_hard = forward_final(action_model, padded, height, width, d7c_gate=hard_gate)
            add_policy_record(policy_records, index, name, "A0", split, base_pred, label, base_psnr)
            add_policy_record(policy_records, index, name, "W_U_G_D_ACTION_HARD", split, pred_hard, label, base_psnr)

        grad_open_raw, _, _ = gradient_at_alpha(action_model, padded, label, hard_gate, 0.0, height, width)
        grad_close_raw, pred_hard_grad, _ = gradient_at_alpha(action_model, padded, label, hard_gate, 1.0, height, width)
        open_score = -grad_open_raw * hard_gate
        keep_score = -grad_close_raw * hard_gate
        active = hard_gate > 0.5

        with torch.no_grad():
            all_gate = torch.ones_like(hard_gate)
            pred_all = forward_final(action_model, padded, height, width, d7c_gate=all_gate)
            add_policy_record(policy_records, index, name, "W_U_G_1_ACTION_ALL", split, pred_all, label, base_psnr)
            alpha_oracle = ((keep_score >= 0) & active).to(hard_gate.dtype)
            pred_oracle = forward_final(action_model, padded, height, width, d7c_gate=hard_gate * alpha_oracle)
            add_policy_record(
                policy_records,
                index,
                name,
                "ACTION_CLOSE_FILTER_GRAD_ORACLE",
                split,
                pred_oracle,
                label,
                base_psnr,
            )

        if not active.any():
            continue
        flat_active = torch.nonzero(active.reshape(-1), as_tuple=False).flatten().cpu()
        sample_count = min(args.sample_per_image, int(flat_active.numel()))
        if sample_count < int(flat_active.numel()):
            perm = torch.randperm(int(flat_active.numel()), generator=generator)[:sample_count]
            sample_indices = flat_active[perm]
        else:
            sample_indices = flat_active
        sample_indices_device = sample_indices.to(device=device)

        if feature_names is None:
            feature_names = sorted(features.keys())
        for feature in feature_names:
            values = features[feature].reshape(-1)[sample_indices_device].detach().float().cpu().numpy()
            feature_chunks[feature].append(values.astype(np.float32))
        image_index_chunks.append(np.full(sample_count, index, dtype=np.int32))
        keep_vals = keep_score.reshape(-1)[sample_indices_device].detach().float().cpu().numpy()
        open_vals = open_score.reshape(-1)[sample_indices_device].detach().float().cpu().numpy()
        keep_score_chunks.append(keep_vals.astype(np.float32))
        open_score_chunks.append(open_vals.astype(np.float32))
        keep_positive_chunks.append((keep_vals >= 0).astype(np.int8))
        open_positive_chunks.append((open_vals > 0).astype(np.int8))

    if not keep_score_chunks:
        raise RuntimeError("No active action-site samples were collected.")

    feature_values = {key: np.concatenate(chunks) for key, chunks in feature_chunks.items()}
    image_indices = np.concatenate(image_index_chunks)
    keep_score_np = np.concatenate(keep_score_chunks)
    open_score_np = np.concatenate(open_score_chunks)
    keep_positive_np = np.concatenate(keep_positive_chunks)
    open_positive_np = np.concatenate(open_positive_chunks)
    calib_mask = (image_indices % 2) == 0
    holdout_mask = ~calib_mask
    all_mask = np.ones_like(calib_mask, dtype=bool)

    feature_rows = []
    for split, mask in [("calib_even", calib_mask), ("holdout_odd", holdout_mask), ("all", all_mask)]:
        for row in feature_metrics_for_split(
            feature_values, keep_positive_np, open_positive_np, keep_score_np, open_score_np, mask
        ):
            row["split"] = split
            feature_rows.append(row)

    candidates = select_replay_candidates(
        feature_values,
        image_indices,
        keep_score_np,
        calib_mask,
        args.fractions,
        args.replay_max_candidates,
    )

    for index, name in enumerate(names):
        input_img, label = load_pair(args.data_dir, args.source_split, name)
        input_img = input_img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        padded, height, width = pad_to_factor(input_img)
        label = label[:, :, :height, :width]
        split = split_name_for_index(index)
        with torch.no_grad():
            base_full = forward_full(base, padded)
            base_pred = base_full[:, :, :height, :width]
            _, base_psnr = metric_pair(base_pred, label)
            hard_gate, _, features = action_site_maps(action_model, gate_producer, padded, base_full)
            for candidate_index, candidate in enumerate(candidates):
                feature = features[candidate["feature"]]
                alpha = top_fraction_alpha(
                    feature,
                    hard_gate,
                    candidate["fraction"],
                    largest=candidate["direction"] == "high",
                )
                pred = forward_final(action_model, padded, height, width, d7c_gate=hard_gate * alpha)
                policy = (
                    f"FEATURE_{candidate_index:02d}_{candidate['feature']}_"
                    f"{candidate['direction']}_{candidate['fraction']:.2f}"
                )
                add_policy_record(policy_records, index, name, policy, split, pred, label, base_psnr)

    policy_summary = summarize_policy_rows(policy_records)
    for row in policy_summary:
        if row["split"] in ("calib_even", "holdout_odd"):
            continue
        row["split"] = row["split"]

    # Add all-split summaries by duplicating from per-image policy records.
    all_records = []
    for row in policy_records:
        copied = dict(row)
        copied["split"] = "all"
        all_records.append(copied)
    policy_summary.extend(summarize_policy_rows(all_records))

    holdout_feature_rows = [r for r in feature_rows if r["split"] == "holdout_odd"]
    best_feature = max(
        holdout_feature_rows,
        key=lambda r: (
            r["keep_dir_auroc"] if not math.isnan(r["keep_dir_auroc"]) else -1,
            r["keep_ap"] if not math.isnan(r["keep_ap"]) else -1,
        ),
    )
    feature_policy_names = [r["policy"] for r in policy_summary if r["split"] == "holdout_odd" and r["policy"].startswith("FEATURE_")]
    best_policy = None
    if feature_policy_names:
        best_policy = max(
            [policy_summary_lookup(policy_summary, "holdout_odd", name) for name in sorted(set(feature_policy_names))],
            key=lambda r: (r["mean_psnr_delta"], -r["regression_le_0p2_count"]),
        )
    hard_holdout = policy_summary_lookup(policy_summary, "holdout_odd", "W_U_G_D_ACTION_HARD")
    oracle_holdout = policy_summary_lookup(policy_summary, "holdout_odd", "ACTION_CLOSE_FILTER_GRAD_ORACLE")

    feature_gate = bool(best_feature["keep_dir_auroc"] >= 0.56)
    replay_gate = False
    if best_policy is not None and hard_holdout is not None:
        replay_gate = bool(
            best_policy["mean_psnr_delta"] > hard_holdout["mean_psnr_delta"] + 0.02
            and best_policy["regression_le_0p2_count"] <= hard_holdout["regression_le_0p2_count"]
        )

    if feature_gate and replay_gate:
        decision = "V3H_OPERATOR_CONTEXT_FEATURE_SIGNAL_PRESENT_REPLAY_POSITIVE_NO_TRAINING_YET"
        next_action = "Authorize a stricter OOF no-training operator-context controller audit; do not train yet."
    elif feature_gate:
        decision = "V3H_OPERATOR_CONTEXT_FEATURE_SEPARABILITY_PRESENT_REPLAY_WEAK_NO_TRAINING"
        next_action = "Diagnose why separability does not replay safely before any training."
    else:
        decision = "V3H_OPERATOR_CONTEXT_FEATURES_WEAK_NO_ROUTER_TRAINING"
        next_action = "Stop current FAM2 operator-context route; new information or target semantics required before training."

    feature_csv = Path(args.output_dir) / "v3h_operator_context_feature_summary.csv"
    with open(feature_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split",
            "feature",
            "keep_auroc",
            "keep_dir_auroc",
            "keep_direction",
            "keep_ap",
            "keep_positive_rate",
            "keep_spearman",
            "open_auroc",
            "open_dir_auroc",
            "open_direction",
            "open_ap",
            "open_positive_rate",
            "open_spearman",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in feature_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    candidate_csv = Path(args.output_dir) / "v3h_operator_context_replay_candidates.csv"
    with open(candidate_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "feature",
            "direction",
            "fraction",
            "calib_sample_mean_keep_score",
            "calib_sample_positive_ratio",
            "calib_sample_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    policy_csv = Path(args.output_dir) / "v3h_operator_context_policy_summary.csv"
    with open(policy_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split",
            "policy",
            "n",
            "mean_psnr_delta",
            "median_psnr_delta",
            "p10_psnr_delta",
            "worst_psnr_delta",
            "positive_ratio",
            "regression_le_0p2_count",
            "regression_le_0p5_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in policy_summary:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "route_id": "haze4k_v5_chd_rm_v3h_operator_site_context_audit_20260710",
        "phase": "no_training_operator_site_context_feature_audit",
        "decision": decision,
        "training_authorized": False,
        "locked_test_touched": False,
        "sample_count": len(names),
        "active_sample_count": int(len(keep_score_np)),
        "split_rule": "even image index calibration, odd image index holdout",
        "target": {
            "primary": "keep_score = - gradient wrt alpha at alpha=1 under D7c hard action gate",
            "positive_label": "keep_score >= 0, equivalent to v3g ACTION_CLOSE_FILTER_GRAD_ORACLE keeping alpha active",
            "secondary": "open_score = - gradient wrt alpha at alpha=0",
        },
        "metric_contract": {
            "baseline": "A0 PSNR on internal train-derived val_inner split",
            "replay": "feature top/bottom fraction alpha masks under same D7c hard action gate",
            "gate": "feature holdout keep_dir_auroc >= 0.56 and replay holdout mean > hard D7c action by 0.02 dB without worse <= -0.2 dB count",
        },
        "best_holdout_feature": best_feature,
        "selected_replay_candidates": candidates,
        "best_holdout_feature_policy": best_policy,
        "hard_d7c_holdout": hard_holdout,
        "oracle_holdout": oracle_holdout,
        "feature_gate_pass": feature_gate,
        "replay_gate_pass": replay_gate,
        "next_action": next_action,
        "artifacts": {
            "feature_summary_csv": str(feature_csv),
            "replay_candidates_csv": str(candidate_csv),
            "policy_summary_csv": str(policy_csv),
        },
    }
    write_json(Path(args.output_dir) / "v3h_operator_context_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
