#!/usr/bin/env python3
"""v3i-B tiny full-context OOF probe and policy replay.

This diagnostic trains only small action-value heads on sampled FAM2 action
sites. A0, W_U, D7c, and the ConvIR backbone stay frozen. The script writes
compact text evidence only; probe weights and raw feature tensors are not saved.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
if str(ITS_ROOT) not in sys.path:
    sys.path.insert(0, str(ITS_ROOT))

from d7c_gate import build_d7c_gate_producer, density_pred_from_head, load_checkpoint_state  # noqa: E402
from models.ConvIR import build_net  # noqa: E402


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


def ranks01(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float32)
    return (ranks / (len(values) - 1)).astype(np.float32)


def clean_id(name):
    stem = Path(name).stem
    return stem.split("_")[0]


def fold_assignments(names, fold_count):
    ids = sorted({clean_id(name) for name in names})
    id_to_fold = {cid: idx % fold_count for idx, cid in enumerate(ids)}
    return np.asarray([id_to_fold[clean_id(name)] for name in names], dtype=np.int64), id_to_fold


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


def forward_full(model, padded, d7c_gate=None):
    if d7c_gate is None:
        out = model(padded)[2]
    else:
        out = model(padded, d7c_gate=d7c_gate)[2]
    return torch.clamp(out, 0, 1)


def forward_final(model, padded, height, width, d7c_gate=None):
    return forward_full(model, padded, d7c_gate=d7c_gate)[:, :, :height, :width]


def action_shape_for_input(padded):
    return (padded.shape[-2] // 2, padded.shape[-1] // 2)


def resize_map(tensor, shape, mode="bilinear"):
    if tensor.shape[-2:] == shape:
        return tensor
    if mode == "nearest":
        return F.interpolate(tensor, size=shape, mode=mode)
    return F.interpolate(tensor, size=shape, mode=mode, align_corners=False)


def action_gate_from_full(gate_full, shape):
    return F.interpolate(gate_full, size=shape, mode="nearest")


def gradient_at_alpha(model, padded, label, hard_action_gate, alpha_value, height, width):
    alpha = torch.full_like(hard_action_gate, float(alpha_value), requires_grad=True)
    model.zero_grad(set_to_none=True)
    pred = forward_final(model, padded, height, width, d7c_gate=hard_action_gate * alpha)
    loss = F.mse_loss(pred, label)
    loss.backward()
    grad = alpha.grad.detach()
    del pred, loss, alpha
    return grad


def full_context_maps(action_model, gate_producer, padded):
    shape = action_shape_for_input(padded)
    with torch.no_grad():
        x_2 = F.interpolate(padded, scale_factor=0.5, mode="bilinear", align_corners=False)
        z2 = action_model.SCM2(x_2)
        x0 = action_model.feat_extract[0](padded)
        res1 = action_model.Encoder[0](x0)
        z = action_model.feat_extract[1](res1)
        fused = action_model.FAM2.merge(torch.cat([z, z2], dim=1))
        gamma, beta = action_model.FAM2.modulator(z2).chunk(2, dim=1)
        delta = fused * gamma + beta
        gate_full, score_full, logits_full = gate_producer(padded)
        gate = action_gate_from_full(gate_full, shape)
        score = resize_map(score_full, shape)
        logits = resize_map(logits_full, shape)
        density = resize_map(density_pred_from_head(gate_producer.density_head, res1), shape)
        height, width = shape
        yy = torch.linspace(0, 1, height, device=padded.device, dtype=padded.dtype).view(1, 1, height, 1)
        xx = torch.linspace(0, 1, width, device=padded.device, dtype=padded.dtype).view(1, 1, 1, width)
        y_map = yy.expand(1, 1, height, width)
        x_map = xx.expand(1, 1, height, width)
        border = torch.minimum(torch.minimum(y_map, 1 - y_map), torch.minimum(x_map, 1 - x_map))
        feature = torch.cat([z, z2, fused, gamma, beta, delta, logits, score, density, x_map, y_map, border], dim=1)
        coord = torch.cat([x_map, y_map, border], dim=1)
    return feature, coord, gate, score


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


def random_top_fraction_alpha(hard_gate, fraction, rng):
    alpha = torch.zeros_like(hard_gate)
    active = torch.nonzero((hard_gate > 0.5).reshape(-1), as_tuple=False).reshape(-1)
    count = int(active.numel())
    if count == 0:
        return alpha
    keep = max(1, int(round(count * fraction)))
    selected = rng.choice(active.detach().cpu().numpy(), size=min(keep, count), replace=False)
    flat = alpha.reshape(-1)
    flat[torch.as_tensor(selected, device=alpha.device, dtype=torch.long)] = 1.0
    return alpha


def residual_abs_top25(base_full, padded, hard_gate):
    shape = hard_gate.shape[-2:]
    residual = resize_map((padded - base_full).abs().mean(dim=1, keepdim=True), shape)
    return top_fraction_alpha(residual, hard_gate, 0.25, largest=True)


def extract_sample_patches(feature, coord, hard_gate, open_score, sample_count, rng):
    active = torch.nonzero((hard_gate > 0.5).reshape(-1), as_tuple=False).reshape(-1)
    if int(active.numel()) == 0:
        return None
    count = min(sample_count, int(active.numel()))
    chosen = rng.choice(active.detach().cpu().numpy(), size=count, replace=False)
    chosen_t = torch.as_tensor(chosen, device=feature.device, dtype=torch.long)
    height, width = feature.shape[-2:]
    ys = torch.div(chosen_t, width, rounding_mode="floor")
    xs = chosen_t % width
    padded_feature = F.pad(feature, (1, 1, 1, 1), mode="replicate")
    patches = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        patches.append(padded_feature[0, :, y : y + 3, x : x + 3].detach().cpu().numpy())
    patch_np = np.stack(patches).astype(np.float16)
    center_np = patch_np[:, :, 1, 1].astype(np.float32)
    coord_flat = coord.reshape(coord.shape[1], -1).transpose(0, 1)[chosen_t].detach().cpu().numpy().astype(np.float32)
    q_np = open_score.reshape(-1)[chosen_t].detach().cpu().numpy().astype(np.float32)
    return patch_np, center_np, coord_flat, q_np


class LinearProbe(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Linear(channels, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)

    def forward_map(self, fmap):
        weight = self.linear.weight[:, :, None, None]
        return F.conv2d(fmap, weight, self.linear.bias)


class PatchDWProbe(nn.Module):
    def __init__(self, channels, proj_channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, proj_channels, 1)
        self.depthwise = nn.Conv2d(proj_channels, proj_channels, 3, padding=1, groups=proj_channels)
        self.pointwise = nn.Conv2d(proj_channels, 1, 1)

    def forward(self, patch):
        out = self.pointwise(self.depthwise(self.proj(patch)))
        return out[:, 0, 1, 1]

    def forward_map(self, fmap):
        return self.pointwise(self.depthwise(self.proj(fmap)))


def normalize_center(x, mean, std):
    return (x - mean) / std


def normalize_patch(x, mean, std):
    return (x - mean[:, None, None]) / std[:, None, None]


def train_probe(kind, arrays, train_mask, args, device, fold):
    rng = np.random.default_rng(args.seed + fold * 997 + {"coord": 1, "linear": 2, "dw3x3": 3}[kind])
    if kind == "coord":
        x_all = arrays["coord"]
        channels = x_all.shape[1]
        mean = x_all[train_mask].mean(axis=0).astype(np.float32)
        std = x_all[train_mask].std(axis=0).astype(np.float32) + 1e-6
        model = LinearProbe(channels).to(device)
        steps = args.coord_steps
    elif kind == "linear":
        x_all = arrays["center"]
        channels = x_all.shape[1]
        mean = x_all[train_mask].mean(axis=0).astype(np.float32)
        std = x_all[train_mask].std(axis=0).astype(np.float32) + 1e-6
        model = LinearProbe(channels).to(device)
        steps = args.linear_steps
    elif kind == "dw3x3":
        x_all = arrays["patch"]
        center = arrays["center"]
        channels = center.shape[1]
        mean = center[train_mask].mean(axis=0).astype(np.float32)
        std = center[train_mask].std(axis=0).astype(np.float32) + 1e-6
        model = PatchDWProbe(channels, args.proj_channels).to(device)
        steps = args.dw_steps
    else:
        raise ValueError(kind)
    y_all = arrays["rank"]
    train_indices = np.flatnonzero(train_mask)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    history = []
    for step in range(steps):
        batch_idx = rng.choice(train_indices, size=min(args.batch_size, len(train_indices)), replace=len(train_indices) < args.batch_size)
        y = torch.from_numpy(y_all[batch_idx]).to(device=device, dtype=torch.float32)
        if kind == "dw3x3":
            xb = torch.from_numpy(x_all[batch_idx].astype(np.float32)).to(device=device)
            xb = normalize_patch(xb, torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device))
        else:
            xb = torch.from_numpy(x_all[batch_idx].astype(np.float32)).to(device=device)
            xb = normalize_center(xb, torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device))
        score = model(xb)
        label = (y >= 0.5).float()
        loss_cls = F.binary_cross_entropy_with_logits(score, label)
        loss_val = F.smooth_l1_loss(torch.sigmoid(score), y)
        loss = loss_cls + args.value_loss_weight * loss_val
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        if (step + 1) % max(1, args.log_every) == 0 or step == steps - 1:
            history.append({"fold": fold, "probe": kind, "step": step + 1, "loss": float(loss.item())})
    return model.eval(), mean, std, history


def score_map_for_probe(kind, model, fmap, coord, mean, std):
    mean_t = torch.from_numpy(mean).to(device=fmap.device, dtype=fmap.dtype)
    std_t = torch.from_numpy(std).to(device=fmap.device, dtype=fmap.dtype)
    with torch.no_grad():
        if kind == "coord":
            x = normalize_patch(coord, mean_t, std_t)
            return model.forward_map(x)
        x = normalize_patch(fmap, mean_t, std_t)
        return model.forward_map(x)


def summarize_policy(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(float(row["psnr_delta"]))
    out = []
    for policy, vals in sorted(grouped.items()):
        arr = np.asarray(vals, dtype=np.float64)
        out.append(
            {
                "policy": policy,
                "n": len(vals),
                "mean_psnr_delta": float(np.mean(arr)),
                "median_psnr_delta": percentile(vals, 50),
                "p10_psnr_delta": percentile(vals, 10),
                "p05_psnr_delta": percentile(vals, 5),
                "worst_psnr_delta": float(np.min(arr)),
                "positive_ratio": float(np.mean(arr > 0)),
                "regression_le_0p2_count": int(np.sum(arr <= -0.2)),
                "regression_le_0p5_count": int(np.sum(arr <= -0.5)),
            }
        )
    return out


def bootstrap_delta(candidate_rows, hard_by_name, rng, draws):
    deltas = []
    names = [row["name"] for row in candidate_rows if row["name"] in hard_by_name]
    diffs = np.asarray([float(row["psnr_delta"]) - hard_by_name[row["name"]] for row in candidate_rows if row["name"] in hard_by_name])
    if len(diffs) == 0:
        return {"n": 0, "mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    for _ in range(draws):
        pick = rng.integers(0, len(diffs), size=len(diffs))
        deltas.append(float(np.mean(diffs[pick])))
    return {
        "n": len(names),
        "mean": float(np.mean(diffs)),
        "ci95_low": percentile(deltas, 2.5),
        "ci95_high": percentile(deltas, 97.5),
    }


def load_v3i_a_controls(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    out = {}
    for row in rows:
        out[(row["name"], row["policy"])] = float(row["psnr_delta"])
    return out


def audit(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    names = load_names(args.split_json, args.split_key, args.max_samples)
    folds_by_image, id_to_fold = fold_assignments(names, args.fold_count)
    base = build_model("original", args.a0_checkpoint, device)
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)

    patches = []
    centers = []
    coords = []
    q_values = []
    sample_image_indices = []
    sample_folds = []
    sample_names = []
    sample_rank = []

    for index, name in enumerate(names):
        input_img, label = load_pair(args.data_dir, args.source_split, name)
        input_img = input_img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        padded, height, width = pad_to_factor(input_img)
        label = label[:, :, :height, :width]
        fmap, coord, hard_gate, _ = full_context_maps(action_model, gate_producer, padded)
        grad_open_raw = gradient_at_alpha(action_model, padded, label, hard_gate, 0.0, height, width)
        open_score = -grad_open_raw * hard_gate
        sampled = extract_sample_patches(fmap, coord, hard_gate, open_score, args.sample_per_image, rng)
        if sampled is not None:
            patch_np, center_np, coord_np, q_np = sampled
            ranks = ranks01(q_np)
            patches.append(patch_np)
            centers.append(center_np)
            coords.append(coord_np)
            q_values.append(q_np)
            sample_image_indices.append(np.full(len(q_np), index, dtype=np.int32))
            sample_folds.append(np.full(len(q_np), folds_by_image[index], dtype=np.int16))
            sample_names.extend([name] * len(q_np))
            sample_rank.append(ranks)
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"v3i_b_extract_progress {index + 1}/{len(names)}", flush=True)
        del input_img, label, padded, fmap, coord, hard_gate, grad_open_raw, open_score
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    arrays = {
        "patch": np.concatenate(patches, axis=0),
        "center": np.concatenate(centers, axis=0),
        "coord": np.concatenate(coords, axis=0),
        "q": np.concatenate(q_values, axis=0),
        "image_index": np.concatenate(sample_image_indices, axis=0),
        "fold": np.concatenate(sample_folds, axis=0),
        "rank": np.concatenate(sample_rank, axis=0),
    }
    feature_manifest = {
        "sample_count": int(arrays["q"].shape[0]),
        "sample_per_image": args.sample_per_image,
        "feature_channels": int(arrays["center"].shape[1]),
        "coord_channels": int(arrays["coord"].shape[1]),
        "fold_count": args.fold_count,
        "clean_reference_count": len(id_to_fold),
        "raw_feature_tensors_saved": False,
    }

    histories = []
    replay_rows = []
    control_rows = []
    v3i_a_controls = load_v3i_a_controls(args.v3i_a_policy_per_image)
    hard_by_name = {}
    oracle_by_name = {}
    for name in names:
        hard_delta = v3i_a_controls.get((name, "HARD_D7C_ALPHA1"))
        oracle_delta = v3i_a_controls.get((name, "OPEN_TOP_0.5"))
        if hard_delta is not None:
            hard_by_name[name] = hard_delta
            control_rows.append({"name": name, "policy": "HARD_D7C_ALPHA1", "psnr_delta": hard_delta})
        if oracle_delta is not None:
            oracle_by_name[name] = oracle_delta
            control_rows.append({"name": name, "policy": "GT_OPEN_TOP50_ORACLE", "psnr_delta": oracle_delta})
        control_rows.append({"name": name, "policy": "A0", "psnr_delta": 0.0})

    for fold in range(args.fold_count):
        train_mask = arrays["fold"] != fold
        fold_models = {}
        for kind in ("coord", "linear", "dw3x3"):
            model, mean, std, history = train_probe(kind, arrays, train_mask, args, device, fold)
            fold_models[kind] = (model, mean, std)
            histories.extend(history)
        for index, name in enumerate(names):
            if folds_by_image[index] != fold:
                continue
            input_img, label = load_pair(args.data_dir, args.source_split, name)
            input_img = input_img.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            padded, height, width = pad_to_factor(input_img)
            label = label[:, :, :height, :width]
            with torch.no_grad():
                base_full = forward_full(base, padded)
                base_pred = base_full[:, :, :height, :width]
                base_mse, base_psnr = metric_pair(base_pred, label)
                fmap, coord, hard_gate, _ = full_context_maps(action_model, gate_producer, padded)
                random_alpha = random_top_fraction_alpha(hard_gate, 0.5, rng)
                residual_alpha = residual_abs_top25(base_full, padded, hard_gate)
                for policy, alpha in [
                    ("RANDOM_TOP50", random_alpha),
                    ("V3H_RESIDUAL_ABS_TOP25", residual_alpha),
                ]:
                    pred = forward_final(action_model, padded, height, width, d7c_gate=hard_gate * alpha)
                    _, psnr = metric_pair(pred, label)
                    replay_rows.append(
                        {"fold": fold, "index": index, "name": name, "policy": policy, "psnr_delta": psnr - base_psnr}
                    )
                for kind, (model, mean, std) in fold_models.items():
                    score = score_map_for_probe(kind, model, fmap, coord, mean, std)
                    alpha = top_fraction_alpha(score, hard_gate, 0.5, largest=True)
                    pred = forward_final(action_model, padded, height, width, d7c_gate=hard_gate * alpha)
                    _, psnr = metric_pair(pred, label)
                    replay_rows.append(
                        {
                            "fold": fold,
                            "index": index,
                            "name": name,
                            "policy": f"OOF_{kind.upper()}_TOP50",
                            "psnr_delta": psnr - base_psnr,
                        }
                    )
            del input_img, label, padded, base_full, base_pred, fmap, coord, hard_gate
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"v3i_b_fold_done {fold}", flush=True)

    all_rows = control_rows + replay_rows
    policy_summary = summarize_policy(all_rows)
    summary_by_policy = {row["policy"]: row for row in policy_summary}
    hard = summary_by_policy["HARD_D7C_ALPHA1"]
    oracle = summary_by_policy["GT_OPEN_TOP50_ORACLE"]
    gate_line = hard["mean_psnr_delta"] + args.min_oracle_gap_retention * (
        oracle["mean_psnr_delta"] - hard["mean_psnr_delta"]
    )
    rng_boot = np.random.default_rng(args.seed + 2026)
    bootstrap_rows = []
    replay_by_policy = defaultdict(list)
    for row in replay_rows:
        replay_by_policy[row["policy"]].append(row)
    for policy, rows in sorted(replay_by_policy.items()):
        boot = bootstrap_delta(rows, hard_by_name, rng_boot, args.bootstrap_draws)
        boot["policy"] = policy
        bootstrap_rows.append(boot)
    candidate_policies = ["OOF_LINEAR_TOP50", "OOF_DW3X3_TOP50"]
    pass_policies = []
    strong_pass_policies = []
    for policy in candidate_policies:
        row = summary_by_policy.get(policy)
        boot = next((b for b in bootstrap_rows if b["policy"] == policy), None)
        if not row or not boot:
            continue
        min_pass = (
            boot["ci95_low"] > 0
            and row["regression_le_0p2_count"] <= hard["regression_le_0p2_count"]
            and row["p10_psnr_delta"] >= hard["p10_psnr_delta"]
        )
        strong_pass = (
            min_pass
            and row["mean_psnr_delta"] > args.ungated_mean_line
            and row["mean_psnr_delta"] >= gate_line
        )
        if min_pass:
            pass_policies.append(policy)
        if strong_pass:
            strong_pass_policies.append(policy)
    if strong_pass_policies:
        decision = "V3I_B_FULL_CONTEXT_REPLAY_PASS_AUTHORIZE_CONTROLLER_ONLY_CANARY"
        next_action = "Authorize controller-only canary planning; no locked test."
    elif pass_policies:
        decision = "V3I_B_FULL_CONTEXT_REPLAY_WEAK_PASS_REQUIRE_CONFIRM_OR_FEATURE_AUDIT"
        next_action = "Do not canary yet; run confirm/stability or counterfactual feature audit."
    else:
        decision = "V3I_B_SINGLE_FORWARD_OBSERVABILITY_FAIL_AUTHORIZE_COUNTERFACTUAL_FEATURE_AUDIT_ONLY"
        next_action = "Stop single-forward controller training; audit counterfactual/disagreement signals only."
    summary = {
        "phase": "v3i-B tiny full-context group OOF probe replay",
        "decision": decision,
        "next_action": next_action,
        "training_authorized": False,
        "controller_canary_authorized": bool(strong_pass_policies),
        "locked_test_touched": False,
        "sample_count": len(names),
        "feature_manifest": feature_manifest,
        "a0_checkpoint": args.a0_checkpoint,
        "a0_checkpoint_sha256": sha256_file(args.a0_checkpoint),
        "control_checkpoint": args.control_checkpoint,
        "control_checkpoint_sha256": sha256_file(args.control_checkpoint),
        "metric_contract": {
            "split": "5-fold clean-reference OOF over internal val_inner 600",
            "target": "sampled within-image open_score rank from alpha=0",
            "replay": "D7c hard veto times predicted top-50 active action sites",
            "minimum_pass": "paired mean gain over hard D7c CI95 low > 0, severe regressions <= hard, p10 >= hard",
            "strong_pass": "minimum pass plus mean > ungated line and >= 25% oracle gap retention",
            "ungated_mean_line": args.ungated_mean_line,
            "oracle_gap_gate_line": gate_line,
        },
        "hard_policy": hard,
        "oracle_policy": oracle,
        "policy_summary": policy_summary,
        "bootstrap_vs_hard": bootstrap_rows,
        "minimum_pass_policies": pass_policies,
        "strong_pass_policies": strong_pass_policies,
    }
    write_csv(output_dir / "v3i_b_probe_training_history.csv", histories)
    write_csv(output_dir / "v3i_b_policy_replay_per_image.csv", all_rows)
    write_csv(output_dir / "v3i_b_policy_replay_summary.csv", policy_summary)
    write_csv(output_dir / "v3i_b_bootstrap_vs_hard.csv", bootstrap_rows)
    write_json(output_dir / "v3i_b_full_context_probe_summary.json", summary)
    write_json(output_dir / "v3i_b_full_tensor_feature_manifest.json", feature_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--v3i_a_policy_per_image", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--split_key", default="val_inner")
    parser.add_argument("--max_samples", type=int, default=600)
    parser.add_argument("--sample_per_image", type=int, default=192)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--coord_steps", type=int, default=180)
    parser.add_argument("--linear_steps", type=int, default=320)
    parser.add_argument("--dw_steps", type=int, default=420)
    parser.add_argument("--proj_channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_norm", type=float, default=5.0)
    parser.add_argument("--value_loss_weight", type=float, default=0.2)
    parser.add_argument("--min_oracle_gap_retention", type=float, default=0.25)
    parser.add_argument("--ungated_mean_line", type=float, default=0.03306524052036514)
    parser.add_argument("--bootstrap_draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--d7c_threshold", type=float, default=D7C_THRESHOLD)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
