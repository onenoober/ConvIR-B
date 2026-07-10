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
from pytorch_msssim import ssim
from torchvision.transforms import functional as TF


REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
sys.path.insert(0, str(ITS_ROOT))

from d7c_gate import build_d7c_gate_producer  # noqa: E402
from models.ConvIR import build_net  # noqa: E402


ROUTE_ID = "haze4k_v5_chd_rm_v3f_operator_correctability_ranker_20260710"
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


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


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
        "mean": mean(values),
        "min": min(values) if values else float("nan"),
        "p10": percentile(values, 10),
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
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


def auroc(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)
    pos = int(labels.sum())
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def auprc(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)
    pos = int(labels.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def spearman(scores, targets):
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    if len(scores) < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(scores), rankdata(targets))[0, 1])


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


def metric_summary(pred, label):
    pred = torch.clamp(pred, 0, 1)
    label = torch.clamp(label, 0, 1)
    mse = F.mse_loss(pred, label)
    psnr = float("inf") if mse.item() == 0 else (10.0 * torch.log10(1.0 / mse)).item()
    height, width = pred.shape[2], pred.shape[3]
    down_ratio = max(1, round(min(height, width) / 256))
    pooled_pred = F.adaptive_avg_pool2d(pred, (int(height / down_ratio), int(width / down_ratio)))
    pooled_label = F.adaptive_avg_pool2d(label, (int(height / down_ratio), int(width / down_ratio)))
    ssim_val = ssim(pooled_pred, pooled_label, data_range=1, size_average=False).mean().item()
    return psnr, ssim_val


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
    return model


def masked_replay(base_pred, control_pred, mask):
    return torch.clamp(base_pred + (control_pred - base_pred) * mask, 0, 1)


def top_fraction_mask(score, veto, fraction):
    mask = torch.zeros_like(score)
    active = veto > 0.5
    count = int(active.sum().item())
    if count == 0 or fraction <= 0:
        return mask
    keep = max(1, int(round(count * fraction)))
    active_scores = score[active]
    if keep >= count:
        mask[active] = 1.0
        return mask
    threshold = torch.topk(active_scores.reshape(-1), keep, largest=True).values.min()
    mask[(score >= threshold) & active] = 1.0
    return mask


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
    control = build_model("fam2_modres", args.control_checkpoint, device)
    d7c_model = build_model("fam2_d7c_noop", args.d7c_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    names = load_names(args.split_json, args.split_key, args.max_samples)

    feature_vectors = {
        "d7c_score": [],
        "d7c_gate": [],
        "control_change": [],
        "residual_abs_input_a0": [],
        "score_x_change": [],
        "gate_x_change": [],
        "control_change_inside_gate": [],
    }
    target_gain = []
    target_positive = []
    target_harmful = []
    hard_gate_sample = []

    policy_names = [
        "A0",
        "W_U_G_1",
        "W_U_G_D",
        "D7C_GAIN_ORACLE_POS",
        "GLOBAL_GAIN_ORACLE_POS",
    ]
    for frac in args.top_fractions:
        policy_names.append(f"D7C_TOP_CHANGE_{frac:g}")
        policy_names.append(f"D7C_TOP_SCOREXCHANGE_{frac:g}")

    per_image_rows = []
    with torch.no_grad():
        for index, name in enumerate(names):
            input_img, label = load_pair(args.data_dir, args.source_split, name)
            input_img = input_img.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            padded, height, width = pad_to_factor(input_img)
            base_pred = torch.clamp(base(padded)[2][:, :, :height, :width], 0, 1)
            control_pred = torch.clamp(control(padded)[2][:, :, :height, :width], 0, 1)
            gate_raw, score_raw, _ = gate_producer(padded)
            d7c_pred = torch.clamp(d7c_model(padded, d7c_gate=gate_raw)[2][:, :, :height, :width], 0, 1)
            label = label[:, :, :height, :width]
            gate = F.interpolate(gate_raw, size=(height, width), mode="nearest")
            score = F.interpolate(score_raw, size=(height, width), mode="bilinear", align_corners=False)
            base_err = (base_pred - label).abs().mean(dim=1, keepdim=True)
            control_err = (control_pred - label).abs().mean(dim=1, keepdim=True)
            gain = base_err - control_err
            change = (control_pred - base_pred).abs().mean(dim=1, keepdim=True)
            residual = (padded[:, :, :height, :width] - base_pred).abs().mean(dim=1, keepdim=True)
            score_change = score * change
            gate_change = gate * change
            sample_count = min(args.pixel_sample_per_image, gain.numel())
            choice = rng.choice(gain.numel(), size=sample_count, replace=False)
            flat = lambda tensor: tensor.detach().cpu().numpy().reshape(-1)[choice]
            feature_vectors["d7c_score"].append(flat(score).astype(np.float32))
            feature_vectors["d7c_gate"].append(flat(gate).astype(np.float32))
            feature_vectors["control_change"].append(flat(change).astype(np.float32))
            feature_vectors["residual_abs_input_a0"].append(flat(residual).astype(np.float32))
            feature_vectors["score_x_change"].append(flat(score_change).astype(np.float32))
            feature_vectors["gate_x_change"].append(flat(gate_change).astype(np.float32))
            feature_vectors["control_change_inside_gate"].append(flat(gate_change).astype(np.float32))
            gain_sample = flat(gain).astype(np.float32)
            target_gain.append(gain_sample)
            target_positive.append(gain_sample > args.gain_eps)
            target_harmful.append(gain_sample < -args.gain_eps)
            hard_gate_sample.append(flat(gate).astype(np.float32) > 0.5)

            policy_masks = {
                "A0": torch.zeros_like(gate),
                "W_U_G_1": torch.ones_like(gate),
                "W_U_G_D": gate,
                "D7C_GAIN_ORACLE_POS": ((gain > args.gain_eps) & (gate > 0.5)).to(gate.dtype),
                "GLOBAL_GAIN_ORACLE_POS": (gain > args.gain_eps).to(gate.dtype),
            }
            for frac in args.top_fractions:
                policy_masks[f"D7C_TOP_CHANGE_{frac:g}"] = top_fraction_mask(change, gate, frac)
                policy_masks[f"D7C_TOP_SCOREXCHANGE_{frac:g}"] = top_fraction_mask(score_change, gate, frac)

            row = {
                "index": index,
                "name": name,
                "gate_coverage": gate.mean().item(),
                "ungated_positive_gain_fraction": (gain > args.gain_eps).float().mean().item(),
                "d7c_positive_gain_fraction": ((gain > args.gain_eps) & (gate > 0.5)).float().sum().item()
                / max(float((gate > 0.5).float().sum().item()), 1.0),
            }
            for policy in policy_names:
                pred = masked_replay(base_pred, control_pred, policy_masks[policy])
                psnr_val, ssim_val = metric_summary(pred, label)
                base_psnr, base_ssim = metric_summary(base_pred, label)
                row[f"{policy}_psnr_delta"] = psnr_val - base_psnr
                row[f"{policy}_ssim_delta"] = ssim_val - base_ssim
                row[f"{policy}_mask_mean"] = policy_masks[policy].mean().item()
            d7c_psnr, d7c_ssim = metric_summary(d7c_pred, label)
            base_psnr, base_ssim = metric_summary(base_pred, label)
            row["W_D_G_D_psnr_delta"] = d7c_psnr - base_psnr
            row["W_D_G_D_ssim_delta"] = d7c_ssim - base_ssim
            per_image_rows.append(row)
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(f"v3f_a_progress {index + 1}/{len(names)}", flush=True)

    features = {key: np.concatenate(value) for key, value in feature_vectors.items()}
    gains = np.concatenate(target_gain)
    positives = np.concatenate(target_positive)
    harmful = np.concatenate(target_harmful)
    gate_samples = np.concatenate(hard_gate_sample)
    feature_rows = []
    for key, values in features.items():
        feature_rows.append(
            {
                "feature": key,
                "positive_gain_auroc": auroc(values, positives),
                "positive_gain_auprc": auprc(values, positives),
                "gain_spearman": spearman(values, gains),
                "harmful_gain_auroc_inverted": auroc(-values, harmful),
                "mean": float(np.mean(values)),
                "p90": float(np.percentile(values, 90)),
            }
        )
    policy_rows = []
    for policy in policy_names + ["W_D_G_D"]:
        deltas = [row[f"{policy}_psnr_delta"] for row in per_image_rows]
        policy_rows.append(
            {
                "policy": policy,
                "mean_psnr_delta": mean(deltas),
                "median_psnr_delta": percentile(deltas, 50),
                "p10_psnr_delta": percentile(deltas, 10),
                "worst_psnr_delta": min(deltas),
                "positive_ratio": mean(1.0 if v > 0 else 0.0 for v in deltas),
                "regression_le_0p2_count": sum(v <= -0.2 for v in deltas),
                "regression_le_0p5_count": sum(v <= -0.5 for v in deltas),
                "mean_mask": mean(row.get(f"{policy}_mask_mean", float("nan")) for row in per_image_rows),
            }
        )
    best_proxy = max(feature_rows, key=lambda r: r["positive_gain_auroc"])
    oracle = next(row for row in policy_rows if row["policy"] == "D7C_GAIN_ORACLE_POS")
    wugd = next(row for row in policy_rows if row["policy"] == "W_U_G_D")
    wug1 = next(row for row in policy_rows if row["policy"] == "W_U_G_1")
    best_deploy = max(
        [row for row in policy_rows if row["policy"].startswith("D7C_TOP_")],
        key=lambda r: (r["mean_psnr_delta"], -r["regression_le_0p2_count"]),
    )
    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3f-A no-training correctability target and separability audit",
        "sample_count": len(names),
        "pixel_sample_count": int(len(gains)),
        "locked_test_touched": False,
        "training_authorized": False,
        "a0_checkpoint": args.a0_checkpoint,
        "a0_checkpoint_sha256": sha256_file(args.a0_checkpoint),
        "control_checkpoint": args.control_checkpoint,
        "control_checkpoint_sha256": sha256_file(args.control_checkpoint),
        "d7c_checkpoint": args.d7c_checkpoint,
        "d7c_checkpoint_sha256": sha256_file(args.d7c_checkpoint),
        "target": {
            "gain_eps": args.gain_eps,
            "positive_rate": float(np.mean(positives)),
            "harmful_rate": float(np.mean(harmful)),
            "d7c_gate_sample_coverage": float(np.mean(gate_samples)),
            "positive_rate_inside_gate": float(np.mean(positives[gate_samples])) if gate_samples.any() else float("nan"),
            "positive_rate_outside_gate": float(np.mean(positives[~gate_samples])) if (~gate_samples).any() else float("nan"),
        },
        "best_feature": best_proxy,
        "oracle_policy": oracle,
        "ungated_control_policy": wug1,
        "d7c_veto_control_policy": wugd,
        "best_deployable_proxy_policy": best_deploy,
        "decision": "V3F_A_CORRECTABILITY_TARGET_AUDIT_DONE",
        "interpretation": (
            "Review oracle gap and feature AUROC before authorizing any supervised "
            "operator-correctability ranker training."
        ),
    }
    if best_proxy["positive_gain_auroc"] < args.min_feature_auroc_for_training:
        summary["next_action"] = (
            "Do not train a lightweight ranker from the audited scalar proxies alone; "
            "close or redesign features/operator context before ranker training."
        )
        summary["training_authorized"] = False
        summary["decision"] = "V3F_A_SCALAR_PROXY_SEPARABILITY_WEAK_NO_RANKER_TRAINING"
    elif oracle["mean_psnr_delta"] <= wugd["mean_psnr_delta"] + args.min_oracle_gain_margin:
        summary["next_action"] = "Oracle correctability mask does not add enough value over D7c veto; do not train ranker."
        summary["training_authorized"] = False
        summary["decision"] = "V3F_A_ORACLE_CORRECTABILITY_UPPER_BOUND_WEAK_NO_RANKER_TRAINING"
    else:
        summary["next_action"] = "Authorize a separate v3f-B lightweight internal ranker screen with fixed features and no locked test."
        summary["training_authorized"] = True
        summary["decision"] = "V3F_A_AUTHORIZE_LIGHTWEIGHT_INTERNAL_RANKER_SCREEN_ONLY"

    write_csv(output_dir / "v3f_a_correctability_feature_separability.csv", feature_rows)
    write_csv(output_dir / "v3f_a_correctability_policy_summary.csv", policy_rows)
    write_csv(output_dir / "v3f_a_correctability_per_image.csv", per_image_rows)
    write_json(output_dir / "v3f_a_correctability_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--d7c_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--split_key", default="val_inner")
    parser.add_argument("--max_samples", type=int, default=600)
    parser.add_argument("--pixel_sample_per_image", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--d7c_threshold", type=float, default=D7C_THRESHOLD)
    parser.add_argument("--gain_eps", type=float, default=0.0)
    parser.add_argument("--top_fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--min_feature_auroc_for_training", type=float, default=0.56)
    parser.add_argument("--min_oracle_gain_margin", type=float, default=0.01)
    parser.add_argument("--progress_every", type=int, default=50)
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
