import argparse
import csv
import json
import os
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_ROOT))

from audit_haze4k_apdr_v0_4b_derived_basis import (  # noqa: E402
    collect_low_targets,
    feature_matrix,
    group_name,
    parse_int_list,
    pca_bases,
    safe_mean,
    sigma_label,
    summarize_rows,
    weighted_project,
    write_csv,
)
from overfit_haze4k_apdr_v0_4_freeparam_lowcolor import (  # noqa: E402
    build_apdr_model,
    build_loader,
    correlation,
    frozen_apdr_tensors,
    gaussian_lowpass,
    percentile,
    psnr,
)
from overfit_haze4k_apdr_v0_4a_lowfield import read_correctability  # noqa: E402


class CoeffRouterMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def standardizer(x, indices):
    source = x[indices]
    mean = source.mean(dim=0, keepdim=True)
    std = source.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return mean, std


def coeff_standardizer(coeffs, indices):
    source = coeffs[indices]
    mean = source.mean(dim=0, keepdim=True)
    std = source.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return mean, std


def normalize(x, mean, std):
    return (x - mean) / std


def tv_loss_from_flat(field, low_size):
    image = field.view(field.shape[0], 3, low_size, low_size)
    if image.shape[-1] < 2 or image.shape[-2] < 2:
        return image.new_tensor(0.0)
    dx = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()
    dy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean()
    return dx + dy


def weighted_smooth_l1(pred, target, weight, beta):
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    return (weight * loss).sum() / weight.sum().clamp_min(1e-12)


def corr_flat(x, y, weight=None):
    if weight is not None:
        keep = weight.flatten() > 0
        x = x.flatten()[keep]
        y = y.flatten()[keep]
    else:
        x = x.flatten()
        y = y.flatten()
    if x.numel() < 2:
        return None
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    if denom.item() <= 1e-12:
        return None
    return (x * y).sum().div(denom).item()


def choose_train_indices(meta_rows, fit_count, eval_count, train_count):
    scope_count = int(train_count) if int(train_count) > 0 else int(eval_count)
    if fit_count <= 0:
        active = [
            row["index"]
            for row in meta_rows
            if row["index"] < scope_count and row["low_weight_sum"] > 1e-8
        ]
        if not active:
            raise RuntimeError("No active samples inside the evaluation subset.")
        return active
    active = [row["index"] for row in meta_rows if row["low_weight_sum"] > 1e-8]
    if len(active) < fit_count:
        raise RuntimeError(f"Only {len(active)} active samples available, need {fit_count}.")
    return active[:fit_count]


def lowspace_metrics(indices, pred_coeffs, projection, low_data):
    bases = projection["bases"]
    mean = projection["mean"]
    coeffs = projection["coeffs"]
    targets = low_data["flat_targets"]
    weights = low_data["flat_weights"]
    pred_flat = mean + pred_coeffs @ bases
    target = targets[indices]
    weight = weights[indices]
    true_coeff = coeffs[indices]
    field_l1 = (weight * (pred_flat - target).abs()).sum().item() / max(weight.sum().item(), 1e-12)
    zero_l1 = (weight * target.abs()).sum().item() / max(weight.sum().item(), 1e-12)
    return {
        "count": len(indices),
        "coeff_mse": (pred_coeffs - true_coeff).square().mean().item(),
        "coeff_mae": (pred_coeffs - true_coeff).abs().mean().item(),
        "lowspace_weighted_field_l1": field_l1,
        "lowspace_zero_field_l1": zero_l1,
        "lowspace_field_l1_drop": (zero_l1 - field_l1) / max(zero_l1, 1e-12),
        "lowspace_field_corr": corr_flat(pred_flat, target, weight),
    }


def predict_coeffs(router, features, feature_mean, feature_std, coeff_mean, coeff_std):
    x = normalize(features, feature_mean, feature_std)
    pred_norm = router(x)
    return pred_norm * coeff_std + coeff_mean


def train_router_for_projection(features, projection, low_data, train_indices, args, device):
    feature_mean, feature_std = standardizer(features, train_indices)
    coeff_mean, coeff_std = coeff_standardizer(projection["coeffs"], train_indices)
    router = CoeffRouterMLP(
        input_dim=features.shape[1],
        output_dim=projection["K"],
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    x_train = normalize(features[train_indices], feature_mean, feature_std).to(device)
    coeff_target = projection["coeffs"][train_indices].to(device)
    coeff_mean_d = coeff_mean.to(device)
    coeff_std_d = coeff_std.to(device)
    coeff_target_norm = normalize(coeff_target, coeff_mean_d, coeff_std_d)
    bases = projection["bases"].to(device)
    field_mean = projection["mean"].to(device)
    target_flat = low_data["flat_targets"][train_indices].to(device)
    weight_flat = low_data["flat_weights"][train_indices].to(device)
    history = []
    for step in range(1, args.steps + 1):
        pred_norm = router(x_train)
        pred_coeff = pred_norm * coeff_std_d + coeff_mean_d
        pred_flat = field_mean + pred_coeff @ bases
        loss_coeff = F.smooth_l1_loss(pred_norm, coeff_target_norm, beta=args.coeff_beta)
        loss_field = weighted_smooth_l1(pred_flat, target_flat, weight_flat, args.field_beta)
        loss_tv = tv_loss_from_flat(pred_flat, projection["low_size"])
        loss = loss_field + args.coeff_lambda * loss_coeff + args.tv_lambda * loss_tv
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(router.parameters(), args.grad_clip_norm)
        optimizer.step()
        if step == 1 or (args.progress_freq and step % args.progress_freq == 0) or step == args.steps:
            with torch.no_grad():
                pred_eval = pred_coeff.detach().cpu()
                metrics = lowspace_metrics(train_indices, pred_eval, projection, low_data)
            history.append(
                {
                    "K": projection["K"],
                    "low_size": projection["low_size"],
                    "step": step,
                    "loss": loss.item(),
                    "loss_field": loss_field.item(),
                    "loss_coeff": loss_coeff.item(),
                    "loss_tv": loss_tv.item(),
                    **metrics,
                }
            )
            print(
                f"K={projection['K']} step={step} loss={loss.item():.6f} "
                f"field_l1={metrics['lowspace_weighted_field_l1']:.6f} "
                f"coeff_mse={metrics['coeff_mse']:.6f}",
                flush=True,
            )
    return {
        "router": router.cpu().eval(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "coeff_mean": coeff_mean,
        "coeff_std": coeff_std,
        "history": history,
        "train_lowspace": history[-1],
    }


def gateb_checks(summary):
    return {
        "weighted_delta_l1_drop": {
            "observed": summary["weighted_delta_l1_drop"],
            "required": ">= 0.50",
            "pass": summary["weighted_delta_l1_drop"] >= 0.50,
        },
        "pred_target_corr": {
            "observed": summary["pred_target_corr"],
            "required": ">= 0.50",
            "pass": summary["pred_target_corr"] is not None and summary["pred_target_corr"] >= 0.50,
        },
        "oracle_recovery": {
            "observed": summary["oracle_recovery"],
            "required": ">= 0.30",
            "pass": summary["oracle_recovery"] >= 0.30,
        },
        "hard_train_gain": {
            "observed": summary["hard_bottom25_output_gain"],
            "required": ">= +0.30 dB",
            "pass": summary["hard_bottom25_output_gain"] >= 0.30,
        },
        "easy_train_gain": {
            "observed": summary["easy_top25_output_gain"],
            "required": ">= -0.010 dB",
            "pass": summary["easy_top25_output_gain"] >= -0.010,
        },
        "strong_reference_regressions": {
            "observed": summary["strong_reference_regressions"],
            "required": "== 0",
            "pass": summary["strong_reference_regressions"] == 0,
        },
        "severe_regressions": {
            "observed": summary["severe_regressions"],
            "required": "== 0",
            "pass": summary["severe_regressions"] == 0,
        },
    }


def split_summary(rows):
    den = sum(row["weighted_den"] for row in rows)
    initial = sum(row["initial_l1_num"] for row in rows) / max(den, 1e-12)
    final = sum(row["final_l1_num"] for row in rows) / max(den, 1e-12)
    summary = summarize_rows(rows, initial, final, safe_mean([row["corr"] for row in rows]))
    summary["weighted_den"] = den
    return summary


def evaluate_full(apdr_model, loader, device, args, tau, scores, features, projections, router_states):
    combo_rows = {key: [] for key in projections}
    sums = {key: {"initial": 0.0, "final": 0.0, "den": 0.0, "corrs": []} for key in projections}
    for index, (input_img, label_img, name) in enumerate(loader):
        image_name = name[0]
        input_img = input_img.to(device)
        label_img = label_img.to(device)
        anchor, m_safe, _delta_star, low_target, _color_target = frozen_apdr_tensors(
            apdr_model,
            input_img,
            label_img,
            args,
        )
        score = scores.get(image_name)
        if score is None:
            raise KeyError(f"Missing correctability score for {image_name}")
        p_benefit = 1.0 if score >= tau else 0.0
        weight = m_safe * p_benefit
        expanded = weight.expand_as(low_target)
        oracle = (anchor + weight * low_target).clamp(0, 1)
        anchor_psnr = psnr(anchor, label_img)
        oracle_psnr = psnr(oracle, label_img)
        split = "train" if args.train_count <= 0 or index < args.train_count else "mini_val"
        for key, projection in projections.items():
            state = router_states[key]
            router = state["router"]
            with torch.no_grad():
                pred_coeff = predict_coeffs(
                    router,
                    features[index : index + 1],
                    state["feature_mean"],
                    state["feature_std"],
                    state["coeff_mean"],
                    state["coeff_std"],
                )
                low_size = projection["low_size"]
                pred_low = projection["mean"] + pred_coeff[0] @ projection["bases"]
                pred = pred_low.view(1, 3, low_size, low_size).to(device)
                pred = F.interpolate(pred, size=anchor.shape[-2:], mode="bilinear", align_corners=False)
                pred = gaussian_lowpass(pred, args.kernel_size, args.sigma)
            output = (anchor + weight * pred).clamp(0, 1)
            output_psnr = psnr(output, label_img)
            corr = correlation(pred.cpu(), low_target.cpu(), expanded.cpu())
            initial_num = (expanded * low_target.abs()).sum().item()
            final_num = (expanded * (pred - low_target).abs()).sum().item()
            weighted_den = expanded.sum().item()
            if corr is not None:
                sums[key]["corrs"].append(corr)
            sums[key]["initial"] += initial_num
            sums[key]["final"] += final_num
            sums[key]["den"] += weighted_den
            combo_rows[key].append(
                {
                    "low_size": projection["low_size"],
                    "K": projection["K"],
                    "split": split,
                    "index": index,
                    "name": image_name,
                    "anchor_psnr": anchor_psnr,
                    "output_psnr": output_psnr,
                    "oracle_psnr": oracle_psnr,
                    "output_gain": output_psnr - anchor_psnr,
                    "oracle_gain": oracle_psnr - anchor_psnr,
                    "corr": corr,
                    "P_benefit": p_benefit,
                    "proxy_score": score,
                    "M_safe_mean": m_safe.mean().item(),
                    "target_abs_mean": low_target.abs().mean().item(),
                    "pred_abs_mean": pred.abs().mean().item(),
                    "weighted_den": weighted_den,
                    "initial_l1_num": initial_num,
                    "final_l1_num": final_num,
                    "weighted_field_l1": final_num / max(weighted_den, 1e-12),
                }
            )
    summaries = {}
    split_summaries = []
    per_image = []
    group_rows = []
    for key, rows in combo_rows.items():
        den = max(sums[key]["den"], 1e-12)
        summary = summarize_rows(
            rows,
            sums[key]["initial"] / den,
            sums[key]["final"] / den,
            safe_mean(sums[key]["corrs"]),
        )
        summary["low_size"] = projections[key]["low_size"]
        summary["K"] = projections[key]["K"]
        summary["gateb_checks"] = gateb_checks(summary)
        summary["gateb_pass"] = all(item["pass"] for item in summary["gateb_checks"].values())
        summaries[key] = summary
        for split_name in sorted({row["split"] for row in rows}):
            members = [row for row in rows if row["split"] == split_name]
            item = split_summary(members)
            item["low_size"] = projections[key]["low_size"]
            item["K"] = projections[key]["K"]
            item["split"] = split_name
            item["gateb_checks"] = gateb_checks(item)
            item["gateb_pass"] = all(check["pass"] for check in item["gateb_checks"].values())
            split_summaries.append(item)
        hard_cut = percentile([row["anchor_psnr"] for row in rows], 25)
        easy_cut = percentile([row["anchor_psnr"] for row in rows], 75)
        for row in rows:
            per_image.append({**row, "group": group_name(row, hard_cut, easy_cut)})
        for group in sorted({group_name(row, hard_cut, easy_cut) for row in rows}):
            members = [row for row in rows if group_name(row, hard_cut, easy_cut) == group]
            group_rows.append(
                {
                    "low_size": projections[key]["low_size"],
                    "K": projections[key]["K"],
                    "group": group,
                    "count": len(members),
                    "mean_output_gain": statistics.mean(row["output_gain"] for row in members),
                    "mean_oracle_gain": statistics.mean(row["oracle_gain"] for row in members),
                    "mean_corr": safe_mean([row["corr"] for row in members]),
                }
            )
    return summaries, split_summaries, per_image, group_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--correctability_json", required=True)
    parser.add_argument("--correctability_train_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_4b_basis_router_gateb_sigma3_seed3407")
    parser.add_argument("--stage_label", default=None)
    parser.add_argument("--artifact_prefix", default="basis_router_gateb")
    parser.add_argument("--basis_num_images", type=int, default=0)
    parser.add_argument("--train_count", type=int, default=0)
    parser.add_argument("--eval_count", type=int, default=128)
    parser.add_argument("--fit_count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca_device", default="cpu")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--low_size", type=int, default=32)
    parser.add_argument("--k_values", default="16,32")
    parser.add_argument("--projection_ridge", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--field_beta", type=float, default=0.01)
    parser.add_argument("--coeff_beta", type=float, default=0.1)
    parser.add_argument("--coeff_lambda", type=float, default=0.2)
    parser.add_argument("--tv_lambda", type=float, default=0.001)
    parser.add_argument("--pca_oversample", type=int, default=8)
    parser.add_argument("--pca_niter", type=int, default=4)
    parser.add_argument("--progress_freq", type=int, default=100)
    args = parser.parse_args()

    if args.eval_count <= 0:
        raise ValueError("--eval_count must be positive")
    if args.train_count < 0:
        raise ValueError("--train_count must be non-negative")
    if args.fit_count < 0:
        raise ValueError("--fit_count must be non-negative")
    if args.train_count > args.eval_count:
        raise ValueError("--train_count cannot exceed --eval_count")
    if args.basis_num_images > 0 and args.basis_num_images < args.eval_count:
        raise ValueError("--basis_num_images must be 0/full or at least --eval_count")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.pca_device == "cuda" and not torch.cuda.is_available():
        args.pca_device = "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = sigma_label(args.sigma)
    k_values = parse_int_list(args.k_values)
    stage_label = args.stage_label
    if stage_label is None:
        if args.train_count > 0 and args.eval_count > args.train_count:
            stage_label = "APDR-v0.4B basis-only coefficient router Gate C train/mini-val"
        else:
            stage_label = "APDR-v0.4B basis-only coefficient router Gate B"

    tau, scores = read_correctability(args.correctability_json, args.correctability_train_csv)
    apdr_model = build_apdr_model(args.selector_checkpoint, args.residual_max, device)
    collect_loader = build_loader(args.data_dir, args.basis_num_images, args.num_workers, shuffle=False)
    low_targets, low_weights, feature_rows, meta_rows = collect_low_targets(
        apdr_model,
        collect_loader,
        device,
        args,
        tau,
        scores,
        [args.low_size],
    )
    feature_names, features = feature_matrix(feature_rows)
    pca = pca_bases(low_targets[args.low_size], low_weights[args.low_size], k_values, args)
    flat_targets = low_targets[args.low_size].flatten(1)
    flat_weights = low_weights[args.low_size].repeat(1, 3, 1, 1).flatten(1).clamp_min(0.0)
    low_data = {"flat_targets": flat_targets, "flat_weights": flat_weights, "pca": pca}
    train_indices = choose_train_indices(meta_rows, args.fit_count, args.eval_count, args.train_count)
    projections = {}
    router_states = {}
    history = []
    train_rows = []
    for k in k_values:
        bases = pca["bases"][:k]
        coeffs, recons = weighted_project(
            flat_targets,
            flat_weights,
            pca["mean"],
            bases,
            args.projection_ridge,
        )
        key = (args.low_size, k)
        projections[key] = {
            "low_size": args.low_size,
            "K": k,
            "coeffs": coeffs,
            "recons": recons,
            "bases": bases,
            "mean": pca["mean"],
            "explained_weighted_energy": pca["explained"].get(k),
        }
        router_states[key] = train_router_for_projection(
            features,
            projections[key],
            low_data,
            train_indices,
            args,
            device,
        )
        history.extend(router_states[key]["history"])
        final_train = router_states[key]["train_lowspace"]
        train_rows.append(final_train)

    eval_loader = build_loader(args.data_dir, args.eval_count, args.num_workers, shuffle=False)
    summaries, split_summaries, per_image, group_rows = evaluate_full(
        apdr_model,
        eval_loader,
        device,
        args,
        tau,
        scores,
        features,
        projections,
        router_states,
    )
    result = {
        "stage": stage_label,
        "tag": args.tag,
        "sigma": args.sigma,
        "correctability_tau": tau,
        "basis_num_images": len(meta_rows),
        "train_count": args.train_count,
        "train_scope_count": args.train_count if args.train_count > 0 else args.eval_count,
        "fit_count": args.fit_count,
        "eval_count": args.eval_count,
        "train_indices": train_indices,
        "feature_names": feature_names,
        "summaries": list(summaries.values()),
        "split_summaries": split_summaries,
        "train_lowspace": train_rows,
        "args": vars(args),
    }
    prefix = args.artifact_prefix
    summary_path = output_dir / f"{prefix}_summary_{label}.json"
    summary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / f"{prefix}_history_{label}.csv", history)
    write_csv(output_dir / f"{prefix}_per_image_{label}.csv", per_image)
    write_csv(output_dir / f"{prefix}_groups_{label}.csv", group_rows)
    write_csv(output_dir / f"{prefix}_train_lowspace_{label}.csv", train_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
