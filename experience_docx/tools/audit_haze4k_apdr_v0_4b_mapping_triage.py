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
    corr_flat,
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


def write_csv_union(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(value):
    out = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise ValueError(f"Empty float list: {value}")
    return out


def standardizer(source):
    mean = source.mean(dim=0, keepdim=True)
    std = source.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return mean, std


def normalize(x, mean, std):
    return (x - mean) / std


def fit_ridge(x_train, y_train, ridge):
    ones = torch.ones(x_train.shape[0], 1, dtype=torch.float32)
    x_aug = torch.cat([x_train, ones], dim=1)
    gram = x_aug.t() @ x_aug
    reg = torch.eye(gram.shape[0], dtype=torch.float32) * float(ridge)
    reg[-1, -1] = 0.0
    rhs = x_aug.t() @ y_train
    return torch.linalg.solve(gram + reg, rhs)


def predict_ridge(x, weights):
    ones = torch.ones(x.shape[0], 1, dtype=torch.float32)
    return torch.cat([x, ones], dim=1) @ weights


def fit_pls(x_train, y_train, components, eps=1e-8, max_iter=100):
    x_mean, x_std = standardizer(x_train)
    y_mean, y_std = standardizer(y_train)
    x_work = normalize(x_train, x_mean, x_std).clone()
    y_work = normalize(y_train, y_mean, y_std).clone()
    max_components = min(int(components), x_train.shape[0] - 1, x_train.shape[1], y_train.shape[1])
    if max_components <= 0:
        raise ValueError("PLS needs at least one component.")
    weights = []
    loadings = []
    y_loadings = []
    for _ in range(max_components):
        u = y_work[:, :1].clone()
        if u.square().sum().item() <= eps:
            break
        for _inner in range(max_iter):
            w = x_work.t() @ u
            w = w / w.norm().clamp_min(eps)
            t = x_work @ w
            q = y_work.t() @ t / (t.t() @ t).clamp_min(eps)
            q_norm = q.norm().clamp_min(eps)
            u_next = y_work @ q / q_norm
            if (u_next - u).abs().max().item() < 1e-6:
                u = u_next
                break
            u = u_next
        p = x_work.t() @ t / (t.t() @ t).clamp_min(eps)
        q = y_work.t() @ t / (t.t() @ t).clamp_min(eps)
        x_work = x_work - t @ p.t()
        y_work = y_work - t @ q.t()
        weights.append(w.squeeze(1))
        loadings.append(p.squeeze(1))
        y_loadings.append(q.squeeze(1))
        if x_work.square().mean().item() <= eps:
            break
    if not weights:
        raise RuntimeError("PLS failed to extract a component.")
    w_mat = torch.stack(weights, dim=1)
    p_mat = torch.stack(loadings, dim=1)
    q_mat = torch.stack(y_loadings, dim=1)
    beta = w_mat @ torch.linalg.pinv(p_mat.t() @ w_mat) @ q_mat.t()
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "beta": beta,
        "components": w_mat.shape[1],
    }


def predict_pls(x, model):
    y_norm = normalize(x, model["x_mean"], model["x_std"]) @ model["beta"]
    return y_norm * model["y_std"] + model["y_mean"]


def pairwise_dist(x, y):
    return torch.cdist(x.float(), y.float(), p=2)


def knn_predict(train_x, train_y, eval_x, k, kernel=False):
    distances = pairwise_dist(eval_x, train_x)
    k = max(1, min(int(k), train_x.shape[0]))
    top_dist, top_idx = torch.topk(distances, k=k, dim=1, largest=False)
    gathered = train_y[top_idx]
    if kernel:
        bandwidth = top_dist[:, -1:].median().clamp_min(1e-6)
        weights = torch.exp(-top_dist.square() / (2.0 * bandwidth.square()))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pred = (gathered * weights.unsqueeze(-1)).sum(dim=1)
        confidence = weights.max(dim=1).values
    else:
        pred = gathered.mean(dim=1)
        confidence = torch.full((eval_x.shape[0],), 1.0 / float(k), dtype=torch.float32)
    return pred, {
        "nn_distance": top_dist[:, 0],
        "knn_mean_distance": top_dist.mean(dim=1),
        "kernel_confidence": confidence,
    }


class TinyCoeffMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def train_early_mlp(x_train_raw, y_train_raw, x_eval_raw, args, seed):
    rng = random.Random(seed)
    order = list(range(x_train_raw.shape[0]))
    rng.shuffle(order)
    valid_count = max(1, min(len(order) // 5, len(order) - 1))
    valid_idx = order[:valid_count]
    train_idx = order[valid_count:]
    feature_mean, feature_std = standardizer(x_train_raw[train_idx])
    y_mean, y_std = standardizer(y_train_raw[train_idx])
    x_train = normalize(x_train_raw[train_idx], feature_mean, feature_std)
    y_train = normalize(y_train_raw[train_idx], y_mean, y_std)
    x_valid = normalize(x_train_raw[valid_idx], feature_mean, feature_std)
    y_valid = normalize(y_train_raw[valid_idx], y_mean, y_std)
    model = TinyCoeffMLP(x_train.shape[1], y_train.shape[1], args.mlp_hidden_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.mlp_learning_rate,
        weight_decay=args.mlp_weight_decay,
    )
    best_state = None
    best_loss = float("inf")
    stale = 0
    history = []
    for step in range(1, args.mlp_steps + 1):
        pred = model(x_train)
        loss = F.smooth_l1_loss(pred, y_train, beta=args.mlp_beta)
        optimizer.zero_grad()
        loss.backward()
        if args.mlp_grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.mlp_grad_clip_norm)
        optimizer.step()
        with torch.no_grad():
            valid_loss = F.smooth_l1_loss(model(x_valid), y_valid, beta=args.mlp_beta).item()
        if valid_loss + 1e-8 < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if step == 1 or step % args.progress_freq == 0 or step == args.mlp_steps:
            history.append({"step": step, "train_loss": loss.item(), "valid_loss": valid_loss})
        if stale >= args.mlp_patience:
            history.append({"step": step, "train_loss": loss.item(), "valid_loss": valid_loss})
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_eval = model(normalize(x_eval_raw, feature_mean, feature_std)) * y_std + y_mean
    return pred_eval, {
        "mlp_best_valid_loss": best_loss,
        "mlp_steps_used": history[-1]["step"] if history else args.mlp_steps,
        "mlp_internal_valid_count": len(valid_idx),
    }, history


def coeff_summary(pred, truth):
    pred = pred.float()
    truth = truth.float()
    err = pred - truth
    mse = err.square().mean().item()
    mae = err.abs().mean().item()
    truth_mse_floor = truth.square().mean().item()
    truth_mae_floor = truth.abs().mean().item()
    keep = truth.abs() > 1e-8
    sign_acc = None
    if keep.any().item():
        sign_acc = (pred[keep].sign() == truth[keep].sign()).float().mean().item()
    pred_norm = pred.norm(dim=1)
    truth_norm = truth.norm(dim=1).clamp_min(1e-12)
    return {
        "coeff_mse": mse,
        "coeff_mae": mae,
        "coeff_mse_norm": mse / max(truth_mse_floor, 1e-12),
        "coeff_mae_norm": mae / max(truth_mae_floor, 1e-12),
        "coeff_corr": corr_flat(pred, truth),
        "sign_accuracy": sign_acc,
        "pred_true_coeff_norm_ratio": (pred_norm / truth_norm).mean().item(),
    }


def split_summary(rows):
    den = sum(row["weighted_den"] for row in rows)
    initial = sum(row["initial_l1_num"] for row in rows) / max(den, 1e-12)
    final = sum(row["final_l1_num"] for row in rows) / max(den, 1e-12)
    summary = summarize_rows(rows, initial, final, safe_mean([row["corr"] for row in rows]))
    summary["weighted_den"] = den
    return summary


def ks_statistic(a, b):
    a = sorted(float(x) for x in a)
    b = sorted(float(x) for x in b)
    if not a or not b:
        return None
    i = 0
    j = 0
    best = 0.0
    while i < len(a) or j < len(b):
        if j >= len(b) or (i < len(a) and a[i] <= b[j]):
            value = a[i]
        else:
            value = b[j]
        while i < len(a) and a[i] <= value:
            i += 1
        while j < len(b) and b[j] <= value:
            j += 1
        best = max(best, abs(i / len(a) - j / len(b)))
    return best


def rbf_mmd(x, y):
    if x.shape[0] == 0 or y.shape[0] == 0:
        return None
    pooled = torch.cat([x, y], dim=0)
    d = pairwise_dist(pooled, pooled).square()
    positive = d[d > 0]
    gamma = 1.0 / positive.median().clamp_min(1e-6) if positive.numel() else torch.tensor(1.0)
    k_xx = torch.exp(-gamma * pairwise_dist(x, x).square()).mean()
    k_yy = torch.exp(-gamma * pairwise_dist(y, y).square()).mean()
    k_xy = torch.exp(-gamma * pairwise_dist(x, y).square()).mean()
    return (k_xx + k_yy - 2.0 * k_xy).item()


def feature_shift_rows(features, feature_names, train_indices, mini_indices):
    train_x = features[train_indices]
    mini_x = features[mini_indices]
    rows = []
    train_mean, train_std = standardizer(train_x)
    train_std = train_std.squeeze(0)
    for idx, name in enumerate(feature_names):
        a = train_x[:, idx]
        b = mini_x[:, idx]
        pooled_std = torch.cat([a, b]).std(unbiased=False).item()
        rows.append(
            {
                "feature": name,
                "train_count": len(train_indices),
                "mini_val_count": len(mini_indices),
                "train_mean": a.mean().item(),
                "mini_val_mean": b.mean().item(),
                "train_std": a.std(unbiased=False).item(),
                "mini_val_std": b.std(unbiased=False).item(),
                "standardized_mean_diff": (b.mean().item() - a.mean().item()) / max(pooled_std, 1e-12),
                "ks_stat": ks_statistic(a.tolist(), b.tolist()),
                "mmd_rbf": None,
                "mini_val_nn_distance_mean": None,
                "mini_val_nn_distance_p50": None,
                "mini_val_nn_distance_p90": None,
            }
        )
    train_z = normalize(train_x, train_mean, train_std.unsqueeze(0))
    mini_z = normalize(mini_x, train_mean, train_std.unsqueeze(0))
    nn_dist = pairwise_dist(mini_z, train_z).min(dim=1).values
    sorted_nn = sorted(nn_dist.tolist())
    p50 = sorted_nn[len(sorted_nn) // 2] if sorted_nn else None
    p90 = sorted_nn[min(len(sorted_nn) - 1, int(0.9 * len(sorted_nn)))] if sorted_nn else None
    rows.append(
        {
            "feature": "__all__",
            "train_count": len(train_indices),
            "mini_val_count": len(mini_indices),
            "train_mean": None,
            "mini_val_mean": None,
            "train_std": None,
            "mini_val_std": None,
            "standardized_mean_diff": None,
            "ks_stat": None,
            "mmd_rbf": rbf_mmd(train_z, mini_z),
            "mini_val_nn_distance_mean": nn_dist.mean().item() if nn_dist.numel() else None,
            "mini_val_nn_distance_p50": p50,
            "mini_val_nn_distance_p90": p90,
        }
    )
    return rows


def per_component_cv_rows(features, projections, active_indices, args):
    rows = []
    folds = list(range(len(active_indices)))
    rng = random.Random(args.seed)
    rng.shuffle(folds)
    fold_count = max(2, min(args.folds, len(active_indices)))
    fold_buckets = [folds[idx::fold_count] for idx in range(fold_count)]
    for key, projection in projections.items():
        coeffs = projection["coeffs"]
        pred_all = torch.zeros(len(active_indices), projection["K"], dtype=torch.float32)
        truth_all = coeffs[active_indices]
        active_pos = {image_index: pos for pos, image_index in enumerate(active_indices)}
        for bucket in fold_buckets:
            valid_indices = [active_indices[pos] for pos in bucket]
            valid_set = set(valid_indices)
            train_indices = [idx for idx in active_indices if idx not in valid_set]
            x_mean, x_std = standardizer(features[train_indices])
            x_train = normalize(features[train_indices], x_mean, x_std)
            x_valid = normalize(features[valid_indices], x_mean, x_std)
            weights = fit_ridge(x_train, coeffs[train_indices], args.cv_ridge)
            pred = predict_ridge(x_valid, weights)
            for row_pos, image_index in enumerate(valid_indices):
                pred_all[active_pos[image_index]] = pred[row_pos]
        for comp in range(projection["K"]):
            pred = pred_all[:, comp]
            truth = truth_all[:, comp]
            mse = (pred - truth).square().mean().item()
            var = (truth - truth.mean()).square().mean().item()
            rows.append(
                {
                    "low_size": projection["low_size"],
                    "K": projection["K"],
                    "component": comp,
                    "active_count": len(active_indices),
                    "ridge": args.cv_ridge,
                    "coeff_mse": mse,
                    "coeff_r2": 1.0 - mse / max(var, 1e-12),
                    "coeff_corr": corr_flat(pred, truth),
                    "truth_std": truth.std(unbiased=False).item(),
                    "pred_std": pred.std(unbiased=False).item(),
                    "pred_true_norm_ratio": pred.abs().mean().item() / max(truth.abs().mean().item(), 1e-12),
                }
            )
    return rows


def make_predictions(features, projections, train_indices, eval_indices, args):
    predictions = {}
    x_mean, x_std = standardizer(features[train_indices])
    x_train = normalize(features[train_indices], x_mean, x_std)
    x_eval = normalize(features[eval_indices], x_mean, x_std)
    nn_distance = pairwise_dist(x_eval, x_train).min(dim=1).values
    base_extra = {
        int(eval_indices[pos]): {
            "nn_distance": nn_distance[pos].item(),
            "confidence_proxy": 1.0 / (1.0 + nn_distance[pos].item()),
        }
        for pos in range(len(eval_indices))
    }
    mlp_history = []
    for key, projection in projections.items():
        coeffs = projection["coeffs"]
        train_y = coeffs[train_indices]
        k_dim = projection["K"]
        zeros = torch.zeros(len(eval_indices), k_dim, dtype=torch.float32)
        predictions[(key, "zero_field")] = {
            "pred_coeffs": zeros,
            "zero_field": True,
            "extras": base_extra,
            "family": "zero",
        }
        mean_coeff = train_y.mean(dim=0, keepdim=True).repeat(len(eval_indices), 1)
        predictions[(key, "mean_coeff")] = {
            "pred_coeffs": mean_coeff,
            "zero_field": False,
            "extras": base_extra,
            "family": "mean",
        }
        for ridge in args.ridge_values:
            weights = fit_ridge(x_train, train_y, ridge)
            name = f"ridge_{ridge:g}"
            predictions[(key, name)] = {
                "pred_coeffs": predict_ridge(x_eval, weights),
                "zero_field": False,
                "extras": base_extra,
                "family": "ridge",
                "ridge": ridge,
            }
        for comp in args.pls_components:
            if comp > min(x_train.shape[0] - 1, x_train.shape[1], train_y.shape[1]):
                continue
            model = fit_pls(features[train_indices], train_y, comp)
            name = f"pls_{model['components']}"
            predictions[(key, name)] = {
                "pred_coeffs": predict_pls(features[eval_indices], model),
                "zero_field": False,
                "extras": base_extra,
                "family": "pls",
                "components": model["components"],
            }
        for knn_k in args.knn_values:
            pred, extra = knn_predict(x_train, train_y, x_eval, knn_k, kernel=False)
            merged = {}
            for pos, image_index in enumerate(eval_indices):
                merged[int(image_index)] = {
                    **base_extra[int(image_index)],
                    "knn_mean_distance": extra["knn_mean_distance"][pos].item(),
                    "kernel_confidence": extra["kernel_confidence"][pos].item(),
                }
            predictions[(key, f"knn_{knn_k}")] = {
                "pred_coeffs": pred,
                "zero_field": False,
                "extras": merged,
                "family": "knn",
                "knn_k": knn_k,
            }
            pred, extra = knn_predict(x_train, train_y, x_eval, knn_k, kernel=True)
            merged = {}
            for pos, image_index in enumerate(eval_indices):
                merged[int(image_index)] = {
                    **base_extra[int(image_index)],
                    "knn_mean_distance": extra["knn_mean_distance"][pos].item(),
                    "kernel_confidence": extra["kernel_confidence"][pos].item(),
                }
            predictions[(key, f"kernel_knn_{knn_k}")] = {
                "pred_coeffs": pred,
                "zero_field": False,
                "extras": merged,
                "family": "kernel_knn",
                "knn_k": knn_k,
            }
        pred, extra, history = train_early_mlp(
            features[train_indices],
            train_y,
            features[eval_indices],
            args,
            args.seed + k_dim,
        )
        for row in history:
            mlp_history.append({"low_size": projection["low_size"], "K": projection["K"], **row})
        predictions[(key, "mlp_early_h32_wd1e-2")] = {
            "pred_coeffs": pred,
            "zero_field": False,
            "extras": {
                int(image_index): {**base_extra[int(image_index)], **extra}
                for image_index in eval_indices
            },
            "family": "mlp_early",
            **extra,
        }
    return predictions, mlp_history


def evaluate_predictions(apdr_model, loader, device, args, tau, scores, projections, predictions, eval_indices):
    per_image = []
    eval_set = set(eval_indices)
    for index, (input_img, label_img, name) in enumerate(loader):
        if index not in eval_set:
            continue
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
        split = "train" if index < args.train_count else "mini_val"
        for (key, mapper_name), item in predictions.items():
            low_size, k = key
            projection = projections[key]
            pos = list(eval_indices).index(index)
            pred_coeff = item["pred_coeffs"][pos : pos + 1]
            if item.get("zero_field"):
                pred_low = torch.zeros(1, 3, low_size, low_size)
            else:
                pred_low = (projection["mean"] + pred_coeff[0] @ projection["bases"]).view(
                    1,
                    3,
                    low_size,
                    low_size,
                )
            pred = F.interpolate(
                pred_low.to(device),
                size=anchor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pred = gaussian_lowpass(pred, args.kernel_size, args.sigma)
            output = (anchor + weight * pred).clamp(0, 1)
            output_psnr = psnr(output, label_img)
            corr = correlation(pred.cpu(), low_target.cpu(), expanded.cpu())
            initial_num = (expanded * low_target.abs()).sum().item()
            final_num = (expanded * (pred - low_target).abs()).sum().item()
            weighted_den = expanded.sum().item()
            true_coeff = projection["coeffs"][index]
            coeff_err = coeff_summary(pred_coeff.cpu(), true_coeff.view(1, -1))
            row = {
                "low_size": low_size,
                "K": k,
                "mapper": mapper_name,
                "family": item.get("family"),
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
                **coeff_err,
                **item.get("extras", {}).get(index, {}),
            }
            per_image.append(row)
        if args.progress_freq and (index + 1) % args.progress_freq == 0:
            print(f"evaluated={index + 1}", flush=True)
    return per_image


def aggregate_mapper_rows(per_image, projections, predictions, eval_indices):
    rows = []
    coeff_rows = []
    group_rows = []
    eval_pos = {int(image_index): pos for pos, image_index in enumerate(eval_indices)}
    for (key, mapper_name), item in predictions.items():
        low_size, k = key
        members_all = [
            row for row in per_image if row["low_size"] == low_size and row["K"] == k and row["mapper"] == mapper_name
        ]
        hard_cut = percentile([row["anchor_psnr"] for row in members_all], 25)
        easy_cut = percentile([row["anchor_psnr"] for row in members_all], 75)
        for row in members_all:
            row["group"] = group_name(row, hard_cut, easy_cut)
        for split_name in ("train", "mini_val", "overall"):
            members = members_all if split_name == "overall" else [row for row in members_all if row["split"] == split_name]
            open_members = [row for row in members if row["P_benefit"] >= 0.5]
            if not members:
                continue
            summary = split_summary(members)
            coeff_metric = {
                "coeff_mse": None,
                "coeff_mae": None,
                "coeff_mse_norm": None,
                "coeff_mae_norm": None,
                "coeff_corr": None,
                "sign_accuracy": None,
                "pred_true_coeff_norm_ratio": None,
            }
            if open_members:
                indices = [int(row["index"]) for row in open_members]
                positions = [eval_pos[index] for index in indices]
                coeff_metric = coeff_summary(
                    item["pred_coeffs"][positions],
                    projections[key]["coeffs"][indices],
                )
            rows.append(
                {
                    "low_size": low_size,
                    "K": k,
                    "mapper": mapper_name,
                    "family": item.get("family"),
                    "split": split_name,
                    "count": len(members),
                    "open_count": len(open_members),
                    **summary,
                    **coeff_metric,
                    "nn_distance_mean": safe_mean([row.get("nn_distance") for row in open_members]),
                    "confidence_proxy_mean": safe_mean([row.get("confidence_proxy") for row in open_members]),
                }
            )
            coeff_rows.append(
                {
                    "low_size": low_size,
                    "K": k,
                    "mapper": mapper_name,
                    "split": split_name,
                    "count": len(open_members),
                    **coeff_metric,
                    "lowspace_weighted_field_l1": safe_mean(
                        [row["weighted_field_l1"] for row in open_members]
                    ),
                }
            )
        for group in sorted({row["group"] for row in members_all}):
            members = [row for row in members_all if row["group"] == group]
            group_rows.append(
                {
                    "low_size": low_size,
                    "K": k,
                    "mapper": mapper_name,
                    "family": item.get("family"),
                    "group": group,
                    "count": len(members),
                    "mean_output_gain": statistics.mean(row["output_gain"] for row in members),
                    "mean_oracle_gain": statistics.mean(row["oracle_gain"] for row in members),
                    "mean_corr": safe_mean([row["corr"] for row in members]),
                    "mean_coeff_mse_norm": safe_mean([row["coeff_mse_norm"] for row in members]),
                    "mean_nn_distance": safe_mean([row.get("nn_distance") for row in members]),
                }
            )
    return rows, coeff_rows, group_rows


def open_easy_failure_rows(per_image):
    rows = []
    for row in per_image:
        if row.get("split") == "mini_val" and row.get("group") == "open_easy":
            rows.append(
                {
                    "low_size": row["low_size"],
                    "K": row["K"],
                    "mapper": row["mapper"],
                    "family": row["family"],
                    "index": row["index"],
                    "name": row["name"],
                    "anchor_psnr": row["anchor_psnr"],
                    "oracle_gain": row["oracle_gain"],
                    "output_gain": row["output_gain"],
                    "P_benefit": row["P_benefit"],
                    "proxy_score": row["proxy_score"],
                    "M_safe_mean": row["M_safe_mean"],
                    "coeff_mse_norm": row["coeff_mse_norm"],
                    "coeff_mae_norm": row["coeff_mae_norm"],
                    "coeff_corr": row["coeff_corr"],
                    "sign_accuracy": row["sign_accuracy"],
                    "pred_true_coeff_norm_ratio": row["pred_true_coeff_norm_ratio"],
                    "weighted_field_l1": row["weighted_field_l1"],
                    "nn_distance": row.get("nn_distance"),
                    "confidence_proxy": row.get("confidence_proxy"),
                    "kernel_confidence": row.get("kernel_confidence"),
                }
            )
    return sorted(rows, key=lambda row: (row["K"], row["mapper"], row["output_gain"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--correctability_json", required=True)
    parser.add_argument("--correctability_train_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_4b_mapping_triage_sigma3_seed3407")
    parser.add_argument("--basis_num_images", type=int, default=0)
    parser.add_argument("--train_count", type=int, default=128)
    parser.add_argument("--eval_count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca_device", default="cpu")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--low_size", type=int, default=32)
    parser.add_argument("--k_values", default="8,16,32")
    parser.add_argument("--projection_ridge", type=float, default=1e-5)
    parser.add_argument("--cv_ridge", type=float, default=1e-3)
    parser.add_argument("--ridge_values", default="0.0001,0.001,0.01,0.1,1.0,10.0")
    parser.add_argument("--pls_components", default="2,4,8,12,16")
    parser.add_argument("--knn_values", default="1,3,5,9")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--mlp_hidden_dim", type=int, default=32)
    parser.add_argument("--mlp_steps", type=int, default=800)
    parser.add_argument("--mlp_learning_rate", type=float, default=5e-4)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-2)
    parser.add_argument("--mlp_beta", type=float, default=0.1)
    parser.add_argument("--mlp_patience", type=int, default=80)
    parser.add_argument("--mlp_grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--pca_oversample", type=int, default=8)
    parser.add_argument("--pca_niter", type=int, default=4)
    parser.add_argument("--progress_freq", type=int, default=100)
    args = parser.parse_args()

    if args.train_count <= 0 or args.eval_count <= args.train_count:
        raise ValueError("--eval_count must be greater than --train_count for mapping triage.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.pca_device == "cuda" and not torch.cuda.is_available():
        args.pca_device = "cpu"
    args.ridge_values = parse_float_list(args.ridge_values)
    args.pls_components = parse_int_list(args.pls_components)
    args.knn_values = parse_int_list(args.knn_values)
    k_values = parse_int_list(args.k_values)
    label = sigma_label(args.sigma)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    train_indices = [
        row["index"]
        for row in meta_rows
        if row["index"] < args.train_count and row["low_weight_sum"] > 1e-8
    ]
    mini_indices = [
        row["index"]
        for row in meta_rows
        if args.train_count <= row["index"] < args.eval_count and row["low_weight_sum"] > 1e-8
    ]
    if not train_indices or not mini_indices:
        raise RuntimeError("Need open train and mini-val samples for mapping triage.")
    eval_indices = list(range(args.eval_count))

    flat_targets = low_targets[args.low_size].flatten(1)
    flat_weights = low_weights[args.low_size].repeat(1, 3, 1, 1).flatten(1).clamp_min(0.0)
    pca = pca_bases(low_targets[args.low_size], low_weights[args.low_size], k_values, args)
    projections = {}
    for k in k_values:
        if k > pca["bases"].shape[0]:
            continue
        bases = pca["bases"][:k]
        coeffs, recons = weighted_project(
            flat_targets,
            flat_weights,
            pca["mean"],
            bases,
            args.projection_ridge,
        )
        projections[(args.low_size, k)] = {
            "low_size": args.low_size,
            "K": k,
            "coeffs": coeffs,
            "recons": recons,
            "bases": bases,
            "mean": pca["mean"],
            "explained_weighted_energy": pca["explained"].get(k),
        }

    predictions, mlp_history = make_predictions(features, projections, train_indices, eval_indices, args)
    eval_loader = build_loader(args.data_dir, args.eval_count, args.num_workers, shuffle=False)
    per_image = evaluate_predictions(
        apdr_model,
        eval_loader,
        device,
        args,
        tau,
        scores,
        projections,
        predictions,
        eval_indices,
    )
    mapper_rows, coeff_rows, group_rows = aggregate_mapper_rows(
        per_image,
        projections,
        predictions,
        eval_indices,
    )
    feature_shift = feature_shift_rows(features, feature_names, train_indices, mini_indices)
    component_cv = per_component_cv_rows(
        features,
        projections,
        [row["index"] for row in meta_rows if row["low_weight_sum"] > 1e-8],
        args,
    )
    open_easy_rows = open_easy_failure_rows(per_image)

    result = {
        "stage": "APDR-v0.4B-MT coefficient mapping triage",
        "tag": args.tag,
        "sigma": args.sigma,
        "correctability_tau": tau,
        "basis_num_images": len(meta_rows),
        "train_count": args.train_count,
        "eval_count": args.eval_count,
        "train_open_count": len(train_indices),
        "mini_val_open_count": len(mini_indices),
        "low_size": args.low_size,
        "k_values": [projection["K"] for projection in projections.values()],
        "mapper_count": len(predictions),
        "feature_names": feature_names,
        "args": vars(args),
    }

    summary_path = output_dir / f"mapping_triage_summary_{label}.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "mapper_family_train128_minival256.csv", mapper_rows)
    write_csv(output_dir / f"coeff_error_by_split_{label}.csv", coeff_rows)
    write_csv(output_dir / f"coeff_cv_per_component_{label}.csv", component_cv)
    write_csv(output_dir / "feature_shift_train_vs_minival.csv", feature_shift)
    write_csv(output_dir / f"open_easy_failure_table_{label}.csv", open_easy_rows)
    write_csv_union(output_dir / f"mapping_triage_per_image_{label}.csv", per_image)
    write_csv(output_dir / f"mapping_triage_groups_{label}.csv", group_rows)
    write_csv(output_dir / f"mapping_triage_mlp_history_{label}.csv", mlp_history)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
