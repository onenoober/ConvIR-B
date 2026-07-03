#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TVF


TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)


SEVERE = -0.20
STRONG_REG = -0.05


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def cvar(values: list[float], pct: float = 5.0) -> float:
    if not values:
        return float("nan")
    k = max(1, int(round(len(values) * pct / 100.0)))
    return mean(sorted(values)[:k])


def first_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"none of {names} under {root}")


def train_dirs(data_dir: Path) -> tuple[Path, Path]:
    train = data_dir / "train"
    return first_dir(train, ("IN", "haze", "hazy")), first_dir(train, ("GT", "gt"))


def label_path(gt_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    ext = Path(image_name).suffix
    candidates = [image_name]
    if "_" in stem:
        base = stem.split("_")[0]
        candidates.extend([f"{base}{ext}", f"{base}.png"])
    for candidate in candidates:
        p = gt_dir / candidate
        if p.is_file():
            return p
    raise FileNotFoundError(f"no GT for {image_name} under {gt_dir}")


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    return TVF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def pad_to(x: torch.Tensor, factor: int = 32) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(x, (0, pw, 0, ph), "reflect"), h, w


def load_state(path: Path, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def tensor_psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    mse = F.mse_loss(torch.clamp(pred, 0, 1), label).clamp_min(1e-12)
    return float((10 * torch.log10(1 / mse)).detach().cpu())


def haar_dwt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) / 2.0
    lh = (a - b + c - d) / 2.0
    hl = (a + b - c - d) / 2.0
    hh = (a - b - c + d) / 2.0
    return ll, lh, hl, hh, h, w


def haar_iwt(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor, h: int, w: int) -> torch.Tensor:
    a = (ll + lh + hl + hh) / 2.0
    b = (ll - lh + hl - hh) / 2.0
    c = (ll + lh - hl - hh) / 2.0
    d = (ll - lh - hl + hh) / 2.0
    out = torch.empty(
        (ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2),
        dtype=ll.dtype,
        device=ll.device,
    )
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out[:, :, :h, :w]


def forward_parts(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    x_2 = F.interpolate(x, scale_factor=0.5)
    x_4 = F.interpolate(x_2, scale_factor=0.5)
    z2 = model.SCM2(x_2)
    z4 = model.SCM1(x_4)
    x_ = model.feat_extract[0](x)
    res1 = model.Encoder[0](x_)
    z = model.feat_extract[1](res1)
    z = model.FAM2(z, z2)
    res2 = model.Encoder[1](z)
    z = model.feat_extract[2](res2)
    z = model.FAM1(z, z4)
    z = model.Encoder[2](z)
    z = model.Decoder[0](z)
    z = model.feat_extract[3](z)
    z = torch.cat([z, res2], dim=1)
    z = model.Convs[0](z)
    mid = model.Decoder[1](z)
    z = model.feat_extract[4](mid)
    z = torch.cat([z, res1], dim=1)
    z = model.Convs[1](z)
    final = model.Decoder[2](z)
    return {"mid": mid, "final": final}


def head_from_final(model: torch.nn.Module, final: torch.Tensor, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return (model.feat_extract[5](final) + x)[:, :, :h, :w]


def channel_scale(ll: torch.Tensor, delta_scale: float) -> torch.Tensor:
    flat = ll.detach().flatten(2)
    scale = flat.std(dim=2).view(ll.shape[0], ll.shape[1], 1, 1)
    return (scale.clamp_min(1e-4) * delta_scale).detach()


def optimize_o1_delta(
    *,
    model: torch.nn.Module,
    final: torch.Tensor,
    x: torch.Tensor,
    gt: torch.Tensor,
    h: int,
    w: int,
    steps: int,
    lr: float,
    delta_scale: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base_ll, lh, hl, hh, fh, fw = haar_dwt(final.detach())
    scale = channel_scale(base_ll, delta_scale)
    raw = torch.zeros((1, base_ll.shape[1], 1, 1), device=base_ll.device, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    best_psnr = -1.0
    best_loss = float("inf")
    best_delta: torch.Tensor | None = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        delta = torch.tanh(raw) * scale
        recon = haar_iwt(base_ll + delta, lh, hl, hh, fh, fw)
        pred = head_from_final(model, recon, x, h, w)
        loss = F.l1_loss(torch.clamp(pred, 0, 1), gt) + 1e-4 * delta.abs().mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            psnr = tensor_psnr(pred, gt)
            if psnr > best_psnr:
                best_psnr = psnr
                best_loss = float(loss.detach().cpu())
                best_delta = (torch.tanh(raw.detach()) * scale).detach()
    assert best_delta is not None
    return best_delta.flatten().detach().cpu(), {
        "oracle_best_psnr": best_psnr,
        "oracle_best_loss": best_loss,
        "oracle_delta_abs_mean": float(best_delta.abs().mean().detach().cpu()),
        "oracle_delta_rms": float(torch.sqrt(torch.mean(best_delta.detach() ** 2)).cpu()),
        "oracle_delta_abs_max": float(best_delta.abs().max().detach().cpu()),
    }


def pooled_ll_context(final: torch.Tensor) -> torch.Tensor:
    ll, _, _, _, _, _ = haar_dwt(final.detach())
    mean_feat = ll.mean(dim=(2, 3)).flatten()
    std_feat = ll.flatten(2).std(dim=2).flatten()
    return torch.cat([mean_feat, std_feat], dim=0).detach().cpu()


def apply_delta_and_psnr(
    model: torch.nn.Module,
    final: torch.Tensor,
    x: torch.Tensor,
    gt: torch.Tensor,
    h: int,
    w: int,
    delta_vec: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    base_ll, lh, hl, hh, fh, fw = haar_dwt(final.detach())
    delta = delta_vec.to(base_ll.device, dtype=base_ll.dtype).view(1, base_ll.shape[1], 1, 1)
    recon = haar_iwt(base_ll + delta, lh, hl, hh, fh, fw)
    pred = head_from_final(model, recon, x, h, w)
    return tensor_psnr(pred, gt), delta.detach().flatten().cpu()


def standardize(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = train.mean(dim=0, keepdim=True)
    sigma = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train - mu) / sigma, (test - mu) / sigma, mu, sigma


def fit_ridge(x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    ones_train = torch.ones((x_train.shape[0], 1), dtype=x_train.dtype)
    ones_test = torch.ones((x_test.shape[0], 1), dtype=x_test.dtype)
    xb = torch.cat([x_train, ones_train], dim=1)
    xt = torch.cat([x_test, ones_test], dim=1)
    eye = torch.eye(xb.shape[1], dtype=x_train.dtype)
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(xb.T @ xb + ridge_lambda * eye, xb.T @ y_train)
    return xt @ weights


class SmallMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_mlp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    hidden: int,
    epochs: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    model = SmallMLP(x_train.shape[1], hidden, y_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return model(x_test).detach()


def summarize_rows(rows: list[dict[str, Any]], variant: str, scope: str = "all") -> dict[str, Any]:
    vals = [float(r["dPSNR"]) for r in rows if r["variant"] == variant]
    a0 = [float(r["A0_PSNR"]) for r in rows if r["variant"] == variant]
    hard_cut = percentile(a0, 25)
    strong_cut = percentile(a0, 75)
    hard = [float(r["dPSNR"]) for r in rows if r["variant"] == variant and float(r["A0_PSNR"]) <= hard_cut]
    easy = [float(r["dPSNR"]) for r in rows if r["variant"] == variant and float(r["A0_PSNR"]) >= strong_cut]
    strong = [float(r["dPSNR"]) for r in rows if r["variant"] == variant and float(r["A0_PSNR"]) >= strong_cut]
    return {
        "variant": variant,
        "scope": scope,
        "count": len(vals),
        "mean_dPSNR": mean(vals),
        "median_dPSNR": median(vals),
        "p01_dPSNR": percentile(vals, 1),
        "p05_dPSNR": percentile(vals, 5),
        "CVaR5_dPSNR": cvar(vals, 5),
        "hard_bottom25_dPSNR": mean(hard),
        "easy_top25_dPSNR": mean(easy),
        "positive_ratio": sum(v > 0 for v in vals) / len(vals) if vals else float("nan"),
        "severe_count": sum(v <= SEVERE for v in vals),
        "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
        "strong_reference_count": len(strong),
        "strong_reference_regressions": sum(v <= STRONG_REG for v in strong),
    }


def p1_gate(summary: dict[str, Any], shuffled_summary: dict[str, Any]) -> dict[str, Any]:
    strong_limit = max(3, int(round(0.05 * float(summary["strong_reference_count"]))))
    control_gap = float(summary["mean_dPSNR"]) - float(shuffled_summary["mean_dPSNR"])
    checks = {
        "mean_dPSNR_ge_0p15": float(summary["mean_dPSNR"]) >= 0.15,
        "hard_bottom25_ge_0p25": float(summary["hard_bottom25_dPSNR"]) >= 0.25,
        "easy_top25_ge_0": float(summary["easy_top25_dPSNR"]) >= 0.0,
        "p05_ge_neg0p15": float(summary["p05_dPSNR"]) >= -0.15,
        "severe_rate_le_0p025": float(summary["severe_rate"]) <= 0.025,
        "strong_regressions_le_5pct": int(summary["strong_reference_regressions"]) <= strong_limit,
        "real_beats_shuffled_by_0p10": control_gap >= 0.10,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "strong_regression_limit": strong_limit,
        "control_gap": control_gap,
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(torch.linalg.norm(a) * torch.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(a, b) / denom)


def run_p1(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    write_text(
        args.out_dir / "v218_p1_o1_action_regression_protocol.md",
        "\n".join(
            [
                "# v2.18 P1 O1 Action Regression Protocol",
                "",
                "This audit regenerates O1 global final-feature LL oracle targets on train-derived Haze4K images.",
                "It then fits 5-fold deployable pooled-LL context predictors and replays predicted deltas through official ConvIR-B.",
                "No deployable model is trained here and locked Haze4K is untouched.",
                "",
                "Primary predictor: small MLP on final-feature LL channel mean and std context.",
                "Diagnostics: ridge predictor and shuffled-target ridge control.",
                f"O1 optimization steps: `{args.steps_o1}`; delta scale: `{args.delta_scale}`.",
                f"MLP epochs: `{args.mlp_epochs}`; hidden: `{args.mlp_hidden}`.",
            ]
        ),
    )
    split_rows = read_csv(args.split_csv)
    names = [row["name"] for row in split_rows]
    if args.max_images:
        names = names[: args.max_images]
    split_by_name = {row["name"]: row for row in split_rows}
    input_dir, gt_dir = train_dirs(args.data_dir)

    feature_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    meta_rows: list[dict[str, Any]] = []

    for idx, name in enumerate(names, 1):
        hazy = image_tensor(input_dir / name, device)
        gt_full = image_tensor(label_path(gt_dir, name), device)
        x, h, w = pad_to(hazy, 32)
        gt = gt_full[:, :, :h, :w]
        with torch.no_grad():
            parts = forward_parts(model, x)
            a0_pred = head_from_final(model, parts["final"], x, h, w)
            a0_psnr = tensor_psnr(a0_pred, gt)
            features = pooled_ll_context(parts["final"])
        delta, stats = optimize_o1_delta(
            model=model,
            final=parts["final"],
            x=x,
            gt=gt,
            h=h,
            w=w,
            steps=args.steps_o1,
            lr=args.lr,
            delta_scale=args.delta_scale,
        )
        feature_rows.append(features.float())
        target_rows.append(delta.float())
        fold = int(split_by_name[name].get("oof_fold", 0))
        meta_rows.append(
            {
                "name": name,
                "oof_fold": fold,
                "A0_PSNR": a0_psnr,
                "oracle_candidate_PSNR": stats["oracle_best_psnr"],
                "oracle_dPSNR": stats["oracle_best_psnr"] - a0_psnr,
                "oracle_best_loss": stats["oracle_best_loss"],
                "oracle_delta_abs_mean": stats["oracle_delta_abs_mean"],
                "oracle_delta_rms": stats["oracle_delta_rms"],
                "oracle_delta_abs_max": stats["oracle_delta_abs_max"],
                "context_abs_mean": float(features.abs().mean()),
                "context_rms": float(torch.sqrt(torch.mean(features**2))),
            }
        )
        if idx % args.print_freq == 0:
            print(f"V218_P1_TARGET {idx}/{len(names)}", flush=True)

    write_csv(args.out_dir / "v218_p1_o1_action_target_summary.csv", meta_rows)

    x_all = torch.stack(feature_rows).float()
    y_all = torch.stack(target_rows).float()
    folds = sorted({int(row["oof_fold"]) for row in meta_rows})
    pred_mlp = torch.zeros_like(y_all)
    pred_ridge = torch.zeros_like(y_all)
    pred_shuffle = torch.zeros_like(y_all)
    fold_report: list[dict[str, Any]] = []
    rng = random.Random(args.seed)

    for fold in folds:
        train_idx = [i for i, row in enumerate(meta_rows) if int(row["oof_fold"]) != fold]
        test_idx = [i for i, row in enumerate(meta_rows) if int(row["oof_fold"]) == fold]
        x_train_raw = x_all[train_idx]
        x_test_raw = x_all[test_idx]
        y_train_raw = y_all[train_idx]
        y_test_raw = y_all[test_idx]
        x_train, x_test, _, _ = standardize(x_train_raw, x_test_raw)
        y_train, _, y_mu, y_sigma = standardize(y_train_raw, y_test_raw)

        mlp_std = fit_mlp(
            x_train,
            y_train,
            x_test,
            hidden=args.mlp_hidden,
            epochs=args.mlp_epochs,
            seed=args.seed + fold,
        )
        ridge_std = fit_ridge(x_train, y_train, x_test, args.ridge_lambda)
        shuffled_order = list(range(y_train.shape[0]))
        rng.shuffle(shuffled_order)
        shuffle_std = fit_ridge(x_train, y_train[shuffled_order], x_test, args.ridge_lambda)

        pred_mlp[test_idx] = mlp_std * y_sigma + y_mu
        pred_ridge[test_idx] = ridge_std * y_sigma + y_mu
        pred_shuffle[test_idx] = shuffle_std * y_sigma + y_mu

        fold_report.append(
            {
                "fold": fold,
                "train_count": len(train_idx),
                "test_count": len(test_idx),
                "target_abs_mean_test": float(y_test_raw.abs().mean()),
                "mlp_target_mse": float(F.mse_loss(pred_mlp[test_idx], y_test_raw)),
                "ridge_target_mse": float(F.mse_loss(pred_ridge[test_idx], y_test_raw)),
                "shuffled_target_mse": float(F.mse_loss(pred_shuffle[test_idx], y_test_raw)),
            }
        )

    replay_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    pred_by_variant = {
        "P1_mlp_pooled_ll": pred_mlp,
        "P1_ridge_pooled_ll": pred_ridge,
        "P1_shuffled_target_control": pred_shuffle,
    }
    for idx, row in enumerate(meta_rows):
        name = str(row["name"])
        hazy = image_tensor(input_dir / name, device)
        gt_full = image_tensor(label_path(gt_dir, name), device)
        x, h, w = pad_to(hazy, 32)
        gt = gt_full[:, :, :h, :w]
        with torch.no_grad():
            parts = forward_parts(model, x)
        target = y_all[idx]
        for variant, pred_all in pred_by_variant.items():
            psnr, applied = apply_delta_and_psnr(model, parts["final"], x, gt, h, w, pred_all[idx])
            dpsnr = psnr - float(row["A0_PSNR"])
            replay_rows.append(
                {
                    "name": name,
                    "oof_fold": row["oof_fold"],
                    "variant": variant,
                    "A0_PSNR": row["A0_PSNR"],
                    "candidate_PSNR": psnr,
                    "dPSNR": dpsnr,
                    "oracle_dPSNR": row["oracle_dPSNR"],
                    "pred_delta_abs_mean": float(applied.abs().mean()),
                    "pred_delta_rms": float(torch.sqrt(torch.mean(applied**2))),
                    "target_delta_abs_mean": row["oracle_delta_abs_mean"],
                    "target_delta_rms": row["oracle_delta_rms"],
                    "cosine_to_oracle_delta": cosine(applied, target),
                    "dot_to_oracle_delta": float(torch.dot(applied, target)),
                    "wrong_direction": int(float(torch.dot(applied, target)) <= 0.0),
                }
            )
        if (idx + 1) % args.print_freq == 0:
            print(f"V218_P1_REPLAY {idx + 1}/{len(meta_rows)}", flush=True)

    variants = sorted(pred_by_variant)
    summary_rows = [summarize_rows(replay_rows, variant) for variant in variants]
    for variant in variants:
        rows = [r for r in replay_rows if r["variant"] == variant]
        direction_rows.append(
            {
                "variant": variant,
                "count": len(rows),
                "mean_cosine_to_oracle_delta": mean([float(r["cosine_to_oracle_delta"]) for r in rows]),
                "median_cosine_to_oracle_delta": median([float(r["cosine_to_oracle_delta"]) for r in rows]),
                "wrong_direction_rate": mean([float(r["wrong_direction"]) for r in rows]),
                "mean_pred_delta_abs_mean": mean([float(r["pred_delta_abs_mean"]) for r in rows]),
                "mean_target_delta_abs_mean": mean([float(r["target_delta_abs_mean"]) for r in rows]),
            }
        )
        for fold in folds:
            subset = [r for r in replay_rows if r["variant"] == variant and int(r["oof_fold"]) == fold]
            fold_report.append(summarize_rows(subset, variant, scope=f"replay_fold{fold}"))

    write_csv(args.out_dir / "v218_p1_o1_action_regression_fold_report.csv", fold_report)
    write_csv(args.out_dir / "v218_p1_o1_action_replay_metrics.csv", replay_rows)
    write_csv(args.out_dir / "v218_p1_o1_action_direction_stats.csv", direction_rows)
    write_csv(
        args.out_dir / "v218_p1_o1_shuffled_target_control.csv",
        [row for row in summary_rows if row["variant"] in {"P1_mlp_pooled_ll", "P1_shuffled_target_control"}],
    )

    primary_summary = next(row for row in summary_rows if row["variant"] == "P1_mlp_pooled_ll")
    shuffled_summary = next(row for row in summary_rows if row["variant"] == "P1_shuffled_target_control")
    gate = p1_gate(primary_summary, shuffled_summary)
    for row in summary_rows:
        if row["variant"] == "P1_mlp_pooled_ll":
            row.update({f"gate_{k}": v for k, v in gate["checks"].items()})
            row["gate_pass"] = gate["pass"]
            row["control_gap_vs_shuffled"] = gate["control_gap"]
    # Re-write replay summary as a compact table embedded in closeout and fold report sidecar.
    write_csv(args.out_dir / "v218_p1_o1_action_replay_summary.csv", summary_rows)

    primary_rows = [r for r in replay_rows if r["variant"] == "P1_mlp_pooled_ll"]
    top_tail = sorted(primary_rows, key=lambda r: float(r["dPSNR"]))[:160]
    write_csv(args.out_dir / "v218_p1_top_tail_manifest.csv", top_tail)

    if gate["pass"]:
        decision = "P1_PASS_O1_GLOBAL_ACTION_LEARNABLE_BY_POOLED_LL_POLICY"
    else:
        decision = "P1_FAIL_O1_GLOBAL_ACTION_NOT_SAFELY_LEARNED_BY_POOLED_LL_POLICY"
    write_text(
        args.out_dir / "v218_p1_decision.md",
        "\n".join(
            [
                "# v2.18 P1 O1 Action Learnability Decision",
                "",
                f"Decision: `{decision}`",
                "",
                "Primary replay summary:",
                "",
                f"- mean dPSNR: `{primary_summary['mean_dPSNR']}`",
                f"- hard bottom25 dPSNR: `{primary_summary['hard_bottom25_dPSNR']}`",
                f"- easy top25 dPSNR: `{primary_summary['easy_top25_dPSNR']}`",
                f"- p05 dPSNR: `{primary_summary['p05_dPSNR']}`",
                f"- CVaR5 dPSNR: `{primary_summary['CVaR5_dPSNR']}`",
                f"- severe rate: `{primary_summary['severe_rate']}`",
                f"- strong-reference regressions: `{primary_summary['strong_reference_regressions']}` / `{primary_summary['strong_reference_count']}`",
                f"- control gap vs shuffled: `{gate['control_gap']}`",
                "",
                "Interpretation:",
                "",
                "- P1 is about deployable policy learnability, not oracle headroom.",
                "- If P1 fails while v2.17 O2/O3 remains strong, WLDB-A2 global policy should not train yet; design spatial WLDB-B learnability next.",
                "- Locked Haze4K remains untouched.",
            ]
        ),
    )
    closeout = {
        "decision": decision,
        "gate": gate,
        "summary": summary_rows,
        "direction": direction_rows,
        "locked_test_touched": False,
        "training_launched": False,
    }
    write_json(args.out_dir / "v218_p1_closeout.json", closeout)
    print("V218_P1_OK", decision, flush=True)
    return {
        "decision": decision,
        "gate_pass": bool(gate["pass"]),
        "summary": summary_rows,
        "replay_rows": replay_rows,
        "target_rows": meta_rows,
    }


def run_p2_p3(args: argparse.Namespace, p1: dict[str, Any]) -> dict[str, Any]:
    write_text(
        args.out_dir / "v218_p2_objective_replay_protocol.md",
        "\n".join(
            [
                "# v2.18 P2 Tail-Aware Objective Replay Protocol",
                "",
                "No training is run here. This replay asks whether the proposed tail and preserve objective terms would activate on the known v2.16 WLDB-A failure mode.",
                "",
                "Tail hinge: activates when candidate dPSNR vs A0 is below `-0.15`.",
                "Severe hinge: audits dPSNR below `-0.20`.",
                "Preserve hinge: activates on strong/easy samples below `-0.05`.",
                "Action budget sweep: thresholds are derived from P1 safe oracle/predicted O1 action norms and applied to v2.16 WLDB-A action norms.",
                "Locked Haze4K remains untouched.",
            ]
        ),
    )
    tail_manifest = read_csv(args.v217_dir / "v217_r1_tail_case_manifest.csv")
    action_rows = read_csv(args.v217_dir / "v217_r1_action_norm_stats.csv")
    loss_rows = read_csv(args.v217_dir / "v217_r3_per_image_loss_terms.csv")
    manifest_by_name = {row["name"]: row for row in tail_manifest}
    action_by_key = {(row["checkpoint"], row["name"]): row for row in action_rows}

    replay_loss_rows: list[dict[str, Any]] = []
    for row in loss_rows:
        checkpoint = row["checkpoint"]
        name = row["name"]
        dpsnr = float(row["dPSNR"])
        m = manifest_by_name.get(name, {})
        strong = int(m.get("is_strong_reference", "0") or 0)
        easy = int(m.get("is_easy_top25", "0") or 0)
        hard = int(m.get("is_hard_bottom25", "0") or 0)
        tail_hinge = max(0.0, -0.15 - dpsnr)
        severe_hinge = max(0.0, SEVERE - dpsnr)
        preserve_mask = int((strong or easy) and dpsnr <= STRONG_REG)
        preserve_hinge = max(0.0, STRONG_REG - dpsnr) if (strong or easy) else 0.0
        replay_loss_rows.append(
            {
                **row,
                "is_hard_bottom25": hard,
                "is_easy_top25": easy,
                "is_strong_reference": strong,
                "tail_hinge_margin_neg0p15": tail_hinge,
                "severe_hinge_margin_neg0p20": severe_hinge,
                "preserve_mask_strong_or_easy": int(strong or easy),
                "preserve_hinge_margin_neg0p05": preserve_hinge,
                "tail_hinge_active": int(tail_hinge > 0),
                "preserve_hinge_active": int(preserve_hinge > 0),
                "positive_sample": int(dpsnr > 0),
            }
        )
    write_csv(args.out_dir / "v218_p2_per_image_loss_terms.csv", replay_loss_rows)

    checkpoints = sorted({row["checkpoint"] for row in replay_loss_rows})
    tail_report: list[dict[str, Any]] = []
    preserve_report: list[dict[str, Any]] = []
    for ckpt in checkpoints:
        rows = [r for r in replay_loss_rows if r["checkpoint"] == ckpt]
        severe_rows = [r for r in rows if float(r["dPSNR"]) <= SEVERE]
        positive_rows = [r for r in rows if float(r["dPSNR"]) > 0]
        tail_report.append(
            {
                "checkpoint": ckpt,
                "count": len(rows),
                "severe_count": len(severe_rows),
                "tail_hinge_active_count": sum(int(r["tail_hinge_active"]) for r in rows),
                "tail_hinge_coverage_on_severe": (
                    sum(int(r["tail_hinge_active"]) for r in severe_rows) / len(severe_rows) if severe_rows else 1.0
                ),
                "positive_tail_hinge_activation_rate": (
                    sum(int(r["tail_hinge_active"]) for r in positive_rows) / len(positive_rows) if positive_rows else 0.0
                ),
                "mean_tail_hinge": mean([float(r["tail_hinge_margin_neg0p15"]) for r in rows]),
                "mean_severe_hinge": mean([float(r["severe_hinge_margin_neg0p20"]) for r in rows]),
            }
        )
        preserve_fail = [r for r in rows if int(r["preserve_mask_strong_or_easy"]) and float(r["dPSNR"]) <= STRONG_REG]
        preserve_positive = [r for r in rows if int(r["preserve_mask_strong_or_easy"]) and float(r["dPSNR"]) > 0]
        preserve_report.append(
            {
                "checkpoint": ckpt,
                "strong_or_easy_count": sum(int(r["preserve_mask_strong_or_easy"]) for r in rows),
                "strong_or_easy_regression_count": len(preserve_fail),
                "preserve_hinge_active_count": sum(int(r["preserve_hinge_active"]) for r in rows),
                "preserve_hinge_coverage_on_regressions": (
                    sum(int(r["preserve_hinge_active"]) for r in preserve_fail) / len(preserve_fail) if preserve_fail else 1.0
                ),
                "positive_preserve_hinge_activation_rate": (
                    sum(int(r["preserve_hinge_active"]) for r in preserve_positive) / len(preserve_positive)
                    if preserve_positive
                    else 0.0
                ),
                "mean_preserve_hinge": mean([float(r["preserve_hinge_margin_neg0p05"]) for r in rows]),
            }
        )
    write_csv(args.out_dir / "v218_p2_tail_hinge_activation_report.csv", tail_report)
    write_csv(args.out_dir / "v218_p2_preserve_mask_activation_report.csv", preserve_report)

    model5_tail = next(row for row in tail_report if row["checkpoint"] == "model_5")
    model5_preserve = next(row for row in preserve_report if row["checkpoint"] == "model_5")
    p2_pass = (
        float(model5_tail["tail_hinge_coverage_on_severe"]) >= 0.95
        and float(model5_preserve["preserve_hinge_coverage_on_regressions"]) >= 0.95
        and float(model5_tail["positive_tail_hinge_activation_rate"]) <= 0.10
        and float(model5_preserve["positive_preserve_hinge_activation_rate"]) <= 0.10
    )
    p2_decision = "P2_PASS_TAIL_PRESERVE_REPLAY_COVERS_WLDB_A_FAILURE" if p2_pass else "P2_FAIL_TAIL_PRESERVE_REPLAY_INSUFFICIENT"
    write_text(
        args.out_dir / "v218_p2_decision.md",
        "\n".join(
            [
                "# v2.18 P2 Tail-Aware Objective Replay Decision",
                "",
                f"Decision: `{p2_decision}`",
                "",
                f"- model_5 severe coverage by tail hinge: `{model5_tail['tail_hinge_coverage_on_severe']}`",
                f"- model_5 positive tail-hinge activation rate: `{model5_tail['positive_tail_hinge_activation_rate']}`",
                f"- model_5 strong/easy regression coverage by preserve hinge: `{model5_preserve['preserve_hinge_coverage_on_regressions']}`",
                f"- model_5 positive preserve-hinge activation rate: `{model5_preserve['positive_preserve_hinge_activation_rate']}`",
                "",
                "Interpretation:",
                "",
                "- This is an objective replay, not training.",
                "- A pass means the proposed terms would notice the known v2.16 failure mode; it does not prove a model can optimize them.",
                "- Locked Haze4K remains untouched.",
            ]
        ),
    )

    p1_target = p1["target_rows"]
    p1_replay = [r for r in p1["replay_rows"] if r["variant"] == "P1_mlp_pooled_ll"]
    safe_norms = [float(r["oracle_delta_abs_mean"]) for r in p1_target]
    pred_norms = [float(r["pred_delta_abs_mean"]) for r in p1_replay]
    threshold_specs = []
    for label, values in (("oracle", safe_norms), ("predicted_mlp", pred_norms)):
        for pct in (25, 50, 75, 90):
            threshold_specs.append((f"{label}_p{pct}", percentile(values, pct)))

    budget_rows: list[dict[str, Any]] = []
    norm_tail_rows: list[dict[str, Any]] = []
    for row in action_rows:
        key = (row["checkpoint"], row["name"])
        m = manifest_by_name.get(row["name"], {})
        dpsnr_key = f"{row['checkpoint']}_dPSNR"
        dpsnr = float(m.get(dpsnr_key, "nan"))
        norm_tail_rows.append(
            {
                "checkpoint": row["checkpoint"],
                "name": row["name"],
                "ll_delta_abs_mean": row["ll_delta_abs_mean"],
                "image_action_l1": row["image_action_l1"],
                "dPSNR": dpsnr,
                "is_model5_severe": int(row["checkpoint"] == "model_5" and dpsnr <= SEVERE),
                "is_model5_positive": int(row["checkpoint"] == "model_5" and dpsnr > 0),
            }
        )
        _ = key
    write_csv(args.out_dir / "v218_p3_action_norm_vs_tail_damage.csv", norm_tail_rows)

    model5_actions = [r for r in norm_tail_rows if r["checkpoint"] == "model_5"]
    model5_severe = [r for r in model5_actions if int(r["is_model5_severe"])]
    for label, threshold in threshold_specs:
        activation_all = [float(r["ll_delta_abs_mean"]) > threshold for r in model5_actions]
        activation_severe = [float(r["ll_delta_abs_mean"]) > threshold for r in model5_severe]
        safe_over = [v > threshold for v in safe_norms]
        pred_over = [v > threshold for v in pred_norms]
        budget_rows.append(
            {
                "threshold_label": label,
                "threshold_value": threshold,
                "identity_penalized": False,
                "model5_activation_rate": sum(activation_all) / len(activation_all) if activation_all else float("nan"),
                "model5_severe_coverage": sum(activation_severe) / len(activation_severe) if activation_severe else 1.0,
                "safe_oracle_overactivation_rate": sum(safe_over) / len(safe_over) if safe_over else float("nan"),
                "predicted_mlp_overactivation_rate": sum(pred_over) / len(pred_over) if pred_over else float("nan"),
                "passes_p3_threshold_gate": (
                    threshold >= 0.0
                    and (sum(activation_all) / len(activation_all) if activation_all else 0.0) > 0.0
                    and (sum(activation_severe) / len(activation_severe) if activation_severe else 1.0) >= 0.50
                    and (sum(safe_over) / len(safe_over) if safe_over else 1.0) <= 0.50
                ),
            }
        )
    write_csv(args.out_dir / "v218_p2_action_budget_sweep.csv", budget_rows)
    write_csv(args.out_dir / "v218_p3_action_budget_calibration.csv", budget_rows)
    p3_pass_rows = [r for r in budget_rows if r["passes_p3_threshold_gate"]]
    p3_decision = "P3_PASS_NONZERO_ACTION_BUDGET_CALIBRATION_FOUND" if p3_pass_rows else "P3_FAIL_NORM_ONLY_ACTION_BUDGET_NOT_CALIBRATED"
    write_text(
        args.out_dir / "v218_p3_budget_threshold_decision.md",
        "\n".join(
            [
                "# v2.18 P3 Action Budget Threshold Decision",
                "",
                f"Decision: `{p3_decision}`",
                "",
                f"- passing threshold count: `{len(p3_pass_rows)}`",
                "- identity action is unpenalized for all nonnegative thresholds.",
                "- threshold candidates are derived only from train-derived safe oracle/predicted action norms.",
                "- locked Haze4K remains untouched.",
            ]
        ),
    )

    write_json(
        args.out_dir / "v218_p2_closeout.json",
        {
            "decision": p2_decision,
            "p2_pass": p2_pass,
            "tail_report": tail_report,
            "preserve_report": preserve_report,
            "locked_test_touched": False,
            "training_launched": False,
        },
    )
    write_json(
        args.out_dir / "v218_p3_closeout.json",
        {
            "decision": p3_decision,
            "p3_pass": bool(p3_pass_rows),
            "passing_thresholds": p3_pass_rows,
            "locked_test_touched": False,
            "training_launched": False,
        },
    )
    print("V218_P2_OK", p2_decision, flush=True)
    print("V218_P3_OK", p3_decision, flush=True)
    return {"p2_decision": p2_decision, "p2_pass": p2_pass, "p3_decision": p3_decision, "p3_pass": bool(p3_pass_rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--v216-dir", type=Path, required=True)
    ap.add_argument("--v217-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--steps-o1", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--delta-scale", type=float, default=0.50)
    ap.add_argument("--mlp-epochs", type=int, default=320)
    ap.add_argument("--mlp-hidden", type=int, default=64)
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=218)
    ap.add_argument("--print-freq", type=int, default=25)
    args = ap.parse_args()

    from models.ConvIR import build_net

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_net("base", "Haze4K", "original").to(device)
    model.load_state_dict(load_state(args.checkpoint, device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    p1 = run_p1(args, model, device)
    p23 = run_p2_p3(args, p1)
    final_decision = "V218_PREFLIGHT_AUDITS_PASS_READY_FOR_SEPARATE_N3_PLAN"
    if not p1["gate_pass"]:
        final_decision = "V218_PAUSE_P1_GLOBAL_POLICY_LEARNABILITY_FAIL"
    elif not p23["p2_pass"]:
        final_decision = "V218_PAUSE_P2_OBJECTIVE_REPLAY_FAIL"
    elif not p23["p3_pass"]:
        final_decision = "V218_PAUSE_P3_ACTION_BUDGET_CALIBRATION_FAIL"
    write_json(
        args.out_dir / "v218_p1_p2_p3_closeout.json",
        {
            "decision": final_decision,
            "p1_decision": p1["decision"],
            "p2_decision": p23["p2_decision"],
            "p3_decision": p23["p3_decision"],
            "locked_test_touched": False,
            "training_launched": False,
        },
    )
    print("V218_P1_P2_P3_CLOSEOUT", final_decision, flush=True)


if __name__ == "__main__":
    main()
