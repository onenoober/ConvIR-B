#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
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


def final_output(out: Any, h: int, w: int) -> torch.Tensor:
    pred = out[2] if isinstance(out, (list, tuple)) else out
    return pred[:, :, :h, :w]


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


def haar_iwt(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
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
    return {"x": x, "res1": res1, "mid": mid, "final": final}


def head_from_final(model: torch.nn.Module, final: torch.Tensor, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return (model.feat_extract[5](final) + x)[:, :, :h, :w]


def final_from_mid(model: torch.nn.Module, mid: torch.Tensor, res1: torch.Tensor) -> torch.Tensor:
    z = model.feat_extract[4](mid)
    z = torch.cat([z, res1], dim=1)
    z = model.Convs[1](z)
    return model.Decoder[2](z)


def channel_scale(ll: torch.Tensor, delta_scale: float) -> torch.Tensor:
    flat = ll.detach().flatten(2)
    scale = flat.std(dim=2).view(ll.shape[0], ll.shape[1], 1, 1)
    return (scale.clamp_min(1e-4) * delta_scale).detach()


def spatial_delta(raw: torch.Tensor, target_hw: tuple[int, int], scale: torch.Tensor) -> torch.Tensor:
    delta = torch.tanh(raw) * scale
    if delta.shape[-2:] != target_hw:
        delta = F.interpolate(delta, size=target_hw, mode="bilinear", align_corners=False)
    return delta


def optimize_final_ll(
    *,
    model: torch.nn.Module,
    final: torch.Tensor,
    x: torch.Tensor,
    gt: torch.Tensor,
    h: int,
    w: int,
    mode: str,
    grid: int,
    steps: int,
    lr: float,
    delta_scale: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base_ll, lh, hl, hh, fh, fw = haar_dwt(final.detach())
    scale = channel_scale(base_ll, delta_scale)
    if mode == "global":
        raw = torch.zeros((1, base_ll.shape[1], 1, 1), device=base_ll.device, requires_grad=True)
    elif mode == "spatial":
        raw = torch.zeros((1, base_ll.shape[1], grid, grid), device=base_ll.device, requires_grad=True)
    else:
        raise ValueError(mode)
    opt = torch.optim.Adam([raw], lr=lr)
    best_psnr = -1.0
    best_pred: torch.Tensor | None = None
    best_loss = float("inf")
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        delta = spatial_delta(raw, base_ll.shape[-2:], scale)
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
                best_pred = pred.detach()
    assert best_pred is not None
    delta = spatial_delta(raw.detach(), base_ll.shape[-2:], scale)
    return best_pred, {
        "best_psnr": best_psnr,
        "best_loss": best_loss,
        "delta_abs_mean": float(delta.abs().mean().detach().cpu()),
        "delta_rms": float(torch.sqrt(torch.mean(delta.detach() ** 2)).cpu()),
    }


def optimize_mid_ll(
    *,
    model: torch.nn.Module,
    mid: torch.Tensor,
    res1: torch.Tensor,
    x: torch.Tensor,
    gt: torch.Tensor,
    h: int,
    w: int,
    include_final: bool,
    mid_grid: int,
    final_grid: int,
    steps: int,
    lr: float,
    delta_scale: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    mid_ll, mid_lh, mid_hl, mid_hh, mh, mw = haar_dwt(mid.detach())
    mid_scale = channel_scale(mid_ll, delta_scale)
    raw_mid = torch.zeros((1, mid_ll.shape[1], mid_grid, mid_grid), device=mid.device, requires_grad=True)
    params: list[torch.Tensor] = [raw_mid]

    with torch.no_grad():
        base_final = final_from_mid(model, mid.detach(), res1.detach())
        final_ll, _, _, _, _, _ = haar_dwt(base_final)
        final_scale = channel_scale(final_ll, delta_scale)
    raw_final = None
    if include_final:
        raw_final = torch.zeros((1, final_ll.shape[1], final_grid, final_grid), device=mid.device, requires_grad=True)
        params.append(raw_final)

    opt = torch.optim.Adam(params, lr=lr)
    best_psnr = -1.0
    best_pred: torch.Tensor | None = None
    best_loss = float("inf")
    last_mid_delta = None
    last_final_delta = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        mid_delta = spatial_delta(raw_mid, mid_ll.shape[-2:], mid_scale)
        mid_recon = haar_iwt(mid_ll + mid_delta, mid_lh, mid_hl, mid_hh, mh, mw)
        final = final_from_mid(model, mid_recon, res1.detach())
        final_delta = None
        if raw_final is not None:
            ll, lh, hl, hh, fh, fw = haar_dwt(final)
            final_delta = spatial_delta(raw_final, ll.shape[-2:], final_scale)
            final = haar_iwt(ll + final_delta, lh, hl, hh, fh, fw)
        pred = head_from_final(model, final, x, h, w)
        reg = mid_delta.abs().mean()
        if final_delta is not None:
            reg = reg + final_delta.abs().mean()
        loss = F.l1_loss(torch.clamp(pred, 0, 1), gt) + 1e-4 * reg
        loss.backward()
        opt.step()
        with torch.no_grad():
            psnr = tensor_psnr(pred, gt)
            if psnr > best_psnr:
                best_psnr = psnr
                best_loss = float(loss.detach().cpu())
                best_pred = pred.detach()
                last_mid_delta = mid_delta.detach()
                last_final_delta = final_delta.detach() if final_delta is not None else None
    assert best_pred is not None and last_mid_delta is not None
    payload = {
        "best_psnr": best_psnr,
        "best_loss": best_loss,
        "mid_delta_abs_mean": float(last_mid_delta.abs().mean().cpu()),
        "mid_delta_rms": float(torch.sqrt(torch.mean(last_mid_delta ** 2)).cpu()),
    }
    if last_final_delta is not None:
        payload["final_delta_abs_mean"] = float(last_final_delta.abs().mean().cpu())
        payload["final_delta_rms"] = float(torch.sqrt(torch.mean(last_final_delta ** 2)).cpu())
    return best_pred, payload


def summarize_variant(rows: list[dict[str, Any]], variant: str, scope: str = "all") -> dict[str, Any]:
    vals = [float(r["dPSNR"]) for r in rows if r["variant"] == variant]
    ssim = [float(r.get("dSSIM_proxy", 0.0)) for r in rows if r["variant"] == variant]
    a0 = [float(r["A0_PSNR"]) for r in rows if r["variant"] == variant]
    hard_cut = percentile(a0, 25)
    easy_cut = percentile(a0, 75)
    strong_cut = easy_cut
    hard = [float(r["dPSNR"]) for r in rows if r["variant"] == variant and float(r["A0_PSNR"]) <= hard_cut]
    easy = [float(r["dPSNR"]) for r in rows if r["variant"] == variant and float(r["A0_PSNR"]) >= easy_cut]
    strong = [float(r["dPSNR"]) for r in rows if r["variant"] == variant and float(r["A0_PSNR"]) >= strong_cut]
    return {
        "variant": variant,
        "scope": scope,
        "count": len(vals),
        "mean_dPSNR": mean(vals),
        "median_dPSNR": median(vals),
        "p05_dPSNR": percentile(vals, 5),
        "p01_dPSNR": percentile(vals, 1),
        "cvar5_dPSNR": mean(sorted(vals)[: max(1, int(round(0.05 * len(vals))))]) if vals else float("nan"),
        "hard_bottom25_dPSNR": mean(hard),
        "easy_top25_dPSNR": mean(easy),
        "positive_ratio": sum(v > 0 for v in vals) / len(vals) if vals else float("nan"),
        "severe_count": sum(v <= SEVERE for v in vals),
        "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
        "strong_reference_count": len(strong),
        "strong_reference_regressions": sum(v <= STRONG_REG for v in strong),
        "mean_dSSIM_proxy": mean(ssim),
    }


def gate(summary: dict[str, Any]) -> dict[str, Any]:
    strong_limit = max(3, int(round(0.05 * float(summary["strong_reference_count"]))))
    checks = {
        "mean_dPSNR_ge_0p20": float(summary["mean_dPSNR"]) >= 0.20,
        "hard_bottom25_ge_0p30": float(summary["hard_bottom25_dPSNR"]) >= 0.30,
        "easy_top25_ge_0": float(summary["easy_top25_dPSNR"]) >= 0.0,
        "severe_rate_le_0p025": float(summary["severe_rate"]) <= 0.025,
        "p05_ge_neg0p20": float(summary["p05_dPSNR"]) >= -0.20,
        "strong_regressions_le_5pct": int(summary["strong_reference_regressions"]) <= strong_limit,
    }
    return {"pass": all(checks.values()), "checks": checks, "strong_regression_limit": strong_limit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--steps-final", type=int, default=25)
    ap.add_argument("--steps-mid", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--delta-scale", type=float, default=0.50)
    ap.add_argument("--o2-grid", type=int, default=16)
    ap.add_argument("--mid-grid", type=int, default=8)
    ap.add_argument("--print-freq", type=int, default=25)
    args = ap.parse_args()

    from models.ConvIR import build_net

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        args.out_dir / "v217_r2_capacity_ladder_protocol.md",
        "\n".join(
            [
                "# v2.17 R2 Capacity Ladder Protocol",
                "",
                "No deployable model is trained here. Each oracle optimizes per-image bounded feature-LL corrections against train-derived GT only to test representational headroom.",
                "",
                "- O0: official A0 identity.",
                "- O1: final-feature LL global per-channel offset.",
                "- O2: final-feature LL bounded spatial correction.",
                "- O3: insertion-point oracles at final, mid, and mid+final feature LL.",
                "- O4: RGB LL oracle reference copied from v2.16 T1.",
                "",
                f"Correction bound: tanh(raw) times `{args.delta_scale}` channel-wise LL std.",
                "Locked Haze4K test remains untouched.",
            ]
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_net("base", "Haze4K", "original").to(device)
    model.load_state_dict(load_state(args.checkpoint, device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    split_rows = read_csv(args.split_csv)
    names = [row["name"] for row in split_rows]
    if args.max_images:
        names = names[: args.max_images]
    split_by_name = {row["name"]: row for row in split_rows}
    input_dir, gt_dir = train_dirs(args.data_dir)

    o1_rows: list[dict[str, Any]] = []
    o2_rows: list[dict[str, Any]] = []
    o3_rows: list[dict[str, Any]] = []
    o4_rows: list[dict[str, Any]] = []

    for idx, name in enumerate(names, 1):
        hazy = image_tensor(input_dir / name, device)
        gt = image_tensor(label_path(gt_dir, name), device)
        x, h, w = pad_to(hazy, 32)
        gt = gt[:, :, :h, :w]
        with torch.no_grad():
            parts = forward_parts(model, x)
            a0_pred = head_from_final(model, parts["final"], x, h, w)
            a0_psnr = tensor_psnr(a0_pred, gt)
        fold = int(split_by_name[name].get("oof_fold", 0))

        pred, stats = optimize_final_ll(
            model=model,
            final=parts["final"],
            x=x,
            gt=gt,
            h=h,
            w=w,
            mode="global",
            grid=1,
            steps=args.steps_final,
            lr=args.lr,
            delta_scale=args.delta_scale,
        )
        o1_rows.append(
            {
                "name": name,
                "oof_fold": fold,
                "variant": "O1_global_final_feature_ll",
                "A0_PSNR": a0_psnr,
                "candidate_PSNR": tensor_psnr(pred, gt),
                "dPSNR": tensor_psnr(pred, gt) - a0_psnr,
                **stats,
            }
        )

        pred, stats = optimize_final_ll(
            model=model,
            final=parts["final"],
            x=x,
            gt=gt,
            h=h,
            w=w,
            mode="spatial",
            grid=args.o2_grid,
            steps=args.steps_final,
            lr=args.lr,
            delta_scale=args.delta_scale,
        )
        o2_row = {
            "name": name,
            "oof_fold": fold,
            "variant": f"O2_spatial_final_feature_ll_grid{args.o2_grid}",
            "A0_PSNR": a0_psnr,
            "candidate_PSNR": tensor_psnr(pred, gt),
            "dPSNR": tensor_psnr(pred, gt) - a0_psnr,
            **stats,
        }
        o2_rows.append(o2_row)
        o3_rows.append({**o2_row, "variant": f"O3_final_feature_ll_grid{args.o2_grid}"})

        pred, stats = optimize_mid_ll(
            model=model,
            mid=parts["mid"],
            res1=parts["res1"],
            x=x,
            gt=gt,
            h=h,
            w=w,
            include_final=False,
            mid_grid=args.mid_grid,
            final_grid=args.o2_grid,
            steps=args.steps_mid,
            lr=args.lr,
            delta_scale=args.delta_scale,
        )
        o3_rows.append(
            {
                "name": name,
                "oof_fold": fold,
                "variant": f"O3_mid_feature_ll_grid{args.mid_grid}",
                "A0_PSNR": a0_psnr,
                "candidate_PSNR": tensor_psnr(pred, gt),
                "dPSNR": tensor_psnr(pred, gt) - a0_psnr,
                **stats,
            }
        )

        pred, stats = optimize_mid_ll(
            model=model,
            mid=parts["mid"],
            res1=parts["res1"],
            x=x,
            gt=gt,
            h=h,
            w=w,
            include_final=True,
            mid_grid=args.mid_grid,
            final_grid=args.o2_grid,
            steps=args.steps_mid,
            lr=args.lr,
            delta_scale=args.delta_scale,
        )
        o3_rows.append(
            {
                "name": name,
                "oof_fold": fold,
                "variant": f"O3_mid_grid{args.mid_grid}_plus_final_grid{args.o2_grid}",
                "A0_PSNR": a0_psnr,
                "candidate_PSNR": tensor_psnr(pred, gt),
                "dPSNR": tensor_psnr(pred, gt) - a0_psnr,
                **stats,
            }
        )

        src = split_by_name[name]
        ll_d = float(src.get("LL_oracle_dPSNR", 0.0))
        o4_rows.append(
            {
                "name": name,
                "oof_fold": fold,
                "variant": "O4_rgb_ll_reference",
                "A0_PSNR": float(src.get("A0_PSNR_runtime", a0_psnr)),
                "candidate_PSNR": float(src.get("A0_PSNR_runtime", a0_psnr)) + ll_d,
                "dPSNR": ll_d,
            }
        )

        if idx % args.print_freq == 0:
            print(f"V217_R2_CAPACITY {idx}/{len(names)}", flush=True)

    write_csv(args.out_dir / "v217_r2_o1_global_feature_ll_oracle.csv", o1_rows)
    write_csv(args.out_dir / "v217_r2_o2_spatial_feature_ll_oracle.csv", o2_rows)
    write_csv(args.out_dir / "v217_r2_o3_insertion_point_oracle.csv", o3_rows)
    write_csv(args.out_dir / "v217_r2_o4_rgb_ll_reference.csv", o4_rows)

    all_rows = o1_rows + o2_rows + o3_rows + o4_rows
    variants = sorted({row["variant"] for row in all_rows})
    summary_rows = []
    for variant in variants:
        s = summarize_variant(all_rows, variant)
        s.update({f"gate_{k}": v for k, v in gate(s)["checks"].items()})
        s["gate_pass"] = gate(s)["pass"]
        summary_rows.append(s)
    write_csv(args.out_dir / "v217_r2_ladder_summary.csv", summary_rows)

    fold_rows = []
    for variant in variants:
        for fold in sorted({int(row["oof_fold"]) for row in all_rows if row["variant"] == variant}):
            subset = [row for row in all_rows if row["variant"] == variant and int(row["oof_fold"]) == fold]
            fold_rows.append(summarize_variant(subset, variant, scope=f"fold{fold}"))
    write_csv(args.out_dir / "v217_r2_fold_report.csv", fold_rows)

    group_rows = []
    for variant in variants:
        rows = [row for row in all_rows if row["variant"] == variant]
        a0_values = [float(row["A0_PSNR"]) for row in rows]
        hard_cut = percentile(a0_values, 25)
        easy_cut = percentile(a0_values, 75)
        groups = {
            "hard_bottom25": [row for row in rows if float(row["A0_PSNR"]) <= hard_cut],
            "easy_top25": [row for row in rows if float(row["A0_PSNR"]) >= easy_cut],
            "middle50": [row for row in rows if hard_cut < float(row["A0_PSNR"]) < easy_cut],
        }
        for group, subset in groups.items():
            group_rows.append(summarize_variant(subset, variant, scope=group))
    write_csv(args.out_dir / "v217_r2_group_report.csv", group_rows)

    summary_by_variant = {row["variant"]: row for row in summary_rows}
    o1_pass = any(row["variant"].startswith("O1_") and row["gate_pass"] for row in summary_rows)
    o2_pass = any(row["variant"].startswith("O2_") and row["gate_pass"] for row in summary_rows)
    o3_pass = any(row["variant"].startswith("O3_") and row["gate_pass"] for row in summary_rows)
    o4_pass = any(row["variant"].startswith("O4_") and row["gate_pass"] for row in summary_rows)
    if o1_pass:
        decision = "R2_O1_GLOBAL_FEATURE_LL_PASS_REVIEW_WLDB_A2_OBJECTIVE"
    elif o2_pass:
        decision = "R2_O1_FAIL_O2_SPATIAL_FEATURE_LL_PASS_AUTHORIZE_R3"
    elif o3_pass:
        decision = "R2_O2_FAIL_O3_INSERTION_POINT_PASS_AUTHORIZE_R3_MID_FINAL"
    elif o4_pass:
        decision = "R2_INTERNAL_FEATURE_LOWBAND_FAIL_RGB_LL_PASS_STOP_WLDB_B_DESIGN"
    else:
        decision = "R2_ALL_ORACLES_FAIL_RECHECK_T1_PIPELINE_STOP_LOWBAND"

    best_variant = max(summary_rows, key=lambda row: float(row["mean_dPSNR"]))["variant"] if summary_rows else "NA"
    write_text(
        args.out_dir / "v217_r2_decision.md",
        "\n".join(
            [
                "# v2.17 R2 Capacity Ladder Decision",
                "",
                f"Decision: `{decision}`",
                "",
                f"- O1 pass: `{o1_pass}`",
                f"- O2 pass: `{o2_pass}`",
                f"- O3 pass: `{o3_pass}`",
                f"- O4 pass: `{o4_pass}`",
                f"- Best mean variant: `{best_variant}`",
                "",
                "Interpretation:",
                "",
                "- If only O2/O3 passes, close WLDB-A but keep spatial/internal feature-lowband open.",
                "- If no internal feature oracle passes while O4 passes, RGB LL headroom did not transfer to frozen internal feature correction.",
                "- Locked Haze4K test remains untouched.",
            ]
        ),
    )
    write_json(
        args.out_dir / "v217_r2_closeout.json",
        {
            "decision": decision,
            "summary_by_variant": summary_by_variant,
            "o1_pass": o1_pass,
            "o2_pass": o2_pass,
            "o3_pass": o3_pass,
            "o4_pass": o4_pass,
            "locked_test_touched": False,
            "training_launched": False,
        },
    )
    print("V217_R2_CAPACITY_LADDER_OK", decision, flush=True)


if __name__ == "__main__":
    main()
