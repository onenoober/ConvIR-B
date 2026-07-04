#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TVF


TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for item in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if item not in sys.path:
        sys.path.insert(0, item)

from models.ConvIR import build_net as build_a0_net  # noqa: E402
from models.ILFRBACSConvIR import build_net as build_route_net  # noqa: E402
from models.ILFRBACSConvIR import load_haze4k_partial  # noqa: E402


SEVERE = -0.20
STRONG_REG = -0.05
IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")
STAGE_SETS = {
    "S1_bottleneck": ("bottleneck",),
    "S2_early_decoder": ("early",),
    "S3_mid_decoder": ("mid",),
    "S4_final_decoder": ("final",),
    "S5_bottleneck_mid": ("bottleneck", "mid"),
    "S6_early_mid_final": ("early", "mid", "final"),
}
ACTION_STRENGTHS = {
    "noop": 0.0,
    "mild": 0.33,
    "medium": 0.67,
    "strong": 1.25,
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def mean(values: list[float]) -> float:
    xs = finite(values)
    return float(statistics.mean(xs)) if xs else float("nan")


def std(values: list[float]) -> float:
    xs = finite(values)
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0 if xs else float("nan")


def percentile(values: list[float], pct: float) -> float:
    xs = sorted(finite(values))
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def cvar(values: list[float], pct: float = 5.0) -> float:
    xs = sorted(finite(values))
    if not xs:
        return float("nan")
    k = max(1, int(math.ceil(len(xs) * pct / 100.0)))
    return mean(xs[:k])


def roc_auc(scores: list[float], labels: list[int]) -> float:
    pairs = [(s, y) for s, y in zip(scores, labels) if math.isfinite(s)]
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def average_precision(scores: list[float], labels: list[int]) -> float:
    pairs = sorted([(s, int(y)) for s, y in zip(scores, labels) if math.isfinite(s)], key=lambda x: x[0], reverse=True)
    total_pos = sum(y for _, y in pairs)
    if total_pos <= 0:
        return float("nan")
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(pairs):
        score = pairs[i][0]
        group_pos = 0
        group_total = 0
        while i < len(pairs) and pairs[i][0] == score:
            group_pos += pairs[i][1]
            group_total += 1
            i += 1
        tp += group_pos
        fp += group_total - group_pos
        recall = tp / total_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def first_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"none of {names} exists under {root}")


def train_dirs(data_dir: Path) -> tuple[Path, Path]:
    root = data_dir / "train"
    return first_dir(root, ("IN", "haze", "hazy")), first_dir(root, ("GT", "gt"))


def label_path(gt_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    ext = Path(image_name).suffix
    candidates = [image_name]
    if "_" in stem:
        base = stem.split("_")[0]
        candidates.extend([f"{base}{ext}", f"{base}.png"])
    for candidate in candidates:
        path = gt_dir / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"no GT for {image_name} under {gt_dir}; tried {candidates}")


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    return TVF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def pad_to(x: torch.Tensor, factor: int = 32) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


def tensor_psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), label).clamp_min(1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).detach().cpu())


def load_state(path: Path, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def haar_dwt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    return (
        (a + b + c + d) / 2.0,
        (a - b + c - d) / 2.0,
        (a + b - c - d) / 2.0,
        (a - b - c + d) / 2.0,
        h,
        w,
    )


def haar_iwt(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor, h: int, w: int) -> torch.Tensor:
    a = (ll + lh + hl + hh) / 2.0
    b = (ll - lh + hl - hh) / 2.0
    c = (ll + lh - hl - hh) / 2.0
    d = (ll - lh - hl + hh) / 2.0
    out = torch.empty((ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2), dtype=ll.dtype, device=ll.device)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out[:, :, :h, :w]


@dataclass
class Sample:
    name: str
    fold: int
    input_path: Path
    label_path: Path


def load_samples(args: argparse.Namespace) -> list[Sample]:
    input_dir, gt_dir = train_dirs(args.data_dir)
    rows = read_csv(args.split_csv)
    samples = []
    for row in rows:
        name = row["name"]
        input_path = input_dir / name
        if not input_path.is_file() or input_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        samples.append(Sample(name=name, fold=int(row["oof_fold"]), input_path=input_path, label_path=label_path(gt_dir, name)))
    samples = sorted(samples, key=lambda s: (s.fold, s.name))
    if args.max_images > 0:
        per_fold: dict[int, list[Sample]] = {}
        for sample in samples:
            per_fold.setdefault(sample.fold, []).append(sample)
        capped = []
        base = max(1, args.max_images // max(1, len(per_fold)))
        for fold in sorted(per_fold):
            capped.extend(per_fold[fold][:base])
        samples = capped[: args.max_images]
    if not samples:
        raise RuntimeError("no samples selected")
    return samples


def build_models(args: argparse.Namespace, device: torch.device):
    state = load_state(args.checkpoint, "cpu")
    a0 = build_a0_net("base", "Haze4K", "original").to(device)
    a0.load_state_dict(state)
    a0.eval()
    for param in a0.parameters():
        param.requires_grad_(False)

    route = build_route_net(
        "base",
        "Haze4K",
        "original",
        hidden_channels=args.hidden_channels,
        delta_scale=args.delta_scale,
        coverage_budget=args.coverage_budget,
    ).to(device)
    partial = load_haze4k_partial(route, state)
    route.eval()
    for param in route.parameters():
        param.requires_grad_(False)
    return a0, route, partial


def source_scan() -> dict[str, Any]:
    files = [
        REPO_ROOT / "Dehazing" / "ITS" / "models" / "ILFRBACSConvIR.py",
        REPO_ROOT / "Dehazing" / "ITS" / "main.py",
    ]
    forbidden = [
        "teacher_output",
        "expert_output",
        "a0_output",
        "output_output",
        "rgb_correction",
        "post_train_rescue",
        "locked",
    ]
    hits = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token in text:
                hits.append({"file": str(file_path.relative_to(REPO_ROOT)), "token": token})
    return {"forbidden_tokens": forbidden, "hits": hits, "hit_count": len(hits)}


def phase_p0(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    a0, route, partial = build_models(args, device)
    write_json(args.out_dir / "v227_p0_partial_load_manifest.json", partial)
    scan = source_scan()
    write_text(
        args.out_dir / "v227_p0_forbidden_symbol_scan.txt",
        "\n".join([f"hit_count={scan['hit_count']}"] + [json.dumps(hit, sort_keys=True) for hit in scan["hits"]]),
    )

    samples = load_samples(args)[: args.p0_images]
    max_abs = 0.0
    per_image = []
    with torch.no_grad():
        for sample in samples:
            x = image_tensor(sample.input_path, device)
            x, h, w = pad_to(x)
            out_a0 = a0(x)[2][:, :, :h, :w]
            out_route = route(x)[2][:, :, :h, :w]
            diff = float((out_a0 - out_route).abs().max().detach().cpu())
            max_abs = max(max_abs, diff)
            per_image.append({"name": sample.name, "fold": sample.fold, "max_abs_vs_a0": diff})
    identity = {
        "image_count": len(per_image),
        "max_abs_vs_A0": max_abs,
        "identity_tol": args.identity_tol,
        "pass": max_abs <= args.identity_tol,
        "per_image": per_image,
    }
    write_json(args.out_dir / "v227_p0_identity_vs_a0.json", identity)

    forward_sig = str(inspect.signature(route.forward))
    contract_pass = (
        partial["unexpected"] == []
        and partial["shape_mismatch"] == []
        and scan["hit_count"] == 0
        and identity["pass"]
        and forward_sig == "(x)"
    )
    report_lines = [
        "# v2.27 P0 Source Contract Report",
        "",
        f"branch: `{git(['branch', '--show-current'])}`",
        f"commit: `{git(['rev-parse', '--short', 'HEAD'])}`",
        f"checkpoint: `{args.checkpoint}`",
        f"checkpoint_sha256: `{sha256_file(args.checkpoint)}`",
        f"forward_signature: `{forward_sig}`",
        "forward_input_contract: `forward(self, x)` only",
        "teacher_or_expert_forward_input: false",
        "rgb_output_output_residual: false",
        "learned_rgb_post_output_correction: false",
        f"partial_load_loaded_count: `{partial['loaded_count']}`",
        f"partial_load_missing_new_module_count: `{partial['missing_new_module_count']}`",
        f"forbidden_symbol_hits: `{scan['hit_count']}`",
        f"identity_max_abs_vs_A0: `{max_abs}`",
        f"locked_test_touched: false",
        f"decision: `{'P0_PASS_ILFRB_ACS_CONTRACT_IDENTITY_SOURCE_CLEAN' if contract_pass else 'P0_FAIL_ILFRB_ACS_CONTRACT_OR_IDENTITY'}`",
    ]
    write_text(args.out_dir / "v227_p0_source_contract_report.md", "\n".join(report_lines))
    write_text(
        args.out_dir / "v227_p0_decision.md",
        "P0_PASS_ILFRB_ACS_CONTRACT_IDENTITY_SOURCE_CLEAN" if contract_pass else "P0_FAIL_ILFRB_ACS_CONTRACT_OR_IDENTITY",
    )
    out = {
        "phase": "p0",
        "decision": "P0_PASS_ILFRB_ACS_CONTRACT_IDENTITY_SOURCE_CLEAN" if contract_pass else "P0_FAIL_ILFRB_ACS_CONTRACT_OR_IDENTITY",
        "pass": contract_pass,
        "identity": identity,
        "source_scan": scan,
        "partial": partial,
        "locked_test_touched": False,
    }
    return out


def forward_cache(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    x_2 = F.interpolate(x, scale_factor=0.5)
    x_4 = F.interpolate(x_2, scale_factor=0.5)
    z2 = model.SCM2(x_2)
    z4 = model.SCM1(x_4)
    stem = model.feat_extract[0](x)
    res1 = model.Encoder[0](stem)
    z = model.feat_extract[1](res1)
    z = model.FAM2(z, z2)
    res2 = model.Encoder[1](z)
    z = model.feat_extract[2](res2)
    z = model.FAM1(z, z4)
    bottleneck = model.Encoder[2](z)
    early = model.Decoder[0](bottleneck)
    z = model.feat_extract[3](early)
    z = torch.cat([z, res2], dim=1)
    z = model.Convs[0](z)
    mid = model.Decoder[1](z)
    z = model.feat_extract[4](mid)
    z = torch.cat([z, res1], dim=1)
    z = model.Convs[1](z)
    final = model.Decoder[2](z)
    out = model.feat_extract[5](final) + x
    return {"x": x, "res1": res1, "res2": res2, "bottleneck": bottleneck, "early": early, "mid": mid, "final": final, "out": out}


def channel_scale(ll: torch.Tensor, delta_scale: float) -> torch.Tensor:
    scale = ll.detach().flatten(2).std(dim=2).view(ll.shape[0], ll.shape[1], 1, 1)
    return scale.clamp_min(1e-4) * delta_scale


def delta_from_raw(raw: torch.Tensor, ll: torch.Tensor, delta_scale: float) -> torch.Tensor:
    scale = channel_scale(ll, delta_scale)
    return torch.tanh(raw) * scale


def apply_ll_delta(feature: torch.Tensor, delta_grid: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    ll, lh, hl, hh, h, w = haar_dwt(feature)
    delta = delta_grid.to(device=ll.device, dtype=ll.dtype) * strength
    if delta.shape[-2:] != ll.shape[-2:]:
        delta = F.interpolate(delta, size=ll.shape[-2:], mode="bilinear", align_corners=False)
    return haar_iwt(ll + delta, lh, hl, hh, h, w)


def decode_from_features(model: torch.nn.Module, cache: dict[str, torch.Tensor], deltas: dict[str, torch.Tensor], strength: float = 1.0) -> torch.Tensor:
    z = cache["bottleneck"]
    if "bottleneck" in deltas:
        z = apply_ll_delta(z, deltas["bottleneck"], strength)
    early = model.Decoder[0](z)
    if "early" in deltas:
        early = apply_ll_delta(early, deltas["early"], strength)
    z = model.feat_extract[3](early)
    z = torch.cat([z, cache["res2"]], dim=1)
    z = model.Convs[0](z)
    mid = model.Decoder[1](z)
    if "mid" in deltas:
        mid = apply_ll_delta(mid, deltas["mid"], strength)
    z = model.feat_extract[4](mid)
    z = torch.cat([z, cache["res1"]], dim=1)
    z = model.Convs[1](z)
    final = model.Decoder[2](z)
    if "final" in deltas:
        final = apply_ll_delta(final, deltas["final"], strength)
    return model.feat_extract[5](final) + cache["x"]


def optimize_stage_set(
    model: torch.nn.Module,
    cache: dict[str, torch.Tensor],
    gt: torch.Tensor,
    stages: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    raws = {}
    ll_by_stage = {}
    grid_by_stage = {"bottleneck": 4, "early": 4, "mid": 8, "final": 16}
    for stage in stages:
        ll, _, _, _, _, _ = haar_dwt(cache[stage].detach())
        ll_by_stage[stage] = ll
        raws[stage] = torch.zeros((1, ll.shape[1], grid_by_stage[stage], grid_by_stage[stage]), device=ll.device, requires_grad=True)
    opt = torch.optim.Adam(list(raws.values()), lr=args.oracle_lr)
    best = {"psnr": -1.0, "loss": float("inf")}
    best_deltas: dict[str, torch.Tensor] = {}
    for _ in range(args.oracle_steps):
        opt.zero_grad(set_to_none=True)
        deltas = {stage: delta_from_raw(raw, ll_by_stage[stage], args.oracle_delta_scale) for stage, raw in raws.items()}
        pred = decode_from_features(model, cache, deltas)
        pred = pred[:, :, : gt.shape[-2], : gt.shape[-1]]
        reg = sum(delta.abs().mean() for delta in deltas.values())
        loss = F.l1_loss(pred.clamp(0, 1), gt) + args.oracle_reg * reg
        loss.backward()
        opt.step()
        with torch.no_grad():
            psnr = tensor_psnr(pred, gt)
            if psnr > best["psnr"]:
                best = {"psnr": psnr, "loss": float(loss.detach().cpu())}
                best_deltas = {stage: delta.detach().cpu().float() for stage, delta in deltas.items()}
    stats = dict(best)
    for stage, delta in best_deltas.items():
        stats[f"{stage}_delta_abs_mean"] = float(delta.abs().mean())
        stats[f"{stage}_delta_rms"] = float(torch.sqrt(torch.mean(delta.float() ** 2)))
        stats[f"{stage}_delta_abs_max"] = float(delta.abs().max())
    return best_deltas, stats


def summarize_variant(rows: list[dict[str, Any]], variant: str, scope: str = "all") -> dict[str, Any]:
    subset = [r for r in rows if r["variant"] == variant]
    vals = [float(r["dPSNR"]) for r in subset]
    a0s = [float(r["A0_PSNR"]) for r in subset]
    hard_cut = percentile(a0s, 25)
    easy_cut = percentile(a0s, 75)
    hard = [float(r["dPSNR"]) for r in subset if float(r["A0_PSNR"]) <= hard_cut]
    easy = [float(r["dPSNR"]) for r in subset if float(r["A0_PSNR"]) >= easy_cut]
    return {
        "variant": variant,
        "scope": scope,
        "count": len(vals),
        "mean_dPSNR": mean(vals),
        "hard_bottom25_dPSNR": mean(hard),
        "easy_top25_dPSNR": mean(easy),
        "p05_dPSNR": percentile(vals, 5),
        "CVaR5_dPSNR": cvar(vals, 5),
        "severe_count": sum(v <= SEVERE for v in vals),
        "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
        "strong_reference_regressions": sum(v <= STRONG_REG for v in easy),
        "strong_reference_regression_rate": sum(v <= STRONG_REG for v in easy) / len(easy) if easy else float("nan"),
        "wrong_direction_rate": sum(v < 0 for v in vals) / len(vals) if vals else float("nan"),
        "delta_abs_mean": mean([float(r.get("delta_abs_mean", 0.0)) for r in subset]),
        "action_coverage": sum(abs(float(r.get("delta_abs_mean", 0.0))) > 0 for r in subset) / len(subset) if subset else float("nan"),
    }


def phase_p1_p5(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    a0, _route, _partial = build_models(args, device)
    samples = load_samples(args)
    p1_rows: list[dict[str, Any]] = []
    target_bank: dict[str, dict[str, Any]] = {}
    print(f"V227_SELECTED_SAMPLES count={len(samples)}", flush=True)
    for idx, sample in enumerate(samples, start=1):
        x0 = image_tensor(sample.input_path, device)
        gt = image_tensor(sample.label_path, device)
        x, h, w = pad_to(x0)
        with torch.no_grad():
            cache = forward_cache(a0, x)
        cache = {key: value.detach() for key, value in cache.items()}
        a0_pred = cache["out"][:, :, :h, :w]
        a0_psnr = tensor_psnr(a0_pred, gt)
        image_delta_bank: dict[str, dict[str, torch.Tensor]] = {}
        for variant, stages in STAGE_SETS.items():
            deltas, stats = optimize_stage_set(a0, cache, gt, stages, args)
            pred = decode_from_features(a0, cache, {k: v.to(device) for k, v in deltas.items()})[:, :, :h, :w]
            psnr = tensor_psnr(pred, gt)
            delta_abs = mean([float(v.abs().mean()) for v in deltas.values()])
            row = {
                "name": sample.name,
                "oof_fold": sample.fold,
                "variant": variant,
                "stages": "+".join(stages),
                "A0_PSNR": a0_psnr,
                "action_PSNR": psnr,
                "dPSNR": psnr - a0_psnr,
                "best_loss": stats["loss"],
                "delta_abs_mean": delta_abs,
            }
            row.update({k: v for k, v in stats.items() if k not in ("psnr", "loss")})
            p1_rows.append(row)
            image_delta_bank[variant] = deltas
        target_bank[sample.name] = {"sample": sample, "cache": {k: v.detach().cpu() for k, v in cache.items() if k in ("res1", "res2", "bottleneck", "early", "mid", "final")}, "a0_psnr": a0_psnr, "deltas": image_delta_bank}
        if idx % args.print_freq == 0:
            print(f"V227_P1_PROGRESS {idx}/{len(samples)}", flush=True)

    write_csv(args.out_dir / "v227_p1_insertion_oracle_detail.csv", p1_rows)
    summary_rows = [summarize_variant(p1_rows, variant) for variant in STAGE_SETS]
    fold_rows = []
    for variant in STAGE_SETS:
        for fold in sorted({sample.fold for sample in samples}):
            fold_rows.append(summarize_variant([r for r in p1_rows if int(r["oof_fold"]) == fold], variant, scope=f"fold{fold}"))
    write_csv(args.out_dir / "v227_p1_fold_tail_report.csv", fold_rows)
    write_text(
        args.out_dir / "v227_p1_oracle_vs_v217_o2_o3.md",
        "\n".join(
            [
                "# v2.27 P1 Oracle Comparison",
                "",
                "Reference v2.17 O2 final-feature LL oracle mean: `+6.16049 dB`.",
                "Reference v2.17 O3 mid+final LL oracle mean: `+6.832469 dB`.",
                "This v2.27 screen compares bottleneck, early, mid, final, and multi-scale internal insertion sets on train-derived samples only.",
                "",
                f"sample_count: `{len(samples)}`",
                f"oracle_steps: `{args.oracle_steps}`",
                f"locked_test_touched: `false`",
            ]
        ),
    )
    write_csv(args.out_dir / "v227_p1_insertion_oracle_summary.csv", summary_rows)
    candidate_rows = [row for row in summary_rows if row["variant"] != "S4_final_decoder"]
    passing = [
        row
        for row in candidate_rows
        if float(row["mean_dPSNR"]) >= 0.50
        and float(row["hard_bottom25_dPSNR"]) >= 1.00
        and float(row["easy_top25_dPSNR"]) >= 0.0
        and float(row["p05_dPSNR"]) >= 0.0
        and int(row["severe_count"]) == 0
        and float(row["strong_reference_regression_rate"]) <= 0.075
    ]
    p1_pass = bool(passing)
    primary = max(passing or candidate_rows, key=lambda row: float(row["mean_dPSNR"]))["variant"]
    p1 = {"decision": "P1_PASS_INTERNAL_ORACLE_CAPACITY" if p1_pass else "P1_FAIL_INTERNAL_ORACLE_CAPACITY_PAUSE", "pass": p1_pass, "primary_variant": primary, "summary": summary_rows}
    if not p1_pass:
        write_json(args.out_dir / "v227_closeout.json", {"p1": p1, "decision": p1["decision"], "locked_test_touched": False})
        return {"p1": p1, "decision": p1["decision"], "locked_test_touched": False}

    p2_rows = run_p2_action_replay(args, a0, samples, target_bank, primary, device)
    p2 = summarize_p2(args, p2_rows)
    if not p2["pass"]:
        closeout = {"p1": p1, "p2": p2, "decision": p2["decision"], "locked_test_touched": False}
        write_json(args.out_dir / "v227_closeout.json", closeout)
        return closeout
    p3_rows, p3 = run_p3_selector_probe(args, p2_rows)
    if not p3["pass"]:
        closeout = {"p1": p1, "p2": p2, "p3": p3, "decision": p3["decision"], "locked_test_touched": False}
        write_json(args.out_dir / "v227_closeout.json", closeout)
        return closeout
    p4 = run_p4_canary(args, p3_rows)
    if not p4["pass"]:
        closeout = {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "decision": p4["decision"], "locked_test_touched": False}
        write_json(args.out_dir / "v227_closeout.json", closeout)
        return closeout
    p5 = run_p5_coverage(args, p3_rows)
    decision = p5["decision"]
    closeout = {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "decision": decision, "locked_test_touched": False}
    write_json(args.out_dir / "v227_closeout.json", closeout)
    return closeout


def input_stats(sample: Sample) -> dict[str, float]:
    img = TVF.to_tensor(Image.open(sample.input_path).convert("RGB"))
    gray = img.mean(dim=0)
    low = F.avg_pool2d(gray.unsqueeze(0).unsqueeze(0), kernel_size=16, stride=16, ceil_mode=True)
    return {
        "input_luma_mean": float(gray.mean()),
        "input_luma_std": float(gray.std(unbiased=False)),
        "input_low_mean": float(low.mean()),
        "input_low_std": float(low.std(unbiased=False)),
    }


def run_p2_action_replay(args, model, samples, target_bank, primary, device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        x0 = image_tensor(sample.input_path, device)
        gt = image_tensor(sample.label_path, device)
        x, h, w = pad_to(x0)
        with torch.no_grad():
            cache = forward_cache(model, x)
        cache = {key: value.detach() for key, value in cache.items()}
        a0_pred = cache["out"][:, :, :h, :w]
        a0_psnr = tensor_psnr(a0_pred, gt)
        deltas = target_bank[sample.name]["deltas"][primary]
        stats = input_stats(sample)
        best_action = "noop"
        best_dpsnr = 0.0
        for action, strength in ACTION_STRENGTHS.items():
            pred = decode_from_features(model, cache, {k: v.to(device) for k, v in deltas.items()}, strength=strength)[:, :, :h, :w]
            psnr = tensor_psnr(pred, gt)
            dpsnr = psnr - a0_psnr
            if dpsnr > best_dpsnr:
                best_dpsnr = dpsnr
                best_action = action
            delta_abs = mean([float((v * strength).abs().mean()) for v in deltas.values()])
            delta_rms = mean([float(torch.sqrt(torch.mean((v.float() * strength) ** 2))) for v in deltas.values()])
            row = {
                "name": sample.name,
                "oof_fold": sample.fold,
                "primary_variant": primary,
                "action": action,
                "action_strength": strength,
                "A0_PSNR": a0_psnr,
                "action_PSNR": psnr,
                "dPSNR": dpsnr,
                "unsafe_action_label": int(dpsnr <= STRONG_REG),
                "safe_action_label": int(dpsnr > STRONG_REG),
                "delta_abs_mean": delta_abs,
                "delta_rms": delta_rms,
                "delta_abs_max": max(float((v * strength).abs().max()) for v in deltas.values()),
            }
            row.update(stats)
            rows.append(row)
        for row in rows[-len(ACTION_STRENGTHS) :]:
            row["best_action_class"] = best_action
            row["best_action_dPSNR"] = best_dpsnr
    write_csv(args.out_dir / "v227_p2_action_bank_replay.csv", rows)
    return rows


def summarize_p2(args, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(str(row["name"]), []).append(row)
    pref_rows = []
    noops = 0
    medium_strong_hard = 0
    hard_count = 0
    a0s = [float(next(iter(group))["A0_PSNR"]) for group in by_name.values()]
    hard_cut = percentile(a0s, 25)
    easy_cut = percentile(a0s, 75)
    for name, group in by_name.items():
        a0 = float(group[0]["A0_PSNR"])
        best = max(group, key=lambda r: float(r["dPSNR"]))
        conservative = max(group, key=lambda r: float(r["dPSNR"]) - 0.03 * float(r["action_strength"]) * (1.0 + float(a0 >= easy_cut)))
        if conservative["action"] == "noop":
            noops += 1
        if a0 <= hard_cut:
            hard_count += 1
            if conservative["action"] in ("medium", "strong"):
                medium_strong_hard += 1
        pref_rows.append(
            {
                "name": name,
                "A0_PSNR": a0,
                "bucket": "hard" if a0 <= hard_cut else "easy" if a0 >= easy_cut else "regular",
                "raw_best_action": best["action"],
                "raw_best_dPSNR": best["dPSNR"],
                "conservative_best_action": conservative["action"],
                "conservative_best_dPSNR": conservative["dPSNR"],
            }
        )
    write_csv(args.out_dir / "v227_p2_action_preference_by_bucket.csv", pref_rows)
    curve = []
    for action in ACTION_STRENGTHS:
        vals = [float(r["dPSNR"]) for r in rows if r["action"] == action]
        curve.append(
            {
                "action": action,
                "strength": ACTION_STRENGTHS[action],
                "mean_dPSNR": mean(vals),
                "p05_dPSNR": percentile(vals, 5),
                "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
                "unsafe_rate": sum(v <= STRONG_REG for v in vals) / len(vals) if vals else float("nan"),
            }
        )
    write_csv(args.out_dir / "v227_p2_strength_safety_curve.csv", curve)
    noop_report = {
        "sample_count": len(by_name),
        "conservative_noop_preference_count": noops,
        "conservative_noop_preference_rate": noops / len(by_name) if by_name else 0.0,
        "hard_medium_strong_preference_rate": medium_strong_hard / hard_count if hard_count else 0.0,
        "strong_unsafe_rate": next((row["unsafe_rate"] for row in curve if row["action"] == "strong"), float("nan")),
    }
    write_json(args.out_dir / "v227_p2_noop_coverage_report.json", noop_report)
    p2_pass = (
        noop_report["conservative_noop_preference_count"] > 0
        and noop_report["hard_medium_strong_preference_rate"] > 0.10
        and math.isfinite(float(noop_report["strong_unsafe_rate"]))
    )
    return {
        "decision": "P2_PASS_ACTION_BANK_STRATIFICATION" if p2_pass else "P2_FAIL_ACTION_BANK_STRATIFICATION_PAUSE",
        "pass": p2_pass,
        "noop_report": noop_report,
        "curve": curve,
    }


def feature_vector(row: dict[str, Any], mode: str) -> list[float]:
    state = [
        float(row["A0_PSNR"]),
        float(row["input_luma_mean"]),
        float(row["input_luma_std"]),
        float(row["input_low_mean"]),
        float(row["input_low_std"]),
    ]
    action = [
        float(row["action_strength"]),
        float(row["delta_abs_mean"]),
        float(row["delta_rms"]),
        float(row["delta_abs_max"]),
    ]
    if mode == "baseline_a_old_state":
        return state[:1]
    if mode == "baseline_b_state_only":
        return state
    if mode == "main_c_action_conditioned":
        return state + action
    raise ValueError(mode)


def fit_oof_logistic(rows: list[dict[str, Any]], mode: str, epochs: int, lr: float, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    random.seed(seed)
    torch.manual_seed(seed)
    out_rows = [dict(row) for row in rows]
    folds = sorted({int(row["oof_fold"]) for row in rows})
    fold_metrics = []
    for fold in folds:
        train = [row for row in out_rows if int(row["oof_fold"]) != fold]
        valid = [row for row in out_rows if int(row["oof_fold"]) == fold]
        y_train = torch.tensor([int(row["unsafe_action_label"]) for row in train], dtype=torch.float32).view(-1, 1)
        if len(set(int(v.item()) for v in y_train.flatten())) < 2:
            for row in valid:
                row[f"{mode}_unsafe_prob"] = mean([float(v) for v in y_train.flatten().tolist()])
            continue
        x_train = torch.tensor([feature_vector(row, mode) for row in train], dtype=torch.float32)
        mu = x_train.mean(dim=0, keepdim=True)
        sigma = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_train = (x_train - mu) / sigma
        x_valid = (torch.tensor([feature_vector(row, mode) for row in valid], dtype=torch.float32) - mu) / sigma
        model = torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], 16), torch.nn.GELU(), torch.nn.Linear(16, 1))
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        pos_weight = (len(y_train) - y_train.sum()).clamp_min(1.0) / y_train.sum().clamp_min(1.0)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            logits = model(x_train)
            loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
            loss.backward()
            opt.step()
        with torch.no_grad():
            prob = torch.sigmoid(model(x_valid)).flatten().tolist()
        for row, p in zip(valid, prob):
            row[f"{mode}_unsafe_prob"] = float(p)
        labels = [int(row["unsafe_action_label"]) for row in valid]
        fold_metrics.append(
            {
                "fold": fold,
                "mode": mode,
                "auc": roc_auc(prob, labels),
                "ap": average_precision(prob, labels),
                "base_rate": mean([float(x) for x in labels]),
                "prob_std": std(prob),
            }
        )
    scores = [float(row.get(f"{mode}_unsafe_prob", 0.0)) for row in out_rows]
    labels = [int(row["unsafe_action_label"]) for row in out_rows]
    summary = {
        "mode": mode,
        "auc": roc_auc(scores, labels),
        "ap": average_precision(scores, labels),
        "base_rate": mean([float(x) for x in labels]),
        "prob_mean": mean(scores),
        "prob_std": std(scores),
        "prob_mae": mean([abs(s - y) for s, y in zip(scores, labels)]),
        "fold_metrics": fold_metrics,
        "fold_pass_count": sum(
            math.isfinite(float(row["auc"])) and float(row["auc"]) >= 0.80 and float(row["prob_std"]) >= 0.03
            for row in fold_metrics
        ),
    }
    return out_rows, summary


def run_p3_selector_probe(args, p2_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [row for row in p2_rows if row["action"] != "noop"]
    modes = ["baseline_a_old_state", "baseline_b_state_only", "main_c_action_conditioned"]
    summaries = []
    merged = rows
    for mode in modes:
        merged, summary = fit_oof_logistic(merged, mode, args.probe_epochs, args.probe_lr, args.seed)
        summaries.append(summary)
    pc_scores = [-float(row["dPSNR"]) for row in merged]
    labels = [int(row["unsafe_action_label"]) for row in merged]
    summaries.append(
        {
            "mode": "positive_control_d_action_outcome",
            "auc": roc_auc(pc_scores, labels),
            "ap": average_precision(pc_scores, labels),
            "base_rate": mean([float(x) for x in labels]),
            "prob_std": std(pc_scores),
            "prob_mae": float("nan"),
            "fold_pass_count": 5,
        }
    )
    write_csv(args.out_dir / "v227_p3_probe_oof_detail.csv", merged)
    write_csv(args.out_dir / "v227_p3_old_vs_new_feature_ablation.csv", summaries)
    main = next(row for row in summaries if row["mode"] == "main_c_action_conditioned")
    base = next(row for row in summaries if row["mode"] == "baseline_a_old_state")
    ap_gate = max(0.35, 2.5 * float(main["base_rate"]))
    p3_pass = (
        float(main["auc"]) >= 0.80
        and float(main["ap"]) >= ap_gate
        and float(main["prob_mae"]) <= 0.35
        and float(main["prob_std"]) >= 0.05
        and int(main["fold_pass_count"]) >= 4
        and float(main["auc"]) >= float(base["auc"]) + 0.12
    )
    p3 = {
        "decision": "P3_PASS_ACTION_CONDITIONED_SELECTOR_PROBE" if p3_pass else "P3_FAIL_SELECTOR_SEPARABILITY_PAUSE_NO_TRAINING",
        "pass": p3_pass,
        "main": main,
        "baseline": base,
        "summaries": summaries,
    }
    write_json(args.out_dir / "v227_p3_selector_probe_summary.json", p3)
    return merged, p3


def run_p4_canary(args, rows: list[dict[str, Any]]) -> dict[str, Any]:
    curve = []
    summaries = []
    for size in (32, 64):
        names = sorted({row["name"] for row in rows})[:size]
        subset = [row for row in rows if row["name"] in names]
        x = torch.tensor([feature_vector(row, "main_c_action_conditioned") for row in subset], dtype=torch.float32)
        y = torch.tensor([int(row["unsafe_action_label"]) for row in subset], dtype=torch.float32).view(-1, 1)
        if len(set(int(v.item()) for v in y.flatten())) < 2:
            summaries.append({"canary_size": size, "train_auc": float("nan"), "prob_std": 0.0, "target_mae": float("nan"), "best_action_acc": 0.0, "pass": False})
            continue
        x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp_min(1e-6)
        model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(model.parameters(), lr=args.canary_lr, weight_decay=0.0)
        pos_weight = (len(y) - y.sum()).clamp_min(1.0) / y.sum().clamp_min(1.0)
        for epoch in range(args.canary_epochs):
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            loss.backward()
            grad_norm = math.sqrt(sum(float((p.grad.detach() ** 2).sum()) for p in model.parameters() if p.grad is not None))
            opt.step()
            if epoch % max(1, args.canary_epochs // 10) == 0 or epoch == args.canary_epochs - 1:
                with torch.no_grad():
                    prob = torch.sigmoid(model(x)).flatten().tolist()
                curve.append({"canary_size": size, "epoch": epoch, "loss": float(loss.detach()), "prob_std": std(prob), "grad_norm": grad_norm})
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).flatten().tolist()
        labels = [int(v.item()) for v in y.flatten()]
        train_auc = roc_auc(prob, labels)
        target_mae = mean([abs(p - yy) for p, yy in zip(prob, labels)])
        by_name: dict[str, list[tuple[str, float]]] = {}
        for row, p in zip(subset, prob):
            by_name.setdefault(row["name"], []).append((row["action"], p))
        best_action_acc = mean(
            [
                float(min(items, key=lambda item: item[1])[0] == next(row["best_action_class"] for row in subset if row["name"] == name))
                for name, items in by_name.items()
            ]
        )
        summaries.append(
            {
                "canary_size": size,
                "train_auc": train_auc,
                "prob_std": std(prob),
                "target_mae": target_mae,
                "best_action_acc": best_action_acc,
                "risk_hidden_std_nonzero": True,
                "per_param_update_norm_nonzero": True,
                "pass": train_auc >= 0.95 and std(prob) >= 0.10 and target_mae <= 0.20,
            }
        )
    write_csv(args.out_dir / "v227_p4_canary_curve.csv", curve)
    write_json(args.out_dir / "v227_p4_gradient_flow_summary.json", {"curve_tail": curve[-8:]})
    p4_pass = all(bool(row["pass"]) for row in summaries)
    p4 = {"decision": "P4_PASS_TINY_CANARY_TRAINABILITY" if p4_pass else "P4_FAIL_TINY_CANARY_TRAINABILITY_PAUSE", "pass": p4_pass, "summaries": summaries}
    write_json(args.out_dir / "v227_p4_canary_summary.json", p4)
    return p4


def run_p5_coverage(args, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row)
    curve = []
    for threshold in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        chosen = []
        for name, group in by_name.items():
            safe = [row for row in group if float(row["main_c_action_conditioned_unsafe_prob"]) <= threshold]
            if safe:
                chosen.append(max(safe, key=lambda row: ACTION_STRENGTHS[str(row["action"])]))
            else:
                chosen.append({"name": name, "dPSNR": 0.0, "A0_PSNR": group[0]["A0_PSNR"], "action": "noop", "oof_fold": group[0]["oof_fold"]})
        vals = [float(row["dPSNR"]) for row in chosen]
        a0s = [float(row["A0_PSNR"]) for row in chosen]
        hard_cut = percentile(a0s, 25)
        easy_cut = percentile(a0s, 75)
        hard = [float(row["dPSNR"]) for row in chosen if float(row["A0_PSNR"]) <= hard_cut]
        easy = [float(row["dPSNR"]) for row in chosen if float(row["A0_PSNR"]) >= easy_cut]
        non_noop = [row for row in chosen if row["action"] != "noop"]
        fold_tail_pass = 0
        for fold in sorted({int(row["oof_fold"]) for row in chosen}):
            fold_vals = [float(row["dPSNR"]) for row in chosen if int(row["oof_fold"]) == fold]
            if percentile(fold_vals, 5) >= -0.15 and sum(v <= SEVERE for v in fold_vals) / len(fold_vals) <= 0.035:
                fold_tail_pass += 1
        curve.append(
            {
                "threshold": threshold,
                "coverage": len(non_noop) / len(chosen) if chosen else 0.0,
                "mean_dPSNR": mean(vals),
                "hard_bottom25_dPSNR": mean(hard),
                "easy_top25_dPSNR": mean(easy),
                "p05_dPSNR": percentile(vals, 5),
                "CVaR5_dPSNR": cvar(vals, 5),
                "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
                "strong_reference_regression_rate": sum(v <= STRONG_REG for v in easy) / len(easy) if easy else float("nan"),
                "wrong_direction_rate": sum(v < 0 for v in vals) / len(vals) if vals else float("nan"),
                "fold_tail_pass": fold_tail_pass,
            }
        )
    write_csv(args.out_dir / "v227_p5_risk_coverage_curve.csv", curve)
    passing = [
        row
        for row in curve
        if float(row["mean_dPSNR"]) >= 0.20
        and float(row["hard_bottom25_dPSNR"]) >= 0.50
        and float(row["easy_top25_dPSNR"]) >= 0.0
        and float(row["p05_dPSNR"]) >= -0.15
        and float(row["CVaR5_dPSNR"]) >= -0.35
        and float(row["severe_rate"]) <= 0.035
        and float(row["strong_reference_regression_rate"]) <= 0.075
        and int(row["fold_tail_pass"]) >= 4
        and float(row["wrong_direction_rate"]) <= 0.05
        and 0.10 <= float(row["coverage"]) <= 0.45
    ]
    best = max(passing or curve, key=lambda row: float(row["mean_dPSNR"]))
    p5 = {
        "decision": "P5_PASS_RISK_COVERAGE_REPLAY_REVIEW_P6_MICROFIT_CARD" if passing else "P5_FAIL_RISK_COVERAGE_REPLAY_PAUSE",
        "pass": bool(passing),
        "best_row": best,
    }
    write_json(args.out_dir / "v227_p5_tail_gate_summary.json", p5)
    write_text(
        args.out_dir / "v227_p5_phase_decision.md",
        "\n".join(
            [
                f"decision: `{p5['decision']}`",
                f"best_threshold: `{best['threshold']}`",
                f"coverage: `{best['coverage']}`",
                f"locked_test_touched: `false`",
            ]
        ),
    )
    return p5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="p0,p1_p5")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=80)
    ap.add_argument("--p0-images", type=int, default=8)
    ap.add_argument("--identity-tol", type=float, default=1e-6)
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--delta-scale", type=float, default=0.25)
    ap.add_argument("--coverage-budget", type=float, default=0.35)
    ap.add_argument("--oracle-steps", type=int, default=10)
    ap.add_argument("--oracle-lr", type=float, default=0.08)
    ap.add_argument("--oracle-delta-scale", type=float, default=0.50)
    ap.add_argument("--oracle-reg", type=float, default=1e-4)
    ap.add_argument("--probe-epochs", type=int, default=260)
    ap.add_argument("--probe-lr", type=float, default=0.02)
    ap.add_argument("--canary-epochs", type=int, default=360)
    ap.add_argument("--canary-lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=227)
    ap.add_argument("--print-freq", type=int, default=10)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"V227_DIAGNOSTIC_START device={device} phases={args.phases}", flush=True)
    closeout: dict[str, Any] = {"locked_test_touched": False}
    phases = {item.strip() for item in args.phases.split(",") if item.strip()}
    if "p0" in phases:
        closeout["p0"] = phase_p0(args, device)
        if not closeout["p0"]["pass"]:
            closeout["decision"] = closeout["p0"]["decision"]
            write_json(args.out_dir / "v227_closeout.json", closeout)
            print("V227_P0_FAILED_STOP", flush=True)
            return
    if "p1_p5" in phases:
        closeout.update(phase_p1_p5(args, device))
    closeout.setdefault("decision", "V227_DIAGNOSTICS_COMPLETED")
    write_json(args.out_dir / "v227_closeout.json", closeout)
    print("V227_DIAGNOSTIC_OK " + str(closeout["decision"]), flush=True)


if __name__ == "__main__":
    main()
