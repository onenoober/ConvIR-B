#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import statistics
import sys
import time
import types
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim
from torchvision.transforms import functional as TVF

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_udpnet_v15_phase0_repro import infer_one, load_a0_model, load_convir_builders  # noqa: E402


DEFAULT_ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0]
PRIMARY_ALPHA = 0.375
SEVERE_DPSNR = -0.20
BOOT_N = 400


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def alpha_key(alpha: float) -> str:
    return (("a%.6f" % alpha).rstrip("0").rstrip(".")).replace(".", "p")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tensor_psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    mse = F.mse_loss(pred, label).clamp_min(1e-12)
    return float((10 * torch.log10(1.0 / mse)).item())


def metric_pair(pred: torch.Tensor, label: torch.Tensor) -> tuple[float, float]:
    psnr = tensor_psnr(pred, label)
    _, _, h, w = pred.shape
    down = max(1, round(min(h, w) / 256))
    ssim_value = ssim(
        F.adaptive_avg_pool2d(pred, (int(h / down), int(w / down))),
        F.adaptive_avg_pool2d(label, (int(h / down), int(w / down))),
        data_range=1,
        size_average=False,
    ).mean().item()
    return psnr, float(ssim_value)


def pad_to(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(x, (0, pw, 0, ph), "reflect"), h, w


def grad_mean(x: torch.Tensor) -> float:
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return float((gx.abs().mean() + gy.abs().mean()).item() * 0.5)


def qvalue(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.detach().flatten().float().cpu(), q).item())


def wilson_lcb(pos: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = pos / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    rad = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (center - rad) / denom)


def wilson_ucb(pos: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = pos / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    rad = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return min(1.0, (center + rad) / denom)


def bootstrap_lcb(values: list[float], rng: np.random.Generator, q: float = 0.05) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0])
    idx = rng.integers(0, len(arr), size=(BOOT_N, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, q))


def percentile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def qbins(values: list[float]) -> list[str]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return []
    q1, q2, q3 = np.quantile(np.asarray(finite, dtype=np.float64), [0.25, 0.5, 0.75])
    out = []
    for value in values:
        if not math.isfinite(value):
            out.append("nan")
        elif value <= q1:
            out.append("q1")
        elif value <= q2:
            out.append("q2")
        elif value <= q3:
            out.append("q3")
        else:
            out.append("q4")
    return out


def summarize(rows: list[dict[str, Any]], dkey: str = "dPSNR", sskey: str = "dSSIM") -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"count": 0}
    deltas = [fnum(row[dkey]) for row in rows]
    ssim_deltas = [fnum(row.get(sskey, 0.0)) for row in rows]
    a0_psnr = [fnum(row["A0_PSNR"]) for row in rows]
    order = sorted(range(n), key=lambda idx: a0_psnr[idx])
    bucket = max(1, n // 4)
    hard = [deltas[idx] for idx in order[:bucket]]
    easy = [deltas[idx] for idx in order[-bucket:]]
    severe = sum(delta <= SEVERE_DPSNR for delta in deltas)
    positive = sum(delta > 0 for delta in deltas)
    rng = np.random.default_rng(2700 + n)
    return {
        "count": n,
        "A0_mean_PSNR": statistics.mean(a0_psnr),
        "candidate_mean_PSNR": statistics.mean([fnum(row.get("candidate_PSNR", 0.0)) for row in rows])
        if "candidate_PSNR" in rows[0]
        else "",
        "mean_dPSNR": statistics.mean(deltas),
        "mean_bootstrap_LCB": bootstrap_lcb(deltas, rng),
        "median_dPSNR": statistics.median(deltas),
        "p5_dPSNR": percentile(deltas, 0.05),
        "p95_dPSNR": percentile(deltas, 0.95),
        "hard_bottom25_dPSNR": statistics.mean(hard),
        "hard_bootstrap_LCB": bootstrap_lcb(hard, rng),
        "easy_top25_dPSNR": statistics.mean(easy),
        "worst_dPSNR": min(deltas),
        "best_dPSNR": max(deltas),
        "dSSIM": statistics.mean(ssim_deltas),
        "positive_ratio": positive / n,
        "positive_Wilson_LCB": wilson_lcb(positive, n),
        "nonnegative_ratio": sum(delta >= 0 for delta in deltas) / n,
        "severe_loss_count": severe,
        "severe_loss_per_600": severe / n * 600.0,
        "severe_Wilson_UCB": wilson_ucb(severe, n),
    }


def fake_optional_wdmamba_imports() -> None:
    try:
        import transformers.generation as tg

        for name in ["GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"]:
            if not hasattr(tg, name):
                setattr(tg, name, type(name, (object,), {}))
    except Exception:
        pass


def load_wdmamba(args: argparse.Namespace, device: torch.device):
    fake_optional_wdmamba_imports()
    repo = Path(args.wdmamba_repo)

    def pkg(name: str, path: Path) -> None:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = mod

    def load_mod(name: str, file: Path):
        spec = importlib.util.spec_from_file_location(name, file)
        if spec is None or spec.loader is None:
            raise RuntimeError(file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    for key in list(sys.modules):
        if key == "basicsr" or key.startswith("basicsr."):
            del sys.modules[key]
    pkg("basicsr", repo / "basicsr")
    pkg("basicsr.archs", repo / "basicsr/archs")
    pkg("basicsr.utils", repo / "basicsr/utils")
    load_mod("basicsr.utils.registry", repo / "basicsr/utils/registry.py")
    load_mod("basicsr.archs.Ublock", repo / "basicsr/archs/Ublock.py")
    load_mod("basicsr.archs.detail_enhance_net", repo / "basicsr/archs/detail_enhance_net.py")
    load_mod("basicsr.archs.wavelet", repo / "basicsr/archs/wavelet.py")
    wavemamba = load_mod("basicsr.archs.wavemamba_arch", repo / "basicsr/archs/wavemamba_arch.py")
    with redirect_stdout(io.StringIO()):
        model = wavemamba.WaveMamba(in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2.0).to(device)
    ckpt = torch.load(args.wdmamba_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["params"], strict=True)
    model.eval()
    return model


def infer_wdmamba(model, rgb: torch.Tensor) -> torch.Tensor:
    padded, h, w = pad_to(rgb, 4)
    out = model.restoration_network(padded)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return torch.clamp(out[:, :, :h, :w], 0, 1)


def list_pairs(data_dir: Path) -> list[dict[str, str]]:
    hazy_files = sorted(data_dir.glob("*_hazy.png"))
    pairs: list[dict[str, str]] = []
    missing = []
    for hazy in hazy_files:
        image_id = hazy.name[: -len("_hazy.png")]
        gt = data_dir / f"{image_id}_GT.png"
        if not gt.is_file():
            missing.append({"hazy": hazy.name, "expected_gt": gt.name})
            continue
        pairs.append({"image_id": image_id, "hazy": str(hazy), "gt": str(gt)})
    if missing:
        raise FileNotFoundError(f"missing GT files: {missing[:5]}")
    if not pairs:
        raise FileNotFoundError(f"no *_hazy.png pairs under {data_dir}")
    return pairs


def load_rgb(path: Path) -> torch.Tensor:
    return TVF.to_tensor(Image.open(path).convert("RGB")).float()


def add_features(row: dict[str, Any], rgb: torch.Tensor, a0_pred: torch.Tensor, wd_pred: torch.Tensor) -> None:
    residual = wd_pred - a0_pred
    brightness = rgb.mean(dim=1, keepdim=True)
    saturation = rgb.max(dim=1, keepdim=True).values - rgb.min(dim=1, keepdim=True).values
    dark = rgb.min(dim=1, keepdim=True).values
    row["input_mean"] = float(rgb.mean().item())
    row["input_std"] = float(rgb.std(unbiased=False).item())
    row["input_grad_mean"] = grad_mean(rgb)
    row["dark_channel_mean"] = float(dark.mean().item())
    row["airlight_proxy_p99"] = qvalue(brightness, 0.99)
    row["sky_highlight_proxy"] = float(((brightness > 0.78) & (saturation < 0.18)).float().mean().item())
    row["a0_mean"] = float(a0_pred.mean().item())
    row["a0_std"] = float(a0_pred.std(unbiased=False).item())
    row["wdmamba_mean"] = float(wd_pred.mean().item())
    row["wdmamba_std"] = float(wd_pred.std(unbiased=False).item())
    row["wdmamba_residual_signed_mean"] = float(residual.mean().item())
    row["wdmamba_residual_abs_mean"] = float(residual.abs().mean().item())
    row["wdmamba_residual_abs_p90"] = qvalue(residual.abs(), 0.90)
    row["wdmamba_residual_grad_mean"] = grad_mean(residual)
    row["a0_wdmamba_psnr"] = tensor_psnr(a0_pred, wd_pred)


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list_pairs(Path(args.data_dir))
    selected = [pair for idx, pair in enumerate(pairs) if idx % args.shard_count == args.shard_index]
    if args.max_images > 0:
        selected = selected[: args.max_images]
    if not selected:
        raise ValueError(f"empty shard {args.shard_index}/{args.shard_count}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, build_convir = load_convir_builders(Path(args.convir_its_dir))
    a0 = load_a0_model(build_convir, Path(args.a0_checkpoint), device)
    wdmamba = load_wdmamba(args, device)
    alphas = sorted({round(float(alpha), 6) for alpha in args.alphas})
    rows: list[dict[str, Any]] = []
    start_all = time.time()

    with torch.inference_mode():
        for local_idx, pair in enumerate(selected, 1):
            image_start = time.time()
            rgb_cpu = load_rgb(Path(pair["hazy"]))
            gt_cpu = load_rgb(Path(pair["gt"]))
            if rgb_cpu.shape != gt_cpu.shape:
                raise ValueError(f"shape mismatch for {pair['image_id']}: {tuple(rgb_cpu.shape)} vs {tuple(gt_cpu.shape)}")
            rgb = rgb_cpu.unsqueeze(0).to(device)
            gt = gt_cpu.unsqueeze(0).to(device)
            _, _, h, w = rgb.shape

            a0_input, _, _ = pad_to(rgb, 32)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            a0_start = time.time()
            a0_pred = infer_one(a0, a0_input, h, w)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            a0_time = time.time() - a0_start

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wd_start = time.time()
            wd_pred = infer_wdmamba(wdmamba, rgb)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wd_time = time.time() - wd_start

            input_psnr, input_ssim = metric_pair(rgb, gt)
            a0_psnr, a0_ssim = metric_pair(a0_pred, gt)
            wd_psnr, wd_ssim = metric_pair(wd_pred, gt)
            row: dict[str, Any] = {
                "image_id": pair["image_id"],
                "hazy_file": Path(pair["hazy"]).name,
                "gt_file": Path(pair["gt"]).name,
                "width": w,
                "height": h,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "input_PSNR": input_psnr,
                "input_SSIM": input_ssim,
                "A0_PSNR": a0_psnr,
                "A0_SSIM": a0_ssim,
                "WDMamba_PSNR": wd_psnr,
                "WDMamba_SSIM": wd_ssim,
                "WDMamba_dPSNR": wd_psnr - a0_psnr,
                "WDMamba_dSSIM": wd_ssim - a0_ssim,
                "A0_time_sec": a0_time,
                "WDMamba_time_sec": wd_time,
                "image_time_sec": time.time() - image_start,
            }
            add_features(row, rgb, a0_pred, wd_pred)
            for alpha in alphas:
                key = alpha_key(alpha)
                pred = torch.clamp(a0_pred + alpha * (wd_pred - a0_pred), 0, 1)
                psnr, ssim_value = metric_pair(pred, gt)
                row[f"alpha_{key}_PSNR"] = psnr
                row[f"alpha_{key}_SSIM"] = ssim_value
                row[f"alpha_{key}_dPSNR"] = psnr - a0_psnr
                row[f"alpha_{key}_dSSIM"] = ssim_value - a0_ssim
            rows.append(row)
            if local_idx % args.print_freq == 0 or local_idx == len(selected):
                mean_0375 = statistics.mean(fnum(r[f"alpha_{alpha_key(PRIMARY_ALPHA)}_dPSNR"]) for r in rows)
                print(
                    f"progress shard={args.shard_index}/{args.shard_count} "
                    f"{local_idx}/{len(selected)} mean_wd0375={mean_0375:.6f} "
                    f"elapsed={time.time() - start_all:.1f}s",
                    flush=True,
                )
            del rgb, gt, a0_input, a0_pred, wd_pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    per_image_path = out_dir / f"{args.prefix}_per_image.csv"
    write_csv(per_image_path, rows)
    manifest = {
        "route": "haze4k_v2_7_nhhaze_transfer_20260616",
        "mode": "evaluate_shard",
        "prefix": args.prefix,
        "data_dir": str(Path(args.data_dir)),
        "pair_count_total": len(pairs),
        "pair_count_shard": len(rows),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "alpha_grid": alphas,
        "primary_alpha": PRIMARY_ALPHA,
        "a0_checkpoint": str(args.a0_checkpoint),
        "a0_sha256": sha256_file(Path(args.a0_checkpoint)),
        "wdmamba_checkpoint": str(args.wdmamba_checkpoint),
        "wdmamba_sha256": sha256_file(Path(args.wdmamba_checkpoint)),
        "wdmamba_repo": str(args.wdmamba_repo),
        "locked_haze4k_touched": False,
        "nhhaze_tuning": False,
        "elapsed_sec": time.time() - start_all,
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"V27_NHHAZE_EVAL_SHARD_OK shard={args.shard_index} rows={len(rows)}")
    return rows


def group_metrics(rows: list[dict[str, Any]], alphas: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = [
        ("A0-PSNR_q4", "A0_PSNR"),
        ("input-PSNR_q4", "input_PSNR"),
        ("WDMamba-endpoint-dPSNR_q4", "WDMamba_dPSNR"),
        ("WDMamba-residual-abs_q4", "wdmamba_residual_abs_mean"),
        ("WDMamba-residual-signed_q4", "wdmamba_residual_signed_mean"),
        ("A0-WDMamba-PSNR_q4", "a0_wdmamba_psnr"),
        ("dark-channel_q4", "dark_channel_mean"),
        ("airlight-proxy_q4", "airlight_proxy_p99"),
        ("input-grad_q4", "input_grad_mean"),
        ("sky-highlight_q4", "sky_highlight_proxy"),
    ]
    metrics: list[dict[str, Any]] = []
    for label, key in specs:
        values = [fnum(row.get(key), float("nan")) for row in rows]
        bins = qbins(values)
        for bin_name in ["q1", "q2", "q3", "q4"]:
            idxs = [idx for idx, b in enumerate(bins) if b == bin_name]
            if not idxs:
                continue
            for alpha in alphas:
                akey = alpha_key(alpha)
                sub = [
                    {
                        **rows[idx],
                        "dPSNR": rows[idx][f"alpha_{akey}_dPSNR"],
                        "dSSIM": rows[idx][f"alpha_{akey}_dSSIM"],
                    }
                    for idx in idxs
                ]
                metrics.append({"alpha": alpha, "group": label, "bin": bin_name, **summarize(sub)})

    mins: list[dict[str, Any]] = []
    for alpha in alphas:
        sub = [row for row in metrics if fnum(row["alpha"]) == alpha]
        if not sub:
            continue
        mean_min = min(sub, key=lambda row: fnum(row["mean_dPSNR"]))
        hard_min = min(sub, key=lambda row: fnum(row["hard_bottom25_dPSNR"]))
        easy_min = min(sub, key=lambda row: fnum(row["easy_top25_dPSNR"]))
        positive_min = min(sub, key=lambda row: fnum(row["positive_ratio"]))
        severe_max = max(sub, key=lambda row: fnum(row["severe_loss_per_600"]))
        mins.append(
            {
                "alpha": alpha,
                "group_count": len(sub),
                "min_group_mean_dPSNR": fnum(mean_min["mean_dPSNR"]),
                "min_group_mean_group": mean_min["group"],
                "min_group_mean_bin": mean_min["bin"],
                "min_group_hard_bottom25_dPSNR": fnum(hard_min["hard_bottom25_dPSNR"]),
                "min_group_hard_group": hard_min["group"],
                "min_group_hard_bin": hard_min["bin"],
                "min_group_easy_top25_dPSNR": fnum(easy_min["easy_top25_dPSNR"]),
                "min_group_easy_group": easy_min["group"],
                "min_group_easy_bin": easy_min["bin"],
                "min_group_positive_ratio": fnum(positive_min["positive_ratio"]),
                "min_group_positive_group": positive_min["group"],
                "min_group_positive_bin": positive_min["bin"],
                "max_group_severe_per_600": fnum(severe_max["severe_loss_per_600"]),
                "max_group_severe_group": severe_max["group"],
                "max_group_severe_bin": severe_max["bin"],
            }
        )
    return metrics, mins


def aggregate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [Path(path) for path in args.input_csvs]
    rows: list[dict[str, Any]] = []
    for path in input_paths:
        rows.extend(read_csv(path))
    rows = sorted(rows, key=lambda row: row["image_id"])
    if not rows:
        raise ValueError("no per-image rows to aggregate")
    alphas = sorted({round(float(alpha), 6) for alpha in args.alphas})
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate image_id detected across shards")

    per_image = out_dir / f"{args.prefix}_per_image.csv"
    write_csv(per_image, rows)
    alpha_rows: list[dict[str, Any]] = []
    for alpha in alphas:
        key = alpha_key(alpha)
        sub = [
            {
                **row,
                "candidate_PSNR": row[f"alpha_{key}_PSNR"],
                "dPSNR": row[f"alpha_{key}_dPSNR"],
                "dSSIM": row[f"alpha_{key}_dSSIM"],
            }
            for row in rows
        ]
        alpha_rows.append({"alpha": alpha, "role": "fixed_from_haze4k" if alpha == PRIMARY_ALPHA else "diagnostic_grid", **summarize(sub)})
    write_csv(out_dir / f"{args.prefix}_alpha_grid.csv", alpha_rows)
    gm, gmin = group_metrics(rows, alphas)
    write_csv(out_dir / f"{args.prefix}_group_metrics.csv", gm)
    write_csv(out_dir / f"{args.prefix}_group_min.csv", gmin)

    primary = next(row for row in alpha_rows if fnum(row["alpha"]) == PRIMARY_ALPHA)
    endpoint = next(row for row in alpha_rows if fnum(row["alpha"]) == 1.0)
    a0 = next(row for row in alpha_rows if fnum(row["alpha"]) == 0.0)
    safe_alpha_rows = [
        row
        for row in alpha_rows
        if fnum(row["mean_dPSNR"]) > 0
        and fnum(row["hard_bottom25_dPSNR"]) > 0
        and fnum(row["easy_top25_dPSNR"]) > 0
        and fnum(row["dSSIM"]) >= 0
        and fnum(row["positive_ratio"]) >= 0.70
    ]
    primary_positive = (
        fnum(primary["mean_dPSNR"]) > 0
        and fnum(primary["hard_bottom25_dPSNR"]) > 0
        and fnum(primary["easy_top25_dPSNR"]) > 0
        and fnum(primary["dSSIM"]) >= 0
        and fnum(primary["positive_ratio"]) >= 0.70
    )
    primary_tail_safer_than_full = (
        fnum(primary["severe_loss_count"]) <= fnum(endpoint["severe_loss_count"])
        and fnum(primary["worst_dPSNR"]) >= fnum(endpoint["worst_dPSNR"])
    )
    if primary_positive and primary_tail_safer_than_full:
        decision = "V27_NHHAZE_FIXED_WD0375_TRANSFER_SUPPORTS_SHRINKAGE"
    elif primary_positive:
        decision = "V27_NHHAZE_FIXED_WD0375_POSITIVE_TAIL_MIXED"
    else:
        decision = "V27_NHHAZE_FIXED_WD0375_TRANSFER_NOT_SUPPORTED"

    summary = {
        "route": "haze4k_v2_7_nhhaze_transfer_20260616",
        "decision": decision,
        "state_label": "COMPLETED_GATE_PASS" if primary_positive else "COMPLETED_GATE_FAIL",
        "locked_haze4k_touched": False,
        "nhhaze_tuning": False,
        "dataset": "NH-HAZE",
        "dataset_root": str(args.data_dir),
        "count": len(rows),
        "alpha_grid": alphas,
        "primary_fixed_alpha": PRIMARY_ALPHA,
        "primary_alpha_summary": primary,
        "endpoint_alpha_summary": endpoint,
        "alpha0_summary": a0,
        "safe_positive_alpha_set_diagnostic": [fnum(row["alpha"]) for row in safe_alpha_rows],
        "primary_tail_safer_than_full": primary_tail_safer_than_full,
        "primary_positive_transfer": primary_positive,
        "input_files": [str(path) for path in input_paths],
    }
    (out_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    compact_rows = []
    for row in alpha_rows:
        compact_rows.append(
            {
                "alpha": row["alpha"],
                "role": row["role"],
                "mean_dPSNR": row["mean_dPSNR"],
                "hard_bottom25_dPSNR": row["hard_bottom25_dPSNR"],
                "easy_top25_dPSNR": row["easy_top25_dPSNR"],
                "dSSIM": row["dSSIM"],
                "positive_ratio": row["positive_ratio"],
                "nonnegative_ratio": row["nonnegative_ratio"],
                "severe_loss_count": row["severe_loss_count"],
                "severe_loss_per_600": row["severe_loss_per_600"],
                "worst_dPSNR": row["worst_dPSNR"],
            }
        )
    write_csv(out_dir / f"{args.prefix}_compact_alpha_comparison.csv", compact_rows)

    decision_md = (
        "# Haze4K v2.7 NH-HAZE Fixed WDMamba Transfer Decision\n\n"
        f"Decision: `{decision}`\n\n"
        "This route evaluates NH-HAZE as a cross-dataset fixed-transfer diagnostic. "
        "The primary row is the Haze4K-selected fixed `WD0375 = A0 + 0.375 * (WDMamba - A0)` profile. "
        "Other alpha rows are reported only as a predeclared diagnostic curve and are not used to tune NH-HAZE.\n\n"
        "## Primary Fixed Row\n\n"
        f"- count: `{len(rows)}`\n"
        f"- alpha: `{PRIMARY_ALPHA}`\n"
        f"- mean/hard/easy dPSNR: `{fnum(primary['mean_dPSNR']):+.6f}` / "
        f"`{fnum(primary['hard_bottom25_dPSNR']):+.6f}` / `{fnum(primary['easy_top25_dPSNR']):+.6f}`\n"
        f"- dSSIM: `{fnum(primary['dSSIM']):+.8f}`\n"
        f"- positive/nonnegative: `{fnum(primary['positive_ratio']):.6f}` / `{fnum(primary['nonnegative_ratio']):.6f}`\n"
        f"- severe: `{int(fnum(primary['severe_loss_count']))}/{len(rows)}` "
        f"(`{fnum(primary['severe_loss_per_600']):.2f}/600`)\n"
        f"- worst dPSNR: `{fnum(primary['worst_dPSNR']):+.6f}`\n\n"
        "## Full Expert Endpoint\n\n"
        f"- alpha `1.0` mean/hard/easy dPSNR: `{fnum(endpoint['mean_dPSNR']):+.6f}` / "
        f"`{fnum(endpoint['hard_bottom25_dPSNR']):+.6f}` / `{fnum(endpoint['easy_top25_dPSNR']):+.6f}`\n"
        f"- alpha `1.0` severe: `{int(fnum(endpoint['severe_loss_count']))}/{len(rows)}` "
        f"(`{fnum(endpoint['severe_loss_per_600']):.2f}/600`), worst `{fnum(endpoint['worst_dPSNR']):+.6f}`\n\n"
        "## Protocol Notes\n\n"
        "- Haze4K locked test touched: `false`.\n"
        "- NH-HAZE alpha tuning: `false`.\n"
        "- NH-HAZE has 55 paired full-resolution PNG images, each 1600x1200.\n"
        "- Metrics are PSNR/SSIM against NH-HAZE GT; hard/easy buckets are bottom/top quartiles by A0 PSNR.\n"
    )
    (out_dir / "v27_decision.md").write_text(decision_md, encoding="utf-8")

    readme = (
        "# Haze4K v2.7 NH-HAZE Transfer Evidence\n\n"
        "- Route card: `experience_docx/experiment_cards/2026-06-16-haze4k-v2-7-nhhaze-transfer.md`\n"
        "- Central index: `experience_docx/EXPERIMENT_INDEX.md`\n"
        f"- Decision: `{decision}`\n"
        "- Runtime host: `convir-4090`\n"
        "- Dataset: `/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE/`\n"
        "- Primary fixed profile: `WD0375 = A0 + 0.375 * (WDMamba - A0)`\n"
        "- Haze4K locked test touched: `false`\n"
        "- NH-HAZE tuning: `false`\n\n"
        "Primary files:\n\n"
        f"- `{args.prefix}_summary.json`\n"
        f"- `{args.prefix}_alpha_grid.csv`\n"
        f"- `{args.prefix}_compact_alpha_comparison.csv`\n"
        f"- `{args.prefix}_per_image.csv`\n"
        f"- `{args.prefix}_group_metrics.csv`\n"
        f"- `{args.prefix}_group_min.csv`\n"
        "- `v27_decision.md`\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"V27_NHHAZE_AGGREGATE_OK decision={decision} rows={len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evaluate", "aggregate", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="v27_nhhaze_wdmamba_transfer")
    parser.add_argument("--convir-its-dir", type=Path, default=Path("Dehazing/ITS"))
    parser.add_argument("--a0-checkpoint", type=Path, required=False)
    parser.add_argument("--wdmamba-checkpoint", type=Path, required=False)
    parser.add_argument("--wdmamba-repo", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba"))
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--print-freq", type=int, default=5)
    parser.add_argument("--input-csvs", nargs="*", default=[])
    args = parser.parse_args()
    args.alphas = sorted({round(float(alpha), 6) for alpha in args.alphas})
    if args.mode in {"evaluate", "all"}:
        if args.a0_checkpoint is None:
            raise ValueError("--a0-checkpoint is required for evaluate/all")
        if args.wdmamba_checkpoint is None:
            raise ValueError("--wdmamba-checkpoint is required for evaluate/all")
        if args.shard_count <= 0:
            raise ValueError("--shard-count must be positive")
        if not (0 <= args.shard_index < args.shard_count):
            raise ValueError("--shard-index must be in [0, shard_count)")
    if args.mode == "aggregate" and not args.input_csvs:
        raise ValueError("--input-csvs is required for aggregate")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "aggregate":
        aggregate(args)
    else:
        evaluate(args)
        args.input_csvs = [str(Path(args.out_dir) / f"{args.prefix}_per_image.csv")]
        aggregate(args)


if __name__ == "__main__":
    main()
