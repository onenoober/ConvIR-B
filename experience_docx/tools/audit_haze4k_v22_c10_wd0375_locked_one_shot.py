#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim
import torchvision.transforms.functional as TVF

import audit_haze4k_v20_c2_outputdiff_router as c2
from eval_udpnet_v15_phase0_repro import load_convir_builders, load_a0_model, infer_one

ALPHA = 0.375
SEVERE = -0.20
AUTHORIZED_DECISION = "C10_FORMAL_5X3_WD0375_PASS_AUTHORIZE_LOCKED_ONE_SHOT_REVIEW"


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pad_to(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(x, (0, pw, 0, ph), "reflect"), h + ph, w + pw


def tensor_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).clamp_min(1e-12)
    return float((10 * torch.log10(1 / mse)).item())


def metric(pred: torch.Tensor, label: torch.Tensor, hp: int, wp: int) -> tuple[float, float]:
    psnr = tensor_psnr(pred, label)
    down = max(1, round(min(hp, wp) / 256))
    ss = ssim(
        F.adaptive_avg_pool2d(pred, (int(hp / down), int(wp / down))),
        F.adaptive_avg_pool2d(label, (int(hp / down), int(wp / down))),
        data_range=1,
        size_average=False,
    ).mean().item()
    return psnr, float(ss)


def first_existing_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"none of {names} exists under {root}")


def label_path(gt_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    ext = Path(image_name).suffix
    candidates = [image_name]
    if "_" in stem:
        candidates.extend([f"{stem.split('_')[0]}{ext}", f"{stem.split('_')[0]}.png"])
    for candidate in candidates:
        p = gt_dir / candidate
        if p.is_file():
            return p
    raise FileNotFoundError(f"no GT for {image_name}")


def load_sample(data_dir: Path, split: str, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    root = data_dir / split
    input_dir = first_existing_dir(root, ("IN", "haze", "hazy"))
    gt_dir = first_existing_dir(root, ("GT", "gt"))
    hazy = Image.open(input_dir / name).convert("RGB")
    gt = Image.open(label_path(gt_dir, name)).convert("RGB")
    return TVF.to_tensor(hazy), TVF.to_tensor(gt)


def list_images(data_dir: Path, split: str) -> list[str]:
    input_dir = first_existing_dir(data_dir / split, ("IN", "haze", "hazy"))
    return sorted(p.name for p in input_dir.iterdir() if p.is_file())


def load_wdmamba(repo: Path, checkpoint: Path, device: torch.device):
    try:
        import transformers.generation as tg
        for name in ["GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"]:
            if not hasattr(tg, name):
                setattr(tg, name, type(name, (object,), {}))
    except Exception:
        pass

    def pkg(name: str, path: Path) -> None:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = mod

    def load_mod(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(path)
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
    model = wavemamba.WaveMamba(in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2.0).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["params"], strict=True)
    model.eval()
    return model


def infer_wdmamba(model, rgb: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, int, int]:
    x, hp, wp = pad_to(rgb, 4)
    out = model.restoration_network(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return torch.clamp(out[:, :, :h, :w], 0, 1), hp, wp


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ds = [fnum(r["dPSNR"]) for r in rows]
    ss = [fnum(r["dSSIM"]) for r in rows]
    a0 = [fnum(r["A0_PSNR"]) for r in rows]
    order = sorted(range(n), key=lambda i: a0[i])
    k = max(1, n // 4)
    hard = [ds[i] for i in order[:k]]
    easy = [ds[i] for i in order[-k:]]
    severe = sum(d <= SEVERE for d in ds)
    return {
        "count": n,
        "mean_dPSNR": statistics.mean(ds),
        "hard_bottom25_dPSNR": statistics.mean(hard),
        "easy_top25_dPSNR": statistics.mean(easy),
        "dSSIM": statistics.mean(ss),
        "positive_ratio": sum(d > 0 for d in ds) / n,
        "nonnegative_ratio": sum(d >= 0 for d in ds) / n,
        "severe_loss_count": severe,
        "severe_loss_per_600": severe / n * 600.0,
    }


def assert_authorized(path: Path) -> None:
    txt = path.read_text(encoding="utf-8")
    if f"Decision: `{AUTHORIZED_DECISION}`" not in txt:
        raise RuntimeError(f"locked one-shot not authorized by {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convir-its-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--data-split", default="test")
    ap.add_argument("--a0-checkpoint", type=Path, required=True)
    ap.add_argument("--wdmamba-repo", type=Path, required=True)
    ap.add_argument("--wdmamba-checkpoint", type=Path, required=True)
    ap.add_argument("--c10-decision", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", default="v22_locked_wd0375_one_shot")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--print-freq", type=int, default=50)
    args = ap.parse_args()

    assert_authorized(args.c10_decision)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, build_convir = load_convir_builders(args.convir_its_dir)
    a0 = load_a0_model(build_convir, args.a0_checkpoint, device)
    wdmamba = load_wdmamba(args.wdmamba_repo, args.wdmamba_checkpoint, device)
    names = list_images(args.data_dir, args.data_split)
    if args.max_images:
        names = names[: args.max_images]

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, name in enumerate(names, 1):
            hazy, gt = load_sample(args.data_dir, args.data_split, name)
            rgb = hazy.unsqueeze(0).to(device)
            lab = gt.unsqueeze(0).to(device)
            _, _, h, w = rgb.shape
            rgb32, hp32, wp32 = pad_to(rgb, 32)
            a0_pred = infer_one(a0, rgb32, h, w)
            a0_psnr, a0_ssim = metric(a0_pred, lab, hp32, wp32)
            expert_pred, hp4, wp4 = infer_wdmamba(wdmamba, rgb, h, w)
            expert_psnr, expert_ssim = metric(expert_pred, lab, hp4, wp4)
            pred = torch.clamp(a0_pred + ALPHA * (expert_pred - a0_pred), 0, 1)
            psnr, ss = metric(pred, lab, hp32, wp32)
            rows.append({
                "name": name,
                "split": args.data_split,
                "A0_PSNR": a0_psnr,
                "A0_SSIM": a0_ssim,
                "WDMamba_PSNR": expert_psnr,
                "WDMamba_SSIM": expert_ssim,
                "WD0375_PSNR": psnr,
                "WD0375_SSIM": ss,
                "dPSNR": psnr - a0_psnr,
                "dSSIM": ss - a0_ssim,
                "alpha": ALPHA,
            })
            if idx % args.print_freq == 0:
                print(f"locked_wd0375 {idx}/{len(names)}", flush=True)

    write_csv(args.out_dir / f"{args.prefix}_per_image.csv", rows)
    summary = {
        "decision": "LOCKED_WD0375_ONE_SHOT_RECORDED",
        "locked_test_touched": True,
        "one_shot": True,
        "no_tuning_from_locked": True,
        "fixed_profile": "WD0375",
        "alpha": ALPHA,
        "data_split": args.data_split,
        "a0_checkpoint": str(args.a0_checkpoint),
        "a0_sha256": sha256(args.a0_checkpoint),
        "wdmamba_checkpoint": str(args.wdmamba_checkpoint),
        "wdmamba_sha256": sha256(args.wdmamba_checkpoint),
        "summary": summarize(rows),
    }
    (args.out_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    s = summary["summary"]
    locked_pass = (
        s["mean_dPSNR"] >= 1.0
        and s["hard_bottom25_dPSNR"] >= 1.0
        and s["easy_top25_dPSNR"] >= 0.0
        and s["positive_ratio"] >= 0.80
        and s["severe_loss_per_600"] <= 60.0
    )
    decision = "LOCKED_WD0375_ONE_SHOT_PASS_REVIEW_DISTILLATION_LATER" if locked_pass else "LOCKED_WD0375_ONE_SHOT_FAIL_NO_TUNING"
    (args.out_dir / f"{args.prefix}_decision.md").write_text(
        "# Locked WD0375 One-Shot Decision\n\n"
        f"Decision: `{decision}`\n\n"
        f"mean/hard/easy/positive/severe: `{s['mean_dPSNR']:.6f}` / "
        f"`{s['hard_bottom25_dPSNR']:.6f}` / `{s['easy_top25_dPSNR']:.6f}` / "
        f"`{s['positive_ratio']:.6f}` / `{s['severe_loss_per_600']:.2f}/600`.\n\n"
        "Locked output is evidence only. It must not tune alpha, features, checkpoints, profiles, actions, experts, or distillation targets.\n",
        encoding="utf-8",
    )
    print(f"LOCKED_WD0375_ONE_SHOT_OK decision={decision}")


if __name__ == "__main__":
    main()
