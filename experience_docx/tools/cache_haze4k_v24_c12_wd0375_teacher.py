#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim
import torchvision.transforms.functional as TVF


ALPHA = 0.375
IMG_EXT = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def first_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"none of {names} under {root}")


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


def load_convir(convir_dir: Path):
    sys.path.insert(0, str(convir_dir))
    from models.ConvIR import build_net  # type: ignore

    return build_net


def load_a0(convir_dir: Path, checkpoint: Path, device: torch.device):
    build_net = load_convir(convir_dir)
    model = build_net("base", "Haze4K", "original").to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def infer_convir(model, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    out = model(x)
    pred = out[2] if isinstance(out, (list, tuple)) else out
    return torch.clamp(pred[:, :, :h, :w], 0, 1)


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
    arch = load_mod("basicsr.archs.wavemamba_arch", repo / "basicsr/archs/wavemamba_arch.py")
    model = arch.WaveMamba(in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2.0).to(device)
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_names(split_manifest: Path, scope: str) -> list[str]:
    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    if scope == "train_core":
        return list(payload["train_core"])
    if scope == "val":
        return [r["name"] for r in payload["val"]]
    raise ValueError(scope)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--convir-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--scope", choices=["train_core", "val"], default="train_core")
    ap.add_argument("--a0-checkpoint", type=Path, required=True)
    ap.add_argument("--wdmamba-repo", type=Path, required=True)
    ap.add_argument("--wdmamba-checkpoint", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--print-freq", type=int, default=25)
    args = ap.parse_args()

    names = load_names(args.split_manifest, args.scope)
    names = [name for i, name in enumerate(names) if i % args.num_shards == args.shard_index]
    if args.max_images:
        names = names[: args.max_images]

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0 = load_a0(args.convir_dir, args.a0_checkpoint, device)
    wdmamba = load_wdmamba(args.wdmamba_repo, args.wdmamba_checkpoint, device)
    input_dir = first_dir(args.data_dir / "train", ("IN", "haze", "hazy"))
    gt_dir = first_dir(args.data_dir / "train", ("GT", "gt"))

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, name in enumerate(names, 1):
            hazy = TVF.to_tensor(Image.open(input_dir / name).convert("RGB")).unsqueeze(0).to(device)
            label = TVF.to_tensor(Image.open(label_path(gt_dir, name)).convert("RGB")).unsqueeze(0).to(device)
            _, _, h, w = hazy.shape
            x32, hp32, wp32 = pad_to(hazy, 32)
            a0_pred = infer_convir(a0, x32, h, w)
            wd_pred, hp4, wp4 = infer_wdmamba(wdmamba, hazy, h, w)
            teacher = torch.clamp(a0_pred + ALPHA * (wd_pred - a0_pred), 0, 1)
            out_path = args.cache_dir / name
            TVF.to_pil_image((teacher.squeeze(0).cpu() + 0.5 / 255).clamp(0, 1), "RGB").save(out_path)
            a0_psnr, a0_ssim = metric(a0_pred, label, hp32, wp32)
            wd_psnr, wd_ssim = metric(wd_pred, label, hp4, wp4)
            t_psnr, t_ssim = metric(teacher, label, hp32, wp32)
            rows.append({
                "name": name,
                "scope": args.scope,
                "cache_path": str(out_path),
                "A0_PSNR": a0_psnr,
                "A0_SSIM": a0_ssim,
                "WDMamba_PSNR": wd_psnr,
                "WDMamba_SSIM": wd_ssim,
                "WD0375_PSNR": t_psnr,
                "WD0375_SSIM": t_ssim,
                "WD0375_dPSNR": t_psnr - a0_psnr,
                "WD0375_dSSIM": t_ssim - a0_ssim,
            })
            if idx % args.print_freq == 0:
                print(f"C12_TEACHER_CACHE {idx}/{len(names)} shard={args.shard_index}/{args.num_shards}")

    prefix = f"v24_c12_teacher_cache_{args.scope}_shard{args.shard_index:02d}_of{args.num_shards:02d}"
    write_csv(args.out_dir / f"{prefix}_metrics.csv", rows)
    manifest = {
        "route": "Haze4K v2.4 C12 WD0375 distillation",
        "locked_test_touched": False,
        "locked_per_image_read": False,
        "scope": args.scope,
        "count": len(rows),
        "alpha": ALPHA,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "cache_dir": str(args.cache_dir),
        "a0_checkpoint": str(args.a0_checkpoint),
        "a0_sha256": sha256(args.a0_checkpoint),
        "wdmamba_checkpoint": str(args.wdmamba_checkpoint),
        "wdmamba_sha256": sha256(args.wdmamba_checkpoint),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256(args.split_manifest),
    }
    write_json(args.out_dir / f"{prefix}_manifest.json", manifest)
    print("C12_TEACHER_CACHE_OK", json.dumps({"count": len(rows), "shard": args.shard_index}, sort_keys=True))


if __name__ == "__main__":
    main()
