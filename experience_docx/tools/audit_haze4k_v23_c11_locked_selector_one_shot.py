#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
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
from eval_udpnet_v15_phase0_repro import load_convir_builders, load_a0_model, load_udpnet_model, infer_one

SEVERE = -0.20
AUTHORIZED_DECISION = "C11E_SEALED_SELECTOR_PASS_READY_FOR_LOCKED_ONE_SHOT_REVIEW"


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


def load_c11_module(repo_root: Path):
    path = repo_root / "experience_docx/tools/analyze_haze4k_v23_c11_wd_fs_selector.py"
    spec = importlib.util.spec_from_file_location("c11_selector_analysis_locked", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def decode_model(payload: str) -> dict[str, Any]:
    return pickle.loads(base64.b64decode(payload.encode("ascii")))


def pad_to(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int, int, int]:
    _, _, h, w = x.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(x, (0, pw, 0, ph), "reflect"), h, w, h + ph, w + pw


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


def list_images(data_dir: Path, split: str) -> list[str]:
    input_dir = first_existing_dir(data_dir / split, ("IN", "haze", "hazy"))
    return sorted(p.name for p in input_dir.iterdir() if p.is_file())


def depth_path(depth_cache: Path, split: str, image_name: str) -> Path:
    candidates = [
        depth_cache / split / f"{image_name.replace('/', '__')}.npy",
        depth_cache / split / f"{image_name}.npy",
        depth_cache / f"{image_name.replace('/', '__')}.npy",
        depth_cache / f"{image_name}.npy",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing depth cache for {image_name}")


def normalize_depth_minmax(depth: np.ndarray) -> np.ndarray:
    lo = float(np.nanmin(depth))
    hi = float(np.nanmax(depth))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - lo) / (hi - lo + 1e-6)).astype(np.float32)


def load_sample(data_dir: Path, depth_cache: Path, split: str, image_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    root = data_dir / split
    input_dir = first_existing_dir(root, ("IN", "haze", "hazy"))
    gt_dir = first_existing_dir(root, ("GT", "gt"))
    trans_dir = root / "trans"
    image = Image.open(input_dir / image_name).convert("RGB")
    label = Image.open(label_path(gt_dir, image_name)).convert("RGB")
    depth_arr = np.load(depth_path(depth_cache, split, image_name)).astype(np.float32)
    depth_arr = np.nan_to_num(depth_arr, nan=0.0, posinf=0.0, neginf=0.0)
    if depth_arr.ndim == 3:
        depth_arr = np.squeeze(depth_arr)
    depth_arr = normalize_depth_minmax(depth_arr)
    depth_img = Image.fromarray(depth_arr, mode="F")
    if depth_img.size != image.size:
        depth_img = depth_img.resize(image.size, resample=Image.BICUBIC)
    trans = load_transmission(trans_dir, image_name, image.size) if trans_dir.is_dir() else None
    return TVF.to_tensor(image), TVF.to_tensor(label), TVF.to_tensor(depth_img).float(), trans


def load_transmission(trans_dir: Path, name: str, size: tuple[int, int]) -> torch.Tensor | None:
    stem = Path(name).stem
    clean_id = stem.split("_", 1)[0]
    candidates = [trans_dir / name, trans_dir / f"{stem}.png", trans_dir / f"{clean_id}.png"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        matches = list(trans_dir.glob(clean_id + ".*"))
        path = matches[0] if matches else None
    if path is None or not path.is_file():
        return None
    img = Image.open(path).convert("L")
    if img.size != size:
        img = img.resize(size, resample=Image.BICUBIC)
    return TVF.to_tensor(img).float()


def load_udp_builder_file(udp_repo: Path, filename: str):
    models_dir = udp_repo / "Dehazing/ITS/models"
    model_file = models_dir / filename
    package_name = "udpnet_c11_locked_models_" + Path(filename).stem.lower()
    package = types.ModuleType(package_name)
    package.__path__ = [str(models_dir)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    module_name = f"{package_name}.{Path(filename).stem}"
    if filename == "FSNet_UDPNet.py":
        src = model_file.read_text(encoding="utf-8").replace("num_heads=1", "num_heads=2")
        mod = types.ModuleType(module_name)
        mod.__file__ = str(model_file)
        mod.__package__ = package_name
        sys.modules[module_name] = mod
        exec(compile(src, str(model_file) + "#c11_fsudp_heads2", "exec"), mod.__dict__)
        return mod.build_net
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(model_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.build_net


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
    x, _, _, hp, wp = pad_to(rgb, 4)
    out = model.restoration_network(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return torch.clamp(out[:, :, :h, :w], 0, 1), hp, wp


def infer_fsudp(model, rgb: torch.Tensor, depth: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, int, int]:
    x, _, _, hp, wp = pad_to(torch.cat([rgb, depth], 1), 32)
    return infer_one(model, x, h, w), hp, wp


def grad_mean(x: torch.Tensor) -> float:
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return float((gx.abs().mean() + gy.abs().mean()).item() * 0.5)


def qvalue(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.detach().flatten().float().cpu(), q).item())


def add_residual_features(row: dict[str, Any], prefix: str, pred: torch.Tensor, a0_pred: torch.Tensor, full_pred: torch.Tensor) -> None:
    full_res = full_pred - a0_pred
    exp_res = pred - a0_pred
    row[f"{prefix}_residual_signed_mean"] = float(exp_res.mean().item())
    row[f"{prefix}_residual_abs_mean"] = float(exp_res.abs().mean().item())
    row[f"{prefix}_residual_grad_mean"] = grad_mean(exp_res)
    row[f"{prefix}_fulludp_mae"] = float((pred - full_pred).abs().mean().item())
    denom = (full_res.flatten().norm() * exp_res.flatten().norm()).clamp_min(1e-12)
    row[f"{prefix}_residual_cosine"] = float(torch.dot(full_res.flatten(), exp_res.flatten()).div(denom).item())


def add_base_features(row: dict[str, Any], rgb: torch.Tensor, depth: torch.Tensor, trans: torch.Tensor | None, a0_pred: torch.Tensor, full_pred: torch.Tensor) -> None:
    for k, v in c2.feature_dict(rgb, depth, a0_pred, full_pred).items():
        row[f"feature_{k}"] = v
    row["input_edge_density"] = grad_mean(rgb)
    row["input_low_texture_proxy"] = -float(rgb.std().item())
    dark = rgb.min(dim=1, keepdim=True).values
    row["dark_channel_mean"] = float(dark.mean().item())
    brightness = rgb.mean(dim=1, keepdim=True)
    sat = rgb.max(dim=1, keepdim=True).values - rgb.min(dim=1, keepdim=True).values
    row["sky_highlight_proxy"] = float(((brightness > 0.78) & (sat < 0.18)).float().mean().item())
    row["airlight_proxy_p99"] = qvalue(brightness, 0.99)
    if trans is not None:
        t = trans.unsqueeze(0).to(rgb.device) if trans.ndim == 3 else trans.to(rgb.device)
        row["transmission_mean"] = float(t.mean().item())
        row["transmission_std"] = float(t.std().item())
        row["haze_density_mean"] = float((1.0 - t).mean().item())
        row["haze_density_p90"] = qvalue(1.0 - t, 0.90)
    else:
        for k in ["transmission_mean", "transmission_std", "haze_density_mean", "haze_density_p90"]:
            row[k] = float("nan")


def summarize(rows: list[dict[str, Any]], dkey: str = "dPSNR", sskey: str = "dSSIM") -> dict[str, Any]:
    n = len(rows)
    ds = [fnum(r[dkey]) for r in rows]
    ss = [fnum(r.get(sskey, 0.0)) for r in rows]
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


def assert_authorized(selector_path: Path, decision_path: Path) -> dict[str, Any]:
    sealed = json.loads(selector_path.read_text(encoding="utf-8"))
    if sealed.get("stage") != "C11-E sealed train-derived selector":
        raise RuntimeError("selector is not a C11-E sealed selector")
    txt = decision_path.read_text(encoding="utf-8")
    if f"Decision: `{AUTHORIZED_DECISION}`" not in txt:
        raise RuntimeError(f"locked C11 selector not authorized by {decision_path}")
    if sealed.get("locked_test_touched") is not False or sealed.get("locked_per_image_read") is not False:
        raise RuntimeError("sealed selector provenance is not train-derived-only")
    return sealed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--convir-its-dir", type=Path, required=True)
    ap.add_argument("--udp-repo", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--data-split", default="test")
    ap.add_argument("--depth-cache-dir", type=Path, required=True)
    ap.add_argument("--a0-checkpoint", type=Path, required=True)
    ap.add_argument("--fulludp-checkpoint", type=Path, required=True)
    ap.add_argument("--fsudp-checkpoint", type=Path, required=True)
    ap.add_argument("--wdmamba-repo", type=Path, required=True)
    ap.add_argument("--wdmamba-checkpoint", type=Path, required=True)
    ap.add_argument("--sealed-selector", type=Path, required=True)
    ap.add_argument("--sealed-decision", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", default="v23_locked_c11_selector_one_shot")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--print-freq", type=int, default=25)
    args = ap.parse_args()

    sealed = assert_authorized(args.sealed_selector, args.sealed_decision)
    c11 = load_c11_module(args.repo_root)
    selector_model = pickle.loads(base64.b64decode(sealed["model_pickle_b64"].encode("ascii")))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, build_convir = load_convir_builders(args.convir_its_dir)
    a0 = load_a0_model(build_convir, args.a0_checkpoint, device)
    fulludp = load_udpnet_model(load_udp_builder_file(args.udp_repo, "ConvIR_UDPNet.py"), args.fulludp_checkpoint, device)[0]
    fsudp = load_udpnet_model(load_udp_builder_file(args.udp_repo, "FSNet_UDPNet.py"), args.fsudp_checkpoint, device)[0]
    wdmamba = load_wdmamba(args.wdmamba_repo, args.wdmamba_checkpoint, device)

    names = list_images(args.data_dir, args.data_split)
    if args.max_images:
        names = names[: args.max_images]

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, name in enumerate(names, 1):
            hazy, gt, depth, trans = load_sample(args.data_dir, args.depth_cache_dir, args.data_split, name)
            rgb = hazy.unsqueeze(0).to(device)
            lab = gt.unsqueeze(0).to(device)
            dep = depth.unsqueeze(0).to(device)
            _, _, h, w = rgb.shape
            rgb32, _, _, hp32, wp32 = pad_to(rgb, 32)
            dep32 = F.interpolate(dep, size=(hp32, wp32), mode="bicubic", align_corners=False)
            a0_pred = infer_one(a0, rgb32, h, w)
            a0_psnr, a0_ssim = metric(a0_pred, lab, hp32, wp32)
            full_pred = infer_one(fulludp, torch.cat([rgb32, dep32], 1), h, w)
            full_psnr, full_ssim = metric(full_pred, lab, hp32, wp32)
            wd_pred, hp4, wp4 = infer_wdmamba(wdmamba, rgb, h, w)
            fs_pred, hpfs, wpfs = infer_fsudp(fsudp, rgb, dep, h, w)
            wd0375 = torch.clamp(a0_pred + 0.375 * (wd_pred - a0_pred), 0, 1)
            fs050 = torch.clamp(a0_pred + 0.5 * (fs_pred - a0_pred), 0, 1)
            wd_psnr, wd_ssim = metric(wd0375, lab, hp32, wp32)
            fs_psnr, fs_ssim = metric(fs050, lab, hpfs, wpfs)
            rec: dict[str, Any] = {
                "name": name,
                "split": args.data_split,
                "A0_PSNR": a0_psnr,
                "A0_SSIM": a0_ssim,
                "FullUDP_PSNR": full_psnr,
                "FullUDP_SSIM": full_ssim,
                "fulludp_dPSNR": full_psnr - a0_psnr,
                "fulludp_dSSIM": full_ssim - a0_ssim,
                "WD0375_PSNR": wd_psnr,
                "WD0375_SSIM": wd_ssim,
                "WD0375_dPSNR": wd_psnr - a0_psnr,
                "WD0375_dSSIM": wd_ssim - a0_ssim,
                "FS050_PSNR": fs_psnr,
                "FS050_SSIM": fs_ssim,
                "FS050_dPSNR": fs_psnr - a0_psnr,
                "FS050_dSSIM": fs_ssim - a0_ssim,
            }
            add_base_features(rec, rgb, dep, trans, a0_pred, full_pred)
            add_residual_features(rec, "wdmamba", wd_pred, a0_pred, full_pred)
            add_residual_features(rec, "fsudp", fs_pred, a0_pred, full_pred)
            wd_signed = 0.375 * fnum(rec["wdmamba_residual_signed_mean"])
            fs_signed = 0.5 * fnum(rec["fsudp_residual_signed_mean"])
            wd_abs = 0.375 * fnum(rec["wdmamba_residual_abs_mean"])
            fs_abs = 0.5 * fnum(rec["fsudp_residual_abs_mean"])
            rec["wd0375_fs050_signed_gap_proxy"] = wd_signed - fs_signed
            rec["wd0375_fs050_abs_gap_proxy"] = wd_abs - fs_abs
            rec["wd0375_fs050_disagreement_proxy"] = abs(wd_signed - fs_signed) + abs(wd_abs - fs_abs)
            action = c11.predict_model(selector_model, [rec])[0]
            d, s = c11.action_value(rec, action)
            rec["selected_action"] = action
            rec["dPSNR"] = d
            rec["dSSIM"] = s
            rec["selected_PSNR"] = a0_psnr + d
            rec["selected_SSIM"] = a0_ssim + s
            rows.append(rec)
            if idx % args.print_freq == 0:
                print(f"locked_c11_selector {idx}/{len(names)} action={action}", flush=True)

    write_csv(args.out_dir / f"{args.prefix}_per_image.csv", rows)
    summary = {
        "decision": "LOCKED_C11_SELECTOR_ONE_SHOT_RECORDED",
        "locked_test_touched": True,
        "one_shot": True,
        "no_tuning_from_locked": True,
        "sealed_selector": str(args.sealed_selector),
        "sealed_selector_sha256": sha256(args.sealed_selector),
        "sealed_config": sealed["selected_config"],
        "a0_checkpoint": str(args.a0_checkpoint),
        "a0_sha256": sha256(args.a0_checkpoint),
        "wdmamba_checkpoint": str(args.wdmamba_checkpoint),
        "wdmamba_sha256": sha256(args.wdmamba_checkpoint),
        "fsudp_checkpoint": str(args.fsudp_checkpoint),
        "fsudp_sha256": sha256(args.fsudp_checkpoint),
        "summary": summarize(rows),
        "wd0375_summary": summarize([{**r, "dPSNR": r["WD0375_dPSNR"], "dSSIM": r["WD0375_dSSIM"]} for r in rows]),
        "fs050_summary": summarize([{**r, "dPSNR": r["FS050_dPSNR"], "dSSIM": r["FS050_dSSIM"]} for r in rows]),
        "action_distribution": {action: sum(r["selected_action"] == action for r in rows) for action in ["WD0375", "FS050", "A0"]},
    }
    (args.out_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dist = []
    for action, count in summary["action_distribution"].items():
        dist.append({"action": action, "count": count, "usage": count / max(1, len(rows))})
    write_csv(args.out_dir / f"{args.prefix}_action_distribution.csv", dist)
    s = summary["summary"]
    locked_pass = (
        s["mean_dPSNR"] >= 1.0
        and s["hard_bottom25_dPSNR"] >= 1.0
        and s["easy_top25_dPSNR"] >= 0.0
        and s["positive_ratio"] >= 0.80
        and s["severe_loss_per_600"] <= 60.0
    )
    decision = "LOCKED_C11_SELECTOR_ONE_SHOT_PASS_REVIEW_DISTILLATION_LATER" if locked_pass else "LOCKED_C11_SELECTOR_ONE_SHOT_FAIL_NO_TUNING"
    (args.out_dir / f"{args.prefix}_decision.md").write_text(
        "# Locked C11 Selector One-Shot Decision\n\n"
        f"Decision: `{decision}`\n\n"
        f"mean/hard/easy/positive/severe: `{s['mean_dPSNR']:.6f}` / "
        f"`{s['hard_bottom25_dPSNR']:.6f}` / `{s['easy_top25_dPSNR']:.6f}` / "
        f"`{s['positive_ratio']:.6f}` / `{s['severe_loss_per_600']:.2f}/600`.\n\n"
        "Locked output is evidence only. It must not tune alpha, features, checkpoints, profiles, actions, experts, or distillation targets.\n",
        encoding="utf-8",
    )
    print(f"LOCKED_C11_SELECTOR_ONE_SHOT_OK decision={decision}")


if __name__ == "__main__":
    main()
