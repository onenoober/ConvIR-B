#!/usr/bin/env python3
"""v2.34 NoPost teacher-delta projection and bridge audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from data.data_load import DeblurDataset  # noqa: E402
from models.ConvIR import build_net  # noqa: E402

ROUTE_ID = "haze4k_v2_34_nopost_teacher_delta_projection_and_multistage_bridge_audit_20260706"
INSERTION_GROUPS: dict[str, list[str]] = {
    "S4_encoder_late": ["S4_encoder_late"],
    "S5_bottleneck_mid": ["S5_bottleneck_mid"],
    "S6_decoder_early": ["S6_decoder_early"],
    "S4_plus_S6": ["S4_encoder_late", "S6_decoder_early"],
    "S5_plus_S6": ["S5_bottleneck_mid", "S6_decoder_early"],
    "S4_plus_S5_plus_S6": ["S4_encoder_late", "S5_bottleneck_mid", "S6_decoder_early"],
    "decoder_mid": ["decoder_mid"],
}


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def cvar_low(values: list[float], frac: float = 0.05) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = max(1, math.ceil(len(ordered) * frac))
    return statistics.mean(ordered[:k])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_model(path: str | Path, map_location: Any) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def build_official(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_net("base", "Haze4K", "original").to(device)
    model.load_state_dict(load_checkpoint_model(checkpoint, device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_wdmamba(repo: str | Path, checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    repo = Path(repo)
    checkpoint = Path(checkpoint)
    try:
        import transformers.generation as tg
        for name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
            if not hasattr(tg, name):
                setattr(tg, name, type(name, (object,), {}))
    except Exception:
        pass

    def pkg(name: str, path: Path) -> None:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = mod

    def load_mod(name: str, path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load module {name} from {path}")
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
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["params"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def pad_to_factor(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    return F.pad(x, (0, pad_w, 0, pad_h), "reflect"), h, w


def infer_wdmamba(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    padded, h, w = pad_to_factor(x, 4)
    out = model.restoration_network(padded)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return torch.clamp(out[:, :, :h, :w], 0, 1)


def make_dataset(data_dir: str | Path, split: str = "train") -> DeblurDataset:
    return DeblurDataset(str(Path(data_dir) / split), "Haze4K")


def ensure_crop_size(tensor: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = tensor.shape
    if h >= size and w >= size:
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=(max(size, h), max(size, w)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def crop_pair(
    input_img: torch.Tensor,
    label_img: torch.Tensor,
    size: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_img = ensure_crop_size(input_img, size)
    label_img = ensure_crop_size(label_img, size)
    _, h, w = input_img.shape
    if seed is None:
        top = max(0, (h - size) // 2)
        left = max(0, (w - size) // 2)
    else:
        rng = random.Random(seed)
        top = rng.randint(0, max(0, h - size))
        left = rng.randint(0, max(0, w - size))
    return input_img[:, top:top + size, left:left + size], label_img[:, top:top + size, left:left + size]


def psnr_per_sample(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    mse = (pred - label).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def charbonnier_loss(pred: torch.Tensor, label: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - label).pow(2) + eps * eps).mean()


def lowpass(tensor: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    return F.avg_pool2d(tensor, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)


def tensor_rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.pow(2).mean().sqrt()


def lowfreq_psnr(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return psnr_per_sample(lowpass(pred), lowpass(label))


def load_wdmamba_table(path: str | Path) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    table_path = Path(path)
    if not table_path.exists():
        return {}
    return {row.get("name", ""): row for row in read_csv(table_path)}


def load_v233_p4_reference(path: str | Path) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    table_path = Path(path)
    if not table_path.exists():
        return {}
    rows = read_csv(table_path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("control_id") == "a0_baseline":
            out[row.get("sample_name", "")] = row
    return out


def bucket_samples(rows: list[dict[str, Any]]) -> None:
    values = [float(row["base_psnr_A0"]) for row in rows]
    hard_cut = percentile(values, 25) or 0.0
    easy_cut = percentile(values, 75) or 0.0
    for row in rows:
        value = float(row["base_psnr_A0"])
        if value <= hard_cut:
            row["hardness_bucket"] = "hard_bottom25_by_A0_canary"
        elif value >= easy_cut:
            row["hardness_bucket"] = "easy_top25_by_A0_canary"
        else:
            row["hardness_bucket"] = "mid_by_A0_canary"
        row["strong_reference_bucket"] = "strong_top25_by_A0_canary" if value >= easy_cut else "not_strong"
        row["easy_guard"] = row["strong_reference_bucket"] == "strong_top25_by_A0_canary"


def prepare_samples(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    official = build_official(args.checkpoint, device)
    wdmamba = load_wdmamba(args.wdmamba_repo, args.wdmamba_checkpoint, device)
    dataset = make_dataset(args.data_dir, "train")
    samples: list[dict[str, Any]] = []
    count = min(args.sample_count, len(dataset))
    with torch.no_grad():
        for index in range(count):
            inp, label = dataset[index]
            inp, label = crop_pair(inp, label, args.crop_size, seed=args.seed + index)
            x = inp.unsqueeze(0).to(device)
            y = label.unsqueeze(0).to(device)
            a0 = torch.clamp(official(x)[2], 0, 1)
            teacher_full = infer_wdmamba(wdmamba, x)
            teacher_blend = torch.clamp(a0 + args.teacher_alpha * (teacher_full - a0), 0, 1)
            a0_psnr = psnr_per_sample(a0, y).item()
            teacher_full_psnr = psnr_per_sample(teacher_full, y).item()
            teacher_blend_psnr = psnr_per_sample(teacher_blend, y).item()
            teacher_lowfreq_delta = (lowfreq_psnr(teacher_blend, y) - lowfreq_psnr(a0, y)).item()
            samples.append({
                "sample_index": index,
                "sample_name": dataset.image_list[index],
                "split_source": "Haze4K/train_canary_first32_seeded_crop",
                "x": x.detach().cpu(),
                "y": y.detach().cpu(),
                "a0": a0.detach().cpu(),
                "teacher_full": teacher_full.detach().cpu(),
                "teacher_blend": teacher_blend.detach().cpu(),
                "base_psnr_A0": a0_psnr,
                "teacher_full_psnr": teacher_full_psnr,
                "teacher_psnr": teacher_blend_psnr,
                "teacher_delta_vs_A0": teacher_blend_psnr - a0_psnr,
                "teacher_lowfreq_LL_delta": teacher_lowfreq_delta,
                "p4_eligible_mask": (teacher_blend_psnr - a0_psnr) >= args.p4_mask_min_gain,
            })
    del wdmamba
    if device.type == "cuda":
        torch.cuda.empty_cache()
    bucket_samples(samples)
    return samples


def run_p0(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    samples = prepare_samples(args, device)
    p1_table = load_wdmamba_table(args.wdmamba_table)
    v233_ref = load_v233_p4_reference(args.v233_p4_per_image)
    audit_rows: list[dict[str, Any]] = []
    benefit_rows: list[dict[str, Any]] = []
    missing_p1 = 0
    p1_p4_disagree = 0
    v233_delta_abs_diffs: list[float] = []
    for sample in samples:
        name = str(sample["sample_name"])
        p1 = p1_table.get(name, {})
        p1_delta = to_float(p1.get("expert_a0p5_dPSNR"))
        p1_dssim = to_float(p1.get("expert_a0p5_dSSIM"))
        p1_eligible = bool(p1 and p1_delta is not None and p1_dssim is not None and p1_delta >= 0.10 and p1_dssim >= -0.001)
        if not p1:
            missing_p1 += 1
        if p1 and p1_eligible != bool(sample["p4_eligible_mask"]):
            p1_p4_disagree += 1
        if not p1:
            mask_reason = "missing_from_p1_wdmamba_table"
        elif p1_eligible and not sample["p4_eligible_mask"]:
            mask_reason = "p1_full_image_positive_but_crop_alpha0p5_below_p4_threshold"
        elif (not p1_eligible) and sample["p4_eligible_mask"]:
            mask_reason = "crop_alpha0p5_positive_but_p1_full_image_guard_failed"
        elif sample["p4_eligible_mask"]:
            mask_reason = "eligible_in_both_p1_and_p4"
        else:
            mask_reason = "not_eligible_in_p1_or_p4"
        ref = v233_ref.get(name, {})
        ref_delta = to_float(ref.get("teacher_blend_delta"))
        if ref_delta is not None:
            v233_delta_abs_diffs.append(abs(ref_delta - float(sample["teacher_delta_vs_A0"])))
        common = {
            "sample_name": name,
            "sample_index": sample["sample_index"],
            "split_source": sample["split_source"],
            "base_psnr_A0": sample["base_psnr_A0"],
            "teacher_id": "wdmamba_alpha0p5",
            "teacher_psnr": sample["teacher_psnr"],
            "teacher_full_psnr": sample["teacher_full_psnr"],
            "teacher_delta_vs_A0": sample["teacher_delta_vs_A0"],
            "teacher_lowfreq_LL_delta": sample["teacher_lowfreq_LL_delta"],
            "p1_table_delta_vs_A0": p1_delta,
            "p1_table_dssim": p1_dssim,
            "p1_eligible_mask": p1_eligible,
            "p4_eligible_mask": sample["p4_eligible_mask"],
            "v233_p4_teacher_blend_delta": ref_delta,
            "mask_reason": mask_reason,
            "hardness_bucket": sample["hardness_bucket"],
            "strong_reference_bucket": sample["strong_reference_bucket"],
            "easy_guard": sample["easy_guard"],
            "loss_weight_teacher": 1.0,
            "loss_weight_preservation": 0.15,
        }
        audit_rows.append(common)
        benefit_rows.append(common)
    deltas = [float(row["teacher_delta_vs_A0"]) for row in benefit_rows]
    hard = [float(row["teacher_delta_vs_A0"]) for row in benefit_rows if str(row["hardness_bucket"]).startswith("hard")]
    easy = [float(row["teacher_delta_vs_A0"]) for row in benefit_rows if str(row["hardness_bucket"]).startswith("easy")]
    p4_eligible_count = sum(1 for row in audit_rows if row["p4_eligible_mask"])
    p1_eligible_count = sum(1 for row in audit_rows if row["p1_eligible_mask"])
    direct_mean = mean(deltas) or 0.0
    hard_mean = mean(hard) or 0.0
    gate_pass = missing_p1 == 0 and direct_mean >= 0.30 and hard_mean >= 0.50
    fields = [
        "sample_name", "sample_index", "split_source", "base_psnr_A0", "teacher_id",
        "teacher_psnr", "teacher_full_psnr", "teacher_delta_vs_A0", "teacher_lowfreq_LL_delta",
        "p1_table_delta_vs_A0", "p1_table_dssim", "p1_eligible_mask", "p4_eligible_mask",
        "v233_p4_teacher_blend_delta", "mask_reason", "hardness_bucket", "strong_reference_bucket",
        "easy_guard", "loss_weight_teacher", "loss_weight_preservation",
    ]
    write_csv(out_dir / "v234_p0_mask_join_audit.csv", audit_rows, fields)
    write_csv(out_dir / "v234_p0_exact_canary_teacher_direct_benefit.csv", benefit_rows, fields)
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P0 mask/join audit and exact canary direct teacher benefit",
        "locked_test_touched": False,
        "sample_count": len(samples),
        "teacher_id": "wdmamba_alpha0p5",
        "teacher_alpha": args.teacher_alpha,
        "direct_teacher_mean_delta": direct_mean,
        "direct_teacher_hard_delta": hard_mean,
        "direct_teacher_easy_delta": mean(easy),
        "direct_teacher_p05": percentile(deltas, 5),
        "direct_teacher_cvar5": cvar_low(deltas),
        "p1_eligible_count": p1_eligible_count,
        "p1_eligible_coverage": p1_eligible_count / len(samples) if samples else None,
        "p4_eligible_count": p4_eligible_count,
        "p4_eligible_coverage": p4_eligible_count / len(samples) if samples else None,
        "p1_p4_mask_disagreement_count": p1_p4_disagree,
        "missing_from_p1_table": missing_p1,
        "v233_reference_rows": len(v233_ref),
        "v233_recompute_mean_abs_delta_diff": mean(v233_delta_abs_diffs),
        "gate_pass": gate_pass,
        "p1_free_tensor_authorized": gate_pass,
        "blocked_reason": "" if gate_pass else "P0 direct teacher benefit or join gate failed",
        "wdmamba_checkpoint": str(args.wdmamba_checkpoint),
        "wdmamba_sha256": file_sha256(args.wdmamba_checkpoint),
    }
    write_json(out_dir / "v234_p0_closeout.json", payload)
    return payload


def forward_with_deltas(
    model: torch.nn.Module,
    x: torch.Tensor,
    deltas: dict[str, torch.Tensor] | None = None,
    capture_shapes: dict[str, tuple[int, ...]] | None = None,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    deltas = deltas or {}
    used: dict[str, torch.Tensor] = {}

    def apply_delta(name: str, feature: torch.Tensor) -> torch.Tensor:
        if capture_shapes is not None:
            capture_shapes[name] = tuple(feature.shape)
        if name in deltas:
            used[name] = deltas[name]
            return feature + deltas[name]
        return feature

    x_2 = F.interpolate(x, scale_factor=0.5)
    x_4 = F.interpolate(x_2, scale_factor=0.5)
    z2 = model.SCM2(x_2)
    z4 = model.SCM1(x_4)
    outputs: list[torch.Tensor] = []
    x_ = model.feat_extract[0](x)
    res1 = model.Encoder[0](x_)
    z = model.feat_extract[1](res1)
    z = model.FAM2(z, z2)
    res2 = model.Encoder[1](z)
    res2 = apply_delta("S4_encoder_late", res2)
    z = model.feat_extract[2](res2)
    z = model.FAM1(z, z4)
    z = model.Encoder[2](z)
    z = apply_delta("S5_bottleneck_mid", z)
    z = model.Decoder[0](z)
    z = apply_delta("S6_decoder_early", z)
    z_ = model.ConvsOut[0](z)
    z = model.feat_extract[3](z)
    outputs.append(z_ + x_4)
    z = torch.cat([z, res2], dim=1)
    z = model.Convs[0](z)
    z = model.Decoder[1](z)
    z = apply_delta("decoder_mid", z)
    z_ = model.ConvsOut[1](z)
    z = model.feat_extract[4](z)
    outputs.append(z_ + x_2)
    z = torch.cat([z, res1], dim=1)
    z = model.Convs[1](z)
    z = model.Decoder[2](z)
    z = model.feat_extract[5](z)
    outputs.append(z + x)
    return outputs, used


def projection_loss(pred: torch.Tensor, target: torch.Tensor, deltas: dict[str, torch.Tensor], energy_weight: float) -> torch.Tensor:
    loss = charbonnier_loss(lowpass(pred), lowpass(target)) + 0.25 * charbonnier_loss(pred, target)
    if energy_weight > 0:
        energy = sum(tensor_rms(delta) for delta in deltas.values())
        loss = loss + energy_weight * energy
    return loss


def optimize_free_tensor_for_sample(
    model: torch.nn.Module,
    sample: dict[str, Any],
    insertion_point: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    x = sample["x"].to(device)
    y = sample["y"].to(device)
    a0 = sample["a0"].to(device)
    teacher_blend = sample["teacher_blend"].to(device)
    active_points = INSERTION_GROUPS[insertion_point]
    shapes: dict[str, tuple[int, ...]] = {}
    with torch.no_grad():
        forward_with_deltas(model, x, capture_shapes=shapes)
    deltas = {
        point: torch.zeros(shapes[point], device=device, requires_grad=True)
        for point in active_points
    }
    optimizer = torch.optim.AdamW(list(deltas.values()), lr=args.learning_rate, weight_decay=0.0)
    losses: list[float] = []
    finite_grad = True
    max_grad_norm = 0.0
    for _step in range(args.projection_steps):
        optimizer.zero_grad(set_to_none=True)
        outputs, _used = forward_with_deltas(model, x, deltas=deltas)
        pred = torch.clamp(outputs[2], 0, 1)
        loss = projection_loss(pred, teacher_blend, deltas, args.energy_weight)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(list(deltas.values()), args.grad_clip_norm)
        norm_value = float(total_norm.detach().cpu()) if torch.is_tensor(total_norm) else float(total_norm)
        max_grad_norm = max(max_grad_norm, norm_value)
        finite_grad = finite_grad and math.isfinite(norm_value)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        outputs, _used = forward_with_deltas(model, x, deltas=deltas)
        pred = torch.clamp(outputs[2], 0, 1)
        out_delta = pred - a0
        feature_energy = sum(tensor_rms(delta.detach()).item() for delta in deltas.values())
        output_energy = tensor_rms(out_delta).item()
        low_ratio = (tensor_rms(lowpass(out_delta)) / tensor_rms(out_delta).clamp_min(1e-12)).item()
        base_psnr = float(sample["base_psnr_A0"])
        free_psnr = psnr_per_sample(pred, y).item()
        teacher_delta = float(sample["teacher_delta_vs_A0"])
        free_delta = free_psnr - base_psnr
    return {
        "insertion_point": insertion_point,
        "sample_name": sample["sample_name"],
        "sample_index": sample["sample_index"],
        "hardness_bucket": sample["hardness_bucket"],
        "strong_reference_bucket": sample["strong_reference_bucket"],
        "eligible_mask": sample["p4_eligible_mask"],
        "base_psnr_A0": sample["base_psnr_A0"],
        "teacher_psnr": sample["teacher_psnr"],
        "direct_teacher_delta": teacher_delta,
        "free_tensor_psnr": free_psnr,
        "free_tensor_delta": free_delta,
        "projection_ratio_vs_teacher": (free_delta / teacher_delta) if teacher_delta > 1e-9 else None,
        "feature_delta_energy": feature_energy,
        "output_delta_energy": output_energy,
        "lowfreq_output_ratio": low_ratio,
        "highfreq_leakage": max(0.0, 1.0 - low_ratio),
        "steps": args.projection_steps,
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "loss_drop": (losses[0] - losses[-1]) if losses else 0.0,
        "max_grad_norm_before_clip": max_grad_norm,
        "gradient_finite": finite_grad,
    }


def summarize_projection(point: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["free_tensor_delta"]) for row in rows]
    teacher = [float(row["direct_teacher_delta"]) for row in rows]
    hard = [float(row["free_tensor_delta"]) for row in rows if str(row["hardness_bucket"]).startswith("hard")]
    easy = [float(row["free_tensor_delta"]) for row in rows if str(row["hardness_bucket"]).startswith("easy")]
    strong = [float(row["free_tensor_delta"]) for row in rows if row["strong_reference_bucket"] == "strong_top25_by_A0_canary"]
    eligible = [row for row in rows if row["eligible_mask"]]
    teacher_mean = mean(teacher) or 0.0
    free_mean = mean(deltas) or 0.0
    p05 = percentile(deltas, 5)
    cvar = cvar_low(deltas)
    severe = sum(1 for value in deltas if value <= -0.20) / len(deltas) if deltas else None
    strong_reg = sum(1 for value in strong if value <= -0.05) / len(strong) if strong else None
    projection_ratio = free_mean / teacher_mean if teacher_mean > 1e-9 else None
    gate_pass = bool(
        projection_ratio is not None and projection_ratio >= 0.10
        and free_mean >= 0.05
        and (p05 is not None and p05 >= -0.03)
        and (severe is not None and severe <= 0.05)
        and (strong_reg is not None and strong_reg <= 0.05)
    )
    return {
        "insertion_point": point,
        "count": len(rows),
        "eligible_count": len(eligible),
        "direct_teacher_mean_delta": teacher_mean,
        "free_tensor_mean_delta": free_mean,
        "free_tensor_hard_delta": mean(hard),
        "free_tensor_easy_delta": mean(easy),
        "p05": p05,
        "cvar5": cvar,
        "severe": severe,
        "strong_reference_regression_rate": strong_reg,
        "projection_ratio_vs_teacher": projection_ratio,
        "feature_delta_energy": mean([float(row["feature_delta_energy"]) for row in rows]),
        "output_delta_energy": mean([float(row["output_delta_energy"]) for row in rows]),
        "lowfreq_output_ratio": mean([float(row["lowfreq_output_ratio"]) for row in rows]),
        "highfreq_leakage": mean([float(row["highfreq_leakage"]) for row in rows]),
        "steps": rows[0]["steps"] if rows else None,
        "loss_start": mean([float(row["loss_start"]) for row in rows if row["loss_start"] is not None]),
        "loss_end": mean([float(row["loss_end"]) for row in rows if row["loss_end"] is not None]),
        "gate_pass": gate_pass,
    }


def run_p1(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    samples = prepare_samples(args, device)
    model = build_official(args.checkpoint, device)
    per_image_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for point in INSERTION_GROUPS:
        point_rows = []
        for sample in samples:
            row = optimize_free_tensor_for_sample(model, sample, point, args, device)
            point_rows.append(row)
            per_image_rows.append(row)
        summary_rows.append(summarize_projection(point, point_rows))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(out_dir / "v234_p1_free_tensor_projection_per_image.csv", per_image_rows)
    fields = [
        "insertion_point", "count", "eligible_count", "direct_teacher_mean_delta",
        "free_tensor_mean_delta", "free_tensor_hard_delta", "free_tensor_easy_delta",
        "p05", "cvar5", "severe", "strong_reference_regression_rate",
        "projection_ratio_vs_teacher", "feature_delta_energy", "output_delta_energy",
        "lowfreq_output_ratio", "highfreq_leakage", "steps", "loss_start", "loss_end",
        "gate_pass",
    ]
    write_csv(out_dir / "v234_p1_free_tensor_projection_by_insertion.csv", summary_rows, fields)
    passing = [row for row in summary_rows if row["gate_pass"]]
    best = max(
        summary_rows,
        key=lambda row: float(row["projection_ratio_vs_teacher"] if row["projection_ratio_vs_teacher"] is not None else -999.0),
    )
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P1 free-tensor teacher-delta projection by insertion",
        "locked_test_touched": False,
        "sample_count": len(samples),
        "teacher_id": "wdmamba_alpha0p5",
        "teacher_alpha": args.teacher_alpha,
        "projection_steps": args.projection_steps,
        "gate_pass": bool(passing),
        "passing_insertion_points": [row["insertion_point"] for row in passing],
        "best_insertion_point": best["insertion_point"],
        "best_projection_ratio_vs_teacher": best["projection_ratio_vs_teacher"],
        "p2_generator_gap_authorized": bool(passing),
        "p3_gradient_conflict_authorized": bool(passing),
        "p4_microcanary32_authorized": False,
        "summary": summary_rows,
    }
    write_json(out_dir / "v234_p1_closeout.json", payload)
    return payload


def run_closeout(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    p0_path = out_dir / "v234_p0_closeout.json"
    p1_path = out_dir / "v234_p1_closeout.json"
    p0 = json.loads(p0_path.read_text(encoding="utf-8")) if p0_path.exists() else {}
    p1 = json.loads(p1_path.read_text(encoding="utf-8")) if p1_path.exists() else {}
    p0_pass = bool(p0.get("gate_pass"))
    p1_pass = bool(p1.get("gate_pass"))
    if not p0:
        decision = "P0_NOT_RUN"
    elif not p0_pass:
        decision = "P0_FAIL_MASK_JOIN_OR_DIRECT_TEACHER_CANARY"
    elif not p1:
        decision = "P1_AUTHORIZED_NOT_RUN"
    elif not p1_pass:
        decision = "P1_FAIL_FREE_TENSOR_PROJECTION_NO_REPRESENTABLE_HEADROOM"
    else:
        decision = "P1_PASS_AUTHORIZE_P2_P3_DIAGNOSTICS"
    payload = {
        "primary_question": "Is WDMamba teacher benefit representable inside ConvIR-B with a NoPost in-network carrier, or is the current route blocked by S5-BILFCF compression capacity?",
        "closed_reference": "v2.33 P4_FAIL_MASKED_CANARY32_NO_CANARY80",
        "not_a_continuation_of_v233_s5_bilfcf": True,
        "locked_test_touched": False,
        "canary80_authorized": False,
        "p0_mask_join_audit_pass": p0_pass if p0 else None,
        "p1_free_tensor_projection_pass": p1_pass if p1 else None,
        "p2_generator_gap_pass": None,
        "p3_gradient_conflict_pass": None,
        "p4_microcanary32_launched": False,
        "decision": decision,
    }
    write_json(out_dir / "v234_closeout.json", payload)
    (out_dir / "v234_decision_tree.md").write_text(
        "\n".join([
            "# v2.34 Decision Tree",
            "",
            f"Decision: `{decision}`",
            "",
            f"- P0 gate pass: `{p0_pass if p0 else None}`",
            f"- P1 gate pass: `{p1_pass if p1 else None}`",
            "- P2 launched: `False`",
            "- P3 launched: `False`",
            "- P4 launched: `False`",
            "- canary80 authorized: `False`",
            "- locked test touched: `False`",
            "",
        ]),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "\n".join([
            "# Haze4K v2.34 NoPost Teacher-Delta Projection and Multi-Stage Bridge Audit",
            "",
            f"State: `{decision}`",
            "",
            "Route card: `experience_docx/experiment_cards/2026-07-06-haze4k-v2-34-nopost-teacher-delta-projection-and-multistage-bridge-audit.md`.",
            "Central index: `experience_docx/EXPERIMENT_INDEX.md`.",
            "",
            "Primary evidence:",
            "",
            "- `v234_p0_mask_join_audit.csv`",
            "- `v234_p0_exact_canary_teacher_direct_benefit.csv`",
            "- `v234_p0_closeout.json`",
            "- `v234_p1_free_tensor_projection_by_insertion.csv`",
            "- `v234_p1_free_tensor_projection_per_image.csv`",
            "- `v234_p1_closeout.json`",
            "- `v234_closeout.json`",
            "",
            "Locked test was not touched.",
            "",
        ]),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["p0", "p1", "closeout"])
    parser.add_argument("--data_dir", default="/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K")
    parser.add_argument("--checkpoint", default="/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--wdmamba_repo", default="/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba")
    parser.add_argument("--wdmamba_checkpoint", default="/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth")
    parser.add_argument("--wdmamba_table", default="")
    parser.add_argument("--v233_p4_per_image", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--sample_count", type=int, default=32)
    parser.add_argument("--teacher_alpha", type=float, default=0.5)
    parser.add_argument("--p4_mask_min_gain", type=float, default=0.05)
    parser.add_argument("--projection_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--energy_weight", type=float, default=0.0001)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.phase == "p0":
        payload = run_p0(args)
    elif args.phase == "p1":
        payload = run_p1(args)
    else:
        payload = run_closeout(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

