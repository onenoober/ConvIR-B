#!/usr/bin/env python3
"""v2.33 NoPost teacher-source and BILFCF trainability audits."""
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
from models.ConvIR import build_bilfcf_net, build_net  # noqa: E402

ALLOWED_NEW_PREFIXES = ("BILFCF_",)
ROUTE_ID = "haze4k_v2_33_nopost_teacher_benefit_source_and_bilfcf_trainability_audit_20260705"


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


def fold_for_name(name: str, folds: int = 5) -> int:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_checkpoint_model(path: str | Path, map_location: Any) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_haze4k_partial(model: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    state = load_checkpoint_model(checkpoint_path, "cpu")
    model_state = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    shape_mismatch = []
    unexpected = []
    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
        elif model_state[key].shape != value.shape:
            shape_mismatch.append([key, list(value.shape), list(model_state[key].shape)])
        else:
            loaded[key] = value
    missing = [key for key in model_state if key not in loaded]
    bad_missing = [key for key in missing if not any(key.startswith(prefix) for prefix in ALLOWED_NEW_PREFIXES)]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            "partial-load failed: "
            f"unexpected={unexpected[:20]} shape_mismatch={shape_mismatch[:20]} bad_missing={bad_missing[:20]}"
        )
    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    return {"loaded_count": len(loaded), "missing_new_modules": sorted(missing)}


def build_official(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_net("base", "Haze4K", "original").to(device)
    model.load_state_dict(load_checkpoint_model(checkpoint, device))
    model.eval()
    return model


def build_candidate(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_bilfcf_net(
        "base", "Haze4K", "original",
        insertion="s5", alpha_max=0.02, gate_bias=-4.0,
        hidden_channels=32, lowpass_kernel=5,
    ).to(device)
    load_haze4k_partial(model, checkpoint)
    return model


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def set_adapter_only(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(ALLOWED_NEW_PREFIXES)
        if param.requires_grad:
            trainable.append(param)
    if not trainable:
        raise RuntimeError("No BILFCF trainable parameters found")
    return trainable


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


def crop_pair(input_img: torch.Tensor, label_img: torch.Tensor, size: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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


def make_dataset(data_dir: str | Path, split: str = "train") -> DeblurDataset:
    return DeblurDataset(str(Path(data_dir) / split), "Haze4K")


def psnr_per_sample(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    mse = (pred - label).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def charbonnier_loss(pred: torch.Tensor, label: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - label).pow(2) + eps * eps).mean()


def lowpass(tensor: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    return F.avg_pool2d(tensor, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)


def tensor_rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.pow(2).mean().sqrt()


def run_p0(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    official = build_official(args.checkpoint, device)
    candidate = build_candidate(args.checkpoint, device)
    candidate.eval()
    dataset = make_dataset(args.data_dir, "train")
    input_img, _label = dataset[0]
    input_img, _ = crop_pair(input_img, input_img, args.crop_size)
    x = input_img.unsqueeze(0).to(device)
    with torch.no_grad():
        official_out = official(x)[2]
        candidate_out = candidate(x)[2]
    diff = (official_out - candidate_out).abs()
    trainable = set_adapter_only(candidate)
    payload = {
        "route_id": ROUTE_ID,
        "closed_reference": "v2.32 P2_FAIL_BOUNDED_FIELD_TRAINABILITY_PAUSE",
        "not_a_continuation_of_failed_v232_training": True,
        "forward_contract": "forward(self, x)",
        "teacher_or_expert_forward_input": False,
        "rgb_output_output_residual": False,
        "learned_rgb_post_output_correction": False,
        "locked_test_touched": False,
        "identity_max_abs_vs_A0": diff.max().item(),
        "identity_mean_abs_vs_A0": diff.mean().item(),
        "checkpoint": str(args.checkpoint),
        "trainable_param_count": sum(param.numel() for param in trainable),
        "pass": diff.max().item() <= 1e-7,
    }
    write_json(out_dir / "v233_p0_identity_zero_init_report.json", payload)
    (out_dir / "v233_p0_arch_contract_delta.md").write_text(
        "\n".join([
            "# v2.33 P0 Architecture Contract Delta",
            "",
            f"Route id: `{ROUTE_ID}`",
            "",
            "## Closed Reference",
            "",
            "- v2.32 S5-only alpha=0.02 loss_C adapter-only canary failed.",
            "- v2.33 does not continue canary80, P3 objective ablation, P2B selector, or locked test from v2.32.",
            "",
            "## Contract",
            "",
            f"- forward_contract: `{payload['forward_contract']}`",
            f"- teacher_or_expert_forward_input: `{payload['teacher_or_expert_forward_input']}`",
            f"- rgb_output_output_residual: `{payload['rgb_output_output_residual']}`",
            f"- learned_rgb_post_output_correction: `{payload['learned_rgb_post_output_correction']}`",
            f"- locked_test_touched: `{payload['locked_test_touched']}`",
            "",
            "## Identity",
            "",
            f"- identity_max_abs_vs_A0: `{payload['identity_max_abs_vs_A0']}`",
            f"- identity_mean_abs_vs_A0: `{payload['identity_mean_abs_vs_A0']}`",
            f"- trainable_param_count: `{payload['trainable_param_count']}`",
            f"- pass: `{payload['pass']}`",
            "",
        ]),
        encoding="utf-8",
    )
    (out_dir / "v233_p0_v232_failure_reference.md").write_text(
        "\n".join([
            "# v2.33 P0 v2.32 Failure Reference",
            "",
            "Closed reference: `v2.32 P2_FAIL_BOUNDED_FIELD_TRAINABILITY_PAUSE`.",
            "",
            "Not allowed in v2.33 before new gates pass:",
            "",
            "- no canary80 continuation from v2.32;",
            "- no P3 objective ablation continuation from v2.32;",
            "- no P2B selector probe;",
            "- no locked test;",
            "- no longer S5-only adapter-only BILFCF training with loss_C as a scaling attempt.",
            "",
        ]),
        encoding="utf-8",
    )
    return payload


def normalize_udpnet(path: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(path):
        rows.append({
            "teacher_id": "udpnet_full",
            "expert_family": "UDPNet",
            "source_path": str(path),
            "name": raw.get("name", ""),
            "split": raw.get("split", "unknown"),
            "a0_psnr": to_float(raw.get("a0_psnr"), 0.0) or 0.0,
            "teacher_psnr": to_float(raw.get("udpnet_psnr"), 0.0) or 0.0,
            "delta_psnr": to_float(raw.get("delta_psnr"), 0.0) or 0.0,
            "delta_ssim": to_float(raw.get("delta_ssim"), 0.0) or 0.0,
            "haze_density": None,
        })
    return rows


def normalize_wdmamba(path: Path, teacher_id: str, psnr_col: str, dpsnr_col: str, dssim_col: str) -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(path):
        a0 = to_float(raw.get("A0_PSNR"), 0.0) or 0.0
        delta = to_float(raw.get(dpsnr_col), 0.0) or 0.0
        rows.append({
            "teacher_id": teacher_id,
            "expert_family": "WDMamba",
            "source_path": str(path),
            "name": raw.get("name", ""),
            "split": raw.get("split", "unknown"),
            "a0_psnr": a0,
            "teacher_psnr": to_float(raw.get(psnr_col), a0 + delta) or (a0 + delta),
            "delta_psnr": delta,
            "delta_ssim": to_float(raw.get(dssim_col), 0.0) or 0.0,
            "haze_density": to_float(raw.get("haze_density_mean")),
        })
    return rows


def assign_buckets(rows: list[dict[str, Any]]) -> None:
    by_teacher: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_teacher.setdefault(str(row["teacher_id"]), []).append(row)
    for teacher_rows in by_teacher.values():
        psnrs = [float(row["a0_psnr"]) for row in teacher_rows]
        hard_cut = percentile(psnrs, 25) or 0.0
        easy_cut = percentile(psnrs, 75) or 0.0
        haze_values = [float(row["haze_density"]) for row in teacher_rows if row.get("haze_density") is not None]
        haze_high = percentile(haze_values, 75) if haze_values else None
        for row in teacher_rows:
            psnr = float(row["a0_psnr"])
            if psnr <= hard_cut:
                row["hardness_bucket"] = "hard_bottom25_by_a0"
            elif psnr >= easy_cut:
                row["hardness_bucket"] = "easy_top25_by_a0"
            else:
                row["hardness_bucket"] = "mid_by_a0"
            if haze_high is None or row.get("haze_density") is None:
                row["lowfreq_haze_bucket"] = "unknown"
            elif float(row["haze_density"]) >= haze_high:
                row["lowfreq_haze_bucket"] = "high_haze_density_top25"
            else:
                row["lowfreq_haze_bucket"] = "lower_haze_density"
            row["strong_reference_bucket"] = "strong_top25_by_a0" if psnr >= easy_cut else "not_strong"
            row["fold"] = fold_for_name(str(row["name"]))
            row["eligible"] = float(row["delta_psnr"]) >= 0.10 and float(row["delta_ssim"]) >= -0.001


def summarize_teacher(teacher_id: str, bucket: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["delta_psnr"]) for row in subset]
    hard = [row for row in subset if row["hardness_bucket"].startswith("hard")]
    easy = [row for row in subset if row["hardness_bucket"].startswith("easy")]
    strong = [row for row in subset if row["strong_reference_bucket"] == "strong_top25_by_a0"]
    lowfreq = [row for row in subset if row["lowfreq_haze_bucket"] == "high_haze_density_top25"]
    eligible = [row for row in subset if row["eligible"]]
    eligible_deltas = [float(row["delta_psnr"]) for row in eligible]
    fold_passes = 0
    for fold in range(5):
        frows = [row for row in subset if int(row["fold"]) == fold]
        if not frows:
            continue
        felig = [row for row in frows if row["eligible"]]
        fd = [float(row["delta_psnr"]) for row in felig]
        fhard = [float(row["delta_psnr"]) for row in frows if row["hardness_bucket"].startswith("hard")]
        severe = sum(1 for v in fd if v <= -0.20) / len(fd) if fd else 1.0
        if (
            mean(fhard) is not None and mean(fhard) >= 0.50
            and len(felig) / len(frows) >= 0.15
            and (percentile(fd, 5) or -999) >= -0.15
            and (cvar_low(fd) or -999) >= -0.35
            and severe <= 0.035
        ):
            fold_passes += 1
    lowfreq_gain = mean([float(row["delta_psnr"]) for row in lowfreq])
    payload = {
        "teacher_id": teacher_id,
        "expert_family": subset[0]["expert_family"] if subset else "unknown",
        "bucket": bucket,
        "hardness_bucket": "mixed" if bucket in {"all", "val_regular", "val_hard"} else bucket,
        "lowfreq_haze_bucket": "mixed_or_unknown",
        "strong_reference_bucket": "mixed",
        "count": len(subset),
        "mean_delta_psnr_vs_A0": mean(deltas),
        "hard_delta_psnr_vs_A0": mean([float(row["delta_psnr"]) for row in hard]),
        "easy_delta_psnr_vs_A0": mean([float(row["delta_psnr"]) for row in easy]),
        "p05_delta_vs_A0": percentile(deltas, 5),
        "cvar5_delta_vs_A0": cvar_low(deltas),
        "severe_regression_rate_vs_A0": sum(1 for v in deltas if v <= -0.20) / len(deltas) if deltas else None,
        "strong_reference_regression_rate": sum(1 for row in strong if float(row["delta_psnr"]) <= -0.05) / len(strong) if strong else None,
        "lowfreq_LL_delta_gain": lowfreq_gain,
        "global_structure_proxy_gain": mean([float(row["delta_ssim"]) for row in subset]),
        "color_shift_risk": "not_measured_table_only",
        "teacher_worse_rate": sum(1 for v in deltas if v < 0) / len(deltas) if deltas else None,
        "eligible_mask_coverage": len(eligible) / len(subset) if subset else None,
        "eligible_mask_mean_gain": mean(eligible_deltas),
        "eligible_mask_p05": percentile(eligible_deltas, 5),
        "eligible_mask_cvar5": cvar_low(eligible_deltas),
        "eligible_mask_severe": sum(1 for v in eligible_deltas if v <= -0.20) / len(eligible_deltas) if eligible_deltas else None,
        "fold_pass": fold_passes,
    }
    payload["gate_pass"] = bool(
        payload["hard_delta_psnr_vs_A0"] is not None and payload["hard_delta_psnr_vs_A0"] >= 0.50
        and payload["lowfreq_LL_delta_gain"] is not None and payload["lowfreq_LL_delta_gain"] >= 0.30
        and payload["eligible_mask_coverage"] is not None and payload["eligible_mask_coverage"] >= 0.15
        and payload["eligible_mask_p05"] is not None and payload["eligible_mask_p05"] >= -0.15
        and payload["eligible_mask_cvar5"] is not None and payload["eligible_mask_cvar5"] >= -0.35
        and payload["eligible_mask_severe"] is not None and payload["eligible_mask_severe"] <= 0.035
        and payload["fold_pass"] >= 4
    )
    return payload


def run_p1(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    notes = []
    if args.udp_table and Path(args.udp_table).exists():
        rows.extend(normalize_udpnet(Path(args.udp_table)))
        notes.append({"teacher_id": "udpnet_full", "source": args.udp_table, "status": "loaded"})
    if args.wdmamba_table and Path(args.wdmamba_table).exists():
        path = Path(args.wdmamba_table)
        rows.extend(normalize_wdmamba(path, "wdmamba_full", "expert_PSNR", "dPSNR_endpoint", "dSSIM_endpoint"))
        rows.extend(normalize_wdmamba(path, "wdmamba_alpha0p5", "expert_a0p5_PSNR", "expert_a0p5_dPSNR", "expert_a0p5_dSSIM"))
        notes.append({"teacher_id": "wdmamba_full/wdmamba_alpha0p5", "source": str(path), "status": "loaded"})
    if not rows:
        raise RuntimeError("No teacher source rows loaded")
    assign_buckets(rows)
    audit_rows = []
    negative_rows = []
    for teacher_id in sorted({str(row["teacher_id"]) for row in rows}):
        trows = [row for row in rows if row["teacher_id"] == teacher_id]
        groups: list[tuple[str, list[dict[str, Any]]]] = [("all", trows)]
        groups += [(split, [row for row in trows if row["split"] == split]) for split in sorted({str(row["split"]) for row in trows})]
        groups += [(hb, [row for row in trows if row["hardness_bucket"] == hb]) for hb in sorted({str(row["hardness_bucket"]) for row in trows})]
        for bucket, subset in groups:
            audit_rows.append(summarize_teacher(teacher_id, bucket, subset))
        for split in sorted({str(row["split"]) for row in trows} | {"all"}):
            subset = trows if split == "all" else [row for row in trows if row["split"] == split]
            deltas = [float(row["delta_psnr"]) for row in subset]
            negative_rows.append({
                "teacher_id": teacher_id,
                "split": split,
                "count": len(subset),
                "teacher_worse_rate": sum(1 for v in deltas if v < 0) / len(deltas) if deltas else None,
                "p05_delta_vs_A0": percentile(deltas, 5),
                "cvar5_delta_vs_A0": cvar_low(deltas),
                "severe_regression_rate_vs_A0": sum(1 for v in deltas if v <= -0.20) / len(deltas) if deltas else None,
            })
    fields = [
        "teacher_id", "expert_family", "bucket", "hardness_bucket", "lowfreq_haze_bucket",
        "strong_reference_bucket", "count", "mean_delta_psnr_vs_A0", "hard_delta_psnr_vs_A0",
        "easy_delta_psnr_vs_A0", "p05_delta_vs_A0", "cvar5_delta_vs_A0",
        "severe_regression_rate_vs_A0", "strong_reference_regression_rate", "lowfreq_LL_delta_gain",
        "global_structure_proxy_gain", "color_shift_risk", "teacher_worse_rate", "eligible_mask_coverage",
        "eligible_mask_mean_gain", "eligible_mask_p05", "eligible_mask_cvar5", "eligible_mask_severe",
        "fold_pass", "gate_pass",
    ]
    write_csv(out_dir / "v233_p1_teacher_benefit_audit.csv", audit_rows, fields)
    write_csv(out_dir / "v233_p1_teacher_negative_transfer_audit.csv", negative_rows)
    all_rows = [row for row in audit_rows if row["bucket"] == "all"]
    gate_pass_teachers = [row["teacher_id"] for row in all_rows if row.get("gate_pass")]
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P1 teacher-benefit source audit",
        "locked_test_touched": False,
        "source_notes": notes,
        "teacher_count": len(all_rows),
        "gate_pass_teachers": gate_pass_teachers,
        "gate_pass": bool(gate_pass_teachers),
    }
    write_json(out_dir / "v233_p1_teacher_benefit_closeout.json", payload)
    (out_dir / "v233_p1_teacher_delta_compressibility.md").write_text(
        "# v2.33 P1 Teacher Delta Compressibility Note\n\n"
        "This P1 audit is table-only and uses existing internal Haze4K A0-vs-teacher per-image evidence.\n"
        "It does not run inference and does not touch locked test data.\n\n"
        f"Gate pass teachers: `{', '.join(gate_pass_teachers) if gate_pass_teachers else 'none'}`.\n",
        encoding="utf-8",
    )
    return payload


def pick_hard_sample(dataset: DeblurDataset, official: torch.nn.Module, device: torch.device, crop_size: int, scan_count: int) -> tuple[int, float, str]:
    best_index = 0
    best_psnr = float("inf")
    for index in range(min(scan_count, len(dataset))):
        inp, label = dataset[index]
        inp, label = crop_pair(inp, label, crop_size)
        x = inp.unsqueeze(0).to(device)
        y = label.unsqueeze(0).to(device)
        with torch.no_grad():
            value = psnr_per_sample(torch.clamp(official(x)[2], 0, 1), y).item()
        if value < best_psnr:
            best_index = index
            best_psnr = value
    return best_index, best_psnr, dataset.image_list[best_index]


def objective_loss(objective: str, candidate_final: torch.Tensor, official_final: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    if objective == "preservation_only":
        return charbonnier_loss(candidate_final, official_final)
    if objective == "one_image_gt_overfit":
        return charbonnier_loss(candidate_final, label) + 0.25 * charbonnier_loss(lowpass(candidate_final), lowpass(label))
    if objective == "lowfreq_positive_delta":
        return charbonnier_loss(lowpass(candidate_final), lowpass(label)) + 0.10 * charbonnier_loss(candidate_final, official_final)
    if objective == "lowfreq_sign_flip_control":
        target = torch.clamp(2.0 * official_final - label, 0, 1)
        return charbonnier_loss(lowpass(candidate_final), lowpass(target)) + 0.10 * charbonnier_loss(candidate_final, official_final)
    raise ValueError(objective)


def train_micro_objective(args: argparse.Namespace, objective: str, sample_index: int, sample_name: str, base_psnr: float) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    official = build_official(args.checkpoint, device)
    candidate = build_candidate(args.checkpoint, device)
    trainable = set_adapter_only(candidate)
    # Keep the frozen ConvIR-B backbone and its BatchNorm layers in eval mode;
    # the one-image probe only updates BILFCF parameters.
    candidate.eval()
    dataset = make_dataset(args.data_dir, "train")
    inp, label = dataset[sample_index]
    inp, label = crop_pair(inp, label, args.crop_size, seed=args.seed)
    x = inp.unsqueeze(0).to(device)
    y = label.unsqueeze(0).to(device)
    with torch.no_grad():
        official_final = official(x)[2].detach()
        base_psnr_crop = psnr_per_sample(torch.clamp(official_final, 0, 1), y).item()
        before_psnr = psnr_per_sample(torch.clamp(candidate(x)[2], 0, 1), y).item()
    initial = [param.detach().clone() for param in trainable]
    opt = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    losses = []
    max_grad_norm = 0.0
    finite_grad = True
    for _step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        final = candidate(x)[2]
        loss = objective_loss(objective, final, official_final, y)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip_norm)
        norm_value = float(total_norm.detach().cpu()) if torch.is_tensor(total_norm) else float(total_norm)
        max_grad_norm = max(max_grad_norm, norm_value)
        finite_grad = finite_grad and math.isfinite(norm_value)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    candidate.eval()
    with torch.no_grad():
        final = torch.clamp(candidate(x)[2], 0, 1)
        after_psnr = psnr_per_sample(final, y).item()
        identity_vs_a0 = (final - torch.clamp(official_final, 0, 1)).abs().mean().item()
    update_norm = math.sqrt(sum(float((param.detach() - init).pow(2).sum().cpu()) for param, init in zip(trainable, initial)))
    loss_drop = losses[0] - losses[-1] if losses else 0.0
    row = {
        "check": objective,
        "sample_name": sample_name,
        "sample_index": sample_index,
        "base_psnr_scan_crop": base_psnr,
        "base_psnr": base_psnr_crop,
        "before_psnr": before_psnr,
        "after_psnr": after_psnr,
        "psnr_gain_vs_A0": after_psnr - base_psnr_crop,
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "loss_drop": loss_drop,
        "train_steps": args.steps,
        "trainable_param_count": sum(param.numel() for param in trainable),
        "max_grad_norm_before_clip": max_grad_norm,
        "gradient_finite": finite_grad,
        "update_norm": update_norm,
        "identity_mean_abs_vs_A0_after": identity_vs_a0,
        "pass": False,
    }
    if objective == "preservation_only":
        row["pass"] = (after_psnr - base_psnr) >= -0.01 and identity_vs_a0 <= 1e-4 and finite_grad
    elif objective == "one_image_gt_overfit":
        row["pass"] = (after_psnr - base_psnr) > 0.10 or loss_drop > 0
    elif objective == "lowfreq_positive_delta":
        row["pass"] = loss_drop > 0 and (after_psnr - base_psnr) > -0.20
    elif objective == "lowfreq_sign_flip_control":
        row["pass"] = loss_drop > 0
    return row


def run_p2(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    official = build_official(args.checkpoint, device)
    dataset = make_dataset(args.data_dir, "train")
    index, base_psnr, name = pick_hard_sample(dataset, official, device, args.crop_size, args.hard_scan_count)
    rows = [train_micro_objective(args, objective, index, name, base_psnr) for objective in (
        "preservation_only", "one_image_gt_overfit", "lowfreq_positive_delta", "lowfreq_sign_flip_control"
    )]
    pos = next(row for row in rows if row["check"] == "lowfreq_positive_delta")
    neg = next(row for row in rows if row["check"] == "lowfreq_sign_flip_control")
    rows.append({
        "check": "sign_flip_positive_vs_reverse",
        "sample_name": name,
        "positive_psnr_gain": pos["psnr_gain_vs_A0"],
        "reverse_psnr_gain": neg["psnr_gain_vs_A0"],
        "positive_loss_drop": pos["loss_drop"],
        "reverse_loss_drop": neg["loss_drop"],
        "pass": pos["loss_drop"] > 0 and pos["psnr_gain_vs_A0"] >= neg["psnr_gain_vs_A0"] - 0.05,
    })
    write_csv(out_dir / "v233_p2_loss_gradient_scale_sanity.csv", rows)
    write_csv(out_dir / "v233_p2_one_image_overfit_report.csv", [row for row in rows if row["check"] in {"one_image_gt_overfit", "lowfreq_positive_delta"}])
    write_csv(out_dir / "v233_p2_sign_flip_control.csv", [row for row in rows if "sign_flip" in row["check"] or "lowfreq" in row["check"]])
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P2 loss/gradient/scale sanity",
        "locked_test_touched": False,
        "sample_name": name,
        "sample_index": index,
        "base_psnr": base_psnr,
        "all_checks_pass": all(bool(row.get("pass")) for row in rows),
        "rows": rows,
    }
    write_json(out_dir / "v233_p2_loss_gradient_scale_closeout.json", payload)
    return payload


def normalized_lowfreq_noise(feature: torch.Tensor, seed: int, epsilon: float, kernel: int = 9) -> torch.Tensor:
    generator = torch.Generator(device=feature.device)
    generator.manual_seed(seed)
    noise = torch.randn(feature.shape, generator=generator, device=feature.device, dtype=feature.dtype)
    noise = lowpass(noise, kernel)
    scale = tensor_rms(feature.detach()).clamp_min(1e-12) * epsilon
    return noise / tensor_rms(noise).clamp_min(1e-12) * scale


def maybe_inject(feature: torch.Tensor, point: str, active_point: str, seed: int, epsilon: float) -> tuple[torch.Tensor, torch.Tensor | None]:
    if point != active_point:
        return feature, None
    delta = normalized_lowfreq_noise(feature, seed=seed, epsilon=epsilon)
    return feature + delta, delta


def forward_with_injection(model: torch.nn.Module, x: torch.Tensor, point: str, seed: int, epsilon: float) -> tuple[list[torch.Tensor], torch.Tensor]:
    x_2 = F.interpolate(x, scale_factor=0.5)
    x_4 = F.interpolate(x_2, scale_factor=0.5)
    z2 = model.SCM2(x_2)
    z4 = model.SCM1(x_4)
    outputs = []
    injected = None
    x_ = model.feat_extract[0](x)
    res1 = model.Encoder[0](x_)
    z = model.feat_extract[1](res1)
    z = model.FAM2(z, z2)
    res2 = model.Encoder[1](z)
    res2, delta = maybe_inject(res2, point, "S4_encoder_late", seed, epsilon)
    injected = delta if delta is not None else injected
    z = model.feat_extract[2](res2)
    z = model.FAM1(z, z4)
    z = model.Encoder[2](z)
    z, delta = maybe_inject(z, point, "S5_bottleneck_mid", seed, epsilon)
    injected = delta if delta is not None else injected
    z = model.Decoder[0](z)
    z, delta = maybe_inject(z, point, "S6_decoder_early", seed, epsilon)
    injected = delta if delta is not None else injected
    z_ = model.ConvsOut[0](z)
    z = model.feat_extract[3](z)
    outputs.append(z_ + x_4)
    z = torch.cat([z, res2], dim=1)
    z = model.Convs[0](z)
    z = model.Decoder[1](z)
    z, delta = maybe_inject(z, point, "decoder_mid", seed, epsilon)
    injected = delta if delta is not None else injected
    z_ = model.ConvsOut[1](z)
    z = model.feat_extract[4](z)
    outputs.append(z_ + x_2)
    z = torch.cat([z, res1], dim=1)
    z = model.Convs[1](z)
    z = model.Decoder[2](z)
    z, delta = maybe_inject(z, point, "decoder_pre_output_feature", seed, epsilon)
    injected = delta if delta is not None else injected
    z = model.feat_extract[5](z)
    outputs.append(z + x)
    if injected is None:
        raise ValueError(f"Injection point not reached: {point}")
    return outputs, injected


def run_p3(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = build_official(args.checkpoint, device)
    dataset = make_dataset(args.data_dir, "train")
    indices = list(range(min(args.sample_count, len(dataset))))
    points = ["S4_encoder_late", "S5_bottleneck_mid", "S6_decoder_early", "decoder_mid", "decoder_pre_output_feature"]
    sample_meta = []
    rows = []
    with torch.no_grad():
        for idx in indices:
            inp, label = dataset[idx]
            inp, label = crop_pair(inp, label, args.crop_size)
            x = inp.unsqueeze(0).to(device)
            y = label.unsqueeze(0).to(device)
            base = torch.clamp(model(x)[2], 0, 1)
            sample_meta.append((idx, dataset.image_list[idx], psnr_per_sample(base, y).item()))
        hard_cut = percentile([m[2] for m in sample_meta], 25) or 0.0
        easy_cut = percentile([m[2] for m in sample_meta], 75) or 0.0
        for point in points:
            for idx, name, base_psnr in sample_meta:
                inp, label = dataset[idx]
                inp, label = crop_pair(inp, label, args.crop_size)
                x = inp.unsqueeze(0).to(device)
                y = label.unsqueeze(0).to(device)
                base = torch.clamp(model(x)[2], 0, 1)
                outputs, injected = forward_with_injection(model, x, point, seed=args.seed + idx, epsilon=args.epsilon)
                pert = torch.clamp(outputs[2], 0, 1)
                out_delta = pert - base
                feature_energy = tensor_rms(injected).item()
                out_energy = tensor_rms(out_delta).item()
                low_ratio = (tensor_rms(lowpass(out_delta)) / tensor_rms(out_delta).clamp_min(1e-12)).item()
                pert_psnr = psnr_per_sample(pert, y).item()
                bucket = "hard_bottom25_by_a0" if base_psnr <= hard_cut else "easy_top25_by_a0" if base_psnr >= easy_cut else "mid_by_a0"
                rows.append({
                    "insertion_point": point,
                    "name": name,
                    "bucket": bucket,
                    "feature_delta_energy": feature_energy,
                    "output_rgb_delta_energy": out_energy,
                    "jacobian_amplification": out_energy / max(feature_energy, 1e-12),
                    "lowfreq_output_ratio": low_ratio,
                    "highfreq_output_leakage": max(0.0, 1.0 - low_ratio),
                    "base_psnr": base_psnr,
                    "perturbed_psnr": pert_psnr,
                    "psnr_delta_vs_A0": pert_psnr - base_psnr,
                })
    summary_rows = []
    for point in points:
        subset = [row for row in rows if row["insertion_point"] == point]
        easy = [row for row in subset if row["bucket"].startswith("easy")]
        hard = [row for row in subset if row["bucket"].startswith("hard")]
        summary_rows.append({
            "insertion_point": point,
            "count": len(subset),
            "feature_delta_energy": mean([float(row["feature_delta_energy"]) for row in subset]),
            "output_rgb_delta_energy": mean([float(row["output_rgb_delta_energy"]) for row in subset]),
            "psnr_drop_per_unit_feature_delta": mean([(-float(row["psnr_delta_vs_A0"])) / max(float(row["feature_delta_energy"]), 1e-12) for row in subset]),
            "lowfreq_output_ratio": mean([float(row["lowfreq_output_ratio"]) for row in subset]),
            "highfreq_output_leakage": mean([float(row["highfreq_output_leakage"]) for row in subset]),
            "easy_regression_rate": sum(1 for row in easy if float(row["psnr_delta_vs_A0"]) <= -0.05) / len(easy) if easy else None,
            "hard_response_gain_proxy": mean([float(row["psnr_delta_vs_A0"]) for row in hard]),
            "jacobian_amplification": mean([float(row["jacobian_amplification"]) for row in subset]),
        })
    write_csv(out_dir / "v233_p3_jacobian_sensitivity_by_insertion.csv", summary_rows)
    write_csv(out_dir / "v233_p3_jacobian_sensitivity_per_sample.csv", rows)
    best = min(summary_rows, key=lambda row: float(row["jacobian_amplification"] or 999999))
    s5 = next(row for row in summary_rows if row["insertion_point"] == "S5_bottleneck_mid")
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P3 Jacobian sensitivity by insertion",
        "locked_test_touched": False,
        "sample_count": len(sample_meta),
        "epsilon": args.epsilon,
        "lowest_amplification_point": best["insertion_point"],
        "s5_jacobian_amplification": s5["jacobian_amplification"],
        "summary": summary_rows,
        "s5_is_lowest_amplification": best["insertion_point"] == "S5_bottleneck_mid",
    }
    write_json(out_dir / "v233_p3_jacobian_closeout.json", payload)
    return payload


def make_p4_samples(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    official = build_official(args.checkpoint, device)
    wdmamba = load_wdmamba(args.wdmamba_repo, args.wdmamba_checkpoint, device)
    dataset = make_dataset(args.data_dir, "train")
    samples: list[dict[str, Any]] = []
    count = min(args.p4_sample_count, len(dataset))
    with torch.no_grad():
        for index in range(count):
            inp, label = dataset[index]
            inp, label = crop_pair(inp, label, args.crop_size, seed=args.seed + index)
            x = inp.unsqueeze(0).to(device)
            y = label.unsqueeze(0).to(device)
            a0 = torch.clamp(official(x)[2], 0, 1)
            teacher = infer_wdmamba(wdmamba, x)
            teacher_blend = torch.clamp(a0 + args.teacher_alpha * (teacher - a0), 0, 1)
            a0_psnr = psnr_per_sample(a0, y).item()
            teacher_psnr = psnr_per_sample(teacher, y).item()
            teacher_blend_psnr = psnr_per_sample(teacher_blend, y).item()
            samples.append({
                "index": index,
                "name": dataset.image_list[index],
                "x": x.detach().cpu(),
                "y": y.detach().cpu(),
                "a0": a0.detach().cpu(),
                "teacher": teacher.detach().cpu(),
                "teacher_blend": teacher_blend.detach().cpu(),
                "a0_psnr": a0_psnr,
                "teacher_psnr": teacher_psnr,
                "teacher_blend_psnr": teacher_blend_psnr,
                "teacher_blend_delta": teacher_blend_psnr - a0_psnr,
                "eligible": teacher_blend_psnr - a0_psnr >= args.p4_mask_min_gain,
            })
    return samples


def sample_batch(samples: list[dict[str, Any]], indices: list[int], key: str, device: torch.device) -> torch.Tensor:
    return torch.cat([samples[index][key] for index in indices], dim=0).to(device)


def p4_control_loss(control_id: str, final: torch.Tensor, y: torch.Tensor, a0: torch.Tensor, teacher_blend: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    if control_id == "gt_only_bilfcf":
        return charbonnier_loss(final, y) + 0.25 * charbonnier_loss(lowpass(final), lowpass(y)) + 0.05 * charbonnier_loss(final, a0)
    if control_id == "unmasked_teacher_distill":
        return charbonnier_loss(lowpass(final), lowpass(teacher_blend)) + 0.10 * charbonnier_loss(final, teacher_blend)
    if control_id == "teacher_benefit_masked_ll_delta":
        target = torch.where(eligible, teacher_blend, a0)
        return charbonnier_loss(lowpass(final), lowpass(target))
    if control_id == "teacher_benefit_masked_ll_delta_preservation":
        target = torch.where(eligible, teacher_blend, a0)
        return charbonnier_loss(lowpass(final), lowpass(target)) + 0.15 * charbonnier_loss(final, a0)
    raise ValueError(control_id)


def train_p4_control(args: argparse.Namespace, control_id: str, samples: list[dict[str, Any]], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    candidate = build_candidate(args.checkpoint, device)
    trainable = set_adapter_only(candidate)
    candidate.eval()
    initial = [param.detach().clone() for param in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    losses: list[float] = []
    finite_grad = True
    max_grad_norm = 0.0
    sample_count = len(samples)
    batch_size = min(args.p4_batch_size, sample_count)
    for step in range(args.steps):
        indices = [((step * batch_size) + offset) % sample_count for offset in range(batch_size)]
        x = sample_batch(samples, indices, "x", device)
        y = sample_batch(samples, indices, "y", device)
        a0 = sample_batch(samples, indices, "a0", device)
        teacher_blend = sample_batch(samples, indices, "teacher_blend", device)
        eligible = torch.tensor(
            [bool(samples[index]["eligible"]) for index in indices],
            dtype=torch.bool,
            device=device,
        ).view(-1, 1, 1, 1)
        optimizer.zero_grad(set_to_none=True)
        final = torch.clamp(candidate(x)[2], 0, 1)
        loss = p4_control_loss(control_id, final, y, a0, teacher_blend, eligible)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip_norm)
        norm_value = float(total_norm.detach().cpu()) if torch.is_tensor(total_norm) else float(total_norm)
        max_grad_norm = max(max_grad_norm, norm_value)
        finite_grad = finite_grad and math.isfinite(norm_value)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    update_norm = math.sqrt(sum(float((param.detach() - init).pow(2).sum().cpu()) for param, init in zip(trainable, initial)))
    meta = {
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "loss_drop": (losses[0] - losses[-1]) if losses else 0.0,
        "max_grad_norm_before_clip": max_grad_norm,
        "gradient_finite": finite_grad,
        "update_norm": update_norm,
        "train_steps": args.steps,
        "trainable_param_count": sum(param.numel() for param in trainable),
    }
    return candidate, meta


def summarize_p4_control(control_id: str, display_name: str, rows: list[dict[str, Any]], train_meta: dict[str, Any], eligible_coverage: float) -> dict[str, Any]:
    deltas = [float(row["delta_psnr_vs_A0"]) for row in rows]
    a0_values = [float(row["a0_psnr"]) for row in rows]
    order = sorted(range(len(rows)), key=lambda i: a0_values[i])
    bucket = max(1, len(rows) // 4)
    hard = [deltas[index] for index in order[:bucket]]
    easy = [deltas[index] for index in order[-bucket:]]
    summary = {
        "control_id": control_id,
        "control": display_name,
        "count": len(rows),
        "mean_delta": mean(deltas),
        "hard_gain": mean(hard),
        "easy_gain": mean(easy),
        "p05": percentile(deltas, 5),
        "cvar5": cvar_low(deltas),
        "severe": sum(1 for value in deltas if value <= -0.20) / len(deltas) if deltas else None,
        "strong_reference_regression_rate": sum(1 for value in easy if value <= -0.05) / len(easy) if easy else None,
        "eligible_mask_coverage": eligible_coverage,
        "loss_start": train_meta.get("loss_start"),
        "loss_end": train_meta.get("loss_end"),
        "loss_drop": train_meta.get("loss_drop"),
        "max_grad_norm_before_clip": train_meta.get("max_grad_norm_before_clip"),
        "gradient_finite": train_meta.get("gradient_finite"),
        "update_norm": train_meta.get("update_norm"),
        "train_steps": train_meta.get("train_steps", 0),
        "trainable_param_count": train_meta.get("trainable_param_count", 0),
        "gate_pass": False,
    }
    return summary


def evaluate_p4_model(control_id: str, display_name: str, model: torch.nn.Module | None, samples: list[dict[str, Any]], device: torch.device, train_meta: dict[str, Any], eligible_coverage: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            if model is None:
                pred = sample["a0"].to(device)
            else:
                pred = torch.clamp(model(x)[2], 0, 1)
            pred_psnr = psnr_per_sample(pred, y).item()
            rows.append({
                "control_id": control_id,
                "control": display_name,
                "sample_index": sample["index"],
                "sample_name": sample["name"],
                "eligible_mask": sample["eligible"],
                "a0_psnr": sample["a0_psnr"],
                "teacher_psnr": sample["teacher_psnr"],
                "teacher_blend_psnr": sample["teacher_blend_psnr"],
                "teacher_blend_delta": sample["teacher_blend_delta"],
                "pred_psnr": pred_psnr,
                "delta_psnr_vs_A0": pred_psnr - float(sample["a0_psnr"]),
            })
    return summarize_p4_control(control_id, display_name, rows, train_meta, eligible_coverage), rows


def run_p4(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    samples = make_p4_samples(args, device)
    if not samples:
        raise RuntimeError("P4 requires at least one train-derived sample")
    eligible_count = sum(1 for sample in samples if sample["eligible"])
    eligible_coverage = eligible_count / len(samples)
    controls = [
        ("a0_baseline", "A0 baseline"),
        ("gt_only_bilfcf", "GT-only BILFCF"),
        ("unmasked_teacher_distill", "unmasked teacher distill"),
        ("teacher_benefit_masked_ll_delta", "teacher-benefit-masked LL delta distill"),
        ("teacher_benefit_masked_ll_delta_preservation", "teacher-benefit-masked LL delta + preservation"),
    ]
    summary_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    baseline_summary, baseline_rows = evaluate_p4_model(
        "a0_baseline", "A0 baseline", None, samples, device,
        {"train_steps": 0, "trainable_param_count": 0, "gradient_finite": True, "update_norm": 0.0},
        eligible_coverage,
    )
    summary_rows.append(baseline_summary)
    per_image_rows.extend(baseline_rows)
    for control_id, display_name in controls[1:]:
        model, train_meta = train_p4_control(args, control_id, samples, device)
        summary, rows = evaluate_p4_model(control_id, display_name, model, samples, device, train_meta, eligible_coverage)
        summary_rows.append(summary)
        per_image_rows.extend(rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    by_id = {str(row["control_id"]): row for row in summary_rows}
    masked = by_id["teacher_benefit_masked_ll_delta_preservation"]
    unmasked = by_id["unmasked_teacher_distill"]
    mask_effect_easy = float(masked["easy_gain"] or 0.0) - float(unmasked["easy_gain"] or 0.0)
    mask_effect_p05 = float(masked["p05"] or 0.0) - float(unmasked["p05"] or 0.0)
    gate_pass = (
        float(masked["mean_delta"] or -999.0) >= 0.05
        and float(masked["hard_gain"] or -999.0) >= 0.20
        and float(masked["easy_gain"] or -999.0) >= -0.03
        and float(masked["p05"] or -999.0) >= -0.20
        and float(masked["cvar5"] or -999.0) >= -0.50
        and float(masked["severe"] or 999.0) <= 0.05
        and float(masked["strong_reference_regression_rate"] or 999.0) <= 0.05
        and eligible_coverage >= 0.15
        and max(mask_effect_easy, mask_effect_p05) > 0.0
    )
    for row in summary_rows:
        row["mask_effect_easy_vs_unmasked"] = mask_effect_easy if row["control_id"] == "teacher_benefit_masked_ll_delta_preservation" else ""
        row["mask_effect_p05_vs_unmasked"] = mask_effect_p05 if row["control_id"] == "teacher_benefit_masked_ll_delta_preservation" else ""
        if row["control_id"] == "teacher_benefit_masked_ll_delta_preservation":
            row["gate_pass"] = gate_pass
    write_csv(out_dir / "v233_p4_teacher_benefit_masked_canary32_report.csv", summary_rows)
    write_csv(out_dir / "v233_p4_teacher_benefit_masked_canary32_per_image.csv", per_image_rows)
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P4 teacher-benefit-gated micro canary32",
        "locked_test_touched": False,
        "sample_count": len(samples),
        "teacher": "wdmamba_alpha0p5" if args.teacher_alpha == 0.5 else f"wdmamba_alpha{args.teacher_alpha}",
        "teacher_alpha": args.teacher_alpha,
        "wdmamba_repo": str(args.wdmamba_repo),
        "wdmamba_checkpoint": str(args.wdmamba_checkpoint),
        "wdmamba_sha256": file_sha256(args.wdmamba_checkpoint),
        "eligible_count": eligible_count,
        "eligible_mask_coverage": eligible_coverage,
        "mask_min_gain": args.p4_mask_min_gain,
        "mask_effect_easy_vs_unmasked": mask_effect_easy,
        "mask_effect_p05_vs_unmasked": mask_effect_p05,
        "gate_pass": gate_pass,
        "canary80_oof_authorized": gate_pass,
        "canary80_oof_launched": False,
        "selected_control": "teacher_benefit_masked_ll_delta_preservation",
        "selected_control_summary": masked,
        "summary": summary_rows,
    }
    write_json(out_dir / "v233_p4_teacher_benefit_masked_canary32_closeout.json", payload)
    return payload


def run_closeout(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    p1 = json.loads((out_dir / "v233_p1_teacher_benefit_closeout.json").read_text(encoding="utf-8")) if (out_dir / "v233_p1_teacher_benefit_closeout.json").exists() else {}
    p2 = json.loads((out_dir / "v233_p2_loss_gradient_scale_closeout.json").read_text(encoding="utf-8")) if (out_dir / "v233_p2_loss_gradient_scale_closeout.json").exists() else {}
    p3 = json.loads((out_dir / "v233_p3_jacobian_closeout.json").read_text(encoding="utf-8")) if (out_dir / "v233_p3_jacobian_closeout.json").exists() else {}
    p4 = json.loads((out_dir / "v233_p4_teacher_benefit_masked_canary32_closeout.json").read_text(encoding="utf-8")) if (out_dir / "v233_p4_teacher_benefit_masked_canary32_closeout.json").exists() else {}
    p1_pass = bool(p1.get("gate_pass"))
    p2_pass = bool(p2.get("all_checks_pass"))
    p3_pass = bool(p3.get("lowest_amplification_point") and not p3.get("s5_is_lowest_amplification", False))
    p4_launched = bool(p4)
    p4_pass = bool(p4.get("gate_pass"))
    if not p1_pass:
        decision = "P1_FAIL_TEACHER_SOURCE_NOT_MASKABLY_SAFE"
    elif not p2_pass:
        decision = "P2_FAIL_LOSS_GRADIENT_SCALE_OR_CARRIER_TRAINABILITY"
    elif not p3_pass:
        decision = "P3_FAIL_OR_INCONCLUSIVE_INSERTION_SENSITIVITY"
    elif not p4_launched:
        decision = "P4_MICRO_CANARY_AUTHORIZED_NOT_LAUNCHED_BY_CLOSEOUT"
    elif not p4_pass:
        decision = "P4_FAIL_MASKED_CANARY32_NO_CANARY80"
    else:
        decision = "P4_PASS_AUTHORIZE_CANARY80_OOF_NOT_LAUNCHED"
    payload = {
        "primary_question": "Does a teacher/expert provide stable, maskable hard-haze and low-frequency benefit over ConvIR-B, and can BILFCF safely compress that benefit?",
        "closed_reference": "v2.32 P2_FAIL_BOUNDED_FIELD_TRAINABILITY_PAUSE",
        "not_a_continuation_of_failed_v232_training": True,
        "p1_gate_pass": p1_pass,
        "p2_gate_pass": p2_pass,
        "p3_gate_pass": p3_pass,
        "p4_gate_pass": p4_pass,
        "p4_canary32_authorized": p1_pass and p2_pass and p3_pass,
        "p4_canary32_launched": p4_launched,
        "p4_canary80_authorized": p4_pass,
        "canary80_oof_launched": False,
        "p2b_selector_probe_launched": False,
        "locked_test_touched": False,
        "decision": decision,
    }
    write_json(out_dir / "v233_closeout.json", payload)
    (out_dir / "v233_decision_tree.md").write_text(
        "\n".join([
            "# v2.33 Decision Tree",
            "",
            f"Decision: `{decision}`",
            "",
            f"- P1 gate pass: `{p1_pass}`",
            f"- P2 gate pass: `{p2_pass}`",
            f"- P3 gate pass: `{p3_pass}`",
            f"- P4 canary32 launched: `{p4_launched}`",
            f"- P4 gate pass: `{p4_pass}`",
            f"- canary80 OOF authorized: `{p4_pass}`",
            "- canary80 OOF launched: `False`",
            "- locked test touched: `False`",
            "",
        ]),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "\n".join([
            "# Haze4K v2.33 NoPost Teacher-Benefit Source and BILFCF Trainability Audit",
            "",
            f"State: `{decision}`",
            "",
            "Route card: `experience_docx/experiment_cards/2026-07-05-haze4k-v2-33-nopost-teacher-benefit-source-and-bilfcf-trainability-audit.md`.",
            "Central index: `experience_docx/EXPERIMENT_INDEX.md`.",
            "",
            "Primary evidence:",
            "",
            "- `v233_p0_arch_contract_delta.md`",
            "- `v233_p1_teacher_benefit_audit.csv`",
            "- `v233_p1_teacher_negative_transfer_audit.csv`",
            "- `v233_p2_loss_gradient_scale_sanity.csv`",
            "- `v233_p3_jacobian_sensitivity_by_insertion.csv`",
            "- `v233_p4_teacher_benefit_masked_canary32_report.csv`",
            "- `v233_closeout.json`",
            "",
            "Locked test was not touched.",
            "",
        ]),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["p0", "p1", "p2", "p3", "p4", "closeout"])
    parser.add_argument("--data_dir", default="/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K")
    parser.add_argument("--checkpoint", default="/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--udp_table", default="")
    parser.add_argument("--wdmamba_table", default="")
    parser.add_argument("--wdmamba_repo", default="/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba")
    parser.add_argument("--wdmamba_checkpoint", default="/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--sample_count", type=int, default=16)
    parser.add_argument("--hard_scan_count", type=int, default=32)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=0.001)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--teacher_alpha", type=float, default=0.5)
    parser.add_argument("--p4_sample_count", type=int, default=32)
    parser.add_argument("--p4_batch_size", type=int, default=4)
    parser.add_argument("--p4_mask_min_gain", type=float, default=0.05)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.phase == "p0":
        payload = run_p0(args)
    elif args.phase == "p1":
        payload = run_p1(args)
    elif args.phase == "p2":
        payload = run_p2(args)
    elif args.phase == "p3":
        payload = run_p3(args)
    elif args.phase == "p4":
        payload = run_p4(args)
    else:
        payload = run_closeout(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
