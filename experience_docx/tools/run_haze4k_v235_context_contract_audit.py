#!/usr/bin/env python3
"""Haze4K v2.35 full-image teacher context-contract audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
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

ROUTE_ID = "haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706"
ALPHAS = (("a0p375", 0.375), ("a0p5", 0.5))
SEVERE_THRESHOLD = -0.30
STRONG_REFERENCE_REGRESSION_THRESHOLD = -0.05
INSERTION_GROUPS: dict[str, list[str]] = {
    "S6_decoder_early": ["S6_decoder_early"],
    "S4_encoder_late": ["S4_encoder_late"],
    "S4_plus_S6": ["S4_encoder_late", "S6_decoder_early"],
    "S5_plus_S6": ["S5_bottleneck_mid", "S6_decoder_early"],
    "S4_plus_S5_plus_S6": ["S4_encoder_late", "S5_bottleneck_mid", "S6_decoder_early"],
}


def fnum(value: Any, default: float | None = None) -> float | None:
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


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path = Path(path)
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


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(tuple(arr.shape)).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def save_tensor(path: str | Path, tensor: torch.Tensor, *, fp16: bool = True) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = tensor.detach().cpu()
    if fp16:
        payload = payload.to(torch.float16)
    torch.save(payload, path)
    return file_sha256(path)


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


def infer_official(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    padded, h, w = pad_to_factor(x, 32)
    out = model(padded)[2]
    return torch.clamp(out[:, :, :h, :w], 0, 1)


def infer_wdmamba(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    padded, h, w = pad_to_factor(x, 4)
    out = model.restoration_network(padded)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return torch.clamp(out[:, :, :h, :w], 0, 1)


def psnr_per_sample(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    mse = (pred - label).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def charbonnier_loss(pred: torch.Tensor, label: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - label).pow(2) + eps * eps).mean()


def lowpass(tensor: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    return F.avg_pool2d(tensor, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)


def tensor_rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.pow(2).mean().clamp_min(1e-12).sqrt()


def psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    return float(psnr_per_sample(torch.clamp(pred, 0, 1), label).item())


def blend(a0: torch.Tensor, expert: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.clamp(a0 + alpha * (expert - a0), 0, 1)


def make_dataset(data_dir: str | Path, split: str = "train") -> DeblurDataset:
    return DeblurDataset(str(Path(data_dir) / split), "Haze4K")


def dataset_index_by_name(dataset: DeblurDataset) -> dict[str, int]:
    return {name: index for index, name in enumerate(dataset.image_list)}


def label_path_for(dataset: DeblurDataset, name: str) -> Path:
    return Path(dataset._label_path(name))  # type: ignore[attr-defined]


def input_path_for(dataset: DeblurDataset, name: str) -> Path:
    return Path(dataset.input_dir) / name


def hardness_bucket(row: dict[str, Any]) -> str:
    bucket = str(row.get("selection_source_bucket") or row.get("hardness_bucket") or "")
    if bucket.startswith("hard"):
        return "hard"
    if bucket.startswith("easy") or bucket.startswith("strong"):
        return "easy"
    return "mid"


def summarize_values(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    hard = [float(row[key]) for row in rows if row.get(key) not in ("", None) and hardness_bucket(row) == "hard"]
    easy = [float(row[key]) for row in rows if row.get(key) not in ("", None) and hardness_bucket(row) == "easy"]
    mid = [float(row[key]) for row in rows if row.get(key) not in ("", None) and hardness_bucket(row) == "mid"]
    strong = easy
    severe = [value for value in values if value <= SEVERE_THRESHOLD]
    strong_regress = [value for value in strong if value < STRONG_REFERENCE_REGRESSION_THRESHOLD]
    return {
        "mean": mean(values),
        "hard": mean(hard),
        "easy": mean(easy),
        "mid": mean(mid),
        "p05": percentile(values, 5),
        "CVaR5": cvar_low(values),
        "negative_count": sum(1 for value in values if value < 0),
        "severe_count": len(severe),
        "severe_rate": len(severe) / len(values) if values else None,
        "strong_reference_count": len(strong),
        "strong_reference_regression_count": len(strong_regress),
        "strong_reference_regression_rate": len(strong_regress) / len(strong) if strong else None,
        "sample_count": len(values),
        "severe_threshold_dB": SEVERE_THRESHOLD,
        "strong_reference_regression_threshold_dB": STRONG_REFERENCE_REGRESSION_THRESHOLD,
    }


def gate_p0d(stats: dict[str, Any]) -> bool:
    return bool(
        (stats.get("mean") is not None and stats["mean"] >= 0.30)
        and (stats.get("p05") is not None and stats["p05"] >= -0.05)
        and (stats.get("CVaR5") is not None and stats["CVaR5"] >= -0.10)
        and (stats.get("severe_rate") == 0)
    )


def gate_p2(stats: dict[str, Any]) -> bool:
    return bool(
        (stats.get("mean") is not None and stats["mean"] >= 0.30)
        and (stats.get("hard") is not None and stats["hard"] >= 0.50)
        and (stats.get("easy") is not None and stats["easy"] >= -0.03)
        and (stats.get("p05") is not None and stats["p05"] >= -0.05)
        and (stats.get("CVaR5") is not None and stats["CVaR5"] >= -0.10)
        and (stats.get("severe_rate") == 0)
    )


def gate_p3(stats: dict[str, Any], positive_count: int, sample_count: int) -> bool:
    return bool(
        positive_count >= min(12, sample_count)
        and (stats.get("mean") is not None and stats["mean"] >= 0.20)
        and (stats.get("hard") is not None and stats["hard"] >= 0.50)
        and (stats.get("easy") is not None and stats["easy"] >= -0.02)
        and (stats.get("p05") is not None and stats["p05"] >= -0.03)
        and (stats.get("CVaR5") is not None and stats["CVaR5"] >= -0.05)
        and (stats.get("severe_rate") == 0)
        and (stats.get("strong_reference_regression_rate") == 0)
    )


def run_p0d(args: argparse.Namespace) -> None:
    source_rows = read_csv(args.p0c_csv)
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        out: dict[str, Any] = {
            "sample_name": row.get("sample_name", ""),
            "sample_index": row.get("sample_index", ""),
            "crop_top": row.get("crop_top", ""),
            "crop_left": row.get("crop_left", ""),
            "selection_source_bucket": row.get("selection_source_bucket", ""),
            "crop_direct_A0_PSNR": fnum(row.get("crop_direct_A0_PSNR")),
            "crop_fullslice_A0_PSNR": fnum(row.get("crop_fullslice_A0_PSNR")),
            "crop_A0_context_gap_direct_minus_fullslice": fnum(row.get("crop_A0_context_gap_direct_minus_fullslice")),
            "crop_fullslice_a0p375_dPSNR_vs_fullslice_A0": fnum(row.get("crop_fullslice_a0p375_dPSNR")),
            "crop_fullslice_a0p5_dPSNR_vs_fullslice_A0": fnum(row.get("crop_fullslice_a0p5_dPSNR")),
            "crop_direct_a0p375_dPSNR_vs_crop_direct_A0": fnum(row.get("crop_direct_a0p375_dPSNR")),
            "crop_direct_a0p5_dPSNR_vs_crop_direct_A0": fnum(row.get("crop_direct_a0p5_dPSNR")),
            "contract_label": "Contract_C_fullimage_teacher_slice_vs_crop_direct_A0",
        }
        gap = fnum(row.get("crop_A0_context_gap_direct_minus_fullslice"), 0.0) or 0.0
        for label, _alpha in ALPHAS:
            fullslice_delta = fnum(row.get(f"crop_fullslice_{label}_dPSNR"))
            out[f"rebased_fullslice_{label}_dPSNR_vs_crop_direct_A0"] = (
                fullslice_delta - gap if fullslice_delta is not None else None
            )
        rows.append(out)

    fields = [
        "sample_name",
        "sample_index",
        "crop_top",
        "crop_left",
        "selection_source_bucket",
        "crop_direct_A0_PSNR",
        "crop_fullslice_A0_PSNR",
        "crop_A0_context_gap_direct_minus_fullslice",
        "crop_fullslice_a0p375_dPSNR_vs_fullslice_A0",
        "crop_fullslice_a0p5_dPSNR_vs_fullslice_A0",
        "rebased_fullslice_a0p375_dPSNR_vs_crop_direct_A0",
        "rebased_fullslice_a0p5_dPSNR_vs_crop_direct_A0",
        "crop_direct_a0p375_dPSNR_vs_crop_direct_A0",
        "crop_direct_a0p5_dPSNR_vs_crop_direct_A0",
        "contract_label",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "v235_p0d_rebased_contract_delta_per_image.csv", rows, fields)

    summary_rows = []
    summary: dict[str, Any] = {
        "route_id": ROUTE_ID,
        "phase": "P0D rebased full-image-slice teacher contract delta",
        "source_p0c_csv": str(args.p0c_csv),
        "locked_test_touched": False,
        "contract": "full-image teacher slice target rebased against crop-direct A0 student baseline",
        "gate": {
            "mean_min_dB": 0.30,
            "p05_min_dB": -0.05,
            "CVaR5_min_dB": -0.10,
            "severe_rate_required": 0,
        },
    }
    any_pass = False
    for label, alpha in ALPHAS:
        key = f"rebased_fullslice_{label}_dPSNR_vs_crop_direct_A0"
        stats = summarize_values(rows, key)
        passed = gate_p0d(stats)
        any_pass = any_pass or passed
        row = {"alpha": alpha, "alpha_label": label, "metric_key": key, "gate_pass": passed, **stats}
        summary_rows.append(row)
        summary[label] = row
    summary["any_alpha_pass"] = any_pass
    summary["decision"] = (
        "P0D_PASS_CROP_INPUT_FULLSLICE_TARGET_CAN_CONTINUE_TO_CONTEXT_AUDIT"
        if any_pass
        else "P0D_FAIL_BLOCK_256_CROP_INPUT_FULLSLICE_TARGET"
    )
    summary["crop_input_student_fullslice_target_authorized"] = any_pass
    summary["next_authorized"] = (
        "P1/P2/P3 context audit may continue; 256-crop Contract C is eligible only for passing alpha"
        if any_pass
        else "Do not train 256 crop-input student on full-image-slice target; continue context-size/full-image contract audit"
    )
    write_csv(args.out_dir / "v235_p0d_rebased_contract_delta_summary.csv", summary_rows)
    write_json(args.out_dir / "v235_p0d_rebased_contract_delta_summary.json", summary)
    write_json(
        args.out_dir / "v235_p0d_closeout.json",
        {
            "route_id": ROUTE_ID,
            "phase": "P0D",
            "decision": summary["decision"],
            "locked_test_touched": False,
            "crop_input_student_fullslice_target_authorized": any_pass,
            "p1_fullimage_cache_audit_authorized": True,
            "p2_context_size_sweep_authorized": True,
            "p3_authorized": False,
            "p4_free_tensor_projection_authorized": False,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("V235_P0D_REBASED_CONTRACT_DELTA_OK")


def table_rows_for_dataset(table_path: Path, dataset: DeblurDataset, limit: int | None) -> list[dict[str, str]]:
    name_to_index = dataset_index_by_name(dataset)
    rows = []
    for row in read_csv(table_path):
        name = row.get("name", "")
        if name not in name_to_index:
            continue
        out = dict(row)
        out["sample_index"] = str(name_to_index[name])
        rows.append(out)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def run_p1_cache(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    dataset = make_dataset(args.data_dir, "train")
    rows = table_rows_for_dataset(args.wdmamba_table, dataset, args.limit)
    if not rows:
        raise RuntimeError("No WDMamba table rows matched Haze4K train data")
    official = build_official(args.checkpoint, device)
    wdmamba = load_wdmamba(args.wdmamba_repo, args.wdmamba_checkpoint, device)
    cache_root = args.cache_root
    manifest: list[dict[str, Any]] = []
    consistency: list[dict[str, Any]] = []

    with torch.no_grad():
        for i, rec in enumerate(rows, start=1):
            name = rec["name"]
            index = int(rec["sample_index"])
            inp, label = dataset[index]
            x = inp.unsqueeze(0).to(device)
            y = label.unsqueeze(0).to(device)
            a0 = infer_official(official, x)
            wd = infer_wdmamba(wdmamba, x)
            a0_psnr = psnr(a0, y)
            wd_psnr = psnr(wd, y)
            input_path = input_path_for(dataset, name)
            gt_path = label_path_for(dataset, name)
            image_id = Path(name).stem
            image_cache_dir = cache_root / image_id
            a0_path = image_cache_dir / "A0_full.pt"
            wd_path = image_cache_dir / "WDMamba_full.pt"
            a0_sha = save_tensor(a0_path, a0, fp16=args.cache_fp16) if args.write_cache else tensor_sha256(a0)
            wd_sha = save_tensor(wd_path, wd, fp16=args.cache_fp16) if args.write_cache else tensor_sha256(wd)
            for label_key, alpha in ALPHAS:
                blend_tensor = blend(a0, wd, alpha)
                blend_psnr = psnr(blend_tensor, y)
                blend_path = image_cache_dir / f"blend_{label_key}.pt"
                blend_sha = (
                    save_tensor(blend_path, blend_tensor, fp16=args.cache_fp16)
                    if args.write_cache
                    else tensor_sha256(blend_tensor)
                )
                table_delta = fnum(rec.get(f"expert_{label_key}_dPSNR"))
                recompute_delta = blend_psnr - a0_psnr
                diff = abs(recompute_delta - table_delta) if table_delta is not None else None
                row = {
                    "image_id": image_id,
                    "image_name": name,
                    "sample_index": index,
                    "table_split": rec.get("split", ""),
                    "input_path": str(input_path),
                    "gt_path": str(gt_path),
                    "input_sha256": file_sha256(input_path),
                    "gt_sha256": file_sha256(gt_path),
                    "A0_full_output_path": str(a0_path) if args.write_cache else "",
                    "A0_full_output_sha256": a0_sha,
                    "A0_full_PSNR": a0_psnr,
                    "WDMamba_full_output_path": str(wd_path) if args.write_cache else "",
                    "WDMamba_full_output_sha256": wd_sha,
                    "WDMamba_full_PSNR": wd_psnr,
                    "blend_alpha": alpha,
                    "blend_formula": "clamp(A0 + alpha * (WDMamba - A0), 0, 1)",
                    "blend_output_path": str(blend_path) if args.write_cache else "",
                    "blend_output_sha256": blend_sha,
                    "blend_full_PSNR": blend_psnr,
                    "table_A0_PSNR": fnum(rec.get("A0_PSNR")),
                    "table_alpha_dPSNR": table_delta,
                    "recompute_alpha_dPSNR": recompute_delta,
                    "table_vs_recompute_abs_diff": diff,
                    "locked_test_touched": False,
                }
                manifest.append(row)
                consistency.append({
                    "image_name": name,
                    "table_split": rec.get("split", ""),
                    "blend_alpha": alpha,
                    "table_A0_PSNR": row["table_A0_PSNR"],
                    "recompute_A0_PSNR": a0_psnr,
                    "table_alpha_dPSNR": table_delta,
                    "recompute_alpha_dPSNR": recompute_delta,
                    "table_vs_recompute_abs_diff": diff,
                })
            print(f"p1_cache_progress {i}/{len(rows)} {name}", flush=True)

    fields = [
        "image_id",
        "image_name",
        "sample_index",
        "table_split",
        "input_path",
        "gt_path",
        "input_sha256",
        "gt_sha256",
        "A0_full_output_path",
        "A0_full_output_sha256",
        "A0_full_PSNR",
        "WDMamba_full_output_path",
        "WDMamba_full_output_sha256",
        "WDMamba_full_PSNR",
        "blend_alpha",
        "blend_formula",
        "blend_output_path",
        "blend_output_sha256",
        "blend_full_PSNR",
        "table_A0_PSNR",
        "table_alpha_dPSNR",
        "recompute_alpha_dPSNR",
        "table_vs_recompute_abs_diff",
        "locked_test_touched",
    ]
    write_csv(args.out_dir / "v235_p1_fullimage_teacher_cache_manifest.csv", manifest, fields)
    write_csv(args.out_dir / "v235_p1_table_vs_recompute_consistency.csv", consistency)
    diffs = [float(row["table_vs_recompute_abs_diff"]) for row in consistency if row.get("table_vs_recompute_abs_diff") not in ("", None)]
    missing_sha = [
        row for row in manifest
        if not row.get("input_sha256") or not row.get("gt_sha256")
        or not row.get("A0_full_output_sha256") or not row.get("WDMamba_full_output_sha256")
        or not row.get("blend_output_sha256")
    ]
    summary = {
        "route_id": ROUTE_ID,
        "phase": "P1 full-image teacher cache manifest",
        "locked_test_touched": False,
        "image_count": len(rows),
        "manifest_rows": len(manifest),
        "cache_written": bool(args.write_cache),
        "cache_root": str(cache_root),
        "cache_fp16": bool(args.cache_fp16),
        "cache_coverage": 1.0 if not missing_sha else (len(manifest) - len(missing_sha)) / len(manifest),
        "missing_sha_count": len(missing_sha),
        "table_vs_recompute_mean_abs_diff": mean(diffs),
        "table_vs_recompute_max_abs_diff": max(diffs) if diffs else None,
        "gate_pass": (
            len(missing_sha) == 0
            and bool(diffs)
            and (mean(diffs) or 0.0) <= args.table_epsilon
            and all(not row.get("locked_test_touched") for row in manifest)
        ),
        "table_epsilon": args.table_epsilon,
    }
    summary["decision"] = "P1_PASS_FULLIMAGE_CACHE_CONTRACT" if summary["gate_pass"] else "P1_FAIL_FULLIMAGE_CACHE_CONTRACT"
    write_json(args.out_dir / "v235_p1_fullimage_teacher_cache_summary.json", summary)
    write_json(
        args.out_dir / "v235_p1_closeout.json",
        {
            "route_id": ROUTE_ID,
            "phase": "P1",
            "decision": summary["decision"],
            "locked_test_touched": False,
            "p2_context_size_sweep_authorized": True,
            "p3_authorized": False,
            "p4_free_tensor_projection_authorized": False,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("V235_P1_FULLIMAGE_TEACHER_CACHE_OK")


def context_origin(h: int, w: int, crop_top: int, crop_left: int, loss_size: int, context_size: int) -> tuple[int, int] | None:
    if context_size < loss_size:
        return None
    if h < context_size or w < context_size:
        return None
    center_y = crop_top + loss_size // 2
    center_x = crop_left + loss_size // 2
    top = min(max(0, center_y - context_size // 2), h - context_size)
    left = min(max(0, center_x - context_size // 2), w - context_size)
    if crop_top < top or crop_left < left:
        return None
    if crop_top + loss_size > top + context_size or crop_left + loss_size > left + context_size:
        return None
    return top, left


def run_p2_context(args: argparse.Namespace) -> None:
    p0c_rows = read_csv(args.p0c_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    dataset = make_dataset(args.data_dir, "train")
    official = build_official(args.checkpoint, device)
    context_tokens = [token.strip() for token in args.contexts.split(",") if token.strip()]
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for i, rec in enumerate(p0c_rows, start=1):
            index = int(rec["sample_index"])
            name = rec["sample_name"]
            crop_top = int(rec["crop_top"])
            crop_left = int(rec["crop_left"])
            inp, label = dataset[index]
            _, h, w = inp.shape
            y_full = label.unsqueeze(0).to(device)
            y_crop = y_full[:, :, crop_top:crop_top + args.loss_crop_size, crop_left:crop_left + args.loss_crop_size]
            for token in context_tokens:
                if token == "full_image_slice":
                    base_psnr = fnum(rec.get("crop_fullslice_A0_PSNR"))
                    a0_gap_vs_crop256 = base_psnr - (fnum(rec.get("crop_direct_A0_PSNR")) or 0.0) if base_psnr is not None else None
                    a0_gap_vs_fullimage = 0.0
                    available = True
                else:
                    context_size = int(token)
                    if context_size == args.loss_crop_size:
                        base_psnr = fnum(rec.get("crop_direct_A0_PSNR"))
                        a0_gap_vs_crop256 = 0.0
                        fullslice = fnum(rec.get("crop_fullslice_A0_PSNR"))
                        a0_gap_vs_fullimage = base_psnr - fullslice if base_psnr is not None and fullslice is not None else None
                        available = True
                    else:
                        origin = context_origin(h, w, crop_top, crop_left, args.loss_crop_size, context_size)
                        if origin is None:
                            base_psnr = None
                            a0_gap_vs_crop256 = None
                            a0_gap_vs_fullimage = None
                            available = False
                        else:
                            top, left = origin
                            x_ctx = inp[:, top:top + context_size, left:left + context_size].unsqueeze(0).to(device)
                            a0_ctx = infer_official(official, x_ctx)
                            rel_top = crop_top - top
                            rel_left = crop_left - left
                            a0_slice = a0_ctx[:, :, rel_top:rel_top + args.loss_crop_size, rel_left:rel_left + args.loss_crop_size]
                            base_psnr = psnr(a0_slice, y_crop)
                            crop256 = fnum(rec.get("crop_direct_A0_PSNR"))
                            fullslice = fnum(rec.get("crop_fullslice_A0_PSNR"))
                            a0_gap_vs_crop256 = base_psnr - crop256 if crop256 is not None else None
                            a0_gap_vs_fullimage = base_psnr - fullslice if fullslice is not None else None
                            available = True
                for label_key, alpha in ALPHAS:
                    teacher_delta_fullslice = fnum(rec.get(f"crop_fullslice_{label_key}_dPSNR"))
                    fullslice_base = fnum(rec.get("crop_fullslice_A0_PSNR"))
                    teacher_psnr = (
                        fullslice_base + teacher_delta_fullslice
                        if fullslice_base is not None and teacher_delta_fullslice is not None
                        else None
                    )
                    delta_vs_context = teacher_psnr - base_psnr if teacher_psnr is not None and base_psnr is not None else None
                    rows.append({
                        "sample_name": name,
                        "sample_index": index,
                        "crop_top": crop_top,
                        "crop_left": crop_left,
                        "selection_source_bucket": rec.get("selection_source_bucket", ""),
                        "context_size": token,
                        "context_available": available,
                        "loss_crop_size": args.loss_crop_size,
                        "A0_context_PSNR": base_psnr,
                        "teacher_fullslice_PSNR": teacher_psnr,
                        "teacher_delta_vs_A0_context": delta_vs_context,
                        "A0_context_gap_vs_crop256": a0_gap_vs_crop256,
                        "A0_context_gap_vs_fullimage": a0_gap_vs_fullimage,
                        "alpha": alpha,
                        "alpha_label": label_key,
                        "p05": "",
                        "CVaR5": "",
                        "negative": delta_vs_context is not None and delta_vs_context < 0,
                    })
            print(f"p2_context_progress {i}/{len(p0c_rows)} {name}", flush=True)
    write_csv(args.out_dir / "v235_p2_context_size_sweep_per_image.csv", rows)
    summary_rows = []
    summary: dict[str, Any] = {
        "route_id": ROUTE_ID,
        "phase": "P2 context-size sweep",
        "source_p0c_csv": str(args.p0c_csv),
        "locked_test_touched": False,
        "contexts": context_tokens,
        "loss_crop_size": args.loss_crop_size,
    }
    any_pass = False
    for token in context_tokens:
        for label_key, alpha in ALPHAS:
            subset = [
                row for row in rows
                if row["context_size"] == token
                and row["alpha_label"] == label_key
                and row.get("teacher_delta_vs_A0_context") not in ("", None)
            ]
            stats = summarize_values(subset, "teacher_delta_vs_A0_context")
            passed = gate_p2(stats)
            any_pass = any_pass or passed
            item = {
                "context_size": token,
                "alpha": alpha,
                "alpha_label": label_key,
                "gate_pass": passed,
                **stats,
            }
            summary_rows.append(item)
            summary[f"{token}_{label_key}"] = item
    summary["any_context_pass"] = any_pass
    passing = [row for row in summary_rows if row["gate_pass"]]
    if passing:
        passing.sort(key=lambda row: (row.get("mean") or -999.0), reverse=True)
        summary["best_passing_contract"] = {
            "context_size": passing[0]["context_size"],
            "alpha": passing[0]["alpha"],
            "alpha_label": passing[0]["alpha_label"],
            "mean": passing[0]["mean"],
        }
        decision = "P2_PASS_AT_LEAST_ONE_CONTEXT_CONTRACT"
    else:
        summary["best_passing_contract"] = None
        decision = "P2_FAIL_NO_CONTEXT_CONTRACT"
    summary["decision"] = decision
    write_csv(args.out_dir / "v235_p2_context_size_sweep_summary.csv", summary_rows)
    write_json(args.out_dir / "v235_p2_context_size_sweep_summary.json", summary)
    write_json(
        args.out_dir / "v235_p2_closeout.json",
        {
            "route_id": ROUTE_ID,
            "phase": "P2",
            "decision": decision,
            "locked_test_touched": False,
            "p3_authorized": any_pass,
            "p4_free_tensor_projection_authorized": False,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("V235_P2_CONTEXT_SIZE_SWEEP_OK")


def run_p3_substrate(args: argparse.Namespace) -> None:
    rows = read_csv(args.p2_csv)
    summary = json.loads(args.p2_summary.read_text(encoding="utf-8"))
    contract = summary.get("best_passing_contract")
    if args.context_size and args.alpha_label:
        contract = {"context_size": args.context_size, "alpha_label": args.alpha_label, "alpha": args.alpha}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not contract:
        payload = {
            "route_id": ROUTE_ID,
            "phase": "P3 same-contract positive-substrate manifest",
            "decision": "P3_SKIPPED_NO_P2_PASSING_CONTEXT",
            "locked_test_touched": False,
            "p4_free_tensor_projection_authorized": False,
        }
        write_csv(args.out_dir / "v235_p3_same_contract_positive_substrate_manifest.csv", [])
        write_json(args.out_dir / "v235_p3_same_contract_positive_substrate_summary.json", payload)
        write_json(args.out_dir / "v235_p3_closeout.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        print("V235_P3_SKIPPED_NO_PASSING_CONTEXT_OK")
        return
    context_size = str(contract["context_size"])
    alpha_label = str(contract["alpha_label"])
    alpha = fnum(contract.get("alpha"))
    selected = [
        row for row in rows
        if row.get("context_size") == context_size
        and row.get("alpha_label") == alpha_label
        and row.get("teacher_delta_vs_A0_context") not in ("", None)
    ]
    manifest: list[dict[str, Any]] = []
    for row in selected:
        delta = fnum(row.get("teacher_delta_vs_A0_context"), 0.0) or 0.0
        eligible = delta > 0
        if eligible:
            mask_reason = "positive_same_contract_teacher_delta"
        elif delta <= SEVERE_THRESHOLD:
            mask_reason = "severe_regression_same_contract_teacher_delta"
        else:
            mask_reason = "nonpositive_same_contract_teacher_delta"
        manifest.append({
            "sample_name": row.get("sample_name", ""),
            "sample_index": row.get("sample_index", ""),
            "teacher_mode": "full_image_output_slice",
            "student_context": context_size,
            "baseline_context": context_size,
            "alpha": alpha,
            "alpha_label": alpha_label,
            "teacher_delta_vs_same_contract_A0": delta,
            "eligible": eligible,
            "mask_reason": mask_reason,
            "hardness_bucket": hardness_bucket(row),
            "strong_reference_bucket": "strong_reference" if hardness_bucket(row) == "easy" else "not_strong",
        })
    stats = summarize_values(manifest, "teacher_delta_vs_same_contract_A0")
    positive_count = sum(1 for row in manifest if row["eligible"])
    passed = gate_p3(stats, positive_count, len(manifest))
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P3 same-contract positive-substrate manifest",
        "context_size": context_size,
        "alpha": alpha,
        "alpha_label": alpha_label,
        "locked_test_touched": False,
        "positive_teacher_count": positive_count,
        "sample_count": len(manifest),
        "gate_pass": passed,
        "decision": "P3_PASS_SAME_CONTRACT_POSITIVE_SUBSTRATE" if passed else "P3_FAIL_SAME_CONTRACT_POSITIVE_SUBSTRATE",
        "p4_free_tensor_projection_authorized": passed,
        **stats,
    }
    write_csv(args.out_dir / "v235_p3_same_contract_positive_substrate_manifest.csv", manifest)
    write_json(args.out_dir / "v235_p3_same_contract_positive_substrate_summary.json", payload)
    write_json(args.out_dir / "v235_p3_closeout.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V235_P3_SAME_CONTRACT_SUBSTRATE_OK")


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


def prepare_p4_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    p3 = json.loads(args.p3_summary.read_text(encoding="utf-8"))
    context_size = str(p3.get("context_size", ""))
    if context_size != "full_image_slice":
        raise RuntimeError(f"P4 currently supports full_image_slice same-contract only, got {context_size}")
    alpha = float(p3["alpha"])
    p0c_rows = read_csv(args.p0c_csv)
    dataset = make_dataset(args.data_dir, "train")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    official = build_official(args.checkpoint, device)
    wdmamba = load_wdmamba(args.wdmamba_repo, args.wdmamba_checkpoint, device)
    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for rec in p0c_rows:
            index = int(rec["sample_index"])
            name = rec["sample_name"]
            crop_top = int(rec["crop_top"])
            crop_left = int(rec["crop_left"])
            inp, label = dataset[index]
            x_full = inp.unsqueeze(0).to(device)
            y_full = label.unsqueeze(0).to(device)
            y_crop = y_full[:, :, crop_top:crop_top + args.loss_crop_size, crop_left:crop_left + args.loss_crop_size]
            a0_full = infer_official(official, x_full)
            wd_full = infer_wdmamba(wdmamba, x_full)
            teacher_full = blend(a0_full, wd_full, alpha)
            a0_slice = a0_full[:, :, crop_top:crop_top + args.loss_crop_size, crop_left:crop_left + args.loss_crop_size]
            teacher_slice = teacher_full[:, :, crop_top:crop_top + args.loss_crop_size, crop_left:crop_left + args.loss_crop_size]
            x_padded, _, _ = pad_to_factor(x_full, 32)
            base_psnr = psnr(a0_slice, y_crop)
            teacher_psnr = psnr(teacher_slice, y_crop)
            bucket = hardness_bucket(rec)
            samples.append({
                "sample_name": name,
                "sample_index": index,
                "crop_top": crop_top,
                "crop_left": crop_left,
                "x": x_padded.detach().cpu(),
                "y_crop": y_crop.detach().cpu(),
                "a0_slice": a0_slice.detach().cpu(),
                "teacher_slice": teacher_slice.detach().cpu(),
                "base_psnr_A0": base_psnr,
                "teacher_psnr": teacher_psnr,
                "teacher_delta_vs_A0": teacher_psnr - base_psnr,
                "hardness_bucket": bucket,
                "strong_reference_bucket": "strong_reference" if bucket == "easy" else "not_strong",
            })
            print(f"p4_prepare_sample {len(samples)}/{len(p0c_rows)} {name}", flush=True)
    del official
    del wdmamba
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return samples


def optimize_projection_sample(
    model: torch.nn.Module,
    sample: dict[str, Any],
    insertion_point: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    x = sample["x"].to(device)
    y_crop = sample["y_crop"].to(device)
    a0_slice = sample["a0_slice"].to(device)
    teacher_slice = sample["teacher_slice"].to(device)
    crop_top = int(sample["crop_top"])
    crop_left = int(sample["crop_left"])
    active_points = INSERTION_GROUPS[insertion_point]
    shapes: dict[str, tuple[int, ...]] = {}
    with torch.no_grad():
        forward_with_deltas(model, x, capture_shapes=shapes)
    deltas = {point: torch.zeros(shapes[point], device=device, requires_grad=True) for point in active_points}
    optimizer = torch.optim.AdamW(list(deltas.values()), lr=args.learning_rate, weight_decay=0.0)
    losses: list[float] = []
    finite_grad = True
    max_grad_norm = 0.0
    for _step in range(args.projection_steps):
        optimizer.zero_grad(set_to_none=True)
        outputs, _used = forward_with_deltas(model, x, deltas=deltas)
        pred_full = torch.clamp(outputs[2], 0, 1)
        pred = pred_full[:, :, crop_top:crop_top + args.loss_crop_size, crop_left:crop_left + args.loss_crop_size]
        loss = projection_loss(pred, teacher_slice, deltas, args.energy_weight)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(list(deltas.values()), args.grad_clip_norm)
        norm_value = float(total_norm.detach().cpu()) if torch.is_tensor(total_norm) else float(total_norm)
        max_grad_norm = max(max_grad_norm, norm_value)
        finite_grad = finite_grad and math.isfinite(norm_value)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        outputs, _used = forward_with_deltas(model, x, deltas=deltas)
        pred_full = torch.clamp(outputs[2], 0, 1)
        pred = pred_full[:, :, crop_top:crop_top + args.loss_crop_size, crop_left:crop_left + args.loss_crop_size]
        out_delta = pred - a0_slice
        feature_energy = sum(tensor_rms(delta.detach()).item() for delta in deltas.values())
        output_energy = tensor_rms(out_delta).item()
        low_ratio = (tensor_rms(lowpass(out_delta)) / tensor_rms(out_delta).clamp_min(1e-12)).item()
        base_psnr = float(sample["base_psnr_A0"])
        free_psnr = psnr(pred, y_crop)
        teacher_delta = float(sample["teacher_delta_vs_A0"])
        free_delta = free_psnr - base_psnr
    return {
        "insertion_point": insertion_point,
        "sample_name": sample["sample_name"],
        "sample_index": sample["sample_index"],
        "crop_top": crop_top,
        "crop_left": crop_left,
        "student_context": "full_image_slice",
        "baseline_context": "full_image_slice",
        "hardness_bucket": sample["hardness_bucket"],
        "strong_reference_bucket": sample["strong_reference_bucket"],
        "base_psnr_A0": sample["base_psnr_A0"],
        "teacher_psnr": sample["teacher_psnr"],
        "same_contract_teacher_delta": teacher_delta,
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


def summarize_p4_projection(point: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["free_tensor_delta"]) for row in rows]
    teacher = [float(row["same_contract_teacher_delta"]) for row in rows]
    hard = [float(row["free_tensor_delta"]) for row in rows if row["hardness_bucket"] == "hard"]
    easy = [float(row["free_tensor_delta"]) for row in rows if row["hardness_bucket"] == "easy"]
    strong = [float(row["free_tensor_delta"]) for row in rows if row["strong_reference_bucket"] == "strong_reference"]
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
        "same_contract_teacher_mean_delta": teacher_mean,
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


def run_p4_projection(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    samples = prepare_p4_samples(args)
    model = build_official(args.checkpoint, device)
    per_image_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for point in INSERTION_GROUPS:
        point_rows = []
        for index, sample in enumerate(samples, start=1):
            row = optimize_projection_sample(model, sample, point, args, device)
            point_rows.append(row)
            per_image_rows.append(row)
            print(f"p4_projection_progress {point} {index}/{len(samples)} {sample['sample_name']}", flush=True)
        summary_rows.append(summarize_p4_projection(point, point_rows))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(args.out_dir / "v235_p4_same_contract_free_tensor_projection_per_image.csv", per_image_rows)
    write_csv(args.out_dir / "v235_p4_same_contract_free_tensor_projection_by_insertion.csv", summary_rows)
    passing = [row for row in summary_rows if row["gate_pass"]]
    best = max(
        summary_rows,
        key=lambda row: float(row["projection_ratio_vs_teacher"] if row["projection_ratio_vs_teacher"] is not None else -999.0),
    )
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P4 same-contract free-tensor projection",
        "locked_test_touched": False,
        "student_context": "full_image_slice",
        "baseline_context": "full_image_slice",
        "teacher_mode": "full_image_output_slice",
        "projection_steps": args.projection_steps,
        "gate_pass": bool(passing),
        "passing_insertion_points": [row["insertion_point"] for row in passing],
        "best_insertion_point": best["insertion_point"],
        "best_projection_ratio_vs_teacher": best["projection_ratio_vs_teacher"],
        "next_generator_or_bridge_design_authorized": bool(passing),
        "canary80_authorized": False,
        "locked_test_authorized": False,
        "summary": summary_rows,
    }
    payload["decision"] = (
        "P4_PASS_SAME_CONTRACT_FREE_TENSOR_PROJECTION"
        if passing
        else "P4_FAIL_SAME_CONTRACT_FREE_TENSOR_PROJECTION"
    )
    write_json(args.out_dir / "v235_p4_closeout.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V235_P4_SAME_CONTRACT_FREE_TENSOR_PROJECTION_OK")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)

    p0d = sub.add_parser("p0d")
    add_common(p0d)
    p0d.add_argument("--p0c-csv", type=Path, required=True)

    p1 = sub.add_parser("p1-cache")
    add_common(p1)
    p1.add_argument("--data-dir", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K"))
    p1.add_argument("--checkpoint", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl"))
    p1.add_argument("--wdmamba-repo", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba"))
    p1.add_argument("--wdmamba-checkpoint", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth"))
    p1.add_argument("--wdmamba-table", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/v22_c8_1_wdmamba_full_per_image.csv"))
    p1.add_argument("--cache-root", type=Path, required=True)
    p1.add_argument("--device", default="cuda")
    p1.add_argument("--limit", type=int, default=600)
    p1.add_argument("--write-cache", action="store_true")
    p1.add_argument("--cache-fp16", action="store_true", default=True)
    p1.add_argument("--table-epsilon", type=float, default=1e-4)

    p2 = sub.add_parser("p2-context")
    add_common(p2)
    p2.add_argument("--p0c-csv", type=Path, required=True)
    p2.add_argument("--data-dir", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K"))
    p2.add_argument("--checkpoint", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl"))
    p2.add_argument("--device", default="cuda")
    p2.add_argument("--contexts", default="256,384,512,768,full_image_slice")
    p2.add_argument("--loss-crop-size", type=int, default=256)

    p3 = sub.add_parser("p3-substrate")
    add_common(p3)
    p3.add_argument("--p2-csv", type=Path, required=True)
    p3.add_argument("--p2-summary", type=Path, required=True)
    p3.add_argument("--context-size", default="")
    p3.add_argument("--alpha-label", default="")
    p3.add_argument("--alpha", type=float, default=None)

    p4 = sub.add_parser("p4-projection")
    add_common(p4)
    p4.add_argument("--p0c-csv", type=Path, required=True)
    p4.add_argument("--p3-summary", type=Path, required=True)
    p4.add_argument("--data-dir", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K"))
    p4.add_argument("--checkpoint", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl"))
    p4.add_argument("--wdmamba-repo", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba"))
    p4.add_argument("--wdmamba-checkpoint", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth"))
    p4.add_argument("--device", default="cuda")
    p4.add_argument("--loss-crop-size", type=int, default=256)
    p4.add_argument("--projection-steps", type=int, default=32)
    p4.add_argument("--learning-rate", type=float, default=0.03)
    p4.add_argument("--energy-weight", type=float, default=0.0001)
    p4.add_argument("--grad-clip-norm", type=float, default=1.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.phase == "p0d":
        run_p0d(args)
    elif args.phase == "p1-cache":
        run_p1_cache(args)
    elif args.phase == "p2-context":
        run_p2_context(args)
    elif args.phase == "p3-substrate":
        run_p3_substrate(args)
    elif args.phase == "p4-projection":
        run_p4_projection(args)
    else:
        raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    main()
