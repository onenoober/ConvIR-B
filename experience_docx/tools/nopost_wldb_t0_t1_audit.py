#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim
import torchvision.transforms.functional as TVF


TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)


SEVERE_RISK = -0.20
LOWBAND_NEED = 0.20
SHRINK_ALPHAS = (0.125, 0.25, 0.375, 0.50)
GROUP_ORDER = (
    "G1_v215_hazy_runtime_top100",
    "G2_v215_all_runtime_top100",
    "G3_v215_lost_severe",
    "G4_v215_gained_false_positive",
    "G5_all_wd0375_severe_risk",
    "G6_a0_hard_bottom25",
    "G7_a0_easy_top25",
)


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


def pad_to(x: torch.Tensor, factor: int = 32) -> tuple[torch.Tensor, int, int, int, int]:
    _, _, h, w = x.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(x, (0, pw, 0, ph), "reflect"), h, w, h + ph, w + pw


def tensor_psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    mse = F.mse_loss(pred, label).clamp_min(1e-12)
    return float((10 * torch.log10(1 / mse)).item())


def metric(pred: torch.Tensor, label: torch.Tensor, hp: int, wp: int) -> tuple[float, float]:
    pred = torch.clamp(pred, 0, 1)
    psnr = tensor_psnr(pred, label)
    down = max(1, round(min(hp, wp) / 256))
    ss = ssim(
        F.adaptive_avg_pool2d(pred, (int(hp / down), int(wp / down))),
        F.adaptive_avg_pool2d(label, (int(hp / down), int(wp / down))),
        data_range=1,
        size_average=False,
    ).mean().item()
    return psnr, float(ss)


def load_state(path: Path, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def build_official(device: torch.device):
    from models.ConvIR import build_net

    return build_net("base", "Haze4K", "original").to(device)


def load_official_checkpoint(model, checkpoint: Path, device: torch.device) -> None:
    model.load_state_dict(load_state(checkpoint, device))


def infer_final(model, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    out = model(x)
    pred = out[2] if isinstance(out, (list, tuple)) else out
    return torch.clamp(pred[:, :, :h, :w], 0, 1)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["name"]): row for row in rows}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[idx])


def bool_int(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes"} else 0
    return int(bool(value))


def top_names(rows: list[dict[str, Any]], score_col: str, k: int = 100) -> set[str]:
    ordered = sorted(rows, key=lambda r: float(r[score_col]), reverse=True)
    return {str(r["name"]) for r in ordered[:k]}


def named_set(rows: list[dict[str, Any]]) -> set[str]:
    return {str(r["name"]) for r in rows}


def set_stats(a: set[str], b: set[str]) -> dict[str, Any]:
    inter = len(a & b)
    union = len(a | b)
    return {
        "count_a": len(a),
        "count_b": len(b),
        "intersection": inter,
        "union": union,
        "jaccard": inter / union if union else float("nan"),
        "a_in_b_ratio": inter / len(a) if a else float("nan"),
        "b_in_a_ratio": inter / len(b) if b else float("nan"),
    }


def summarize_numeric(rows: list[dict[str, Any]], col: str) -> dict[str, Any]:
    vals = [float(r[col]) for r in rows if col in r and str(r[col]) != ""]
    return {
        "count": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "p05": quantile(vals, 0.05),
        "p25": quantile(vals, 0.25),
        "p75": quantile(vals, 0.75),
        "p95": quantile(vals, 0.95),
    }


def summarize_delta(rows: list[dict[str, Any]], col: str) -> dict[str, Any]:
    vals = [float(r[col]) for r in rows]
    return {
        "count": len(vals),
        f"{col}_mean": mean(vals),
        f"{col}_median": median(vals),
        f"{col}_p05": quantile(vals, 0.05),
        f"{col}_p95": quantile(vals, 0.95),
        f"{col}_positive_ratio": sum(v > 0 for v in vals) / len(vals) if vals else float("nan"),
        f"{col}_severe_loss_count": sum(v <= SEVERE_RISK for v in vals),
    }


def even_pad(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    ph = h % 2
    pw = w % 2
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


def haar_dwt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    x, h, w = even_pad(x)
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


def psnr_only(pred: torch.Tensor, label: torch.Tensor) -> float:
    return tensor_psnr(torch.clamp(pred, 0, 1), label)


def band_metrics(a0: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    a_ll, a_lh, a_hl, a_hh, h, w = haar_dwt(a0)
    g_ll, g_lh, g_hl, g_hh, _, _ = haar_dwt(gt)
    ll_oracle = haar_iwt(g_ll, a_lh, a_hl, a_hh, h, w)
    hf_oracle = haar_iwt(a_ll, g_lh, g_hl, g_hh, h, w)
    full_oracle = haar_iwt(g_ll, g_lh, g_hl, g_hh, h, w)
    base = psnr_only(a0, gt)
    out = {
        "A0_PSNR_runtime": base,
        "LL_oracle_PSNR": psnr_only(ll_oracle, gt),
        "HF_oracle_PSNR": psnr_only(hf_oracle, gt),
        "LL_HF_oracle_PSNR": psnr_only(full_oracle, gt),
        "a0_ll_l1_to_gt": float((a_ll - g_ll).abs().mean().detach().cpu()),
        "a0_hf_l1_to_gt": float(
            ((a_lh - g_lh).abs().mean() + (a_hl - g_hl).abs().mean() + (a_hh - g_hh).abs().mean()).detach().cpu()
            / 3.0
        ),
    }
    out["LL_oracle_dPSNR"] = out["LL_oracle_PSNR"] - base
    out["HF_oracle_dPSNR"] = out["HF_oracle_PSNR"] - base
    out["LL_HF_oracle_dPSNR"] = out["LL_HF_oracle_PSNR"] - base
    for alpha in SHRINK_ALPHAS:
        ll = a_ll + alpha * (g_ll - a_ll)
        shrink = haar_iwt(ll, a_lh, a_hl, a_hh, h, w)
        key = f"LL_shrink_{str(alpha).replace('.', 'p')}_dPSNR"
        out[key] = psnr_only(shrink, gt) - base
    return out


def build_t1_rows(
    *,
    data_dir: Path,
    checkpoint: Path,
    names: list[str],
    feature_by_name: dict[str, dict[str, Any]],
    device: torch.device,
    print_freq: int,
) -> list[dict[str, Any]]:
    model = build_official(device)
    load_official_checkpoint(model, checkpoint, device)
    model.eval()
    input_dir, gt_dir = train_dirs(data_dir)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, name in enumerate(names, 1):
            hazy = image_tensor(input_dir / name, device)
            gt = image_tensor(label_path(gt_dir, name), device)
            x, h, w, hp, wp = pad_to(hazy, 32)
            a0 = infer_final(model, x, h, w)
            a0_psnr, a0_ssim = metric(a0, gt, hp, wp)
            row: dict[str, Any] = {
                "name": name,
                "A0_PSNR_runtime": a0_psnr,
                "A0_SSIM_runtime": a0_ssim,
                "hazy_PSNR_runtime": tensor_psnr(hazy[:, :, :h, :w], gt),
            }
            if name in feature_by_name:
                src = feature_by_name[name]
                row.update(
                    {
                        "source_split": src.get("source_split", "train_core"),
                        "oof_fold": src.get("oof_fold", ""),
                        "WD0375_dPSNR": float(src["WD0375_dPSNR"]),
                        "WD0375_severe_risk": int(float(src["WD0375_dPSNR"]) <= SEVERE_RISK),
                    }
                )
                if "A0_PSNR" in src and str(src["A0_PSNR"]) != "":
                    row["A0_PSNR_feature_table"] = float(src["A0_PSNR"])
            row.update(band_metrics(a0, gt))
            row["lowband_need_label"] = int(float(row["LL_oracle_dPSNR"]) >= LOWBAND_NEED)
            rows.append(row)
            if idx % print_freq == 0:
                print(f"T1_WAVELET_AUDIT {idx}/{len(names)}", flush=True)
    return rows


def group_summary(rows: list[dict[str, Any]], groups: dict[str, set[str]]) -> list[dict[str, Any]]:
    by_name = rows_by_name(rows)
    out: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        names = groups[group]
        present = [by_name[n] for n in sorted(names) if n in by_name]
        row: dict[str, Any] = {"group": group, "manifest_count": len(names), "present_count": len(present)}
        for col in ("LL_oracle_dPSNR", "HF_oracle_dPSNR", "LL_HF_oracle_dPSNR"):
            row.update(summarize_delta(present, col) if present else {f"{col}_mean": float("nan")})
        row["lowband_need_count"] = sum(int(r.get("lowband_need_label", 0)) for r in present)
        row["wd0375_severe_count"] = sum(int(r.get("WD0375_severe_risk", 0)) for r in present)
        out.append(row)
    return out


def make_groups(
    feature_rows: list[dict[str, Any]],
    v215_predictions: list[dict[str, Any]],
    v215_top100: list[dict[str, Any]],
    lost: list[dict[str, Any]],
    gained: list[dict[str, Any]],
) -> dict[str, set[str]]:
    feature_sorted_a0 = sorted(feature_rows, key=lambda r: float(r["A0_PSNR"]))
    k = max(1, len(feature_sorted_a0) // 4)
    return {
        "G1_v215_hazy_runtime_top100": top_names(v215_predictions, "B0_hazy_runtime_v214_pred", 100),
        "G2_v215_all_runtime_top100": {str(r["name"]) for r in v215_top100 if bool_int(r.get("in_all_runtime_top100", 0))},
        "G3_v215_lost_severe": named_set(lost),
        "G4_v215_gained_false_positive": named_set(gained),
        "G5_all_wd0375_severe_risk": {str(r["name"]) for r in feature_rows if float(r["WD0375_dPSNR"]) <= SEVERE_RISK},
        "G6_a0_hard_bottom25": {str(r["name"]) for r in feature_sorted_a0[:k]},
        "G7_a0_easy_top25": {str(r["name"]) for r in feature_sorted_a0[-k:]},
    }


def build_t0_outputs(
    *,
    out_dir: Path,
    feature_rows: list[dict[str, Any]],
    t1_rows: list[dict[str, Any]],
    groups: dict[str, set[str]],
) -> dict[str, Any]:
    feature_by = rows_by_name(feature_rows)
    t1_by = rows_by_name(t1_rows)
    wd_severe = groups["G5_all_wd0375_severe_risk"]
    a0_hard = groups["G6_a0_hard_bottom25"]
    a0_easy = groups["G7_a0_easy_top25"]
    low_need = {name for name, row in t1_by.items() if int(row.get("lowband_need_label", 0)) == 1}
    sets = {
        "wd0375_severe_risk": wd_severe,
        "a0_hard_bottom25": a0_hard,
        "a0_easy_top25": a0_easy,
        "lowband_need_ll_oracle_ge_0p20": low_need,
        **groups,
    }
    overlap_rows = []
    for a_name, a_set in sets.items():
        for b_name, b_set in sets.items():
            stats = set_stats(a_set, b_set)
            overlap_rows.append({"set_a": a_name, "set_b": b_name, **stats})
    write_csv(out_dir / "v216_t0_target_overlap_matrix.csv", overlap_rows)

    risk_vs_a0 = []
    for label, names in (("wd0375_severe", wd_severe), ("wd0375_nonsevere", set(feature_by) - wd_severe)):
        present = [feature_by[n] for n in names if n in feature_by and "A0_PSNR" in feature_by[n]]
        row = {"group": label}
        row.update(summarize_numeric(present, "A0_PSNR"))
        risk_vs_a0.append(row)
    write_csv(out_dir / "v216_t0_wd0375_risk_vs_a0_weakness.csv", risk_vs_a0)

    risk_vs_low = []
    for label, names in (("wd0375_severe", wd_severe), ("wd0375_nonsevere", set(t1_by) - wd_severe)):
        present = [t1_by[n] for n in names if n in t1_by]
        row = {"group": label, "count": len(present)}
        row.update(summarize_delta(present, "LL_oracle_dPSNR") if present else {})
        row["lowband_need_count"] = sum(int(r.get("lowband_need_label", 0)) for r in present)
        row["lowband_need_rate"] = row["lowband_need_count"] / len(present) if present else float("nan")
        risk_vs_low.append(row)
    write_csv(out_dir / "v216_t0_wd0375_risk_vs_lowband_need.csv", risk_vs_low)

    manifest = {
        "groups": {key: sorted(value) for key, value in groups.items()},
        "derived_sets": {
            "wd0375_severe_risk": sorted(wd_severe),
            "a0_hard_bottom25": sorted(a0_hard),
            "a0_easy_top25": sorted(a0_easy),
            "lowband_need_ll_oracle_ge_0p20": sorted(low_need),
        },
        "thresholds": {
            "wd0375_severe_risk": SEVERE_RISK,
            "lowband_need_ll_oracle_dpsnr": LOWBAND_NEED,
        },
        "locked_test_touched": False,
        "training_launched": False,
    }
    write_json(out_dir / "v216_t0_stress_group_manifest.json", manifest)

    wd_low = set_stats(wd_severe, low_need)
    wd_a0 = set_stats(wd_severe, a0_hard)
    decision = (
        "T0_TARGET_DECOUPLED"
        if (wd_low["jaccard"] < 0.50 and wd_a0["jaccard"] < 0.50)
        else "T0_TARGET_OVERLAP_HIGH_REVIEW_BEFORE_TRAINING"
    )
    write_text(
        out_dir / "v216_t0_decision.md",
        "\n".join(
            [
                "# v2.16 T0 Decision",
                "",
                f"Decision: `{decision}`",
                "",
                f"- WD0375 severe vs lowband-need Jaccard: `{wd_low['jaccard']:.6f}`",
                f"- WD0375 severe vs A0 hard-bottom25 Jaccard: `{wd_a0['jaccard']:.6f}`",
                "- Locked Haze4K test: untouched.",
                "- Training launched: false.",
            ]
        ),
    )
    return {"decision": decision, "wd_severe_lowband_jaccard": wd_low["jaccard"], "wd_severe_a0_hard_jaccard": wd_a0["jaccard"]}


def build_t1_outputs(*, out_dir: Path, t1_rows: list[dict[str, Any]], groups: dict[str, set[str]]) -> dict[str, Any]:
    write_csv(out_dir / "v216_t1_per_image_band_deltas.csv", t1_rows)
    manifest = {
        "transform": "one-level Haar DWT on RGB A0 prediction and GT, train-derived images only",
        "ll_oracle": "replace A0 LL band with GT LL band, keep A0 high-frequency bands",
        "hf_oracle": "keep A0 LL band, replace high-frequency bands with GT bands",
        "ll_hf_oracle": "replace all bands with GT bands; used only as a ceiling sanity check",
        "shrink_alphas": list(SHRINK_ALPHAS),
        "lowband_need_label": f"LL_oracle_dPSNR >= {LOWBAND_NEED}",
        "locked_test_touched": False,
        "training_launched": False,
    }
    write_json(out_dir / "v216_t1_wavelet_band_manifest.json", manifest)
    for col, filename in (
        ("LL_oracle_dPSNR", "v216_t1_ll_oracle_summary.csv"),
        ("HF_oracle_dPSNR", "v216_t1_hf_oracle_summary.csv"),
        ("LL_HF_oracle_dPSNR", "v216_t1_ll_hf_oracle_summary.csv"),
    ):
        write_csv(out_dir / filename, [{"scope": "all_train_core", **summarize_delta(t1_rows, col)}])
    report = group_summary(t1_rows, groups)
    write_csv(out_dir / "v216_t1_group_report.csv", report)

    ordered_ll = sorted(t1_rows, key=lambda r: float(r["LL_oracle_dPSNR"]), reverse=True)
    visual_lines = [
        "# v2.16 T1 Visual Grid Index",
        "",
        "No image artifacts are committed by default. This index records train-derived exemplars for optional cloud-only visual inspection.",
        "",
        "## Top LL Oracle Headroom",
    ]
    for row in ordered_ll[:24]:
        visual_lines.append(f"- `{row['name']}` LL dPSNR `{float(row['LL_oracle_dPSNR']):.6f}`")
    visual_lines.extend(["", "## Lowest LL Oracle Headroom"])
    for row in ordered_ll[-24:]:
        visual_lines.append(f"- `{row['name']}` LL dPSNR `{float(row['LL_oracle_dPSNR']):.6f}`")
    write_text(out_dir / "v216_t1_visual_grid_index.md", "\n".join(visual_lines))

    all_summary = summarize_delta(t1_rows, "LL_oracle_dPSNR")
    hard_row = next((r for r in report if r["group"] == "G6_a0_hard_bottom25"), {})
    easy_row = next((r for r in report if r["group"] == "G7_a0_easy_top25"), {})
    ll_mean = float(all_summary["LL_oracle_dPSNR_mean"])
    hard_mean = float(hard_row.get("LL_oracle_dPSNR_mean", float("nan")))
    easy_mean = float(easy_row.get("LL_oracle_dPSNR_mean", float("nan")))
    severe_losses = int(all_summary["LL_oracle_dPSNR_severe_loss_count"])
    low_need_rate = sum(int(r.get("lowband_need_label", 0)) for r in t1_rows) / len(t1_rows)
    passed = (
        ll_mean >= 0.20
        and hard_mean >= 0.30
        and easy_mean >= -0.05
        and severe_losses <= max(3, int(0.01 * len(t1_rows)))
        and low_need_rate >= 0.20
    )
    decision = "T1_LOWBAND_HEADROOM_PASS_ALLOW_T2" if passed else "T1_LOWBAND_HEADROOM_FAIL_STOP_NO_T2_NO_TRAINING"
    write_text(
        out_dir / "v216_t1_decision.md",
        "\n".join(
            [
                "# v2.16 T1 Decision",
                "",
                f"Decision: `{decision}`",
                "",
                "Gate criteria:",
                "",
                f"- all-image LL oracle mean dPSNR >= `0.20`: `{ll_mean:.6f}`",
                f"- A0 hard-bottom25 LL oracle mean dPSNR >= `0.30`: `{hard_mean:.6f}`",
                f"- A0 easy-top25 LL oracle mean dPSNR >= `-0.05`: `{easy_mean:.6f}`",
                f"- severe LL-oracle regressions <= `max(3, 1%)`: `{severe_losses}`",
                f"- lowband-need rate >= `0.20`: `{low_need_rate:.6f}`",
                "",
                "Locked Haze4K test remains untouched. No training was launched.",
            ]
        ),
    )
    return {
        "decision": decision,
        "ll_mean_dpsnr": ll_mean,
        "hard_ll_mean_dpsnr": hard_mean,
        "easy_ll_mean_dpsnr": easy_mean,
        "ll_severe_loss_count": severe_losses,
        "lowband_need_rate": low_need_rate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--feature-table", type=Path, required=True)
    ap.add_argument("--v215-predictions", type=Path, required=True)
    ap.add_argument("--v215-top100", type=Path, required=True)
    ap.add_argument("--v215-lost-severe", type=Path, required=True)
    ap.add_argument("--v215-gained-fp", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--print-freq", type=int, default=50)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = read_rows(args.feature_table)
    v215_predictions = read_rows(args.v215_predictions)
    v215_top100 = read_rows(args.v215_top100)
    lost = read_rows(args.v215_lost_severe)
    gained = read_rows(args.v215_gained_fp)
    groups = make_groups(feature_rows, v215_predictions, v215_top100, lost, gained)
    names = [str(row["name"]) for row in feature_rows]
    if args.max_images:
        keep = set(names[: args.max_images])
        names = names[: args.max_images]
        groups = {key: {name for name in value if name in keep} for key, value in groups.items()}

    protocol = [
        "# v2.16 T0/T1 Protocol",
        "",
        "Route: `codex/haze4k-v2-16-nopost-wavelet-lowband-decoder`",
        "",
        "This is a no-training, no-locked-test diagnostic for the NoPost-WLDB route.",
        "T0 and T1 are executed in one cloud script because T0 risk-vs-lowband decoupling needs the T1 LL-oracle labels.",
        "",
        "Forbidden: A0/WD0375/teacher/expert outputs as model forward inputs, output-output deltas as deployable features, learned RGB post-correction, locked Haze4K.",
        "",
        "Allowed here: train-derived A0 prediction and GT are used only for oracle/headroom measurement, not as a deployable forward contract.",
    ]
    write_text(args.out_dir / "v216_t0_protocol.md", "\n".join(protocol))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"T0_T1_DEVICE {device}", flush=True)
    t1_rows = build_t1_rows(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        names=names,
        feature_by_name=rows_by_name(feature_rows),
        device=device,
        print_freq=args.print_freq,
    )
    feature_by = rows_by_name(feature_rows)
    for row in t1_rows:
        if row["name"] in feature_by and "A0_PSNR" in feature_by[row["name"]]:
            row["A0_PSNR_feature_table"] = float(feature_by[row["name"]]["A0_PSNR"])

    t1_decision = build_t1_outputs(out_dir=args.out_dir, t1_rows=t1_rows, groups=groups)
    t0_decision = build_t0_outputs(out_dir=args.out_dir, feature_rows=feature_rows, t1_rows=t1_rows, groups=groups)
    closeout = {
        "route": "haze4k-v2-16-nopost-wavelet-lowband-decoder",
        "rows": len(t1_rows),
        "t0": t0_decision,
        "t1": t1_decision,
        "locked_test_touched": False,
        "training_launched": False,
        "next_action": "IMPLEMENT_T2_CONTRACT_IDENTITY_ONLY" if "PASS_ALLOW_T2" in t1_decision["decision"] else "STOP_NO_T2_NO_TRAINING",
    }
    write_json(args.out_dir / "v216_t0_t1_closeout.json", closeout)
    print("V216_T0_T1_AUDIT_OK", json.dumps(closeout, sort_keys=True))


if __name__ == "__main__":
    main()
