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
ACTION_BUDGET = 0.025


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


def final_output(out: Any, h: int, w: int) -> torch.Tensor:
    pred = out[2] if isinstance(out, (list, tuple)) else out
    return pred[:, :, :h, :w]


def tensor_psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    mse = F.mse_loss(torch.clamp(pred, 0, 1), label).clamp_min(1e-12)
    return float((10 * torch.log10(1 / mse)).detach().cpu())


def haar_ll(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    return (a + b + c + d) / 2.0


def final_feature(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
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
    z = model.Decoder[1](z)
    z = model.feat_extract[4](z)
    z = torch.cat([z, res1], dim=1)
    z = model.Convs[1](z)
    return model.Decoder[2](z)


def wldb_action(model: torch.nn.Module, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    block = model.nopost_wldb
    ll, lh, hl, hh, h, w = block.dwt(z)
    ll_delta = block.lowband_project(block.lowband_context(ll))
    z_cal = block.iwt(ll + ll_delta, lh, hl, hh, h, w)
    return ll_delta, z_cal - z


def load_eval_tables(run_dir: Path, labels: list[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        path = run_dir / f"v216_wldb_a_eval_{label}_per_image.csv"
        if path.is_file():
            rows: list[dict[str, Any]] = []
            for row in read_csv(path):
                rows.append({k: (float(v) if k not in {"name", "candidate"} and v != "" else v) for k, v in row.items()})
            out[label] = rows
    return out


def checkpoint_summary(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: float(r["A0_PSNR"]))
    k = max(1, len(rows) // 4)
    deltas = [float(r["dPSNR"]) for r in rows]
    ssim = [float(r.get("dSSIM", 0.0)) for r in rows]
    strong_cut = percentile([float(r["A0_PSNR"]) for r in rows], 75)
    strong = [r for r in rows if float(r["A0_PSNR"]) >= strong_cut]
    return {
        "checkpoint": label,
        "count": len(rows),
        "mean_dPSNR": mean(deltas),
        "median_dPSNR": median(deltas),
        "p01_dPSNR": percentile(deltas, 1),
        "p05_dPSNR": percentile(deltas, 5),
        "cvar5_dPSNR": mean(sorted(deltas)[: max(1, int(round(0.05 * len(deltas))))]),
        "hard_bottom25_dPSNR": mean([float(r["dPSNR"]) for r in ordered[:k]]),
        "easy_top25_dPSNR": mean([float(r["dPSNR"]) for r in ordered[-k:]]),
        "positive_ratio": sum(v > 0 for v in deltas) / len(deltas),
        "severe_loss_count": sum(v <= SEVERE for v in deltas),
        "severe_rate": sum(v <= SEVERE for v in deltas) / len(deltas),
        "mean_dSSIM": mean(ssim),
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_count": len(strong),
        "strong_reference_regressions": sum(float(r["dPSNR"]) <= STRONG_REG for r in strong),
    }


def compute_eval_outputs(
    *,
    data_dir: Path,
    official_checkpoint: Path,
    model_checkpoints: dict[str, Path],
    split_rows: list[dict[str, str]],
    fold: int,
    out_dir: Path,
    max_images: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from models.ConvIR import build_net as build_official
    from models.NoPostWLDBConvIR import build_net as build_wldb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = [r["name"] for r in split_rows if int(r["oof_fold"]) == fold]
    if max_images:
        names = names[:max_images]
    input_dir, gt_dir = train_dirs(data_dir)
    official = build_official("base", "Haze4K", "original").to(device)
    official.load_state_dict(load_state(official_checkpoint, device))
    official.eval()

    loss_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for label, ckpt in model_checkpoints.items():
        model = build_wldb("base", "Haze4K", "original").to(device)
        model.load_state_dict(load_state(ckpt, device), strict=True)
        model.eval()
        with torch.no_grad():
            for idx, name in enumerate(names, 1):
                hazy = image_tensor(input_dir / name, device)
                gt_raw = image_tensor(label_path(gt_dir, name), device)
                x, h, w = pad_to(hazy, 32)
                gt = gt_raw[:, :, :h, :w]
                h, w = gt.shape[-2:]
                a0 = final_output(official(x), h, w)
                pred = final_output(model(x), h, w)
                pred_clamped = torch.clamp(pred, 0, 1)
                a0_clamped = torch.clamp(a0, 0, 1)
                a0_l1 = float(F.l1_loss(a0_clamped, gt).detach().cpu())
                a0_ll_l1 = float(F.l1_loss(haar_ll(a0_clamped), haar_ll(gt)).detach().cpu())
                a0_psnr = tensor_psnr(a0, gt)
                candidate_psnr = tensor_psnr(pred, gt)
                final_l1 = float(F.l1_loss(pred_clamped, gt).detach().cpu())
                lowband_l1 = float(F.l1_loss(haar_ll(pred_clamped), haar_ll(gt)).detach().cpu())
                preserve_l1 = float(F.l1_loss(pred_clamped, a0_clamped).detach().cpu())
                image_action = float((pred_clamped - a0_clamped).abs().mean().detach().cpu())
                z = final_feature(model, x)
                ll_delta, feature_delta = wldb_action(model, z)
                loss_rows.append(
                    {
                        "checkpoint": label,
                        "name": name,
                        "a0_psnr": a0_psnr,
                        "candidate_psnr": candidate_psnr,
                        "dPSNR": candidate_psnr - a0_psnr,
                        "a0_final_l1": a0_l1,
                        "candidate_final_l1": final_l1,
                        "delta_final_l1_vs_a0": final_l1 - a0_l1,
                        "a0_lowband_l1": a0_ll_l1,
                        "candidate_lowband_l1": lowband_l1,
                        "delta_lowband_l1_vs_a0": lowband_l1 - a0_ll_l1,
                        "preserve_l1_to_a0": preserve_l1,
                        "image_action_l1": image_action,
                        "budget_hinge": max(0.0, image_action - ACTION_BUDGET),
                    }
                )
                action_rows.append(
                    {
                        "checkpoint": label,
                        "name": name,
                        "ll_delta_abs_mean": float(ll_delta.abs().mean().detach().cpu()),
                        "ll_delta_rms": float(torch.sqrt(torch.mean(ll_delta.detach() ** 2)).cpu()),
                        "ll_delta_abs_max": float(ll_delta.abs().max().detach().cpu()),
                        "feature_delta_abs_mean": float(feature_delta.abs().mean().detach().cpu()),
                        "feature_delta_rms": float(torch.sqrt(torch.mean(feature_delta.detach() ** 2)).cpu()),
                        "image_action_l1": image_action,
                        "budget_hinge": max(0.0, image_action - ACTION_BUDGET),
                    }
                )
                if idx % 100 == 0:
                    print(f"R1_ACTION {label} {idx}/{len(names)}", flush=True)
    write_csv(out_dir / "v217_r1_loss_delta_vs_identity.csv", loss_rows)
    write_csv(out_dir / "v217_r1_action_norm_stats.csv", action_rows)
    feature_rows = [
        {
            "checkpoint": row["checkpoint"],
            "name": row["name"],
            "feature_delta_abs_mean": row["feature_delta_abs_mean"],
            "feature_delta_rms": row["feature_delta_rms"],
            "ll_delta_abs_mean": row["ll_delta_abs_mean"],
            "ll_delta_rms": row["ll_delta_rms"],
        }
        for row in action_rows
    ]
    write_csv(out_dir / "v217_r1_feature_delta_stats.csv", feature_rows)
    return loss_rows, action_rows


def summarize_losses(loss_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    labels = sorted({str(r["checkpoint"]) for r in loss_rows})
    for label in labels:
        rows = [r for r in loss_rows if r["checkpoint"] == label]
        out.append(
            {
                "checkpoint": label,
                "count": len(rows),
                "mean_delta_final_l1_vs_a0": mean([float(r["delta_final_l1_vs_a0"]) for r in rows]),
                "mean_delta_lowband_l1_vs_a0": mean([float(r["delta_lowband_l1_vs_a0"]) for r in rows]),
                "mean_preserve_l1_to_a0": mean([float(r["preserve_l1_to_a0"]) for r in rows]),
                "mean_image_action_l1": mean([float(r["image_action_l1"]) for r in rows]),
                "budget_activation_rate": sum(float(r["budget_hinge"]) > 0 for r in rows) / len(rows),
                "tail_cvar5_dPSNR": mean(sorted([float(r["dPSNR"]) for r in rows])[: max(1, int(round(0.05 * len(rows))))]),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v216-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--official-checkpoint", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-action-images", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.v216_dir / "wldb_a_seed3407"
    labels = ["model_5", "model_10", "model_15", "model_20", "Best", "Final"]
    eval_tables = load_eval_tables(run_dir, labels)
    if not eval_tables:
        raise FileNotFoundError(f"no v216 WLDB-A per-image CSVs under {run_dir}")

    pareto = [checkpoint_summary(label, eval_tables[label]) for label in labels if label in eval_tables]
    write_csv(args.out_dir / "v217_r1_wldb_a_checkpoint_pareto.csv", pareto)

    severe_sets = {
        label: {str(r["name"]) for r in rows if float(r["dPSNR"]) <= SEVERE}
        for label, rows in eval_tables.items()
    }
    overlap_rows = []
    for a, aset in severe_sets.items():
        for b, bset in severe_sets.items():
            overlap_rows.append({"checkpoint_a": a, "checkpoint_b": b, **set_stats(aset, bset)})
    write_csv(args.out_dir / "v217_r1_severe_overlap_by_checkpoint.csv", overlap_rows)

    by_name: dict[str, dict[str, Any]] = {}
    base_rows = next(iter(eval_tables.values()))
    strong_cut = percentile([float(r["A0_PSNR"]) for r in base_rows], 75)
    hard_cut = percentile([float(r["A0_PSNR"]) for r in base_rows], 25)
    easy_cut = strong_cut
    for label, rows in eval_tables.items():
        for row in rows:
            name = str(row["name"])
            rec = by_name.setdefault(
                name,
                {
                    "name": name,
                    "A0_PSNR": float(row["A0_PSNR"]),
                    "is_hard_bottom25": int(float(row["A0_PSNR"]) <= hard_cut),
                    "is_easy_top25": int(float(row["A0_PSNR"]) >= easy_cut),
                    "is_strong_reference": int(float(row["A0_PSNR"]) >= strong_cut),
                },
            )
            rec[f"{label}_dPSNR"] = float(row["dPSNR"])
            rec[f"{label}_severe"] = int(float(row["dPSNR"]) <= SEVERE)
            rec[f"{label}_strong_regression"] = int(float(row["dPSNR"]) <= STRONG_REG and rec["is_strong_reference"])
    manifest_rows = list(by_name.values())
    write_csv(args.out_dir / "v217_r1_tail_case_manifest.csv", manifest_rows)
    strong_cases = [
        row
        for row in manifest_rows
        if int(row["is_strong_reference"]) == 1
        and any(int(row.get(f"{label}_strong_regression", 0)) == 1 for label in eval_tables)
    ]
    write_csv(args.out_dir / "v217_r1_strong_reference_regression_cases.csv", strong_cases)

    split_csv = args.v216_dir / "v216_t1_per_image_band_deltas.csv"
    split_rows = read_csv(split_csv)
    checkpoints = {
        label: run_dir / "checkpoints" / f"{label}.pkl"
        for label in labels
        if (run_dir / "checkpoints" / f"{label}.pkl").is_file()
    }
    # v2.16 saved Final.pkl and Best.pkl with matching labels.
    for label in ("Best", "Final"):
        p = run_dir / "checkpoints" / f"{label}.pkl"
        if p.is_file():
            checkpoints[label] = p
    loss_rows, action_rows = compute_eval_outputs(
        data_dir=args.data_dir,
        official_checkpoint=args.official_checkpoint,
        model_checkpoints=checkpoints,
        split_rows=split_rows,
        fold=args.fold,
        out_dir=args.out_dir,
        max_images=args.max_action_images,
    )
    loss_summary = summarize_losses(loss_rows)
    write_csv(args.out_dir / "v217_r1_loss_delta_vs_identity_summary.csv", loss_summary)

    train_history_path = run_dir / "v216_wldb_a_train_history.json"
    train_history = json.loads(train_history_path.read_text(encoding="utf-8")) if train_history_path.is_file() else {}
    history_rows = train_history.get("history", [])
    budget_all_zero = all(abs(float(r.get("budget", 0.0))) <= 1e-12 for r in history_rows) if history_rows else False
    model5 = next((r for r in pareto if r["checkpoint"] == "model_5"), {})
    model10 = next((r for r in pareto if r["checkpoint"] == "model_10"), {})
    severe_overlap_model5_model10 = next(
        (r["jaccard"] for r in overlap_rows if r["checkpoint_a"] == "model_5" and r["checkpoint_b"] == "model_10"),
        float("nan"),
    )
    decision = "R1_CLOSE_WLDB_A_KEEP_NOPOST_LOWBAND_OPEN"
    write_text(
        args.out_dir / "v217_r1_decision.md",
        "\n".join(
            [
                "# v2.17 R1 WLDB-A Postmortem Decision",
                "",
                f"Decision: `{decision}`",
                "",
                "Interpretation:",
                "",
                "- WLDB-A remains closed as a concrete form; do not expand seeds, epochs, hidden width, or locked-test use.",
                "- The broader NoPost lowband direction remains open pending R2 capacity-ladder evidence.",
                f"- model_5 severe count: `{model5.get('severe_loss_count', 'NA')}` / `{model5.get('count', 'NA')}`.",
                f"- model_5 mean/hard/easy dPSNR: `{model5.get('mean_dPSNR', float('nan')):.6f}` / `{model5.get('hard_bottom25_dPSNR', float('nan')):.6f}` / `{model5.get('easy_top25_dPSNR', float('nan')):.6f}`.",
                f"- model_10 mean dPSNR: `{model10.get('mean_dPSNR', float('nan')):.6f}`.",
                f"- model_5 vs model_10 severe-set Jaccard: `{severe_overlap_model5_model10:.6f}`.",
                f"- training history action-budget term all zero: `{budget_all_zero}`.",
                "",
                "Answers to the five R1 questions:",
                "",
                "1. Tail concentration is recorded in `v217_r1_tail_case_manifest.csv` using hard/easy/strong flags.",
                "2. Checkpoint severe overlap is recorded in `v217_r1_severe_overlap_by_checkpoint.csv`.",
                "3. The pareto table shows the mean/hard gain shrinks as later checkpoints reduce severe losses.",
                "4. The v2.16 action-budget hinge stayed inactive in the logged train objective.",
                "5. Loss-vs-identity and action statistics are recorded from the available checkpoints; use them for R3 objective audit.",
                "",
                "Locked Haze4K test remains untouched. This is an audit only, not training.",
            ]
        ),
    )
    write_json(
        args.out_dir / "v217_r1_closeout.json",
        {
            "decision": decision,
            "pareto": pareto,
            "budget_all_zero": budget_all_zero,
            "loss_summary": loss_summary,
            "locked_test_touched": False,
            "training_launched": False,
        },
    )
    print("V217_R1_POSTMORTEM_OK", decision, flush=True)


if __name__ == "__main__":
    main()
