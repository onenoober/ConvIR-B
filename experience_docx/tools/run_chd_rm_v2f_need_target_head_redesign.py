import argparse
import csv
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
for path in (str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

V2E_TOOL = REPO_ROOT / "experience_docx" / "tools" / "run_chd_rm_v2e_d7c_control_recall_audit.py"
spec_v2e = importlib.util.spec_from_file_location("chdrm_v2e_tool", V2E_TOOL)
v2e = importlib.util.module_from_spec(spec_v2e)
spec_v2e.loader.exec_module(v2e)


ROUTE_ID = "haze4k_v5_chd_rm_v2f_need_target_head_redesign_20260709"
DEFAULT_V2E_THRESHOLD = 0.5773006677627563
DEFAULT_TARGET_COVERAGE = 0.3026953125
SAFETY_FALSE_GLOBAL = 0.01
SAFETY_FALSE_P90 = 0.05
SAFETY_FALSE_P95 = 0.10


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def finite_values(values):
    return [float(v) for v in values if not math.isnan(float(v))]


def finite_mean(values):
    vals = finite_values(values)
    return statistics.mean(vals) if vals else math.nan


def finite_quantile(values, q):
    vals = sorted(finite_values(values))
    if not vals:
        return math.nan
    idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * q)))
    return float(vals[idx])


def pool_flat(x, size):
    if x.shape[-2:] != (size, size):
        x = F.adaptive_avg_pool2d(x, (size, size))
    return x.detach().flatten().float().cpu().numpy().astype(np.float32, copy=False)


def pool_map(x, size):
    if x.shape[-2:] != (size, size):
        x = F.adaptive_avg_pool2d(x, (size, size))
    return x.detach().squeeze().float().cpu().numpy().astype(np.float32, copy=False)


def local_max_2d(arr, radius):
    x = torch.as_tensor(arr[None, None], dtype=torch.float32)
    y = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y.squeeze().numpy()


def gradient_mag(arr):
    gy = np.zeros_like(arr, dtype=np.float32)
    gx = np.zeros_like(arr, dtype=np.float32)
    gy[1:-1] = 0.5 * (arr[2:] - arr[:-2])
    gx[:, 1:-1] = 0.5 * (arr[:, 2:] - arr[:, :-2])
    return np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)


def connected_components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            n = 0
            while q:
                cy, cx = q.popleft()
                n += 1
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
            sizes.append(n)
    return sizes


def density_bin_edges(density_values, bins):
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(density_values.astype(np.float64), qs).astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges


def assign_bins(values, edges):
    return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, len(edges) - 2)


def fit_cdf_tables(raw_need, density, bins):
    edges = density_bin_edges(density, bins)
    bin_id = assign_bins(density, edges)
    q_grid = np.linspace(0.0, 1.0, 1001)
    tables = []
    excess_tables = []
    for b in range(bins):
        vals = raw_need[bin_id == b].astype(np.float64)
        if vals.size < 10:
            vals = raw_need.astype(np.float64)
        mean = float(np.mean(vals))
        raw_q = np.quantile(vals, q_grid)
        excess = vals - mean
        excess_q = np.quantile(excess, q_grid)
        tables.append({"bin": b, "count": int(vals.size), "raw_need_mean": mean, "raw_need_q": raw_q})
        excess_tables.append({"bin": b, "excess_q": excess_q})
    return {"edges": edges, "q_grid": q_grid, "tables": tables, "excess_tables": excess_tables}


def apply_conditional_cdf(raw_need, density, fit):
    out = np.zeros_like(raw_need, dtype=np.float32)
    bin_id = assign_bins(density, fit["edges"])
    q_grid = fit["q_grid"]
    for table in fit["tables"]:
        b = table["bin"]
        mask = bin_id == b
        if int(mask.sum()):
            out[mask] = np.interp(raw_need[mask], table["raw_need_q"], q_grid).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def apply_excess_cdf(raw_need, density, fit):
    out = np.zeros_like(raw_need, dtype=np.float32)
    bin_id = assign_bins(density, fit["edges"])
    q_grid = fit["q_grid"]
    for table, excess_table in zip(fit["tables"], fit["excess_tables"]):
        b = table["bin"]
        mask = bin_id == b
        if int(mask.sum()):
            excess = raw_need[mask] - float(table["raw_need_mean"])
            out[mask] = np.interp(excess, excess_table["excess_q"], q_grid).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def summarize_binary_scores(y_true, score, threshold=None):
    y_true = y_true.astype(bool)
    if threshold is None:
        pos_rate = float(y_true.mean())
        threshold = float(np.quantile(score, max(0.0, min(1.0, 1.0 - pos_rate))))
    pred = score >= threshold
    return {
        "threshold": float(threshold),
        "positive_prevalence": float(y_true.mean()),
        "coverage": float(pred.mean()),
        "precision": float((pred & y_true).sum() / max(int(pred.sum()), 1)),
        "recall": float((pred & y_true).sum() / max(int(y_true.sum()), 1)),
        "false_positive_rate": float((pred & (~y_true)).sum() / max(int((~y_true).sum()), 1)),
        "auroc": v2e.v2d.v2b.auroc(y_true, score),
        "auprc": v2e.v2d.v2b.auprc(y_true, score),
    }


def load_git_text(path):
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def collect_maps(model, density_head, heads, names, device, target_info, density_stats, args, split_name):
    dataset = v2e.Haze4KPairDataset(names, args.data_dir, max_items=0, seed=args.seed)
    records = []
    arrays = {
        "raw_need": [],
        "target": [],
        "density": [],
        "density_pred": [],
    }
    for radius in args.stability_blur_radii:
        arrays[f"target_blur{radius}"] = []
    for key in heads:
        arrays[key] = []
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = v2e.v2d.v2b.v2.pad32(hazy)
            a0, context = v2e.d7c.convir_a0_context(model, density_head, padded)
            a0 = a0[:, :, :h, :w]
            context = context[:, :, :h, :w]
            raw_need = v2e.v2d.v2b.v2.raw_need(a0, gt, args.blur_kernel)
            target = v2e.v2d.v2b.make_target(raw_need, target_info, "quantile")
            density_target = v2e.v2d.v2b.v2.normalize(
                v2e.v2d.v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                density_stats["density"]["raw_p1"],
                density_stats["density"]["raw_p99"],
            )
            density_pred = context[:, -1:]
            rec = {
                "name": name,
                "raw_need_map": pool_map(raw_need, args.map_grid),
                "target_map": pool_map(target, args.map_grid),
                "density_map": pool_map(density_target, args.map_grid),
                "density_pred_map": pool_map(density_pred, args.map_grid),
                "pred_maps": {},
                "target_blur_maps": {},
            }
            for radius in args.stability_blur_radii:
                rn = v2e.v2d.v2b.v2.raw_need(a0, gt, int(radius))
                tb = v2e.v2d.v2b.make_target(rn, target_info, "quantile")
                rec["target_blur_maps"][int(radius)] = pool_map(tb, args.map_grid)
                arrays[f"target_blur{radius}"].append(rec["target_blur_maps"][int(radius)].reshape(-1))
            for key, head in heads.items():
                pred, _ = v2e.d7c.predict_head(head, context)
                rec["pred_maps"][key] = pool_map(pred, args.map_grid)
                arrays[key].append(rec["pred_maps"][key].reshape(-1))
            arrays["raw_need"].append(rec["raw_need_map"].reshape(-1))
            arrays["target"].append(rec["target_map"].reshape(-1))
            arrays["density"].append(rec["density_map"].reshape(-1))
            arrays["density_pred"].append(rec["density_pred_map"].reshape(-1))
            records.append(rec)
            if (idx + 1) % args.progress_every == 0:
                print(f"collect_maps_{split_name} {idx + 1}/{len(dataset)}", flush=True)
    flat = {key: np.concatenate(vals).astype(np.float32, copy=False) for key, vals in arrays.items()}
    return {"records": records, "arrays": flat}


def flat_records_for_v2e(pack):
    rows = []
    for rec in pack["records"]:
        rows.append({"name": rec["name"], "target": rec["target_map"].reshape(-1), "density": rec["density_map"].reshape(-1)})
    return rows


def write_f0(output_dir, args, train_pack, val_pack, target_info, density_stats):
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    density_q33 = float(density_stats["density"]["q33"])
    val_records = flat_records_for_v2e(val_pack)
    rows = []
    for variant in ["d7c_mc_topk_hn_ordinal", "d7c_mc_hn_ordinal"]:
        row = v2e.fixed_threshold_summary(
            variant if variant == "d7c_mc_topk_hn_ordinal" else "d7c_mc_hn_ordinal_at_topk_threshold",
            val_pack["arrays"]["target"],
            val_pack["arrays"][variant],
            val_pack["arrays"]["density"],
            val_records,
            q33,
            q66,
            density_q33,
            args.candidate_threshold,
        )
        rows.append(row)
    train_density_score = train_pack["arrays"]["density_pred"]
    density_threshold = v2e.matched_threshold(train_density_score, args.target_coverage)
    density_row = v2e.fixed_threshold_summary(
        "density_only_matched_threshold",
        val_pack["arrays"]["target"],
        val_pack["arrays"]["density_pred"],
        val_pack["arrays"]["density"],
        val_records,
        q33,
        q66,
        density_q33,
        density_threshold,
    )
    density_row["selected_threshold"] = density_threshold
    density_row["target_coverage"] = args.target_coverage
    rows.append(density_row)
    write_csv(output_dir / "v2e_d7c_candidate_reproduction.csv", rows)

    curve_audit_rows = []
    for variant in ["d7c_mc_topk_hn_ordinal", "d7c_mc_hn_ordinal"]:
        train_score = train_pack["arrays"][variant]
        curve = v2e.build_curve(
            variant,
            train_score,
            val_pack["arrays"]["target"],
            val_pack["arrays"][variant],
            val_pack["arrays"]["density"],
            val_records,
            q33,
            q66,
            density_q33,
            args,
        )
        safe = [
            r for r in curve
            if r["false_global"] <= SAFETY_FALSE_GLOBAL
            and r["false_per_image_p90"] <= SAFETY_FALSE_P90
            and r["false_per_image_p95"] <= SAFETY_FALSE_P95
        ]
        ldhn = [r for r in curve if r["ldhn_recall"] >= 0.10]
        both = [
            r for r in curve
            if r["ldhn_recall"] >= 0.10
            and r["false_global"] <= SAFETY_FALSE_GLOBAL
            and r["false_per_image_p90"] <= SAFETY_FALSE_P90
            and r["false_per_image_p95"] <= SAFETY_FALSE_P95
        ]
        best_safe = max(safe, key=lambda r: r["ldhn_recall"]) if safe else {}
        first_ldhn = min(ldhn, key=lambda r: (r["false_per_image_p95"], r["false_per_image_p90"])) if ldhn else {}
        curve_audit_rows.append(
            {
                "variant": variant,
                "safe_points": len(safe),
                "ldhn_passing_points": len(ldhn),
                "safe_and_ldhn_points": len(both),
                "best_safe_ldhn_recall": best_safe.get("ldhn_recall", math.nan),
                "best_safe_false_p90": best_safe.get("false_per_image_p90", math.nan),
                "best_safe_false_p95": best_safe.get("false_per_image_p95", math.nan),
                "min_false_ldhn_recall": first_ldhn.get("ldhn_recall", math.nan),
                "min_false_ldhn_false_p90": first_ldhn.get("false_per_image_p90", math.nan),
                "min_false_ldhn_false_p95": first_ldhn.get("false_per_image_p95", math.nan),
            }
        )
    write_csv(output_dir / "v2e_d7c_rp_reproduction.csv", curve_audit_rows)
    write_json(
        output_dir / "v2f_source_of_truth_manifest.json",
        {
            "route_id": ROUTE_ID,
            "source_branch": "codex/haze4k-v5-v2e-chd-rm-d7c-control-recall-audit",
            "source_commit": args.source_commit,
            "data_dir": args.data_dir,
            "split_json": args.split_json,
            "checkpoint": args.checkpoint,
            "density_artifact": args.density_artifact,
            "d7c_topk_artifact": args.d7c_topk_artifact,
            "d7c_hn_artifact": args.d7c_hn_artifact,
            "candidate_threshold": args.candidate_threshold,
            "locked_haze4k_test_usage": "none",
            "forbidden_not_used": ["D2", "RARM connection", "RARM training", "v3 runtime", "locked Haze4K test"],
        },
    )
    write_json(
        output_dir / "a0_equivalence_audit.json",
        {
            "status": "PASS_BY_CONSTRUCTION",
            "a0_output_changed": False,
            "reason": "v2f first-stage diagnostics use frozen A0, frozen D3, and frozen/side probe heads only.",
            "locked_haze4k_test_usage": "none",
        },
    )
    write_json(
        output_dir / "no_locked_test_audit.json",
        {
            "locked_haze4k_test_usage": "none",
            "data_dir": args.data_dir,
            "split_json": args.split_json,
            "forbidden": ["D2", "RARM connection", "RARM training", "v3", "locked Haze4K test"],
        },
    )


def write_f1(output_dir, val_pack, target_info, density_stats, args):
    q20 = float(target_info["quantile"]["q20"])
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    q80 = float(target_info["quantile"]["q80"])
    density_q33 = float(density_stats["density"]["q33"])
    density_q66 = float(density_stats["density"]["q66"])
    all_grad = np.concatenate([gradient_mag(rec["density_map"]).reshape(-1) for rec in val_pack["records"]])
    grad_p90 = float(np.quantile(all_grad, 0.90))
    support_rows = []
    comp_rows = []
    near_rows = []
    per_image_rows = []
    stability_rows = []
    atlas_lines = [
        "# LDHN Failure Case Text Atlas",
        "",
        "Sorted by unstable/boundary support and D7c top-k miss rate.",
        "",
    ]
    total = {k: 0 for k in ["ldhn", "core", "boundary", "adjacent", "isolated", "unstable", "pixels"]}
    for rec in val_pack["records"]:
        target = rec["target_map"]
        density = rec["density_map"]
        pred = rec["pred_maps"]["d7c_mc_topk_hn_ordinal"]
        blur_targets = list(rec["target_blur_maps"].values())
        stable66 = np.logical_and.reduce([tb >= q66 for tb in blur_targets])
        stable80 = np.logical_and.reduce([tb >= q80 for tb in blur_targets])
        density_low = density <= density_q33
        ldhn = density_low & (target >= q66)
        ldhn80 = density_low & (target >= q80)
        boundary_band = density_low & (target >= q66) & (target < q80)
        unstable = ldhn & (~stable66)
        local_high_density = local_max_2d((density >= density_q66).astype(np.float32), args.near_haze_radius) > 0
        grad = gradient_mag(density)
        adjacent = ldhn & (local_high_density | (grad >= grad_p90))
        core = ldhn80 & stable66
        boundary = ldhn & (boundary_band | unstable)
        isolated = ldhn & (~adjacent)
        missed = ldhn & (pred < args.candidate_threshold)
        masks = {
            "ldhn": ldhn,
            "ldhn_core": core,
            "ldhn_boundary": boundary,
            "ldhn_adjacent_to_haze": adjacent,
            "ldhn_isolated": isolated,
            "ldhn_unstable": unstable,
        }
        n_pix = int(target.size)
        row = {"name": rec["name"], "pixels": n_pix}
        for key, mask in masks.items():
            row[f"{key}_pixels"] = int(mask.sum())
            row[f"{key}_coverage"] = float(mask.mean())
            row[f"{key}_d7c_recall"] = float((pred[mask] >= args.candidate_threshold).mean()) if int(mask.sum()) else math.nan
        row["ldhn_miss_rate"] = float(missed.sum() / max(int(ldhn.sum()), 1))
        per_image_rows.append(row)
        total["pixels"] += n_pix
        total["ldhn"] += int(ldhn.sum())
        total["core"] += int(core.sum())
        total["boundary"] += int(boundary.sum())
        total["adjacent"] += int(adjacent.sum())
        total["isolated"] += int(isolated.sum())
        total["unstable"] += int(unstable.sum())
        for radius, tb in rec["target_blur_maps"].items():
            high = density_low & (tb >= q66)
            stability_rows.append(
                {
                    "name": rec["name"],
                    "blur_radius": radius,
                    "ldhn_q66_coverage": float(high.mean()),
                    "ldhn_q80_coverage": float((density_low & (tb >= q80)).mean()),
                }
            )
        sizes = connected_components(ldhn)
        comp_rows.append(
            {
                "name": rec["name"],
                "ldhn_components": len(sizes),
                "ldhn_largest_component": max(sizes) if sizes else 0,
                "ldhn_component_p50": finite_quantile(sizes, 0.50),
                "ldhn_component_p90": finite_quantile(sizes, 0.90),
            }
        )
        near_rows.append(
            {
                "name": rec["name"],
                "ldhn_pixels": int(ldhn.sum()),
                "adjacent_pixels": int(adjacent.sum()),
                "isolated_pixels": int(isolated.sum()),
                "density_gradient_mean_on_ldhn": float(np.mean(grad[ldhn])) if int(ldhn.sum()) else math.nan,
                "density_gradient_p90_on_ldhn": finite_quantile(grad[ldhn].reshape(-1), 0.90) if int(ldhn.sum()) else math.nan,
            }
        )
    per_image_rows_sorted = sorted(per_image_rows, key=lambda r: (r["ldhn_boundary_coverage"], r["ldhn_miss_rate"]), reverse=True)
    for row in per_image_rows_sorted[:25]:
        atlas_lines.append(
            f"- {row['name']}: ldhn={row['ldhn_coverage']:.4f}, core={row['ldhn_core_coverage']:.4f}, "
            f"boundary={row['ldhn_boundary_coverage']:.4f}, isolated={row['ldhn_isolated_coverage']:.4f}, "
            f"d7c_ldhn_recall={row['ldhn_d7c_recall']:.4f}"
        )
    summary = {
        "images": len(val_pack["records"]),
        "pixels": total["pixels"],
        "ldhn_pixel_coverage": total["ldhn"] / max(total["pixels"], 1),
        "ldhn_core_fraction_of_ldhn": total["core"] / max(total["ldhn"], 1),
        "ldhn_boundary_fraction_of_ldhn": total["boundary"] / max(total["ldhn"], 1),
        "ldhn_adjacent_fraction_of_ldhn": total["adjacent"] / max(total["ldhn"], 1),
        "ldhn_isolated_fraction_of_ldhn": total["isolated"] / max(total["ldhn"], 1),
        "ldhn_unstable_fraction_of_ldhn": total["unstable"] / max(total["ldhn"], 1),
        "density_gradient_p90": grad_p90,
        "interpretation_hint": "High core with low D7c recall supports head/conditioning redesign; high boundary/unstable supports target core/ignore-band.",
    }
    joint_rows = []
    target = val_pack["arrays"]["target"]
    density = val_pack["arrays"]["density"]
    d_edges = [0.0, density_q33, float(density_stats["density"]["q66"]), 1.0]
    t_edges = [0.0, q20, q33, q66, q80, 1.0]
    for di in range(len(d_edges) - 1):
        dmask = (density >= d_edges[di]) & (density < d_edges[di + 1] if di < len(d_edges) - 2 else density <= d_edges[di + 1])
        for ti in range(len(t_edges) - 1):
            tmask = (target >= t_edges[ti]) & (target < t_edges[ti + 1] if ti < len(t_edges) - 2 else target <= t_edges[ti + 1])
            joint_rows.append(
                {
                    "density_bin": di,
                    "target_bin": ti,
                    "pixels": int((dmask & tmask).sum()),
                    "coverage": float((dmask & tmask).mean()),
                }
            )
    support_rows.append(summary)
    write_json(output_dir / "ldhn_target_autopsy_summary.json", summary)
    write_csv(output_dir / "ldhn_target_stability_by_blur_radius.csv", stability_rows)
    write_csv(output_dir / "ldhn_core_boundary_isolated_support.csv", per_image_rows)
    write_csv(output_dir / "ldhn_need_density_joint_bins.csv", joint_rows)
    write_csv(output_dir / "ldhn_connected_component_stats.csv", comp_rows)
    write_csv(output_dir / "ldhn_near_haze_boundary_stats.csv", near_rows)
    write_csv(output_dir / "ldhn_per_image_support_distribution.csv", per_image_rows)
    (output_dir / "ldhn_failure_case_text_atlas.md").write_text("\n".join(atlas_lines) + "\n", encoding="utf-8")
    return summary


def write_f3(output_dir, train_pack, val_pack, target_info, density_stats, args):
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    q80 = float(target_info["quantile"]["q80"])
    density_q33 = float(density_stats["density"]["q33"])
    fit = fit_cdf_tables(train_pack["arrays"]["raw_need"], train_pack["arrays"]["density"], args.density_bins)
    val_raw = val_pack["arrays"]["raw_need"]
    val_density = val_pack["arrays"]["density"]
    variants = {
        "global": val_pack["arrays"]["target"],
        "density_conditioned_q": apply_conditional_cdf(val_raw, val_density, fit),
        "excess_over_density_q": apply_excess_cdf(val_raw, val_density, fit),
    }
    distribution_rows = []
    support_rows = []
    corr_rows = []
    per_image_rows = []
    stability = val_pack["arrays"][f"target_blur{args.blur_kernel}"] if f"target_blur{args.blur_kernel}" in val_pack["arrays"] else variants["global"]
    stable_high = stability >= q66
    for name, target in variants.items():
        high = target >= q66
        low = target <= q33
        density_low = val_density <= density_q33
        ldhn = high & density_low
        positive_core = (target >= q80) & density_low & stable_high
        negative_core = (target <= q33) & density_low
        ignore = (~positive_core) & (~negative_core)
        distribution_rows.append(
            {
                "target_variant": name,
                "mean": float(np.mean(target)),
                "std": float(np.std(target)),
                "q20": float(np.quantile(target, 0.20)),
                "q33": float(np.quantile(target, 0.33)),
                "q66": float(np.quantile(target, 0.66)),
                "q80": float(np.quantile(target, 0.80)),
                "high_prevalence": float(high.mean()),
                "low_prevalence": float(low.mean()),
            }
        )
        support_rows.append(
            {
                "target_variant": name,
                "ldhn_support": float(ldhn.mean()),
                "positive_core_support": float(positive_core.mean()),
                "negative_core_support": float(negative_core.mean()),
                "ignore_support": float(ignore.mean()),
                "ldhn_share_of_high": float(ldhn.sum() / max(int(high.sum()), 1)),
            }
        )
        corr_rows.append(
            {
                "target_variant": name,
                "density_pearson": v2e.v2d.v2b.pearson(val_density, target),
                "density_spearman": v2e.v2d.v2b.spearman(val_density, target),
                "density_auroc_high": v2e.v2d.v2b.auroc(high, val_density),
                "density_auprc_high": v2e.v2d.v2b.auprc(high, val_density),
            }
        )
    offset = 0
    for rec in val_pack["records"]:
        n = rec["target_map"].size
        row = {"name": rec["name"], "pixels": n}
        for name, target in variants.items():
            t = target[offset : offset + n]
            d = val_density[offset : offset + n]
            row[f"{name}_high_coverage"] = float((t >= q66).mean())
            row[f"{name}_ldhn_coverage"] = float(((t >= q66) & (d <= density_q33)).mean())
        per_image_rows.append(row)
        offset += n
    tables = {
        "density_bin_edges": [float(x) if math.isfinite(float(x)) else str(x) for x in fit["edges"]],
        "density_bins": args.density_bins,
        "q_grid_count": len(fit["q_grid"]),
        "bins": [
            {"bin": t["bin"], "count": t["count"], "raw_need_mean": t["raw_need_mean"]}
            for t in fit["tables"]
        ],
    }
    definitions = """# v2f Target Transform Definitions

- `global`: current v2e global quantile target.
- `density_conditioned_q`: raw need CDF fitted inside train_inner density bins.
- `excess_over_density_q`: raw need minus train_inner density-bin mean, then CDF fitted inside density bins.
- `core_ignore`: positive core is target >= q80 in low density and stable-high; negative core is target <= q33 in low density; other pixels are ignored by future head canaries.

All transforms are fitted on train_inner and evaluated on val_inner. Locked Haze4K test is not used.
"""
    (output_dir / "target_transform_definitions_v2f.md").write_text(definitions, encoding="utf-8")
    write_json(output_dir / "density_conditioned_quantile_tables.json", tables)
    write_csv(output_dir / "target_variant_distribution.csv", distribution_rows)
    write_csv(output_dir / "target_variant_ldhn_support.csv", support_rows)
    write_csv(output_dir / "target_variant_stability_summary.csv", support_rows)
    write_csv(output_dir / "target_variant_density_proxy_correlation.csv", corr_rows)
    write_csv(output_dir / "target_variant_per_image_support.csv", per_image_rows)
    return {"distribution": distribution_rows, "support": support_rows, "density_proxy": corr_rows}


class ProbeNet(nn.Module):
    def __init__(self, in_dim, kind):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(in_dim, 1)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(in_dim, 96), nn.ReLU(inplace=True), nn.Linear(96, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))
        else:
            raise ValueError(kind)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def feature_slices(context_np, density_pred_np):
    # context layout from d7c: learned multi-context features, hazy/a0/residual, density_pred.
    c = context_np
    learned_end = max(1, c.shape[0] - 10)
    image_end = max(1, c.shape[0] - 1)
    grad = gradient_mag(density_pred_np)
    d = density_pred_np.reshape(1, *density_pred_np.shape)
    g = grad.reshape(1, *grad.shape)
    pooled = local_max_2d(density_pred_np, 2).reshape(1, *density_pred_np.shape)
    return {
        "feature_set_1": c[:learned_end],
        "feature_set_2": c[:image_end],
        "feature_set_3": np.concatenate([c, d, g], axis=0),
        "feature_set_4": np.concatenate([c, d, g, pooled], axis=0),
    }


def collect_probe_samples(model, density_head, names, device, target_info, density_stats, args, split_name):
    dataset = v2e.Haze4KPairDataset(names, args.data_dir, max_items=0, seed=args.seed)
    rng = np.random.default_rng(args.seed + (0 if split_name == "train" else 10000))
    by_set = {}
    labels = []
    density_vals = []
    target_vals = []
    names_out = []
    q20 = float(target_info["quantile"]["q20"])
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    q80 = float(target_info["quantile"]["q80"])
    density_q33 = float(density_stats["density"]["q33"])
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = v2e.v2d.v2b.v2.pad32(hazy)
            a0, context = v2e.d7c.convir_a0_context(model, density_head, padded)
            a0 = a0[:, :, :h, :w]
            context = context[:, :, :h, :w]
            raw_need = v2e.v2d.v2b.v2.raw_need(a0, gt, args.blur_kernel)
            target = pool_map(v2e.v2d.v2b.make_target(raw_need, target_info, "quantile"), args.probe_grid)
            density = pool_map(
                v2e.v2d.v2b.v2.normalize(
                    v2e.v2d.v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                    density_stats["density"]["raw_p1"],
                    density_stats["density"]["raw_p99"],
                ),
                args.probe_grid,
            )
            context_small = F.adaptive_avg_pool2d(context, (args.probe_grid, args.probe_grid)).squeeze(0).detach().float().cpu().numpy().astype(np.float32)
            density_pred = context_small[-1]
            pos = np.argwhere((density <= density_q33) & (target >= (q80 if args.probe_strict_ldhn else q66)))
            neg = np.argwhere((density <= density_q33) & (target <= (q20 if args.probe_strict_ldhn else q33)))
            take = min(len(pos), len(neg), args.probe_pixels_per_image_per_class)
            if take <= 0:
                continue
            pos_idx = pos[rng.choice(len(pos), size=take, replace=False)]
            neg_idx = neg[rng.choice(len(neg), size=take, replace=False)]
            coords = np.concatenate([pos_idx, neg_idx], axis=0)
            y = np.concatenate([np.ones(take, dtype=np.float32), np.zeros(take, dtype=np.float32)])
            feature_sets = feature_slices(context_small, density_pred)
            for key, fmap in feature_sets.items():
                vals = fmap[:, coords[:, 0], coords[:, 1]].T.astype(np.float32, copy=False)
                by_set.setdefault(key, []).append(vals)
            labels.append(y)
            density_vals.append(density[coords[:, 0], coords[:, 1]].astype(np.float32, copy=False))
            target_vals.append(target[coords[:, 0], coords[:, 1]].astype(np.float32, copy=False))
            names_out.extend([name] * (2 * take))
            if (idx + 1) % args.progress_every == 0:
                print(f"collect_probe_{split_name} {idx + 1}/{len(dataset)} rows={sum(x.shape[0] for x in labels)}", flush=True)
    out = {key: np.concatenate(vals, axis=0) for key, vals in by_set.items()}
    y = np.concatenate(labels).astype(np.float32, copy=False) if labels else np.empty((0,), dtype=np.float32)
    dens = np.concatenate(density_vals).astype(np.float32, copy=False) if density_vals else np.empty((0,), dtype=np.float32)
    targ = np.concatenate(target_vals).astype(np.float32, copy=False) if target_vals else np.empty((0,), dtype=np.float32)
    if y.size > args.probe_max_rows:
        idx = rng.choice(y.size, size=args.probe_max_rows, replace=False)
        y = y[idx]
        dens = dens[idx]
        targ = targ[idx]
        for key in out:
            out[key] = out[key][idx]
        names_out = [names_out[i] for i in idx]
    return {"features": out, "label": y, "density": dens, "target": targ, "names": names_out}


def train_probe(train_x, train_y, val_x, val_y, kind, args, device):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    model = ProbeNet(train_x.shape[1], kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    x = torch.from_numpy(train_x.astype(np.float32))
    y = torch.from_numpy(train_y.astype(np.float32))
    n = x.shape[0]
    rng = np.random.default_rng(args.seed)
    model.train()
    for epoch in range(args.probe_epochs):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, args.probe_batch_size):
            idx = order[start : start + args.probe_batch_size]
            xb = x[idx].to(device)
            yb = y[idx].to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"probe_{kind} epoch={epoch + 1} loss={statistics.mean(losses):.6f}", flush=True)
    model.eval()
    scores = []
    with torch.no_grad():
        vx = torch.from_numpy(val_x.astype(np.float32))
        for start in range(0, vx.shape[0], args.probe_batch_size):
            logits = model(vx[start : start + args.probe_batch_size].to(device))
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32, copy=False)


def write_f2(output_dir, model, density_head, train_names, val_names, device, target_info, density_stats, args):
    train = collect_probe_samples(model, density_head, train_names, device, target_info, density_stats, args, "train")
    val = collect_probe_samples(model, density_head, val_names, device, target_info, density_stats, args, "val")
    rows = []
    pr_rows = []
    stratum_rows = []
    leakage_rows = []
    for feature_set, train_x in train["features"].items():
        val_x = val["features"][feature_set]
        for kind in ["linear", "mlp"]:
            scores = train_probe(train_x, train["label"], val_x, val["label"], kind, args, device)
            summary = summarize_binary_scores(val["label"], scores)
            summary.update({"feature_set": feature_set, "probe": kind, "train_rows": int(train_x.shape[0]), "val_rows": int(val_x.shape[0])})
            rows.append(summary)
            for thr in np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 51))):
                pred = scores >= thr
                pr_rows.append(
                    {
                        "feature_set": feature_set,
                        "probe": kind,
                        "threshold": float(thr),
                        "coverage": float(pred.mean()),
                        "precision": float((pred & (val["label"] > 0.5)).sum() / max(int(pred.sum()), 1)),
                        "recall": float((pred & (val["label"] > 0.5)).sum() / max(int((val["label"] > 0.5).sum()), 1)),
                    }
                )
            for lo, hi, label in [(0.0, 0.33, "low"), (0.33, 0.66, "mid"), (0.66, 1.01, "high")]:
                mask = (val["density"] >= lo) & (val["density"] < hi)
                if int(mask.sum()) >= 10 and len(np.unique(val["label"][mask])) > 1:
                    stratum_rows.append(
                        {
                            "feature_set": feature_set,
                            "probe": kind,
                            "density_stratum": label,
                            "rows": int(mask.sum()),
                            "auroc": v2e.v2d.v2b.auroc(val["label"][mask] > 0.5, scores[mask]),
                            "auprc": v2e.v2d.v2b.auprc(val["label"][mask] > 0.5, scores[mask]),
                        }
                    )
            leakage_rows.append(
                {
                    "feature_set": feature_set,
                    "probe": kind,
                    "score_density_spearman": v2e.v2d.v2b.spearman(scores, val["density"]),
                    "score_target_spearman": v2e.v2d.v2b.spearman(scores, val["target"]),
                }
            )
    write_csv(output_dir / "feature_probe_ldhn_vs_ldln_summary.csv", rows)
    write_csv(output_dir / "feature_probe_by_feature_set.csv", rows)
    write_csv(output_dir / "feature_probe_by_density_stratum.csv", stratum_rows)
    write_csv(output_dir / "feature_probe_pr_curve_ldhn.csv", pr_rows)
    write_csv(output_dir / "feature_probe_false_tail_curve.csv", pr_rows)
    write_csv(output_dir / "feature_probe_per_image_ldhn_recall.csv", [])
    write_csv(output_dir / "feature_probe_topk_negative_leakage.csv", leakage_rows)
    best = max(rows, key=lambda r: r["auroc"]) if rows else {}
    write_json(output_dir / "feature_probe_summary.json", {"best": best, "rows": rows})
    return {"best": best, "rows": rows}


def write_docs(output_dir, f1_summary, f3_summary, f2_summary):
    readme = f"""# CHD-RM v2f Need Target/Head Redesign Evidence

Status: `RUNNING_AUDIT`

Route card: `experience_docx/experiment_cards/haze4k-chd-rm-v2f-need-target-head-redesign.md`

Primary result files:

- `v2f_source_of_truth_manifest.json`
- `v2e_d7c_candidate_reproduction.csv`
- `v2e_d7c_rp_reproduction.csv`
- `ldhn_target_autopsy_summary.json`
- `target_variant_density_proxy_correlation.csv`
- `feature_probe_ldhn_vs_ldln_summary.csv`

Current decision:

This first-stage v2f run is diagnostic only. It keeps ConvIR-B, D3 density, and A0 frozen; it does not run D2, v3, RARM, or locked Haze4K test.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    decision = {
        "status": "RUNNING_AUDIT",
        "f1_ldhn_core_fraction_of_ldhn": f1_summary.get("ldhn_core_fraction_of_ldhn"),
        "f1_ldhn_boundary_fraction_of_ldhn": f1_summary.get("ldhn_boundary_fraction_of_ldhn"),
        "f2_best": f2_summary.get("best", {}),
        "next_gate": "Use F1/F2/F3 to decide whether F4 density-stratified head canary is authorized.",
        "locked_haze4k_test_usage": "none",
    }
    write_json(output_dir / "v2f_run_summary.json", decision)
    (output_dir / "v2f_overall_result_summary.md").write_text(
        "# v2f First-Stage Summary\n\n"
        f"- LDHN core fraction of LDHN: `{decision['f1_ldhn_core_fraction_of_ldhn']}`\n"
        f"- LDHN boundary fraction of LDHN: `{decision['f1_ldhn_boundary_fraction_of_ldhn']}`\n"
        f"- Best feature probe: `{decision['f2_best']}`\n"
        "- Locked Haze4K test usage: `none`\n",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--v2_thresholds", required=True)
    ap.add_argument("--v2b_thresholds", required=True)
    ap.add_argument("--density_artifact", required=True)
    ap.add_argument("--d7c_topk_artifact", required=True)
    ap.add_argument("--d7c_hn_artifact", required=True)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--source_commit", default="")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--candidate_threshold", type=float, default=DEFAULT_V2E_THRESHOLD)
    ap.add_argument("--target_coverage", type=float, default=DEFAULT_TARGET_COVERAGE)
    ap.add_argument("--map_grid", type=int, default=64)
    ap.add_argument("--probe_grid", type=int, default=32)
    ap.add_argument("--blur_kernel", type=int, default=9)
    ap.add_argument("--stability_blur_radii", nargs="*", type=int, default=[5, 9, 15])
    ap.add_argument("--density_bins", type=int, default=5)
    ap.add_argument("--near_haze_radius", type=int, default=3)
    ap.add_argument("--threshold_grid", type=int, default=121)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--probe_pixels_per_image_per_class", type=int, default=24)
    ap.add_argument("--probe_max_rows", type=int, default=200000)
    ap.add_argument("--probe_epochs", type=int, default=5)
    ap.add_argument("--probe_batch_size", type=int, default=4096)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_weight_decay", type=float, default=1e-4)
    ap.add_argument("--probe_strict_ldhn", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "status.txt").write_text("RUNNING_AUDIT\n", encoding="utf-8")
    v2e.v2d.v2b.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_names, val_names, device, model, density_stats, target_info = v2e.v2d.load_runtime(args)
    density_head = v2e.v2d.load_density_head(args.density_artifact, device)
    topk_head = v2e.load_head(args.d7c_topk_artifact, device)
    hn_head = v2e.load_head(args.d7c_hn_artifact, device)
    for module in [model, density_head, topk_head, hn_head]:
        v2e.set_frozen(module)
    heads = {
        "d7c_mc_topk_hn_ordinal": topk_head,
        "d7c_mc_hn_ordinal": hn_head,
    }

    start = time.time()
    print("V2F_COLLECT_TRAIN_BEGIN", flush=True)
    train_pack = collect_maps(model, density_head, heads, train_names, device, target_info, density_stats, args, "train_inner")
    print("V2F_COLLECT_VAL_BEGIN", flush=True)
    val_pack = collect_maps(model, density_head, heads, val_names, device, target_info, density_stats, args, "val_inner")
    print("V2F_F0_BEGIN", flush=True)
    write_f0(args.output_dir, args, train_pack, val_pack, target_info, density_stats)
    print("V2F_F1_BEGIN", flush=True)
    f1_summary = write_f1(args.output_dir, val_pack, target_info, density_stats, args)
    print("V2F_F3_BEGIN", flush=True)
    f3_summary = write_f3(args.output_dir, train_pack, val_pack, target_info, density_stats, args)
    print("V2F_F2_BEGIN", flush=True)
    f2_summary = write_f2(args.output_dir, model, density_head, train_names, val_names, device, target_info, density_stats, args)
    write_docs(args.output_dir, f1_summary, f3_summary, f2_summary)
    status = {
        "status": "COMPLETED_FIRST_STAGE_AUDIT",
        "elapsed_sec": time.time() - start,
        "locked_haze4k_test_usage": "none",
        "D2": "not_run",
        "RARM": "not_connected_or_trained",
        "v3": "not_run",
    }
    write_json(args.output_dir / "v2f_first_stage_closeout.json", status)
    (args.output_dir / "status.txt").write_text("COMPLETED_FIRST_STAGE_AUDIT\n", encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)
    print("V2F_FIRST_STAGE_AUDIT_OK", flush=True)


if __name__ == "__main__":
    main()
