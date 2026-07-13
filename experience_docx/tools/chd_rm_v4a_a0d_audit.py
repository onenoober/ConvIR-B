#!/usr/bin/env python3
"""A0D descriptive risk decomposition from the canonical A0R r1 final state.

This program deliberately reuses the frozen v3z entrypoint and A0R retained
state. It never selects an update method, window, threshold, or candidate.
Raw rows remain in the cloud output directory; only compact summaries and a
typed closeout are eligible for Git evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROUTE_ID = "haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714"
SOURCE: Any = None
AUDIT: argparse.Namespace | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_source(v3z_root: Path) -> Any:
    source_path = v3z_root / "experience_docx" / "tools" / "chd_rm_v3x_projected_safety_constraint.py"
    spec = importlib.util.spec_from_file_location("v4a_a0d_v3z", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen v3z source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_name(name: str) -> tuple[str, float, float]:
    pieces = Path(name).stem.split("_")
    image_id = pieces[0]
    haze_1 = float(pieces[1]) if len(pieces) > 2 else math.nan
    haze_2 = float(pieces[2]) if len(pieces) > 2 else math.nan
    return image_id, haze_1, haze_2


def psnr(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-30))


def finite_row(row: dict[str, Any]) -> bool:
    for value in row.values():
        if isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def row_metrics(args: Any, v3s: Any, legacy: Any, frozen: Any, name: str, fold: int, kind: str, model: torch.nn.Module, device: torch.device, split: str) -> list[dict[str, Any]]:
    assert SOURCE is not None
    sample = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, name, fold, device)
    sample["delta_bound"] = args.delta_bound
    image_id, haze_1, haze_2 = parse_name(name)
    rows: list[dict[str, Any]] = []
    for operator in SOURCE.V3W.OPERATORS:
        delta = SOURCE.V3W.delta_for(kind, model, sample, operator)
        old_low, old_high, new_low, new_high = v3s.candidate_predictions(sample["base"], sample["steps"][operator], delta)
        old_low_mse = float(v3s.per_image_mse(old_low, sample["label"]).item())
        old_high_mse = float(v3s.per_image_mse(old_high, sample["label"]).item())
        new_high_mse = float(v3s.per_image_mse(new_high, sample["label"]).item())
        raw_old_high = sample["base"] + 0.25 * sample["steps"][operator]
        outside = (raw_old_high < 0.0) | (raw_old_high > 1.0)
        support_fraction = float(sample["support"].mean().item())
        inherited = max(old_high_mse - old_low_mse, 0.0)
        total = max(new_high_mse - old_low_mse, 0.0)
        delta_psnr = psnr(new_high_mse) - psnr(old_high_mse)
        rows.append(
            {
                "schema_version": 1,
                "split": split,
                "name": name,
                "clean_reference_group": image_id,
                "operator": operator,
                "haze_param_1": haze_1,
                "haze_param_2": haze_2,
                "old_125_mse": old_low_mse,
                "old_250_mse": old_high_mse,
                "new_250_mse": new_high_mse,
                "old_250_psnr": psnr(old_high_mse),
                "new_250_psnr": psnr(new_high_mse),
                "delta_psnr": delta_psnr,
                "H_inherited": inherited,
                "H_total": total,
                "H_intervention": total - inherited,
                "H_predecessor_positive": max(new_high_mse - old_high_mse, 0.0),
                "support_fraction": support_fraction,
                "support_normalized_total": total / support_fraction if support_fraction > 0.0 else math.nan,
                "old_step_l2": float(torch.sqrt(torch.mean(sample["steps"][operator].square())).item()),
                "old_preclamp_fraction": float(outside.float().mean().item()),
            }
        )
    return rows


def percentile_edges(values: list[float]) -> tuple[float, float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        raise RuntimeError("cannot define quartiles from no finite observations")
    result = np.quantile(np.asarray(finite, dtype=np.float64), [0.25, 0.50, 0.75], method="linear")
    return float(result[0]), float(result[1]), float(result[2])


def quartile(value: float, edges: tuple[float, float, float]) -> str:
    if not math.isfinite(value):
        return "missing"
    if value <= edges[0]:
        return "q1"
    if value <= edges[1]:
        return "q2"
    if value <= edges[2]:
        return "q3"
    return "q4"


def cvar(values: list[float], fraction: float) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        return math.nan
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.mean(sorted(values, reverse=True)[:count]))


def percentile(values: list[float], q: float) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        return math.nan
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def summary_row(rows: list[dict[str, Any]], split: str, operator: str, group_name: str, group_value: str, planned_n: int) -> dict[str, Any]:
    inherited = [float(row["H_inherited"]) for row in rows]
    total = [float(row["H_total"]) for row in rows]
    intervention = [float(row["H_intervention"]) for row in rows]
    predecessor = [float(row["H_predecessor_positive"]) for row in rows]
    delta_psnr = [float(row["delta_psnr"]) for row in rows]
    normalized = [float(row["support_normalized_total"]) for row in rows]
    finite = all(finite_row(row) for row in rows)
    return {
        "split": split,
        "operator": operator,
        "group_name": group_name,
        "group_value": group_value,
        "planned_n": planned_n,
        "observed_n": len(rows),
        "nonfinite_n": sum(not finite_row(row) for row in rows),
        "H_inherited_mean": float(np.mean(inherited)) if finite and rows else math.nan,
        "H_total_mean": float(np.mean(total)) if finite and rows else math.nan,
        "H_intervention_mean": float(np.mean(intervention)) if finite and rows else math.nan,
        "H_predecessor_positive_mean": float(np.mean(predecessor)) if finite and rows else math.nan,
        "H_total_median": percentile(total, 0.50),
        "H_total_p95": percentile(total, 0.95),
        "H_total_cvar10": cvar(total, 0.10),
        "support_normalized_total_mean": float(np.mean(normalized)) if finite and rows else math.nan,
        "old_250_psnr_mean": float(np.mean([float(row["old_250_psnr"]) for row in rows])) if finite and rows else math.nan,
        "new_250_psnr_mean": float(np.mean([float(row["new_250_psnr"]) for row in rows])) if finite and rows else math.nan,
        "delta_psnr_p05": percentile(delta_psnr, 0.05),
        "severe_count": sum(value <= -0.2 for value in delta_psnr),
        "hard_count": sum(value <= -0.5 for value in delta_psnr),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty table: {path}")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_a0d(args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str], folds: dict[str, int], device: torch.device, output_dir: str) -> dict[str, Any]:
    assert SOURCE is not None and AUDIT is not None
    audit = AUDIT
    a0r_closeout = json.loads(Path(audit.a0r_closeout).read_text(encoding="utf-8"))
    required = {
        "route_id": ROUTE_ID,
        "state": "COMPLETED_GATE_PASS",
        "decision": "V4A_A0R_REPRODUCTION_PASS_AUTHORIZE_A0D_AND_A0P",
        "authorizes": "A0D_AND_A0P_ONLY",
    }
    for key, expected in required.items():
        if a0r_closeout.get(key) != expected:
            raise RuntimeError(f"A0R closeout {key} mismatch: {a0r_closeout.get(key)!r}")
    state_payload = torch.load(audit.a0r_final_state, map_location=device)
    if state_payload.get("state_kind") != "final" or state_payload.get("replicate_id") != "r1":
        raise RuntimeError("A0D requires the canonical r1 final retained state")
    all_names, _ = v3s.load_names_and_folds(args, legacy)
    heldout = all_names[args.sample_count:args.sample_count * 2]
    if list(names) != list(all_names[:args.sample_count]) or len(heldout) != args.sample_count:
        raise RuntimeError("A0D update/heldout population mismatch")
    models = SOURCE.V3W.import_v3w_models()
    first = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, names[0], folds[names[0]], device)
    cells = SOURCE.V3W.build_cells(models, first, args, device)
    label, (kind, objective, model) = next(iter(cells.items()))
    if objective != "safety_curriculum":
        raise RuntimeError(f"unexpected A0R objective: {objective}")
    model.load_state_dict(state_payload["model_state"], strict=True)
    rows: list[dict[str, Any]] = []
    for split, split_names in (("update128", names), ("heldout128", heldout)):
        for name in split_names:
            rows.extend(row_metrics(args, v3s, legacy, frozen, name, folds[name], kind, model, device, split))
    expected_rows = 2 * args.sample_count * len(SOURCE.V3W.OPERATORS)
    if len(rows) != expected_rows or not all(finite_row(row) for row in rows):
        raise RuntimeError("A0D row completeness or finiteness failure")
    initial_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        initial_by_key[(str(row["name"]), str(row["operator"]))] = row
    group_fields = ("haze_param_1", "haze_param_2", "old_step_l2", "support_fraction", "H_inherited", "old_preclamp_fraction")
    edges = {field: percentile_edges([float(row[field]) for row in rows]) for field in group_fields}
    summaries: list[dict[str, Any]] = []
    for split in ("update128", "heldout128"):
        for operator in SOURCE.V3W.OPERATORS:
            base = [row for row in rows if row["split"] == split and row["operator"] == operator]
            summaries.append(summary_row(base, split, operator, "all", "all", args.sample_count))
            for field in group_fields:
                for bucket in ("q1", "q2", "q3", "q4"):
                    grouped = [row for row in base if quartile(float(row[field]), edges[field]) == bucket]
                    summaries.append(summary_row(grouped, split, operator, field, bucket, args.sample_count))
    output = Path(output_dir)
    raw_path = output / f"{args.run_tag}_a0d_rows_cloud_only.csv"
    summary_path = output / "v4a_a0d_group_tail_summary.csv"
    write_rows(raw_path, rows)
    write_rows(summary_path, summaries)
    closeout = {
        "route_id": ROUTE_ID,
        "stage": "v4a-A0D-descriptive-risk-decomposition",
        "state": "COMPLETED_GATE_PASS",
        "decision": "V4A_A0D_DESCRIPTIVE_COMPLETE_AUTHORIZE_A0P_INTERPRETATION_ONLY",
        "authorizes": "A0P_INTERPRETATION_ONLY",
        "evidence_role": "development_screening",
        "metric_contract": "v4a amended A0D descriptive risk decomposition",
        "a0r_closeout_sha256": sha256_file(Path(audit.a0r_closeout)),
        "a0r_final_state_sha256": sha256_file(Path(audit.a0r_final_state)),
        "row_count": len(rows),
        "summary_count": len(summaries),
        "group_edges": edges,
        "raw_rows_cloud_only": str(raw_path),
        "summary": str(summary_path),
        "locked_test_touched": False,
        "canary_touched": False,
        "candidate_selected": False,
    }
    closeout["contract_id"] = canonical_hash(closeout)
    write_json(output / "v4a_a0d_closeout.json", closeout)
    print(json.dumps(closeout, sort_keys=True), flush=True)
    return closeout


def audit(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v3z-root", required=True)
    parser.add_argument("--a0r-closeout", required=True)
    parser.add_argument("--a0r-final-state", required=True)
    args, v3z_args = parser.parse_known_args(argv)
    if not v3z_args:
        raise ValueError("frozen v3z arguments are required after A0D arguments")
    global SOURCE, AUDIT
    AUDIT = args
    SOURCE = load_source(Path(args.v3z_root).resolve())
    SOURCE.run_projected = run_a0d
    original = sys.argv[:]
    try:
        sys.argv = [str(SOURCE.__file__), *v3z_args]
        SOURCE.main()
    finally:
        sys.argv = original


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "audit":
        raise SystemExit("usage: chd_rm_v4a_a0d_audit.py audit --v3z-root ... --a0r-closeout ... --a0r-final-state ... projected ...")
    audit(sys.argv[2:])


if __name__ == "__main__":
    main()
