#!/usr/bin/env python3
"""Review the frozen tail-controller reconstruction discrepancy without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import run_haze4k_frozen_output_tail_controller_qualification as frozen
from route_program_api import (
    asset_path,
    atomic_json,
    load_completed_unit_ledger,
    load_context,
    output_file,
    prepare_phase_output,
    record_completed_unit,
    write_contract_progress,
    write_contract_result,
    write_gate_result,
    write_workload_progress,
)


ROUTE_ID = "haze4k-frozen-output-tail-controller-evidence-review-v1"
OPERATION_ID = "HAZE4K_FROZEN_OUTPUT_TAIL_CONTROLLER_EVIDENCE_REVIEW"
RUN_ID = "haze4k-frozen-output-tail-controller-evidence-review-v1-r3"
PARENT_ROUTE_ID = "haze4k-frozen-output-tail-controller-qualification-v1"
PARENT_ROUTE_COMMIT = "4ab3443359e352a2e1ad3590cc6f4b326e899f07"
PARENT_RECEIPT = "56e89b4a6646c26e6c2d9a42db9769a428ffe086a34d8bd519cab6e19bc95575"
PARENT_ENTRYPOINT_SHA256 = "b892caa106d3a5e8052e09076a69a3e29bb844fc47af7c040a5373169133bd96"
PARENT_CLOSEOUT_SHA256 = "54b61f0f96d7442ec73b70b950090762631220c1ed2f01654b8f01b55cdaeac1"
PARENT_CONCLUSION_SHA256 = "4a1f6292759766e406cebc566230ecac60ed580d2ea87cba539ab1e0531ba682"
PARENT_SUMMARY_SHA256 = "4b1232bf30ac9dd1a623b12fdde61d5e1d035a1959eb052114ab10ab21669024"
PARENT_GATE_SHA256 = "ef5af047fe7e795b58db742de8ec127fa431c3d25345d6902d4dba4a61f3d199"
PARENT_SELECTION_SHA256 = "06420d49c3e1a4000ede155d7e8d441f9c7b98414152d16644fb6f50e1b906e9"
PARENT_RAW_INVENTORY_SHA256 = "fdebfc7cdad65d355b551fcca9f4fa4130e468525c35c96557e2f6b67ae0e188"
PARENT_INVENTORY_VERIFICATION_SHA256 = (
    "a1a11a4e3001820bb1fac4cb14164d9b4c3d83a73ffe30ffcda98e93000e797b"
)
UPSTREAM_RAW_INVENTORY_SHA256 = (
    "9a5249f66c994f9c94245145cc65e02da643ada71cc047cd5144db475b3c66ff"
)
RUNTIME_ENVIRONMENT_SHA256 = (
    "35600a8354cfca6c0f3ed3c6159a362377e5c558795208f325e59d74b59b569b"
)
ANCHOR_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"

SELECTED_CANDIDATE_ID = "R0_L1__weighted_q__raw__setwise__cap_0p01"
SELECTED_CELL = "R0_L1"
SELECTED_AREA_CAP = 0.01
EXPECTED_SCENES = 600
EXPECTED_CELLS = 9
EXPECTED_COMPARISON_FILES = EXPECTED_SCENES * EXPECTED_CELLS
TOTAL_UNITS = EXPECTED_SCENES + 2
THRESHOLDS = (1.0e-6, 1.0e-5, 1.0e-4)
QUANTILES = (0.50, 0.90, 0.95, 0.99, 0.999)
PRECISION_CRITICAL_VALUE = 2.2414027276049464
PRECISION_TARGET_HALF_WIDTH = 0.05
EXPECTED_REQUIRED_SCENES = 503
EXPECTED_CURRENT_INVENTORY = {
    "file_count": 745,
    "total_bytes": 183620315,
    "inventory_sha256": "6851661520aef8367dafa34633f63d6620e41fbc06ddffe839d85f21171f8602",
    "classes": {
        "calibration_prediction": {
            "file_count": 45,
            "total_bytes": 85823910,
            "inventory_sha256": "0b726b5f9d2abd1a6ff1cbe231591ae492b6b0e651c157d18f49cfa58bfc9d30",
        },
        "oof_controller_scene": {
            "file_count": 600,
            "total_bytes": 96926085,
            "inventory_sha256": "d01aeee4098777b908f299f35f71bf7b04f8dd328651656f8ad34383a9e50a7f",
        },
        "test_controller_scene": {
            "file_count": 100,
            "total_bytes": 870320,
            "inventory_sha256": "d9696582f43e8f4f5fe7e58eccd2f4429f99ef2d9744b348f92d81083bb966bd",
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


class DifferenceTracker:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        self.element_counts: dict[tuple[str, str, float], int] = defaultdict(int)
        self.cells: dict[tuple[str, str, float], set[str]] = defaultdict(set)
        self.scenes: dict[tuple[str, str, float], set[str]] = defaultdict(set)
        self.files: dict[tuple[str, str, float], set[str]] = defaultdict(set)
        self.current_files: dict[tuple[str, str, float], set[str]] = defaultdict(set)
        self.worst: dict[tuple[str, str], dict[str, Any]] = {}
        self.nonfinite_elements: dict[tuple[str, str], int] = defaultdict(int)

    def add(
        self,
        source: str,
        metric: str,
        current: np.ndarray,
        reference: np.ndarray,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if current.shape != reference.shape:
            raise ValueError("shape mismatch")
        finite = np.isfinite(current) & np.isfinite(reference)
        self.nonfinite_elements[(source, metric)] += int(current.size - np.sum(finite))
        absolute = np.abs(
            current.astype(np.float64, copy=False)
            - reference.astype(np.float64, copy=False)
        )
        finite_values = absolute[finite]
        if finite_values.size:
            self.values[(source, metric)].append(finite_values.reshape(-1).copy())
            finite_absolute = np.where(finite, absolute, -np.inf)
            flat_index = int(np.argmax(finite_absolute))
            maximum = float(finite_absolute.reshape(-1)[flat_index])
            key = (source, metric)
            if key not in self.worst or maximum > float(self.worst[key]["absolute_error"]):
                index = np.unravel_index(flat_index, absolute.shape)
                point = dict(metadata)
                point.update({
                    "source": source,
                    "metric": metric,
                    "absolute_error": maximum,
                    "action": int(index[0]),
                    "action_name": frozen.ACTION_NAMES[int(index[0])],
                    "row": int(index[1]),
                    "column": int(index[2]),
                    "current_value": float(current[index]),
                    "reference_value": float(reference[index]),
                    "current_dtype": str(current.dtype),
                    "reference_dtype": str(reference.dtype),
                    "shape": list(current.shape),
                })
                self.worst[key] = point
        counts: dict[str, int] = {}
        for threshold in THRESHOLDS:
            count = int(np.sum(finite & (absolute > threshold)))
            counts[f"gt_{threshold:.0e}"] = count
            if count:
                key = (source, metric, threshold)
                self.element_counts[key] += count
                self.cells[key].add(str(metadata["cell"]))
                self.scenes[key].add(str(metadata["scene"]))
                self.files[key].add(str(metadata["comparison_file_id"]))
                self.current_files[key].add(str(metadata["current_file_relpath"]))
        return {
            "maximum_absolute_error": (
                float(np.max(finite_values)) if finite_values.size else None
            ),
            "nonfinite_elements": int(current.size - np.sum(finite)),
            **counts,
        }

    def maximum(self, source: str) -> float:
        values = [
            float(item["absolute_error"])
            for (item_source, _), item in self.worst.items()
            if item_source == source
        ]
        return max(values, default=0.0)

    def quantile_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for source in sorted({key[0] for key in self.values}):
            metric_arrays: list[np.ndarray] = []
            for metric in ("utility", "risk"):
                arrays = self.values.get((source, metric), [])
                values = np.concatenate(arrays) if arrays else np.empty(0)
                metric_arrays.append(values)
                rows.append(self._quantile_row(source, metric, values))
            combined = np.concatenate(metric_arrays) if metric_arrays else np.empty(0)
            rows.append(self._quantile_row(source, "combined", combined))
        return rows

    @staticmethod
    def _quantile_row(
        source: str, metric: str, values: np.ndarray,
    ) -> dict[str, Any]:
        quantiles = np.quantile(values, QUANTILES) if values.size else [math.nan] * 5
        return {
            "source": source,
            "metric": metric,
            "finite_element_count": int(values.size),
            "p50": float(quantiles[0]),
            "p90": float(quantiles[1]),
            "p95": float(quantiles[2]),
            "p99": float(quantiles[3]),
            "p99_9": float(quantiles[4]),
            "max": float(np.max(values)) if values.size else math.nan,
            "quantile_method": "numpy_linear_finite_elements",
        }

    def threshold_rows(self) -> list[dict[str, Any]]:
        rows = []
        sources = sorted({key[0] for key in self.values})
        for source in sources:
            for metric in ("utility", "risk"):
                for threshold in THRESHOLDS:
                    key = (source, metric, threshold)
                    rows.append({
                        "source": source,
                        "metric": metric,
                        "threshold_strictly_greater_than": threshold,
                        "element_count": self.element_counts[key],
                        "cell_count": len(self.cells[key]),
                        "scene_count": len(self.scenes[key]),
                        "comparison_file_count": len(self.files[key]),
                        "current_file_count": len(self.current_files[key]),
                    })
        return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    frozen.write_csv(path, rows)


def inventory_items(
    root: Path, specifications: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for artifact_class, pattern in specifications:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"invalid inventory file: {path}")
            items.append({
                "artifact_class": artifact_class,
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    return items


def verify_current_inventory(
    root: Path, expected_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = read_json(expected_path)
    items = inventory_items(root, (
        ("calibration_prediction", "workload/calibration_predictions/*.npz"),
        ("oof_controller_scene", "workload/oof_controller_scenes/*.npz"),
        ("test_controller_scene", "workload/test_controller_scenes/*.npz"),
    ))
    observed = frozen.compact_inventory(items)
    checks = {
        "expected_inventory_file_identity": sha256_file(expected_path)
        == PARENT_RAW_INVENTORY_SHA256,
        "file_count": observed["file_count"]
        == expected.get("file_count")
        == EXPECTED_CURRENT_INVENTORY["file_count"],
        "total_bytes": observed["total_bytes"]
        == expected.get("total_bytes")
        == EXPECTED_CURRENT_INVENTORY["total_bytes"],
        "inventory_sha256": observed["inventory_sha256"]
        == expected.get("inventory_sha256")
        == EXPECTED_CURRENT_INVENTORY["inventory_sha256"],
        "class_inventory": observed["classes"]
        == expected.get("classes")
        == EXPECTED_CURRENT_INVENTORY["classes"],
    }
    return {
        "schema_version": 1,
        "route_id": PARENT_ROUTE_ID,
        "receipt": PARENT_RECEIPT,
        "checks": checks,
        "matched": all(checks.values()),
        "observed": {
            key: observed[key]
            for key in ("file_count", "total_bytes", "inventory_sha256", "classes")
        },
    }, items


def expected_current_keys() -> set[str]:
    keys = {"scene", "fold", "selected_metrics"}
    for cell in frozen.CELL_IDS:
        keys.update({f"{cell}_ensemble_utility", f"{cell}_ensemble_risk"})
        for seed_index in range(len(frozen.SEED_OFFSETS)):
            keys.update({
                f"{cell}_seed_{seed_index}_utility",
                f"{cell}_seed_{seed_index}_risk",
            })
    keys.update({
        "actions_selected", "actions_uniform", "actions_shuffled",
        "actions_permuted", "actions_parent_p11", "actions_raw_p00",
        "actions_observable_utility_gt_risk",
        "actions_gt_utility_observable_risk", "actions_gt_gt",
    })
    return keys


def selected_weighted_value(
    actions: np.ndarray, values: np.ndarray, areas: np.ndarray,
) -> dict[str, Any]:
    rows, columns = np.nonzero(actions > 0)
    total_area = float(np.sum(areas))
    if rows.size == 0:
        return {
            "selected_tiles": 0,
            "selected_area_fraction": 0.0,
            "area_weighted_value": None,
        }
    selected_areas = areas[rows, columns].astype(np.float64)
    selected_values = values[actions[rows, columns] - 1, rows, columns].astype(np.float64)
    return {
        "selected_tiles": int(rows.size),
        "selected_area_fraction": float(np.sum(selected_areas) / total_area),
        "area_weighted_value": float(
            np.sum(selected_areas * selected_values) / np.sum(selected_areas)
        ),
    }


def checkpoint_records(
    upstream_root: Path,
    upstream_lookup: dict[str, dict[str, Any]],
    cell: str,
    fold: int,
) -> list[dict[str, Any]]:
    output = []
    for seed_index in range(len(frozen.SEED_OFFSETS)):
        path = frozen.qual_checkpoint_path(upstream_root, cell, fold, seed_index)
        relative = path.relative_to(upstream_root).as_posix()
        item = upstream_lookup.get(relative)
        if item is None:
            raise RuntimeError(f"checkpoint absent from verified inventory: {relative}")
        output.append({
            "seed_index": seed_index,
            "relative_path": relative,
            "sha256": item["sha256"],
            "bytes": int(item["bytes"]),
        })
    return output


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    if context.route_id != ROUTE_ID or context.operation_id != OPERATION_ID:
        raise RuntimeError("evidence-review contract identity mismatch")
    prepare_phase_output(context)
    started = time.monotonic()
    entrypoint = asset_path(context, "review_entrypoint", kind="file")
    parent_entrypoint = asset_path(context, "parent_qualification_entrypoint", kind="file")
    runtime_environment = asset_path(context, "runtime_environment", kind="file")
    current_inventory = asset_path(context, "current_raw_inventory", kind="file")
    upstream_inventory = asset_path(context, "upstream_raw_inventory", kind="file")
    generator = np.random.default_rng(20260728)
    utility = generator.normal(0.25, 0.05, size=(2, 4, 5)).astype(np.float32)
    risk = generator.uniform(0.01, 0.04, size=(2, 4, 5)).astype(np.float32)
    upstream_utility = utility.copy()
    upstream_utility[0, 0, 0] += np.float32(5.0e-6)
    areas = np.ones((4, 5), dtype=np.float64)
    seeds = np.stack([utility, utility, utility])
    tracker = DifferenceTracker()
    metadata = {
        "cell": SELECTED_CELL,
        "scene": "synthetic",
        "fold": 0,
        "comparison_file_id": "synthetic",
        "current_file_relpath": "synthetic",
        "current_file_sha256": "0" * 64,
        "reference_file_relpath": "synthetic",
        "reference_file_sha256": "0" * 64,
        "checkpoint_files": [],
    }
    tracker.add("upstream_vs_current", "utility", utility, upstream_utility, metadata)
    tracker.add("upstream_vs_current", "risk", risk, risk.copy(), metadata)
    tracker.add(
        "saved_seed_ensemble", "utility", utility, np.mean(seeds, axis=0), metadata,
    )
    current_actions = frozen.qual_choose_setwise(
        utility, risk, areas, area_cap=SELECTED_AREA_CAP,
    )
    upstream_actions = frozen.qual_choose_setwise(
        upstream_utility, risk, areas, area_cap=SELECTED_AREA_CAP,
    )
    completed_iterations = 18
    write_contract_progress(
        context,
        completed_iterations=completed_iterations,
        total_iterations=completed_iterations,
        stage="cpu_exact_array_and_action_review",
    )
    checks = {
        "entrypoint_identity": context.assets["review_entrypoint"].sha256
        == sha256_file(entrypoint),
        "parent_entrypoint_identity": sha256_file(parent_entrypoint)
        == PARENT_ENTRYPOINT_SHA256,
        "runtime_environment_identity": sha256_file(runtime_environment)
        == RUNTIME_ENVIRONMENT_SHA256,
        "current_inventory_metadata_identity": sha256_file(current_inventory)
        == PARENT_RAW_INVENTORY_SHA256,
        "upstream_inventory_metadata_identity": sha256_file(upstream_inventory)
        == UPSTREAM_RAW_INVENTORY_SHA256,
        "official_anchor_identity": context.assets["official_anchor_checkout"].commit
        == ANCHOR_COMMIT,
        "scientific_workspaces_absent": all(
            identifier not in context.assets
            for identifier in ("current_raw_workspace", "upstream_raw_workspace")
        ),
        "bounded_discrepancy_path": (
            1.0e-6 < tracker.maximum("upstream_vs_current") <= 1.0e-5
        ),
        "seed_ensemble_path": tracker.maximum("saved_seed_ensemble") <= 1.0e-6,
        "action_replay_path": np.array_equal(current_actions, upstream_actions),
        "quantile_and_threshold_paths": (
            len(tracker.quantile_rows()) == 6
            and len(tracker.threshold_rows()) == 12
        ),
    }
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "cpu_exact",
            "device": "cpu",
            "fixture": {"batch": 1, "channels": 2, "height": 4, "width": 5},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
            "cost": {
                "observed_iterations": completed_iterations,
                "observed_wall_seconds": time.monotonic() - started,
                "observed_peak_memory_mib": 0.0,
            },
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    if context.route_id != ROUTE_ID or context.operation_id != OPERATION_ID \
            or context.run_id != RUN_ID:
        raise RuntimeError("evidence-review runtime identity mismatch")
    prepare_phase_output(context)
    current_root = asset_path(context, "current_raw_workspace", kind="directory")
    upstream_root = asset_path(context, "upstream_raw_workspace", kind="directory")
    current_inventory_path = asset_path(context, "current_raw_inventory", kind="file")
    upstream_inventory_path = asset_path(context, "upstream_raw_inventory", kind="file")
    closeout_path = asset_path(context, "parent_closeout", kind="file")
    conclusion_path = asset_path(context, "parent_conclusion", kind="file")
    summary_path = asset_path(context, "parent_summary", kind="file")
    gate_path = asset_path(context, "parent_gate_summary", kind="file")
    selection_path = asset_path(context, "parent_selection_freeze", kind="file")
    parent_verification_path = asset_path(
        context, "parent_inventory_verification", kind="file",
    )

    closeout = read_json(closeout_path)
    conclusion = read_json(conclusion_path)
    parent_summary = read_json(summary_path)
    parent_gate = read_json(gate_path)
    selection = read_json(selection_path)
    parent_verification = read_json(parent_verification_path)
    selected_candidate = selection.get("selected_candidate", {})
    parent_identity_checks = {
        "closeout_sha256": sha256_file(closeout_path) == PARENT_CLOSEOUT_SHA256,
        "conclusion_sha256": sha256_file(conclusion_path) == PARENT_CONCLUSION_SHA256,
        "summary_sha256": sha256_file(summary_path) == PARENT_SUMMARY_SHA256,
        "gate_sha256": sha256_file(gate_path) == PARENT_GATE_SHA256,
        "selection_sha256": sha256_file(selection_path) == PARENT_SELECTION_SHA256,
        "parent_inventory_verification_sha256": sha256_file(parent_verification_path)
        == PARENT_INVENTORY_VERIFICATION_SHA256,
        "route_identity": closeout.get("route_id") == PARENT_ROUTE_ID,
        "route_commit": closeout.get("route_commit") == PARENT_ROUTE_COMMIT,
        "terminal": (
            closeout.get("state") == "COMPLETED_INCONCLUSIVE"
            and closeout.get("decision")
            == "HAZE4K_FROZEN_OUTPUT_TAIL_CONTROLLER_QUALIFICATION_INCONCLUSIVE"
            and closeout.get("authorizes")
            == "FROZEN_OUTPUT_TAIL_CONTROLLER_EVIDENCE_REVIEW_ONLY"
        ),
        "conclusion_identity": (
            conclusion.get("decision") == closeout.get("decision")
            and conclusion.get("authorizes") == closeout.get("authorizes")
        ),
        "selected_candidate": (
            selected_candidate.get("candidate_id") == SELECTED_CANDIDATE_ID
            and selected_candidate.get("cell") == SELECTED_CELL
            and selected_candidate.get("risk_method") == "weighted_q"
            and selected_candidate.get("utility_method") == "raw"
            and selected_candidate.get("selector") == "setwise"
            and float(selected_candidate.get("area_cap", -1.0)) == SELECTED_AREA_CAP
        ),
        "selection_time_isolation": (
            selection.get("formal_oof_arrays_parsed_before_freeze") is False
            and closeout.get("details", {}).get("selection_freeze_sha256")
            == PARENT_SELECTION_SHA256
        ),
        "parent_identity_failure_is_scoped": (
            parent_summary.get("identity_checks", {}).get(
                "raw_prediction_reconstruction"
            ) is False
            and all(
                value is True
                for key, value in parent_summary.get("identity_checks", {}).items()
                if key != "raw_prediction_reconstruction"
            )
        ),
        "parent_gate_copy": parent_gate.get("gate_outcomes")
        == closeout.get("gate_outcomes"),
        "parent_inventory_copy": parent_verification.get("matched") is True,
        "protected_assets_absent": all(
            identifier not in context.assets
            for identifier in (
                "candidate_confirmation", "canary", "locked_test",
                "nh_haze", "reside_its", "reside_ots",
            )
        ),
    }

    current_verification, current_items = verify_current_inventory(
        current_root, current_inventory_path,
    )
    upstream_verification, upstream_items = frozen.qual_verify_parent_inventory(
        upstream_root, upstream_inventory_path,
    )
    current_lookup = {item["relative_path"]: item for item in current_items}
    upstream_lookup = {item["relative_path"]: item for item in upstream_items}
    input_verification = {
        "schema_version": 1,
        "parent_route_id": PARENT_ROUTE_ID,
        "parent_route_commit": PARENT_ROUTE_COMMIT,
        "parent_receipt": PARENT_RECEIPT,
        "parent_identity_checks": parent_identity_checks,
        "current_raw_workspace": current_verification,
        "upstream_raw_workspace": upstream_verification,
    }
    input_verification_path = output_file(context, "input_inventory_verification.json")
    atomic_json(input_verification_path, input_verification)
    record_completed_unit(
        context,
        unit_id="input_inventory_verification",
        input_sha256=sha256_text("|".join([
            PARENT_RECEIPT,
            PARENT_RAW_INVENTORY_SHA256,
            UPSTREAM_RAW_INVENTORY_SHA256,
            PARENT_SELECTION_SHA256,
        ])),
        output_relpath=input_verification_path.name,
    )
    write_workload_progress(
        context, completed_units=1, stage="receipt_bound_input_inventories_verified",
    )

    tracker = DifferenceTracker()
    anomaly_counts = defaultdict(int)
    processed_scenes: set[str] = set()
    processed_comparison_files: set[str] = set()
    observed_folds: set[int] = set()
    raw_output_items: list[dict[str, Any]] = []
    action_counts = defaultdict(int)
    action_scenes: dict[str, set[str]] = defaultdict(set)
    selected_action_replay_scenes: set[str] = set()
    selected_risk_differences: list[float] = []
    selected_utility_differences: list[float] = []
    expected_keys = expected_current_keys()
    current_files = sorted(
        (current_root / "workload" / "oof_controller_scenes").glob("*.npz")
    )
    if len(current_files) != EXPECTED_SCENES:
        anomaly_counts["current_oof_file_count"] += 1

    raw_root = output_file(context, "raw_scene_cell_diagnostics")
    raw_root.mkdir()
    for scene_index, current_path in enumerate(current_files, start=1):
        current_relative = current_path.relative_to(current_root).as_posix()
        current_item = current_lookup.get(current_relative)
        if current_item is None:
            raise RuntimeError(f"current file absent from verified inventory: {current_relative}")
        scene_rows: list[dict[str, Any]] = []
        scene_input_identities = [str(current_item["sha256"]), PARENT_SELECTION_SHA256]
        with np.load(current_path, allow_pickle=False) as current:
            if set(current.files) != expected_keys:
                anomaly_counts["current_key_contract"] += 1
            scene = str(current["scene"].item())
            fold = int(current["fold"].item())
            processed_scenes.add(scene)
            observed_folds.add(fold)
            if current_path.name != f"fold_{fold}_{scene[:24]}.npz":
                anomaly_counts["current_filename_identity"] += 1
            cache_path = upstream_root / "workload" / "scene_cache" / (
                f"training_{scene[:24]}.npz"
            )
            cache_relative = cache_path.relative_to(upstream_root).as_posix()
            cache_item = upstream_lookup.get(cache_relative)
            if cache_item is None:
                anomaly_counts["scene_cache_inventory"] += 1
                raise RuntimeError(f"scene cache absent from inventory: {cache_relative}")
            scene_input_identities.append(str(cache_item["sha256"]))
            with np.load(cache_path, allow_pickle=False) as cache:
                areas = cache["areas"].astype(np.float64)
            areas_valid = bool(
                areas.ndim == 2
                and areas.size > 0
                and np.isfinite(areas).all()
                and np.all(areas > 0.0)
            )
            if not areas_valid:
                anomaly_counts["scene_cache_areas"] += 1
            selected_action_detail: dict[str, Any] | None = None
            for cell in frozen.CELL_IDS:
                upstream_path = upstream_root / "workload" / "raw_predictions" \
                    / "haze4k_train_oof" / cell \
                    / f"fold_{fold}_{scene[:24]}.npz"
                upstream_relative = upstream_path.relative_to(upstream_root).as_posix()
                upstream_item = upstream_lookup.get(upstream_relative)
                if upstream_item is None:
                    anomaly_counts["upstream_prediction_inventory"] += 1
                    raise RuntimeError(
                        f"upstream prediction absent from inventory: {upstream_relative}"
                    )
                checkpoints = checkpoint_records(
                    upstream_root, upstream_lookup, cell, fold,
                )
                scene_input_identities.append(str(upstream_item["sha256"]))
                scene_input_identities.extend(item["sha256"] for item in checkpoints)
                comparison_file_id = f"{cell}|{fold}|{scene}"
                processed_comparison_files.add(comparison_file_id)
                metadata = {
                    "cell": cell,
                    "scene": scene,
                    "fold": fold,
                    "comparison_file_id": comparison_file_id,
                    "current_file_relpath": current_relative,
                    "current_file_sha256": current_item["sha256"],
                    "reference_file_relpath": upstream_relative,
                    "reference_file_sha256": upstream_item["sha256"],
                    "checkpoint_files": checkpoints,
                    "production_source_device_class": "cuda_sm89",
                    "comparison_device_path": "numpy_cpu_float64_subtraction",
                }
                row: dict[str, Any] = {
                    "cell": cell,
                    "scene": scene,
                    "fold": fold,
                    "current_file_relpath": current_relative,
                    "current_file_sha256": current_item["sha256"],
                    "upstream_file_relpath": upstream_relative,
                    "upstream_file_sha256": upstream_item["sha256"],
                    "checkpoint_files": checkpoints,
                }
                with np.load(upstream_path, allow_pickle=False) as upstream:
                    if set(upstream.files) != {
                        "utility", "risk", "utility_lower", "risk_upper", "actions",
                    }:
                        anomaly_counts["upstream_key_contract"] += 1
                    current_arrays = {
                        metric: current[f"{cell}_ensemble_{metric}"]
                        for metric in ("utility", "risk")
                    }
                    upstream_arrays = {
                        metric: upstream[metric] for metric in ("utility", "risk")
                    }
                    seed_arrays = {
                        metric: [
                            current[f"{cell}_seed_{seed_index}_{metric}"]
                            for seed_index in range(len(frozen.SEED_OFFSETS))
                        ]
                        for metric in ("utility", "risk")
                    }
                    for metric in ("utility", "risk"):
                        current_array = current_arrays[metric]
                        upstream_array = upstream_arrays[metric]
                        expected_shape = (
                            areas_valid
                            and current_array.ndim == 3
                            and current_array.shape[0] == len(frozen.ACTION_NAMES)
                            and current_array.shape[1:] == areas.shape
                        )
                        if current_array.shape != upstream_array.shape \
                                or not expected_shape:
                            anomaly_counts["upstream_current_shape"] += 1
                            row[f"{metric}_shape_match"] = False
                            continue
                        row[f"{metric}_shape_match"] = True
                        if current_array.dtype != np.float32 \
                                or upstream_array.dtype != np.float32:
                            anomaly_counts["upstream_current_dtype"] += 1
                        row[f"{metric}_upstream_current"] = tracker.add(
                            "upstream_vs_current",
                            metric,
                            current_array,
                            upstream_array,
                            metadata,
                        )
                        if any(item.shape != current_array.shape for item in seed_arrays[metric]):
                            anomaly_counts["saved_seed_shape"] += 1
                            row[f"{metric}_seed_shape_match"] = False
                            continue
                        row[f"{metric}_seed_shape_match"] = True
                        if any(item.dtype != np.float32 for item in seed_arrays[metric]):
                            anomaly_counts["saved_seed_dtype"] += 1
                        saved_seed_mean = np.mean(
                            np.stack(seed_arrays[metric]), axis=0,
                        ).astype(np.float32)
                        row[f"{metric}_saved_seed_ensemble"] = tracker.add(
                            "saved_seed_ensemble",
                            metric,
                            current_array,
                            saved_seed_mean,
                            metadata,
                        )

                    if cell == SELECTED_CELL \
                            and areas_valid \
                            and all(
                                current_arrays[metric].shape
                                == upstream_arrays[metric].shape
                                and current_arrays[metric].ndim == 3
                                and current_arrays[metric].shape[0]
                                == len(frozen.ACTION_NAMES)
                                for metric in ("utility", "risk")
                            ) \
                            and current_arrays["utility"].shape[1:] == areas.shape:
                        saved_actions = current["actions_selected"].astype(np.int8)
                        if saved_actions.shape != areas.shape:
                            anomaly_counts["selected_action_shape"] += 1
                            row["selected_action_shape_match"] = False
                            scene_rows.append(row)
                            continue
                        row["selected_action_shape_match"] = True
                        current_actions = frozen.qual_choose_setwise(
                            current_arrays["utility"],
                            current_arrays["risk"],
                            areas,
                            area_cap=SELECTED_AREA_CAP,
                        )
                        upstream_actions = frozen.qual_choose_setwise(
                            upstream_arrays["utility"],
                            upstream_arrays["risk"],
                            areas,
                            area_cap=SELECTED_AREA_CAP,
                        )
                        action_pairs = {
                            "saved_vs_current_replay": (saved_actions, current_actions),
                            "saved_vs_upstream_replay": (saved_actions, upstream_actions),
                            "current_vs_upstream_replay": (
                                current_actions, upstream_actions,
                            ),
                        }
                        pair_counts = {}
                        for name, (left, right) in action_pairs.items():
                            count = int(np.sum(left != right))
                            pair_counts[name] = count
                            action_counts[name] += count
                            if count:
                                action_scenes[name].add(scene)
                        current_risk = selected_weighted_value(
                            saved_actions, current_arrays["risk"], areas,
                        )
                        upstream_risk = selected_weighted_value(
                            saved_actions, upstream_arrays["risk"], areas,
                        )
                        current_utility = selected_weighted_value(
                            saved_actions, current_arrays["utility"], areas,
                        )
                        upstream_utility = selected_weighted_value(
                            saved_actions, upstream_arrays["utility"], areas,
                        )
                        if current_risk["area_weighted_value"] is not None:
                            selected_risk_differences.append(abs(
                                float(current_risk["area_weighted_value"])
                                - float(upstream_risk["area_weighted_value"])
                            ))
                            selected_utility_differences.append(abs(
                                float(current_utility["area_weighted_value"])
                                - float(upstream_utility["area_weighted_value"])
                            ))
                        selected_action_detail = {
                            "pair_mismatch_elements": pair_counts,
                            "saved_action_tiles": int(np.sum(saved_actions > 0)),
                            "current_weighted_q": current_risk,
                            "upstream_weighted_q": upstream_risk,
                            "current_raw_utility": current_utility,
                            "upstream_raw_utility": upstream_utility,
                        }
                        selected_action_replay_scenes.add(scene)
                        row["selected_action_replay"] = selected_action_detail
                scene_rows.append(row)

        raw_path = raw_root / f"fold_{fold}_{scene[:24]}.json"
        atomic_json(raw_path, {
            "schema_version": 1,
            "scene": scene,
            "fold": fold,
            "current_file_relpath": current_relative,
            "current_file_sha256": current_item["sha256"],
            "scene_cache_relpath": cache_relative,
            "scene_cache_sha256": cache_item["sha256"],
            "cells": scene_rows,
            "selected_action_replay": selected_action_detail,
        })
        raw_output_items.append({
            "artifact_class": "scene_cell_diagnostic",
            "relative_path": raw_path.relative_to(context.phase_output_path).as_posix(),
            "sha256": sha256_file(raw_path),
            "bytes": raw_path.stat().st_size,
        })
        record_completed_unit(
            context,
            unit_id=f"scene_{fold}_{scene[:24]}",
            input_sha256=sha256_text("|".join(sorted(set(scene_input_identities)))),
            output_relpath=raw_path.relative_to(context.phase_output_path).as_posix(),
        )
        if scene_index % 25 == 0 or scene_index == len(current_files):
            write_workload_progress(
                context,
                completed_units=1 + scene_index,
                stage="finite_population_scene_array_census",
            )

    quantile_rows = tracker.quantile_rows()
    threshold_rows = tracker.threshold_rows()
    mismatch_scenes = set()
    for metric in ("utility", "risk"):
        mismatch_scenes.update(
            tracker.scenes[("upstream_vs_current", metric, 1.0e-6)]
        )
    prevalence = frozen.wilson(
        len(mismatch_scenes), EXPECTED_SCENES, family_size=2,
    )
    precision_half_width = max(
        float(prevalence["estimate"]) - float(prevalence["lower"]),
        float(prevalence["upper"]) - float(prevalence["estimate"]),
    )
    maximum_discrepancy = tracker.maximum("upstream_vs_current")
    maximum_seed_inconsistency = tracker.maximum("saved_seed_ensemble")
    nonfinite_count = sum(tracker.nonfinite_elements.values())
    structural_anomaly_count = sum(anomaly_counts.values())
    action_mismatch_count = sum(action_counts.values())

    identity_checks = {
        **parent_identity_checks,
        "current_raw_inventory": current_verification["matched"],
        "upstream_raw_inventory": upstream_verification["matched"],
    }
    coverage_checks = {
        "current_oof_scenes": len(current_files) == EXPECTED_SCENES,
        "unique_scene_identities": len(processed_scenes) == EXPECTED_SCENES,
        "comparison_files": len(processed_comparison_files)
        == EXPECTED_COMPARISON_FILES,
        "factorial_cells": EXPECTED_CELLS == len(frozen.CELL_IDS),
        "folds": observed_folds == set(range(frozen.OOF_FOLDS)),
        "selected_action_replays": len(selected_action_replay_scenes)
        == EXPECTED_SCENES,
        "scene_diagnostic_files": len(raw_output_items) == EXPECTED_SCENES,
    }
    evidence_identity_outcome = "pass" if all(identity_checks.values()) else "fail"
    coverage_outcome = "pass" if all(coverage_checks.values()) else "fail"
    actions_decision_invariant = action_mismatch_count == 0
    terminal_gates_decision_invariant = (
        actions_decision_invariant
        and parent_identity_checks["selected_candidate"]
        and parent_identity_checks["selection_time_isolation"]
    )
    if evidence_identity_outcome != "pass" or coverage_outcome != "pass":
        materiality_outcome = "invalid"
        discrepancy_class = "input_identity_or_coverage_invalid"
    elif structural_anomaly_count or nonfinite_count or action_mismatch_count \
            or maximum_seed_inconsistency > 1.0e-6 \
            or maximum_discrepancy > 1.0e-4:
        materiality_outcome = "unfavorable"
        discrepancy_class = "material_identity_failure"
    elif 1.0e-6 < maximum_discrepancy <= 1.0e-5 \
            and terminal_gates_decision_invariant:
        materiality_outcome = "favorable"
        discrepancy_class = "bounded_decision_invariant_discrepancy"
    elif maximum_discrepancy <= 1.0e-6:
        materiality_outcome = "indeterminate"
        discrepancy_class = "parent_failure_not_reproduced"
    else:
        materiality_outcome = "indeterminate"
        discrepancy_class = "intermediate_discrepancy_requires_external_evidence"
    precision_outcome = (
        "met" if len(processed_scenes) == EXPECTED_SCENES
        and precision_half_width <= PRECISION_TARGET_HALF_WIDTH else "unmet"
    )
    gate_outcomes = {
        "evidence_identity": evidence_identity_outcome,
        "coverage": coverage_outcome,
        "reconstruction_materiality": materiality_outcome,
        "precision": precision_outcome,
    }

    raw_inventory = frozen.compact_inventory(raw_output_items)
    quantiles_path = output_file(
        context, "haze4k_frozen_output_tail_controller_evidence_review_quantiles.csv",
    )
    thresholds_path = output_file(
        context, "haze4k_frozen_output_tail_controller_evidence_review_threshold_counts.csv",
    )
    worst_path = output_file(
        context, "haze4k_frozen_output_tail_controller_evidence_review_worst_points.json",
    )
    action_path = output_file(
        context, "haze4k_frozen_output_tail_controller_evidence_review_action_replay.json",
    )
    inventory_path = output_file(
        context, "haze4k_frozen_output_tail_controller_evidence_review_inventory.json",
    )
    gate_summary_path = output_file(
        context, "haze4k_frozen_output_tail_controller_evidence_review_gate_summary.json",
    )
    summary_output_path = output_file(
        context, "haze4k_frozen_output_tail_controller_evidence_review_summary.json",
    )
    write_csv(quantiles_path, quantile_rows)
    write_csv(thresholds_path, threshold_rows)
    atomic_json(worst_path, {
        "schema_version": 1,
        "comparison_level": "saved_ensemble",
        "production_source_device_class": "cuda_sm89",
        "comparison_device_path": "numpy_cpu_float64_subtraction",
        "seed_level_reconstruction_from_upstream": "not_available_from_saved_upstream_ensemble_files",
        "worst_points": [tracker.worst[key] for key in sorted(tracker.worst)],
        "nonfinite_elements": {
            f"{source}:{metric}": count
            for (source, metric), count in sorted(tracker.nonfinite_elements.items())
        },
        "structural_anomaly_counts": dict(sorted(anomaly_counts.items())),
    })
    action_summary = {
        "schema_version": 1,
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "selected_cell": SELECTED_CELL,
        "risk_method": "weighted_q",
        "utility_method": "raw",
        "selector": "setwise",
        "area_cap": SELECTED_AREA_CAP,
        "selection_freeze_sha256": PARENT_SELECTION_SHA256,
        "selection_frozen_before_formal_oof": parent_identity_checks[
            "selection_time_isolation"
        ],
        "pair_mismatch_elements": dict(sorted(action_counts.items())),
        "pair_mismatch_scenes": {
            key: len(value) for key, value in sorted(action_scenes.items())
        },
        "calibrated_risk_identity": "weighted_q_is_the_saved_raw_risk_array",
        "maximum_selected_area_weighted_risk_difference": max(
            selected_risk_differences, default=0.0,
        ),
        "maximum_selected_area_weighted_utility_difference": max(
            selected_utility_differences, default=0.0,
        ),
        "actions_decision_invariant": actions_decision_invariant,
        "terminal_scientific_gates_decision_invariant": terminal_gates_decision_invariant,
        "parent_gate_outcomes": closeout.get("gate_outcomes"),
        "parent_qualification_retroactively_passed": False,
    }
    atomic_json(action_path, action_summary)
    atomic_json(inventory_path, {
        "schema_version": 1,
        "input_verification": input_verification,
        "raw_scene_diagnostic_inventory": raw_inventory,
        "raw_retention_scope": "cloud_only",
        "github_retention_scope": "compact_aggregate_gate_evidence_only",
    })
    gate_summary = {
        "schema_version": 1,
        "gate_outcomes": gate_outcomes,
        "discrepancy_class": discrepancy_class,
        "maximum_upstream_current_absolute_error": maximum_discrepancy,
        "maximum_saved_seed_ensemble_absolute_error": maximum_seed_inconsistency,
        "nonfinite_element_count": nonfinite_count,
        "structural_anomaly_count": structural_anomaly_count,
        "action_mismatch_count": action_mismatch_count,
        "scene_mismatch_prevalence": prevalence,
        "precision_half_width": precision_half_width,
        "precision_target_half_width": PRECISION_TARGET_HALF_WIDTH,
        "actions_decision_invariant": actions_decision_invariant,
        "terminal_scientific_gates_decision_invariant": terminal_gates_decision_invariant,
    }
    atomic_json(gate_summary_path, gate_summary)
    summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "scope": "receipt-bound frozen-output tail-controller evidence review",
        "evidence_role": "development_screening",
        "independent_unit": "original_clear_scene",
        "comparison_population": {
            "scenes": len(processed_scenes),
            "cells": len(frozen.CELL_IDS),
            "ensemble_comparison_files": len(processed_comparison_files),
            "metrics": ["utility", "risk"],
            "comparison_level": "saved_ensemble",
        },
        "input_identity_checks": identity_checks,
        "coverage_checks": coverage_checks,
        "gate_outcomes": gate_outcomes,
        "discrepancy_class": discrepancy_class,
        "maximum_upstream_current_absolute_error": maximum_discrepancy,
        "maximum_saved_seed_ensemble_absolute_error": maximum_seed_inconsistency,
        "scene_mismatch_prevalence": prevalence,
        "precision": {
            "method": "binomial_worst_case",
            "critical_value": PRECISION_CRITICAL_VALUE,
            "target_half_width": PRECISION_TARGET_HALF_WIDTH,
            "required_scenes": EXPECTED_REQUIRED_SCENES,
            "available_scenes": EXPECTED_SCENES,
            "observed_half_width": precision_half_width,
            "met": precision_outcome == "met",
        },
        "action_replay": action_summary,
        "production_comparison_boundary": {
            "upstream_and_current_prediction_source": "cuda_sm89",
            "saved_array_dtype_expected": "float32",
            "comparison_path": "NumPy CPU with float64 subtraction",
            "single_seed_upstream_reconstruction": "not available from ensemble-only upstream files",
        },
        "time_isolation": {
            "selection_freeze_sha256": PARENT_SELECTION_SHA256,
            "selection_frozen_before_parent_formal_oof": parent_identity_checks[
                "selection_time_isolation"
            ],
            "review_reran_candidate_selection": False,
            "review_read_calibration_predictions": False,
            "review_changed_thresholds_or_gates": False,
        },
        "interpretation_boundary": {
            "parent_qualification_retroactively_passed": False,
            "bounded_discrepancy_authorization_ceiling": (
                "decision-focused controller contract authoring only"
            ),
            "scientific_utility_and_activation_parent_outcomes_unchanged": True,
        },
        "compact_evidence_files": {
            "quantiles": quantiles_path.name,
            "threshold_counts": thresholds_path.name,
            "worst_points": worst_path.name,
            "action_replay": action_path.name,
            "inventory": inventory_path.name,
            "gate_summary": gate_summary_path.name,
        },
        "raw_scene_diagnostics": {
            "retention": "cloud_only",
            "directory": raw_root.name,
            "file_count": len(raw_output_items),
            "inventory_sha256": raw_inventory["inventory_sha256"],
        },
        "limitations": [
            "All arrays are development-screening evidence; no confirmation, canary, locked-test, or cross-domain data were read.",
            "The 600 original-clear scenes are the only independent units; cells, files, tiles, actions, and elements do not increase n.",
            "The upstream files save ensemble predictions, so an upstream single-seed discrepancy cannot be reconstructed from those files alone.",
            "A bounded decision-invariant discrepancy cannot retroactively qualify the parent controller because utility and activation were independently unfavorable.",
        ],
        "marker": "HAZE4K_FROZEN_OUTPUT_TAIL_CONTROLLER_EVIDENCE_REVIEW_COMPLETE",
    }
    atomic_json(summary_output_path, summary)

    compact_paths = {
        "summary": summary_output_path,
        "gate_summary": gate_summary_path,
        "quantiles": quantiles_path,
        "threshold_counts": thresholds_path,
        "worst_points": worst_path,
        "action_replay": action_path,
        "inventory": inventory_path,
    }
    finalization_path = output_file(context, "finalization_unit.json")
    compact_sha256 = {
        key: sha256_file(path) for key, path in sorted(compact_paths.items())
    }
    atomic_json(finalization_path, {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "discrepancy_class": discrepancy_class,
        "gate_outcomes": gate_outcomes,
        "compact_evidence_sha256": compact_sha256,
        "raw_scene_diagnostic_inventory_sha256": raw_inventory["inventory_sha256"],
    })
    record_completed_unit(
        context,
        unit_id="finalization",
        input_sha256=sha256_text("|".join([
            discrepancy_class,
            raw_inventory["inventory_sha256"],
            *[f"{key}:{value}" for key, value in sorted(compact_sha256.items())],
        ])),
        output_relpath=finalization_path.name,
    )
    if len(load_completed_unit_ledger(context)) != TOTAL_UNITS:
        raise RuntimeError("evidence-review completed-unit ledger is incomplete")
    write_workload_progress(
        context, completed_units=TOTAL_UNITS, stage="evidence_review_gate_finalization",
    )
    write_gate_result(
        context,
        gate_outcomes=gate_outcomes,
        details={
            "parent_route_id": PARENT_ROUTE_ID,
            "parent_route_commit": PARENT_ROUTE_COMMIT,
            "parent_receipt": PARENT_RECEIPT,
            "selected_candidate_id": SELECTED_CANDIDATE_ID,
            "discrepancy_class": discrepancy_class,
            "maximum_upstream_current_absolute_error": maximum_discrepancy,
            "maximum_saved_seed_ensemble_absolute_error": maximum_seed_inconsistency,
            "scene_mismatch_count": len(mismatch_scenes),
            "independent_scene_count": EXPECTED_SCENES,
            "precision_half_width": precision_half_width,
            "completed_unit_ledger_count": TOTAL_UNITS,
            "action_mismatch_count": action_mismatch_count,
            "summary_file": summary_output_path.name,
            "gate_summary_file": gate_summary_path.name,
            "quantiles_file": quantiles_path.name,
            "threshold_counts_file": thresholds_path.name,
            "worst_points_file": worst_path.name,
            "action_replay_file": action_path.name,
            "inventory_file": inventory_path.name,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.phase == "contract":
        contract(arguments.context)
    else:
        run(arguments.context)


if __name__ == "__main__":
    main()
