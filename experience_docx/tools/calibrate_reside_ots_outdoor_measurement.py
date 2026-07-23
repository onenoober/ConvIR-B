#!/usr/bin/env python3
"""Qualify an OTS synthetic-outdoor local optical-thickness field."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from route_program_api import (
    asset_path,
    load_context,
    output_file,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
    write_workload_progress,
)


EXPECTED_CLEAR = 8970
EXPECTED_DEPTH = 8970
EXPECTED_HAZE = 313950
EXPECTED_EXCLUSIONS = 964
EXPECTED_ELIGIBLE = 8006
DEFINITION_SCENES = 375
VALIDATION_SCENES = 625
SELECTED_SCENES = DEFINITION_SCENES + VALIDATION_SCENES
VARIANTS_PER_SCENE = 35
IMAGE_SIZE = 96
PERMUTATION_SHIFT = 313
TOTAL_UNITS = 37000
SELECTION_SALT = "reside-ots-outdoor-measurement-v1"
AIRLIGHT_GRID = (0.80, 0.85, 0.90, 0.95, 1.00)
BETA_GRID = (0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20)
EXPECTED_EXCLUSION_DIGEST = "5ba09f2b49f0d8d1846fb3638e58a93d340de26be977ea84efb5526e14c39ecd"


def digest_lines(lines: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


def image_files(path: Path) -> list[Path]:
    suffixes = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def parse_variant(path: Path) -> tuple[str, float, float]:
    parts = path.stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"invalid OTS haze filename: {path.name}")
    scene_id, first, second = parts
    values = float(first), float(second)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite OTS haze filename token: {path.name}")
    return scene_id, values[0], values[1]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int]:
    if total <= 0:
        return {"successes": successes, "total": total, "estimate": 0.0, "lower": 0.0, "upper": 1.0}
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "estimate": estimate,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
    }


def image_array(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0


def reconstruction_rmse(
    clear: np.ndarray, haze: np.ndarray, depth: np.ndarray, airlight: float, beta: float,
) -> float:
    transmission = np.exp((-float(beta) * depth).astype(np.float64)).astype(np.float32)
    reconstructed = clear * transmission[..., None] + float(airlight) * (1.0 - transmission[..., None])
    return float(np.sqrt(np.mean(np.square(haze - reconstructed), dtype=np.float64)))


def scene_measurement(
    scene_id: str,
    clear_path: Path,
    variants: list[Path],
    depth: np.ndarray,
    permuted_depth: np.ndarray,
) -> dict[str, Any]:
    clear = image_array(clear_path)
    true_errors: list[float] = []
    reversed_errors: list[float] = []
    permuted_errors: list[float] = []
    beta_errors: dict[float, list[float]] = defaultdict(list)
    beta_permuted: dict[float, list[float]] = defaultdict(list)
    observed_grid: set[tuple[float, float]] = set()
    for haze_path in variants:
        observed_scene, airlight, beta = parse_variant(haze_path)
        if observed_scene != scene_id:
            raise ValueError(f"variant scene mismatch: {haze_path.name}")
        observed_grid.add((round(airlight, 2), round(beta, 2)))
        haze = image_array(haze_path)
        true_error = reconstruction_rmse(clear, haze, depth, airlight, beta)
        reversed_error = reconstruction_rmse(clear, haze, depth, beta, airlight)
        permuted_error = reconstruction_rmse(clear, haze, permuted_depth, airlight, beta)
        true_errors.append(true_error)
        reversed_errors.append(reversed_error)
        permuted_errors.append(permuted_error)
        beta_errors[round(beta, 2)].append(true_error)
        beta_permuted[round(beta, 2)].append(permuted_error)
    expected_grid = {(round(a, 2), round(b, 2)) for a in AIRLIGHT_GRID for b in BETA_GRID}
    if len(variants) != VARIANTS_PER_SCENE or observed_grid != expected_grid:
        raise ValueError(f"incomplete OTS parameter grid for scene {scene_id}")
    true_median = float(np.median(true_errors))
    true_q95 = float(np.quantile(true_errors, 0.95))
    reversed_median = float(np.median(reversed_errors))
    permuted_median = float(np.median(permuted_errors))
    depth_q05, depth_q95 = (float(value) for value in np.quantile(depth, [0.05, 0.95]))
    tau_range_heavy = max(BETA_GRID) * (depth_q95 - depth_q05)
    reconstruction_pass = true_median <= 0.025 and true_q95 <= 0.050
    heterogeneity_pass = tau_range_heavy >= 0.050
    negative_control_improvement = permuted_median - true_median
    negative_control_pass = negative_control_improvement >= 0.005
    return {
        "scene_id": scene_id,
        "true_rmse_median": true_median,
        "true_rmse_q95": true_q95,
        "reversed_rmse_median": reversed_median,
        "permuted_rmse_median": permuted_median,
        "negative_control_improvement": negative_control_improvement,
        "depth_q05": depth_q05,
        "depth_q95": depth_q95,
        "tau_range_heavy": tau_range_heavy,
        "reconstruction_pass": reconstruction_pass,
        "heterogeneity_pass": heterogeneity_pass,
        "negative_control_pass": negative_control_pass,
        "joint_pass": reconstruction_pass and heterogeneity_pass and negative_control_pass,
        "beta_true_rmse_median": {format(beta, ".2f"): float(np.median(values)) for beta, values in sorted(beta_errors.items())},
        "beta_permuted_rmse_median": {format(beta, ".2f"): float(np.median(values)) for beta, values in sorted(beta_permuted.items())},
    }


def _octave_quote(path: Path) -> str:
    return str(path).replace("'", "''")


def octave_export_depths(
    entries: list[tuple[Path, int, int, Path]], script_path: Path, *, timeout: int,
) -> None:
    lines = ["more off;"]
    for source, height, width, destination in entries:
        lines.extend([
            f"s=load('{_octave_quote(source)}');",
            "if ~isfield(s,'depth'), error('missing depth field'); end;",
            "d=double(s.depth);",
            f"if size(d,1)=={width} && size(d,2)=={height}, d=d'; end;",
            f"if size(d,1)~={height} || size(d,2)~={width}, error('depth shape mismatch'); end;",
            f"[xq,yq]=meshgrid(linspace(1,size(d,2),{IMAGE_SIZE}),linspace(1,size(d,1),{IMAGE_SIZE}));",
            "small=interp2(d,xq,yq,'linear');",
            "if any(~isfinite(small(:))), error('nonfinite depth'); end;",
            f"fid=fopen('{_octave_quote(destination)}','wb');",
            "if fid<0, error('depth output open failed'); end;",
            "fwrite(fid,single(small),'single'); fclose(fid);",
        ])
    lines.append("disp('OTS_OCTAVE_DEPTH_EXPORT_OK');")
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    completed = subprocess.run(
        ["/usr/bin/octave", "--quiet", "--no-gui", str(script_path)],
        text=True, capture_output=True, timeout=timeout, check=False,
    )
    if completed.returncode or "OTS_OCTAVE_DEPTH_EXPORT_OK" not in completed.stdout:
        detail = (completed.stdout + completed.stderr)[-4096:]
        raise RuntimeError(f"Octave depth export failed: {detail}")
    expected_bytes = IMAGE_SIZE * IMAGE_SIZE * 4
    for _, _, _, destination in entries:
        if not destination.is_file() or destination.stat().st_size != expected_bytes:
            raise RuntimeError(f"invalid exported depth: {destination}")


def load_depth(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size != IMAGE_SIZE * IMAGE_SIZE or not np.isfinite(values).all():
        raise ValueError(f"invalid depth transport: {path}")
    return values.reshape(IMAGE_SIZE, IMAGE_SIZE, order="F")


def aggregate_numeric(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "q05", "median", "mean", "q95", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def aggregate_split(records: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    joint = sum(bool(item["joint_pass"]) for item in records)
    return {
        "expected_scene_count": expected,
        "completed_scene_count": len(records),
        "joint_qualification": wilson(joint, expected),
        "reconstruction_pass": wilson(sum(bool(item["reconstruction_pass"]) for item in records), expected),
        "heterogeneity_pass": wilson(sum(bool(item["heterogeneity_pass"]) for item in records), expected),
        "negative_control_pass": wilson(sum(bool(item["negative_control_pass"]) for item in records), expected),
        "true_rmse_median": aggregate_numeric([float(item["true_rmse_median"]) for item in records]),
        "true_rmse_q95": aggregate_numeric([float(item["true_rmse_q95"]) for item in records]),
        "reversed_rmse_median": aggregate_numeric([float(item["reversed_rmse_median"]) for item in records]),
        "permuted_rmse_median": aggregate_numeric([float(item["permuted_rmse_median"]) for item in records]),
        "negative_control_improvement": aggregate_numeric([float(item["negative_control_improvement"]) for item in records]),
        "tau_range_heavy": aggregate_numeric([float(item["tau_range_heavy"]) for item in records]),
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    size = IMAGE_SIZE
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    clear = np.stack((xx / size, yy / size, (xx + yy) / (2 * size)), axis=-1)
    depth = 1.0 + xx / size + 0.5 * yy / size
    permuted = np.flip(depth, axis=1).copy()
    airlight, beta = 0.9, 0.16
    transmission = np.exp(-beta * depth).astype(np.float32)
    haze = clear * transmission[..., None] + airlight * (1.0 - transmission[..., None])
    true_rmse = reconstruction_rmse(clear, haze, depth, airlight, beta)
    reversed_rmse = reconstruction_rmse(clear, haze, depth, beta, airlight)
    permuted_rmse = reconstruction_rmse(clear, haze, permuted, airlight, beta)
    clear_path = output_file(context, "fixture_clear.png")
    Image.fromarray(np.uint8(np.clip(clear * 255.0, 0, 255))).save(clear_path)
    variant_paths: list[Path] = []
    for fixture_airlight in AIRLIGHT_GRID:
        for fixture_beta in BETA_GRID:
            fixture_transmission = np.exp(-fixture_beta * depth).astype(np.float32)
            fixture_haze = (
                clear * fixture_transmission[..., None]
                + fixture_airlight * (1.0 - fixture_transmission[..., None])
            )
            variant_path = output_file(
                context, f"fixture_{fixture_airlight:.2f}_{fixture_beta:.2f}.png",
            )
            Image.fromarray(np.uint8(np.clip(fixture_haze * 255.0, 0, 255))).save(variant_path)
            variant_paths.append(variant_path)
    fixture_scene = scene_measurement(
        "fixture", clear_path, variant_paths, depth, permuted,
    )

    synthetic_mat = output_file(context, "synthetic_depth.mat")
    create = subprocess.run(
        ["/usr/bin/octave", "--quiet", "--no-gui", "--eval",
         f"depth=reshape(1:{size * size},{size},{size}); save('-hdf5','{_octave_quote(synthetic_mat)}','depth');"],
        text=True, capture_output=True, timeout=30, check=False,
    )
    synthetic_raw = output_file(context, "synthetic_depth.f32")
    transport_ok = create.returncode == 0
    if transport_ok:
        octave_export_depths(
            [(synthetic_mat, size, size, synthetic_raw)],
            output_file(context, "synthetic_export.m"), timeout=30,
        )
        transported = load_depth(synthetic_raw)
        transport_ok = transported.shape == (size, size) and np.isfinite(transported).all()

    checks = {
        "octave_depth_transport": bool(transport_ok),
        "optical_reconstruction_exact": true_rmse <= 1e-6,
        "reversed_mapping_separates": reversed_rmse > 0.10,
        "permuted_depth_separates": permuted_rmse - true_rmse > 0.005,
        "wilson_reference": abs(float(wilson(563, 625)["lower"]) - 0.8748596417) < 1e-6,
        "scene_measurement_path": (
            fixture_scene["true_rmse_median"] <= 0.025
            and fixture_scene["reversed_rmse_median"] > fixture_scene["true_rmse_median"]
            and fixture_scene["tau_range_heavy"] >= 0.05
        ),
    }
    write_contract_result(
        context, checks=checks,
        engineering={
            "mode": "cpu_exact",
            "device": "cpu",
            "fixture": {"batch": 35, "channels": 3, "height": size, "width": size},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    reside = asset_path(context, "reside_root", kind="directory")
    parent_closeout_path = asset_path(context, "parent_closeout", kind="file")
    exclusions_path = asset_path(context, "parent_exclusions", kind="file")
    parent = json.loads(parent_closeout_path.read_text(encoding="utf-8"))
    exclusions = sorted({line.strip() for line in exclusions_path.read_text(encoding="utf-8").splitlines() if line.strip()})
    parent_identity = (
        parent.get("state") == "COMPLETED_GATE_PASS"
        and parent.get("decision") == "OTS_TARGETED_GEOMETRY_PASS"
        and parent.get("authorizes") == "OTS_OUTDOOR_MEASUREMENT_DESIGN"
        and parent.get("details", {}).get("deduplicated_exclusion_count") == EXPECTED_EXCLUSIONS
        and parent.get("details", {}).get("eligible_ots_scenes") == EXPECTED_ELIGIBLE
        and parent.get("details", {}).get("exclusion_digest") == EXPECTED_EXCLUSION_DIGEST
        and len(exclusions) == EXPECTED_EXCLUSIONS
        and digest_lines(exclusions) == EXPECTED_EXCLUSION_DIGEST
    )

    clear_paths = image_files(reside / "official/OTS_ALPHA/clear_images")
    depth_paths = sorted((reside / "official/OTS_ALPHA/depth").glob("*.mat"))
    haze_paths = image_files(reside / "official/OTS_ALPHA/OTS")
    clear_by_id = {path.stem: path for path in clear_paths}
    depth_by_id = {path.stem: path for path in depth_paths}
    haze_by_id: dict[str, list[Path]] = defaultdict(list)
    observed_grid_by_id: dict[str, set[tuple[float, float]]] = defaultdict(set)
    parse_failures = 0
    for path in haze_paths:
        try:
            scene_id, first, second = parse_variant(path)
        except ValueError:
            parse_failures += 1
            continue
        haze_by_id[scene_id].append(path)
        observed_grid_by_id[scene_id].add((round(first, 2), round(second, 2)))
    expected_grid = {(round(a, 2), round(b, 2)) for a in AIRLIGHT_GRID for b in BETA_GRID}
    clear_ids = set(clear_by_id)
    dataset_identity = (
        len(clear_paths) == EXPECTED_CLEAR
        and len(depth_paths) == EXPECTED_DEPTH
        and len(haze_paths) == EXPECTED_HAZE
        and parse_failures == 0
        and clear_ids == set(depth_by_id) == set(haze_by_id)
        and all(len(haze_by_id[item]) == VARIANTS_PER_SCENE for item in clear_ids)
        and all(observed_grid_by_id[item] == expected_grid for item in clear_ids)
    )
    eligible = sorted(clear_ids - set(exclusions))
    dataset_identity = dataset_identity and len(eligible) == EXPECTED_ELIGIBLE
    selected = sorted(eligible, key=lambda item: hashlib.sha256(f"{SELECTION_SALT}|{item}".encode("utf-8")).hexdigest())[:SELECTED_SCENES]
    definition_ids = selected[:DEFINITION_SCENES]
    validation_ids = selected[DEFINITION_SCENES:]
    split_overlap = set(definition_ids) & set(validation_ids)
    output_file(context, "selected_scene_ids.txt").write_text(
        "".join(f"definition\t{item}\n" for item in definition_ids)
        + "".join(f"validation\t{item}\n" for item in validation_ids),
        encoding="utf-8",
    )
    write_workload_progress(context, completed_units=1000, stage="dataset_identity")

    with Image.open(clear_by_id[selected[0]]) as first_image:
        _ = first_image.size
    raw_root = output_file(context, "depth96")
    raw_root.mkdir()
    entries: list[tuple[Path, int, int, Path]] = []
    for scene_id in selected:
        with Image.open(clear_by_id[scene_id]) as image:
            width, height = image.size
        entries.append((depth_by_id[scene_id], height, width, raw_root / f"{scene_id}.f32"))
    octave_export_depths(entries, output_file(context, "export_depths.m"), timeout=1800)
    depths = {scene_id: load_depth(raw_root / f"{scene_id}.f32") for scene_id in selected}
    write_workload_progress(context, completed_units=2000, stage="depth_export")

    permuted_validation = validation_ids[PERMUTATION_SHIFT:] + validation_ids[:PERMUTATION_SHIFT]
    permuted_by_id = {scene_id: depths[other] for scene_id, other in zip(validation_ids, permuted_validation)}
    for scene_id in definition_ids:
        permuted_by_id[scene_id] = depths[definition_ids[(definition_ids.index(scene_id) + PERMUTATION_SHIFT) % DEFINITION_SCENES]]

    def measure(scene_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            return scene_id, scene_measurement(
                scene_id, clear_by_id[scene_id], haze_by_id[scene_id], depths[scene_id], permuted_by_id[scene_id],
            ), None
        except Exception as exc:  # bounded per-scene exclusion remains visible
            return scene_id, None, str(exc)[:512]

    workers = int(os.environ.get("CONVIR_ROUTE_OTS_WORKERS", "8"))
    measured: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (scene_id, record, error) in enumerate(executor.map(measure, selected), start=1):
            if record is None:
                failures.append({"scene_id": scene_id, "reason": error or "unknown"})
            else:
                measured[scene_id] = record
            if index % 100 == 0 or index == SELECTED_SCENES:
                write_workload_progress(
                    context, completed_units=2000 + index * VARIANTS_PER_SCENE, stage="scene_measurement",
                )

    definition_records = [measured[item] for item in definition_ids if item in measured]
    validation_records = [measured[item] for item in validation_ids if item in measured]
    definition_true = [float(item["true_rmse_median"]) for item in definition_records]
    definition_reversed = [float(item["reversed_rmse_median"]) for item in definition_records]
    definition_mapping = (
        len(definition_records) == DEFINITION_SCENES
        and float(np.median(definition_true)) <= 0.025
        and float(np.median(definition_true)) <= 0.5 * float(np.median(definition_reversed))
    )
    validation_aggregate = aggregate_split(validation_records, VALIDATION_SCENES)
    definition_aggregate = aggregate_split(definition_records, DEFINITION_SCENES)
    validation_complete = len(validation_records) == VALIDATION_SCENES
    joint_lower = float(validation_aggregate["joint_qualification"]["lower"])
    split_disjoint = not split_overlap
    gates = {
        "parent_authorization": parent_identity,
        "dataset_identity": dataset_identity,
        "definition_token_mapping": definition_mapping,
        "validation_scene_count": validation_complete,
        "joint_scene_qualification": joint_lower >= 0.90,
        "split_disjointness": split_disjoint,
    }
    if not parent_identity or not dataset_identity:
        state, decision, authorizes = ("COMPLETED_INCONCLUSIVE", "OTS_OUTDOOR_MEASUREMENT_INCONCLUSIVE", "OTS_MEASUREMENT_SUPPLEMENT_ONLY")
        gate_reasons = ["parent authorization or frozen OTS dataset identity changed"]
    elif all(gates.values()):
        state, decision, authorizes = ("COMPLETED_GATE_PASS", "OTS_OUTDOOR_MEASUREMENT_PASS", "OTS_BASELINE_BEHAVIOR_MEASUREMENT_DESIGN")
        gate_reasons = ["all frozen source-identity, token-mapping, coverage, reconstruction, heterogeneity, negative-control and precision gates passed"]
    else:
        state, decision, authorizes = ("COMPLETED_GATE_FAIL", "OTS_OUTDOOR_MEASUREMENT_FAIL", "NONE")
        gate_reasons = [key for key, passed in gates.items() if not passed]

    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "measurement_scope": "synthetic outdoor OTS optical thickness only",
        "dataset_identity": {
            "root": str(reside),
            "parent_identity_match": parent_identity,
            "count_and_grid_integrity": dataset_identity,
            "clear_scenes": len(clear_paths),
            "depth_files": len(depth_paths),
            "haze_variants": len(haze_paths),
            "excluded_scenes": len(exclusions),
            "eligible_scenes": len(eligible),
            "exclusion_digest": digest_lines(exclusions),
        },
        "sampling": {
            "selection_salt_sha256": hashlib.sha256(SELECTION_SALT.encode("utf-8")).hexdigest(),
            "selected_scene_digest": digest_lines(selected),
            "definition_scene_digest": digest_lines(definition_ids),
            "validation_scene_digest": digest_lines(validation_ids),
            "definition_scenes": len(definition_ids),
            "validation_scenes": len(validation_ids),
            "variants_per_scene": VARIANTS_PER_SCENE,
            "effective_independent_unit": "original OTS clear scene",
            "permutation_shift": PERMUTATION_SHIFT,
        },
        "measurement": {
            "tau": "beta times official OTS scene depth; synthetic optical thickness only",
            "transmission": "exp(-beta times depth)",
            "reconstruction": "I_hat=J*transmission+A*(1-transmission)",
            "filename_mapping": "scene_airlight_beta",
            "airlight_grid": list(AIRLIGHT_GRID),
            "beta_grid": list(BETA_GRID),
            "negative_control": f"validation depth circularly shifted by {PERMUTATION_SHIFT} independent scenes",
        },
        "aggregates": {"definition": definition_aggregate, "validation": validation_aggregate},
        "definition_mapping": {
            "completed_scenes": len(definition_records),
            "a_first_beta_second_median_rmse": float(np.median(definition_true)) if definition_true else None,
            "reversed_mapping_median_rmse": float(np.median(definition_reversed)) if definition_reversed else None,
            "passed": definition_mapping,
        },
        "failures": failures[:20],
        "failure_count": len(failures),
        "gates": gates,
        "terminal": {"state": state, "decision": decision, "authorizes": authorizes, "gate_reasons": gate_reasons},
        "limitations": [
            "Passing supports an L1 synthetic-outdoor optical-thickness measurement claim only.",
            "OTS depth and beta are generation fields, not validated real-haze restoration-demand labels.",
            "The permuted-depth control tests spatial information beyond global airlight and beta, but it is not causal evidence.",
            "No model, training, inference, SOTS, Haze4K outcome, NH-HAZE, confirmation, canary, or locked-test data were used.",
        ],
        "marker": "RESIDE_OTS_OUTDOOR_MEASUREMENT_V1_COMPLETE",
    }
    output_file(context, "measurement_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    fields = ["split", "beta", "independent_scenes", "true_rmse_median", "true_rmse_q95", "permuted_rmse_median", "negative_control_improvement_median"]
    with output_file(context, "aggregate_strata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split, records in (("definition", definition_records), ("validation", validation_records)):
            for beta in BETA_GRID:
                key = format(beta, ".2f")
                true_values = [float(item["beta_true_rmse_median"][key]) for item in records]
                permuted_values = [float(item["beta_permuted_rmse_median"][key]) for item in records]
                writer.writerow({
                    "split": split, "beta": key, "independent_scenes": len(records),
                    "true_rmse_median": float(np.median(true_values)) if true_values else "",
                    "true_rmse_q95": float(np.quantile(true_values, 0.95)) if true_values else "",
                    "permuted_rmse_median": float(np.median(permuted_values)) if permuted_values else "",
                    "negative_control_improvement_median": float(np.median(np.asarray(permuted_values) - np.asarray(true_values))) if true_values else "",
                })
    write_workload_progress(context, completed_units=TOTAL_UNITS, stage="measurement_finalize")
    write_run_result(
        context, state=state, decision=decision, authorizes=authorizes,
        details={
            "summary_file": "measurement_summary.json",
            "validation_scenes": len(validation_records),
            "joint_qualification_lower_95": joint_lower,
            "definition_token_mapping_passed": definition_mapping,
            "gate_reasons": gate_reasons,
            "synthetic_outdoor_only": True,
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
