#!/usr/bin/env python3
"""Calibrate the ITS local restoration-direction measurement without model use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
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


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
RESAMPLE = Image.Resampling.BILINEAR
SAMPLE_SALT = "reside-its-local-measurement-v1-scene-salt"
PERMUTATION_SHIFT = 503
ALPHAS = (-0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
SEVERITIES = ("light", "middle", "heavy")
TAU_RANGE_MINIMUM = 0.05
Q_TOLERANCE = 1e-5
Q_DENOMINATOR_MSE_MINIMUM = 1e-5
NEGATIVE_CONTROL_MARGIN = 0.005
EXPECTED_COUNTS = {
    "train_clear": 10000,
    "train_variants": 100000,
    "validation_clear": 1000,
    "validation_variants": 10000,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_lines(lines: Iterable[str]) -> str:
    payload = "\n".join(sorted(lines)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"missing ITS directory: {directory}")
    return sorted(
        item for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def scene_id(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def grouped_by_scene(paths: Iterable[Path]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        result[scene_id(path)].append(path)
    return {key: sorted(value) for key, value in sorted(result.items())}


def hashed_scene_order(scene_ids: Iterable[str], split: str) -> list[str]:
    return sorted(
        scene_ids,
        key=lambda value: (
            hashlib.sha256(f"{SAMPLE_SALT}|{split}|{value}".encode()).hexdigest(),
            value,
        ),
    )


def load_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(
            image.convert("RGB").resize((size, size), RESAMPLE), dtype=np.float64,
        )
    return value / 255.0


def load_transmission(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(
            image.convert("L").resize((size, size), RESAMPLE), dtype=np.float64,
        )
    value = value / 255.0
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"invalid transmission field: {path}")
    return np.clip(value, 1.0 / 255.0, 1.0)


def canonical_clear_digest(path: Path) -> str:
    with Image.open(path) as image:
        value = np.asarray(
            image.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    return hashlib.sha256(value.tobytes()).hexdigest()


def mean_tau(path: Path, size: int) -> float:
    return float(np.mean(-np.log(load_transmission(path, size))))


def select_severities(paths: list[Path], size: int) -> dict[str, dict[str, Any]]:
    if len(paths) != 10:
        raise ValueError(f"expected ten ITS variants, observed {len(paths)}")
    ranked = sorted((mean_tau(path, size), path) for path in paths)
    indices = (0, (len(ranked) - 1) // 2, len(ranked) - 1)
    selected = {
        severity: {"tau_mean": ranked[index][0], "transmission_path": ranked[index][1]}
        for severity, index in zip(SEVERITIES, indices)
    }
    if len({value["transmission_path"].stem for value in selected.values()}) != 3:
        raise ValueError("severity selection did not produce three distinct variants")
    return selected


def wilson_interval(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0:
        return {"successes": successes, "total": total, "estimate": 0.0, "lower": 0.0, "upper": 1.0}
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {
        "successes": successes,
        "total": total,
        "estimate": proportion,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
    }


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "q05": None, "median": None, "q95": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def safe_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or x.std() <= 1e-12 or y.std() <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def content_fields(clear: np.ndarray) -> dict[str, np.ndarray | float]:
    luminance = 0.2126 * clear[..., 0] + 0.7152 * clear[..., 1] + 0.0722 * clear[..., 2]
    dx = np.diff(luminance, axis=1, append=luminance[:, -1:])
    dy = np.diff(luminance, axis=0, append=luminance[-1:, :])
    gradient = np.sqrt(dx * dx + dy * dy)
    maximum = np.max(clear, axis=2)
    minimum = np.min(clear, axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 1e-12,
    )
    return {
        "luminance": luminance,
        "gradient": gradient,
        "saturation": saturation,
        "mean_luminance": float(np.mean(luminance)),
        "mean_gradient": float(np.mean(gradient)),
        "mean_saturation": float(np.mean(saturation)),
    }


def local_correlation(tau: np.ndarray, field: np.ndarray) -> float | None:
    return safe_correlation(tau.reshape(-1).tolist(), field.reshape(-1).tolist())


def reconstruction_rmse(haze: np.ndarray, clear: np.ndarray, transmission: np.ndarray) -> float:
    one_minus = 1.0 - transmission[..., None]
    denominator = float(np.sum(one_minus * one_minus))
    if denominator <= 1e-12:
        return float("inf")
    numerator = np.sum((haze - clear * transmission[..., None]) * one_minus, axis=(0, 1))
    airlight = np.clip(numerator / denominator, 0.0, 1.0)
    reconstructed = clear * transmission[..., None] + airlight * one_minus
    return float(np.sqrt(np.mean((haze - reconstructed) ** 2)))


def q_cells(haze: np.ndarray, clear: np.ndarray, tau: np.ndarray) -> dict[str, Any]:
    first, second = np.quantile(tau, (1.0 / 3.0, 2.0 / 3.0))
    masks = {
        "low_tau": tau <= first,
        "middle_tau": (tau > first) & (tau <= second),
        "high_tau": tau > second,
    }
    residual = haze - clear
    identifiable = 0
    passed = True
    maximum_error = 0.0
    low_denominator_cells = 0
    absolute_error_finite = True
    for mask in masks.values():
        pixels = int(np.sum(mask))
        if pixels < 64:
            low_denominator_cells += 1
            continue
        values = residual[mask]
        denominator = float(np.sum(values * values))
        residual_mse = denominator / (pixels * 3.0)
        if residual_mse < Q_DENOMINATOR_MSE_MINIMUM:
            low_denominator_cells += 1
            for alpha in ALPHAS:
                absolute = float(np.mean(np.abs(alpha * values)))
                absolute_error_finite = absolute_error_finite and math.isfinite(absolute)
            continue
        identifiable += 1
        for alpha in ALPHAS:
            controlled = clear[mask] + alpha * values
            q_value = float(np.sum((controlled - clear[mask]) * values) / (denominator + 1e-12))
            error = abs(q_value - alpha)
            maximum_error = max(maximum_error, error)
            passed = passed and math.isfinite(q_value) and error <= Q_TOLERANCE
    return {
        "identifiable_regions": identifiable,
        "low_denominator_regions": low_denominator_cells,
        "q_pass": passed and identifiable > 0,
        "maximum_q_error": maximum_error,
        "absolute_error_finite": absolute_error_finite,
    }


def scan_split(
    *,
    split: str,
    selected_ids: list[str],
    transmissions: dict[str, list[Path]],
    size: int,
    progress_offset: int,
    context: Any,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, str]]]:
    records: dict[str, dict[str, dict[str, Any]]] = {}
    failures: list[dict[str, str]] = []
    for index, identifier in enumerate(selected_ids, start=1):
        try:
            records[identifier] = select_severities(transmissions.get(identifier, []), size)
        except (OSError, ValueError) as exc:
            failures.append({"scene_id": identifier, "reason": str(exc)[:256]})
        completed = progress_offset + index
        if index == 1 or index % 25 == 0 or index == len(selected_ids):
            write_workload_progress(context, completed_units=completed, stage=f"{split}_severity_scan")
    return records, failures


def process_split(
    *,
    split: str,
    selected_ids: list[str],
    severity_records: dict[str, dict[str, dict[str, Any]]],
    clear_by_id: dict[str, Path],
    haze_by_stem: dict[str, Path],
    size: int,
    progress_offset: int,
    context: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    available_ids = [identifier for identifier in selected_ids if identifier in severity_records]
    if available_ids:
        shift = PERMUTATION_SHIFT % len(available_ids)
        if shift == 0:
            shift = 1
        permuted = {
            identifier: available_ids[(index + shift) % len(available_ids)]
            for index, identifier in enumerate(available_ids)
        }
    else:
        permuted = {}
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, identifier in enumerate(selected_ids, start=1):
        try:
            if identifier not in severity_records:
                raise ValueError("severity scan was unavailable")
            permutation_id = permuted[identifier]
            if permutation_id == identifier:
                raise ValueError("negative-control permutation contains a fixed point")
            clear = load_rgb(clear_by_id[identifier], size)
            fields = content_fields(clear)
            severity_results: dict[str, dict[str, Any]] = {}
            q_scene_pass = True
            identifiable_scene = True
            negative_control_wins = 0
            maximum_q_error = 0.0
            local_associations: dict[str, list[float]] = {
                "tau_luminance": [], "tau_gradient": [], "tau_saturation": [],
            }
            for severity in SEVERITIES:
                selected = severity_records[identifier][severity]
                transmission_path = selected["transmission_path"]
                stem = transmission_path.stem
                if stem not in haze_by_stem:
                    raise ValueError(f"missing paired haze image for {stem}")
                haze = load_rgb(haze_by_stem[stem], size)
                transmission = load_transmission(transmission_path, size)
                tau = -np.log(transmission)
                permuted_path = severity_records[permutation_id][severity]["transmission_path"]
                permuted_transmission = load_transmission(permuted_path, size)
                true_rmse = reconstruction_rmse(haze, clear, transmission)
                permuted_rmse = reconstruction_rmse(haze, clear, permuted_transmission)
                improvement = permuted_rmse - true_rmse
                negative_control_wins += int(improvement > NEGATIVE_CONTROL_MARGIN)
                cells = q_cells(haze, clear, tau)
                identifiable = cells["identifiable_regions"] >= 2
                identifiable_scene = identifiable_scene and identifiable
                q_scene_pass = q_scene_pass and identifiable and cells["q_pass"] and cells["absolute_error_finite"]
                maximum_q_error = max(maximum_q_error, cells["maximum_q_error"])
                correlations = {
                    "tau_luminance": local_correlation(tau, fields["luminance"]),
                    "tau_gradient": local_correlation(tau, fields["gradient"]),
                    "tau_saturation": local_correlation(tau, fields["saturation"]),
                }
                for key, value in correlations.items():
                    if value is not None:
                        local_associations[key].append(value)
                severity_results[severity] = {
                    "tau_mean": float(selected["tau_mean"]),
                    "true_reconstruction_rmse": true_rmse,
                    "permuted_reconstruction_rmse": permuted_rmse,
                    "negative_control_improvement": improvement,
                    "negative_control_pass": improvement > NEGATIVE_CONTROL_MARGIN,
                    "identifiable_regions": cells["identifiable_regions"],
                    "low_denominator_regions": cells["low_denominator_regions"],
                    "q_pass": cells["q_pass"],
                    "maximum_q_error": cells["maximum_q_error"],
                }
            tau_range = (
                severity_results["heavy"]["tau_mean"] - severity_results["light"]["tau_mean"]
            )
            negative_control_pass = negative_control_wins >= 2
            results.append({
                "scene_id": identifier,
                "canonical_clear_digest": canonical_clear_digest(clear_by_id[identifier]),
                "severity": severity_results,
                "tau_range": tau_range,
                "severity_coverage": tau_range >= TAU_RANGE_MINIMUM,
                "identifiable_scene": identifiable_scene,
                "q_scene_pass": q_scene_pass,
                "negative_control_pass": negative_control_pass,
                "joint_qualification": identifiable_scene and q_scene_pass and negative_control_pass,
                "maximum_q_error": maximum_q_error,
                "mean_luminance": fields["mean_luminance"],
                "mean_gradient": fields["mean_gradient"],
                "mean_saturation": fields["mean_saturation"],
                "local_associations": {
                    key: (float(np.mean(value)) if value else None)
                    for key, value in local_associations.items()
                },
            })
        except (OSError, ValueError) as exc:
            failures.append({"scene_id": identifier, "reason": str(exc)[:256]})
        completed = progress_offset + index
        if index == 1 or index % 25 == 0 or index == len(selected_ids):
            write_workload_progress(context, completed_units=completed, stage=f"{split}_measurement")
    return results, failures


def aggregate_split(results: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    severity_coverage = sum(bool(item["severity_coverage"]) for item in results)
    identifiable = sum(bool(item["identifiable_scene"]) for item in results)
    q_pass = sum(bool(item["q_scene_pass"]) for item in results)
    negative = sum(bool(item["negative_control_pass"]) for item in results)
    joint = sum(bool(item["joint_qualification"]) for item in results)
    average_tau = [
        float(np.mean([item["severity"][severity]["tau_mean"] for severity in SEVERITIES]))
        for item in results
    ]
    scene_content = {
        "tau_vs_mean_luminance": safe_correlation(average_tau, [item["mean_luminance"] for item in results]),
        "tau_vs_mean_gradient": safe_correlation(average_tau, [item["mean_gradient"] for item in results]),
        "tau_vs_mean_saturation": safe_correlation(average_tau, [item["mean_saturation"] for item in results]),
    }
    local_content: dict[str, dict[str, float | None]] = {}
    for key in ("tau_luminance", "tau_gradient", "tau_saturation"):
        values = [
            item["local_associations"][key]
            for item in results if item["local_associations"][key] is not None
        ]
        local_content[key] = quantiles(values)
    return {
        "expected_scene_count": expected,
        "completed_scene_count": len(results),
        "severity_coverage": wilson_interval(severity_coverage, len(results)),
        "identifiable_scene_coverage": wilson_interval(identifiable, len(results)),
        "controlled_q_recovery": wilson_interval(q_pass, len(results)),
        "negative_control_separation": wilson_interval(negative, len(results)),
        "joint_qualification": wilson_interval(joint, len(results)),
        "tau_range": quantiles([item["tau_range"] for item in results]),
        "maximum_q_error": quantiles([item["maximum_q_error"] for item in results]),
        "scene_level_content_associations": scene_content,
        "local_content_associations_descriptive": local_content,
    }


def write_strata_csv(path: Path, split_results: dict[str, list[dict[str, Any]]]) -> None:
    rows: list[dict[str, Any]] = []
    for split, results in split_results.items():
        for severity in SEVERITIES:
            selected = [item["severity"][severity] for item in results]
            rows.append({
                "split": split,
                "severity": severity,
                "independent_scenes": len(results),
                "tau_mean": float(np.mean([item["tau_mean"] for item in selected])) if selected else "",
                "tau_median": float(np.median([item["tau_mean"] for item in selected])) if selected else "",
                "true_rmse_mean": float(np.mean([item["true_reconstruction_rmse"] for item in selected])) if selected else "",
                "permuted_rmse_mean": float(np.mean([item["permuted_reconstruction_rmse"] for item in selected])) if selected else "",
                "negative_control_improvement_mean": float(np.mean([item["negative_control_improvement"] for item in selected])) if selected else "",
                "negative_control_pass_rate": float(np.mean([item["negative_control_pass"] for item in selected])) if selected else "",
                "q_pass_rate": float(np.mean([item["q_pass"] for item in selected])) if selected else "",
            })
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def terminal_from_gates(validation: dict[str, Any], overlap_count: int) -> tuple[str, str, str, list[str]]:
    reasons: list[str] = []
    if overlap_count != 0:
        reasons.append("sampled train and validation clear-scene canonical digests overlap")
        return "COMPLETED_GATE_FAIL", "ITS_LOCAL_MEASUREMENT_FAIL", "NONE", reasons
    if validation["completed_scene_count"] != 1000:
        reasons.append("fewer than 1,000 independent validation scenes completed")
    if validation["severity_coverage"]["lower"] < 0.9:
        reasons.append("severity coverage Wilson lower bound is below 0.90")
    if validation["identifiable_scene_coverage"]["lower"] < 0.9:
        reasons.append("identifiable-region coverage Wilson lower bound is below 0.90")
    if reasons:
        return (
            "COMPLETED_INCONCLUSIVE",
            "ITS_LOCAL_MEASUREMENT_INCONCLUSIVE",
            "MEASUREMENT_SUPPLEMENT_ONLY",
            reasons,
        )
    decisive = []
    if validation["controlled_q_recovery"]["lower"] < 0.99:
        decisive.append("controlled q-recovery Wilson lower bound is below 0.99")
    if validation["negative_control_separation"]["lower"] < 0.85:
        decisive.append("paired-transmission separation Wilson lower bound is below 0.85")
    if validation["joint_qualification"]["lower"] < 0.85:
        decisive.append("joint scene-qualification Wilson lower bound is below 0.85")
    if decisive:
        return "COMPLETED_GATE_FAIL", "ITS_LOCAL_MEASUREMENT_FAIL", "NONE", decisive
    return (
        "COMPLETED_GATE_PASS",
        "ITS_LOCAL_MEASUREMENT_PASS",
        "CONVIRB_INDOOR_BASELINE_MEASUREMENT",
        ["all frozen integrity, coverage, precision, q-recovery, and negative-control gates passed"],
    )


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    qualification = context.assets.get("reside_qualification")
    checks = {
        "metadata_only_mode": context.engineering_contract["mode"] == "metadata_only",
        "cpu_contract": context.device == "cpu",
        "dataset_hidden_from_contract": "reside_root" not in context.assets,
        "qualification_identity_bound": qualification is not None and qualification.contract_access is True,
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
        "no_model_or_training_path": True,
        "output_and_finalizer_contract": True,
    }
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "metadata_only",
            "device": "cpu",
            "fixture": None,
            "production_path_exercised": False,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    size = int(os.environ.get("CONVIR_ROUTE_ITS_IMAGE_SIZE", "96"))
    if size != 96:
        raise ValueError("ITS image size differs from the frozen 96-pixel contract")
    reside = asset_path(context, "reside_root", kind="directory")
    qualification_path = asset_path(context, "reside_qualification", kind="file")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if qualification.get("marker") != "RESIDE_MINIMAL_MEASUREMENT_QUALIFICATION_RECORDED":
        raise ValueError("RESIDE qualification marker is missing")
    if qualification.get("decision", {}).get("its_training_role") != "ELIGIBLE_FOR_SCENE_LEVEL_SUBSAMPLING":
        raise ValueError("RESIDE qualification does not authorize ITS scene sampling")
    dataset_identity = qualification["dataset_identity"]
    identity_files = {
        "archive_manifest_sha256": reside / "ARCHIVE_SHA256SUMS.txt",
        "pairing_report_sha256": reside / "PAIRING_VALIDATION.txt",
        "layout_record_sha256": reside / "DATASET_LAYOUT.txt",
    }
    observed_identity = {key: sha256_file(path) for key, path in identity_files.items()}
    if any(observed_identity[key] != dataset_identity[key] for key in observed_identity):
        raise ValueError("RESIDE dataset identity differs from the qualified snapshot")

    directories = {
        "train_clear": reside / "official/ITS/train/ITS_clear",
        "train_haze": reside / "official/ITS/train/ITS_haze",
        "train_transmission": reside / "official/ITS/train/ITS_trans",
        "validation_clear": reside / "official/ITS/val/clear",
        "validation_haze": reside / "official/ITS/val/haze",
        "validation_transmission": reside / "official/ITS/val/trans",
    }
    files = {key: image_files(path) for key, path in directories.items()}
    observed_counts = {
        "train_clear": len(files["train_clear"]),
        "train_variants": len(files["train_haze"]),
        "train_transmissions": len(files["train_transmission"]),
        "validation_clear": len(files["validation_clear"]),
        "validation_variants": len(files["validation_haze"]),
        "validation_transmissions": len(files["validation_transmission"]),
    }
    count_integrity = (
        observed_counts["train_clear"] == EXPECTED_COUNTS["train_clear"]
        and observed_counts["train_variants"] == EXPECTED_COUNTS["train_variants"]
        and observed_counts["train_transmissions"] == EXPECTED_COUNTS["train_variants"]
        and observed_counts["validation_clear"] == EXPECTED_COUNTS["validation_clear"]
        and observed_counts["validation_variants"] == EXPECTED_COUNTS["validation_variants"]
        and observed_counts["validation_transmissions"] == EXPECTED_COUNTS["validation_variants"]
    )

    clear = {
        "definition": {path.stem: path for path in files["train_clear"]},
        "validation": {path.stem: path for path in files["validation_clear"]},
    }
    haze_paths = {
        "definition": files["train_haze"],
        "validation": files["validation_haze"],
    }
    transmission_paths = {
        "definition": files["train_transmission"],
        "validation": files["validation_transmission"],
    }
    haze_by_stem = {
        split: {path.stem: path for path in paths} for split, paths in haze_paths.items()
    }
    transmissions = {
        split: grouped_by_scene(paths) for split, paths in transmission_paths.items()
    }
    mapping_integrity = all(
        set(haze_by_stem[split]) == {path.stem for path in transmission_paths[split]}
        and set(transmissions[split]) == set(clear[split])
        for split in ("definition", "validation")
    )

    definition_order = hashed_scene_order(clear["definition"], "definition")[:1000]
    validation_order = hashed_scene_order(clear["validation"], "validation")
    selected = {"definition": definition_order, "validation": validation_order}
    scans: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    scan_failures: dict[str, list[dict[str, str]]] = {}
    scans["definition"], scan_failures["definition"] = scan_split(
        split="definition", selected_ids=definition_order,
        transmissions=transmissions["definition"], size=size,
        progress_offset=0, context=context,
    )
    scans["validation"], scan_failures["validation"] = scan_split(
        split="validation", selected_ids=validation_order,
        transmissions=transmissions["validation"], size=size,
        progress_offset=1000, context=context,
    )

    split_results: dict[str, list[dict[str, Any]]] = {}
    process_failures: dict[str, list[dict[str, str]]] = {}
    split_results["definition"], process_failures["definition"] = process_split(
        split="definition", selected_ids=definition_order,
        severity_records=scans["definition"], clear_by_id=clear["definition"],
        haze_by_stem=haze_by_stem["definition"], size=size,
        progress_offset=2000, context=context,
    )
    split_results["validation"], process_failures["validation"] = process_split(
        split="validation", selected_ids=validation_order,
        severity_records=scans["validation"], clear_by_id=clear["validation"],
        haze_by_stem=haze_by_stem["validation"], size=size,
        progress_offset=3000, context=context,
    )

    aggregates = {
        "definition": aggregate_split(split_results["definition"], 1000),
        "validation": aggregate_split(split_results["validation"], 1000),
    }
    definition_digests = {item["canonical_clear_digest"] for item in split_results["definition"]}
    validation_digests = {item["canonical_clear_digest"] for item in split_results["validation"]}
    overlap = definition_digests & validation_digests
    state, decision, authorizes, gate_reasons = terminal_from_gates(
        aggregates["validation"], len(overlap),
    )
    if not count_integrity or not mapping_integrity:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "ITS_LOCAL_MEASUREMENT_INCONCLUSIVE"
        authorizes = "MEASUREMENT_SUPPLEMENT_ONLY"
        gate_reasons = ["qualified ITS count or haze/transmission mapping integrity changed"]

    failures = {
        split: (scan_failures[split] + process_failures[split])[:20]
        for split in ("definition", "validation")
    }
    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "measurement_scope": "synthetic indoor ITS only",
        "dataset_identity": {
            "root": str(reside),
            "observed_identity": observed_identity,
            "qualified_identity_match": True,
            "observed_counts": observed_counts,
            "count_integrity": count_integrity,
            "mapping_integrity": mapping_integrity,
        },
        "sampling": {
            "definition_available_scenes": len(clear["definition"]),
            "definition_selected_scenes": len(definition_order),
            "validation_available_scenes": len(clear["validation"]),
            "validation_selected_scenes": len(validation_order),
            "variants_per_scene": 3,
            "severity_selection": "rank ten paired transmission fields by mean tau=-log(clip(t,1/255,1)); choose first, index four, and last",
            "definition_scene_digest": digest_lines(definition_order),
            "validation_scene_digest": digest_lines(validation_order),
            "selection_salt_sha256": hashlib.sha256(SAMPLE_SALT.encode()).hexdigest(),
            "permutation_shift": PERMUTATION_SHIFT,
            "effective_independent_unit": "clear_scene",
            "effective_independent_scene_count": len(split_results["definition"]) + len(split_results["validation"]),
            "nested_pair_count": 3 * (len(split_results["definition"]) + len(split_results["validation"])),
        },
        "measurement": {
            "tau": "-log(clip(t,1/255,1)); synthetic optical thickness only",
            "q": "<Y-J,I-J>/(||I-J||^2+1e-12)",
            "controlled_alpha_grid": list(ALPHAS),
            "q_tolerance": Q_TOLERANCE,
            "q_denominator_mse_minimum": Q_DENOMINATOR_MSE_MINIMUM,
            "negative_control_margin": NEGATIVE_CONTROL_MARGIN,
            "near_clear_rule": "q is not used for low-denominator cells; finite absolute error is recorded instead",
        },
        "aggregates": aggregates,
        "split_overlap": {
            "canonical_overlap_count": len(overlap),
            "overlap_digest": digest_lines(overlap),
        },
        "failures": failures,
        "failure_counts": {
            split: len(scan_failures[split]) + len(process_failures[split])
            for split in ("definition", "validation")
        },
        "gates": {
            "validation_scene_count": aggregates["validation"]["completed_scene_count"] == 1000,
            "split_overlap_zero": len(overlap) == 0,
            "severity_coverage": aggregates["validation"]["severity_coverage"]["lower"] >= 0.9,
            "identifiable_region_coverage": aggregates["validation"]["identifiable_scene_coverage"]["lower"] >= 0.9,
            "controlled_q_recovery": aggregates["validation"]["controlled_q_recovery"]["lower"] >= 0.99,
            "paired_transmission_negative_control": aggregates["validation"]["negative_control_separation"]["lower"] >= 0.85,
            "primary_scene_qualification": aggregates["validation"]["joint_qualification"]["lower"] >= 0.85,
        },
        "terminal": {
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "limitations": [
            "Passing supports an L1 synthetic-indoor measurement/mechanism-feasibility claim only.",
            "Controlled q recovery validates direction and implementation but is algebraic, not causal evidence.",
            "Transmission is synthetic optical thickness and is not a validated real-haze restoration-demand label.",
            "Brightness, gradient, and saturation associations are descriptive and may indicate construct confounding.",
            "Outdoor and real-haze calibration remain mandatory independent stages; no ITS result is transported to them.",
            "No model, insertion site, loss, hyperparameter, training run, inference, SOTS, OTS, Haze4K test, or NH-HAZE access occurred.",
        ],
        "marker": "RESIDE_ITS_LOCAL_MEASUREMENT_V1_COMPLETE",
    }
    summary_path = output_file(context, "measurement_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_strata_csv(output_file(context, "aggregate_strata.csv"), split_results)
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "measurement_summary.json",
            "validation_scenes": len(split_results["validation"]),
            "joint_qualification_lower_95": aggregates["validation"]["joint_qualification"]["lower"],
            "gate_reasons": gate_reasons,
            "outdoor_calibration_required": True,
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
