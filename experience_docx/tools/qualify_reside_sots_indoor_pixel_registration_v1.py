#!/usr/bin/env python3
"""Model-free pixel-registration qualification for RESIDE SOTS Indoor."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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
    write_scientific_review_facts,
    write_workload_progress,
)


EXPECTED_INDOOR_ROOT = Path(
    "/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/SOTS/indoor"
)
EXPECTED_CLEAR = 50
EXPECTED_HAZY = 500
EXPECTED_VARIANTS = 10
EXPECTED_CLEAR_SIZE = (640, 480)
EXPECTED_HAZY_SIZE = (620, 460)
CROP_PER_SIDE = 10
EXPECTED_GT_SHA256 = (
    "f87015ecff62c83a7286017fb1641589df6445556e35737196d81e4fb7dda950"
)
EXPECTED_HAZY_SHA256 = (
    "18543d375ac814208b71844748e85ca8685317da4a52403dfb172ecfeee6f3ee"
)
EXPECTED_COMBINED_SHA256 = (
    "31efa342c4224585772ff15eaad9285e1d4134f9dd3f355eb70e689e19fdbfb4"
)
GEOMETRY_ROUTE_ID = "reside-sots-indoor-geometry-qualification-v2"
GEOMETRY_TERMINAL_RECORD_SHA256 = (
    "a457021296d6487af55e58e2b3c07dcc38fe454f5785db04567a18eff4d298da"
)
GEOMETRY_SUMMARY_SHA256 = (
    "194d8cd661eed78c16dd7cb353f5dc45573cac774c1d74f951826f4dd6f066a3"
)
GEOMETRY_CLOSEOUT_SHA256 = (
    "92fd1fda2ae9e66f6dd3fa8ae7f93e6cab83bc91605bd185495ec94b066f3e80"
)
GEOMETRY_CONCLUSION_SHA256 = (
    "ef50912db082fdf10ec89838642f3b9939ddaa508b41b669be1cbfebbf8ec85d"
)

SEARCH_RADIUS = 4
FAR_CONTROL_SHIFT = 8
SCORE_MARGIN = SEARCH_RADIUS + FAR_CONTROL_SHIFT + 2
EDGE_QUANTILE = 0.75
MIN_EFFECTIVE_PIXELS = 4096
MIN_STRUCTURE_SCORE = 0.45
MIN_ZERO_PEAK_MARGIN = 0.015
MIN_NONZERO_PEAK_MARGIN = 0.04
MIN_WRONG_CLEAR_MARGIN = 0.10
MIN_FAR_SHIFT_MARGIN = 0.05
MIN_CONTROL_VALID_PAIRS = 475
MIN_CONTROL_VALID_SCENES = 48
MIN_SUPPORTIVE_PAIRS = 475
MIN_SUPPORTIVE_SCENES = 48
MIN_SUPPORTIVE_VARIANTS_PER_SCENE = 9
FAIL_MIN_CONTRADICTORY_PAIRS = 50
FAIL_MIN_CONTRADICTORY_SCENES = 10
FAIL_MIN_MODAL_SHIFT_FRACTION = 0.80

SUMMARY_FILENAME = "reside_sots_indoor_pixel_registration_qualification_v1_summary.json"
REVIEW_FACTS_FILENAME = (
    "reside_sots_indoor_pixel_registration_qualification_v1_review_facts.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def direct_png_files(directory: Path) -> tuple[list[Path], int]:
    if not directory.is_dir() or directory.is_symlink():
        return [], 1
    files: list[Path] = []
    unexpected_entries = 0
    for item in directory.iterdir():
        if item.is_symlink() or item.is_dir():
            unexpected_entries += 1
        elif item.is_file() and item.suffix.lower() == ".png":
            files.append(item)
        elif item.is_file() and item.suffix.lower() in {
            ".bmp",
            ".jpeg",
            ".jpg",
            ".tif",
            ".tiff",
        }:
            unexpected_entries += 1
    return sorted(files), unexpected_entries


def aggregate_digest(
    root: Path, files: list[Path], identities: dict[Path, str]
) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(
            f"{relative}\t{path.stat().st_size}\t{identities[path]}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def load_rgb(path: Path, expected_size: tuple[int, int], crop: int) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError(f"unexpected image encoding: {path.name}")
        if tuple(image.size) != expected_size:
            raise ValueError(f"unexpected image size: {path.name}")
        array = np.asarray(image, dtype=np.float64) / 255.0
    if crop:
        array = array[crop:-crop, crop:-crop, :]
    if array.shape != (EXPECTED_HAZY_SIZE[1], EXPECTED_HAZY_SIZE[0], 3):
        raise ValueError(f"unexpected scoring geometry: {path.name}")
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite decoded pixels: {path.name}")
    return array


def structure_map(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    gx = np.zeros_like(gray, dtype=np.float64)
    gy = np.zeros_like(gray, dtype=np.float64)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    return gx, gy


def score_shift(
    clear: tuple[np.ndarray, np.ndarray],
    hazy: tuple[np.ndarray, np.ndarray],
    dx: int,
    dy: int,
) -> tuple[float | None, int]:
    clear_gx, clear_gy = clear
    hazy_gx, hazy_gy = hazy
    height, width = clear_gx.shape
    y0, y1 = SCORE_MARGIN, height - SCORE_MARGIN
    x0, x1 = SCORE_MARGIN, width - SCORE_MARGIN
    cgx = clear_gx[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    cgy = clear_gy[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    hgx = hazy_gx[y0:y1, x0:x1]
    hgy = hazy_gy[y0:y1, x0:x1]
    magnitude = np.hypot(cgx, cgy)
    edge_threshold = float(np.quantile(magnitude, EDGE_QUANTILE))
    mask = magnitude >= edge_threshold
    effective_pixels = int(np.count_nonzero(mask))
    if effective_pixels < MIN_EFFECTIVE_PIXELS or edge_threshold <= 1e-8:
        return None, effective_pixels
    clear_vector = np.concatenate((cgx[mask], cgy[mask]))
    hazy_vector = np.concatenate((hgx[mask], hgy[mask]))
    clear_vector = clear_vector - float(np.mean(clear_vector))
    hazy_vector = hazy_vector - float(np.mean(hazy_vector))
    denominator = float(
        np.linalg.norm(clear_vector) * np.linalg.norm(hazy_vector)
    )
    if denominator <= 1e-12:
        return None, effective_pixels
    score = float(np.dot(clear_vector, hazy_vector) / denominator)
    if not np.isfinite(score):
        return None, effective_pixels
    return max(-1.0, min(1.0, score)), effective_pixels


def score_grid(
    clear: tuple[np.ndarray, np.ndarray],
    hazy: tuple[np.ndarray, np.ndarray],
) -> tuple[dict[tuple[int, int], float], int]:
    scores: dict[tuple[int, int], float] = {}
    minimum_effective = 1 << 60
    for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
        for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            score, effective = score_shift(clear, hazy, dx, dy)
            minimum_effective = min(minimum_effective, effective)
            if score is not None:
                scores[(dx, dy)] = score
    return scores, int(minimum_effective)


def analyze_pair(
    clear: tuple[np.ndarray, np.ndarray],
    hazy: tuple[np.ndarray, np.ndarray],
    wrong_clear: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    true_scores, effective_pixels = score_grid(clear, hazy)
    wrong_scores, wrong_effective_pixels = score_grid(wrong_clear, hazy)
    if len(true_scores) != (2 * SEARCH_RADIUS + 1) ** 2 or not wrong_scores:
        return {
            "classification": "ambiguous",
            "control_valid": False,
            "best_shift": None,
            "effective_pixels": min(effective_pixels, wrong_effective_pixels),
        }
    ordered = sorted(true_scores.items(), key=lambda item: (-item[1], item[0]))
    best_shift, best_score = ordered[0]
    second_score = ordered[1][1]
    zero_score = true_scores[(0, 0)]
    best_nonzero = max(
        score for shift, score in true_scores.items() if shift != (0, 0)
    )
    wrong_best = max(wrong_scores.values())
    far_scores: list[float] = []
    for dx, dy in (
        (-FAR_CONTROL_SHIFT, 0),
        (FAR_CONTROL_SHIFT, 0),
        (0, -FAR_CONTROL_SHIFT),
        (0, FAR_CONTROL_SHIFT),
    ):
        score, _ = score_shift(clear, hazy, dx, dy)
        if score is not None:
            far_scores.append(score)
    if len(far_scores) != 4:
        return {
            "classification": "ambiguous",
            "control_valid": False,
            "best_shift": best_shift,
            "effective_pixels": effective_pixels,
        }
    far_best = max(far_scores)
    wrong_margin = best_score - wrong_best
    far_margin = best_score - far_best
    control_valid = (
        wrong_margin >= MIN_WRONG_CLEAR_MARGIN
        and far_margin >= MIN_FAR_SHIFT_MARGIN
    )
    supportive = (
        control_valid
        and best_shift == (0, 0)
        and zero_score >= MIN_STRUCTURE_SCORE
        and zero_score - best_nonzero >= MIN_ZERO_PEAK_MARGIN
    )
    contradictory = (
        control_valid
        and best_shift != (0, 0)
        and best_score >= MIN_STRUCTURE_SCORE
        and best_score - zero_score >= MIN_NONZERO_PEAK_MARGIN
    )
    classification = (
        "supportive" if supportive else "contradictory" if contradictory else "ambiguous"
    )
    return {
        "classification": classification,
        "control_valid": control_valid,
        "best_shift": best_shift,
        "best_score": best_score,
        "zero_score": zero_score,
        "peak_margin": best_score - second_score,
        "zero_nonzero_margin": zero_score - best_nonzero,
        "wrong_margin": wrong_margin,
        "far_margin": far_margin,
        "effective_pixels": effective_pixels,
    }


def synthetic_fixture() -> dict[str, Any]:
    height = width = 128
    yy, xx = np.mgrid[0:height, 0:width]
    base = (
        0.35
        + 0.18 * np.sin(xx * 0.19)
        + 0.16 * np.cos(yy * 0.13)
        + 0.12 * np.sin((xx + 2 * yy) * 0.07)
    )
    base[18:48, 22:67] += 0.22
    base[72:105, 54:112] -= 0.20
    base = np.clip(base, 0.0, 1.0)
    clear_rgb = np.repeat(base[..., None], 3, axis=2)
    wrong_rgb = np.repeat(np.flipud(np.rot90(base))[..., None], 3, axis=2)
    clear_structure = structure_map(clear_rgb)
    wrong_structure = structure_map(wrong_rgb)
    expected_shifts = [(0, 0), (2, -1), (-3, 2)]
    recovered: list[tuple[int, int] | None] = []
    valid_controls = 0
    minimum_best_score = 1.0
    minimum_peak_margin = 1.0
    for dx, dy in expected_shifts:
        shifted = np.roll(base, shift=(-dy, -dx), axis=(0, 1))
        attenuation = 0.52 + 0.08 * np.sin((xx - yy) * 0.03)
        hazy = np.clip(0.31 + attenuation * shifted, 0.0, 1.0)
        hazy_rgb = np.repeat(hazy[..., None], 3, axis=2)
        result = analyze_pair(clear_structure, structure_map(hazy_rgb), wrong_structure)
        recovered.append(result.get("best_shift"))
        valid_controls += int(bool(result.get("control_valid")))
        minimum_best_score = min(minimum_best_score, float(result.get("best_score", -1.0)))
        minimum_peak_margin = min(
            minimum_peak_margin, float(result.get("peak_margin", -1.0))
        )
    passed = (
        recovered == expected_shifts
        and valid_controls == len(expected_shifts)
        and minimum_best_score >= 0.80
        and minimum_peak_margin >= 0.01
    )
    return {
        "passed": passed,
        "case_count": len(expected_shifts),
        "exact_shift_recovery_count": sum(
            observed == expected
            for observed, expected in zip(recovered, expected_shifts)
        ),
        "valid_control_count": valid_controls,
        "minimum_best_score": minimum_best_score,
        "minimum_peak_margin": minimum_peak_margin,
    }


def geometry_evidence_matches(
    summary: dict[str, Any], closeout: dict[str, Any], conclusion: dict[str, Any]
) -> bool:
    geometry = summary.get("geometry", {})
    identity = summary.get("aggregate_identity", {})
    return (
        summary.get("route_id") == GEOMETRY_ROUTE_ID
        and summary.get("marker")
        == "RESIDE_SOTS_INDOOR_GEOMETRY_QUALIFICATION_V2_COMPLETE"
        and summary.get("gate_outcomes")
        == {
            "current_asset_identity": "pass",
            "geometry_compatibility": "favorable",
            "isolation": "safe",
            "v1_evidence_identity": "pass",
        }
        and identity.get("gt_sha256") == EXPECTED_GT_SHA256
        and identity.get("hazy_sha256") == EXPECTED_HAZY_SHA256
        and identity.get("combined_sha256") == EXPECTED_COMBINED_SHA256
        and identity.get("file_count") == EXPECTED_CLEAR + EXPECTED_HAZY
        and geometry.get("geometry_qualified_clear_scene_count") == EXPECTED_CLEAR
        and geometry.get("geometry_qualified_pair_count") == EXPECTED_HAZY
        and geometry.get("crop_delta_histogram") == {"20x20": EXPECTED_HAZY}
        and geometry.get("per_side_crop_histogram") == {"10x10": EXPECTED_HAZY}
        and closeout.get("route_id") == GEOMETRY_ROUTE_ID
        and closeout.get("state") == "COMPLETED_GATE_PASS"
        and closeout.get("decision")
        == "RESIDE_SOTS_INDOOR_GEOMETRY_QUALIFICATION_PASS"
        and closeout.get("authorizes") == "DATASET_ASSET_REGISTRY_ENTRY_ONLY"
        and closeout.get("evidence_sha256", {}).get(
            "reside_sots_indoor_geometry_qualification_v2_summary.json"
        )
        == GEOMETRY_SUMMARY_SHA256
        and conclusion.get("route_id") == GEOMETRY_ROUTE_ID
        and conclusion.get("state") == "COMPLETED_GATE_PASS"
    )


def score_band(value: float | None) -> str:
    if value is None:
        return "invalid"
    for lower, upper, label in (
        (-2.0, 0.0, "lt_0"),
        (0.0, 0.25, "0_to_lt_0.25"),
        (0.25, 0.45, "0.25_to_lt_0.45"),
        (0.45, 0.60, "0.45_to_lt_0.60"),
        (0.60, 0.80, "0.60_to_lt_0.80"),
        (0.80, 2.0, "ge_0.80"),
    ):
        if lower <= value < upper:
            return label
    return "invalid"


def margin_band(value: float | None) -> str:
    if value is None:
        return "invalid"
    for lower, upper, label in (
        (-2.0, 0.0, "lt_0"),
        (0.0, 0.015, "0_to_lt_0.015"),
        (0.015, 0.04, "0.015_to_lt_0.04"),
        (0.04, 0.10, "0.04_to_lt_0.10"),
        (0.10, 2.0, "ge_0.10"),
    ):
        if lower <= value < upper:
            return label
    return "invalid"


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    geometry_summary = load_json(asset_path(context, "geometry_summary", kind="file"))
    geometry_closeout = load_json(asset_path(context, "geometry_closeout", kind="file"))
    geometry_conclusion = load_json(
        asset_path(context, "geometry_conclusion", kind="file")
    )
    fixture_started = time.perf_counter()
    fixture = synthetic_fixture()
    fixture_wall_seconds = time.perf_counter() - fixture_started
    peak_memory_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    write_contract_progress(
        context,
        completed_iterations=fixture["case_count"],
        total_iterations=fixture["case_count"],
        stage="synthetic_known_shift_contract",
    )
    write_contract_result(
        context,
        checks={
            "cpu_contract": context.device == "cpu",
            "cpu_exact_engineering_contract": (
                context.engineering_contract.get("mode") == "cpu_exact"
            ),
            "protected_permissions_disabled": not any(
                context.protected_data_permissions.values()
            ),
            "dataset_hidden_from_contract_phase": "sots_root" not in context.assets,
            "geometry_v2_evidence_available": geometry_evidence_matches(
                geometry_summary, geometry_closeout, geometry_conclusion
            ),
            "synthetic_known_shift_fixture_passed": fixture["passed"],
        },
        engineering={
            "mode": "cpu_exact",
            "device": "cpu",
            "fixture": {
                "batch": 1,
                "channels": 3,
                "height": 128,
                "width": 128,
            },
            "cost": {
                "observed_iterations": fixture["case_count"],
                "observed_wall_seconds": fixture_wall_seconds,
                "observed_peak_memory_mib": peak_memory_mib,
            },
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    if context.total_units != 1 or context.device != "cpu":
        raise RuntimeError("SOTS Indoor registration runtime contract mismatch")
    if context.engineering_contract.get("mode") != "cpu_exact":
        raise RuntimeError("SOTS Indoor registration requires cpu_exact")
    if any(context.protected_data_permissions.values()):
        raise RuntimeError("SOTS Indoor registration forbids protected data")
    if load_completed_unit_ledger(context):
        raise RuntimeError("fresh registration qualification has completed units")

    geometry_summary_path = asset_path(context, "geometry_summary", kind="file")
    geometry_closeout_path = asset_path(context, "geometry_closeout", kind="file")
    geometry_conclusion_path = asset_path(context, "geometry_conclusion", kind="file")
    geometry_summary = load_json(geometry_summary_path)
    geometry_closeout = load_json(geometry_closeout_path)
    geometry_conclusion = load_json(geometry_conclusion_path)
    geometry_evidence_ok = (
        sha256_file(geometry_summary_path) == GEOMETRY_SUMMARY_SHA256
        and sha256_file(geometry_closeout_path) == GEOMETRY_CLOSEOUT_SHA256
        and sha256_file(geometry_conclusion_path) == GEOMETRY_CONCLUSION_SHA256
        and geometry_evidence_matches(
            geometry_summary, geometry_closeout, geometry_conclusion
        )
    )
    fixture = synthetic_fixture()

    sots_root = asset_path(context, "sots_root", kind="directory")
    indoor_root = sots_root / "indoor"
    clear_root = indoor_root / "gt"
    hazy_root = indoor_root / "hazy"
    clear_files, unexpected_clear = direct_png_files(clear_root)
    hazy_files, unexpected_hazy = direct_png_files(hazy_root)
    clear_by_stem = {path.stem: path for path in clear_files}
    duplicate_clear_stems = len(clear_by_stem) != len(clear_files)
    hazy_by_source: dict[str, list[Path]] = defaultdict(list)
    unmapped_hazy: list[str] = []
    for path in hazy_files:
        source = path.stem.split("_", 1)[0]
        if source not in clear_by_stem:
            unmapped_hazy.append(path.name)
        else:
            hazy_by_source[source].append(path)
    unused_clear = sorted(set(clear_by_stem) - set(hazy_by_source))
    variant_histogram = Counter(len(paths) for paths in hazy_by_source.values())

    all_files = [*clear_files, *hazy_files]
    identities: dict[Path, str] = {}
    for index, path in enumerate(all_files, start=1):
        identities[path] = sha256_file(path)
        if index % 50 == 0:
            write_workload_progress(
                context,
                completed_units=0,
                stage=f"asset_identity_checked_{index}_of_{len(all_files)}",
            )
    gt_digest = aggregate_digest(indoor_root, clear_files, identities)
    hazy_digest = aggregate_digest(indoor_root, hazy_files, identities)
    combined_digest = aggregate_digest(indoor_root, all_files, identities)
    pairing_ok = (
        indoor_root == EXPECTED_INDOOR_ROOT
        and indoor_root.is_dir()
        and not indoor_root.is_symlink()
        and len(clear_files) == EXPECTED_CLEAR
        and len(hazy_files) == EXPECTED_HAZY
        and not duplicate_clear_stems
        and not unmapped_hazy
        and not unused_clear
        and len(hazy_by_source) == EXPECTED_CLEAR
        and variant_histogram == Counter({EXPECTED_VARIANTS: EXPECTED_CLEAR})
        and unexpected_clear == 0
        and unexpected_hazy == 0
    )
    current_asset_ok = (
        pairing_ok
        and gt_digest == EXPECTED_GT_SHA256
        and hazy_digest == EXPECTED_HAZY_SHA256
        and combined_digest == EXPECTED_COMBINED_SHA256
    )

    clear_structures: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    decode_failures = 0
    if current_asset_ok:
        for source in sorted(clear_by_stem):
            try:
                clear_structures[source] = structure_map(
                    load_rgb(
                        clear_by_stem[source], EXPECTED_CLEAR_SIZE, CROP_PER_SIDE
                    )
                )
            except Exception:
                decode_failures += 1

    classification_counts: Counter[str] = Counter()
    best_shift_histogram: Counter[str] = Counter()
    zero_score_histogram: Counter[str] = Counter()
    zero_nonzero_margin_histogram: Counter[str] = Counter()
    wrong_margin_histogram: Counter[str] = Counter()
    far_margin_histogram: Counter[str] = Counter()
    scene_support_histogram: Counter[str] = Counter()
    scene_control_histogram: Counter[str] = Counter()
    contradictory_shift_counts: Counter[tuple[int, int]] = Counter()
    supportive_scenes = 0
    control_valid_scenes = 0
    contradictory_scenes = 0
    scored_pairs = 0
    minimum_effective_pixels: int | None = None

    sources = sorted(hazy_by_source)
    if current_asset_ok and decode_failures == 0 and len(sources) == EXPECTED_CLEAR:
        for scene_index, source in enumerate(sources):
            wrong_source = sources[(scene_index + 1) % len(sources)]
            scene_supportive = 0
            scene_controls = 0
            scene_contradictory = 0
            for hazy_path in sorted(hazy_by_source[source]):
                try:
                    hazy_structure = structure_map(
                        load_rgb(hazy_path, EXPECTED_HAZY_SIZE, 0)
                    )
                    result = analyze_pair(
                        clear_structures[source],
                        hazy_structure,
                        clear_structures[wrong_source],
                    )
                except Exception:
                    decode_failures += 1
                    result = {
                        "classification": "ambiguous",
                        "control_valid": False,
                        "best_shift": None,
                        "effective_pixels": 0,
                    }
                classification = str(result["classification"])
                classification_counts[classification] += 1
                scene_supportive += int(classification == "supportive")
                scene_contradictory += int(classification == "contradictory")
                control_valid = bool(result["control_valid"])
                scene_controls += int(control_valid)
                best_shift = result.get("best_shift")
                if best_shift is None:
                    best_shift_histogram["invalid"] += 1
                else:
                    best_shift_histogram[f"dx={best_shift[0]},dy={best_shift[1]}"] += 1
                    if classification == "contradictory":
                        contradictory_shift_counts[best_shift] += 1
                zero_score_histogram[score_band(result.get("zero_score"))] += 1
                zero_nonzero_margin_histogram[
                    margin_band(result.get("zero_nonzero_margin"))
                ] += 1
                wrong_margin_histogram[margin_band(result.get("wrong_margin"))] += 1
                far_margin_histogram[margin_band(result.get("far_margin"))] += 1
                effective = int(result.get("effective_pixels", 0))
                minimum_effective_pixels = (
                    effective
                    if minimum_effective_pixels is None
                    else min(minimum_effective_pixels, effective)
                )
                scored_pairs += 1
            scene_support_histogram[str(scene_supportive)] += 1
            scene_control_histogram[str(scene_controls)] += 1
            supportive_scenes += int(
                scene_supportive >= MIN_SUPPORTIVE_VARIANTS_PER_SCENE
                and scene_contradictory == 0
            )
            control_valid_scenes += int(scene_controls >= MIN_SUPPORTIVE_VARIANTS_PER_SCENE)
            contradictory_scenes += int(scene_contradictory > 0)
            write_workload_progress(
                context,
                completed_units=0,
                stage=f"registration_scored_{scene_index + 1}_of_{len(sources)}_scenes",
            )

    supportive_pairs = classification_counts["supportive"]
    contradictory_pairs = classification_counts["contradictory"]
    ambiguous_pairs = classification_counts["ambiguous"]
    control_valid_pairs = sum(
        int(count) * int(variants)
        for variants, count in scene_control_histogram.items()
    )
    modal_shift: tuple[int, int] | None = None
    modal_shift_count = 0
    if contradictory_shift_counts:
        modal_shift, modal_shift_count = sorted(
            contradictory_shift_counts.items(), key=lambda item: (-item[1], item[0])
        )[0]
    modal_shift_fraction = (
        modal_shift_count / contradictory_pairs if contradictory_pairs else 0.0
    )

    identity_valid = current_asset_ok and decode_failures == 0
    controls_favorable = (
        fixture["passed"]
        and control_valid_pairs >= MIN_CONTROL_VALID_PAIRS
        and control_valid_scenes >= MIN_CONTROL_VALID_SCENES
    )
    registration_pass = (
        controls_favorable
        and scored_pairs == EXPECTED_HAZY
        and supportive_pairs >= MIN_SUPPORTIVE_PAIRS
        and supportive_scenes >= MIN_SUPPORTIVE_SCENES
        and contradictory_pairs == 0
    )
    stable_nonzero_failure = (
        controls_favorable
        and contradictory_pairs >= FAIL_MIN_CONTRADICTORY_PAIRS
        and contradictory_scenes >= FAIL_MIN_CONTRADICTORY_SCENES
        and modal_shift is not None
        and modal_shift != (0, 0)
        and modal_shift_fraction >= FAIL_MIN_MODAL_SHIFT_FRACTION
    )
    correspondence_outcome = (
        "invalid"
        if not identity_valid
        else "favorable"
        if registration_pass
        else "unfavorable"
        if stable_nonzero_failure
        else "indeterminate"
    )
    control_outcome = (
        "invalid"
        if not identity_valid
        else "met"
        if controls_favorable
        else "unmet"
    )
    forbidden_operation_count = 0
    gate_outcomes = {
        "geometry_evidence_identity": "pass" if geometry_evidence_ok else "fail",
        "current_asset_identity": "pass" if identity_valid else "fail",
        "negative_control_validity": control_outcome,
        "pixel_registration": correspondence_outcome,
        "isolation": "safe" if forbidden_operation_count == 0 else "unsafe",
    }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "model-free RESIDE SOTS Indoor center-crop pixel-registration qualification",
        "geometry_v2_binding": {
            "route_id": GEOMETRY_ROUTE_ID,
            "terminal_record_sha256": GEOMETRY_TERMINAL_RECORD_SHA256,
            "summary_sha256": GEOMETRY_SUMMARY_SHA256,
            "closeout_sha256": GEOMETRY_CLOSEOUT_SHA256,
            "conclusion_sha256": GEOMETRY_CONCLUSION_SHA256,
            "evidence_matches": geometry_evidence_ok,
        },
        "aggregate_identity": {
            "gt_sha256": gt_digest,
            "hazy_sha256": hazy_digest,
            "combined_sha256": combined_digest,
            "file_count": len(all_files),
            "matches_geometry_v2": current_asset_ok,
        },
        "design": {
            "independent_unit": "clear_scene",
            "independent_clear_scenes": EXPECTED_CLEAR,
            "nested_variants_per_scene": EXPECTED_VARIANTS,
            "pair_census_size": EXPECTED_HAZY,
            "clear_center_crop_per_side_pixels": CROP_PER_SIDE,
            "search_radius_pixels": SEARCH_RADIUS,
            "far_shift_control_pixels": FAR_CONTROL_SHIFT,
            "wrong_clear_control": "next clear scene in sorted cyclic order",
            "minimum_structure_score": MIN_STRUCTURE_SCORE,
            "minimum_zero_peak_margin": MIN_ZERO_PEAK_MARGIN,
            "minimum_nonzero_peak_margin": MIN_NONZERO_PEAK_MARGIN,
            "minimum_wrong_clear_margin": MIN_WRONG_CLEAR_MARGIN,
            "minimum_far_shift_margin": MIN_FAR_SHIFT_MARGIN,
            "minimum_effective_pixels": MIN_EFFECTIVE_PIXELS,
        },
        "synthetic_fixture": fixture,
        "registration": {
            "scored_pair_count": scored_pairs,
            "supportive_pair_count": supportive_pairs,
            "ambiguous_pair_count": ambiguous_pairs,
            "contradictory_pair_count": contradictory_pairs,
            "control_valid_pair_count": control_valid_pairs,
            "supportive_scene_count": supportive_scenes,
            "control_valid_scene_count": control_valid_scenes,
            "contradictory_scene_count": contradictory_scenes,
            "minimum_effective_pixels_observed": minimum_effective_pixels,
            "modal_contradictory_nonzero_shift": (
                None
                if modal_shift is None
                else {"dx": modal_shift[0], "dy": modal_shift[1]}
            ),
            "modal_contradictory_shift_count": modal_shift_count,
            "modal_contradictory_shift_fraction": modal_shift_fraction,
            "thresholds": {
                "minimum_control_valid_pairs": MIN_CONTROL_VALID_PAIRS,
                "minimum_control_valid_scenes": MIN_CONTROL_VALID_SCENES,
                "minimum_supportive_pairs": MIN_SUPPORTIVE_PAIRS,
                "minimum_supportive_scenes": MIN_SUPPORTIVE_SCENES,
                "minimum_supportive_variants_per_scene": (
                    MIN_SUPPORTIVE_VARIANTS_PER_SCENE
                ),
                "pass_maximum_contradictory_pairs": 0,
                "fail_minimum_contradictory_pairs": FAIL_MIN_CONTRADICTORY_PAIRS,
                "fail_minimum_contradictory_scenes": FAIL_MIN_CONTRADICTORY_SCENES,
                "fail_minimum_modal_shift_fraction": FAIL_MIN_MODAL_SHIFT_FRACTION,
            },
            "classification_histogram": dict(sorted(classification_counts.items())),
            "best_shift_histogram": dict(sorted(best_shift_histogram.items())),
            "scene_support_histogram": dict(sorted(scene_support_histogram.items())),
            "scene_control_histogram": dict(sorted(scene_control_histogram.items())),
            "zero_score_histogram": dict(sorted(zero_score_histogram.items())),
            "zero_nonzero_margin_histogram": dict(
                sorted(zero_nonzero_margin_histogram.items())
            ),
            "wrong_clear_margin_histogram": dict(
                sorted(wrong_margin_histogram.items())
            ),
            "far_shift_margin_histogram": dict(sorted(far_margin_histogram.items())),
        },
        "decode": {
            "decode_failure_count": decode_failures,
            "unexpected_entry_count": unexpected_clear + unexpected_hazy,
        },
        "isolation": {
            "model_loads": 0,
            "checkpoint_reads": 0,
            "training_calls": 0,
            "inference_calls": 0,
            "restoration_metric_calls": 0,
            "image_display_calls": 0,
            "image_archive_calls": 0,
            "protected_data_operations": 0,
            "per_pair_evidence_rows_archived": 0,
            "forbidden_operation_count": forbidden_operation_count,
            "forbidden_operation_limit": 0,
        },
        "gate_outcomes": gate_outcomes,
        "limitations": [
            "This qualification tests spatial correspondence only after the frozen ten-pixel center crop.",
            "Gradient-structure correspondence is not a restoration-quality metric and does not measure model performance.",
            "Ambiguous evidence and insufficient negative-control separation cannot be converted to PASS or FAIL.",
            "No model, checkpoint, training, inference, restoration metric, image display, image archive, or protected-data access occurs.",
        ],
        "marker": "RESIDE_SOTS_INDOOR_PIXEL_REGISTRATION_QUALIFICATION_V1_COMPLETE",
    }
    summary_path = output_file(context, SUMMARY_FILENAME)
    atomic_json(summary_path, summary)
    summary_sha256 = sha256_file(summary_path)

    def fact(
        fact_id: str,
        metric: str,
        unit: str,
        point: int | float,
        point_pointer: str,
        threshold: int | float,
        threshold_operator: str,
        threshold_pointer: str,
        gate_id: str,
    ) -> dict[str, Any]:
        return {
            "fact_id": fact_id,
            "claim_id": "sots_indoor_pixel_registration_qualification",
            "metric": metric,
            "unit": unit,
            "population": "all 50 clear scenes and 500 nested SOTS Indoor hazy variants",
            "grouping": "clear scene with ten nested hazy variants",
            "point": point,
            "ci_lower": None,
            "ci_upper": None,
            "confidence_level": None,
            "threshold": threshold,
            "threshold_operator": threshold_operator,
            "gate_outcome": gate_outcomes[gate_id],
            "source_filename": SUMMARY_FILENAME,
            "source_sha256": summary_sha256,
            "json_pointers": {
                "point": point_pointer,
                "ci_lower": None,
                "ci_upper": None,
                "confidence_level": None,
                "threshold": threshold_pointer,
                "gate_outcome": f"/gate_outcomes/{gate_id}",
            },
        }

    write_scientific_review_facts(
        context,
        relpath=REVIEW_FACTS_FILENAME,
        facts=[
            fact(
                "control_valid_pair_count",
                "pairs with both cyclic wrong-clear and fixed far-shift separation",
                "pairs",
                control_valid_pairs,
                "/registration/control_valid_pair_count",
                MIN_CONTROL_VALID_PAIRS,
                ">=",
                "/registration/thresholds/minimum_control_valid_pairs",
                "negative_control_validity",
            ),
            fact(
                "supportive_pair_count",
                "pairs supporting zero displacement under all score and control gates",
                "pairs",
                supportive_pairs,
                "/registration/supportive_pair_count",
                MIN_SUPPORTIVE_PAIRS,
                ">=",
                "/registration/thresholds/minimum_supportive_pairs",
                "pixel_registration",
            ),
            fact(
                "supportive_scene_count",
                "clear scenes with at least nine supportive variants and no contradictory variant",
                "clear scenes",
                supportive_scenes,
                "/registration/supportive_scene_count",
                MIN_SUPPORTIVE_SCENES,
                ">=",
                "/registration/thresholds/minimum_supportive_scenes",
                "pixel_registration",
            ),
            fact(
                "contradictory_pair_count",
                "control-valid pairs with a decisive nonzero displacement peak",
                "pairs",
                contradictory_pairs,
                "/registration/contradictory_pair_count",
                0,
                "==",
                "/registration/thresholds/pass_maximum_contradictory_pairs",
                "pixel_registration",
            ),
            fact(
                "forbidden_operation_count",
                "model, metric, display, archive, per-pair archive, or protected operation count",
                "operations",
                forbidden_operation_count,
                "/isolation/forbidden_operation_count",
                0,
                "==",
                "/isolation/forbidden_operation_limit",
                "isolation",
            ),
        ],
    )
    record_completed_unit(
        context,
        unit_id="sots_indoor_pixel_registration_complete_census",
        input_sha256=combined_digest,
        output_relpath=SUMMARY_FILENAME,
    )
    write_workload_progress(
        context,
        completed_units=1,
        stage="sots_indoor_pixel_registration_qualified",
    )
    write_gate_result(
        context,
        gate_outcomes=gate_outcomes,
        details={
            "summary_file": SUMMARY_FILENAME,
            "combined_asset_sha256": combined_digest,
            "geometry_terminal_record_sha256": GEOMETRY_TERMINAL_RECORD_SHA256,
            "geometry_summary_sha256": GEOMETRY_SUMMARY_SHA256,
            "geometry_closeout_sha256": GEOMETRY_CLOSEOUT_SHA256,
            "model_or_checkpoint_accessed": False,
            "training_or_inference_occurred": False,
            "restoration_metrics_computed": False,
            "images_displayed_or_archived": False,
            "protected_data_touched": False,
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
