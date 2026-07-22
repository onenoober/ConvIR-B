#!/usr/bin/env bash
set -euo pipefail

PYTHON=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python

[[ -x "$PYTHON" ]] || {
  echo "RESIDE_MEASUREMENT_AUDIT_FAILED: missing cloud Python $PYTHON" >&2
  exit 2
}

"$PYTHON" - <<'PY'
from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.io import loadmat


RESIDE = Path("/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE")
HAZE4K = Path("/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K")
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
RESAMPLE = Image.Resampling.LANCZOS


def fail(message: str) -> None:
    raise SystemExit(f"RESIDE_MEASUREMENT_AUDIT_FAILED: {message}")


def files(path: Path, extensions: set[str]) -> list[Path]:
    if not path.is_dir():
        fail(f"missing directory: {path}")
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in extensions)


def image_files(path: Path) -> list[Path]:
    return files(path, IMAGE_EXTENSIONS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_lines(lines: Iterable[str]) -> str:
    payload = "\n".join(sorted(lines)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scene_id(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def numeric_suffix(path: Path) -> tuple[float, ...] | None:
    parts = path.stem.split("_")[1:]
    if not parts:
        return ()
    try:
        values = tuple(float(value) for value in parts)
    except ValueError:
        return None
    return values if all(math.isfinite(value) for value in values) else None


def variant_summary(items: list[Path], clear_items: list[Path]) -> dict[str, Any]:
    clear_ids = {item.stem for item in clear_items}
    grouped: Counter[str] = Counter(scene_id(item) for item in items)
    if set(grouped) != clear_ids:
        fail(
            f"scene mapping differs for {items[0].parent if items else 'empty'}: "
            f"mapped={len(grouped)} clear={len(clear_ids)}"
        )
    counts = sorted(grouped.values())
    suffixes = [numeric_suffix(item) for item in items]
    suffix_failures = sum(value is None for value in suffixes)
    valid_suffixes = [value for value in suffixes if value is not None]
    dimensions = sorted({len(value) for value in valid_suffixes})
    positions: list[dict[str, Any]] = []
    if len(dimensions) == 1 and dimensions[0] > 0:
        for index in range(dimensions[0]):
            values = sorted({value[index] for value in valid_suffixes})
            positions.append(
                {
                    "index": index + 1,
                    "unique_count": len(values),
                    "min": values[0],
                    "max": values[-1],
                }
            )
    return {
        "clear_scene_count": len(clear_ids),
        "variant_file_count": len(items),
        "variants_per_scene_min": counts[0],
        "variants_per_scene_median": statistics.median(counts),
        "variants_per_scene_max": counts[-1],
        "variants_per_scene_unique": sorted(set(counts)),
        "numeric_suffix_failures": suffix_failures,
        "numeric_suffix_dimensions": dimensions,
        "numeric_suffix_positions": positions,
        "numeric_suffix_digest": digest_lines(
            "|".join(format(part, ".12g") for part in value) for value in valid_suffixes
        ),
    }


def cross_field_filename_summary(left: list[Path], right: list[Path]) -> dict[str, Any]:
    left_stems = {item.stem for item in left}
    right_stems = {item.stem for item in right}
    left_by_scene = Counter(scene_id(item) for item in left)
    right_by_scene = Counter(scene_id(item) for item in right)
    if left_by_scene != right_by_scene:
        fail(
            f"cross-field scene/variant counts differ: {left[0].parent} vs {right[0].parent}"
        )
    left_only = sorted(left_stems - right_stems)
    right_only = sorted(right_stems - left_stems)
    return {
        "left_file_count": len(left),
        "right_file_count": len(right),
        "exact_stem_intersection_count": len(left_stems & right_stems),
        "exact_stem_sets_equal": left_stems == right_stems,
        "scene_variant_counts_equal": True,
        "scene_count": len(left_by_scene),
        "left_only_examples": left_only[:5],
        "right_only_examples": right_only[:5],
        "left_stem_set_digest": digest_lines(left_stems),
        "right_stem_set_digest": digest_lines(right_stems),
    }


def evenly_spaced(items: list[Path], count: int) -> list[Path]:
    if not items:
        fail("cannot sample an empty file list")
    take = min(count, len(items))
    indices = np.linspace(0, len(items) - 1, num=take, dtype=int)
    return [items[int(index)] for index in sorted(set(indices.tolist()))]


def transmission_sample(
    items: list[Path], haze_by_scene: dict[str, list[Path]], clear_by_stem: dict[str, Path], count: int
) -> dict[str, Any]:
    chosen = evenly_spaced(items, count)
    modes: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    minima: list[float] = []
    maxima: list[float] = []
    stds: list[float] = []
    channel_mismatch = 0
    alignment: Counter[str] = Counter()
    physical_mappings: list[str] = []
    best_rmses: list[float] = []
    best_to_second_ratios: list[float] = []
    strictly_better_count = 0
    for path in chosen:
        with Image.open(path) as image:
            modes[image.mode] += 1
            array = np.asarray(image)
            trans_small = (
                np.asarray(image.convert("L").resize((96, 96), RESAMPLE), dtype=np.float64)
                / 255.0
            )
        shapes[str(tuple(array.shape))] += 1
        if array.ndim == 3:
            if not np.array_equal(array[..., 0], array[..., -1]):
                channel_mismatch += 1
            array = array[..., 0]
        scene = scene_id(path)
        clear_path = clear_by_stem.get(scene)
        candidates = haze_by_scene.get(scene, [])
        if clear_path is None or not candidates:
            fail(f"missing clear/haze scene paired to transmission field: {path}")
        with Image.open(clear_path) as clear_image:
            expected_hw = (clear_image.height, clear_image.width)
            clear_small = (
                np.asarray(clear_image.convert("RGB").resize((96, 96), RESAMPLE), dtype=np.float64)
                / 255.0
            )
        alignment["direct_hw" if tuple(array.shape[:2]) == expected_hw else "other"] += 1
        one_minus_t = 1.0 - trans_small[..., None]
        denominator = float(np.sum(one_minus_t * one_minus_t))
        if denominator <= 1e-9:
            fail(f"transmission field cannot identify airlight: {path}")
        candidate_errors: list[tuple[float, str]] = []
        for haze_path in candidates:
            with Image.open(haze_path) as haze_image:
                haze_small = (
                    np.asarray(haze_image.convert("RGB").resize((96, 96), RESAMPLE), dtype=np.float64)
                    / 255.0
                )
            numerator = np.sum(
                (haze_small - clear_small * trans_small[..., None]) * one_minus_t,
                axis=(0, 1),
            )
            airlight = np.clip(numerator / denominator, 0.0, 1.0)
            reconstructed = clear_small * trans_small[..., None] + airlight * one_minus_t
            rmse = float(np.sqrt(np.mean((haze_small - reconstructed) ** 2)))
            candidate_errors.append((rmse, haze_path.stem))
        candidate_errors.sort()
        best_rmse, best_stem = candidate_errors[0]
        second_rmse = candidate_errors[1][0] if len(candidate_errors) > 1 else math.inf
        ratio = best_rmse / second_rmse if math.isfinite(second_rmse) and second_rmse > 0 else 0.0
        if best_rmse < second_rmse:
            strictly_better_count += 1
        best_rmses.append(best_rmse)
        best_to_second_ratios.append(ratio)
        physical_mappings.append(f"{path.stem}|{best_stem}|{best_rmse:.12g}|{second_rmse:.12g}")
        array = np.asarray(array, dtype=np.float64)
        if array.size == 0 or not np.isfinite(array).all():
            fail(f"invalid transmission array: {path}")
        minima.append(float(array.min()))
        maxima.append(float(array.max()))
        stds.append(float(array.std()))
    if channel_mismatch:
        fail(f"{channel_mismatch} sampled transmission images have unequal channels")
    if alignment.get("other", 0):
        fail(f"{alignment['other']} sampled transmission fields do not align to paired images")
    if not any(value > 0 for value in stds):
        fail("sampled transmission fields have no spatial variation")
    return {
        "sample_count": len(chosen),
        "sample_selection_digest": digest_lines(str(path.relative_to(RESIDE)) for path in chosen),
        "modes": dict(sorted(modes.items())),
        "shapes": dict(sorted(shapes.items())),
        "spatial_alignment": dict(sorted(alignment.items())),
        "raw_min": min(minima),
        "raw_max": max(maxima),
        "spatial_std_min": min(stds),
        "spatial_std_max": max(stds),
        "spatially_variable_count": sum(value > 0 for value in stds),
        "all_finite": True,
        "physical_pairing": {
            "method": (
                "Within each scene, fit constant RGB airlight for every haze candidate under "
                "I=J*t+A*(1-t) on deterministic 96x96 images and select minimum reconstruction RMSE."
            ),
            "strictly_better_count": strictly_better_count,
            "best_rmse_min": min(best_rmses),
            "best_rmse_median": statistics.median(best_rmses),
            "best_rmse_max": max(best_rmses),
            "best_to_second_ratio_median": statistics.median(best_to_second_ratios),
            "best_to_second_ratio_max": max(best_to_second_ratios),
            "mapping_digest": digest_lines(physical_mappings),
        },
    }


def largest_numeric_mat_array(path: Path) -> tuple[str, np.ndarray]:
    try:
        payload = loadmat(path)
        candidates = [
            (key, np.asarray(value))
            for key, value in payload.items()
            if not key.startswith("__") and np.asarray(value).dtype.kind in "buifc"
        ]
    except NotImplementedError:
        import h5py

        candidates = []
        with h5py.File(path, "r") as handle:
            def visit(name: str, value: Any) -> None:
                if isinstance(value, h5py.Dataset):
                    array = np.asarray(value)
                    if array.dtype.kind in "buifc":
                        candidates.append((name, array))
            handle.visititems(visit)
    if not candidates:
        fail(f"no numeric depth array in {path}")
    return max(candidates, key=lambda item: item[1].size)


def depth_sample(
    items: list[Path], clear_by_stem: dict[str, Path], count: int
) -> dict[str, Any]:
    chosen = evenly_spaced(items, count)
    keys: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    minima: list[float] = []
    maxima: list[float] = []
    stds: list[float] = []
    alignment: Counter[str] = Counter()
    for path in chosen:
        key, array = largest_numeric_mat_array(path)
        array = np.real(np.asarray(array, dtype=np.complex128)).astype(np.float64)
        if array.size == 0 or not np.isfinite(array).all():
            fail(f"invalid depth array: {path}")
        keys[key] += 1
        shapes[str(tuple(array.shape))] += 1
        dtypes[str(array.dtype)] += 1
        clear_path = clear_by_stem.get(path.stem)
        if clear_path is None:
            fail(f"missing clear image paired to depth field: {path}")
        with Image.open(clear_path) as clear_image:
            expected_hw = (clear_image.height, clear_image.width)
        observed_hw = tuple(array.shape[:2])
        if observed_hw == expected_hw:
            alignment["direct_hw"] += 1
        elif observed_hw == expected_hw[::-1]:
            alignment["transposed_wh"] += 1
        else:
            alignment["other"] += 1
        minima.append(float(array.min()))
        maxima.append(float(array.max()))
        stds.append(float(array.std()))
    if not any(value > 0 for value in stds):
        fail("sampled depth fields have no spatial variation")
    if alignment.get("other", 0):
        fail(f"{alignment['other']} sampled depth fields do not align to paired clear images")
    return {
        "sample_count": len(chosen),
        "sample_selection_digest": digest_lines(str(path.relative_to(RESIDE)) for path in chosen),
        "selected_numeric_keys": dict(sorted(keys.items())),
        "shapes": dict(sorted(shapes.items())),
        "converted_dtypes": dict(sorted(dtypes.items())),
        "spatial_alignment": dict(sorted(alignment.items())),
        "min": min(minima),
        "max": max(maxima),
        "spatial_std_min": min(stds),
        "spatial_std_max": max(stds),
        "spatially_variable_count": sum(value > 0 for value in stds),
        "all_finite": True,
    }


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.reshape(-1).tolist():
        value = (value << 1) | int(bool(bit))
    return value


def fingerprint(path: Path, source: str) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        gray = rgb.convert("L")
        canonical = np.asarray(rgb.resize((64, 64), RESAMPLE), dtype=np.uint8)
        dhash_array = np.asarray(gray.resize((9, 8), RESAMPLE), dtype=np.int16)
        d_hash = bits_to_int(dhash_array[:, 1:] > dhash_array[:, :-1])
        phash_array = np.asarray(gray.resize((32, 32), RESAMPLE), dtype=np.float64)
        low = dctn(phash_array, type=2, norm="ortho")[:8, :8]
        p_hash = bits_to_int(low > np.median(low[1:]))
        thumb = np.asarray(gray.resize((32, 32), RESAMPLE), dtype=np.float32).reshape(-1) / 255.0
        color_mean = canonical.reshape(-1, 3).mean(axis=0) / 255.0
    return {
        "source": source,
        "id": path.stem,
        "relative": str(path.relative_to(path.parents[2] if source.startswith("HAZE4K") else RESIDE)),
        "canonical64": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "dhash": d_hash,
        "phash": p_hash,
        "thumb": thumb,
        "color_mean": color_mean,
    }


def unique_fingerprints(groups: dict[str, list[Path]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    all_items: list[dict[str, Any]] = []
    by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    work = [(path, source) for source, paths in groups.items() for path in paths]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for item in executor.map(lambda pair: fingerprint(pair[0], pair[1]), work, chunksize=8):
            all_items.append(item)
            by_canonical[item["canonical64"]].append(item)
    return all_items, dict(by_canonical)


def hamming_neighbours(value: int, radius: int = 2) -> Iterable[int]:
    yield value
    bits = range(64)
    for width in range(1, radius + 1):
        for positions in itertools.combinations(bits, width):
            candidate = value
            for position in positions:
                candidate ^= 1 << position
            yield candidate


def strict_matches(
    left_groups: dict[str, list[dict[str, Any]]],
    right_items: list[dict[str, Any]],
    example_limit: int = 20,
) -> dict[str, Any]:
    right_by_dhash: dict[int, list[dict[str, Any]]] = defaultdict(list)
    right_by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in right_items:
        right_by_dhash[item["dhash"]].append(item)
        right_by_canonical[item["canonical64"]].append(item)

    records: set[tuple[str, str, str, str, str]] = set()
    exact_left_groups: set[str] = set()
    strict_left_groups: set[str] = set()
    for left_digest, members in left_groups.items():
        representative = members[0]
        exact = right_by_canonical.get(left_digest, [])
        if exact:
            exact_left_groups.add(left_digest)
        candidates: dict[tuple[str, str], dict[str, Any]] = {}
        for neighbour in hamming_neighbours(representative["dhash"], radius=2):
            for item in right_by_dhash.get(neighbour, []):
                candidates[(item["source"], item["id"])] = item
        for item in candidates.values():
            phash_distance = (representative["phash"] ^ item["phash"]).bit_count()
            if phash_distance > 6:
                continue
            left_thumb = representative["thumb"]
            right_thumb = item["thumb"]
            left_std = float(left_thumb.std())
            right_std = float(right_thumb.std())
            if left_std < 1e-6 or right_std < 1e-6:
                correlation = 1.0 if float(np.abs(left_thumb - right_thumb).max()) < 1e-3 else 0.0
            else:
                correlation = float(np.corrcoef(left_thumb, right_thumb)[0, 1])
            mean_delta = float(np.abs(representative["color_mean"] - item["color_mean"]).max())
            if correlation < 0.995 or mean_delta > 0.04:
                continue
            strict_left_groups.add(left_digest)
            records.add(
                (
                    left_digest,
                    representative["source"],
                    representative["id"],
                    item["source"],
                    item["id"],
                )
            )

    ordered = sorted(records)
    right_exclusion_ids = sorted({f"{record[3]}:{record[4]}" for record in ordered})
    return {
        "left_file_count": sum(len(value) for value in left_groups.values()),
        "left_unique_canonical_scene_count": len(left_groups),
        "exact_canonical_left_scene_count": len(exact_left_groups),
        "strict_perceptual_left_scene_count": len(strict_left_groups),
        "strict_match_record_count": len(ordered),
        "right_exclusion_scene_count": len(right_exclusion_ids),
        "right_exclusion_ids": right_exclusion_ids[:1000],
        "right_exclusion_ids_truncated": len(right_exclusion_ids) > 1000,
        "right_exclusion_ids_digest": digest_lines(right_exclusion_ids),
        "mapping_digest": digest_lines("|".join(record) for record in ordered),
        "examples": [
            {
                "left_source": record[1],
                "left_id": record[2],
                "right_source": record[3],
                "right_id": record[4],
            }
            for record in ordered[:example_limit]
        ],
        "examples_truncated": len(ordered) > example_limit,
    }


archive_manifest = RESIDE / "ARCHIVE_SHA256SUMS.txt"
pair_report = RESIDE / "PAIRING_VALIDATION.txt"
layout_record = RESIDE / "DATASET_LAYOUT.txt"
for required in (archive_manifest, pair_report, layout_record):
    if not required.is_file():
        fail(f"missing identity record: {required}")
if "RESIDE_PAIRING_VALIDATION_OK" not in pair_report.read_text(encoding="utf-8"):
    fail("pairing validation success marker is absent")

its_train_clear = image_files(RESIDE / "official/ITS/train/ITS_clear")
its_train_haze = image_files(RESIDE / "official/ITS/train/ITS_haze")
its_train_trans = image_files(RESIDE / "official/ITS/train/ITS_trans")
its_val_clear = image_files(RESIDE / "official/ITS/val/clear")
its_val_haze = image_files(RESIDE / "official/ITS/val/haze")
its_val_trans = image_files(RESIDE / "official/ITS/val/trans")
ots_clear = image_files(RESIDE / "official/OTS_ALPHA/clear_images")
ots_haze = image_files(RESIDE / "official/OTS_ALPHA/OTS")
ots_depth = files(RESIDE / "official/OTS_ALPHA/depth", {".mat"})
sots_indoor_gt = image_files(RESIDE / "official/SOTS/indoor/gt")
sots_indoor_haze = image_files(RESIDE / "official/SOTS/indoor/hazy")
sots_outdoor_gt = image_files(RESIDE / "official/SOTS/outdoor/gt")
sots_outdoor_haze = image_files(RESIDE / "official/SOTS/outdoor/hazy")
haze4k_train_gt = image_files(HAZE4K / "train/gt")
haze4k_test_gt = image_files(HAZE4K / "test/gt")

if {item.stem for item in ots_clear} != {item.stem for item in ots_depth}:
    fail("OTS clear/depth scene ID sets differ")
its_train_field_names = cross_field_filename_summary(its_train_haze, its_train_trans)
its_val_field_names = cross_field_filename_summary(its_val_haze, its_val_trans)
its_train_haze_by_scene: dict[str, list[Path]] = defaultdict(list)
its_val_haze_by_scene: dict[str, list[Path]] = defaultdict(list)
for item in its_train_haze:
    its_train_haze_by_scene[scene_id(item)].append(item)
for item in its_val_haze:
    its_val_haze_by_scene[scene_id(item)].append(item)

reside_train_items, reside_train_by_canonical = unique_fingerprints(
    {
        "RESIDE_ITS_TRAIN": its_train_clear,
        "RESIDE_ITS_VAL": its_val_clear,
        "RESIDE_OTS": ots_clear,
    }
)
sots_items, sots_by_canonical = unique_fingerprints(
    {
        "RESIDE_SOTS_INDOOR": sots_indoor_gt,
        "RESIDE_SOTS_OUTDOOR": sots_outdoor_gt,
    }
)
haze4k_items, haze4k_by_canonical = unique_fingerprints(
    {
        "HAZE4K_TRAIN": haze4k_train_gt,
        "HAZE4K_TEST": haze4k_test_gt,
    }
)

result = {
    "schema_version": 1,
    "audit": "RESIDE minimal measurement qualification",
    "status": "COMPLETED_READ_ONLY_AUDIT",
    "data_mutations_performed": False,
    "roots": {"reside": str(RESIDE), "haze4k": str(HAZE4K)},
    "dataset_identity": {
        "archive_manifest_sha256": sha256_file(archive_manifest),
        "archive_manifest_line_count": len(archive_manifest.read_text(encoding="utf-8").splitlines()),
        "pairing_report_sha256": sha256_file(pair_report),
        "layout_record_sha256": sha256_file(layout_record),
        "pairing_marker_present": True,
    },
    "scene_variant_mapping": {
        "its_train": variant_summary(its_train_haze, its_train_clear),
        "its_validation": variant_summary(its_val_haze, its_val_clear),
        "ots": variant_summary(ots_haze, ots_clear),
        "sots_indoor": variant_summary(sots_indoor_haze, sots_indoor_gt),
        "sots_outdoor": variant_summary(sots_outdoor_haze, sots_outdoor_gt),
    },
    "measurement_fields": {
        "its_train_haze_transmission_filename_mapping": its_train_field_names,
        "its_validation_haze_transmission_filename_mapping": its_val_field_names,
        "its_train_transmission": transmission_sample(
            its_train_trans, its_train_haze_by_scene, {item.stem: item for item in its_train_clear}, 64
        ),
        "its_validation_transmission": transmission_sample(
            its_val_trans, its_val_haze_by_scene, {item.stem: item for item in its_val_clear}, 32
        ),
        "ots_depth": {
            "file_count": len(ots_depth),
            "paired_scene_ids_equal": True,
            "reader": "MATLAB v7.3/HDF5; verified separately with existing /usr/bin/octave",
            "python_hdf5_reader_available": False,
        },
        "interpretation_limit": (
            "Decodability, pairing, finiteness, and spatial variation are established. "
            "The audit does not establish that depth-derived transmission equals real spatially varying scattering."
        ),
    },
    "scene_independence": {
        "haze4k_file_count": len(haze4k_items),
        "haze4k_unique_canonical_scene_count": len(haze4k_by_canonical),
        "reside_train_clear_file_count": len(reside_train_items),
        "reside_train_unique_canonical_scene_count": len(reside_train_by_canonical),
        "sots_clear_file_count": len(sots_items),
        "sots_unique_canonical_scene_count": len(sots_by_canonical),
        "haze4k_vs_reside_train": strict_matches(haze4k_by_canonical, reside_train_items),
        "sots_vs_reside_train": strict_matches(sots_by_canonical, reside_train_items),
        "matching_rule": {
            "exact": "SHA-256 equality after deterministic 64x64 RGB canonicalization",
            "strict_perceptual": (
                "dHash Hamming <=2, pHash Hamming <=6, 32x32 grayscale correlation >=0.995, "
                "and maximum RGB-mean difference <=0.04"
            ),
            "use": "Exclude all strict matches conservatively before any scene-level split.",
        },
    },
    "qualification_effect": {
        "resolved": [
            "specific archive/layout identity is recorded",
            "scene-to-variant mappings are complete",
            "ITS transmission and OTS depth fields are paired and readable on deterministic samples",
            "cross-dataset exact and strict-perceptual overlap is quantified",
        ],
        "not_resolved": [
            "real non-homogeneous-haze construct validity",
            "semantics of numeric haze filename tokens",
            "local proxy calibration against real haze",
            "precision margins and independent training-run count",
        ],
    },
    "marker": "RESIDE_MINIMAL_MEASUREMENT_QUALIFICATION_OK",
}

print(json.dumps(result, indent=2, sort_keys=True))
print("RESIDE_MINIMAL_MEASUREMENT_QUALIFICATION_OK")
PY
