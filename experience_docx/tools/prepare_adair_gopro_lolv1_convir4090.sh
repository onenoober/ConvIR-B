#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/sda/data/wangyuxin/dataset/data
GOPRO_ARCHIVE=$DATA_ROOT/GOPRO_Large.zip
LOL_ROOT=$DATA_ROOT/LOLdataset
GOPRO_EXTRACT_ROOT=$DATA_ROOT/raw/GOPRO_Large
PYTHON=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python

fail() {
  echo "ADAIR_DATASETS_PREPARE_FAILED: $*" >&2
  exit 2
}

[[ -x "$PYTHON" ]] || fail "required Python is unavailable: $PYTHON"
[[ -d "$DATA_ROOT" ]] || fail "missing data root: $DATA_ROOT"
[[ -f "$GOPRO_ARCHIVE" ]] || fail "missing GoPro archive: $GOPRO_ARCHIVE"
[[ -d "$LOL_ROOT" ]] || fail "missing LOL-V1 root: $LOL_ROOT"

"$PYTHON" - "$DATA_ROOT" "$GOPRO_ARCHIVE" "$LOL_ROOT" "$GOPRO_EXTRACT_ROOT" <<'PY'
from __future__ import annotations

import os
import stat
import sys
import zipfile
from pathlib import Path


DATA_ROOT = Path(sys.argv[1]).resolve()
GOPRO_ARCHIVE = Path(sys.argv[2]).resolve()
LOL_ROOT = Path(sys.argv[3]).resolve()
GOPRO_EXTRACT_ROOT = Path(sys.argv[4]).resolve()
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
EXPECTED_COUNTS = {
    "gopro_train": 2103,
    "gopro_test": 1111,
    "lol_train": 485,
    "lol_test": 15,
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def ensure_empty_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if not path.is_dir():
            fail(f"destination is not a directory: {path}")
        if any(path.iterdir()):
            fail(f"destination is not empty: {path}")


def safe_extract_zip(archive: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if not destination.is_dir() or any(destination.iterdir()):
            fail(f"GoPro extraction destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = Path(member.filename)
            target = (destination / member_path).resolve()
            if target != destination_root and destination_root not in target.parents:
                fail(f"GoPro archive has an unsafe member path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                fail(f"GoPro archive has a symbolic-link member: {member.filename}")
        bundle.extractall(destination)


def directory_images(directory: Path, label: str) -> dict[str, Path]:
    if not directory.is_dir():
        fail(f"missing {label} directory: {directory}")
    result = {item.name: item.resolve() for item in directory.iterdir() if is_image(item)}
    if not result:
        fail(f"no images in {label} directory: {directory}")
    return result


def require_matching_pair(input_dir: Path, target_dir: Path, label: str) -> tuple[dict[str, Path], dict[str, Path]]:
    inputs = directory_images(input_dir, f"{label} input")
    targets = directory_images(target_dir, f"{label} target")
    missing_targets = sorted(set(inputs) - set(targets))
    missing_inputs = sorted(set(targets) - set(inputs))
    if missing_targets or missing_inputs:
        fail(
            f"unpaired images in {label}: missing_targets={missing_targets[:3]}; "
            f"missing_inputs={missing_inputs[:3]}"
        )
    return inputs, targets


def split_for(path: Path, train_names: set[str], test_names: set[str]) -> str | None:
    components = {part.lower() for part in path.parts}
    if components & train_names:
        return "train"
    if components & test_names:
        return "test"
    return None


def gopro_pairs(root: Path, split: str) -> tuple[dict[str, Path], dict[str, Path]]:
    inputs: dict[str, Path] = {}
    targets: dict[str, Path] = {}
    scenes = []
    for sharp_dir in sorted(root.rglob("sharp")):
        if not sharp_dir.is_dir():
            continue
        if split_for(sharp_dir, {"train"}, {"test"}) != split:
            continue
        blur_dir = sharp_dir.parent / "blur_gamma"
        if not blur_dir.is_dir():
            blur_dir = sharp_dir.parent / "blur"
        if not blur_dir.is_dir():
            fail(f"missing blur_gamma (or blur) beside sharp directory: {sharp_dir}")
        scene_name = sharp_dir.parent.name
        source_inputs, source_targets = require_matching_pair(
            blur_dir, sharp_dir, f"GoPro {split} scene {scene_name}"
        )
        for filename in sorted(source_inputs):
            flat_name = f"{scene_name}_{filename}"
            if flat_name in inputs:
                fail(f"duplicate GoPro flattened filename: {flat_name}")
            inputs[flat_name] = source_inputs[filename]
            targets[flat_name] = source_targets[filename]
        scenes.append(scene_name)
    if not scenes:
        fail(f"could not find GoPro {split} scene pairs below {root}")
    return inputs, targets


def lol_pairs(root: Path, split: str) -> tuple[dict[str, Path], dict[str, Path]]:
    expected_parent = "our485" if split == "train" else "eval15"
    inputs: dict[str, Path] = {}
    targets: dict[str, Path] = {}
    for low_dir in sorted(root.rglob("low")):
        if not low_dir.is_dir() or low_dir.parent.name.lower() != expected_parent:
            continue
        high_dir = low_dir.parent / "high"
        if not high_dir.is_dir():
            high_dir = low_dir.parent / "gt"
        if not high_dir.is_dir():
            fail(f"missing high (or gt) beside LOL-V1 low directory: {low_dir}")
        source_inputs, source_targets = require_matching_pair(
            low_dir, high_dir, f"LOL-V1 {split}"
        )
        for filename in sorted(source_inputs):
            if filename in inputs:
                fail(f"duplicate LOL-V1 filename: {filename}")
            inputs[filename] = source_inputs[filename]
            targets[filename] = source_targets[filename]
    if not inputs:
        fail(f"could not find LOL-V1 {expected_parent}/low and matching high directories")
    return inputs, targets


def require_expected_count(label: str, inputs: dict[str, Path]) -> None:
    expected = EXPECTED_COUNTS[label]
    if len(inputs) != expected:
        fail(f"{label} pair count expected {expected}, found {len(inputs)}")


def materialize_pairs(
    inputs: dict[str, Path], targets: dict[str, Path], input_dir: Path, target_dir: Path, label: str
) -> None:
    ensure_empty_directory(input_dir)
    ensure_empty_directory(target_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in sorted(inputs):
        os.symlink(inputs[filename], input_dir / filename)
        os.symlink(targets[filename], target_dir / filename)
    print(f"ADAIR_DATASET_COUNT={len(inputs)}\t{label}")


def audit_pairs(
    inputs: dict[str, Path], targets: dict[str, Path], input_dir: Path, target_dir: Path, label: str
) -> None:
    input_names = {item.name for item in input_dir.iterdir() if is_image(item)}
    target_names = {item.name for item in target_dir.iterdir() if is_image(item)}
    expected_names = set(inputs)
    if input_names != expected_names or target_names != expected_names:
        fail(f"{label} destination filename set does not match its verified source pairs")
    for filename in expected_names:
        input_link = input_dir / filename
        target_link = target_dir / filename
        if not input_link.is_symlink() or not target_link.is_symlink():
            fail(f"{label} contains a non-symbolic-link output: {filename}")
        if not os.path.samefile(input_link, inputs[filename]):
            fail(f"{label} input link resolves to the wrong source: {filename}")
        if not os.path.samefile(target_link, targets[filename]):
            fail(f"{label} target link resolves to the wrong source: {filename}")
    print(f"ADAIR_DATASET_AUDIT={len(expected_names)}\t{label}")


try:
    destination_dirs = (
        (DATA_ROOT / "Train" / "Deblur" / "blur", DATA_ROOT / "Train" / "Deblur" / "sharp"),
        (DATA_ROOT / "test" / "deblur" / "gopro" / "input", DATA_ROOT / "test" / "deblur" / "gopro" / "target"),
        (DATA_ROOT / "Train" / "Enhance" / "low", DATA_ROOT / "Train" / "Enhance" / "gt"),
        (DATA_ROOT / "test" / "enhance" / "lol" / "input", DATA_ROOT / "test" / "enhance" / "lol" / "target"),
    )
    for input_dir, target_dir in destination_dirs:
        ensure_empty_directory(input_dir)
        ensure_empty_directory(target_dir)
    lol_train = lol_pairs(LOL_ROOT, "train")
    lol_test = lol_pairs(LOL_ROOT, "test")
    safe_extract_zip(GOPRO_ARCHIVE, GOPRO_EXTRACT_ROOT)
    gopro_train = gopro_pairs(GOPRO_EXTRACT_ROOT, "train")
    gopro_test = gopro_pairs(GOPRO_EXTRACT_ROOT, "test")
    for label, (inputs, _) in (
        ("gopro_train", gopro_train),
        ("gopro_test", gopro_test),
        ("lol_train", lol_train),
        ("lol_test", lol_test),
    ):
        require_expected_count(label, inputs)

    destinations = (
        (gopro_train, DATA_ROOT / "Train" / "Deblur" / "blur", DATA_ROOT / "Train" / "Deblur" / "sharp", "GOPRO_TRAIN"),
        (gopro_test, DATA_ROOT / "test" / "deblur" / "gopro" / "input", DATA_ROOT / "test" / "deblur" / "gopro" / "target", "GOPRO_TEST"),
        (lol_train, DATA_ROOT / "Train" / "Enhance" / "low", DATA_ROOT / "Train" / "Enhance" / "gt", "LOL_V1_TRAIN"),
        (lol_test, DATA_ROOT / "test" / "enhance" / "lol" / "input", DATA_ROOT / "test" / "enhance" / "lol" / "target", "LOL_V1_TEST"),
    )
    for (inputs, targets), input_dir, target_dir, label in destinations:
        materialize_pairs(inputs, targets, input_dir, target_dir, label)
    for (inputs, targets), input_dir, target_dir, label in destinations:
        audit_pairs(inputs, targets, input_dir, target_dir, label)
    print(f"ADAIR_GOPRO_EXTRACT_ROOT={GOPRO_EXTRACT_ROOT}")
    print("ADAIR_DATASETS_PREPARE_OK")
except (OSError, RuntimeError, zipfile.BadZipFile) as error:
    print(f"ADAIR_DATASETS_PREPARE_FAILED: {error}", file=sys.stderr)
    raise SystemExit(2)
PY
