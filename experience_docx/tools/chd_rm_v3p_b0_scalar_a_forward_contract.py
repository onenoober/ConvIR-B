#!/usr/bin/env python3
"""v3p-B0 train-only scalar-A forward-model contract for Haze4K."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROUTE_ID = "haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712"
EXPECTED_SAMPLE_IDS = set(range(1, 3001))
EXPECTED_HAZE_NON_PNG = (".DS_Store",)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines):
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else None


def stem_id(name):
    try:
        return int(Path(name).stem.split("_", 1)[0])
    except ValueError:
        return None


def collect_layout(data_dir):
    train = Path(data_dir) / "train"
    directories = {name: train / name for name in ("haze", "gt", "trans")}
    for path in directories.values():
        if not path.is_dir():
            raise FileNotFoundError(f"required train directory missing: {path}")
    issues = []
    files = {}
    for name, path in directories.items():
        entries = sorted(item.name for item in path.iterdir() if item.is_file())
        png = [item for item in entries if Path(item).suffix.lower() == ".png"]
        non_png = [item for item in entries if Path(item).suffix.lower() != ".png"]
        files[name] = {"entries": entries, "png": png, "non_png": non_png}
    if files["haze"]["non_png"] != list(EXPECTED_HAZE_NON_PNG):
        issues.append(f"haze non-PNG entries are {files['haze']['non_png']}, expected {list(EXPECTED_HAZE_NON_PNG)}")
    for name in ("gt", "trans"):
        if files[name]["non_png"]:
            issues.append(f"{name} contains non-PNG entries: {files[name]['non_png']}")
    maps = {}
    for name in ("gt", "trans"):
        mapping = {}
        for item in files[name]["png"]:
            sample_id = stem_id(item)
            if sample_id is None or Path(item).stem != str(sample_id):
                issues.append(f"{name} filename is not an integer PNG id: {item}")
                continue
            if sample_id in mapping:
                issues.append(f"duplicate {name} id: {sample_id}")
            mapping[sample_id] = item
        if set(mapping) != EXPECTED_SAMPLE_IDS:
            issues.append(f"{name} numeric id set differs from 1..3000")
        maps[name] = mapping
    haze_mapping = {}
    for item in files["haze"]["png"]:
        sample_id = stem_id(item)
        if sample_id is None:
            issues.append(f"haze filename has no numeric source id: {item}")
            continue
        if sample_id in haze_mapping:
            issues.append(f"multiple haze files map to source id {sample_id}")
        haze_mapping[sample_id] = item
    if set(haze_mapping) != EXPECTED_SAMPLE_IDS:
        issues.append("haze source-id set differs from 1..3000")
    if any(len(files[name]["png"]) != 3000 for name in files):
        issues.append("each train modality must contain exactly 3000 PNG files")
    maps["haze"] = haze_mapping
    layout = {
        "train_root": str(train),
        "entry_count": {name: len(value["entries"]) for name, value in files.items()},
        "png_count": {name: len(value["png"]) for name, value in files.items()},
        "non_png_entries": {name: value["non_png"] for name, value in files.items()},
        "layout_listing_sha256": sha256_lines(
            f"{name}/{item}" for name in sorted(files) for item in files[name]["entries"]
        ),
    }
    return train, maps, layout, issues


def srgb_to_linear(value):
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def read_rgb(path):
    with Image.open(path) as image:
        mode = image.mode
        size = image.size
        raw = np.asarray(image)
        value = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return value, {"mode": mode, "size": size, "dtype": str(raw.dtype), "raw_shape": list(raw.shape)}


def read_transmission(path):
    with Image.open(path) as image:
        mode = image.mode
        size = image.size
        raw = np.asarray(image)
    if raw.dtype != np.uint8:
        raise ValueError(f"transmission dtype must be uint8, got {raw.dtype}")
    if raw.ndim == 2:
        channel_count = 1
        value = raw.astype(np.float64) / 255.0
    elif raw.ndim == 3 and raw.shape[2] == 3:
        channel_count = 3
        if not (np.array_equal(raw[:, :, 0], raw[:, :, 1]) and np.array_equal(raw[:, :, 0], raw[:, :, 2])):
            raise ValueError("three-channel transmission PNG channels differ")
        value = raw[:, :, 0].astype(np.float64) / 255.0
    else:
        raise ValueError(f"unsupported transmission shape: {raw.shape}")
    return value, {"mode": mode, "size": size, "dtype": str(raw.dtype), "raw_shape": list(raw.shape), "channels": channel_count}


def fit_scalar_a(haze, clear, transmission, color_space):
    if color_space == "linear":
        haze = srgb_to_linear(haze)
        clear = srgb_to_linear(clear)
    weight = 1.0 - transmission
    denominator = 3.0 * math.fsum(float(value) for value in (weight * weight).ravel())
    if denominator <= 0.0:
        raise ValueError("scalar-A denominator is zero")
    numerator = math.fsum(float(value) for value in (weight[:, :, None] * (haze - transmission[:, :, None] * clear)).ravel())
    atmospheric_light = numerator / denominator
    reconstruction = transmission[:, :, None] * clear + weight[:, :, None] * atmospheric_light
    residual = haze - reconstruction
    mse = float(np.mean(residual * residual))
    return atmospheric_light, residual, mse


def names_from_manifest(path, key, count):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("locked_test_touched"):
        raise ValueError("B0 requires a train-only split manifest")
    names = sorted(manifest["splits"][key])[:count]
    if len(names) != count:
        raise ValueError(f"expected {count} names from {key}, got {len(names)}")
    return manifest, names


def shape_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["shape"]].append(row)
    result = []
    for shape, values in sorted(groups.items()):
        srmse = [row["srgb_forward_rmse"] for row in values]
        result.append({
            "shape": shape,
            "image_count": len(values),
            "srgb_forward_rmse_mean": float(np.mean(srmse)),
            "srgb_forward_rmse_p95": percentile(srmse, 95),
            "srgb_forward_rmse_p99": percentile(srmse, 99),
            "srgb_forward_rmse_max": float(np.max(srmse)),
        })
    return result


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "closeout": output_dir / f"{args.run_tag}_closeout.json",
        "summary": output_dir / f"{args.run_tag}_summary.json",
        "source": output_dir / f"{args.run_tag}_source_manifest.json",
        "contract": output_dir / f"{args.run_tag}_forward_contract.json",
        "images": output_dir / f"{args.run_tag}_per_image_cloud_only.csv",
        "shapes": output_dir / f"{args.run_tag}_by_shape.csv",
    }
    manifest, names = names_from_manifest(args.fresh_split_manifest, args.train_key, args.expected_images)
    train_root, maps, layout, issues = collect_layout(args.data_dir)
    rows = []
    block_mses = []
    modes = Counter()
    for name in names:
        sample_id = stem_id(name)
        if sample_id is None or maps["haze"].get(sample_id) != name:
            issues.append(f"selected haze name does not map uniquely to its source id: {name}")
            continue
        haze_path = train_root / "haze" / name
        gt_path = train_root / "gt" / maps["gt"].get(sample_id, "")
        trans_path = train_root / "trans" / maps["trans"].get(sample_id, "")
        if not gt_path.is_file() or not trans_path.is_file():
            issues.append(f"missing GT or transmission triplet member for {name}")
            continue
        try:
            haze, haze_info = read_rgb(haze_path)
            clear, clear_info = read_rgb(gt_path)
            transmission, trans_info = read_transmission(trans_path)
        except (OSError, ValueError) as exc:
            issues.append(f"cannot decode {name}: {exc}")
            continue
        if haze_info["mode"] != "RGB" or clear_info["mode"] != "RGB":
            issues.append(f"RGB source mode mismatch for {name}: haze={haze_info['mode']} gt={clear_info['mode']}")
            continue
        if haze_info["dtype"] != "uint8" or clear_info["dtype"] != "uint8":
            issues.append(f"RGB source dtype mismatch for {name}: haze={haze_info['dtype']} gt={clear_info['dtype']}")
            continue
        if haze.shape != clear.shape or haze.shape[:2] != transmission.shape:
            issues.append(f"shape mismatch for {name}: haze={haze.shape} gt={clear.shape} trans={transmission.shape}")
            continue
        srgb_a, srgb_residual, srgb_mse = fit_scalar_a(haze, clear, transmission, "srgb")
        linear_a, _, linear_mse = fit_scalar_a(haze, clear, transmission, "linear")
        for y0 in range(0, haze.shape[0], args.block_size):
            for x0 in range(0, haze.shape[1], args.block_size):
                block = srgb_residual[y0:y0 + args.block_size, x0:x0 + args.block_size]
                block_mses.append(float(np.mean(block * block)))
        modes[(haze_info["mode"], clear_info["mode"], trans_info["mode"], trans_info["channels"])] += 1
        rows.append({
            "name": name,
            "source_id": sample_id,
            "shape": f"{haze.shape[0]}x{haze.shape[1]}",
            "haze_mode": haze_info["mode"],
            "gt_mode": clear_info["mode"],
            "transmission_mode": trans_info["mode"],
            "transmission_channels": trans_info["channels"],
            "scalar_a_srgb": srgb_a,
            "scalar_a_linear": linear_a,
            "srgb_forward_mse": srgb_mse,
            "srgb_forward_rmse": math.sqrt(srgb_mse),
            "linear_forward_mse": linear_mse,
            "linear_forward_rmse": math.sqrt(linear_mse),
        })
    a_bound = 1.0 / 255.0
    max_srgb_rmse = max((row["srgb_forward_rmse"] for row in rows), default=float("inf"))
    scalar_a_in_range = all(-a_bound <= row["scalar_a_srgb"] <= 1.0 + a_bound for row in rows)
    structural_pass = not issues and len(rows) == args.expected_images
    numeric_pass = scalar_a_in_range and max_srgb_rmse <= args.maximum_srgb_rmse
    all_pass = structural_pass and numeric_pass
    if args.run_mode == "smoke":
        decision = (
            "V3P_B0_SCALAR_A_SMOKE_PASS_AUTHORIZE_FORMAL_OOF_ONLY"
            if all_pass else "V3P_B0_SCALAR_A_SMOKE_FAIL_STOP_PHYSICS_ROUTE"
        )
        authorizes = "v3p-B0 formal scalar-A OOF contract only" if all_pass else "none"
    else:
        decision = (
            "V3P_B0_SCALAR_A_FORWARD_CONTRACT_PASS_AUTHORIZE_B1_PRIVILEGED_TA_CEILING_ONLY"
            if all_pass else "V3P_B0_SCALAR_A_FORWARD_CONTRACT_FAIL_STOP_PHYSICS_ESTIMATOR_ROUTE"
        )
        authorizes = "v3p-B1 privileged t+A ceiling only" if all_pass else "none"
    contract = {
        "route_id": ROUTE_ID,
        "stage": "v3p-B0-scalar-A-forward-contract",
        "data_scope": "train-only v3j_controller_train OOF names; no canary or locked test",
        "pairing": "haze name numeric prefix -> gt/<id>.png and trans/<id>.png; ignore only train/haze/.DS_Store",
        "decode": "PIL source RGB PNG for haze/GT, uint8 transmission as L or exact replicated RGB, divided by 255; no resize, crop, interpolation, or BGR conversion",
        "primary_color_space": "sRGB encoded loader semantics",
        "linear_sensitivity": "IEC sRGB-to-linear transform is diagnostic only and cannot replace the frozen sRGB B1 contract",
        "scalar_a_formula": "sum_{p,c}(1-t_p)(I_{p,c}-t_p J_{p,c}) / sum_{p,c}(1-t_p)^2",
        "forward_formula": "I ~= t J + (1-t) A",
        "maximum_srgb_rmse": args.maximum_srgb_rmse,
        "scalar_a_range_tolerance": a_bound,
        "block_size": args.block_size,
        "B1_noise_floor": "block forward-residual MSE p99; B1 must abstain unless its physics score margin exceeds this MSE times block RGB pixel count",
    }
    source = {
        "route_id": ROUTE_ID,
        "script_sha256": sha256_file(__file__),
        "fresh_split_manifest_sha256": sha256_file(args.fresh_split_manifest),
        "selected_names_sha256": sha256_lines(names),
        "selected_images": len(names),
        "data_dir": str(Path(args.data_dir)),
        "layout": layout,
    }
    summary = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": "v3p-B0-scalar-A-forward-contract",
        "run_mode": args.run_mode,
        "state": "COMPLETED_GATE_PASS" if all_pass else "COMPLETED_GATE_FAIL",
        "gate_type": "physics_forward_contract",
        "decision": decision,
        "authorizes": authorizes,
        "reason": (
            "All train-only OOF triplets satisfy the frozen scalar-A pairing, decode, range, and sRGB forward-residual contract."
            if all_pass else "The frozen scalar-A pairing, decode, range, or sRGB forward-residual contract does not hold."
        ),
        "metric_contract": contract,
        "structural_pass": structural_pass,
        "numeric_pass": numeric_pass,
        "issues": issues,
        "image_count": len(rows),
        "expected_images": args.expected_images,
        "max_srgb_forward_rmse": max_srgb_rmse,
        "srgb_forward_rmse_p95": percentile([row["srgb_forward_rmse"] for row in rows], 95),
        "srgb_forward_rmse_p99": percentile([row["srgb_forward_rmse"] for row in rows], 99),
        "linear_forward_rmse_p99": percentile([row["linear_forward_rmse"] for row in rows], 99),
        "scalar_a_srgb_min": min((row["scalar_a_srgb"] for row in rows), default=None),
        "scalar_a_srgb_max": max((row["scalar_a_srgb"] for row in rows), default=None),
        "mode_counts": {"|".join(map(str, key)): value for key, value in sorted(modes.items())},
        "noise_floor_block_forward_residual_mse_p99": percentile(block_mses, 99),
        "noise_floor_block_forward_residual_mse_max": max(block_mses, default=None),
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    image_fields = [
        "name", "source_id", "shape", "haze_mode", "gt_mode", "transmission_mode", "transmission_channels",
        "scalar_a_srgb", "scalar_a_linear", "srgb_forward_mse", "srgb_forward_rmse", "linear_forward_mse", "linear_forward_rmse",
    ]
    write_rows(outputs["images"], image_fields, rows)
    write_rows(outputs["shapes"], [
        "shape", "image_count", "srgb_forward_rmse_mean", "srgb_forward_rmse_p95", "srgb_forward_rmse_p99", "srgb_forward_rmse_max",
    ], shape_summary(rows))
    write_json(outputs["contract"], contract)
    write_json(outputs["source"], source)
    write_json(outputs["closeout"], summary)
    write_json(outputs["summary"], {**summary, "source_manifest": source})
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fresh_split_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--run_mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--expected_images", type=int, required=True)
    parser.add_argument("--train_key", default="v3j_controller_train")
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--maximum_srgb_rmse", type=float, default=8.0 / 255.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
