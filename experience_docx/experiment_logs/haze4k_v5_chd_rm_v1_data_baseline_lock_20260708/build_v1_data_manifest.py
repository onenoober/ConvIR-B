import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(path: Path):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)


def image_id_from_hazy(path: Path) -> str:
    return path.stem.split("_")[0]


def find_trans(trans_dir: Path, hazy_name: str, image_id: str):
    for name in (hazy_name, f"{image_id}.png"):
        p = trans_dir / name
        if p.exists():
            return p
    return None


def stats_for_strata(path: Path):
    img = Image.open(path).convert("RGB").resize((64, 64), Image.BILINEAR)
    pix = list(img.getdata())
    n = len(pix)
    mean_r = sum(p[0] for p in pix) / (255.0 * n)
    mean_g = sum(p[1] for p in pix) / (255.0 * n)
    mean_b = sum(p[2] for p in pix) / (255.0 * n)
    brightness = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b
    saturation = sum(max(p) - min(p) for p in pix) / (255.0 * n)
    gray = [0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in pix]
    w, h = img.size
    total = 0.0
    count = 0
    for y in range(h - 1):
        row = y * w
        nxt = (y + 1) * w
        for x in range(w - 1):
            total += abs(gray[row + x + 1] - gray[row + x])
            total += abs(gray[nxt + x] - gray[row + x])
            count += 2
    gradient = total / (255.0 * max(1, count))
    return brightness, saturation, gradient


def quantile_cuts(values, buckets):
    ordered = sorted(values)
    return [ordered[round((len(ordered) - 1) * i / buckets)] for i in range(1, buckets)]


def bucket(value, cuts):
    return sum(value > c for c in cuts)


def build_manifest(data_dir: Path, split: str):
    root = data_dir / split
    haze_dir = root / "haze"
    gt_dir = root / "gt"
    trans_dir = root / "trans"
    rows = []
    errors = []
    seen_ids = set()
    for hazy in list_images(haze_dir):
        image_id = image_id_from_hazy(hazy)
        gt = gt_dir / f"{image_id}.png"
        trans = find_trans(trans_dir, hazy.name, image_id)
        if image_id in seen_ids:
            errors.append({"type": "duplicate_image_id", "split": split, "image_id": image_id, "hazy": str(hazy)})
            continue
        seen_ids.add(image_id)
        if not gt.exists():
            errors.append({"type": "missing_gt", "split": split, "image_id": image_id, "hazy": str(hazy)})
            continue
        if trans is None:
            errors.append({"type": "missing_trans", "split": split, "image_id": image_id, "hazy": str(hazy)})
            continue
        with Image.open(hazy) as img:
            width, height = img.size
        rows.append({
            "image_id": image_id,
            "hazy_name": hazy.name,
            "hazy_path": str(hazy),
            "gt_path": str(gt),
            "trans_path": str(trans),
            "width": width,
            "height": height,
            "sha256_hazy": sha256(hazy),
            "sha256_gt": sha256(gt),
            "sha256_trans": sha256(trans),
            "split": split,
            "fold_id": "",
        })
    return rows, errors


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def make_internal_split(train_rows, seed):
    stat_rows = []
    for row in train_rows:
        b, s, g = stats_for_strata(Path(row["hazy_path"]))
        stat_rows.append({"image_id": row["image_id"], "brightness": b, "saturation": s, "gradient": g})
    bcuts = quantile_cuts([r["brightness"] for r in stat_rows], 5)
    gcuts = quantile_cuts([r["gradient"] for r in stat_rows], 3)
    strata = defaultdict(list)
    for r in stat_rows:
        key = (bucket(r["brightness"], bcuts), bucket(r["gradient"], gcuts), int(r["brightness"] >= 0.62 and r["gradient"] <= 0.035), int(r["brightness"] >= 0.58 and r["saturation"] <= 0.12))
        strata[key].append(r["image_id"])
    rng = random.Random(seed)
    selected = []
    total = len(train_rows)
    val_count = 600
    for key, ids in sorted(strata.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        quota = round(len(ids) * val_count / total)
        selected.extend(ids[:quota])
    selected = sorted(set(selected))
    if len(selected) < val_count:
        remaining = sorted(set(r["image_id"] for r in train_rows) - set(selected))
        rng.shuffle(remaining)
        selected = sorted(selected + remaining[: val_count - len(selected)])
    elif len(selected) > val_count:
        rng.shuffle(selected)
        selected = sorted(selected[:val_count])
    val = set(selected)
    return [{"image_id": r["image_id"], "split_role": "val_inner" if r["image_id"] in val else "train_inner"} for r in train_rows]


def make_folds(train_rows, seed):
    ids = sorted(r["image_id"] for r in train_rows)
    rng = random.Random(seed + 5000)
    rng.shuffle(ids)
    fold_for = {image_id: idx % 5 for idx, image_id in enumerate(ids)}
    out = []
    for r in train_rows:
        for fold_id in range(5):
            role = "val" if fold_for[r["image_id"]] == fold_id else "train"
            out.append({"image_id": r["image_id"], "fold_id": fold_id, "fold_role": role})
    return out


def duplicate_hashes(rows_a, rows_b, key):
    a = defaultdict(list)
    b = defaultdict(list)
    for r in rows_a:
        a[r[key]].append(r["image_id"])
    for r in rows_b:
        b[r[key]].append(r["image_id"])
    return {h: {"train": a[h], "test": b[h]} for h in sorted(set(a) & set(b))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_rows, train_errors = build_manifest(data_dir, "train")
    test_rows, test_errors = build_manifest(data_dir, "test")
    fields = ["image_id", "hazy_name", "hazy_path", "gt_path", "trans_path", "width", "height", "sha256_hazy", "sha256_gt", "sha256_trans", "split", "fold_id"]
    write_csv(out / "haze4k_manifest_train.csv", train_rows, fields)
    write_csv(out / "haze4k_manifest_test.csv", test_rows, fields)

    split_rows = make_internal_split(train_rows, args.seed)
    write_csv(out / "haze4k_internal_split_2400_600.csv", split_rows, ["image_id", "split_role"])
    fold_rows = make_folds(train_rows, args.seed)
    write_csv(out / "haze4k_oof_folds.csv", fold_rows, ["image_id", "fold_id", "fold_role"])

    split_counts = Counter(r["split_role"] for r in split_rows)
    fold_counts = Counter((r["fold_id"], r["fold_role"]) for r in fold_rows)
    leakage = {
        "train_test_hazy_hash_overlap": duplicate_hashes(train_rows, test_rows, "sha256_hazy"),
        "train_test_gt_hash_overlap": duplicate_hashes(train_rows, test_rows, "sha256_gt"),
        "train_errors": train_errors,
        "test_errors": test_errors,
    }
    leakage["pass"] = not leakage["train_test_hazy_hash_overlap"] and not leakage["train_test_gt_hash_overlap"] and not train_errors and not test_errors
    summary = {
        "data_dir": str(data_dir),
        "seed": args.seed,
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "train_errors": train_errors,
        "test_errors": test_errors,
        "internal_split_counts": dict(split_counts),
        "fold_val_counts": {str(k): v for k, v in sorted(fold_counts.items()) if k[1] == "val"},
        "leakage_pass": leakage["pass"],
        "non_image_note": "Image-extension filtering is required; train/haze contains .DS_Store in the cloud dataset.",
    }
    (out / "haze4k_file_hash_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "leakage_audit.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    (out / "split_policy.md").write_text(
        "# CHD-RM v1 Split Policy\n\n"
        "- Source data: Haze4K train 3000 only for train-derived selection.\n"
        "- Internal split: stratified 2400 train_inner / 600 val_inner, seed 3407.\n"
        "- OOF: five folds from train 3000, 600 validation images per fold.\n"
        "- Locked test: not used for tuning; only counted for leakage accounting.\n"
        "- Non-image files are ignored by extension filtering.\n",
        encoding="utf-8",
    )
    (out / "data_manifest_summary.md").write_text(
        "# CHD-RM v1 Data Manifest Summary\n\n"
        f"- Train pairs: {len(train_rows)}\n"
        f"- Test pairs: {len(test_rows)}\n"
        f"- Internal split: train_inner={split_counts['train_inner']}, val_inner={split_counts['val_inner']}\n"
        "- OOF folds: 5 folds, 600 validation images per fold.\n"
        f"- Leakage audit pass: {leakage['pass']}\n"
        "- Note: `.DS_Store` in `train/haze` is ignored by image-extension filtering.\n",
        encoding="utf-8",
    )
    decision = "COMPLETED_DATA_GATE_PASS" if summary["train_count"] == 3000 and summary["test_count"] == 1000 and split_counts["train_inner"] == 2400 and split_counts["val_inner"] == 600 and leakage["pass"] else "DATA_GATE_FAIL"
    (out / "decision_record.md").write_text(
        "# CHD-RM v1 Data Manifest Decision\n\n"
        f"Decision: `{decision}`\n\n"
        "Locked test status: filenames and hashes inspected only for leakage accounting; no model score or tuning use.\n\n"
        "Next: write and review A0 val600 command contract before launching baseline evaluation.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"decision={decision}")


if __name__ == "__main__":
    main()
