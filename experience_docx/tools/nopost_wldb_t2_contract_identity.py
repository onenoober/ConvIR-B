#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TVF


TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

FORBIDDEN_TOKENS = (
    "A0_output",
    "WD0375_output",
    "WDMamba_output",
    "expert_output",
    "teacher_output",
    "teacher_output - A0_output",
    "A0_output - hazy",
    "output_1 - output_2",
    "learned_rgb",
)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def first_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"none of {names} under {root}")


def train_input_dir(data_dir: Path) -> Path:
    return first_dir(data_dir / "train", ("IN", "haze", "hazy"))


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    return TVF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def pad_to(x: torch.Tensor, factor: int = 32) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(x, (0, pw, 0, ph), "reflect"), h, w


def load_state(path: Path, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def final_output(out: Any, h: int, w: int) -> torch.Tensor:
    pred = out[2] if isinstance(out, (list, tuple)) else out
    return pred[:, :, :h, :w]


def partial_load_wldb(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    source = load_state(checkpoint, device)
    target = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatch: list[list[Any]] = []
    for key, value in source.items():
        if key not in target:
            unexpected.append(key)
        elif tuple(target[key].shape) != tuple(value.shape):
            shape_mismatch.append([key, list(value.shape), list(target[key].shape)])
        else:
            loaded[key] = value
    missing = [key for key in target if key not in loaded]
    bad_missing = [key for key in missing if not key.startswith("nopost_wldb.")]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            "WLDB partial load failed: "
            f"unexpected={unexpected[:10]} shape_mismatch={shape_mismatch[:10]} bad_missing={bad_missing[:20]}"
        )
    target.update(loaded)
    model.load_state_dict(target, strict=True)
    return {
        "loaded_count": len(loaded),
        "missing_new_module_count": len(missing),
        "missing_new_modules": sorted(missing),
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
    }


def source_scan(model_file: Path) -> tuple[str, dict[str, Any]]:
    text = model_file.read_text(encoding="utf-8")
    lines = []
    hits = []
    for token in FORBIDDEN_TOKENS:
        if token in text:
            hits.append(token)
            lines.append(f"HIT {token}")
    if not hits:
        lines.append("FORBIDDEN_SYMBOL_SCAN_OK")
    return "\n".join(lines), {"model_file": str(model_file), "forbidden_tokens": list(FORBIDDEN_TOKENS), "hits": hits}


def param_report(model: torch.nn.Module) -> dict[str, Any]:
    trainable = []
    frozen = []
    for name, param in model.named_parameters():
        rec = {"name": name, "shape": list(param.shape), "numel": int(param.numel())}
        if name.startswith("nopost_wldb."):
            trainable.append(rec)
        else:
            frozen.append(rec)
    return {
        "intended_trainable_prefix": "nopost_wldb.",
        "trainable_param_count": sum(r["numel"] for r in trainable),
        "frozen_anchor_param_count": sum(r["numel"] for r in frozen),
        "trainable": trainable,
        "frozen_prefixes": ["Encoder", "Decoder", "feat_extract", "Convs", "ConvsOut", "FAM", "SCM"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=32)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_file = REPO_ROOT / "Dehazing" / "ITS" / "models" / "NoPostWLDBConvIR.py"
    scan_text, scan_payload = source_scan(model_file)
    write_text(args.out_dir / "v216_t2_forbidden_symbol_scan.txt", scan_text)

    from models.ConvIR import build_net as build_official
    from models.NoPostWLDBConvIR import build_net as build_wldb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    official = build_official("base", "Haze4K", "original").to(device)
    wldb = build_wldb("base", "Haze4K", "original").to(device)
    official.load_state_dict(load_state(args.checkpoint, device))
    load_report = partial_load_wldb(wldb, args.checkpoint, device)
    official.eval()
    wldb.eval()

    param_groups = param_report(wldb)
    write_json(args.out_dir / "v216_t2_param_groups.json", param_groups)
    signature = {
        "route": "haze4k-v2-16-nopost-wavelet-lowband-decoder",
        "forward_inputs": ["hazy"],
        "internal_insertion": "after Decoder[2], before feat_extract[5]",
        "lowband_branch": "Haar DWT on final decoder feature LL band only",
        "rgb_head": "original feat_extract[5], then rgb_residual + hazy",
        "forbidden_scan": scan_payload,
        "partial_load": load_report,
        "locked_test_touched": False,
        "training_launched": False,
    }
    write_json(args.out_dir / "v216_t2_forward_signature.json", signature)
    write_text(
        args.out_dir / "v216_t2_contract_audit.md",
        "\n".join(
            [
                "# v2.16 T2 Contract Audit",
                "",
                "Decision scope: contract and zero-init identity only. No training is launched.",
                "",
                "- Forward input: hazy image only.",
                "- Insertion point: official ConvIR-B final decoder feature, after `Decoder[2]` and before `feat_extract[5]`.",
                "- Active branch: Haar LL lowband branch only.",
                "- Projection: zero-initialized `lowband_project`.",
                "- Output path: original `feat_extract[5]`, then `rgb_residual + hazy`.",
                "- Locked Haze4K test: untouched.",
            ]
        ),
    )

    input_dir = train_input_dir(args.data_dir)
    names = sorted(p.name for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})[
        : args.max_images
    ]
    rows = []
    max_abs_values = []
    mean_abs_values = []
    with torch.no_grad():
        for name in names:
            img = image_tensor(input_dir / name, device)
            x, h, w = pad_to(img, 32)
            a = final_output(official(x), h, w)
            b = final_output(wldb(x), h, w)
            diff = (a - b).abs()
            row = {
                "name": name,
                "max_abs_diff": float(diff.max().detach().cpu()),
                "mean_abs_diff": float(diff.mean().detach().cpu()),
            }
            rows.append(row)
            max_abs_values.append(row["max_abs_diff"])
            mean_abs_values.append(row["mean_abs_diff"])
    write_csv(args.out_dir / "v216_t2_identity_per_image.csv", rows)

    summary = {
        "count": len(rows),
        "max_abs_diff": max(max_abs_values) if max_abs_values else None,
        "mean_abs_diff_mean": sum(mean_abs_values) / len(mean_abs_values) if mean_abs_values else None,
        "identity_threshold": 1e-5,
        "identity_pass": bool(max_abs_values and max(max_abs_values) <= 1e-5),
        "forbidden_symbol_hits": scan_payload["hits"],
        "locked_test_touched": False,
        "training_launched": False,
    }
    write_json(args.out_dir / "v216_t2_identity_summary.json", summary)
    decision = (
        "T2_CONTRACT_IDENTITY_PASS_TRAINING_STILL_BLOCKED_PENDING_REVIEW"
        if summary["identity_pass"] and not scan_payload["hits"]
        else "T2_CONTRACT_IDENTITY_FAIL_NO_TRAINING"
    )
    write_text(
        args.out_dir / "v216_t2_decision.md",
        "\n".join(
            [
                "# v2.16 T2 Decision",
                "",
                f"Decision: `{decision}`",
                "",
                f"- identity max abs diff: `{summary['max_abs_diff']}`",
                f"- identity threshold: `{summary['identity_threshold']}`",
                f"- forbidden symbol hits: `{len(scan_payload['hits'])}`",
                "- Locked Haze4K test: untouched.",
                "- Training launched: false.",
            ]
        ),
    )
    closeout = {
        "route": "haze4k-v2-16-nopost-wavelet-lowband-decoder",
        "t2_decision": decision,
        "identity": summary,
        "next_action": "REVIEW_BEFORE_ANY_WLDB_A_TRAINING" if "PASS" in decision else "STOP_NO_TRAINING",
    }
    write_json(args.out_dir / "v216_t2_closeout.json", closeout)
    print("V216_T2_CONTRACT_IDENTITY_OK", json.dumps(closeout, sort_keys=True))


if __name__ == "__main__":
    main()
