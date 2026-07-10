#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
sys.path.insert(0, str(ITS_ROOT))

from d7c_gate import (  # noqa: E402
    build_d7c_gate_producer,
    forward_with_optional_d7c,
    partial_load_model_state,
)
from models.ConvIR import build_net  # noqa: E402
from train import (  # noqa: E402
    RARM_TRAINABLE_KEYS,
    _configure_train_scope,
    _set_train_mode_for_scope,
)


ROUTE_ID = "haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710"
DECISION_PASS = "V3D_RARM_STAGE0_PREFLIGHT_PASS_AUTHORIZE_STAGE1_1EPOCH_ADAPTER_ONLY"
DECISION_FAIL = "V3D_RARM_STAGE0_PREFLIGHT_FAIL_NO_TRAINING"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint_state(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_split(path):
    with open(path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    if "splits" not in split:
        raise ValueError(f"Split JSON has no 'splits': {path}")
    return split


def label_path_for_hazy(data_dir, source_split, hazy_name):
    stem, ext = os.path.splitext(hazy_name)
    candidates = [hazy_name, f"{stem.split('_')[0]}{ext}", f"{stem.split('_')[0]}.png"]
    label_dir = Path(data_dir) / source_split / "gt"
    for candidate in candidates:
        path = label_dir / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"No GT match for {hazy_name}; tried {candidates} under {label_dir}")


def load_pair(data_dir, source_split, hazy_name):
    hazy_path = Path(data_dir) / source_split / "haze" / hazy_name
    label_path = label_path_for_hazy(data_dir, source_split, hazy_name)
    if not hazy_path.is_file():
        raise FileNotFoundError(f"Hazy image missing: {hazy_path}")
    image = Image.open(hazy_path).convert("RGB")
    label = Image.open(label_path).convert("RGB")
    return TF.to_tensor(image), TF.to_tensor(label)


def pad_to_factor(tensor, factor=32):
    height, width = tensor.shape[2], tensor.shape[3]
    padded_height = ((height + factor) // factor) * factor
    padded_width = ((width + factor) // factor) * factor
    pad_height = padded_height - height if height % factor != 0 else 0
    pad_width = padded_width - width if width % factor != 0 else 0
    if pad_height or pad_width:
        tensor = F.pad(tensor, (0, pad_width, 0, pad_height), "reflect")
    return tensor, height, width


def crop_final(output, height, width):
    return output[:, :, :height, :width]


def tensor_stats(tensor):
    tensor = tensor.detach()
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "max_abs": tensor.abs().max().item(),
    }


def make_gate_args(args):
    class GateArgs:
        pass

    gate_args = GateArgs()
    gate_args.version = "base"
    gate_args.data = "Haze4K"
    gate_args.fam_mode = "fam2_d7c_noop"
    gate_args.d7c_gate_mode = "d7c_fixed"
    gate_args.d7c_base_checkpoint = args.checkpoint
    gate_args.d7c_density_artifact = args.density_artifact
    gate_args.d7c_need_artifact = args.d7c_artifact
    gate_args.d7c_threshold = args.d7c_threshold
    return gate_args


def make_train_args(args, gate_producer):
    class TrainArgs:
        pass

    train_args = TrainArgs()
    train_args.rarm_train_scope = "fam2_modulator_only"
    train_args.d7c_gate_producer = gate_producer
    return train_args


def source_contract(repo_root):
    expected = [
        ("Dehazing/ITS/main.py", "--rarm_train_scope"),
        ("Dehazing/ITS/main.py", "fam2_modulator_only"),
        ("Dehazing/ITS/main.py", "d7c_gate.py"),
        ("Dehazing/ITS/train.py", "RARM_TRAINABLE_KEYS"),
        ("Dehazing/ITS/train.py", "_configure_train_scope(model, args)"),
        ("Dehazing/ITS/train.py", "torch.optim.Adam(trainable_parameters"),
        ("Dehazing/ITS/train.py", "torch.nn.utils.clip_grad_norm_(trainable_parameters"),
    ]
    checks = []
    for relative_path, needle in expected:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        checks.append({"path": relative_path, "needle": needle, "pass": needle in text})
    return {"checks": checks, "pass": all(item["pass"] for item in checks)}


def assert_no_locked_test_paths(args):
    if args.source_split != "train":
        raise ValueError(f"v3d Stage 0 may only sample from Haze4K train, got {args.source_split}")
    lowered = " ".join(
        str(value).lower()
        for value in [
            args.data_dir,
            args.split_json,
            args.output_dir,
            args.checkpoint,
            args.density_artifact,
            args.d7c_artifact,
        ]
    )
    forbidden = ["locked", "final-test", "final_test"]
    hits = [token for token in forbidden if token in lowered]
    if hits:
        raise ValueError(f"Forbidden locked-test-like path token(s): {hits}")


def multiscale_loss(outputs, label_img):
    label_img2 = F.interpolate(label_img, scale_factor=0.5, mode="bilinear")
    label_img4 = F.interpolate(label_img, scale_factor=0.25, mode="bilinear")
    l1 = F.l1_loss(outputs[0], label_img4)
    l2 = F.l1_loss(outputs[1], label_img2)
    l3 = F.l1_loss(outputs[2], label_img)
    loss_content = l1 + l2 + l3

    fft_loss = 0.0
    for pred, label in [(outputs[0], label_img4), (outputs[1], label_img2), (outputs[2], label_img)]:
        label_fft = torch.fft.fft2(label, dim=(-2, -1))
        label_fft = torch.stack((label_fft.real, label_fft.imag), -1)
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        fft_loss = fft_loss + F.l1_loss(pred_fft, label_fft)
    return loss_content + 0.1 * fft_loss, loss_content, fft_loss


def gradient_audit(model):
    rows = []
    nonzero_trainable = 0
    nonzero_frozen = 0
    finite = True
    for name, param in model.named_parameters():
        if param.grad is None:
            grad_max = None
            grad_mean = None
            grad_finite = True
        else:
            grad = param.grad.detach()
            grad_max = grad.abs().max().item()
            grad_mean = grad.abs().mean().item()
            grad_finite = bool(torch.isfinite(grad).all().item())
            finite = finite and grad_finite
            if grad_max > 0 and param.requires_grad:
                nonzero_trainable += 1
            if grad_max > 0 and not param.requires_grad:
                nonzero_frozen += 1
        rows.append(
            {
                "name": name,
                "requires_grad": bool(param.requires_grad),
                "grad_max_abs": grad_max,
                "grad_mean_abs": grad_mean,
                "grad_finite": grad_finite,
            }
        )
    return {
        "rows": rows,
        "nonzero_trainable_grad_count": nonzero_trainable,
        "nonzero_frozen_grad_count": nonzero_frozen,
        "all_gradients_finite": finite,
    }


def run_preflight(args):
    assert_no_locked_test_paths(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in [args.checkpoint, args.data_dir, args.split_json, args.density_artifact, args.d7c_artifact]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_state = load_checkpoint_state(args.checkpoint)
    base = build_net("base", "Haze4K", "original").to(device).eval()
    candidate = build_net("base", "Haze4K", "fam2_d7c_noop").to(device)
    base.load_state_dict(checkpoint_state, strict=True)
    candidate_load = partial_load_model_state(
        candidate,
        checkpoint_state,
        set(RARM_TRAINABLE_KEYS),
    )

    gate_args = make_gate_args(args)
    gate_args.d7c_gate_producer = build_d7c_gate_producer(gate_args, device)
    train_args = make_train_args(args, gate_args.d7c_gate_producer)
    trainable_params = _configure_train_scope(candidate, train_args)
    _set_train_mode_for_scope(candidate, train_args)

    trainable_names = [name for name, param in candidate.named_parameters() if param.requires_grad]
    frozen_count = sum(param.numel() for _, param in candidate.named_parameters() if not param.requires_grad)
    trainable_count = sum(param.numel() for _, param in candidate.named_parameters() if param.requires_grad)

    split = load_split(args.split_json)
    names = sorted(split["splits"][args.split_key])[: args.max_samples]
    if not names:
        raise ValueError(f"No names in split {args.split_key}")

    rows = []
    max_noop_diff = 0.0
    gate_coverages = []
    nontrivial_gates = 0
    with torch.no_grad():
        for index, name in enumerate(names):
            input_img, _ = load_pair(args.data_dir, args.source_split, name)
            input_img = input_img.unsqueeze(0).to(device)
            padded, height, width = pad_to_factor(input_img)
            base_outputs = base(padded)
            candidate_outputs = forward_with_optional_d7c(candidate, train_args, padded)
            gate, _, _ = gate_args.d7c_gate_producer(padded)
            final_diff = (
                crop_final(base_outputs[2], height, width)
                - crop_final(candidate_outputs[2], height, width)
            ).abs()
            max_diff = final_diff.max().item()
            max_noop_diff = max(max_noop_diff, max_diff)
            gate_coverage = gate.mean().item()
            gate_coverages.append(gate_coverage)
            nontrivial = bool(gate.min().item() < gate.max().item())
            nontrivial_gates += int(nontrivial)
            rows.append(
                {
                    "index": index,
                    "name": name,
                    "gate_coverage": gate_coverage,
                    "gate_min": gate.min().item(),
                    "gate_max": gate.max().item(),
                    "gate_nontrivial": nontrivial,
                    "noop_final_max_abs_diff": max_diff,
                }
            )

    first_input, first_label = load_pair(args.data_dir, args.source_split, names[0])
    first_input = first_input.unsqueeze(0).to(device)
    first_label = first_label.unsqueeze(0).to(device)
    padded_input, height, width = pad_to_factor(first_input)
    padded_label, _, _ = pad_to_factor(first_label)

    candidate.zero_grad(set_to_none=True)
    _set_train_mode_for_scope(candidate, train_args)
    outputs = forward_with_optional_d7c(candidate, train_args, padded_input)
    loss, loss_content, loss_fft = multiscale_loss(outputs, padded_label)
    loss.backward()
    grad_audit = gradient_audit(candidate)

    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm).item()
    optimizer.step()
    candidate.eval()

    with torch.no_grad():
        base_outputs = base(padded_input)
        stepped_outputs = forward_with_optional_d7c(candidate, train_args, padded_input)
        one_step_diff = (
            crop_final(base_outputs[2], height, width)
            - crop_final(stepped_outputs[2], height, width)
        ).abs()

    static_contract = source_contract(REPO_ROOT)
    pass_checks = {
        "source_contract": static_contract["pass"],
        "partial_load_missing_exact": candidate_load["missing_candidate_keys"] == list(RARM_TRAINABLE_KEYS),
        "partial_load_clean": not candidate_load["unexpected_keys"]
        and not candidate_load["shape_mismatch"]
        and not candidate_load["bad_missing"],
        "trainable_names_exact": trainable_names == list(RARM_TRAINABLE_KEYS),
        "noop_equivalence": max_noop_diff <= args.noop_tolerance,
        "gate_nontrivial": nontrivial_gates > 0,
        "loss_finite": bool(torch.isfinite(loss).item()),
        "trainable_grad_nonzero": grad_audit["nonzero_trainable_grad_count"] > 0,
        "frozen_grad_zero": grad_audit["nonzero_frozen_grad_count"] == 0,
        "gradients_finite": grad_audit["all_gradients_finite"],
        "one_step_nonzero": one_step_diff.max().item() > args.one_step_min_diff,
        "one_step_bounded": one_step_diff.max().item() <= args.one_step_max_diff,
        "locked_test_touched_false": True,
    }
    passed = all(pass_checks.values())
    decision = DECISION_PASS if passed else DECISION_FAIL

    summary = {
        "route_id": ROUTE_ID,
        "decision": decision,
        "pass": passed,
        "pass_checks": pass_checks,
        "branch": os.popen("git branch --show-current").read().strip(),
        "commit": os.popen("git rev-parse HEAD").read().strip(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_dir": args.data_dir,
        "split_json": args.split_json,
        "split_key": args.split_key,
        "sample_count": len(names),
        "source_split": args.source_split,
        "density_artifact": args.density_artifact,
        "d7c_artifact": args.d7c_artifact,
        "candidate_partial_load": candidate_load,
        "rarm_train_scope": "fam2_modulator_only",
        "rarm_trainable_keys": list(RARM_TRAINABLE_KEYS),
        "trainable_names": trainable_names,
        "parameter_count_trainable": trainable_count,
        "parameter_count_frozen": frozen_count,
        "parameter_count_total": trainable_count + frozen_count,
        "max_noop_final_abs_diff": max_noop_diff,
        "gate_nontrivial_count": nontrivial_gates,
        "gate_coverage_mean": float(sum(gate_coverages) / len(gate_coverages)),
        "gate_coverage_min": float(min(gate_coverages)),
        "gate_coverage_max": float(max(gate_coverages)),
        "one_batch_loss": {
            "total": loss.item(),
            "content": loss_content.item(),
            "fft": loss_fft.item() if hasattr(loss_fft, "item") else float(loss_fft),
            "finite": bool(torch.isfinite(loss).item()),
        },
        "gradient_audit": {
            key: value for key, value in grad_audit.items() if key != "rows"
        },
        "grad_clip_norm_before_clip": grad_norm,
        "one_step_output_diff": tensor_stats(one_step_diff),
        "source_contract": static_contract,
        "locked_test_touched": False,
        "forbidden_flows": {
            "formal_training": False,
            "checkpoint_saved": False,
            "adapter_training": False,
            "locked_test": False,
        },
    }
    write_csv(output_dir / "v3d_stage0_preflight_per_sample.csv", rows)
    write_csv(output_dir / "v3d_stage0_gradient_audit.csv", grad_audit["rows"])
    write_json(output_dir / "v3d_stage0_preflight_summary.json", summary)
    write_json(
        output_dir / "v3d_stage0_preflight_closeout.json",
        {
            "route_id": ROUTE_ID,
            "decision": decision,
            "pass": passed,
            "next_action": (
                "Stage 1 one-epoch adapter-only cloud smoke is authorized."
                if passed
                else "Do not launch RARM training; inspect failed Stage 0 checks."
            ),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--split_key", default="val_inner")
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--d7c_threshold", type=float, default=0.5773006677627563)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=0.001)
    parser.add_argument("--noop_tolerance", type=float, default=1e-7)
    parser.add_argument("--one_step_min_diff", type=float, default=1e-12)
    parser.add_argument("--one_step_max_diff", type=float, default=0.05)
    args = parser.parse_args()
    raise SystemExit(run_preflight(args))


if __name__ == "__main__":
    main()
