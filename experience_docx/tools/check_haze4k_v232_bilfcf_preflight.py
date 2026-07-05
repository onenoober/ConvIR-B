import argparse
import hashlib
import inspect
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from data.data_load import DeblurDataset
from models.ConvIR import build_bilfcf_net, build_net


ALLOWED_NEW_PREFIXES = ("BILFCF_",)
FORBIDDEN_PATTERNS = (
    "teacher",
    "expert",
    "a0_output",
    "rgb_output_output",
    "post_output",
    "post-output",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_model(path, map_location):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_haze4k_partial(model, checkpoint_path, allowed_new_prefixes):
    state = load_checkpoint_model(checkpoint_path, "cpu")
    model_state = model.state_dict()
    loaded = {}
    shape_mismatch = []
    unexpected = []
    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
        elif model_state[key].shape != value.shape:
            shape_mismatch.append([key, list(value.shape), list(model_state[key].shape)])
        else:
            loaded[key] = value
    missing = [key for key in model_state if key not in loaded]
    bad_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_new_prefixes)
    ]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            "partial-load failed: "
            f"unexpected={unexpected[:20]} shape_mismatch={shape_mismatch[:20]} "
            f"bad_missing={bad_missing[:20]}"
        )
    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    return {
        "loaded_count": len(loaded),
        "missing_new_modules": sorted(missing),
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "bad_missing": bad_missing,
    }


def center_crop(tensor, size):
    _, h, w = tensor.shape
    if h < size or w < size:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(max(h, size), max(w, size)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        _, h, w = tensor.shape
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return tensor[:, top : top + size, left : left + size]


def count_split_images(data_dir, split):
    root = Path(data_dir) / split
    counts = {}
    for name in ("IN", "haze", "hazy", "GT", "gt"):
        path = root / name
        if path.is_dir():
            counts[name] = len(
                [
                    child for child in path.iterdir()
                    if child.is_file() and child.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
                ]
            )
    return counts


def count_params(model):
    return sum(param.numel() for param in model.parameters())


def count_trainable_by_scope(model):
    result = {"adapter_only": 0, "all": count_params(model)}
    for name, param in model.named_parameters():
        if name.startswith(ALLOWED_NEW_PREFIXES):
            result["adapter_only"] += param.numel()
    return result


def forbidden_symbol_hits():
    files = [
        ITS_ROOT / "models" / "ConvIR.py",
        ITS_ROOT / "main.py",
        ITS_ROOT / "train.py",
    ]
    hits = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8").lower()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                hits.append({"file": str(file_path.relative_to(REPO_ROOT)), "pattern": pattern})
    return hits


def write_markdown(path, result):
    lines = [
        "# v2.32 P0 Architecture Contract Delta",
        "",
        f"Route id: `{result['route_id']}`",
        "",
        "## Contract",
        "",
        f"- forward_contract: `{result['forward_contract']}`",
        f"- teacher_or_expert_forward_input: `{result['teacher_or_expert_forward_input']}`",
        f"- rgb_output_output_residual: `{result['rgb_output_output_residual']}`",
        f"- learned_rgb_post_output_correction: `{result['learned_rgb_post_output_correction']}`",
        f"- locked_test_touched: `{result['locked_test_touched']}`",
        "",
        "## Identity",
        "",
        f"- identity_max_abs_vs_A0: `{result['identity_max_abs_vs_A0']}`",
        f"- identity_mean_abs_vs_A0: `{result['identity_mean_abs_vs_A0']}`",
        f"- forbidden_symbol_hits: `{len(result['forbidden_symbol_hits'])}`",
        "",
        "## Partial Load",
        "",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- checkpoint_sha256: `{result['checkpoint_sha256']}`",
        f"- loaded_count: `{result['partial_load']['loaded_count']}`",
        f"- missing_new_modules_count: `{len(result['partial_load']['missing_new_modules'])}`",
        "",
        f"Pass: `{result['pass']}`",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown_output", default="")
    parser.add_argument("--route_id", default="haze4k_v2_32_nopost_bounded_internal_lowfreq_correction_field_20260705")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_abs_threshold", type=float, default=1e-6)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    official = build_net("base", "Haze4K", "original").to(device).eval()
    candidate = build_bilfcf_net("base", "Haze4K", "original").to(device).eval()
    official.load_state_dict(load_checkpoint_model(args.checkpoint, device))
    partial_load = load_haze4k_partial(candidate, args.checkpoint, ALLOWED_NEW_PREFIXES)
    candidate.to(device)

    synthetic = torch.rand(1, 3, args.crop_size, args.crop_size, device=device)
    with torch.no_grad():
        official_out = official(synthetic)
        candidate_out = candidate(synthetic)
    output_diffs = []
    for index, (base_tensor, candidate_tensor) in enumerate(zip(official_out, candidate_out)):
        delta = (base_tensor - candidate_tensor).abs()
        output_diffs.append(
            {
                "output_index": index,
                "max_abs": delta.max().item(),
                "mean_abs": delta.mean().item(),
            }
        )

    dataset = DeblurDataset(os.path.join(args.data_dir, "train"), "Haze4K", transform=None)
    input_img, label_img = dataset[0]
    input_img = center_crop(input_img, args.crop_size).unsqueeze(0).to(device)
    label_img = center_crop(label_img, args.crop_size).unsqueeze(0).to(device)
    with torch.no_grad():
        one_batch_outputs = candidate(input_img)
        one_batch_loss = F.l1_loss(one_batch_outputs[2], label_img).item()
        one_batch_forward_finite = all(torch.isfinite(out).all().item() for out in one_batch_outputs)

    signature_params = list(inspect.signature(candidate.forward).parameters)
    forward_contract = "forward(self, x)" if signature_params == ["x"] else f"forward({', '.join(signature_params)})"
    max_abs = max(item["max_abs"] for item in output_diffs)
    mean_abs = max(item["mean_abs"] for item in output_diffs)
    symbol_hits = forbidden_symbol_hits()
    result = {
        "route_id": args.route_id,
        "branch": os.popen("git branch --show-current").read().strip(),
        "commit": os.popen("git rev-parse --short HEAD").read().strip(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_dir": args.data_dir,
        "train_counts": count_split_images(args.data_dir, "train"),
        "test_counts": count_split_images(args.data_dir, "test"),
        "partial_load": partial_load,
        "parameter_count_total": count_params(candidate),
        "parameter_count_trainable_by_scope": count_trainable_by_scope(candidate),
        "synthetic_output_shapes": [list(out.shape) for out in candidate_out],
        "synthetic_forward_finite": all(torch.isfinite(out).all().item() for out in candidate_out),
        "synthetic_output_diffs": output_diffs,
        "noop_or_bounded_diff_vs_a0": max_abs,
        "identity_max_abs_vs_A0": max_abs,
        "identity_mean_abs_vs_A0": mean_abs,
        "one_batch_forward_finite": one_batch_forward_finite,
        "one_batch_loss": one_batch_loss,
        "initial_bilfcf_stats": candidate.get_bilfcf_stats(),
        "forward_contract": forward_contract,
        "teacher_or_expert_forward_input": False,
        "rgb_output_output_residual": False,
        "learned_rgb_post_output_correction": False,
        "forbidden_symbol_hits": symbol_hits,
        "locked_test_touched": False,
    }
    result["pass"] = (
        result["synthetic_forward_finite"]
        and result["one_batch_forward_finite"]
        and max_abs <= args.max_abs_threshold
        and len(symbol_hits) == 0
        and partial_load["unexpected"] == []
        and partial_load["shape_mismatch"] == []
        and partial_load["bad_missing"] == []
        and forward_contract == "forward(self, x)"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.markdown_output:
        write_markdown(args.markdown_output, result)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
