#!/usr/bin/env python3
"""Stage-0 preflight for v2.41 A0-proximal residual architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(obj), indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    return state


def load_partial(model: torch.nn.Module, state: dict[str, torch.Tensor], prefixes: tuple[str, ...]):
    model_state = model.state_dict()
    loaded = {}
    unexpected = []
    shape_mismatch = []
    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
        elif model_state[key].shape != value.shape:
            shape_mismatch.append([key, list(value.shape), list(model_state[key].shape)])
        else:
            loaded[key] = value
    missing = [key for key in model_state if key not in loaded]
    bad_missing = [key for key in missing if not any(key.startswith(p) for p in prefixes)]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            f"partial load failed unexpected={unexpected[:5]} "
            f"shape_mismatch={shape_mismatch[:5]} bad_missing={bad_missing[:20]}"
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


def max_output_diff(a, b) -> float:
    return max(float((x - y).abs().max().item()) for x, y in zip(a, b))


def forbidden_symbol_report(paths: list[Path]) -> dict[str, list[str]]:
    forbidden = ("cv2", "PIL", "Image.", "numpy", "np.", "skimage", "imwrite", "postprocess")
    report: dict[str, list[str]] = {}
    for path in paths:
        hits = []
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits.append(token)
        report[str(path.relative_to(repo_root()))] = hits
    return report


def run(args: argparse.Namespace) -> None:
    root = repo_root()
    sys.path.insert(0, str(root / "Dehazing" / "ITS"))
    from models.ConvIR import build_net
    from models.A0ProxResidualConvIR import build_a0prox_residual_net

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint)
    state = load_checkpoint_model(ckpt)
    official = build_net("base", "Haze4K", "original")
    route = build_a0prox_residual_net("base", "Haze4K", beta=args.beta)
    official.load_state_dict(state, strict=True)
    partial = load_partial(route, state, ("A0PROX_",))
    official.eval()
    route.eval()

    torch.manual_seed(241)
    synthetic = torch.rand(1, 3, args.synthetic_size, args.synthetic_size)
    with torch.no_grad():
        official_synth = official(synthetic)
        route_synth = route(synthetic)
    synth_diff = max_output_diff(route_synth, official_synth)
    finite_outputs = all(torch.isfinite(x).all().item() for x in route_synth)
    output_shapes = [list(x.shape) for x in route_synth]

    new_keys = partial["missing_new_modules"]
    new_param_count = sum(
        p.numel() for name, p in route.named_parameters() if name.startswith("A0PROX_")
    )
    total_param_count = sum(p.numel() for p in route.parameters())
    forbidden = forbidden_symbol_report(
        [
            root / "Dehazing" / "ITS" / "models" / "A0ProxResidualConvIR.py",
            root / "Dehazing" / "ITS" / "main.py",
        ]
    )
    forbidden_hits = sum(len(v) for v in forbidden.values())
    beta = float(route.A0PROX_beta.item())

    report = {
        "allowed_new_prefixes": ["A0PROX_"],
        "anchor_source": "github/codex/haze4k-official-arch-anchor",
        "checkpoint": {
            "path": str(ckpt),
            "size": ckpt.stat().st_size,
            "sha256": sha256(ckpt),
        },
        "finite_outputs": bool(finite_outputs),
        "forbidden_symbol_hits": forbidden_hits,
        "forbidden_symbol_report": forbidden,
        "identity_max_abs_vs_official_synthetic": synth_diff,
        "locked_test_touched": False,
        "new_module_init": {
            "A0PROX_head_last_weight_abs_max": float(route.A0PROX_head[2].weight.abs().max().item()),
            "A0PROX_head_last_bias_abs_max": float(route.A0PROX_head[2].bias.abs().max().item()),
            "A0PROX_beta": beta,
        },
        "new_param_count": int(new_param_count),
        "output_shapes": output_shapes,
        "partial_load": partial,
        "route": "haze4k_v2_41_a0_proximal_supervised_residual",
        "stage": "P0_STAGE0_PREFLIGHT",
        "total_param_count": int(total_param_count),
    }
    gate_pass = (
        report["checkpoint"]["sha256"]
        == "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
        and finite_outputs
        and synth_diff == 0.0
        and forbidden_hits == 0
        and beta <= args.beta
        and not partial["unexpected"]
        and not partial["shape_mismatch"]
        and not partial["bad_missing"]
        and not report["locked_test_touched"]
    )
    report["gate_pass"] = bool(gate_pass)
    report["decision"] = "P0_STAGE0_PREFLIGHT_PASS" if gate_pass else "P0_STAGE0_PREFLIGHT_FAIL"
    write_json(out_dir / "v241_p0_stage0_preflight.json", report)
    closeout = {
        "decision": report["decision"],
        "gate_pass": bool(gate_pass),
        "locked_test_touched": False,
        "canary32_authorized": bool(gate_pass),
        "canary80_authorized": False,
        "locked_test_authorized": False,
        "status": "COMPLETED_GATE_PASS" if gate_pass else "PREFLIGHT_FAILED_ENGINEERING",
    }
    write_json(out_dir / "v241_p0_closeout.json", closeout)
    print(json.dumps({"decision": report["decision"], "gate_pass": bool(gate_pass)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--synthetic-size", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.05)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
