#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def first_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"none of {names} under {root}")


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


def scan_forbidden(source: str) -> dict[str, Any]:
    patterns = {
        "A0_symbol": r"\bA0\b",
        "WD0375_symbol": r"\bWD0375\b",
        "WDMamba_symbol": r"\bWDMamba\b",
        "teacher_symbol": r"\bteacher\b",
        "expert_symbol": r"\bexpert\b",
        "output_subtraction_hint": r"output\s*-\s*output|candidate\s*-\s*A0|A0\s*-\s*candidate",
        "rgb_post_correction_hint": r"post.?correction|rgb.?correction",
    }
    hits: dict[str, Any] = {}
    for label, pattern in patterns.items():
        matches = []
        for lineno, line in enumerate(source.splitlines(), 1):
            if re.search(pattern, line, flags=re.IGNORECASE):
                matches.append({"line": lineno, "text": line.strip()})
        hits[label] = matches
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=8)
    args = ap.parse_args()

    from models.ConvIR import build_net as build_official
    from models.NoPostMidFinalContextLowbandConvIR import NoPostMidFinalContextLowbandConvIR

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_path = REPO_ROOT / "Dehazing" / "ITS" / "models" / "NoPostMidFinalContextLowbandConvIR.py"
    source = source_path.read_text(encoding="utf-8")
    hits = scan_forbidden(source)
    forbidden_hit_count = sum(len(v) for v in hits.values())
    write_text(
        args.out_dir / "v220_p0_forbidden_symbol_scan.txt",
        "\n".join(
            [
                "v2.20 P0 forbidden symbol scan",
                f"source={source_path}",
                f"forbidden_hit_count={forbidden_hit_count}",
                json.dumps(hits, indent=2, sort_keys=True),
            ]
        ),
    )

    signature = str(inspect.signature(NoPostMidFinalContextLowbandConvIR.forward))
    write_json(
        args.out_dir / "v220_p0_forward_signature.json",
        {
            "class": "NoPostMidFinalContextLowbandConvIR",
            "forward_signature": signature,
            "accepts_only_self_and_x": signature == "(self, x)",
            "locked_test_touched": False,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_state(args.checkpoint, device)
    official = build_official("base", "Haze4K", "original").to(device)
    official.load_state_dict(state)
    official.eval()

    input_dir = first_dir(args.data_dir / "train", ("IN", "haze", "hazy"))
    names = sorted(p.name for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})[: args.max_images]

    model = NoPostMidFinalContextLowbandConvIR("base", "Haze4K", hidden_channels=32).to(device)
    load_report = model.load_state_dict(state, strict=False)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    new_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("nopost_midfinal_context_policy."))
    frozen_official_params = total_params - new_params
    missing = list(load_report.missing_keys)
    unexpected = list(load_report.unexpected_keys)
    param_row = {
        "policy_mode": "mid_final_context",
        "total_params": total_params,
        "new_nopost_midfinal_context_policy_params": new_params,
        "official_params_loaded_or_reused": frozen_official_params,
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "missing_keys_all_new_prefix": all(k.startswith("nopost_midfinal_context_policy.") for k in missing),
        "unexpected_keys": unexpected,
        "missing_keys": missing,
    }

    max_abs = 0.0
    mean_abs_values = []
    with torch.no_grad():
        for name in names:
            img = image_tensor(input_dir / name, device)
            x, h, w = pad_to(img, 32)
            official_pred = final_output(official(x), h, w)
            policy_pred = final_output(model(x), h, w)
            diff = (policy_pred - official_pred).abs()
            max_abs = max(max_abs, float(diff.max().detach().cpu()))
            mean_abs_values.append(float(diff.mean().detach().cpu()))

    identity_row = {
        "policy_mode": "mid_final_context",
        "image_count": len(names),
        "max_abs_vs_A0": max_abs,
        "mean_abs_vs_A0": sum(mean_abs_values) / len(mean_abs_values) if mean_abs_values else 0.0,
        "identity_pass_1e-6": max_abs <= 1e-6,
    }

    write_json(
        args.out_dir / "v220_p0_param_groups.json",
        {
            "checkpoint": str(args.checkpoint),
            "policy_param_groups": [param_row],
            "partial_load_rule": "official keys load; only nopost_midfinal_context_policy.* may be missing",
            "locked_test_touched": False,
        },
    )
    write_json(
        args.out_dir / "v220_p0_identity_summary.json",
        {
            "identity_rows": [identity_row],
            "locked_test_touched": False,
        },
    )
    contract_pass = (
        forbidden_hit_count == 0
        and signature == "(self, x)"
        and param_row["missing_keys_all_new_prefix"]
        and param_row["unexpected_key_count"] == 0
        and identity_row["identity_pass_1e-6"]
    )
    decision = "P0_PASS_MIDFINAL_CONTEXT_CONTRACT_IDENTITY_SOURCE_CLEAN" if contract_pass else "P0_FAIL_MIDFINAL_CONTEXT_CONTRACT_IDENTITY"
    write_text(
        args.out_dir / "v220_p0_contract_audit.md",
        "\n".join(
            [
                "# v2.20 P0 Mid+Final Context Lowband Contract Audit",
                "",
                f"Decision: `{decision}`",
                "",
                f"- forward signature: `{signature}`",
                f"- forbidden symbol hit count: `{forbidden_hit_count}`",
                "- official checkpoint partial load allows only `nopost_midfinal_context_policy.*` missing keys.",
                "- zero-init mid+final context policy identity is checked against official A0 outputs on train images.",
                "- locked Haze4K remains untouched.",
            ]
        ),
    )
    write_text(
        args.out_dir / "v220_p0_decision.md",
        "\n".join(
            [
                "# v2.20 P0 Contract And Identity Decision",
                "",
                f"Decision: `{decision}`",
                "",
                f"- contract pass: `{contract_pass}`",
                f"- identity rows: `{[identity_row]}`",
                f"- param group rows: `{[param_row]}`",
                "",
                "No training or locked-test command is launched by P0.",
            ]
        ),
    )
    print("V220_P0_CONTRACT_IDENTITY_OK", decision, flush=True)
    if not contract_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
