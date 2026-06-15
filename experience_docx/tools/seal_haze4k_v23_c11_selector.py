#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_c11_module(repo_root: Path):
    path = repo_root / "experience_docx/tools/analyze_haze4k_v23_c11_wd_fs_selector.py"
    spec = importlib.util.spec_from_file_location("c11_selector_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def encode_model(model: dict[str, Any]) -> str:
    return base64.b64encode(pickle.dumps(model, protocol=4)).decode("ascii")


def decode_model(payload: str) -> dict[str, Any]:
    return pickle.loads(base64.b64decode(payload.encode("ascii")))


def predict_from_sealed(sealed: dict[str, Any], rows: list[dict[str, Any]], c11) -> list[str]:
    model = decode_model(sealed["model_pickle_b64"])
    return c11.predict_model(model, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--c8-root", type=Path, default=Path("experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615"))
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    c8_root = args.c8_root if args.c8_root.is_absolute() else repo / args.c8_root
    out_dir.mkdir(parents=True, exist_ok=True)

    c11 = load_c11_module(repo)
    rows = c11.load_merged(c8_root)
    wd = c11.summarize(c11.profile_rows(rows, "WD0375", "WD0375_dPSNR", "WD0375_dSSIM", action="WD0375"))

    cfg = c11.select_config(rows, seed=23011)
    model = c11.train_model(rows, cfg)
    actions = c11.predict_model(model, rows)
    sealed_rows = c11.make_action_rows(rows, "sealed_train_full", actions, config=c11.config_name(cfg))
    summary = c11.summarize(sealed_rows)
    group_bins = c11.group_bins(sealed_rows, "sealed_train_full")
    group_pass = all(bool(r["group_gate_pass"]) for r in group_bins)
    aggregate_pass = c11.aggregate_pass(summary, wd)

    sealed = {
        "route": "Haze4K v2.3 C11 WD0375-FS050 Two-Profile Selector",
        "stage": "C11-E sealed train-derived selector",
        "policy": "Use this exact fitted model for locked replay; do not select config on locked data.",
        "allowed_actions": c11.ACTION_SET,
        "selected_config": c11.config_name(cfg),
        "config": cfg,
        "model_kind": model["kind"],
        "features": model.get("features", []),
        "model_pickle_b64": encode_model(model),
        "train_derived_summary": summary,
        "train_derived_aggregate_pass": aggregate_pass,
        "train_derived_group_pass": group_pass,
        "locked_test_touched": False,
        "locked_per_image_read": False,
        "distillation": False,
        "source_tables": [
            str(c8_root / "v22_c8_1_wdmamba_per_image.csv"),
            str(c8_root / "v22_c8_2_fsudp_per_image.csv"),
            str(c8_root / "v22_c8_3_mbtaylor_per_image.csv"),
        ],
    }
    (out_dir / "v23_c11e_sealed_selector.json").write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out_dir / "v23_c11e_sealed_selector_train_summary.csv", [{"selector": "sealed_train_full", **summary, "aggregate_gate_pass": aggregate_pass, "group_gate_pass": group_pass, "selected_config": c11.config_name(cfg)}])
    write_csv(out_dir / "v23_c11e_sealed_selector_group_bins.csv", group_bins)
    dist = []
    for action in c11.ACTION_SET:
        count = sum(a == action for a in actions)
        dist.append({"selector": "sealed_train_full", "action": action, "count": count, "usage": count / max(1, len(actions))})
    write_csv(out_dir / "v23_c11e_sealed_selector_action_distribution.csv", dist)
    decision = "C11E_SEALED_SELECTOR_PASS_READY_FOR_LOCKED_ONE_SHOT_REVIEW" if aggregate_pass and group_pass else "C11E_SEALED_SELECTOR_FAIL_LOCKED_BLOCKED"
    (out_dir / "v23_c11e_sealed_selector_decision.md").write_text(
        "# C11-E Sealed Selector Decision\n\n"
        f"Decision: `{decision}`\n\n"
        f"Selected config: `{c11.config_name(cfg)}`.\n\n"
        f"Train-derived sealed mean/hard/easy/positive/severe: `{summary['mean_dPSNR']:.6f}` / "
        f"`{summary['hard_bottom25_dPSNR']:.6f}` / `{summary['easy_top25_dPSNR']:.6f}` / "
        f"`{summary['positive_ratio']:.6f}` / `{summary['severe_loss_per_600']:.2f}/600`.\n\n"
        "Locked Haze4K is still untouched by C11-E. Locked replay, if authorized, must use `v23_c11e_sealed_selector.json` exactly.\n",
        encoding="utf-8",
    )
    print("C11E_SEALED_SELECTOR_OK")
    print(json.dumps({"decision": decision, "selected_config": c11.config_name(cfg)}, sort_keys=True))


if __name__ == "__main__":
    main()
