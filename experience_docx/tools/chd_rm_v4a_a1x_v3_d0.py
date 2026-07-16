#!/usr/bin/env python3
"""A1X-v3 D0: matched 2x2 representation/readout development screen."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_accessibility_v3_20260716"
CELLS = {
    "context_local": {"representation": "context", "readout": "local", "shuffled": False},
    "context_global": {"representation": "context", "readout": "global", "shuffled": False},
    "a1x_local": {"representation": "a1x", "readout": "local", "shuffled": False},
    "a1x_global": {"representation": "a1x", "readout": "global", "shuffled": False},
    "a1x_global_shuffled": {"representation": "a1x", "readout": "global", "shuffled": True},
}
PRIMARY = "a1x_global"
SHUFFLED = "a1x_global_shuffled"
LOCAL_CONTROL = "a1x_local"
A1R: Any = None
ORIGINAL_PROBE: Any = None
ORIGINAL_TARGET_FEATURES: Any = None
ORIGINAL_BOOTSTRAP: Any = None
ORIGINAL_RUN: Any = None


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class D0Probe(nn.Module):
    def __init__(self, input_channels: int, readout: str, width: int, delta_bound: tuple[float, ...]):
        super().__init__()
        if readout == "local":
            self.inner = ORIGINAL_PROBE(input_channels, "spatial", width, delta_bound)
            self.global_readout = False
        elif readout == "global":
            repo = Path(os.environ["REMOTE_REPO"])
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from Dehazing.ITS.models.A1XAccess import A1X_ACCESS_Head
            self.inner = A1X_ACCESS_Head(input_channels=input_channels)
            self.register_buffer("_delta_bound", torch.tensor(delta_bound).view(1, 3, 1, 1))
            self.global_readout = True
        else:
            raise ValueError(readout)

    @property
    def delta_bound(self) -> torch.Tensor:
        return self._delta_bound if self.global_readout else self.inner.delta_bound

    def forward(self, features: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        if not self.global_readout:
            return self.inner(features, support)
        raw = self.inner(features)
        return support * (2.0 * self.delta_bound.to(device=raw.device, dtype=raw.dtype) * raw)


def item_features(item: dict[str, Any], representation: str) -> torch.Tensor:
    if representation == "a1x":
        return item["a1x_features"]
    if representation == "context":
        return torch.cat((item["context_base"], item["output_features"]), dim=1)
    raise ValueError(representation)


def endpoint_loss(correction, current, target, support, bound):
    endpoint = torch.maximum(torch.minimum(current + correction, bound), -bound)
    active = (support > 0.0).expand_as(endpoint).to(endpoint.dtype)
    per_pixel = (((endpoint - target) / bound.clamp_min(1e-8)).square() * active)
    per_item = per_pixel.flatten(1).sum(1) / active.flatten(1).sum(1).clamp_min(1.0)
    return per_item.mean()


def target_and_features(**kwargs):
    cached, sample = ORIGINAL_TARGET_FEATURES(**kwargs)
    v3s = kwargs["v3s"]
    for operator, item in cached.items():
        step = sample["steps"][operator]
        current = sample[f"current:{operator}"]
        zero = torch.zeros_like(current)
        old_125, old_250, _, _ = v3s.candidate_predictions(sample["base"], step, zero)
        size = item["target_low"].shape[-2:]
        parts = [sample["hazy"], sample["base"], old_125, old_250, current]
        item["a1x_features"] = torch.cat([
            F.interpolate(value, size=size, mode="bilinear", align_corners=False, antialias=False)
            for value in parts
        ], dim=1).detach().cpu()
    return cached, sample


def bootstrap_summary(rows):
    result = ORIGINAL_BOOTSTRAP(rows)
    names = sorted({str(row["name"]) for row in rows})
    operators = tuple(A1R.SOURCE.V3W.OPERATORS)
    keyed = {(str(r["name"]), str(r["operator"]), str(r["cell"])): r for r in rows}
    arrays = {
        (cell, op): np.asarray([float(keyed[(name, op, cell)]["gain_vs_shrink_db"]) for name in names])
        for cell in (PRIMARY, LOCAL_CONTROL) for op in operators
    }
    rng = np.random.Generator(np.random.PCG64(3407))
    draws = np.empty(4000)
    for index in range(4000):
        selected = rng.integers(0, len(names), size=len(names), endpoint=False)
        draws[index] = min(float(np.mean(arrays[(PRIMARY, op)][selected] - arrays[(LOCAL_CONTROL, op)][selected])) for op in operators)
    ordered = np.sort(draws)
    result["a1x_global_minus_local_db"] = float(draws.mean())
    result["a1x_global_minus_local_db_lcb95"] = float(ordered[199])
    result["a1x_global_minus_local_db_ucb95"] = float(ordered[3799])
    return result


def run_d0(*args, **kwargs):
    closeout = ORIGINAL_RUN(*args, **kwargs)
    output = Path(kwargs.get("output_dir", args[6] if len(args) > 6 else ""))
    bootstrap_path = output / "v4a_a1r_bootstrap_summary.json"
    bootstrap = json.loads(bootstrap_path.read_text())
    primary = bootstrap["cell_results"][PRIMARY]
    passed = bool(
        closeout["structural_valid"]
        and float(primary["worst_operator_gain_vs_shrink_db_lcb95"]) >= 0.020
        and float(bootstrap["primary_oracle_retention_lcb95"]) >= 0.25
        and float(bootstrap["primary_true_minus_shuffle_db_lcb95"]) >= 0.005
        and float(bootstrap["a1x_global_minus_local_db_lcb95"]) > 0.0
    )
    closeout.update({
        "schema_version": 1, "route_id": ROUTE_ID, "stage": "d0",
        "state": "COMPLETED_GATE_PASS" if passed else "COMPLETED_GATE_FAIL",
        "decision": "A1X_V3_D0_PASS_AUTHORIZE_FORMAL_DESIGN_ONLY" if passed else "A1X_V3_D0_GLOBAL_HEAD_CONTRACT_FAIL_STOP",
        "authorizes": "A1X_V3_FORMAL_DESIGN_ONLY" if passed else "NONE",
        "evidence_role": "development_screening", "formal_pass": passed,
        "confirmation_images_targets_outcomes_touched": False,
        "canary_touched": False, "locked_test_touched": False,
        "runner_sha256": os.environ["RUNNER_SHA256"],
        "bootstrap": bootstrap,
    })
    summary = {
        "schema_version": 1, "route_id": ROUTE_ID, "stage": "d0",
        "name_count": 512, "cells": CELLS, "bootstrap": bootstrap,
        "structural_valid": closeout["structural_valid"],
        "decision": closeout["decision"],
        "confirmation_images_targets_outcomes_touched": False,
    }
    summary_path = output / "a1x_v3_d0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    closeout["summary_sha256"] = sha256(summary_path)
    Path(os.environ["CLOSEOUT_PATH"]).write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n")
    print(json.dumps(closeout, sort_keys=True), flush=True)
    return closeout


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--a1r-root", required=True)
    parser.add_argument("--expected-a1r-commit", required=True)
    own, remainder = parser.parse_known_args(sys.argv[2:])
    root = Path(own.a1r_root).resolve()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip()
    if head != own.expected_a1r_commit or dirty:
        raise RuntimeError("A1R source identity mismatch")
    global A1R, ORIGINAL_PROBE, ORIGINAL_TARGET_FEATURES, ORIGINAL_BOOTSTRAP, ORIGINAL_RUN
    A1R = load_module(root / "experience_docx/tools/chd_rm_v4a_a1r_representation_sufficiency.py", "a1x_v3_d0_a1r")
    ORIGINAL_PROBE, ORIGINAL_TARGET_FEATURES = A1R.DeltaEndpointProbe, A1R.target_and_features
    ORIGINAL_BOOTSTRAP, ORIGINAL_RUN = A1R.bootstrap_summary, A1R.run_a1r
    A1R.CELL_SPECS = CELLS
    A1R.PRIMARY_CELL, A1R.SHUFFLED_CELL = PRIMARY, SHUFFLED
    A1R.DeltaEndpointProbe = D0Probe
    A1R.item_features = item_features
    A1R.endpoint_loss = endpoint_loss
    A1R.target_and_features = target_and_features
    A1R.bootstrap_summary = bootstrap_summary
    A1R.run_a1r = run_d0
    A1R.REPAIRABLE_THRESHOLD = 0.0
    A1R.audit(remainder)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "audit":
        raise SystemExit("usage: chd_rm_v4a_a1x_v3_d0.py audit ...")
    main()
