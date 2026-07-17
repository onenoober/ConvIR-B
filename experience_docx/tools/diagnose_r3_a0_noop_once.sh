#!/usr/bin/env bash
set -euo pipefail

PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
OUT=/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_r3_proposal_first_acv_20260717/r3-a0-proposal-r3/workload
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K

"$PY" - "$OUT" "$DATA" <<'PY'
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

out = Path(sys.argv[1])
data = Path(sys.argv[2])
row = json.loads((out / "a0_cache_units_cloud_only.jsonl").open(encoding="utf-8").readline())
payload = torch.load(out / "candidate_cache_cloud_only" / f"{row['unit_key']}.pt", map_location="cpu")
base = payload["base"].float()
step = payload["step"].float()
candidates = payload["candidates"].float()
name = row["name"]
stem, extension = os.path.splitext(name)
label = None
for candidate in (name, f"{stem.split('_')[0]}{extension}", f"{stem.split('_')[0]}.png"):
    path = data / "train/gt" / candidate
    if path.is_file():
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB")).copy()
        label = torch.from_numpy(array.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0)
        break
assert label is not None
if label.shape[-2:] != base.shape[-2:]:
    label = label[:, :, :base.shape[-2], :base.shape[-1]]
old_low = torch.clamp(base + 0.125 * step, 0.0, 1.0)
old_high = torch.clamp(base + 0.25 * step, 0.0, 1.0)
noop_low = torch.clamp(base + 0.125 * (step + candidates[0:1]), 0.0, 1.0)
noop_high = torch.clamp(base + 0.25 * (step + candidates[0:1]), 0.0, 1.0)
old_low_mse = float((old_low - label).square().mean())
old_high_mse = float((old_high - label).square().mean())
noop_low_mse = float((noop_low - label).square().mean())
noop_high_mse = float((noop_high - label).square().mean())
tolerance = 2.0 * (1e-12 + 1e-12 * max(abs(value) for value in (
    old_low_mse, old_high_mse, noop_low_mse, noop_high_mse
)))
result = {
    "candidate_zero_name": payload["candidate_names"][0],
    "base_shape": list(base.shape),
    "step_shape": list(step.shape),
    "candidates_shape": list(candidates.shape),
    "base_finite": bool(torch.isfinite(base).all()),
    "step_finite": bool(torch.isfinite(step).all()),
    "candidate_zero_finite": bool(torch.isfinite(candidates[0:1]).all()),
    "candidate_zero_abs_max": float(candidates[0:1].abs().max()),
    "low_exact_equal": bool(torch.equal(old_low, noop_low)),
    "high_exact_equal": bool(torch.equal(old_high, noop_high)),
    "old_low_mse": old_low_mse,
    "noop_low_mse": noop_low_mse,
    "old_high_mse": old_high_mse,
    "noop_high_mse": noop_high_mse,
    "tolerance": tolerance,
    "safe_noop": noop_low_mse <= old_low_mse + tolerance and noop_high_mse <= old_high_mse + tolerance,
}
print(json.dumps(result, sort_keys=True))
PY

printf '%s\n' R3_A0_NOOP_DIAG_OK
