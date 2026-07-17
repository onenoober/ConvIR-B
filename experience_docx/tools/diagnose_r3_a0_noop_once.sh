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
rows = []
with (out / "a0_cache_units_cloud_only.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        rows.append(json.loads(line))
        if len(rows) == 16:
            break
max_low_delta = 0.0
max_high_delta = 0.0
first_unsafe = None
all_exact = True
for index, row in enumerate(rows, 1):
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
    low = torch.clamp(base + 0.125 * (step + candidates), 0.0, 1.0)
    high = torch.clamp(base + 0.25 * (step + candidates), 0.0, 1.0)
    old_low_mse = float((old_low - label).square().mean())
    old_high_mse = float((old_high - label).square().mean())
    low_mse = (low - label).square().mean(dim=(1, 2, 3))
    high_mse = (high - label).square().mean(dim=(1, 2, 3))
    noop_low_mse = float(low_mse[0])
    noop_high_mse = float(high_mse[0])
    tolerance = 2.0 * (1e-12 + 1e-12 * max(abs(value) for value in (
        old_low_mse, old_high_mse, noop_low_mse, noop_high_mse
    )))
    safe = noop_low_mse <= old_low_mse + tolerance and noop_high_mse <= old_high_mse + tolerance
    all_exact = all_exact and bool(torch.equal(old_low, low[0:1])) and bool(torch.equal(old_high, high[0:1]))
    max_low_delta = max(max_low_delta, abs(noop_low_mse - old_low_mse))
    max_high_delta = max(max_high_delta, abs(noop_high_mse - old_high_mse))
    if not safe and first_unsafe is None:
        first_unsafe = index
result = {
    "units_checked": len(rows),
    "all_noop_tensors_exact_equal": all_exact,
    "max_low_mse_reduction_delta": max_low_delta,
    "max_high_mse_reduction_delta": max_high_delta,
    "first_unsafe_index": first_unsafe,
    "frozen_tolerance_floor": 2e-12,
}
print(json.dumps(result, sort_keys=True))
PY

printf '%s\n' R3_A0_NOOP_DIAG_OK
