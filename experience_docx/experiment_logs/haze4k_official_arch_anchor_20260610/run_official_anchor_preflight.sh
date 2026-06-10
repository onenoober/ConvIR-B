#!/usr/bin/env bash
set -euo pipefail
REMOTE_ROOT=${REMOTE_ROOT:-/root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor}
ITS="$REMOTE_ROOT/Dehazing/ITS"
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_official_arch_anchor_20260610"
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
CKPT=${CKPT:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG="$EVID/official_anchor_preflight.log"
JSON_OUT="$EVID/official_anchor_preflight.json"
STATUS="$EVID/status.txt"
export REMOTE_ROOT DATA CKPT JSON_OUT
mkdir -p "$EVID"
{
  echo "preflight_start haze4k_official_arch_anchor $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "data=$DATA"
  echo "checkpoint=$CKPT"
} | tee -a "$STATUS"
cd "$ITS"
set +e
PYTHONUNBUFFERED=1 "$PY" - <<'PY' 2>&1 | tee "$LOG"
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

repo = Path(os.environ.get("REMOTE_ROOT", "/root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor"))
its = repo / "Dehazing" / "ITS"
sys.path.insert(0, str(its))

from data import train_dataloader
from main import load_init_model
from models.ConvIR import build_net

ckpt = Path(os.environ.get("CKPT", "/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl"))
data_dir = Path(os.environ.get("DATA", "/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K"))
out_path = Path(os.environ["JSON_OUT"])

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

class Args(argparse.Namespace):
    pass

result = {
    "locked_test_touched": False,
    "checkpoint": str(ckpt),
    "checkpoint_sha256": sha256_file(ckpt),
    "data_dir": str(data_dir),
    "checks": {},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(3407)
model = build_net("base", "Haze4K", "original").to(device)
result["parameter_count"] = sum(p.numel() for p in model.parameters())
state_keys = list(model.state_dict())
forbidden_prefixes = ("APDR_", "DPGA_", "PFD_")
forbidden_substrings = ("modulator",)
forbidden_keys = [
    key for key in state_keys
    if key.startswith(forbidden_prefixes) or any(part in key for part in forbidden_substrings)
]
result["checks"]["official_state_clean"] = not forbidden_keys
result["forbidden_state_keys"] = forbidden_keys[:20]

args = Args(init_model=str(ckpt), resume="")
load_init_model(model, args)
result["checks"]["strict_init_model_load"] = True
result["init_model_loaded"] = True

model.eval()
with torch.no_grad():
    x = torch.rand(1, 3, 256, 256, device=device)
    outputs = model(x)
result["synthetic_output_shapes"] = [list(t.shape) for t in outputs]
result["checks"]["synthetic_forward_finite"] = all(torch.isfinite(t).all().item() for t in outputs)
result["checks"]["synthetic_three_scales"] = result["synthetic_output_shapes"] == [
    [1, 3, 64, 64],
    [1, 3, 128, 128],
    [1, 3, 256, 256],
]

loader = train_dataloader(str(data_dir), batch_size=1, num_workers=0, data="Haze4K", use_transform=True)
input_img, label_img = next(iter(loader))
input_img = input_img.to(device)
label_img = label_img.to(device)
with torch.no_grad():
    pred = model(input_img)
    label2 = F.interpolate(label_img, scale_factor=0.5, mode="bilinear")
    label4 = F.interpolate(label_img, scale_factor=0.25, mode="bilinear")
    loss = F.l1_loss(pred[0], label4) + F.l1_loss(pred[1], label2) + F.l1_loss(pred[2], label_img)
result["train_batch_shapes"] = {
    "input": list(input_img.shape),
    "label": list(label_img.shape),
    "outputs": [list(t.shape) for t in pred],
}
result["train_batch_l1_multiscale"] = float(loss.item())
result["checks"]["train_batch_forward_finite"] = torch.isfinite(loss).item()

accepted_learning_rate = False
accepted_leaning_rate = False
# Validate parser compatibility by constructing the same parser path through main.py --help text.
help_text = os.popen(f"{sys.executable} main.py --help").read()
accepted_learning_rate = "--learning_rate" in help_text
accepted_leaning_rate = "--leaning_rate" in help_text
result["checks"]["learning_rate_aliases_present"] = accepted_learning_rate and accepted_leaning_rate

result["pass"] = all(result["checks"].values())
out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True), flush=True)
if not result["pass"]:
    raise SystemExit(1)
PY
rc=${PIPESTATUS[0]}
set -e
echo "preflight_done rc=$rc haze4k_official_arch_anchor $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "OFFICIAL_ANCHOR_PREFLIGHT_OK" | tee -a "$STATUS"
else
  echo "OFFICIAL_ANCHOR_PREFLIGHT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
