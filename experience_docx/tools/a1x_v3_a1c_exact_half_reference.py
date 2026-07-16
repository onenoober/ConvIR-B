"""Pinned A1C exact-half endpoint reference for A1X-v3 S0.

Extracted from A1C commit 9c4bc79cfdadb00aa91ac6c6baed58fdbc6be068;
full upstream source SHA-256: 0b947a36a83178aaa5d8316273a24de835c52af437332b3b2607c74ffe9cac12.
"""
from typing import Any

import torch
import torch.nn.functional as F

UPSTREAM_COMMIT = "9c4bc79cfdadb00aa91ac6c6baed58fdbc6be068"
UPSTREAM_SOURCE_SHA256 = "0b947a36a83178aaa5d8316273a24de835c52af437332b3b2607c74ffe9cac12"
PARENT: Any = None


def endpoint(target, current, support, bound, cell):
    if cell == "full":
        return target, None
    height, width = current.shape[-2:]
    expected = {(400, 400): (208, 208), (480, 640): (240, 320)}
    if (height, width) not in expected:
        raise RuntimeError(f"unsealed A1R transport shape {(height, width)}")
    low_size = expected[(height, width)]
    if cell not in ("exact_half", "antialiased_half"):
        raise ValueError(cell)
    antialias = cell == "antialiased_half"
    low_target = F.interpolate(target, size=low_size, mode="bilinear", align_corners=False, antialias=antialias)
    low_current = F.interpolate(current, size=low_size, mode="bilinear", align_corners=False, antialias=antialias)
    low_difference = low_target - low_current
    if float(torch.clamp(low_difference.abs() - 2.0 * bound, min=0).max().item()) > 1e-6:
        raise RuntimeError("half correction exceeds the theoretical +/-2B range")
    replay = F.interpolate(low_difference, size=(height, width), mode="bilinear", align_corners=False)
    return support * PARENT.clamp_channelwise(current + support * replay, bound), low_size
