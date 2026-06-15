from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvIR import ConvIR
from .layers import BasicConv


FEATURE_MODES = {
    "rgb": 9,
    "rgb_wavelet": 21,
}


def _load_checkpoint_state(checkpoint_path: str | Path) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"expected checkpoint dict, got {type(state)!r}")
    return state


def load_haze4k_partial_checkpoint(module: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Strictly load the official Haze4K ConvIR-B checkpoint into `module`."""

    state = _load_checkpoint_state(checkpoint_path)
    module_state = module.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatch: list[dict[str, Any]] = []

    for key, value in state.items():
        if key not in module_state:
            unexpected.append(key)
            continue
        if tuple(module_state[key].shape) != tuple(value.shape):
            shape_mismatch.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "module_shape": list(module_state[key].shape),
                }
            )
            continue
        loaded[key] = value

    missing = [key for key in module_state if key not in loaded]
    if unexpected or shape_mismatch or missing:
        raise RuntimeError(
            "partial-load failed: "
            f"unexpected={unexpected}, "
            f"shape_mismatch={shape_mismatch}, "
            f"missing={missing}"
        )

    module_state.update(loaded)
    module.load_state_dict(module_state, strict=True)
    return {
        "loaded_keys": sorted(loaded),
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "missing": missing,
        "checkpoint_path": str(checkpoint_path),
    }


def _pad_even(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x


def _haar_dwt2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _pad_even(x)
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def _build_features(hazy: torch.Tensor, a0: torch.Tensor, feature_mode: str) -> torch.Tensor:
    diff = hazy - a0
    if feature_mode == "rgb":
        return torch.cat([hazy, a0, diff], dim=1)
    if feature_mode == "rgb_wavelet":
        comps = [
            F.interpolate(comp, size=diff.shape[-2:], mode="bilinear", align_corners=False)
            for comp in _haar_dwt2(diff)
        ]
        return torch.cat([hazy, a0, diff, *comps], dim=1)
    raise ValueError(f"unsupported feature_mode={feature_mode!r}")


class C13ResidualAdapter(nn.Module):
    def __init__(self, in_channels: int, width: int = 32, depth: int = 3, head_init: str = "kaiming") -> None:
        super().__init__()
        if head_init not in {"kaiming", "zero"}:
            raise ValueError(f"unsupported head_init={head_init!r}")
        blocks: list[nn.Module] = [BasicConv(in_channels, width, kernel_size=3, stride=1, relu=True)]
        for _ in range(max(0, depth - 1)):
            blocks.append(BasicConv(width, width, kernel_size=3, stride=1, relu=True))
        self.C13_body = nn.Sequential(*blocks)
        self.C13_head = nn.Conv2d(width, 3, kernel_size=3, stride=1, padding=1, bias=True)
        if head_init == "kaiming":
            nn.init.kaiming_normal_(self.C13_head.weight, mode="fan_out", nonlinearity="linear")
        else:
            nn.init.zeros_(self.C13_head.weight)
        nn.init.zeros_(self.C13_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.C13_head(self.C13_body(x))

    def body_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.C13_body(x)

    def head_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.C13_head(x)


class C13A0FrozenResidualConvIR(nn.Module):
    def __init__(
        self,
        version: str,
        data: str,
        a0_checkpoint: str | Path,
        feature_mode: str = "rgb_wavelet",
        adapter_width: int = 32,
        adapter_depth: int = 3,
        bootstrap_scale: float = 0.01,
        residual_mode: str = "gated_bootstrap",
        residual_scale: float = 1.0,
        scale_init: float = 0.25,
        head_init: str = "kaiming",
        clamp_output: bool = False,
    ) -> None:
        super().__init__()
        if feature_mode not in FEATURE_MODES:
            raise ValueError(f"unsupported feature_mode={feature_mode!r}")
        if residual_mode not in {"gated_bootstrap", "direct", "adaptive_scalar"}:
            raise ValueError(f"unsupported residual_mode={residual_mode!r}")
        self.feature_mode = feature_mode
        self.clamp_output = clamp_output
        self.residual_mode = residual_mode
        self.a0 = ConvIR(version, data)
        self.C13_adapter = C13ResidualAdapter(
            FEATURE_MODES[feature_mode],
            width=adapter_width,
            depth=adapter_depth,
            head_init=head_init,
        )
        self.C13_gate = nn.Parameter(torch.zeros(3, 1, 1))
        self.register_buffer("C13_bootstrap_scale", torch.tensor(float(bootstrap_scale)))
        self.register_buffer("C13_residual_scale", torch.tensor(float(residual_scale)))
        self.register_buffer("C13_scale_init", torch.tensor(float(scale_init)))
        if residual_mode == "adaptive_scalar":
            self.C13_scale_head = nn.Conv2d(adapter_width, 1, kernel_size=1, stride=1, padding=0, bias=True)
            nn.init.zeros_(self.C13_scale_head.weight)
            clipped_scale = max(1e-4, min(1 - 1e-4, float(scale_init)))
            init_bias = math.log(clipped_scale / (1.0 - clipped_scale))
            nn.init.constant_(self.C13_scale_head.bias, init_bias)
        self.load_a0_checkpoint(a0_checkpoint)
        self.freeze_a0()

    def freeze_a0(self) -> None:
        self.a0.eval()
        for param in self.a0.parameters():
            param.requires_grad_(False)

    def load_a0_checkpoint(self, a0_checkpoint: str | Path) -> dict[str, Any]:
        report = load_haze4k_partial_checkpoint(self.a0, a0_checkpoint)
        return report

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.freeze_a0()
        return self

    def build_features(self, hazy: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
        return _build_features(hazy, a0, self.feature_mode)

    def route_forward(self, hazy: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        with torch.no_grad():
            a0_outputs = self.a0(hazy)
            a0_pred = a0_outputs[2] if isinstance(a0_outputs, (list, tuple)) else a0_outputs
        features = self.build_features(hazy, a0_pred)
        body = self.C13_adapter.body_forward(features)
        raw = self.C13_adapter.head_forward(body)
        gate = torch.tanh(self.C13_gate)
        if self.residual_mode == "gated_bootstrap":
            residual = gate * raw + self.C13_bootstrap_scale * (raw - raw.detach())
            scale = gate
        elif self.residual_mode == "adaptive_scalar":
            scale_logits = self.C13_scale_head(F.adaptive_avg_pool2d(body, 1))
            scale = torch.sigmoid(scale_logits)
            residual = self.C13_residual_scale * scale * raw
        else:
            scale = self.C13_residual_scale.view(1, 1, 1, 1).expand(raw.shape[0], 1, 1, 1)
            residual = self.C13_residual_scale * raw
        pred = a0_pred + residual
        if self.clamp_output:
            pred = torch.clamp(pred, 0.0, 1.0)
        pred_2 = F.interpolate(
            pred,
            size=(max(1, pred.shape[-2] // 2), max(1, pred.shape[-1] // 2)),
            mode="bilinear",
            align_corners=False,
        )
        pred_4 = F.interpolate(
            pred,
            size=(max(1, pred.shape[-2] // 4), max(1, pred.shape[-1] // 4)),
            mode="bilinear",
            align_corners=False,
        )
        return {
            "outputs": [pred_4, pred_2, pred],
            "a0": a0_pred,
            "features": features,
            "body": body,
            "raw_residual": raw,
            "residual": residual,
            "scale": scale,
            "gate": gate,
        }

    def forward(self, hazy: torch.Tensor):  # type: ignore[override]
        return self.route_forward(hazy)["outputs"]

    def collect_route_stats(self, hazy: torch.Tensor) -> dict[str, float]:
        aux = self.route_forward(hazy)
        a0 = aux["a0"]
        pred = aux["outputs"][-1]
        residual = aux["residual"]
        raw = aux["raw_residual"]
        gate = aux["gate"]
        scale = aux["scale"]
        assert isinstance(a0, torch.Tensor)
        assert isinstance(pred, torch.Tensor)
        assert isinstance(residual, torch.Tensor)
        assert isinstance(raw, torch.Tensor)
        assert isinstance(gate, torch.Tensor)
        assert isinstance(scale, torch.Tensor)
        return {
            "gate_mean": float(gate.mean().item()),
            "gate_abs_max": float(gate.abs().max().item()),
            "residual_scale": float(self.C13_residual_scale.item()),
            "scale_mean": float(scale.mean().item()),
            "scale_abs_max": float(scale.abs().max().item()),
            "raw_residual_mean_abs": float(raw.abs().mean().item()),
            "residual_mean_abs": float(residual.abs().mean().item()),
            "student_a0_mean_abs": float((pred - a0).abs().mean().item()),
        }


def build_net(
    version: str,
    data: str,
    a0_checkpoint: str | Path,
    feature_mode: str = "rgb_wavelet",
    adapter_width: int = 32,
    adapter_depth: int = 3,
    bootstrap_scale: float = 0.01,
    residual_mode: str = "gated_bootstrap",
    residual_scale: float = 1.0,
    scale_init: float = 0.25,
    head_init: str = "kaiming",
    clamp_output: bool = False,
) -> C13A0FrozenResidualConvIR:
    return C13A0FrozenResidualConvIR(
        version=version,
        data=data,
        a0_checkpoint=a0_checkpoint,
        feature_mode=feature_mode,
        adapter_width=adapter_width,
        adapter_depth=adapter_depth,
        bootstrap_scale=bootstrap_scale,
        residual_mode=residual_mode,
        residual_scale=residual_scale,
        scale_init=scale_init,
        head_init=head_init,
        clamp_output=clamp_output,
    )
