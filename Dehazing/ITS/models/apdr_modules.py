import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(value):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class RGBHazePriorExtractor(nn.Module):
    def forward(self, x):
        max_rgb = x.max(dim=1, keepdim=True).values
        min_rgb = x.min(dim=1, keepdim=True).values
        saturation = max_rgb - min_rgb
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        grad_x = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        grad_y = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        gradient = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)
        local_mean = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        local_contrast = (gray - local_mean).abs()

        return torch.cat(
            [min_rgb, max_rgb, saturation, gray, gradient, local_contrast],
            dim=1,
        )


class APDRScaleAdapter(nn.Module):
    def __init__(
        self,
        feature_channels,
        hidden_channels=24,
        residual_max=0.04,
        gate_max=0.5,
        gate_init=0.02,
    ):
        super(APDRScaleAdapter, self).__init__()
        if residual_max <= 0:
            raise ValueError("residual_max must be positive.")
        if gate_max <= 0:
            raise ValueError("gate_max must be positive.")
        if not (0.0 < gate_init < gate_max):
            raise ValueError("gate_init must be in (0, gate_max).")

        self.residual_max = float(residual_max)
        self.gate_max = float(gate_max)
        self.gate_init = float(gate_init)
        self.priors = RGBHazePriorExtractor()

        self.image_context = nn.Sequential(
            nn.Conv2d(3 + 3 + 6, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.feature_context = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.residual_head = nn.Conv2d(hidden_channels, 3, kernel_size=3, stride=1, padding=1)
        self.gate_head = nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=1)

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, _logit(self.gate_init / self.gate_max))

    def forward(self, hazy, anchor, feature, force_zero_gate=False):
        priors = self.priors(hazy)
        image_context = self.image_context(torch.cat([hazy, anchor.detach(), priors], dim=1))
        feature_context = self.feature_context(feature)
        context = self.context(torch.cat([image_context, feature_context], dim=1))
        residual_raw = self.residual_max * torch.tanh(self.residual_head(context))
        gate = self.gate_max * torch.sigmoid(self.gate_head(context))
        if force_zero_gate:
            gate = torch.zeros_like(gate)
        residual = gate * residual_raw
        output = anchor + residual
        return output, {
            "gate": gate,
            "residual": residual,
            "residual_raw": residual_raw,
            "anchor": anchor,
            "output": output,
            "prior": priors,
            "gate_max": self.gate_max,
            "residual_max": self.residual_max,
        }
