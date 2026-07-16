"""Frozen A1X deployable-input accessibility diagnostic head.

The official ConvIR builder is intentionally untouched.  This standalone head
accepts only the five preregistered three-channel exact-half tensors and starts
as an exact no-op through its zero-initialized final projection.
"""

import torch
from torch import nn


class A1X_ACCESS_DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride=stride, padding=1,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.GELU()

    def forward(self, value):
        value = self.depthwise(value)
        value = self.pointwise(value)
        return self.activation(self.norm(value))


class A1X_ACCESS_Head(nn.Module):
    """Fixed 24/48/96 global nonlinear head on the exact-half lattice."""

    input_channels = 15
    output_channels = 3

    def __init__(self, input_channels=15):
        super().__init__()
        self.input_channels = input_channels
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, 24, 1, bias=False),
            nn.GroupNorm(1, 24),
            nn.GELU(),
        )
        self.encoder_24 = A1X_ACCESS_DepthwiseSeparable(24, 24)
        self.encoder_48 = A1X_ACCESS_DepthwiseSeparable(24, 48, stride=2)
        self.encoder_96 = A1X_ACCESS_DepthwiseSeparable(48, 96, stride=2)
        self.global_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(96, 24, 1, bias=True),
            nn.GELU(),
        )
        self.decoder_48 = A1X_ACCESS_DepthwiseSeparable(96 + 48, 48)
        self.decoder_24 = A1X_ACCESS_DepthwiseSeparable(48 + 24 + 24, 24)
        self.final_projection = nn.Conv2d(24, self.output_channels, 1, bias=True)
        nn.init.zeros_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)

    def forward(self, deployable_tensors):
        if deployable_tensors.ndim != 4 or deployable_tensors.shape[1] != self.input_channels:
            raise ValueError("A1X_ACCESS_Head requires a 15-channel exact-half tensor")
        x24 = self.encoder_24(self.stem(deployable_tensors))
        x48 = self.encoder_48(x24)
        x96 = self.encoder_96(x48)
        context = self.global_projection(x96).expand(-1, -1, *x24.shape[-2:])
        up48 = torch.nn.functional.interpolate(
            x96, size=x48.shape[-2:], mode="bilinear", align_corners=False,
        )
        d48 = self.decoder_48(torch.cat((up48, x48), dim=1))
        up24 = torch.nn.functional.interpolate(
            d48, size=x24.shape[-2:], mode="bilinear", align_corners=False,
        )
        d24 = self.decoder_24(torch.cat((up24, x24, context), dim=1))
        return torch.tanh(self.final_projection(d24))
