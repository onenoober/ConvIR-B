"""Zero-init Delta-u heads for the v3t zero-lock diagnostic."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DIRT_BoundedHead(nn.Module):
    def __init__(self, input_channels, channels, delta_bound):
        super().__init__()
        if input_channels <= 0 or channels <= 0:
            raise ValueError("input_channels and channels must be positive")
        if len(delta_bound) != 3 or any(float(value) <= 0.0 for value in delta_bound):
            raise ValueError("delta_bound must contain three positive values")
        self.DIRT_stem = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1)
        self.DIRT_depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.DIRT_pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.DIRT_head = nn.Conv2d(channels, 3, kernel_size=1)
        self.DIRT_activation = nn.GELU()
        self.register_buffer(
            "DIRT_delta_bound",
            torch.as_tensor(delta_bound, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.DIRT_stem, self.DIRT_depthwise, self.DIRT_pointwise):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.DIRT_head.weight)
        nn.init.zeros_(self.DIRT_head.bias)

    def _bounded(self, features):
        features = self.DIRT_activation(self.DIRT_stem(features))
        features = self.DIRT_activation(self.DIRT_depthwise(features))
        features = self.DIRT_activation(self.DIRT_pointwise(features))
        raw = self.DIRT_head(features)
        return self.DIRT_delta_bound.to(dtype=raw.dtype) * torch.tanh(raw)


class DIRT_OutputDeltaU(_DIRT_BoundedHead):
    """The v3s output-side input form, retained as a factorial control."""

    def __init__(self, channels=24, delta_bound=(1.0, 1.0, 1.0)):
        super().__init__(input_channels=9, channels=channels, delta_bound=delta_bound)

    def forward(self, hazy, base_prediction, old_step, support):
        if hazy.shape != base_prediction.shape or hazy.shape != old_step.shape:
            raise ValueError("full-resolution inputs must share shape")
        features = torch.cat((hazy, base_prediction, old_step), dim=1)
        return support.to(dtype=features.dtype) * self._bounded(features)


class DIRT_ContextDeltaU(_DIRT_BoundedHead):
    """Use frozen v3l full-context features plus downsampled output inputs."""

    def __init__(self, context_channels, channels=24, delta_bound=(1.0, 1.0, 1.0)):
        super().__init__(input_channels=context_channels + 9, channels=channels, delta_bound=delta_bound)
        self.DIRT_context_channels = int(context_channels)

    def forward(self, context, hazy, base_prediction, old_step, support):
        if context.shape[1] != self.DIRT_context_channels:
            raise ValueError("unexpected frozen context channel count")
        target_size = context.shape[-2:]
        low_inputs = [
            F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            for value in (hazy, base_prediction, old_step)
        ]
        low_delta = self._bounded(torch.cat((context, *low_inputs), dim=1))
        full_delta = F.interpolate(low_delta, size=base_prediction.shape[-2:], mode="bilinear", align_corners=False)
        return support.to(dtype=full_delta.dtype) * full_delta
