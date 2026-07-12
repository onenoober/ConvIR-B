"""Bounded zero-init direction repair for a frozen residual operator.

The module deliberately owns only the new ``DIRR_*`` parameters.  The caller
supplies the frozen base prediction and frozen correction step, so keeping this
module at its zero initialization reproduces the previous operator exactly.
"""

import torch
import torch.nn as nn


class DIRR_DeltaU(nn.Module):
    """Predict a bounded regional correction to a frozen output step.

    ``support`` is the old hard output support.  Applying it after the bounded
    head prevents the repair branch from creating a new intervention region.
    """

    def __init__(self, channels=24, delta_bound=(1.0, 1.0, 1.0)):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if len(delta_bound) != 3 or any(float(value) <= 0.0 for value in delta_bound):
            raise ValueError("delta_bound must contain three positive channel bounds")

        self.DIRR_stem = nn.Conv2d(9, channels, kernel_size=3, padding=1)
        self.DIRR_depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.DIRR_pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.DIRR_head = nn.Conv2d(channels, 3, kernel_size=1)
        self.DIRR_activation = nn.GELU()
        self.register_buffer(
            "DIRR_delta_bound",
            torch.as_tensor(delta_bound, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.DIRR_stem, self.DIRR_depthwise, self.DIRR_pointwise):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.DIRR_head.weight)
        nn.init.zeros_(self.DIRR_head.bias)

    def forward(self, hazy, base_prediction, frozen_step, support):
        if hazy.shape != base_prediction.shape or hazy.shape != frozen_step.shape:
            raise ValueError("hazy, base_prediction, and frozen_step must share shape")
        if support.shape[0] != hazy.shape[0] or support.shape[-2:] != hazy.shape[-2:]:
            raise ValueError("support must share batch and spatial dimensions with the inputs")
        if support.shape[1] != 1:
            raise ValueError("support must have one channel")

        features = torch.cat((hazy, base_prediction, frozen_step), dim=1)
        features = self.DIRR_activation(self.DIRR_stem(features))
        features = self.DIRR_activation(self.DIRR_depthwise(features))
        features = self.DIRR_activation(self.DIRR_pointwise(features))
        raw_delta = self.DIRR_head(features)
        bounded_delta = self.DIRR_delta_bound.to(dtype=raw_delta.dtype) * torch.tanh(raw_delta)
        return support.to(dtype=raw_delta.dtype) * bounded_delta

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
