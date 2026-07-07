import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvIR import ConvIR


class V300A0AnchoredConvIR(ConvIR):
    """A0-anchored ConvIR route with a zero-init low-frequency residual branch."""

    def __init__(self, version, data, residual_scale=1.0):
        super().__init__(version, data)
        self.V300_pool = nn.AvgPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.V300_branch = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )
        self.register_buffer("V300_residual_scale", torch.tensor(float(residual_scale)))
        self._init_v300_branch()

    def _init_v300_branch(self):
        for module in self.V300_branch:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.V300_branch[-1].weight)
        nn.init.zeros_(self.V300_branch[-1].bias)

    def forward(self, x):
        outputs = super().forward(x)
        final = outputs[-1]
        low = self.V300_pool(torch.cat([x, final], dim=1))
        residual = self.V300_branch(low)
        residual = F.interpolate(residual, size=final.shape[-2:], mode="bilinear", align_corners=False)
        outputs[-1] = final + self.V300_residual_scale.to(final.dtype) * residual
        return outputs


def build_v300_a0_anchored_net(version, data, residual_scale=1.0):
    return V300A0AnchoredConvIR(version, data, residual_scale=residual_scale)
