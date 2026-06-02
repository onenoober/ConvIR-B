import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualHardFeatureDelta(nn.Module):
    def __init__(self, channels):
        super(ResidualHardFeatureDelta, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
        )
        final = self.body[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x):
        return self.body(x)

    def stats(self, x):
        with torch.no_grad():
            delta = self.forward(x)
            return {
                "delta_abs_mean": delta.abs().mean().item(),
                "delta_abs_max": delta.abs().max().item(),
            }


class DecoderResidualHazeFeedback(nn.Module):
    def __init__(self, out_channels, hidden_channels=16):
        super(DecoderResidualHazeFeedback, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(6, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        final = self.body[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, restored, residual, target_size):
        x = torch.cat([restored, residual], dim=1)
        delta = self.body(x)
        if delta.shape[-2:] != target_size:
            delta = F.interpolate(delta, size=target_size, mode="bilinear", align_corners=False)
        return delta

    def stats(self, restored, residual, target_size):
        with torch.no_grad():
            delta = self.forward(restored, residual, target_size)
            return {
                "delta_abs_mean": delta.abs().mean().item(),
                "delta_abs_max": delta.abs().max().item(),
            }


class LowPassResidualDelta(nn.Module):
    def __init__(self, channels):
        super(LowPassResidualDelta, self).__init__()
        self.project = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return self.project(low)

    def stats(self, x):
        with torch.no_grad():
            low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            delta = self.project(low)
            return {
                "low_abs_mean": low.abs().mean().item(),
                "delta_abs_mean": delta.abs().mean().item(),
                "delta_abs_max": delta.abs().max().item(),
            }
