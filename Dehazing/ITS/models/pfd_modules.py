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
            feature_norm = x.norm().item()
            delta_norm = delta.norm().item()
            return {
                "delta_abs_mean": delta.abs().mean().item(),
                "delta_abs_max": delta.abs().max().item(),
                "delta_std": delta.std(unbiased=False).item(),
                "delta_norm": delta_norm,
                "feature_norm": feature_norm,
                "delta_norm_ratio": delta_norm / max(feature_norm, 1e-12),
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
            feature_norm = x.norm().item()
            delta_norm = delta.norm().item()
            return {
                "low_abs_mean": low.abs().mean().item(),
                "delta_abs_mean": delta.abs().mean().item(),
                "delta_abs_max": delta.abs().max().item(),
                "delta_std": delta.std(unbiased=False).item(),
                "delta_norm": delta_norm,
                "feature_norm": feature_norm,
                "delta_norm_ratio": delta_norm / max(feature_norm, 1e-12),
            }
