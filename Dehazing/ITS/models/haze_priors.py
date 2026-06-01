import torch
import torch.nn as nn
import torch.nn.functional as F


class HazePriorExtractor(nn.Module):
    def forward(self, x):
        max_rgb = x.max(dim=1, keepdim=True)[0]
        min_rgb = x.min(dim=1, keepdim=True)[0]
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        local_mean = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        local_contrast = (gray - local_mean).abs()
        return torch.cat([min_rgb, saturation, local_contrast], dim=1)


class HazePriorResidual(nn.Module):
    def __init__(self, out_channels, hidden_channels=8):
        super(HazePriorResidual, self).__init__()
        self.priors = HazePriorExtractor()
        self.confidence = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        final = self.project[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x):
        priors = self.priors(x)
        confidence = self.confidence(priors)
        return self.project(priors) * confidence

    def stats(self, x):
        with torch.no_grad():
            priors = self.priors(x)
            confidence = self.confidence(priors)
            delta = self.project(priors) * confidence
            return {
                "prior_mean": priors.mean().item(),
                "prior_std": priors.std(unbiased=False).item(),
                "confidence_mean": confidence.mean().item(),
                "confidence_std": confidence.std(unbiased=False).item(),
                "delta_abs_mean": delta.abs().mean().item(),
                "delta_abs_max": delta.abs().max().item(),
            }
