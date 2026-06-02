import math
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


class SelectiveSafeResidualHardFeatureDelta(nn.Module):
    def __init__(
        self,
        channels,
        gate_max=1.0,
        gate_init=0.10,
        norm_cap=0.0035,
        lowpass_ratio=0.20,
        use_haze_prior=True,
    ):
        super(SelectiveSafeResidualHardFeatureDelta, self).__init__()
        if not (0.0 < gate_init < gate_max):
            raise ValueError("gate_init must be in (0, gate_max).")
        if norm_cap <= 0:
            raise ValueError("norm_cap must be positive.")
        if not (0.0 <= lowpass_ratio <= 1.0):
            raise ValueError("lowpass_ratio must be in [0, 1].")

        hidden = max(channels // 4, 16)
        self.channels = channels
        self.gate_max = float(gate_max)
        self.norm_cap = float(norm_cap)
        self.lowpass_ratio = float(lowpass_ratio)
        self.use_haze_prior = bool(use_haze_prior)

        self.delta_body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
        )

        gate_in_channels = channels + (1 if self.use_haze_prior else 0)
        self.gate_body = nn.Sequential(
            nn.Conv2d(gate_in_channels, hidden, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, stride=1, padding=1),
        )

        final_delta = self.delta_body[-1]
        nn.init.zeros_(final_delta.weight)
        nn.init.zeros_(final_delta.bias)

        final_gate = self.gate_body[-1]
        nn.init.zeros_(final_gate.weight)
        init_prob = gate_init / self.gate_max
        init_logit = math.log(init_prob / (1.0 - init_prob))
        nn.init.constant_(final_gate.bias, init_logit)

    def _remove_drift(self, delta):
        delta = delta - delta.mean(dim=(2, 3), keepdim=True)
        if self.lowpass_ratio < 1.0:
            low = F.avg_pool2d(delta, kernel_size=5, stride=1, padding=2)
            high = delta - low
            delta = high + self.lowpass_ratio * low
        return delta

    def _cap_norm(self, feature, delta):
        batch = delta.shape[0]
        delta_norm = delta.reshape(batch, -1).norm(dim=1).view(batch, 1, 1, 1)
        feature_norm = feature.reshape(batch, -1).norm(dim=1).view(batch, 1, 1, 1)
        max_delta_norm = self.norm_cap * feature_norm.clamp_min(1e-12)
        scale = torch.clamp(max_delta_norm / delta_norm.clamp_min(1e-12), max=1.0)
        return delta * scale

    def _gate_input(self, feature, haze_prior):
        if not self.use_haze_prior:
            return feature
        if haze_prior is None:
            haze_prior = torch.zeros(
                feature.shape[0],
                1,
                feature.shape[2],
                feature.shape[3],
                device=feature.device,
                dtype=feature.dtype,
            )
        return torch.cat([feature, haze_prior.detach()], dim=1)

    def forward(self, feature, haze_prior=None):
        delta = self.delta_body(feature)
        delta = self._remove_drift(delta)
        gate = self.gate_max * torch.sigmoid(self.gate_body(self._gate_input(feature, haze_prior)))
        delta = gate * delta
        return self._cap_norm(feature, delta)

    def stats(self, feature, haze_prior=None):
        with torch.no_grad():
            delta = self.forward(feature, haze_prior)
            gate = self.gate_max * torch.sigmoid(self.gate_body(self._gate_input(feature, haze_prior)))
            feature_norm = feature.norm().item()
            delta_norm = delta.norm().item()
            return {
                "delta_abs_mean": delta.abs().mean().item(),
                "delta_abs_max": delta.abs().max().item(),
                "delta_std": delta.std(unbiased=False).item(),
                "delta_norm": delta_norm,
                "feature_norm": feature_norm,
                "delta_norm_ratio": delta_norm / max(feature_norm, 1e-12),
                "gate_mean": gate.mean().item(),
                "gate_max": gate.max().item(),
                "gate_min": gate.min().item(),
                "gate_std": gate.std(unbiased=False).item(),
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
