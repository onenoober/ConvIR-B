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


class APDRV02ScaleAdapter(nn.Module):
    def __init__(
        self,
        feature_channels,
        hidden_channels=24,
        residual_max=0.04,
        gate_max=0.5,
        gate_init=0.01,
    ):
        super(APDRV02ScaleAdapter, self).__init__()
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
        self.spatial_gate_head = nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=1)
        self.global_gate_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0),
        )

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.spatial_gate_head.weight)
        nn.init.constant_(self.spatial_gate_head.bias, _logit(self.gate_init / self.gate_max))
        nn.init.zeros_(self.global_gate_head[-1].weight)
        nn.init.constant_(self.global_gate_head[-1].bias, _logit(self.gate_init / self.gate_max))

    def forward(self, hazy, anchor, feature, force_zero_gate=False):
        priors = self.priors(hazy)
        image_context = self.image_context(torch.cat([hazy, anchor.detach(), priors], dim=1))
        feature_context = self.feature_context(feature)
        context = self.context(torch.cat([image_context, feature_context], dim=1))

        residual_raw = self.residual_max * torch.tanh(self.residual_head(context))
        spatial_logits = self.spatial_gate_head(context)
        global_logits = self.global_gate_head(context)
        spatial_gate_unit = torch.sigmoid(spatial_logits)
        global_gate_unit = torch.sigmoid(global_logits)
        gate = self.gate_max * spatial_gate_unit * global_gate_unit
        if force_zero_gate:
            gate = torch.zeros_like(gate)
        residual = gate * residual_raw
        output = anchor + residual
        return output, {
            "gate": gate,
            "spatial_gate": self.gate_max * spatial_gate_unit,
            "spatial_gate_unit": spatial_gate_unit,
            "spatial_logits": spatial_logits,
            "global_gate": self.gate_max * global_gate_unit,
            "global_gate_unit": global_gate_unit,
            "global_logits": global_logits,
            "residual": residual,
            "residual_raw": residual_raw,
            "anchor": anchor,
            "output": output,
            "prior": priors,
            "gate_max": self.gate_max,
            "residual_max": self.residual_max,
        }


class GlobalDensityRouter(nn.Module):
    def __init__(self, feature_channels, hidden_channels=24, gate_max=0.5, gate_init=0.01):
        super(GlobalDensityRouter, self).__init__()
        self.gate_max = float(gate_max)
        self.gate_init = float(gate_init)
        stat_channels = 3 + 3 + 6 + 3
        router_channels = max(hidden_channels * 2, 32)
        self.feature_pool = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Linear(feature_channels + stat_channels * 2, router_channels),
            nn.GELU(),
            nn.Linear(router_channels, router_channels),
            nn.GELU(),
            nn.Linear(router_channels, 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, _logit(self.gate_init / self.gate_max))

    def forward(self, hazy, anchor, feature, priors):
        anchor_detached = anchor.detach()
        residual_proxy = (hazy - anchor_detached).abs()
        stats_source = torch.cat([hazy, anchor_detached, priors, residual_proxy], dim=1)
        flat_stats = stats_source.flatten(2)
        means = flat_stats.mean(dim=2)
        stds = flat_stats.std(dim=2, unbiased=False)
        pooled_feature = self.feature_pool(feature).flatten(1)
        logits = self.router(torch.cat([pooled_feature, means, stds], dim=1))
        return logits.view(-1, 1, 1, 1)


class APDRV02RScaleAdapter(nn.Module):
    def __init__(
        self,
        feature_channels,
        hidden_channels=24,
        residual_max=0.04,
        gate_max=0.5,
        gate_init=0.01,
    ):
        super(APDRV02RScaleAdapter, self).__init__()
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
        self.spatial_gate_head = nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=1)
        self.global_router = GlobalDensityRouter(
            feature_channels,
            hidden_channels=hidden_channels,
            gate_max=gate_max,
            gate_init=gate_init,
        )
        self.register_buffer("global_budget_tau", torch.tensor(0.0))
        self.register_buffer("global_budget_temperature", torch.tensor(1.0))

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.spatial_gate_head.weight)
        nn.init.constant_(self.spatial_gate_head.bias, _logit(self.gate_init / self.gate_max))

    def set_global_budget_calibration(self, tau, temperature):
        self.global_budget_tau.fill_(float(tau))
        self.global_budget_temperature.fill_(max(float(temperature), 1e-4))

    def forward(self, hazy, anchor, feature, force_zero_gate=False):
        priors = self.priors(hazy)
        image_context = self.image_context(torch.cat([hazy, anchor.detach(), priors], dim=1))
        feature_context = self.feature_context(feature)
        context = self.context(torch.cat([image_context, feature_context], dim=1))

        residual_raw = self.residual_max * torch.tanh(self.residual_head(context))
        spatial_logits = self.spatial_gate_head(context)
        global_logits = self.global_router(hazy, anchor, feature, priors)
        spatial_gate_unit = torch.sigmoid(spatial_logits)
        global_score_unit = torch.sigmoid(global_logits)
        temperature = self.global_budget_temperature.clamp_min(1e-4)
        global_budget_unit = torch.sigmoid((global_logits - self.global_budget_tau) / temperature)
        gate = self.gate_max * spatial_gate_unit * global_budget_unit
        if force_zero_gate:
            gate = torch.zeros_like(gate)
        residual = gate * residual_raw
        output = anchor + residual
        return output, {
            "gate": gate,
            "spatial_gate": self.gate_max * spatial_gate_unit,
            "spatial_gate_unit": spatial_gate_unit,
            "spatial_logits": spatial_logits,
            "global_gate": self.gate_max * global_budget_unit,
            "global_gate_unit": global_budget_unit,
            "global_budget_unit": global_budget_unit,
            "global_score_unit": global_score_unit,
            "global_logits": global_logits,
            "residual": residual,
            "residual_raw": residual_raw,
            "anchor": anchor,
            "output": output,
            "prior": priors,
            "gate_max": self.gate_max,
            "residual_max": self.residual_max,
        }
