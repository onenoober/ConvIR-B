import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_size(x, size):
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _local_mean(x, kernel_size=9):
    pad = kernel_size // 2
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def _channel_norm(x):
    return torch.sqrt(torch.mean(x * x, dim=1, keepdim=True).clamp_min(1e-12))


def _gradient_magnitude(x):
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


class ContextEvidenceEncoder(nn.Module):
    """Builds allowed hazy/internal feature evidence at the final feature size."""

    def __init__(self, context_channels=32):
        super().__init__()
        in_channels = 13
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, context_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(context_channels, context_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, hazy, final_feature, res1, res2, scm2, scm4):
        size = final_feature.shape[-2:]
        hazy = _same_size(hazy, size)

        brightness = hazy.mean(dim=1, keepdim=True)
        dark = hazy.min(dim=1, keepdim=True).values
        saturation = hazy.max(dim=1, keepdim=True).values - dark
        gradient = _gradient_magnitude(brightness)
        local_brightness = _local_mean(brightness)
        local_variance = _local_mean((brightness - local_brightness) ** 2)
        low_haze_proxy = _local_mean(dark, kernel_size=15)

        final_norm = _channel_norm(final_feature)
        final_mean = final_feature.mean(dim=1, keepdim=True)
        final_low = _local_mean(final_mean)
        final_local_variance = _local_mean((final_mean - final_low) ** 2)
        final_detail_energy = _channel_norm(final_feature - _local_mean(final_feature))

        res1_norm = _channel_norm(_same_size(res1, size))
        res2_norm = _channel_norm(_same_size(res2, size))
        scm2_norm = _channel_norm(_same_size(scm2, size))
        scm4_norm = _channel_norm(_same_size(scm4, size))

        context = torch.cat(
            [
                brightness,
                dark,
                saturation,
                gradient,
                local_variance,
                low_haze_proxy,
                final_norm,
                final_local_variance,
                final_detail_energy,
                res1_norm,
                res2_norm,
                scm2_norm,
                scm4_norm,
            ],
            dim=1,
        )
        return self.encoder(context)


class PatchBandGate(nn.Module):
    def __init__(self, context_channels=32, gate_bias=-3.0, use_detail=False):
        super().__init__()
        self.use_detail = bool(use_detail)
        hidden = max(8, context_channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(context_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(gate_bias))

    def forward(self, context):
        raw = self.net(context)
        g_low = torch.sigmoid(raw[:, 0:1])
        if self.use_detail:
            g_detail = g_low * torch.sigmoid(raw[:, 1:2])
        else:
            g_detail = torch.zeros_like(g_low)
        return g_low, g_detail


class BandFeatureAction(nn.Module):
    def __init__(self, feature_channels=32, context_channels=32, hidden_channels=32, use_detail=False):
        super().__init__()
        self.use_detail = bool(use_detail)
        in_channels = feature_channels + context_channels
        self.low_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, feature_channels, kernel_size=3, padding=1),
        )
        self.detail_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, feature_channels, kernel_size=3, padding=1),
        )

    def forward(self, final_feature, context):
        feature_low = _local_mean(final_feature, kernel_size=9)
        feature_detail = final_feature - feature_low
        low_input = torch.cat([feature_low, context], dim=1)
        delta_low = _local_mean(self.low_head(low_input), kernel_size=9)

        if self.use_detail:
            detail_input = torch.cat([feature_detail, context], dim=1)
            raw_detail = self.detail_head(detail_input)
            delta_detail = raw_detail - _local_mean(raw_detail, kernel_size=9)
        else:
            delta_detail = torch.zeros_like(delta_low)
        return delta_low, delta_detail


class NoPostPBCFGA(nn.Module):
    def __init__(
        self,
        feature_channels=32,
        context_channels=32,
        hidden_channels=32,
        gate_bias=-3.0,
        use_low=True,
        use_detail=False,
    ):
        super().__init__()
        self.use_low = bool(use_low)
        self.use_detail = bool(use_detail)
        self.context = ContextEvidenceEncoder(context_channels=context_channels)
        self.gate = PatchBandGate(
            context_channels=context_channels,
            gate_bias=gate_bias,
            use_detail=use_detail,
        )
        self.action = BandFeatureAction(
            feature_channels=feature_channels,
            context_channels=context_channels,
            hidden_channels=hidden_channels,
            use_detail=use_detail,
        )
        self.zero_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

    def forward(self, hazy, final_feature, res1, res2, scm2, scm4):
        context = self.context(
            hazy=hazy,
            final_feature=final_feature,
            res1=res1,
            res2=res2,
            scm2=scm2,
            scm4=scm4,
        )
        g_low, g_detail = self.gate(context)
        delta_low, delta_detail = self.action(final_feature, context)

        if not self.use_low:
            g_low = torch.zeros_like(g_low)
        raw_action = g_low * delta_low + g_detail * delta_detail
        delta_feature = self.zero_proj(raw_action)
        aux = {
            "context": context,
            "g_low": g_low,
            "g_detail": g_detail,
            "delta_low": delta_low,
            "delta_detail": delta_detail,
            "raw_action": raw_action,
            "delta_feature": delta_feature,
        }
        return delta_feature, aux


def summarize_aux(aux):
    if aux is None:
        return {}
    return {
        "g_low_mean": float(aux["g_low"].detach().mean().cpu()),
        "g_low_max": float(aux["g_low"].detach().max().cpu()),
        "g_detail_mean": float(aux["g_detail"].detach().mean().cpu()),
        "raw_action_abs_mean": float(aux["raw_action"].detach().abs().mean().cpu()),
        "delta_feature_abs_mean": float(aux["delta_feature"].detach().abs().mean().cpu()),
    }
