import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvIR import DBlock, EBlock, FAM, SCM
from .layers import BasicConv


def gradient_magnitude_1ch(x):
    grad_x = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    grad_y = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)


def luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def dark_channel(x, patch=15):
    dark = x.min(dim=1, keepdim=True).values
    return -F.max_pool2d(-dark, kernel_size=patch, stride=1, padding=patch // 2)


def local_mean(x, patch=31):
    return F.avg_pool2d(x, kernel_size=patch, stride=1, padding=patch // 2)


def build_dpga_prior_maps(x, depth, dark_patch=15, local_patch=31):
    if depth is None:
        raise ValueError("DPGA forward requires a DepthAnything prior tensor.")
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    if depth.shape[1] != 1:
        raise ValueError(f"Expected depth prior with one channel, got {depth.shape}")
    if tuple(depth.shape[-2:]) != tuple(x.shape[-2:]):
        depth = F.interpolate(depth, size=x.shape[-2:], mode="bilinear", align_corners=False)
    depth = depth.to(device=x.device, dtype=x.dtype).clamp(0, 1)

    gray = luma(x)
    min_rgb = x.min(dim=1, keepdim=True).values
    max_rgb = x.max(dim=1, keepdim=True).values
    saturation = max_rgb - min_rgb
    local_contrast = (gray - local_mean(gray, local_patch)).abs()
    return torch.cat(
        [
            depth,
            gradient_magnitude_1ch(depth),
            dark_channel(x, dark_patch),
            max_rgb,
            saturation,
            local_contrast,
            gradient_magnitude_1ch(gray),
            gray,
        ],
        dim=1,
    )


DPGA_LEGACY_ADAPTER_NAMES = ("shallow", "bottleneck", "skip")
DPGA_UDP_FUSION_NAMES = ("dpfm1", "dpfm2", "dpfm4")
DPGA_UDP_AGF_NAMES = ("agf1", "agf2")
DPGA_UDP_COMPONENT_NAMES = ("channel", "cross")
DPGA_UDP_FUSION_MODES = ("udp_lite", "udp_bi")


def is_udp_fusion_mode(fusion_mode):
    return str(fusion_mode or "").strip().lower() in DPGA_UDP_FUSION_MODES


def parse_dpga_active_adapters(active_adapters, fusion_mode="legacy"):
    if active_adapters is None:
        active_adapters = "all"
    if isinstance(active_adapters, str):
        value = active_adapters.strip().lower()
        if value in ("", "all"):
            if is_udp_fusion_mode(fusion_mode):
                return set(DPGA_UDP_FUSION_NAMES)
            return set(DPGA_LEGACY_ADAPTER_NAMES)
        if value in ("none", "off", "zero"):
            return set()
        names = set()
        for item in value.split(","):
            item = item.strip().lower()
            if not item:
                continue
            if item in ("dpfm", "udp", "udp_lite", "udp_bi"):
                names.update(DPGA_UDP_FUSION_NAMES)
            elif item == "agf":
                names.update(DPGA_UDP_AGF_NAMES)
            else:
                names.add(item)
    else:
        names = {str(item).strip().lower() for item in active_adapters if str(item).strip()}
    known = set(DPGA_LEGACY_ADAPTER_NAMES + DPGA_UDP_FUSION_NAMES + DPGA_UDP_AGF_NAMES)
    unknown = sorted(names.difference(known))
    if unknown:
        raise ValueError(f"Unknown DPGA adapters: {unknown}; expected any of {sorted(known)}")
    return names


def parse_udp_components(components):
    if components is None:
        components = "all"
    if isinstance(components, str):
        value = components.strip().lower()
        if value in ("", "all"):
            return set(DPGA_UDP_COMPONENT_NAMES)
        if value in ("none", "off", "zero"):
            return set()
        names = {item.strip().lower() for item in value.split(",") if item.strip()}
    else:
        names = {str(item).strip().lower() for item in components if str(item).strip()}
    unknown = sorted(names.difference(DPGA_UDP_COMPONENT_NAMES))
    if unknown:
        raise ValueError(f"Unknown UDP-Lite components: {unknown}; expected any of {DPGA_UDP_COMPONENT_NAMES}")
    return names


class DPGALiteAdapter(nn.Module):
    def __init__(
        self,
        feature_channels,
        prior_channels,
        prior_embed_channels=16,
        reduction=2,
        residual_scale=0.1,
        scale_init=0.0,
        bootstrap_scale=0.01,
    ):
        super().__init__()
        hidden_channels = max(16, feature_channels // reduction)
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(prior_channels, prior_embed_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(prior_embed_channels, prior_embed_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.reduce = nn.Conv2d(feature_channels + prior_embed_channels, hidden_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.act = nn.GELU()
        self.project = nn.Conv2d(hidden_channels, feature_channels, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.residual_scale = float(residual_scale)
        self.bootstrap_scale = float(bootstrap_scale)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self._last_stats = {}

    def forward(self, feature, prior, scale_multiplier=1.0, gate=None):
        if tuple(prior.shape[-2:]) != tuple(feature.shape[-2:]):
            prior = F.interpolate(prior, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        prior = self.prior_encoder(prior)
        delta = torch.cat([feature, prior], dim=1)
        delta = self.reduce(delta)
        delta = self.depthwise(delta)
        delta = self.act(delta)
        delta = self.project(delta)
        scale = (
            (torch.tanh(self.scale) + self.bootstrap_scale)
            * self.residual_scale
            * float(scale_multiplier)
        )
        scaled_delta = scale * delta
        gate_mean = None
        if gate is not None:
            if tuple(gate.shape[-2:]) != tuple(feature.shape[-2:]):
                gate = F.interpolate(gate, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            scaled_delta = scaled_delta * gate
            gate_mean = float(gate.detach().mean().item())
        with torch.no_grad():
            self._last_stats = {
                "scale": float(self.scale.detach().item()),
                "effective_scale": float(scale.detach().item()),
                "scale_multiplier": float(scale_multiplier),
                "gate_mean": gate_mean if gate_mean is not None else 1.0,
                "delta_l1": float(delta.detach().abs().mean().item()),
                "delta_l2": float(torch.sqrt(torch.mean(delta.detach() * delta.detach()) + 1e-12).item()),
                "scaled_delta_l1": float(scaled_delta.detach().abs().mean().item()),
                "scaled_delta_l2": float(
                    torch.sqrt(torch.mean(scaled_delta.detach() * scaled_delta.detach()) + 1e-12).item()
                ),
                "feature_l1": float(feature.detach().abs().mean().item()),
            }
        return feature + scaled_delta


class DPGAHardGate(nn.Module):
    def __init__(self, prior_channels, hidden_channels=16, init_bias=-3.0):
        super().__init__()
        hidden_channels = max(8, int(hidden_channels))
        self.body = nn.Sequential(
            nn.Conv2d(prior_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias, float(init_bias))

    def forward(self, prior):
        logits = self.body(prior)
        return logits, torch.sigmoid(logits)


def _zero_init_last_conv(module):
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Conv2d):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return


def _choose_heads(channels, requested_heads):
    requested_heads = max(1, int(requested_heads))
    for heads in range(min(requested_heads, channels), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


def _window_partition(x, window_size):
    b, c, h, w = x.shape
    x = x.view(b, c, h // window_size, window_size, w // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, window_size * window_size, c)


def _window_reverse(windows, window_size, batch, height, width, channels):
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(batch, channels, height, width)


class DepthPriorPyramid(nn.Module):
    def __init__(self, prior_channels, channels):
        super().__init__()
        c1, c2, c4 = channels
        self.stem = nn.Sequential(
            nn.Conv2d(prior_channels, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(c2, c4, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self._last_stats = {}

    def forward(self, prior):
        d1 = self.stem(prior)
        d2 = self.down2(d1)
        d4 = self.down4(d2)
        with torch.no_grad():
            self._last_stats = {
                "d1_l1": float(d1.detach().abs().mean().item()),
                "d2_l1": float(d2.detach().abs().mean().item()),
                "d4_l1": float(d4.detach().abs().mean().item()),
            }
        return d1, d2, d4


class DepthGuidedChannelAttention(nn.Module):
    def __init__(self, feature_channels, depth_channels, reduction=4):
        super().__init__()
        hidden_channels = max(8, feature_channels // int(max(1, reduction)))
        self.mlp = nn.Sequential(
            nn.Conv2d(feature_channels + depth_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, feature_channels, kernel_size=1),
        )
        _zero_init_last_conv(self.mlp)

    def forward(self, feature, depth_feature):
        if tuple(depth_feature.shape[-2:]) != tuple(feature.shape[-2:]):
            depth_feature = F.interpolate(depth_feature, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(feature, 1),
                F.adaptive_avg_pool2d(depth_feature, 1),
            ],
            dim=1,
        )
        gate_delta = torch.tanh(self.mlp(pooled))
        return feature * gate_delta


class LocalWindowCrossAttention(nn.Module):
    def __init__(
        self,
        feature_channels,
        depth_channels,
        window_size=8,
        num_heads=4,
        query_source="depth",
    ):
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.window_size = max(1, int(window_size))
        self.num_heads = _choose_heads(self.feature_channels, num_heads)
        self.head_dim = self.feature_channels // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.query_source = str(query_source).strip().lower()
        if self.query_source not in ("depth", "feature"):
            raise ValueError("query_source must be 'depth' or 'feature'")
        q_channels = depth_channels if self.query_source == "depth" else feature_channels
        kv_channels = feature_channels if self.query_source == "depth" else depth_channels
        self.q = nn.Conv2d(q_channels, self.feature_channels, kernel_size=1)
        self.kv = nn.Conv2d(kv_channels, self.feature_channels * 2, kernel_size=1)
        self.project = nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, feature, depth_feature):
        if tuple(depth_feature.shape[-2:]) != tuple(feature.shape[-2:]):
            depth_feature = F.interpolate(depth_feature, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        query = depth_feature if self.query_source == "depth" else feature
        key_value = feature if self.query_source == "depth" else depth_feature
        q = self.q(query)
        k, v = self.kv(key_value).chunk(2, dim=1)
        batch, channels, height, width = feature.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h or pad_w:
            q = F.pad(q, (0, pad_w, 0, pad_h), mode="replicate")
            k = F.pad(k, (0, pad_w, 0, pad_h), mode="replicate")
            v = F.pad(v, (0, pad_w, 0, pad_h), mode="replicate")
        padded_h, padded_w = q.shape[-2:]
        q_windows = _window_partition(q, self.window_size)
        k_windows = _window_partition(k, self.window_size)
        v_windows = _window_partition(v, self.window_size)
        tokens = q_windows.shape[1]
        q_windows = q_windows.view(-1, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k_windows = k_windows.view(-1, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v_windows = v_windows.view(-1, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q_windows @ k_windows.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v_windows).transpose(1, 2).contiguous()
        out = out.view(-1, tokens, channels)
        out = _window_reverse(out, self.window_size, batch, padded_h, padded_w, channels)
        if pad_h or pad_w:
            out = out[:, :, :height, :width]
        return self.project(out)


class DPFMLiteFusion(nn.Module):
    def __init__(
        self,
        feature_channels,
        depth_channels,
        residual_scale=0.1,
        scale_init=0.0,
        bootstrap_scale=0.01,
        window_size=8,
        num_heads=4,
        bidirectional=False,
    ):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.channel = DepthGuidedChannelAttention(feature_channels, depth_channels)
        if self.bidirectional:
            self.cross_rgb_from_depth = LocalWindowCrossAttention(
                feature_channels,
                depth_channels,
                window_size=window_size,
                num_heads=num_heads,
                query_source="feature",
            )
            self.cross_depth_from_rgb = LocalWindowCrossAttention(
                feature_channels,
                depth_channels,
                window_size=window_size,
                num_heads=num_heads,
                query_source="depth",
            )
        else:
            self.cross = LocalWindowCrossAttention(
                feature_channels,
                depth_channels,
                window_size=window_size,
                num_heads=num_heads,
            )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.residual_scale = float(residual_scale)
        self.bootstrap_scale = float(bootstrap_scale)
        self._last_stats = {}
        self._last_delta_reg = None

    def _effective_scale(self, scale_multiplier):
        return (
            (torch.tanh(self.scale) + self.bootstrap_scale)
            * self.residual_scale
            * float(scale_multiplier)
        )

    def forward(self, feature, depth_feature, scale_multiplier=1.0, components=None):
        components = parse_udp_components(components)
        if not components:
            self._last_delta_reg = None
            with torch.no_grad():
                self._last_stats = {
                    "scale": float(self.scale.detach().item()),
                    "effective_scale": 0.0,
                    "scale_multiplier": float(scale_multiplier),
                    "channel_delta_l1": 0.0,
                    "cross_delta_l1": 0.0,
                    "cross_rgb_from_depth_delta_l1": 0.0,
                    "cross_depth_from_rgb_delta_l1": 0.0,
                    "scaled_delta_l1": 0.0,
                    "scaled_delta_l2": 0.0,
                    "feature_l1": float(feature.detach().abs().mean().item()),
                    "inactive": 1.0,
                    "bidirectional": float(self.bidirectional),
                }
            return feature
        channel_delta = self.channel(feature, depth_feature) if "channel" in components else torch.zeros_like(feature)
        cross_rgb_delta = torch.zeros_like(feature)
        cross_depth_delta = torch.zeros_like(feature)
        if "cross" in components:
            if self.bidirectional:
                cross_rgb_delta = self.cross_rgb_from_depth(feature, depth_feature)
                cross_depth_delta = self.cross_depth_from_rgb(feature, depth_feature)
                cross_delta = cross_rgb_delta + cross_depth_delta
            else:
                cross_delta = self.cross(feature, depth_feature)
        else:
            cross_delta = torch.zeros_like(feature)
        delta = channel_delta + cross_delta
        effective_scale = self._effective_scale(scale_multiplier)
        scaled_delta = effective_scale * delta
        self._last_delta_reg = torch.mean(scaled_delta * scaled_delta)
        with torch.no_grad():
            self._last_stats = {
                "scale": float(self.scale.detach().item()),
                "effective_scale": float(effective_scale.detach().item()),
                "scale_multiplier": float(scale_multiplier),
                "channel_delta_l1": float(channel_delta.detach().abs().mean().item()),
                "cross_delta_l1": float(cross_delta.detach().abs().mean().item()),
                "cross_rgb_from_depth_delta_l1": float(cross_rgb_delta.detach().abs().mean().item()),
                "cross_depth_from_rgb_delta_l1": float(cross_depth_delta.detach().abs().mean().item()),
                "scaled_delta_l1": float(scaled_delta.detach().abs().mean().item()),
                "scaled_delta_l2": float(torch.sqrt(torch.mean(scaled_delta.detach() * scaled_delta.detach()) + 1e-12).item()),
                "feature_l1": float(feature.detach().abs().mean().item()),
                "channel_active": float("channel" in components),
                "cross_active": float("cross" in components),
                "bidirectional": float(self.bidirectional),
            }
        return feature + scaled_delta


class AGFLiteSkipGate(nn.Module):
    def __init__(
        self,
        skip_channels,
        decoder_channels,
        depth_channels,
        gate_limit=0.25,
        scale_init=0.0,
        bootstrap_scale=0.01,
    ):
        super().__init__()
        hidden_channels = max(8, min(skip_channels, decoder_channels))
        in_channels = skip_channels + decoder_channels + depth_channels
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )
        self.channel_gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, skip_channels, kernel_size=1),
        )
        _zero_init_last_conv(self.spatial_gate)
        _zero_init_last_conv(self.channel_gate)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.bootstrap_scale = float(bootstrap_scale)
        self.gate_limit = float(gate_limit)
        self._last_stats = {}
        self._last_delta_reg = None

    def _effective_scale(self, scale_multiplier):
        return (torch.tanh(self.scale) + self.bootstrap_scale) * self.gate_limit * float(scale_multiplier)

    def forward(self, skip, decoder, depth_feature, scale_multiplier=1.0):
        if tuple(decoder.shape[-2:]) != tuple(skip.shape[-2:]):
            decoder = F.interpolate(decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if tuple(depth_feature.shape[-2:]) != tuple(skip.shape[-2:]):
            depth_feature = F.interpolate(depth_feature, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        merged = torch.cat([skip, decoder, depth_feature], dim=1)
        spatial_delta = torch.tanh(self.spatial_gate(merged))
        channel_delta = torch.tanh(self.channel_gate(F.adaptive_avg_pool2d(merged, 1)))
        effective_scale = self._effective_scale(scale_multiplier)
        gate_delta = effective_scale * (spatial_delta + channel_delta)
        gated = skip * (1.0 + gate_delta)
        delta = gated - skip
        self._last_delta_reg = torch.mean(delta * delta)
        with torch.no_grad():
            self._last_stats = {
                "scale": float(self.scale.detach().item()),
                "effective_scale": float(effective_scale.detach().item()),
                "scale_multiplier": float(scale_multiplier),
                "gate_delta_l1": float(gate_delta.detach().abs().mean().item()),
                "skip_delta_l1": float(delta.detach().abs().mean().item()),
                "scaled_delta_l1": float(delta.detach().abs().mean().item()),
                "scaled_delta_l2": float(torch.sqrt(torch.mean(delta.detach() * delta.detach()) + 1e-12).item()),
                "feature_l1": float(skip.detach().abs().mean().item()),
            }
        return gated


class DPGAConvIR(nn.Module):
    def __init__(
        self,
        version,
        data,
        prior_channels=8,
        prior_embed_channels=16,
        adapter_reduction=2,
        adapter_residual_scale=0.1,
        adapter_scale_init=0.0,
        adapter_bootstrap_scale=0.01,
        hard_gate_init_bias=-3.0,
        dark_patch=15,
        local_patch=31,
        fusion_mode="legacy",
        udp_components="all",
        udp_window_size=8,
        udp_num_heads=4,
        agf_gate_limit=0.25,
    ):
        super(DPGAConvIR, self).__init__()

        if version == 'small':
            num_res = 4
        elif version == 'base':
            num_res = 8
        elif version == 'large':
            num_res = 16
        else:
            raise ValueError(f'Unsupported ConvIR version: {version}')

        base_channel = 32
        self.dark_patch = int(dark_patch)
        self.local_patch = int(local_patch)

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, data),
            EBlock(base_channel * 2, num_res, data),
            EBlock(base_channel * 4, num_res, data),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, data),
            DBlock(base_channel * 2, num_res, data),
            DBlock(base_channel, num_res, data)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4, 'original')
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2, 'original')
        self.SCM2 = SCM(base_channel * 2)

        adapter_kwargs = dict(
            prior_channels=prior_channels,
            prior_embed_channels=prior_embed_channels,
            reduction=adapter_reduction,
            residual_scale=adapter_residual_scale,
            scale_init=adapter_scale_init,
            bootstrap_scale=adapter_bootstrap_scale,
        )
        self.DPGA_shallow = DPGALiteAdapter(base_channel, **adapter_kwargs)
        self.DPGA_bottleneck = DPGALiteAdapter(base_channel * 4, **adapter_kwargs)
        self.DPGA_skip = DPGALiteAdapter(base_channel * 4, **adapter_kwargs)
        self.DPGA_hard_gate = DPGAHardGate(
            prior_channels,
            hidden_channels=prior_embed_channels,
            init_bias=hard_gate_init_bias,
        )
        depth_channels = (
            prior_embed_channels,
            prior_embed_channels * 2,
            prior_embed_channels * 4,
        )
        self.dpga_dpfm_bidirectional = str(fusion_mode or "").strip().lower() == "udp_bi"
        self.DPGA_prior_encoder = DepthPriorPyramid(prior_channels, depth_channels)
        self.DPGA_dpfm1 = DPFMLiteFusion(
            base_channel,
            depth_channels[0],
            residual_scale=adapter_residual_scale,
            scale_init=adapter_scale_init,
            bootstrap_scale=adapter_bootstrap_scale,
            window_size=udp_window_size,
            num_heads=udp_num_heads,
            bidirectional=self.dpga_dpfm_bidirectional,
        )
        self.DPGA_dpfm2 = DPFMLiteFusion(
            base_channel * 2,
            depth_channels[1],
            residual_scale=adapter_residual_scale,
            scale_init=adapter_scale_init,
            bootstrap_scale=adapter_bootstrap_scale,
            window_size=udp_window_size,
            num_heads=udp_num_heads,
            bidirectional=self.dpga_dpfm_bidirectional,
        )
        self.DPGA_dpfm4 = DPFMLiteFusion(
            base_channel * 4,
            depth_channels[2],
            residual_scale=adapter_residual_scale,
            scale_init=adapter_scale_init,
            bootstrap_scale=adapter_bootstrap_scale,
            window_size=udp_window_size,
            num_heads=udp_num_heads,
            bidirectional=self.dpga_dpfm_bidirectional,
        )
        self.DPGA_agf2 = AGFLiteSkipGate(
            base_channel * 2,
            base_channel * 2,
            depth_channels[1],
            gate_limit=agf_gate_limit,
            scale_init=adapter_scale_init,
            bootstrap_scale=adapter_bootstrap_scale,
        )
        self.DPGA_agf1 = AGFLiteSkipGate(
            base_channel,
            base_channel,
            depth_channels[0],
            gate_limit=agf_gate_limit,
            scale_init=adapter_scale_init,
            bootstrap_scale=adapter_bootstrap_scale,
        )
        self._last_hard_gate = None
        self._last_hard_gate_logits = None
        self.set_dpga_runtime_config(
            fusion_mode=fusion_mode,
            udp_components=udp_components,
        )

    def set_dpga_runtime_config(
        self,
        active_adapters="all",
        scale_multiplier=1.0,
        hard_gate_mode="off",
        shallow_scale_multiplier=1.0,
        bottleneck_scale_multiplier=1.0,
        skip_scale_multiplier=1.0,
        fusion_mode=None,
        udp_components=None,
    ):
        if fusion_mode is None:
            fusion_mode = getattr(self, "dpga_fusion_mode", "legacy")
        fusion_mode = str(fusion_mode or "legacy").strip().lower()
        if fusion_mode not in ("legacy", "udp_lite", "udp_bi"):
            raise ValueError("DPGA fusion_mode must be 'legacy', 'udp_lite', or 'udp_bi'")
        if fusion_mode == "udp_bi" and not getattr(self, "dpga_dpfm_bidirectional", False):
            raise ValueError("DPGA model must be constructed with fusion_mode='udp_bi' before enabling udp_bi.")
        if fusion_mode == "udp_lite" and getattr(self, "dpga_dpfm_bidirectional", False):
            raise ValueError("DPGA model constructed with bidirectional DPFM cannot switch back to udp_lite.")
        self.dpga_fusion_mode = fusion_mode
        self.dpga_active_adapters = parse_dpga_active_adapters(active_adapters, fusion_mode=fusion_mode)
        if udp_components is None:
            udp_components = getattr(self, "dpga_udp_components", set(DPGA_UDP_COMPONENT_NAMES))
        self.dpga_udp_components = parse_udp_components(udp_components)
        self.dpga_scale_multiplier = float(scale_multiplier)
        self.dpga_hard_gate_mode = str(hard_gate_mode or "off").strip().lower()
        if self.dpga_hard_gate_mode not in ("off", "bottleneck"):
            raise ValueError("DPGA hard_gate_mode must be 'off' or 'bottleneck'")
        self.dpga_adapter_scale_multipliers = {
            "shallow": float(shallow_scale_multiplier),
            "bottleneck": float(bottleneck_scale_multiplier),
            "skip": float(skip_scale_multiplier),
        }

    def _adapter_scale_multiplier(self, name):
        return self.dpga_scale_multiplier * self.dpga_adapter_scale_multipliers.get(name, 1.0)

    def _apply_dpga_adapter(self, name, adapter, feature, prior, gate=None):
        scale_multiplier = self._adapter_scale_multiplier(name)
        if name not in self.dpga_active_adapters or scale_multiplier == 0.0:
            adapter._last_stats = {
                "scale": float(adapter.scale.detach().item()),
                "effective_scale": 0.0,
                "scale_multiplier": float(scale_multiplier),
                "gate_mean": 0.0 if gate is not None else 1.0,
                "delta_l1": 0.0,
                "delta_l2": 0.0,
                "scaled_delta_l1": 0.0,
                "scaled_delta_l2": 0.0,
                "feature_l1": float(feature.detach().abs().mean().item()),
                "inactive": 1.0,
            }
            return feature
        return adapter(feature, prior, scale_multiplier=scale_multiplier, gate=gate)

    def _apply_dpfm(self, name, module, feature, depth_feature):
        scale_multiplier = self._adapter_scale_multiplier(name)
        if name not in self.dpga_active_adapters or scale_multiplier == 0.0:
            return module(feature, depth_feature, scale_multiplier=0.0, components="none")
        return module(
            feature,
            depth_feature,
            scale_multiplier=scale_multiplier,
            components=self.dpga_udp_components,
        )

    def _apply_agf(self, name, module, skip, decoder, depth_feature):
        scale_multiplier = self._adapter_scale_multiplier(name)
        if name not in self.dpga_active_adapters or scale_multiplier == 0.0:
            module._last_delta_reg = None
            with torch.no_grad():
                module._last_stats = {
                    "scale": float(module.scale.detach().item()),
                    "effective_scale": 0.0,
                    "scale_multiplier": float(scale_multiplier),
                    "gate_delta_l1": 0.0,
                    "skip_delta_l1": 0.0,
                    "scaled_delta_l1": 0.0,
                    "scaled_delta_l2": 0.0,
                    "feature_l1": float(skip.detach().abs().mean().item()),
                    "inactive": 1.0,
                }
            return skip
        return module(skip, decoder, depth_feature, scale_multiplier=scale_multiplier)

    def forward(self, x, depth=None):
        prior = build_dpga_prior_maps(
            x,
            depth,
            dark_patch=self.dark_patch,
            local_patch=self.local_patch,
        )
        self._last_hard_gate = None
        self._last_hard_gate_logits = None
        hard_gate = None
        if self.dpga_hard_gate_mode == "bottleneck":
            hard_gate_logits, hard_gate = self.DPGA_hard_gate(prior)
            self._last_hard_gate = hard_gate
            self._last_hard_gate_logits = hard_gate_logits
        if is_udp_fusion_mode(self.dpga_fusion_mode):
            depth1, depth2, depth4 = self.DPGA_prior_encoder(prior)
        else:
            depth1 = depth2 = depth4 = None
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        if is_udp_fusion_mode(self.dpga_fusion_mode):
            res1 = self._apply_dpfm("dpfm1", self.DPGA_dpfm1, res1, depth1)
        else:
            res1 = self._apply_dpga_adapter("shallow", self.DPGA_shallow, res1, prior)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        if is_udp_fusion_mode(self.dpga_fusion_mode):
            res2 = self._apply_dpfm("dpfm2", self.DPGA_dpfm2, res2, depth2)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        if is_udp_fusion_mode(self.dpga_fusion_mode):
            z = self._apply_dpfm("dpfm4", self.DPGA_dpfm4, z, depth4)
        else:
            z = self._apply_dpga_adapter("bottleneck", self.DPGA_bottleneck, z, prior, gate=hard_gate)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        if is_udp_fusion_mode(self.dpga_fusion_mode):
            res2_skip = self._apply_agf("agf2", self.DPGA_agf2, res2, z, depth2)
            z = torch.cat([z, res2_skip], dim=1)
        else:
            z = torch.cat([z, res2], dim=1)
            z = self._apply_dpga_adapter("skip", self.DPGA_skip, z, prior)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        if is_udp_fusion_mode(self.dpga_fusion_mode):
            res1_skip = self._apply_agf("agf1", self.DPGA_agf1, res1, z, depth1)
            z = torch.cat([z, res1_skip], dim=1)
        else:
            z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs

    def active_dpga_prefixes(self):
        return (
            "DPGA_shallow",
            "DPGA_bottleneck",
            "DPGA_skip",
            "DPGA_hard_gate",
            "DPGA_prior_encoder",
            "DPGA_dpfm1",
            "DPGA_dpfm2",
            "DPGA_dpfm4",
            "DPGA_agf1",
            "DPGA_agf2",
        )

    def dpga_fusion_delta_regularization(self):
        regs = []
        for module in (
            self.DPGA_dpfm1,
            self.DPGA_dpfm2,
            self.DPGA_dpfm4,
            self.DPGA_agf1,
            self.DPGA_agf2,
        ):
            value = getattr(module, "_last_delta_reg", None)
            if value is not None:
                regs.append(value)
        if not regs:
            return None
        return sum(regs) / len(regs)

    def collect_dpga_stats(self):
        stats = {}
        for name, module in self.named_modules():
            if not isinstance(module, DPGALiteAdapter):
                continue
            adapter_name = name.replace("DPGA_", "")
            effective_scale = (
                (torch.tanh(module.scale.detach()) + module.bootstrap_scale)
                * module.residual_scale
                * self._adapter_scale_multiplier(adapter_name)
            ).item()
            module_stats = {
                "scale": module.scale.detach().item(),
                "effective_scale": effective_scale if adapter_name in self.dpga_active_adapters else 0.0,
                "scale_multiplier": self._adapter_scale_multiplier(adapter_name),
                "active": float(adapter_name in self.dpga_active_adapters),
            }
            module_stats.update(getattr(module, "_last_stats", {}))
            stats[name] = module_stats
        for name, module in self.named_modules():
            if not isinstance(module, (DPFMLiteFusion, AGFLiteSkipGate)):
                continue
            module_name = name.replace("DPGA_", "")
            module_stats = {
                "scale": module.scale.detach().item(),
                "effective_scale": getattr(module, "_last_stats", {}).get("effective_scale", 0.0),
                "scale_multiplier": self._adapter_scale_multiplier(module_name),
                "active": float(module_name in self.dpga_active_adapters),
            }
            module_stats.update(getattr(module, "_last_stats", {}))
            stats[name] = module_stats
        if hasattr(self.DPGA_prior_encoder, "_last_stats"):
            stats["DPGA_prior_encoder"] = {
                "active": float(is_udp_fusion_mode(self.dpga_fusion_mode)),
                **self.DPGA_prior_encoder._last_stats,
            }
        if self._last_hard_gate is not None:
            stats["DPGA_hard_gate"] = {
                "active": float(self.dpga_hard_gate_mode == "bottleneck"),
                "gate_mean": float(self._last_hard_gate.detach().mean().item()),
                "gate_min": float(self._last_hard_gate.detach().amin().item()),
                "gate_max": float(self._last_hard_gate.detach().amax().item()),
            }
        return stats


def build_dpga_net(
    version,
    data,
    prior_embed_channels=16,
    adapter_reduction=2,
    adapter_residual_scale=0.1,
    adapter_scale_init=0.0,
    adapter_bootstrap_scale=0.01,
    hard_gate_init_bias=-3.0,
    dark_patch=15,
    local_patch=31,
    active_adapters="all",
    scale_multiplier=1.0,
    hard_gate_mode="off",
    shallow_scale_multiplier=1.0,
    bottleneck_scale_multiplier=1.0,
    skip_scale_multiplier=1.0,
    fusion_mode="legacy",
    udp_components="all",
    udp_window_size=8,
    udp_num_heads=4,
    agf_gate_limit=0.25,
):
    model = DPGAConvIR(
        version,
        data,
        prior_embed_channels=prior_embed_channels,
        adapter_reduction=adapter_reduction,
        adapter_residual_scale=adapter_residual_scale,
        adapter_scale_init=adapter_scale_init,
        adapter_bootstrap_scale=adapter_bootstrap_scale,
        hard_gate_init_bias=hard_gate_init_bias,
        dark_patch=dark_patch,
        local_patch=local_patch,
        fusion_mode=fusion_mode,
        udp_components=udp_components,
        udp_window_size=udp_window_size,
        udp_num_heads=udp_num_heads,
        agf_gate_limit=agf_gate_limit,
    )
    model.set_dpga_runtime_config(
        active_adapters=active_adapters,
        scale_multiplier=scale_multiplier,
        hard_gate_mode=hard_gate_mode,
        shallow_scale_multiplier=shallow_scale_multiplier,
        bottleneck_scale_multiplier=bottleneck_scale_multiplier,
        skip_scale_multiplier=skip_scale_multiplier,
        fusion_mode=fusion_mode,
        udp_components=udp_components,
    )
    return model
