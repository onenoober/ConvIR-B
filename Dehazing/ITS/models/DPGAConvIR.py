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


DPGA_ADAPTER_NAMES = ("shallow", "bottleneck", "skip")


def parse_dpga_active_adapters(active_adapters):
    if active_adapters is None:
        active_adapters = "all"
    if isinstance(active_adapters, str):
        value = active_adapters.strip().lower()
        if value in ("", "all"):
            return set(DPGA_ADAPTER_NAMES)
        if value in ("none", "off", "zero"):
            return set()
        names = {item.strip().lower() for item in value.split(",") if item.strip()}
    else:
        names = {str(item).strip().lower() for item in active_adapters if str(item).strip()}
    unknown = sorted(names.difference(DPGA_ADAPTER_NAMES))
    if unknown:
        raise ValueError(f"Unknown DPGA adapters: {unknown}; expected any of {DPGA_ADAPTER_NAMES}")
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

    def forward(self, feature, prior, scale_multiplier=1.0):
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
        with torch.no_grad():
            self._last_stats = {
                "scale": float(self.scale.detach().item()),
                "effective_scale": float(scale.detach().item()),
                "scale_multiplier": float(scale_multiplier),
                "delta_l1": float(delta.detach().abs().mean().item()),
                "delta_l2": float(torch.sqrt(torch.mean(delta.detach() * delta.detach()) + 1e-12).item()),
                "scaled_delta_l1": float(scaled_delta.detach().abs().mean().item()),
                "scaled_delta_l2": float(
                    torch.sqrt(torch.mean(scaled_delta.detach() * scaled_delta.detach()) + 1e-12).item()
                ),
                "feature_l1": float(feature.detach().abs().mean().item()),
            }
        return feature + scaled_delta


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
        dark_patch=15,
        local_patch=31,
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
        self.set_dpga_runtime_config()

    def set_dpga_runtime_config(self, active_adapters="all", scale_multiplier=1.0):
        self.dpga_active_adapters = parse_dpga_active_adapters(active_adapters)
        self.dpga_scale_multiplier = float(scale_multiplier)

    def _apply_dpga_adapter(self, name, adapter, feature, prior):
        if name not in self.dpga_active_adapters or self.dpga_scale_multiplier == 0.0:
            adapter._last_stats = {
                "scale": float(adapter.scale.detach().item()),
                "effective_scale": 0.0,
                "scale_multiplier": float(self.dpga_scale_multiplier),
                "delta_l1": 0.0,
                "delta_l2": 0.0,
                "scaled_delta_l1": 0.0,
                "scaled_delta_l2": 0.0,
                "feature_l1": float(feature.detach().abs().mean().item()),
                "inactive": 1.0,
            }
            return feature
        return adapter(feature, prior, scale_multiplier=self.dpga_scale_multiplier)

    def forward(self, x, depth=None):
        prior = build_dpga_prior_maps(
            x,
            depth,
            dark_patch=self.dark_patch,
            local_patch=self.local_patch,
        )
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        res1 = self._apply_dpga_adapter("shallow", self.DPGA_shallow, res1, prior)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        z = self._apply_dpga_adapter("bottleneck", self.DPGA_bottleneck, z, prior)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        z = torch.cat([z, res2], dim=1)
        z = self._apply_dpga_adapter("skip", self.DPGA_skip, z, prior)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs

    def active_dpga_prefixes(self):
        return ("DPGA_shallow", "DPGA_bottleneck", "DPGA_skip")

    def collect_dpga_stats(self):
        stats = {}
        for name, module in self.named_modules():
            if not isinstance(module, DPGALiteAdapter):
                continue
            adapter_name = name.replace("DPGA_", "")
            effective_scale = (
                (torch.tanh(module.scale.detach()) + module.bootstrap_scale)
                * module.residual_scale
                * self.dpga_scale_multiplier
            ).item()
            module_stats = {
                "scale": module.scale.detach().item(),
                "effective_scale": effective_scale if adapter_name in self.dpga_active_adapters else 0.0,
                "scale_multiplier": self.dpga_scale_multiplier,
                "active": float(adapter_name in self.dpga_active_adapters),
            }
            module_stats.update(getattr(module, "_last_stats", {}))
            stats[name] = module_stats
        return stats


def build_dpga_net(
    version,
    data,
    prior_embed_channels=16,
    adapter_reduction=2,
    adapter_residual_scale=0.1,
    adapter_scale_init=0.0,
    adapter_bootstrap_scale=0.01,
    dark_patch=15,
    local_patch=31,
    active_adapters="all",
    scale_multiplier=1.0,
):
    model = DPGAConvIR(
        version,
        data,
        prior_embed_channels=prior_embed_channels,
        adapter_reduction=adapter_reduction,
        adapter_residual_scale=adapter_residual_scale,
        adapter_scale_init=adapter_scale_init,
        adapter_bootstrap_scale=adapter_bootstrap_scale,
        dark_patch=dark_patch,
        local_patch=local_patch,
    )
    model.set_dpga_runtime_config(active_adapters=active_adapters, scale_multiplier=scale_multiplier)
    return model
