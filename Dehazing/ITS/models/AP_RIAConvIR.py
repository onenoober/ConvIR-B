import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res, data):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel, out_channel, data) for _ in range(num_res - 1)]
        layers.append(ResBlock(out_channel, out_channel, data, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res, data):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel, channel, data) for _ in range(num_res - 1)]
        layers.append(ResBlock(channel, channel, data, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True),
        )

    def forward(self, x):
        return self.main(x)


class FAM(nn.Module):
    def __init__(self, channel, mode='original'):
        super(FAM, self).__init__()
        if mode not in ('original', 'modres'):
            raise ValueError('Unsupported FAM mode: {}'.format(mode))
        self.mode = mode
        self.merge = BasicConv(channel * 2, channel, kernel_size=3, stride=1, relu=False)
        if self.mode == 'modres':
            rng_state = torch.get_rng_state()
            self.modulator = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
            torch.set_rng_state(rng_state)
            nn.init.zeros_(self.modulator.weight)
            nn.init.zeros_(self.modulator.bias)

    def forward(self, x1, x2):
        fused = self.merge(torch.cat([x1, x2], dim=1))
        if self.mode == 'original':
            return fused
        gamma, beta = self.modulator(x2).chunk(2, dim=1)
        return fused * (1 + gamma) + beta

    def modulation_stats(self, x2):
        if self.mode == 'original':
            return None
        with torch.no_grad():
            gamma, beta = self.modulator(x2).chunk(2, dim=1)
        return {
            'gamma_mean': gamma.mean().item(),
            'gamma_std': gamma.std(unbiased=False).item(),
            'gamma_min': gamma.min().item(),
            'gamma_max': gamma.max().item(),
            'gamma_abs_gt_0.5': (gamma.abs() > 0.5).float().mean().item(),
            'beta_mean': beta.mean().item(),
            'beta_std': beta.std(unbiased=False).item(),
            'beta_min': beta.min().item(),
            'beta_max': beta.max().item(),
            'beta_abs_gt_0.1': (beta.abs() > 0.1).float().mean().item(),
        }


def _avg_lowpass(x, kernel_size=5):
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def _gradient_magnitude(x):
    """Return a simple differentiable grayscale gradient magnitude map."""
    gray = x.mean(dim=1, keepdim=True)
    gx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    gy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _dark_channel(x, kernel_size=15):
    """Approximate dark channel prior map from an RGB image in [0, 1]-like range."""
    min_rgb = torch.min(x, dim=1, keepdim=True)[0]
    pad = kernel_size // 2
    return -F.max_pool2d(-min_rgb, kernel_size=kernel_size, stride=1, padding=pad)


class RuntimeEvidenceBuilder(nn.Module):
    """Build inference-time evidence maps from hazy input, anchor side output, and feature.

    This module never consumes expert outputs, teacher outputs, clean GT, or E-A0.
    It therefore remains valid at inference time.
    """

    def __init__(self, lowpass_kernel=9, dark_kernel=15):
        super(RuntimeEvidenceBuilder, self).__init__()
        self.lowpass_kernel = lowpass_kernel
        self.dark_kernel = dark_kernel

    @property
    def out_channels(self):
        # |A0-I|, local energy, dark channel, brightness, saturation,
        # low-energy, high-energy, gradient magnitude
        return 8

    def forward(self, hazy, anchor_side, feature_hw):
        if anchor_side.shape[-2:] != hazy.shape[-2:]:
            anchor_side = F.interpolate(anchor_side, size=hazy.shape[-2:], mode='bilinear', align_corners=False)

        delta = anchor_side - hazy
        abs_delta = delta.abs().mean(dim=1, keepdim=True)
        local_energy = F.avg_pool2d(abs_delta, kernel_size=7, stride=1, padding=3)

        dark = _dark_channel(hazy, kernel_size=self.dark_kernel)
        brightness = torch.max(hazy, dim=1, keepdim=True)[0]
        saturation = torch.max(hazy, dim=1, keepdim=True)[0] - torch.min(hazy, dim=1, keepdim=True)[0]

        delta_low = _avg_lowpass(delta, kernel_size=self.lowpass_kernel)
        delta_high = delta - delta_low
        low_energy = delta_low.abs().mean(dim=1, keepdim=True)
        high_energy = delta_high.abs().mean(dim=1, keepdim=True)

        grad = _gradient_magnitude(hazy)

        maps = torch.cat(
            [abs_delta, local_energy, dark, brightness, saturation, low_energy, high_energy, grad],
            dim=1,
        )
        if maps.shape[-2:] != feature_hw:
            maps = F.interpolate(maps, size=feature_hw, mode='bilinear', align_corners=False)
        return maps


class EvidenceEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(EvidenceEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, evidence_maps):
        return self.net(evidence_maps)


class LowDetailFeatureCorrectionHead(nn.Module):
    """Predict low-frequency and detail feature corrections from ConvIR features and evidence."""

    def __init__(self, feature_channels=32, evidence_channels=32, hidden_channels=64, lowpass_kernel=5):
        super(LowDetailFeatureCorrectionHead, self).__init__()
        self.lowpass_kernel = lowpass_kernel
        in_ch = feature_channels + evidence_channels

        self.low_head = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, feature_channels, kernel_size=1, stride=1, padding=0),
        )

        self.detail_head = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, feature_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, feature, evidence_feature):
        feature_low = _avg_lowpass(feature, kernel_size=self.lowpass_kernel)
        feature_detail = feature - feature_low

        delta_low_raw = self.low_head(torch.cat([feature_low, evidence_feature], dim=1))
        delta_low = _avg_lowpass(delta_low_raw, kernel_size=self.lowpass_kernel)

        delta_detail_raw = self.detail_head(torch.cat([feature_detail, evidence_feature], dim=1))
        delta_detail = delta_detail_raw - _avg_lowpass(delta_detail_raw, kernel_size=self.lowpass_kernel)

        return delta_low, delta_detail


class RiskAwareDualGate(nn.Module):
    """Predict low/detail injection gates.

    G_detail is parameterized as G_low * q_detail by default, giving a conservative
    detail-injection prior without hand-written thresholding.
    """

    def __init__(self, evidence_channels=32, gate_downsample=8, constrain_detail=True):
        super(RiskAwareDualGate, self).__init__()
        self.gate_downsample = int(gate_downsample)
        self.constrain_detail = bool(constrain_detail)

        self.low_gate = nn.Sequential(
            nn.Conv2d(evidence_channels + 1, evidence_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(evidence_channels, 1, kernel_size=3, stride=1, padding=1),
        )
        self.detail_gate = nn.Sequential(
            nn.Conv2d(evidence_channels + 2, evidence_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(evidence_channels, 1, kernel_size=3, stride=1, padding=1),
        )

    def _pool_for_gate(self, x):
        if self.gate_downsample <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=self.gate_downsample, stride=self.gate_downsample)

    def _upsample_gate(self, g, hw):
        if g.shape[-2:] == hw:
            return g
        return F.interpolate(g, size=hw, mode='bilinear', align_corners=False)

    def forward(self, evidence_feature, delta_low, delta_detail):
        hw = evidence_feature.shape[-2:]

        low_energy = delta_low.abs().mean(dim=1, keepdim=True)
        detail_energy = delta_detail.abs().mean(dim=1, keepdim=True)

        low_in = torch.cat([evidence_feature, low_energy], dim=1)
        low_in_small = self._pool_for_gate(low_in)
        g_low_small = torch.sigmoid(self.low_gate(low_in_small))
        g_low = self._upsample_gate(g_low_small, hw)

        detail_in = torch.cat([evidence_feature, detail_energy, g_low.detach()], dim=1)
        detail_in_small = self._pool_for_gate(detail_in)
        q_detail_small = torch.sigmoid(self.detail_gate(detail_in_small))
        q_detail = self._upsample_gate(q_detail_small, hw)

        if self.constrain_detail:
            g_detail = g_low * q_detail
        else:
            g_detail = q_detail
        return g_low, g_detail


class APRIAAdapter(nn.Module):
    """AP-RIA: Evidence-Guided Anchor-Preserving Residual Injection Adapter.

    It modifies ConvIR internal features, not final RGB images. Expert outputs are not
    consumed by this module; teacher guidance is supplied only through training losses.
    """

    def __init__(
        self,
        feature_channels=32,
        evidence_channels=32,
        hidden_channels=64,
        lowpass_kernel=5,
        gate_downsample=8,
        constrain_detail=True,
        detach_evidence=True,
    ):
        super(APRIAAdapter, self).__init__()
        self.detach_evidence = bool(detach_evidence)

        self.evidence_builder = RuntimeEvidenceBuilder(lowpass_kernel=max(5, lowpass_kernel))
        self.evidence_encoder = EvidenceEncoder(self.evidence_builder.out_channels, evidence_channels)
        self.correction_head = LowDetailFeatureCorrectionHead(
            feature_channels=feature_channels,
            evidence_channels=evidence_channels,
            hidden_channels=hidden_channels,
            lowpass_kernel=lowpass_kernel,
        )
        self.gate = RiskAwareDualGate(
            evidence_channels=evidence_channels,
            gate_downsample=gate_downsample,
            constrain_detail=constrain_detail,
        )
        self.zero_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

    def forward(self, feature, hazy, anchor_side, return_aux=False):
        if self.detach_evidence:
            evidence_maps = self.evidence_builder(hazy.detach(), anchor_side.detach(), feature.shape[-2:])
        else:
            evidence_maps = self.evidence_builder(hazy, anchor_side, feature.shape[-2:])

        evidence_feature = self.evidence_encoder(evidence_maps)
        delta_low, delta_detail = self.correction_head(feature, evidence_feature)
        g_low, g_detail = self.gate(evidence_feature, delta_low, delta_detail)

        injection = g_low * delta_low + g_detail * delta_detail
        calibrated = feature + self.zero_proj(injection)

        if not return_aux:
            return calibrated

        aux = {
            'evidence_maps': evidence_maps,
            'evidence_feature': evidence_feature,
            'delta_low': delta_low,
            'delta_detail': delta_detail,
            'g_low': g_low,
            'g_detail': g_detail,
            'injection_raw': injection,
            'injection': self.zero_proj(injection),
        }
        return calibrated, aux


class ConvIR_AP_RIA(nn.Module):
    """ConvIR-B/ConvIR-* with an internal AP-RIA adapter before the final RGB head.

    The final ConvIR reconstruction interface is:
        final decoder feature -> feat_extract[5] -> RGB residual -> + input

    AP-RIA is inserted between Decoder[2] and feat_extract[5]. This keeps the model
    in-network and avoids output-level E-A0 post-processing.
    """

    def __init__(
        self,
        version,
        data,
        fam_mode='original',
        use_ap_ria=True,
        ap_ria_evidence_channels=32,
        ap_ria_hidden_channels=64,
        ap_ria_lowpass_kernel=5,
        ap_ria_gate_downsample=8,
        ap_ria_constrain_detail=True,
        ap_ria_detach_evidence=True,
    ):
        super(ConvIR_AP_RIA, self).__init__()
        if fam_mode not in ('original', 'modres', 'fam2_modres'):
            raise ValueError('Unsupported ConvIR FAM mode: {}'.format(fam_mode))

        if version == 'small':
            num_res = 4
        elif version == 'base':
            num_res = 8
        elif version == 'large':
            num_res = 16
        else:
            raise ValueError('Unsupported ConvIR version: {}'.format(version))

        base_channel = 32
        self.version = version
        self.data = data
        self.use_ap_ria = bool(use_ap_ria)

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
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, data),
            DBlock(base_channel * 2, num_res, data),
            DBlock(base_channel, num_res, data),
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList([
            BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
        ])

        fam1_mode = 'original' if fam_mode == 'fam2_modres' else fam_mode
        fam2_mode = 'modres' if fam_mode == 'fam2_modres' else fam_mode
        self.FAM1 = FAM(base_channel * 4, fam1_mode)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2, fam2_mode)
        self.SCM2 = SCM(base_channel * 2)

        if self.use_ap_ria:
            self.ap_ria = APRIAAdapter(
                feature_channels=base_channel,
                evidence_channels=ap_ria_evidence_channels,
                hidden_channels=ap_ria_hidden_channels,
                lowpass_kernel=ap_ria_lowpass_kernel,
                gate_downsample=ap_ria_gate_downsample,
                constrain_detail=ap_ria_constrain_detail,
                detach_evidence=ap_ria_detach_evidence,
            )
        else:
            self.ap_ria = None

    def anchor_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith('ap_ria.'):
                yield p

    def ap_ria_parameters(self):
        if self.ap_ria is None:
            return iter(())
        return self.ap_ria.parameters()

    def freeze_anchor(self, freeze=True):
        for name, p in self.named_parameters():
            if name.startswith('ap_ria.'):
                p.requires_grad_(True)
            else:
                p.requires_grad_(not freeze)

    def forward(self, x, return_aux=False):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)

        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = []
        aux = {}

        # 256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        # 128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)

        # 128
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)

        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        final_feature = self.Decoder[2](z)

        # Anchor side output: same head on unadapted feature. It is used only as
        # runtime evidence / optional preservation target, not as output fusion.
        anchor_side = self.feat_extract[5](final_feature) + x

        if self.ap_ria is not None:
            if return_aux:
                calibrated_feature, ap_aux = self.ap_ria(final_feature, x, anchor_side, return_aux=True)
                aux.update(ap_aux)
            else:
                calibrated_feature = self.ap_ria(final_feature, x, anchor_side, return_aux=False)
        else:
            calibrated_feature = final_feature

        rgb_residual = self.feat_extract[5](calibrated_feature)
        outputs.append(rgb_residual + x)

        if return_aux:
            aux['final_feature'] = final_feature
            aux['calibrated_feature'] = calibrated_feature
            aux['anchor_side'] = anchor_side
            return outputs, aux
        return outputs

    def collect_modulation_stats(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        stats = {}
        fam1_stats = self.FAM1.modulation_stats(z4)
        fam2_stats = self.FAM2.modulation_stats(z2)
        if fam1_stats is not None:
            stats['FAM1'] = fam1_stats
        if fam2_stats is not None:
            stats['FAM2'] = fam2_stats
        return stats

    def collect_ap_ria_stats(self, x):
        with torch.no_grad():
            outputs, aux = self.forward(x, return_aux=True)
        if 'g_low' not in aux:
            return {}
        return {
            'g_low_mean': aux['g_low'].mean().item(),
            'g_low_std': aux['g_low'].std(unbiased=False).item(),
            'g_low_min': aux['g_low'].min().item(),
            'g_low_max': aux['g_low'].max().item(),
            'g_detail_mean': aux['g_detail'].mean().item(),
            'g_detail_std': aux['g_detail'].std(unbiased=False).item(),
            'g_detail_min': aux['g_detail'].min().item(),
            'g_detail_max': aux['g_detail'].max().item(),
            'injection_abs_mean': aux['injection'].abs().mean().item(),
            'delta_low_abs_mean': aux['delta_low'].abs().mean().item(),
            'delta_detail_abs_mean': aux['delta_detail'].abs().mean().item(),
            'output_mean': outputs[-1].mean().item(),
        }


def build_net(
    version,
    data,
    fam_mode='original',
    use_ap_ria=True,
    ap_ria_evidence_channels=32,
    ap_ria_hidden_channels=64,
    ap_ria_lowpass_kernel=5,
    ap_ria_gate_downsample=8,
    ap_ria_constrain_detail=True,
    ap_ria_detach_evidence=True,
):
    return ConvIR_AP_RIA(
        version=version,
        data=data,
        fam_mode=fam_mode,
        use_ap_ria=use_ap_ria,
        ap_ria_evidence_channels=ap_ria_evidence_channels,
        ap_ria_hidden_channels=ap_ria_hidden_channels,
        ap_ria_lowpass_kernel=ap_ria_lowpass_kernel,
        ap_ria_gate_downsample=ap_ria_gate_downsample,
        ap_ria_constrain_detail=ap_ria_constrain_detail,
        ap_ria_detach_evidence=ap_ria_detach_evidence,
    )
