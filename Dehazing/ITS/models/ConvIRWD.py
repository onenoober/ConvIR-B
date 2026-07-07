import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvIR import ConvIR
from .layers import BasicConv


def _haar_dwt2(x):
    """One-level differentiable Haar split with reflect padding for odd sizes."""
    pad_h = x.shape[-2] % 2
    pad_w = x.shape[-1] % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 - x01 + x10 - x11)
    hl = 0.5 * (x00 + x01 - x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)
    return ll, lh, hl, hh


class WDFeatureMod(nn.Module):
    def __init__(self, state_channels, feature_channels):
        super(WDFeatureMod, self).__init__()
        hidden = max(32, feature_channels // 2)
        self.main = nn.Sequential(
            BasicConv(state_channels, hidden, kernel_size=3, relu=True),
            BasicConv(hidden, hidden, kernel_size=5, relu=True),
            nn.Conv2d(hidden, feature_channels * 2, kernel_size=1),
        )
        nn.init.zeros_(self.main[-1].weight)
        nn.init.zeros_(self.main[-1].bias)
        self.last_stats = {}

    def forward(self, feature, state):
        if state.shape[-2:] != feature.shape[-2:]:
            state = F.interpolate(state, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        gamma, beta = self.main(state).chunk(2, dim=1)
        gamma = 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)
        self.last_stats = {
            "gamma_abs_mean": float(gamma.detach().abs().mean().cpu()),
            "beta_abs_mean": float(beta.detach().abs().mean().cpu()),
        }
        return feature * (1.0 + gamma) + beta


class ConvIRWDLite(ConvIR):
    """ConvIR plus neutral-init wavelet degradation modulation.

    This is a full-model-line route scaffold, not an A0 output residual. The
    WD modules read a two-level Haar state from the input and inject it inside
    the bottleneck and decoder fusion path. All route projections are zero
    initialized, so partial-loading the official ConvIR checkpoint starts as an
    exact no-op for Stage-0 verification.
    """

    def __init__(self, version, data, state_channels=64):
        super(ConvIRWDLite, self).__init__(version, data)
        base_channel = 32
        self.WD_state_encoder = nn.Sequential(
            BasicConv(12, base_channel, kernel_size=3, relu=True),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True),
            BasicConv(base_channel * 2, state_channels, kernel_size=3, relu=True),
        )
        self.WD_bottleneck_mod = WDFeatureMod(state_channels, base_channel * 4)
        self.WD_decoder2_mod = WDFeatureMod(state_channels, base_channel * 2)
        self.WD_decoder1_mod = WDFeatureMod(state_channels, base_channel)

    def _wd_state(self, x, target_hw):
        ll1, _, _, _ = _haar_dwt2(x)
        ll2, lh2, hl2, hh2 = _haar_dwt2(ll1)
        state = torch.cat([ll2, lh2, hl2, hh2], dim=1)
        if state.shape[-2:] != target_hw:
            state = F.interpolate(state, size=target_hw, mode="bilinear", align_corners=False)
        return self.WD_state_encoder(state)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        wd_state = self._wd_state(x, z.shape[-2:])
        z = self.Encoder[2](z)
        z = self.WD_bottleneck_mod(z, wd_state)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.WD_decoder2_mod(z, wd_state)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.WD_decoder1_mod(z, wd_state)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs

    def collect_wd_stats(self, x):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _ = self(x)
        if was_training:
            self.train()
        return {
            "bottleneck": dict(self.WD_bottleneck_mod.last_stats),
            "decoder2": dict(self.WD_decoder2_mod.last_stats),
            "decoder1": dict(self.WD_decoder1_mod.last_stats),
        }


def build_convir_wd_net(version, data):
    return ConvIRWDLite(version, data)
