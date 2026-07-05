
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res, data):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel, data) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, data, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res, data):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel, data) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, data, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

class ConvIR(nn.Module):
    def __init__(self, version, data):
        super(ConvIR, self).__init__()
        
        if version == 'small':
            num_res = 4
        elif version == 'base':
            num_res = 8
        elif version == 'large':
            num_res = 16

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, data),
            EBlock(base_channel*2, num_res, data),
            EBlock(base_channel*4, num_res, data),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
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

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
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
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


class BoundedInternalLowFrequencyCorrectionField(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels=32,
        alpha_max=0.02,
        gate_bias=-4.0,
        lowpass_kernel=5,
    ):
        super(BoundedInternalLowFrequencyCorrectionField, self).__init__()
        if lowpass_kernel < 1 or lowpass_kernel % 2 != 1:
            raise ValueError("lowpass_kernel must be a positive odd integer")
        self.alpha_max = float(alpha_max)
        self.gate_bias = float(gate_bias)
        self.lowpass_kernel = int(lowpass_kernel)
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1, bias=True)
        self.gate_head = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1, bias=True)
        self._last_correction = None
        self._last_gate = None
        self._last_raw_delta_low = None
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.body:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def _lowpass(self, tensor):
        pad = self.lowpass_kernel // 2
        return F.avg_pool2d(
            tensor,
            kernel_size=self.lowpass_kernel,
            stride=1,
            padding=pad,
            count_include_pad=False,
        )

    def forward(self, feature):
        hidden = self.body(feature)
        raw_delta = self.delta_head(hidden)
        raw_delta_low = self._lowpass(raw_delta)
        gate = torch.sigmoid(self.gate_head(hidden) + self.gate_bias)
        correction = self.alpha_max * torch.tanh(raw_delta_low) * gate
        self._last_correction = correction
        self._last_gate = gate
        self._last_raw_delta_low = raw_delta_low
        return feature + correction

    def regularization(self):
        if self._last_correction is None:
            return None
        return self._last_correction.abs().mean()

    def stats(self):
        if self._last_correction is None or self._last_gate is None:
            return {}
        correction = self._last_correction.detach()
        gate = self._last_gate.detach()
        high = correction - self._lowpass(correction)
        corr_abs = correction.abs().flatten()
        high_rms = high.pow(2).mean().sqrt()
        corr_rms = correction.pow(2).mean().sqrt()
        p95 = torch.quantile(corr_abs, 0.95).item() if corr_abs.numel() else 0.0
        leakage = (high_rms / corr_rms.clamp_min(1e-12)).item()
        return {
            "field_energy_mean": correction.abs().mean().item(),
            "field_energy_rms": corr_rms.item(),
            "field_energy_p95": p95,
            "field_abs_max": corr_abs.max().item() if corr_abs.numel() else 0.0,
            "gate_mean": gate.mean().item(),
            "gate_std": gate.std(unbiased=False).item(),
            "gate_min": gate.min().item(),
            "gate_max": gate.max().item(),
            "highfreq_leakage": leakage,
            "lowfreq_ratio": max(0.0, 1.0 - leakage),
        }


class BILFCFConvIR(ConvIR):
    def __init__(
        self,
        version,
        data,
        insertion="s5",
        alpha_max=0.02,
        gate_bias=-4.0,
        hidden_channels=32,
        lowpass_kernel=5,
    ):
        super(BILFCFConvIR, self).__init__(version, data)
        if insertion != "s5":
            raise ValueError("v2.32 currently enables only the conservative s5 insertion")
        self.bilfcf_insertion = insertion
        self.BILFCF_s5 = BoundedInternalLowFrequencyCorrectionField(
            channels=32 * 4,
            hidden_channels=hidden_channels,
            alpha_max=alpha_max,
            gate_bias=gate_bias,
            lowpass_kernel=lowpass_kernel,
        )

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
        z = self.Encoder[2](z)
        z = self.BILFCF_s5(z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        z = torch.cat([z, res2], dim=1)
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

    def bilfcf_regularization(self):
        reg = self.BILFCF_s5.regularization()
        if reg is None:
            return None
        return reg

    def get_bilfcf_stats(self):
        return {"BILFCF_s5": self.BILFCF_s5.stats()}

    def collect_bilfcf_stats(self, x):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self(x)
            stats = self.get_bilfcf_stats()
        if was_training:
            self.train()
        return stats


def build_net(version, data, fam_mode='original'):
    if fam_mode != 'original':
        raise ValueError(
            "Official ConvIR-B anchor only supports fam_mode='original'. "
            "Create a route branch for architecture variants."
        )
    return ConvIR(version, data)


def build_bilfcf_net(
    version,
    data,
    fam_mode='original',
    insertion='s5',
    alpha_max=0.02,
    gate_bias=-4.0,
    hidden_channels=32,
    lowpass_kernel=5,
):
    if fam_mode != 'original':
        raise ValueError("BILFCF keeps fam_mode='original' for anchor parity.")
    return BILFCFConvIR(
        version,
        data,
        insertion=insertion,
        alpha_max=alpha_max,
        gate_bias=gate_bias,
        hidden_channels=hidden_channels,
        lowpass_kernel=lowpass_kernel,
    )
