import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvIR import ConvIR


class HaarDWT2D(nn.Module):
    def forward(self, x):
        h, w = x.shape[-2:]
        ph = h % 2
        pw = w % 2
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        a = x[:, :, 0::2, 0::2]
        b = x[:, :, 0::2, 1::2]
        c = x[:, :, 1::2, 0::2]
        d = x[:, :, 1::2, 1::2]
        ll = (a + b + c + d) / 2.0
        lh = (a - b + c - d) / 2.0
        hl = (a + b - c - d) / 2.0
        hh = (a - b - c + d) / 2.0
        return ll, lh, hl, hh, h, w


class HaarIWT2D(nn.Module):
    def forward(self, ll, lh, hl, hh, h, w):
        a = (ll + lh + hl + hh) / 2.0
        b = (ll - lh + hl - hh) / 2.0
        c = (ll + lh - hl - hh) / 2.0
        d = (ll - lh - hl + hh) / 2.0
        out = torch.empty(
            (ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2),
            dtype=ll.dtype,
            device=ll.device,
        )
        out[:, :, 0::2, 0::2] = a
        out[:, :, 0::2, 1::2] = b
        out[:, :, 1::2, 0::2] = c
        out[:, :, 1::2, 1::2] = d
        return out[:, :, :h, :w]


class GlobalLowbandPolicyBlock(nn.Module):
    def __init__(self, channels, hidden_channels=32):
        super().__init__()
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, ll):
        return self.project(self.context(ll))


class SpatialLowbandPolicyBlock(nn.Module):
    def __init__(self, channels, hidden_channels=32):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, ll):
        return self.project(self.context(ll))


class LowbandPolicyInsertion(nn.Module):
    def __init__(self, channels, hidden_channels=32, policy_mode="global"):
        super().__init__()
        if policy_mode == "global":
            self.policy = GlobalLowbandPolicyBlock(channels, hidden_channels)
        elif policy_mode == "spatial":
            self.policy = SpatialLowbandPolicyBlock(channels, hidden_channels)
        else:
            raise ValueError(f"unsupported policy_mode: {policy_mode}")
        self.dwt = HaarDWT2D()
        self.iwt = HaarIWT2D()

    def forward(self, z):
        ll, lh, hl, hh, h, w = self.dwt(z)
        ll_delta = self.policy(ll)
        zero_lh = torch.zeros_like(lh)
        zero_hl = torch.zeros_like(hl)
        zero_hh = torch.zeros_like(hh)
        delta = self.iwt(ll_delta, zero_lh, zero_hl, zero_hh, h, w)
        return z + delta


class NoPostLowbandPolicyConvIR(ConvIR):
    def __init__(self, version, data, hidden_channels=32, policy_mode="global"):
        super().__init__(version, data)
        self.nopost_lowband_policy = LowbandPolicyInsertion(
            32,
            hidden_channels=hidden_channels,
            policy_mode=policy_mode,
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
        z = self.nopost_lowband_policy(z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs


def build_net(version, data, fam_mode="original", hidden_channels=32, policy_mode="global"):
    if fam_mode != "original":
        raise ValueError("NoPost lowband policy starts from official ConvIR-B original FAM.")
    return NoPostLowbandPolicyConvIR(
        version,
        data,
        hidden_channels=hidden_channels,
        policy_mode=policy_mode,
    )
