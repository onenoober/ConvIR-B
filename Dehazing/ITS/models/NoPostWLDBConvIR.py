
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


class WaveletLowbandDecoderBlock(nn.Module):
    def __init__(self, channels, hidden_channels=16):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.iwt = HaarIWT2D()
        self.lowband_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )
        self.lowband_project = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.lowband_project.weight)
        nn.init.zeros_(self.lowband_project.bias)

    def forward(self, z):
        ll, lh, hl, hh, h, w = self.dwt(z)
        ll_delta = self.lowband_project(self.lowband_context(ll))
        ll = ll + ll_delta
        return self.iwt(ll, lh, hl, hh, h, w)


class NoPostWLDBConvIR(ConvIR):
    def __init__(self, version, data, hidden_channels=16):
        super().__init__(version, data)
        self.nopost_wldb = WaveletLowbandDecoderBlock(32, hidden_channels=hidden_channels)

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
        z = self.nopost_wldb(z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs


def build_net(version, data, fam_mode="original", hidden_channels=16):
    if fam_mode != "original":
        raise ValueError("NoPost-WLDB starts from the official ConvIR-B original FAM path.")
    return NoPostWLDBConvIR(version, data, hidden_channels=hidden_channels)
