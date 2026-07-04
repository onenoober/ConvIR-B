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


class MidLowbandPolicyBlock(nn.Module):
    def __init__(self, channels, hidden_channels=32):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, ll):
        return self.project(self.context(ll))


class FinalContextLowbandPolicyBlock(nn.Module):
    def __init__(self, final_channels, mid_channels, hidden_channels=32):
        super().__init__()
        self.mid_project = nn.Conv2d(mid_channels, hidden_channels, kernel_size=1, bias=True)
        self.global_project = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(final_channels + mid_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(final_channels + hidden_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.project = nn.Conv2d(hidden_channels, final_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, final_ll, mid_feature):
        mid_ctx = self.mid_project(mid_feature)
        mid_ctx = F.interpolate(mid_ctx, size=final_ll.shape[-2:], mode="bilinear", align_corners=False)
        final_global = F.adaptive_avg_pool2d(final_ll, 1)
        mid_global = F.adaptive_avg_pool2d(mid_feature, 1)
        global_ctx = self.global_project(torch.cat([final_global, mid_global], dim=1))
        global_ctx = global_ctx.expand(-1, -1, final_ll.shape[-2], final_ll.shape[-1])
        return self.project(self.context(torch.cat([final_ll, mid_ctx, global_ctx], dim=1)))


class MidFinalContextLowbandPolicy(nn.Module):
    def __init__(self, mid_channels=64, final_channels=32, hidden_channels=32):
        super().__init__()
        self.mid_policy = MidLowbandPolicyBlock(mid_channels, hidden_channels)
        self.final_policy = FinalContextLowbandPolicyBlock(final_channels, mid_channels, hidden_channels)
        self.dwt = HaarDWT2D()
        self.iwt = HaarIWT2D()

    def apply_lowband(self, z, ll_delta):
        ll, lh, hl, hh, h, w = self.dwt(z)
        if ll_delta.shape[-2:] != ll.shape[-2:]:
            ll_delta = F.interpolate(ll_delta, size=ll.shape[-2:], mode="bilinear", align_corners=False)
        zero_lh = torch.zeros_like(lh)
        zero_hl = torch.zeros_like(hl)
        zero_hh = torch.zeros_like(hh)
        delta = self.iwt(ll_delta, zero_lh, zero_hl, zero_hh, h, w)
        return z + delta

    def forward_mid(self, mid):
        ll, _, _, _, _, _ = self.dwt(mid)
        return self.apply_lowband(mid, self.mid_policy(ll))

    def forward_final(self, final, mid_context):
        ll, _, _, _, _, _ = self.dwt(final)
        return self.apply_lowband(final, self.final_policy(ll, mid_context))


class NoPostMidFinalContextLowbandConvIR(ConvIR):
    def __init__(self, version, data, hidden_channels=32):
        super().__init__(version, data)
        self.nopost_midfinal_context_policy = MidFinalContextLowbandPolicy(
            mid_channels=64,
            final_channels=32,
            hidden_channels=hidden_channels,
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
        z = self.nopost_midfinal_context_policy.forward_mid(z)
        mid_context = z
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.nopost_midfinal_context_policy.forward_final(z, mid_context)
        z = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs


def build_net(version, data, fam_mode="original", hidden_channels=32):
    if fam_mode != "original":
        raise ValueError("NoPost mid/final context lowband policy starts from official ConvIR-B original FAM.")
    return NoPostMidFinalContextLowbandConvIR(
        version,
        data,
        hidden_channels=hidden_channels,
    )
