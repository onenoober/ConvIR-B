
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


def _init_kaiming(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DepthFiLMBlock(nn.Module):
    def __init__(
        self,
        channels,
        prior_channels=16,
        gate_bias=-6.0,
        gate_limit=0.05,
        gamma_limit=0.10,
        beta_limit=0.05,
    ):
        super(DepthFiLMBlock, self).__init__()
        self.channels = channels
        self.gate_limit = gate_limit
        self.gamma_limit = gamma_limit
        self.beta_limit = beta_limit
        self.prior = nn.Sequential(
            nn.Conv2d(2, prior_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(prior_channels, prior_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.head = nn.Conv2d(prior_channels, channels * 2 + 1, kernel_size=1, bias=True)
        self.prior.apply(_init_kaiming)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[channels * 2].fill_(gate_bias)

    def forward(self, feat, prior):
        prior = F.interpolate(prior, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        film = self.head(self.prior(prior))
        gamma, beta, gate = torch.split(film, [self.channels, self.channels, 1], dim=1)
        gamma = torch.tanh(gamma) * self.gamma_limit
        beta = torch.tanh(beta) * self.beta_limit
        gate = torch.sigmoid(gate) * self.gate_limit
        delta = gate * (gamma * feat + beta)
        stats = {
            'gate_mean': gate.detach().mean(),
            'gate_max': gate.detach().max(),
            'gamma_abs_mean': gamma.detach().abs().mean(),
            'beta_abs_mean': beta.detach().abs().mean(),
            'delta_abs_mean': delta.detach().abs().mean(),
        }
        return feat + delta, stats


class DepthTransmissionAdapter(nn.Module):
    def __init__(
        self,
        stage2_channels=64,
        stage3_channels=128,
        prior_channels=16,
        gate_bias=-6.0,
        gate_limit=0.05,
        gamma_limit=0.10,
        beta_limit=0.05,
        alpha_init=1.0,
    ):
        super(DepthTransmissionAdapter, self).__init__()
        self.stage2 = DepthFiLMBlock(
            stage2_channels,
            prior_channels=prior_channels,
            gate_bias=gate_bias,
            gate_limit=gate_limit,
            gamma_limit=gamma_limit,
            beta_limit=beta_limit,
        )
        self.stage3 = DepthFiLMBlock(
            stage3_channels,
            prior_channels=prior_channels,
            gate_bias=gate_bias,
            gate_limit=gate_limit,
            gamma_limit=gamma_limit,
            beta_limit=beta_limit,
        )
        self.log_alpha2 = nn.Parameter(torch.log(torch.tensor(float(alpha_init))))
        self.log_alpha3 = nn.Parameter(torch.log(torch.tensor(float(alpha_init))))
        hidden = max(16, stage2_channels // 2)
        self.transmission_head = nn.Sequential(
            nn.Conv2d(stage2_channels + stage3_channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1, bias=True),
        )
        _init_kaiming(self.transmission_head[0])
        nn.init.zeros_(self.transmission_head[-1].weight)
        nn.init.zeros_(self.transmission_head[-1].bias)
        self.last_aux = {}
        self.last_stats = {}

    @staticmethod
    def normalize_depth(depth):
        if depth is None:
            return None
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.size(1) != 1:
            depth = depth.mean(dim=1, keepdim=True)
        depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
        flat = depth.flatten(2)
        d_min = flat.amin(dim=2).view(depth.size(0), 1, 1, 1)
        d_max = flat.amax(dim=2).view(depth.size(0), 1, 1, 1)
        depth = (depth - d_min) / (d_max - d_min + 1e-6)
        return depth.clamp(0.0, 1.0)

    def _prior(self, depth, size, log_alpha):
        d = F.interpolate(depth, size=size, mode='bilinear', align_corners=False)
        alpha = log_alpha.exp().clamp(0.05, 5.0)
        t_proxy = torch.exp(-alpha * d)
        return torch.cat([d, t_proxy], dim=1), d, t_proxy

    def forward(self, feat2, feat3, depth):
        self.last_aux = {}
        self.last_stats = {}
        depth = self.normalize_depth(depth)
        if depth is None:
            return feat2, feat3

        prior2, d2, t_proxy2 = self._prior(depth, feat2.shape[-2:], self.log_alpha2)
        prior3, d3, t_proxy3 = self._prior(depth, feat3.shape[-2:], self.log_alpha3)
        feat2, stats2 = self.stage2(feat2, prior2)
        feat3, stats3 = self.stage3(feat3, prior3)
        feat3_up = F.interpolate(feat3, size=feat2.shape[-2:], mode='bilinear', align_corners=False)
        t_pred = torch.sigmoid(self.transmission_head(torch.cat([feat2, feat3_up], dim=1)))
        self.last_aux = {
            't_pred': t_pred,
            'depth': d2,
            't_proxy': t_proxy2,
            'depth_stage3': d3,
            't_proxy_stage3': t_proxy3,
        }
        self.last_stats = {
            'stage2_gate_mean': stats2['gate_mean'],
            'stage2_gate_max': stats2['gate_max'],
            'stage2_gamma_abs_mean': stats2['gamma_abs_mean'],
            'stage2_beta_abs_mean': stats2['beta_abs_mean'],
            'stage2_delta_abs_mean': stats2['delta_abs_mean'],
            'stage3_gate_mean': stats3['gate_mean'],
            'stage3_gate_max': stats3['gate_max'],
            'stage3_gamma_abs_mean': stats3['gamma_abs_mean'],
            'stage3_beta_abs_mean': stats3['beta_abs_mean'],
            'stage3_delta_abs_mean': stats3['delta_abs_mean'],
            't_pred_mean': t_pred.detach().mean(),
            't_pred_std': t_pred.detach().std(unbiased=False),
        }
        return feat2, feat3

    def auxiliary_losses(self, rank_pairs=512, min_depth_gap=0.03):
        if not self.last_aux:
            device = next(self.parameters()).device
            zero = torch.zeros((), device=device)
            return {'rank': zero, 'tv': zero, 'proxy': zero}
        t_pred = self.last_aux['t_pred']
        depth = self.last_aux['depth'].detach()
        t_proxy = self.last_aux['t_proxy'].detach()
        rank = self._rank_loss(t_pred, depth, rank_pairs, min_depth_gap)
        tv = self._edge_aware_tv(t_pred, depth)
        proxy = F.l1_loss(t_pred, t_proxy)
        return {'rank': rank, 'tv': tv, 'proxy': proxy}

    @staticmethod
    def _rank_loss(t_pred, depth, rank_pairs, min_depth_gap):
        b, _, h, w = t_pred.shape
        n = h * w
        if rank_pairs <= 0 or n < 2:
            return t_pred.new_zeros(())
        pair_count = min(rank_pairs, n)
        t_flat = t_pred.flatten(2)
        d_flat = depth.flatten(2)
        idx_i = torch.randint(0, n, (b, pair_count), device=t_pred.device)
        idx_j = torch.randint(0, n, (b, pair_count), device=t_pred.device)
        ti = torch.gather(t_flat[:, 0], 1, idx_i)
        tj = torch.gather(t_flat[:, 0], 1, idx_j)
        di = torch.gather(d_flat[:, 0], 1, idx_i)
        dj = torch.gather(d_flat[:, 0], 1, idx_j)
        d_diff = di - dj
        keep = d_diff.abs() >= min_depth_gap
        if not bool(keep.any()):
            return t_pred.new_zeros(())
        signed_t = torch.sign(d_diff[keep]) * (ti[keep] - tj[keep])
        return F.softplus(signed_t).mean()

    @staticmethod
    def _edge_aware_tv(t_pred, depth):
        dx_t = (t_pred[:, :, :, 1:] - t_pred[:, :, :, :-1]).abs()
        dy_t = (t_pred[:, :, 1:, :] - t_pred[:, :, :-1, :]).abs()
        dx_d = (depth[:, :, :, 1:] - depth[:, :, :, :-1]).abs()
        dy_d = (depth[:, :, 1:, :] - depth[:, :, :-1, :]).abs()
        loss_x = (dx_t * torch.exp(-5.0 * dx_d)).mean()
        loss_y = (dy_t * torch.exp(-5.0 * dy_d)).mean()
        return loss_x + loss_y

    def stats(self):
        return {key: float(value.cpu()) for key, value in self.last_stats.items()}


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


class ConvIRDTA(ConvIR):
    def __init__(self, version, data, **dta_kwargs):
        super(ConvIRDTA, self).__init__(version, data)
        self.DTA = DepthTransmissionAdapter(
            stage2_channels=64,
            stage3_channels=128,
            **dta_kwargs,
        )

    def forward(self, x, depth=None):
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
        res2, z = self.DTA(res2, z, depth)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs

    def dta_auxiliary_losses(self, rank_pairs=512, min_depth_gap=0.03):
        return self.DTA.auxiliary_losses(rank_pairs, min_depth_gap)

    def collect_dta_stats(self, x, depth=None):
        self.eval()
        with torch.no_grad():
            h, w = x.shape[2], x.shape[3]
            factor = 32
            padded_h = ((h + factor) // factor) * factor
            padded_w = ((w + factor) // factor) * factor
            padh = padded_h - h if h % factor != 0 else 0
            padw = padded_w - w if w % factor != 0 else 0
            if padh or padw:
                x = F.pad(x, (0, padw, 0, padh), 'reflect')
                if depth is not None:
                    depth = F.pad(depth, (0, padw, 0, padh), 'reflect')
            self.forward(x, depth)
        return self.DTA.stats()


def build_net(
    version,
    data,
    fam_mode='original',
    arch='official_convir',
    dta_prior_channels=16,
    dta_gate_bias=-6.0,
    dta_gate_limit=0.05,
    dta_gamma_limit=0.10,
    dta_beta_limit=0.05,
    dta_alpha_init=1.0,
):
    if fam_mode != 'original':
        raise ValueError(
            "Official ConvIR-B anchor only supports fam_mode='original'. "
            "Create a route branch for architecture variants."
        )
    if arch in ('official_convir', 'convir'):
        return ConvIR(version, data)
    if arch == 'dta':
        return ConvIRDTA(
            version,
            data,
            prior_channels=dta_prior_channels,
            gate_bias=dta_gate_bias,
            gate_limit=dta_gate_limit,
            gamma_limit=dta_gamma_limit,
            beta_limit=dta_beta_limit,
            alpha_init=dta_alpha_init,
        )
    raise ValueError(f'Unsupported arch: {arch}')
