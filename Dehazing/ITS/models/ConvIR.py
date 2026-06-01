
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


def build_haze_prior_maps(x):
    min_rgb = x.min(dim=1, keepdim=True)[0]
    max_rgb = x.max(dim=1, keepdim=True)[0]
    dark = -F.max_pool2d(-min_rgb, kernel_size=15, stride=1, padding=7)
    saturation = max_rgb - min_rgb

    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    grad_x = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    grad_y = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    grad_x = F.pad(grad_x.abs(), (0, 1, 0, 0))
    grad_y = F.pad(grad_y.abs(), (0, 0, 0, 1))
    grad = grad_x + grad_y

    return torch.cat([min_rgb, max_rgb, dark, saturation, grad], dim=1)


class HazePriorSCM(nn.Module):
    def __init__(self, out_plane):
        super(HazePriorSCM, self).__init__()
        self.rgb_scm = SCM(out_plane)

        rng_state = torch.get_rng_state()
        self.prior_branch = nn.Sequential(
            BasicConv(5, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
        )
        torch.set_rng_state(rng_state)
        final_conv = self.prior_branch[-1].main[0]
        nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    def forward(self, x):
        prior = build_haze_prior_maps(x)
        return self.rgb_scm(x) + self.prior_branch(prior)

    def prior_branch_stats(self, x):
        with torch.no_grad():
            prior = build_haze_prior_maps(x)
            prior_feat = self.prior_branch(prior)
            rgb_feat = self.rgb_scm(x)
            return {
                'prior_min_mean': prior[:, 0:1].mean().item(),
                'prior_max_mean': prior[:, 1:2].mean().item(),
                'prior_dark_mean': prior[:, 2:3].mean().item(),
                'prior_saturation_mean': prior[:, 3:4].mean().item(),
                'prior_grad_mean': prior[:, 4:5].mean().item(),
                'rgb_abs_mean': rgb_feat.abs().mean().item(),
                'prior_branch_abs_mean': prior_feat.abs().mean().item(),
                'prior_to_rgb_abs_ratio': (
                    prior_feat.abs().mean() / rgb_feat.abs().mean().clamp_min(1e-6)
                ).item(),
            }

class FAM(nn.Module):
    def __init__(self, channel, mode='original'):
        super(FAM, self).__init__()
        if mode not in (
            'original',
            'modres',
            'modres_bounded',
            'modres_gamma_bounded',
            'modres_gamma_conf_gated',
        ):
            raise ValueError(f'Unsupported FAM mode: {mode}')
        self.mode = mode
        self.gamma_max = 0.10
        self.beta_max = 0.05
        self._last_gate = None
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)
        if self.mode != 'original':
            out_channel = channel if self.mode in (
                'modres_gamma_bounded',
                'modres_gamma_conf_gated',
            ) else channel * 2
            rng_state = torch.get_rng_state()
            self.modulator = nn.Conv2d(channel, out_channel, kernel_size=1, stride=1, padding=0)
            torch.set_rng_state(rng_state)
            nn.init.zeros_(self.modulator.weight)
            nn.init.zeros_(self.modulator.bias)
        if self.mode == 'modres_gamma_conf_gated':
            rng_state = torch.get_rng_state()
            self.gate_head = nn.Linear(channel * 4, 1)
            torch.set_rng_state(rng_state)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.zeros_(self.gate_head.bias)

    def _confidence_gate(self, x2, fused):
        cond = x2.detach()
        base = fused.detach()
        cond_mean = cond.mean(dim=(2, 3))
        cond_std = cond.std(dim=(2, 3), unbiased=False)
        base_mean = base.mean(dim=(2, 3))
        base_std = base.std(dim=(2, 3), unbiased=False)
        descriptor = torch.cat([cond_mean, cond_std, base_mean, base_std], dim=1)
        gate = torch.sigmoid(self.gate_head(descriptor)).view(-1, 1, 1, 1)
        return gate

    def _modulation(self, x2, fused):
        if self.mode == 'original':
            return None, None, {}
        if self.mode == 'modres_gamma_bounded':
            gamma_raw = self.modulator(x2)
            gamma = self.gamma_max * torch.tanh(gamma_raw)
            return gamma, None, {}
        if self.mode == 'modres_gamma_conf_gated':
            gamma_raw = self.modulator(x2)
            gamma_base = self.gamma_max * torch.tanh(gamma_raw)
            gate = self._confidence_gate(x2, fused)
            gamma = gate * gamma_base
            return gamma, None, {
                'gate': gate,
                'gamma_base': gamma_base,
            }

        gamma_raw, beta_raw = self.modulator(x2).chunk(2, dim=1)
        if self.mode == 'modres':
            return gamma_raw, beta_raw, {}

        gamma = self.gamma_max * torch.tanh(gamma_raw)
        scale = fused.detach().std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        beta = self.beta_max * scale * torch.tanh(beta_raw)
        return gamma, beta, {}

    def forward(self, x1, x2):
        fused = self.merge(torch.cat([x1, x2], dim=1))
        self._last_gate = None
        if self.mode == 'original':
            return fused
        gamma, beta, extra = self._modulation(x2, fused)
        self._last_gate = extra.get('gate')
        if beta is None:
            return fused * (1 + gamma)
        return fused * (1 + gamma) + beta

    def gate_budget_loss(self, easy_weight):
        if self._last_gate is None:
            return None
        weight = easy_weight.to(device=self._last_gate.device, dtype=self._last_gate.dtype)
        weight = weight.view(-1, 1, 1, 1)
        return (weight * self._last_gate).mean()

    def modulation_stats(self, x1, x2):
        if self.mode == 'original':
            return None
        with torch.no_grad():
            fused = self.merge(torch.cat([x1, x2], dim=1))
            gamma, beta, extra = self._modulation(x2, fused)
            stats = {
                'beta_present': 0.0 if beta is None else 1.0,
                'gamma_mean': gamma.mean().item(),
                'gamma_abs_mean': gamma.abs().mean().item(),
                'gamma_std': gamma.std(unbiased=False).item(),
                'gamma_min': gamma.min().item(),
                'gamma_max': gamma.max().item(),
                'gamma_abs_gt_0.5': (gamma.abs() > 0.5).float().mean().item(),
                'gamma_abs_gt_0.05': (gamma.abs() > 0.05).float().mean().item(),
                'gamma_abs_gt_0.10': (gamma.abs() > 0.10).float().mean().item(),
                'gamma_abs_gt_0.09': (gamma.abs() > 0.09).float().mean().item(),
            }
            gate = extra.get('gate')
            gamma_base = extra.get('gamma_base')
            if gate is None:
                stats.update({
                    'gate_mean': 0.0,
                    'gate_std': 0.0,
                    'gate_min': 0.0,
                    'gate_max': 0.0,
                    'gamma_base_abs_mean': gamma.abs().mean().item(),
                    'effective_gamma_abs_mean': gamma.abs().mean().item(),
                })
            else:
                stats.update({
                    'gate_mean': gate.mean().item(),
                    'gate_std': gate.std(unbiased=False).item(),
                    'gate_min': gate.min().item(),
                    'gate_max': gate.max().item(),
                    'gamma_base_abs_mean': gamma_base.abs().mean().item(),
                    'effective_gamma_abs_mean': gamma.abs().mean().item(),
                })
            if beta is None:
                stats.update({
                    'beta_mean': 0.0,
                    'beta_abs_mean': 0.0,
                    'beta_std': 0.0,
                    'beta_min': 0.0,
                    'beta_max': 0.0,
                    'beta_abs_gt_0.1': 0.0,
                    'beta_abs_gt_0.02': 0.0,
                    'beta_abs_gt_0.05': 0.0,
                })
            else:
                stats.update({
                    'beta_mean': beta.mean().item(),
                    'beta_abs_mean': beta.abs().mean().item(),
                    'beta_std': beta.std(unbiased=False).item(),
                    'beta_min': beta.min().item(),
                    'beta_max': beta.max().item(),
                    'beta_abs_gt_0.1': (beta.abs() > 0.1).float().mean().item(),
                    'beta_abs_gt_0.02': (beta.abs() > 0.02).float().mean().item(),
                    'beta_abs_gt_0.05': (beta.abs() > 0.05).float().mean().item(),
                })
            return stats

class ConvIR(nn.Module):
    def __init__(self, version, data, fam_mode='original', scm_mode='original'):
        super(ConvIR, self).__init__()
        if fam_mode not in (
            'original',
            'modres',
            'fam2_modres',
            'fam2_modres_bounded',
            'fam2_modres_gamma_bounded',
            'fam2_modres_gamma_conf_gated',
        ):
            raise ValueError(f'Unsupported ConvIR FAM mode: {fam_mode}')
        if scm_mode not in ('original', 'haze_prior'):
            raise ValueError(f'Unsupported ConvIR SCM mode: {scm_mode}')
        
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

        fam2_modes = {
            'fam2_modres': 'modres',
            'fam2_modres_bounded': 'modres_bounded',
            'fam2_modres_gamma_bounded': 'modres_gamma_bounded',
            'fam2_modres_gamma_conf_gated': 'modres_gamma_conf_gated',
        }
        fam1_mode = 'original' if fam_mode in fam2_modes else fam_mode
        fam2_mode = fam2_modes.get(fam_mode, fam_mode)
        scm_cls = HazePriorSCM if scm_mode == 'haze_prior' else SCM

        self.FAM1 = FAM(base_channel * 4, fam1_mode)
        self.SCM1 = scm_cls(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2, fam2_mode)
        self.SCM2 = scm_cls(base_channel * 2)

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

    def collect_modulation_stats(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        stats = {}
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        z = self.feat_extract[1](res1)
        fam2_stats = self.FAM2.modulation_stats(z, z2)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        z = self.feat_extract[2](res2)
        fam1_stats = self.FAM1.modulation_stats(z, z4)
        if fam1_stats is not None:
            stats['FAM1'] = fam1_stats
        if fam2_stats is not None:
            stats['FAM2'] = fam2_stats
        return stats

    def gate_budget_loss(self, easy_weight):
        losses = []
        for fam in (self.FAM1, self.FAM2):
            loss = fam.gate_budget_loss(easy_weight)
            if loss is not None:
                losses.append(loss)
        if not losses:
            return None
        return sum(losses) / len(losses)

    def collect_scm_stats(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        stats = {}
        if hasattr(self.SCM1, 'prior_branch_stats'):
            stats['SCM1'] = self.SCM1.prior_branch_stats(x_4)
        if hasattr(self.SCM2, 'prior_branch_stats'):
            stats['SCM2'] = self.SCM2.prior_branch_stats(x_2)
        return stats


def build_net(version, data, fam_mode='original', scm_mode='original'):
    return ConvIR(version, data, fam_mode, scm_mode)
