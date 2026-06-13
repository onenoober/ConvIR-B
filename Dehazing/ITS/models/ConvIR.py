
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


class CalibratedDepthFiLMBlock(nn.Module):
    def __init__(
        self,
        channels,
        prior_in_channels=6,
        prior_channels=32,
        gate_bias=-6.0,
        gate_limit=0.06,
        gamma_limit=0.12,
        beta_limit=0.06,
    ):
        super(CalibratedDepthFiLMBlock, self).__init__()
        self.channels = channels
        self.gate_limit = gate_limit
        self.gamma_limit = gamma_limit
        self.beta_limit = beta_limit
        self.prior = nn.Sequential(
            nn.Conv2d(prior_in_channels, prior_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(prior_channels, prior_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.head = nn.Conv2d(prior_channels, channels * 2 + 2, kernel_size=1, bias=True)
        self.prior.apply(_init_kaiming)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[channels * 2].fill_(gate_bias)

    def forward(self, feat, prior):
        prior = F.interpolate(prior, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        film = self.head(self.prior(prior))
        gamma, beta, gate, conf_logit = torch.split(
            film, [self.channels, self.channels, 1, 1], dim=1
        )
        gamma = torch.tanh(gamma) * self.gamma_limit
        beta = torch.tanh(beta) * self.beta_limit
        prior_conf = prior[:, -1:, :, :].clamp(0.0, 1.0)
        conf = torch.sigmoid(conf_logit) * prior_conf
        gate = torch.sigmoid(gate) * self.gate_limit * conf
        delta = gate * (gamma * feat + beta)
        stats = {
            'gate_mean': gate.detach().mean(),
            'gate_max': gate.detach().max(),
            'conf_mean': conf.detach().mean(),
            'conf_min': conf.detach().amin(),
            'gamma_abs_mean': gamma.detach().abs().mean(),
            'beta_abs_mean': beta.detach().abs().mean(),
            'delta_abs_mean': delta.detach().abs().mean(),
        }
        return feat + delta, stats


class CalibratedDepthTransmissionAdapter(nn.Module):
    def __init__(
        self,
        stage1_channels=32,
        stage2_channels=64,
        stage3_channels=128,
        prior_channels=32,
        gate_bias=-6.0,
        gate_limit=0.06,
        gamma_limit=0.12,
        beta_limit=0.06,
        alpha_init=1.0,
        depth_mode='normal',
        confidence_floor=0.25,
        confidence_local_scale=6.0,
        output_residual_scale=0.03,
    ):
        super(CalibratedDepthTransmissionAdapter, self).__init__()
        self.depth_mode = depth_mode
        self.confidence_floor = confidence_floor
        self.confidence_local_scale = confidence_local_scale
        self.output_residual_scale = output_residual_scale
        prior_in_channels = 6
        self.stage2 = CalibratedDepthFiLMBlock(
            stage2_channels,
            prior_in_channels=prior_in_channels,
            prior_channels=prior_channels,
            gate_bias=gate_bias,
            gate_limit=gate_limit,
            gamma_limit=gamma_limit,
            beta_limit=beta_limit,
        )
        self.stage3 = CalibratedDepthFiLMBlock(
            stage3_channels,
            prior_in_channels=prior_in_channels,
            prior_channels=prior_channels,
            gate_bias=gate_bias,
            gate_limit=gate_limit,
            gamma_limit=gamma_limit,
            beta_limit=beta_limit,
        )
        self.log_alpha2 = nn.Parameter(torch.log(torch.tensor(float(alpha_init))))
        self.log_alpha3 = nn.Parameter(torch.log(torch.tensor(float(alpha_init))))
        self.log_alpha_full = nn.Parameter(torch.log(torch.tensor(float(alpha_init))))
        hidden = max(32, stage2_channels)
        self.transmission_head = nn.Sequential(
            nn.Conv2d(stage2_channels + stage3_channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=3, padding=1, bias=True),
        )
        self.output_refine = nn.Sequential(
            nn.Conv2d(stage1_channels + prior_in_channels, stage1_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stage1_channels, 3, kernel_size=3, padding=1, bias=True),
        )
        self.transmission_head[0].apply(_init_kaiming)
        self.transmission_head[2].apply(_init_kaiming)
        nn.init.zeros_(self.transmission_head[-1].weight)
        nn.init.zeros_(self.transmission_head[-1].bias)
        self.output_refine[0].apply(_init_kaiming)
        nn.init.zeros_(self.output_refine[-1].weight)
        nn.init.zeros_(self.output_refine[-1].bias)
        self.last_aux = {}
        self.last_stats = {}

    def normalize_depth(self, depth):
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
        depth = depth.clamp(0.0, 1.0)
        if self.depth_mode == 'invert':
            depth = 1.0 - depth
        elif self.depth_mode == 'zero':
            depth = torch.zeros_like(depth)
        elif self.depth_mode not in ('normal', 'shuffle'):
            raise ValueError(f'Unsupported DTA depth_mode: {self.depth_mode}')
        return depth

    def _confidence(self, d):
        local = F.avg_pool2d(d, kernel_size=5, stride=1, padding=2)
        inconsistency = (d - local).abs()
        conf = torch.exp(-self.confidence_local_scale * inconsistency)
        return self.confidence_floor + (1.0 - self.confidence_floor) * conf

    @staticmethod
    def _gradients(d):
        dx = F.pad(d[:, :, :, 1:] - d[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(d[:, :, 1:, :] - d[:, :, :-1, :], (0, 0, 0, 1))
        return dx, dy

    def _prior(self, depth, size, log_alpha):
        d = F.interpolate(depth, size=size, mode='bilinear', align_corners=False)
        alpha = log_alpha.exp().clamp(0.05, 5.0)
        t_proxy = torch.exp(-alpha * d).clamp(1e-4, 1.0)
        neg_log_t = -torch.log(t_proxy)
        dx, dy = self._gradients(d)
        conf = self._confidence(d)
        prior = torch.cat([d, t_proxy, neg_log_t, dx, dy, conf], dim=1)
        return prior, d, t_proxy, conf

    def forward(self, feat2, feat3, depth):
        self.last_aux = {}
        self.last_stats = {}
        depth = self.normalize_depth(depth)
        if depth is None:
            return feat2, feat3

        prior2, d2, t_proxy2, conf2 = self._prior(depth, feat2.shape[-2:], self.log_alpha2)
        prior3, d3, t_proxy3, conf3 = self._prior(depth, feat3.shape[-2:], self.log_alpha3)
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
            'confidence': conf2,
            'confidence_stage3': conf3,
            'depth_full': depth,
        }
        self.last_stats = {
            'stage2_gate_mean': stats2['gate_mean'],
            'stage2_gate_max': stats2['gate_max'],
            'stage2_conf_mean': stats2['conf_mean'],
            'stage2_conf_min': stats2['conf_min'],
            'stage2_gamma_abs_mean': stats2['gamma_abs_mean'],
            'stage2_beta_abs_mean': stats2['beta_abs_mean'],
            'stage2_delta_abs_mean': stats2['delta_abs_mean'],
            'stage3_gate_mean': stats3['gate_mean'],
            'stage3_gate_max': stats3['gate_max'],
            'stage3_conf_mean': stats3['conf_mean'],
            'stage3_conf_min': stats3['conf_min'],
            'stage3_gamma_abs_mean': stats3['gamma_abs_mean'],
            'stage3_beta_abs_mean': stats3['beta_abs_mean'],
            'stage3_delta_abs_mean': stats3['delta_abs_mean'],
            't_pred_mean': t_pred.detach().mean(),
            't_pred_std': t_pred.detach().std(unbiased=False),
        }
        return feat2, feat3

    def refine_output(self, final_feat, output, depth):
        depth = self.normalize_depth(depth)
        if depth is None:
            return output
        prior, _, _, _ = self._prior(depth, output.shape[-2:], self.log_alpha_full)
        delta = torch.tanh(self.output_refine(torch.cat([final_feat, prior], dim=1)))
        delta = delta * self.output_residual_scale
        self.last_stats['output_delta_abs_mean'] = delta.detach().abs().mean()
        return output + delta

    def auxiliary_losses(self, rank_pairs=512, min_depth_gap=0.03):
        if not self.last_aux:
            device = next(self.parameters()).device
            zero = torch.zeros((), device=device)
            return {'rank': zero, 'tv': zero, 'proxy': zero}
        t_pred = self.last_aux['t_pred']
        depth = self.last_aux['depth'].detach()
        t_proxy = self.last_aux['t_proxy'].detach()
        rank = DepthTransmissionAdapter._rank_loss(t_pred, depth, rank_pairs, min_depth_gap)
        tv = DepthTransmissionAdapter._edge_aware_tv(t_pred, depth)
        proxy = F.l1_loss(t_pred, t_proxy)
        return {'rank': rank, 'tv': tv, 'proxy': proxy}

    def supervised_losses(self, trans_gt=None, hazy=None, dehazed=None, airlight=None):
        if not self.last_aux:
            device = next(self.parameters()).device
            zero = torch.zeros((), device=device)
            return {'trans': zero, 'phys': zero, 't_l1': zero, 't_log_l1': zero, 't_nll': zero, 't_spearman_proxy': zero}
        t_pred = self.last_aux['t_pred']
        zero = t_pred.new_zeros(())
        losses = {'trans': zero, 'phys': zero, 't_l1': zero, 't_log_l1': zero, 't_nll': zero, 't_spearman_proxy': zero}
        if trans_gt is None:
            return losses
        if trans_gt.dim() == 3:
            trans_gt = trans_gt.unsqueeze(1)
        trans_gt = trans_gt.float().clamp(1e-4, 1.0)
        trans = F.interpolate(trans_gt, size=t_pred.shape[-2:], mode='bilinear', align_corners=False)
        t_l1 = F.smooth_l1_loss(t_pred, trans)
        losses['trans'] = t_l1
        losses['t_l1'] = t_l1.detach()
        log_t_error = (torch.log(t_pred.clamp(1e-4, 1.0)) - torch.log(trans.clamp(1e-4, 1.0))).abs()
        losses['t_log_l1'] = log_t_error.mean()
        t_log_var = self.last_aux.get('t_log_var')
        if t_log_var is not None:
            t_log_var = t_log_var.clamp(-6.0, 6.0)
            losses['t_nll'] = (torch.exp(-t_log_var) * log_t_error.detach() + t_log_var).mean()
        if hazy is not None and dehazed is not None and airlight is not None:
            if airlight.dim() == 1:
                airlight = airlight.view(-1, 1, 1, 1)
            elif airlight.dim() == 2:
                airlight = airlight.view(airlight.size(0), airlight.size(1), 1, 1)
            airlight = airlight.to(hazy.device).float().clamp(0.0, 1.0)
            if airlight.size(1) == 1:
                airlight = airlight.expand(-1, 3, -1, -1)
            trans_full = F.interpolate(t_pred, size=hazy.shape[-2:], mode='bilinear', align_corners=False)
            recon_hazy = dehazed.clamp(0.0, 1.0) * trans_full + airlight * (1.0 - trans_full)
            losses['phys'] = F.smooth_l1_loss(recon_hazy, hazy)
        return losses

    def stats(self):
        return {key: float(value.cpu()) for key, value in self.last_stats.items()}


class DepthAttributedPreserveAdapter(nn.Module):
    def __init__(
        self,
        stage1_channels=32,
        stage2_channels=64,
        stage3_channels=128,
        prior_channels=32,
        gate_bias=-5.0,
        gate_limit=0.10,
        gamma_limit=0.16,
        beta_limit=0.08,
        alpha_init=1.0,
        depth_mode='invert',
        confidence_floor=0.30,
        confidence_local_scale=6.0,
        r0_residual_scale=0.04,
        depth_residual_scale=0.08,
        depth_mask_easy_budget=0.04,
        depth_mask_dense_budget=0.12,
        depth_mask_density_thresh=0.35,
        depth_mask_bias=-4.0,
        phys_t_min=0.10,
        phase='joint',
        ablation='full',
        safe_mix_enabled=False,
        safe_mix_delta_clip=0.08,
        safe_mix_phys_weight=1.0,
        safe_mix_learned_weight=0.0,
        safe_mix_gate_limit=1.0,
        safe_mix_gate_bias=-3.0,
        router_fusion_enabled=False,
        router_image_gate_limit=1.0,
        router_patch_gate_limit=1.0,
        router_patch_size=32,
        router_image_bias=2.0,
        router_patch_bias=2.0,
        feature_fusion_enabled=False,
        feature_fusion_strength=0.10,
        feature_fusion_gate_limit=1.0,
        feature_fusion_gate_bias=2.0,
    ):
        super(DepthAttributedPreserveAdapter, self).__init__()
        self.depth_mode = depth_mode
        self.confidence_floor = confidence_floor
        self.confidence_local_scale = confidence_local_scale
        self.r0_residual_scale = r0_residual_scale
        self.depth_residual_scale = depth_residual_scale
        self.depth_mask_easy_budget = depth_mask_easy_budget
        self.depth_mask_dense_budget = depth_mask_dense_budget
        self.depth_mask_density_thresh = depth_mask_density_thresh
        self.phys_t_min = phys_t_min
        self.phase = phase
        self.ablation = ablation
        self.safe_mix_enabled = safe_mix_enabled
        self.safe_mix_delta_clip = safe_mix_delta_clip
        self.safe_mix_phys_weight = safe_mix_phys_weight
        self.safe_mix_learned_weight = safe_mix_learned_weight
        self.safe_mix_gate_limit = safe_mix_gate_limit
        self.router_fusion_enabled = router_fusion_enabled
        self.router_image_gate_limit = router_image_gate_limit
        self.router_patch_gate_limit = router_patch_gate_limit
        self.router_patch_size = router_patch_size
        self.feature_fusion_enabled = feature_fusion_enabled
        self.feature_fusion_strength = feature_fusion_strength
        self.feature_fusion_gate_limit = feature_fusion_gate_limit
        prior_in_channels = 6

        self.stage2 = CalibratedDepthFiLMBlock(
            stage2_channels,
            prior_in_channels=prior_in_channels,
            prior_channels=prior_channels,
            gate_bias=gate_bias,
            gate_limit=gate_limit,
            gamma_limit=gamma_limit,
            beta_limit=beta_limit,
        )
        self.stage3 = CalibratedDepthFiLMBlock(
            stage3_channels,
            prior_in_channels=prior_in_channels,
            prior_channels=prior_channels,
            gate_bias=gate_bias,
            gate_limit=gate_limit,
            gamma_limit=gamma_limit,
            beta_limit=beta_limit,
        )
        self.log_alpha2 = nn.Parameter(torch.log(torch.tensor(float(alpha_init))))
        self.log_alpha3 = nn.Parameter(torch.log(torch.tensor(float(alpha_init))))
        self.log_alpha_full = nn.Parameter(torch.log(torch.tensor(float(alpha_init))))
        hidden = max(32, stage2_channels)
        self.transmission_head = nn.Sequential(
            nn.Conv2d(stage2_channels + stage3_channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=3, padding=1, bias=True),
        )
        self.trans_uncertainty_head = nn.Sequential(
            nn.Conv2d(stage2_channels + stage3_channels, hidden // 2, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=3, padding=1, bias=True),
        )
        self.airlight_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(stage2_channels + stage3_channels, hidden // 2, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1, bias=True),
        )
        self.airlight_uncertainty_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(stage2_channels + stage3_channels, hidden // 2, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1, bias=True),
        )
        self.r0_refine = nn.Sequential(
            nn.Conv2d(stage1_channels, stage1_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stage1_channels, 3, kernel_size=3, padding=1, bias=True),
        )
        mask_in_channels = stage1_channels + prior_in_channels + 4
        self.depth_mask_head = nn.Sequential(
            nn.Conv2d(mask_in_channels, stage1_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stage1_channels, 1, kernel_size=3, padding=1, bias=True),
        )
        safe_in_channels = stage1_channels + prior_in_channels + 6
        self.safe_residual_head = nn.Sequential(
            nn.Conv2d(safe_in_channels, stage1_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stage1_channels, 3, kernel_size=3, padding=1, bias=True),
        )
        self.safe_gate_head = nn.Sequential(
            nn.Conv2d(safe_in_channels, stage1_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stage1_channels, 1, kernel_size=3, padding=1, bias=True),
        )
        self.router_image_head = nn.Sequential(
            nn.Conv2d(safe_in_channels, stage1_channels // 2, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stage1_channels // 2, 1, kernel_size=1, bias=True),
        )
        self.router_patch_head = nn.Sequential(
            nn.Conv2d(safe_in_channels, stage1_channels // 2, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stage1_channels // 2, 1, kernel_size=3, padding=1, bias=True),
        )
        self.feature_fusion2 = nn.Sequential(
            nn.Conv2d(stage2_channels + prior_in_channels, stage2_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stage2_channels, stage2_channels, kernel_size=3, padding=1, bias=True),
        )
        self.feature_fusion2_gate = nn.Sequential(
            nn.Conv2d(stage2_channels + prior_in_channels, stage2_channels // 2, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stage2_channels // 2, 1, kernel_size=1, bias=True),
        )
        self.feature_fusion3 = nn.Sequential(
            nn.Conv2d(stage3_channels + prior_in_channels, stage3_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stage3_channels, stage3_channels, kernel_size=3, padding=1, bias=True),
        )
        self.feature_fusion3_gate = nn.Sequential(
            nn.Conv2d(stage3_channels + prior_in_channels, stage3_channels // 2, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stage3_channels // 2, 1, kernel_size=1, bias=True),
        )
        self.feature_fusion_final = nn.Sequential(
            nn.Conv2d(stage1_channels + prior_in_channels, stage1_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stage1_channels, stage1_channels, kernel_size=3, padding=1, bias=True),
        )
        self.feature_fusion_final_gate = nn.Sequential(
            nn.Conv2d(stage1_channels + prior_in_channels, stage1_channels // 2, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stage1_channels // 2, 1, kernel_size=1, bias=True),
        )

        self.transmission_head[0].apply(_init_kaiming)
        self.transmission_head[2].apply(_init_kaiming)
        nn.init.zeros_(self.transmission_head[-1].weight)
        nn.init.zeros_(self.transmission_head[-1].bias)
        self.trans_uncertainty_head[0].apply(_init_kaiming)
        nn.init.zeros_(self.trans_uncertainty_head[-1].weight)
        nn.init.zeros_(self.trans_uncertainty_head[-1].bias)
        self.airlight_head[1].apply(_init_kaiming)
        nn.init.zeros_(self.airlight_head[-1].weight)
        nn.init.zeros_(self.airlight_head[-1].bias)
        self.airlight_uncertainty_head[1].apply(_init_kaiming)
        nn.init.zeros_(self.airlight_uncertainty_head[-1].weight)
        nn.init.zeros_(self.airlight_uncertainty_head[-1].bias)
        self.r0_refine[0].apply(_init_kaiming)
        nn.init.zeros_(self.r0_refine[-1].weight)
        nn.init.zeros_(self.r0_refine[-1].bias)
        self.depth_mask_head[0].apply(_init_kaiming)
        nn.init.zeros_(self.depth_mask_head[-1].weight)
        nn.init.zeros_(self.depth_mask_head[-1].bias)
        self.safe_residual_head[0].apply(_init_kaiming)
        nn.init.zeros_(self.safe_residual_head[-1].weight)
        nn.init.zeros_(self.safe_residual_head[-1].bias)
        self.safe_gate_head[0].apply(_init_kaiming)
        nn.init.zeros_(self.safe_gate_head[-1].weight)
        nn.init.zeros_(self.safe_gate_head[-1].bias)
        self.router_image_head[0].apply(_init_kaiming)
        nn.init.zeros_(self.router_image_head[-1].weight)
        nn.init.zeros_(self.router_image_head[-1].bias)
        self.router_patch_head[0].apply(_init_kaiming)
        nn.init.zeros_(self.router_patch_head[-1].weight)
        nn.init.zeros_(self.router_patch_head[-1].bias)
        for head in (self.feature_fusion2, self.feature_fusion3, self.feature_fusion_final):
            head[0].apply(_init_kaiming)
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        for head in (self.feature_fusion2_gate, self.feature_fusion3_gate, self.feature_fusion_final_gate):
            head[0].apply(_init_kaiming)
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        with torch.no_grad():
            self.depth_mask_head[-1].bias.fill_(depth_mask_bias)
            self.safe_gate_head[-1].bias.fill_(safe_mix_gate_bias)
            self.router_image_head[-1].bias.fill_(router_image_bias)
            self.router_patch_head[-1].bias.fill_(router_patch_bias)
            self.feature_fusion2_gate[-1].bias.fill_(feature_fusion_gate_bias)
            self.feature_fusion3_gate[-1].bias.fill_(feature_fusion_gate_bias)
            self.feature_fusion_final_gate[-1].bias.fill_(feature_fusion_gate_bias)
        self.last_aux = {}
        self.last_stats = {}

    def normalize_depth(self, depth):
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
        depth = depth.clamp(0.0, 1.0)
        if self.depth_mode == 'invert':
            depth = 1.0 - depth
        elif self.depth_mode == 'zero':
            depth = torch.zeros_like(depth)
        elif self.depth_mode not in ('normal', 'shuffle'):
            raise ValueError(f'Unsupported DTA-v3 depth_mode: {self.depth_mode}')
        return depth

    def _confidence(self, d):
        local = F.avg_pool2d(d, kernel_size=5, stride=1, padding=2)
        inconsistency = (d - local).abs()
        conf = torch.exp(-self.confidence_local_scale * inconsistency)
        return self.confidence_floor + (1.0 - self.confidence_floor) * conf

    @staticmethod
    def _gradients(d):
        dx = F.pad(d[:, :, :, 1:] - d[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(d[:, :, 1:, :] - d[:, :, :-1, :], (0, 0, 0, 1))
        return dx, dy

    def _prior(self, depth, size, log_alpha):
        d = F.interpolate(depth, size=size, mode='bilinear', align_corners=False)
        alpha = log_alpha.exp().clamp(0.05, 5.0)
        t_proxy = torch.exp(-alpha * d).clamp(1e-4, 1.0)
        neg_log_t = -torch.log(t_proxy)
        dx, dy = self._gradients(d)
        conf = self._confidence(d)
        prior = torch.cat([d, t_proxy, neg_log_t, dx, dy, conf], dim=1)
        return prior, d, t_proxy, conf

    def _zero_stats(self, feat2, feat3):
        zero2 = feat2.detach().new_zeros(())
        zero3 = feat3.detach().new_zeros(())
        return {
            'stage2_gate_mean': zero2,
            'stage2_gate_max': zero2,
            'stage2_conf_mean': zero2,
            'stage2_conf_min': zero2,
            'stage2_gamma_abs_mean': zero2,
            'stage2_beta_abs_mean': zero2,
            'stage2_delta_abs_mean': zero2,
            'stage3_gate_mean': zero3,
            'stage3_gate_max': zero3,
            'stage3_conf_mean': zero3,
            'stage3_conf_min': zero3,
            'stage3_gamma_abs_mean': zero3,
            'stage3_beta_abs_mean': zero3,
            'stage3_delta_abs_mean': zero3,
        }

    def _uses_depth_features(self):
        return self.phase != 'r0' and self.ablation not in ('r0_only',)

    def _apply_film(self):
        return self.phase != 'r0' and self.ablation in ('full', 'film_only_no_output_refine')

    def _apply_feature_fusion(self):
        return self.feature_fusion_enabled and self.phase != 'r0' and self.ablation in ('full',)

    def _feature_fuse(self, feat, prior, delta_head, gate_head, prefix):
        fusion_input = torch.cat([feat, prior], dim=1)
        gate = torch.sigmoid(gate_head(fusion_input)) * self.feature_fusion_gate_limit
        delta = torch.tanh(delta_head(fusion_input)) * self.feature_fusion_strength
        action = gate * delta
        fused = feat + action
        self.last_aux[f'{prefix}_feature_gate'] = gate
        self.last_aux[f'{prefix}_feature_delta'] = delta
        self.last_aux[f'{prefix}_feature_action'] = action
        self.last_stats[f'{prefix}_feature_gate_mean'] = gate.detach().mean()
        self.last_stats[f'{prefix}_feature_gate_max'] = gate.detach().max()
        self.last_stats[f'{prefix}_feature_delta_abs_mean'] = delta.detach().abs().mean()
        self.last_stats[f'{prefix}_feature_action_abs_mean'] = action.detach().abs().mean()
        return fused

    def forward(self, feat2, feat3, depth):
        self.last_aux = {}
        self.last_stats = {}
        depth = self.normalize_depth(depth)
        if depth is None or not self._uses_depth_features():
            self.last_stats = self._zero_stats(feat2, feat3)
            return feat2, feat3

        prior2, d2, t_proxy2, conf2 = self._prior(depth, feat2.shape[-2:], self.log_alpha2)
        prior3, d3, t_proxy3, conf3 = self._prior(depth, feat3.shape[-2:], self.log_alpha3)
        stats2 = stats3 = None
        if self._apply_film():
            feat2, stats2 = self.stage2(feat2, prior2)
            feat3, stats3 = self.stage3(feat3, prior3)
        else:
            zero_stats = self._zero_stats(feat2, feat3)
            stats2 = {
                'gate_mean': zero_stats['stage2_gate_mean'],
                'gate_max': zero_stats['stage2_gate_max'],
                'conf_mean': conf2.detach().mean(),
                'conf_min': conf2.detach().amin(),
                'gamma_abs_mean': zero_stats['stage2_gamma_abs_mean'],
                'beta_abs_mean': zero_stats['stage2_beta_abs_mean'],
                'delta_abs_mean': zero_stats['stage2_delta_abs_mean'],
            }
            stats3 = {
                'gate_mean': zero_stats['stage3_gate_mean'],
                'gate_max': zero_stats['stage3_gate_max'],
                'conf_mean': conf3.detach().mean(),
                'conf_min': conf3.detach().amin(),
                'gamma_abs_mean': zero_stats['stage3_gamma_abs_mean'],
                'beta_abs_mean': zero_stats['stage3_beta_abs_mean'],
                'delta_abs_mean': zero_stats['stage3_delta_abs_mean'],
            }
        if self._apply_feature_fusion():
            feat2 = self._feature_fuse(
                feat2,
                prior2,
                self.feature_fusion2,
                self.feature_fusion2_gate,
                'stage2',
            )
            feat3 = self._feature_fuse(
                feat3,
                prior3,
                self.feature_fusion3,
                self.feature_fusion3_gate,
                'stage3',
            )
        feature_aux = dict(self.last_aux)
        feature_stats = dict(self.last_stats)
        feat3_up = F.interpolate(feat3, size=feat2.shape[-2:], mode='bilinear', align_corners=False)
        trans_feat = torch.cat([feat2, feat3_up], dim=1)
        t_pred = torch.sigmoid(self.transmission_head(trans_feat))
        t_log_var = self.trans_uncertainty_head(trans_feat).clamp(-6.0, 6.0)
        airlight_pred = torch.sigmoid(self.airlight_head(trans_feat))
        airlight_log_var = self.airlight_uncertainty_head(trans_feat).clamp(-6.0, 6.0)
        self.last_aux = {
            't_pred': t_pred,
            't_log_var': t_log_var,
            'airlight_pred': airlight_pred,
            'airlight_log_var': airlight_log_var,
            'depth': d2,
            't_proxy': t_proxy2,
            'depth_stage3': d3,
            't_proxy_stage3': t_proxy3,
            'confidence': conf2,
            'confidence_stage3': conf3,
            'depth_full': depth,
        }
        self.last_aux.update(feature_aux)
        self.last_stats = {
            'stage2_gate_mean': stats2['gate_mean'],
            'stage2_gate_max': stats2['gate_max'],
            'stage2_conf_mean': stats2['conf_mean'],
            'stage2_conf_min': stats2['conf_min'],
            'stage2_gamma_abs_mean': stats2['gamma_abs_mean'],
            'stage2_beta_abs_mean': stats2['beta_abs_mean'],
            'stage2_delta_abs_mean': stats2['delta_abs_mean'],
            'stage3_gate_mean': stats3['gate_mean'],
            'stage3_gate_max': stats3['gate_max'],
            'stage3_conf_mean': stats3['conf_mean'],
            'stage3_conf_min': stats3['conf_min'],
            'stage3_gamma_abs_mean': stats3['gamma_abs_mean'],
            'stage3_beta_abs_mean': stats3['beta_abs_mean'],
            'stage3_delta_abs_mean': stats3['delta_abs_mean'],
            't_pred_mean': t_pred.detach().mean(),
            't_pred_std': t_pred.detach().std(unbiased=False),
            't_uncertainty_mean': torch.sigmoid(t_log_var.detach()).mean(),
            't_uncertainty_std': torch.sigmoid(t_log_var.detach()).std(unbiased=False),
            'airlight_pred_mean': airlight_pred.detach().mean(),
            'airlight_uncertainty_mean': torch.sigmoid(airlight_log_var.detach()).mean(),
        }
        self.last_stats.update(feature_stats)
        return feat2, feat3

    def fuse_final_feature(self, final_feat, depth):
        if not self._apply_feature_fusion():
            return final_feat
        depth = self.normalize_depth(depth)
        if depth is None:
            return final_feat
        prior_full, _, _, _ = self._prior(depth, final_feat.shape[-2:], self.log_alpha_full)
        return self._feature_fuse(
            final_feat,
            prior_full,
            self.feature_fusion_final,
            self.feature_fusion_final_gate,
            'final',
        )

    @staticmethod
    def _image_texture(brightness):
        dx = F.pad((brightness[:, :, :, 1:] - brightness[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((brightness[:, :, 1:, :] - brightness[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        return 0.5 * (dx + dy)

    @staticmethod
    def _airlight_tensor(hazy, airlight):
        if airlight is None:
            return F.adaptive_max_pool2d(hazy.clamp(0.0, 1.0), 1)
        if not torch.is_tensor(airlight):
            airlight = torch.tensor(airlight, device=hazy.device, dtype=hazy.dtype)
        airlight = airlight.to(hazy.device).float()
        if airlight.dim() == 0:
            airlight = airlight.view(1, 1, 1, 1)
        elif airlight.dim() == 1:
            airlight = airlight.view(-1, 1, 1, 1)
        elif airlight.dim() == 2:
            airlight = airlight.view(airlight.size(0), airlight.size(1), 1, 1)
        if airlight.size(1) == 1:
            airlight = airlight.expand(-1, 3, -1, -1)
        return airlight.clamp(0.0, 1.0)

    @staticmethod
    def _airlight_scalar(airlight, batch, device):
        if airlight is None:
            return None
        if not torch.is_tensor(airlight):
            airlight = torch.tensor(airlight, device=device, dtype=torch.float32)
        airlight = airlight.to(device).float()
        if airlight.dim() == 0:
            airlight = airlight.view(1, 1, 1, 1).expand(batch, -1, -1, -1)
        elif airlight.dim() == 1:
            airlight = airlight.view(-1, 1, 1, 1)
        elif airlight.dim() == 2:
            airlight = airlight.mean(dim=1, keepdim=True).view(airlight.size(0), 1, 1, 1)
        elif airlight.dim() == 4 and airlight.size(1) != 1:
            airlight = airlight.mean(dim=1, keepdim=True)
        return airlight.clamp(0.0, 1.0)

    def _depth_mask(self, final_feat, hazy, t_full, prior_full):
        density = (1.0 - t_full).clamp(0.0, 1.0)
        brightness = hazy.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        texture = self._image_texture(brightness).clamp(0.0, 1.0)
        mask_input = torch.cat([final_feat, prior_full, t_full, density, brightness, texture], dim=1)
        raw = torch.sigmoid(self.depth_mask_head(mask_input))
        dense_open = torch.sigmoid((density - self.depth_mask_density_thresh) * 10.0)
        budget = self.depth_mask_easy_budget + (
            self.depth_mask_dense_budget - self.depth_mask_easy_budget
        ) * dense_open
        prior_conf = prior_full[:, -1:, :, :].clamp(0.0, 1.0)
        bright_conf = torch.sigmoid((0.92 - brightness) * 12.0)
        texture_conf = 0.5 + 0.5 * torch.sigmoid((texture - 0.004) * 80.0)
        mask = raw * budget * prior_conf * bright_conf * texture_conf
        return mask.clamp(0.0, self.depth_mask_dense_budget)

    def refine_output(self, final_feat, output, depth, hazy=None, airlight=None):
        out = output
        r0_enabled = self.ablation in ('full', 'r0_only') and self.phase in ('r0', 'joint', 'depth')
        if r0_enabled:
            r0_delta = torch.tanh(self.r0_refine(final_feat)) * self.r0_residual_scale
            out = out + r0_delta
            self.last_stats['r0_delta_abs_mean'] = r0_delta.detach().abs().mean()
        else:
            self.last_stats['r0_delta_abs_mean'] = output.detach().new_zeros(())

        depth_enabled = (
            self.phase in ('joint', 'depth')
            and self.ablation in ('full', 'phys_blend_only')
            and self.last_aux
            and hazy is not None
        )
        if not depth_enabled:
            self.last_stats['depth_mask_mean'] = output.detach().new_zeros(())
            self.last_stats['depth_delta_abs_mean'] = output.detach().new_zeros(())
            return out

        depth = self.normalize_depth(depth)
        if depth is None:
            return out
        prior_full, _, _, _ = self._prior(depth, output.shape[-2:], self.log_alpha_full)
        t_full = F.interpolate(
            self.last_aux['t_pred'],
            size=output.shape[-2:],
            mode='bilinear',
            align_corners=False,
        ).clamp(self.phys_t_min, 1.0)
        hazy = hazy[:, :, :output.shape[-2], :output.shape[-1]].clamp(0.0, 1.0)
        air = self._airlight_tensor(hazy, airlight)
        j_phys = ((hazy - air * (1.0 - t_full)) / t_full).clamp(0.0, 1.0)
        base_img = (out + hazy).clamp(0.0, 1.0)
        phys_delta_raw = j_phys - base_img
        if self.safe_mix_enabled:
            depth_delta, mask, phys_delta = self._safe_mix_delta(
                final_feat,
                hazy,
                t_full,
                prior_full,
                phys_delta_raw,
                output.shape[-2:],
            )
        else:
            phys_delta = phys_delta_raw.clamp(-self.depth_residual_scale, self.depth_residual_scale)
            mask = self._depth_mask(final_feat, hazy, t_full, prior_full)
            depth_delta = mask * phys_delta
        self.last_aux['depth_mask'] = mask
        self.last_aux['j_phys'] = j_phys
        self.last_stats['depth_mask_mean'] = mask.detach().mean()
        self.last_stats['depth_mask_max'] = mask.detach().max()
        self.last_stats['depth_delta_abs_mean'] = depth_delta.detach().abs().mean()
        self.last_stats['j_phys_delta_abs_mean'] = phys_delta.detach().abs().mean()
        return out + depth_delta

    def _safe_mix_delta(self, final_feat, hazy, t_full, prior_full, phys_delta_raw, output_size):
        clip = float(self.safe_mix_delta_clip)
        phys_delta = phys_delta_raw.clamp(-clip, clip)
        density = (1.0 - t_full).clamp(0.0, 1.0)
        brightness = hazy.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        texture = self._image_texture(brightness).clamp(0.0, 1.0)
        phys_abs = phys_delta.detach().abs().mean(dim=1, keepdim=True)
        t_unc = torch.sigmoid(
            F.interpolate(
                self.last_aux['t_log_var'],
                size=output_size,
                mode='bilinear',
                align_corners=False,
            )
        )
        safe_input = torch.cat(
            [final_feat, prior_full, t_full, density, brightness, texture, phys_abs, t_unc],
            dim=1,
        )
        learned_delta = torch.tanh(self.safe_residual_head(safe_input)) * clip
        raw_gate = torch.sigmoid(self.safe_gate_head(safe_input)) * self.safe_mix_gate_limit
        if self.router_fusion_enabled:
            image_logits = self.router_image_head(safe_input)
            image_router = torch.sigmoid(image_logits.mean(dim=(2, 3), keepdim=True))
            image_router = image_router * self.router_image_gate_limit
            patch_logits = self.router_patch_head(safe_input)
            patch_size = max(1, int(self.router_patch_size))
            if patch_size > 1:
                pooled_patch = F.avg_pool2d(
                    patch_logits,
                    kernel_size=patch_size,
                    stride=patch_size,
                    ceil_mode=True,
                )
                patch_router = F.interpolate(
                    torch.sigmoid(pooled_patch) * self.router_patch_gate_limit,
                    size=output_size,
                    mode='nearest',
                )
            else:
                patch_router = torch.sigmoid(patch_logits) * self.router_patch_gate_limit
        else:
            image_router = torch.ones_like(raw_gate)
            patch_router = torch.ones_like(raw_gate)
        prior_conf = prior_full[:, -1:, :, :].clamp(0.0, 1.0)
        bright_conf = torch.sigmoid((0.94 - brightness) * 10.0)
        texture_conf = 0.5 + 0.5 * torch.sigmoid((texture - 0.003) * 80.0)
        uncertainty_conf = 1.0 - 0.5 * t_unc.clamp(0.0, 1.0)
        gate = (
            raw_gate
            * image_router
            * patch_router
            * prior_conf
            * bright_conf
            * texture_conf
            * uncertainty_conf
        ).clamp(0.0, self.safe_mix_gate_limit)
        mixed_delta = (
            self.safe_mix_phys_weight * phys_delta
            + self.safe_mix_learned_weight * learned_delta
        ).clamp(-clip, clip)
        depth_delta = gate * mixed_delta
        self.last_aux['safe_gate'] = gate
        self.last_aux['safe_raw_gate'] = raw_gate
        self.last_aux['safe_image_router'] = image_router
        self.last_aux['safe_patch_router'] = patch_router
        self.last_aux['safe_learned_delta'] = learned_delta
        self.last_aux['safe_mixed_delta'] = mixed_delta
        self.last_aux['safe_depth_delta'] = depth_delta
        self.last_aux['safe_phys_abs'] = phys_abs
        self.last_stats['safe_gate_mean'] = gate.detach().mean()
        self.last_stats['safe_gate_max'] = gate.detach().max()
        self.last_stats['safe_gate_coverage_gt_001'] = (gate.detach() > 0.01).float().mean()
        self.last_stats['safe_raw_gate_mean'] = raw_gate.detach().mean()
        self.last_stats['safe_image_router_mean'] = image_router.detach().mean()
        self.last_stats['safe_patch_router_mean'] = patch_router.detach().mean()
        self.last_stats['safe_patch_router_max'] = patch_router.detach().max()
        self.last_stats['safe_learned_delta_abs_mean'] = learned_delta.detach().abs().mean()
        self.last_stats['safe_mixed_delta_abs_mean'] = mixed_delta.detach().abs().mean()
        self.last_stats['safe_t_uncertainty_mean'] = t_unc.detach().mean()
        return depth_delta, gate, phys_delta

    def auxiliary_losses(self, rank_pairs=512, min_depth_gap=0.03):
        if not self.last_aux:
            device = next(self.parameters()).device
            zero = torch.zeros((), device=device)
            return {
                'rank': zero,
                'tv': zero,
                'proxy': zero,
                'mask_budget': zero,
                'feature_gate_budget': zero,
                'feature_action_budget': zero,
                'safe_gate_budget': zero,
                'safe_action_budget': zero,
            }
        t_pred = self.last_aux['t_pred']
        depth = self.last_aux['depth'].detach()
        t_proxy = self.last_aux['t_proxy'].detach()
        rank = DepthTransmissionAdapter._rank_loss(t_pred, depth, rank_pairs, min_depth_gap)
        tv = DepthTransmissionAdapter._edge_aware_tv(t_pred, depth)
        proxy = F.l1_loss(t_pred, t_proxy)
        mask = self.last_aux.get('depth_mask')
        mask_budget = mask.mean() if mask is not None else t_pred.new_zeros(())
        feature_gates = []
        feature_actions = []
        for prefix in ('stage2', 'stage3', 'final'):
            gate = self.last_aux.get(f'{prefix}_feature_gate')
            action = self.last_aux.get(f'{prefix}_feature_action')
            if gate is not None:
                feature_gates.append(gate.mean())
            if action is not None:
                feature_actions.append(action.abs().mean())
        feature_gate_budget = (
            torch.stack(feature_gates).mean() if feature_gates else t_pred.new_zeros(())
        )
        feature_action_budget = (
            torch.stack(feature_actions).mean() if feature_actions else t_pred.new_zeros(())
        )
        safe_gate = self.last_aux.get('safe_gate')
        safe_action = self.last_aux.get('safe_depth_delta')
        safe_gate_budget = safe_gate.mean() if safe_gate is not None else t_pred.new_zeros(())
        safe_action_budget = safe_action.abs().mean() if safe_action is not None else t_pred.new_zeros(())
        return {
            'rank': rank,
            'tv': tv,
            'proxy': proxy,
            'mask_budget': mask_budget,
            'feature_gate_budget': feature_gate_budget,
            'feature_action_budget': feature_action_budget,
            'safe_gate_budget': safe_gate_budget,
            'safe_action_budget': safe_action_budget,
        }

    def supervised_losses(self, trans_gt=None, hazy=None, dehazed=None, airlight=None):
        if not self.last_aux:
            device = next(self.parameters()).device
            zero = torch.zeros((), device=device)
            return {
                'trans': zero,
                'phys': zero,
                't_l1': zero,
                't_log_l1': zero,
                't_nll': zero,
                'airlight': zero,
                'airlight_nll': zero,
                't_spearman_proxy': zero,
            }
        t_pred = self.last_aux['t_pred']
        zero = t_pred.new_zeros(())
        losses = {
            'trans': zero,
            'phys': zero,
            't_l1': zero,
            't_log_l1': zero,
            't_nll': zero,
            'airlight': zero,
            'airlight_nll': zero,
            't_spearman_proxy': zero,
        }
        airlight_pred = self.last_aux.get('airlight_pred')
        if airlight_pred is not None and airlight is not None:
            airlight_gt = self._airlight_scalar(airlight, airlight_pred.size(0), airlight_pred.device)
            if airlight_gt is not None:
                a_l1 = F.smooth_l1_loss(airlight_pred, airlight_gt)
                losses['airlight'] = a_l1
                a_log_var = self.last_aux.get('airlight_log_var')
                if a_log_var is not None:
                    a_abs = (airlight_pred - airlight_gt).abs()
                    losses['airlight_nll'] = (torch.exp(-a_log_var) * a_abs.detach() + a_log_var).mean()
        if trans_gt is None:
            return losses
        if trans_gt.dim() == 3:
            trans_gt = trans_gt.unsqueeze(1)
        trans_gt = trans_gt.float().clamp(1e-4, 1.0)
        trans = F.interpolate(trans_gt, size=t_pred.shape[-2:], mode='bilinear', align_corners=False)
        t_l1 = F.smooth_l1_loss(t_pred, trans)
        losses['trans'] = t_l1
        losses['t_l1'] = t_l1.detach()
        log_t_error = (torch.log(t_pred.clamp(1e-4, 1.0)) - torch.log(trans.clamp(1e-4, 1.0))).abs()
        losses['t_log_l1'] = log_t_error.mean()
        t_log_var = self.last_aux.get('t_log_var')
        if t_log_var is not None:
            t_log_var = t_log_var.clamp(-6.0, 6.0)
            losses['t_nll'] = (torch.exp(-t_log_var) * log_t_error.detach() + t_log_var).mean()
        if hazy is not None and dehazed is not None and airlight is not None:
            airlight = self._airlight_tensor(hazy, airlight)
            trans_full = F.interpolate(t_pred, size=hazy.shape[-2:], mode='bilinear', align_corners=False)
            recon_hazy = dehazed.clamp(0.0, 1.0) * trans_full + airlight * (1.0 - trans_full)
            losses['phys'] = F.smooth_l1_loss(recon_hazy, hazy)
        return losses

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
    def __init__(self, version, data, dta_variant='v1', **dta_kwargs):
        super(ConvIRDTA, self).__init__(version, data)
        self.dta_variant = dta_variant
        if dta_variant == 'v3':
            self.DTA = DepthAttributedPreserveAdapter(
                stage1_channels=32,
                stage2_channels=64,
                stage3_channels=128,
                **dta_kwargs,
            )
        elif dta_variant == 'v2':
            self.DTA = CalibratedDepthTransmissionAdapter(
                stage1_channels=32,
                stage2_channels=64,
                stage3_channels=128,
                **dta_kwargs,
            )
        else:
            allowed = {
                key: value for key, value in dta_kwargs.items()
                if key in ('prior_channels', 'gate_bias', 'gate_limit', 'gamma_limit', 'beta_limit', 'alpha_init')
            }
            self.DTA = DepthTransmissionAdapter(
                stage2_channels=64,
                stage3_channels=128,
                **allowed,
            )

    def forward(self, x, depth=None, airlight=None):
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
        final_feat = z
        if hasattr(self.DTA, 'fuse_final_feature'):
            final_feat = self.DTA.fuse_final_feature(final_feat, depth)
        z = self.feat_extract[5](final_feat)
        if hasattr(self.DTA, 'refine_output'):
            z = self.DTA.refine_output(final_feat, z, depth, hazy=x, airlight=airlight)
        outputs.append(z+x)

        return outputs

    def dta_auxiliary_losses(self, rank_pairs=512, min_depth_gap=0.03):
        return self.DTA.auxiliary_losses(rank_pairs, min_depth_gap)

    def dta_supervised_losses(self, trans_gt=None, hazy=None, dehazed=None, airlight=None):
        if hasattr(self.DTA, 'supervised_losses'):
            return self.DTA.supervised_losses(trans_gt, hazy, dehazed, airlight)
        device = next(self.parameters()).device
        zero = torch.zeros((), device=device)
        return {'trans': zero, 'phys': zero, 't_l1': zero, 't_spearman_proxy': zero}

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
    dta_variant='v1',
    dta_prior_channels=16,
    dta_gate_bias=-6.0,
    dta_gate_limit=0.05,
    dta_gamma_limit=0.10,
    dta_beta_limit=0.05,
    dta_alpha_init=1.0,
    dta_depth_mode='normal',
    dta_confidence_floor=0.25,
    dta_confidence_local_scale=6.0,
    dta_output_residual_scale=0.03,
    dta_r0_residual_scale=0.04,
    dta_depth_residual_scale=0.08,
    dta_depth_mask_easy_budget=0.04,
    dta_depth_mask_dense_budget=0.12,
    dta_depth_mask_density_thresh=0.35,
    dta_depth_mask_bias=-4.0,
    dta_phys_t_min=0.10,
    dta_phase='joint',
    dta_ablation='full',
    dta_safe_mix_enabled=False,
    dta_safe_mix_delta_clip=0.08,
    dta_safe_mix_phys_weight=1.0,
    dta_safe_mix_learned_weight=0.0,
    dta_safe_mix_gate_limit=1.0,
    dta_safe_mix_gate_bias=-3.0,
    dta_router_fusion_enabled=False,
    dta_router_image_gate_limit=1.0,
    dta_router_patch_gate_limit=1.0,
    dta_router_patch_size=32,
    dta_router_image_bias=2.0,
    dta_router_patch_bias=2.0,
    dta_feature_fusion_enabled=False,
    dta_feature_fusion_strength=0.10,
    dta_feature_fusion_gate_limit=1.0,
    dta_feature_fusion_gate_bias=2.0,
):
    if fam_mode != 'original':
        raise ValueError(
            "Official ConvIR-B anchor only supports fam_mode='original'. "
            "Create a route branch for architecture variants."
        )
    if arch in ('official_convir', 'convir'):
        return ConvIR(version, data)
    if arch in ('dta', 'dta_v2', 'dta_v3'):
        variant = 'v3' if arch == 'dta_v3' else ('v2' if arch == 'dta_v2' else dta_variant)
        kwargs = {
            'prior_channels': dta_prior_channels,
            'gate_bias': dta_gate_bias,
            'gate_limit': dta_gate_limit,
            'gamma_limit': dta_gamma_limit,
            'beta_limit': dta_beta_limit,
            'alpha_init': dta_alpha_init,
            'depth_mode': dta_depth_mode,
            'confidence_floor': dta_confidence_floor,
            'confidence_local_scale': dta_confidence_local_scale,
        }
        if variant == 'v3':
            kwargs.update(
                {
                    'r0_residual_scale': dta_r0_residual_scale,
                    'depth_residual_scale': dta_depth_residual_scale,
                    'depth_mask_easy_budget': dta_depth_mask_easy_budget,
                    'depth_mask_dense_budget': dta_depth_mask_dense_budget,
                    'depth_mask_density_thresh': dta_depth_mask_density_thresh,
                    'depth_mask_bias': dta_depth_mask_bias,
                    'phys_t_min': dta_phys_t_min,
                    'phase': dta_phase,
                    'ablation': dta_ablation,
                    'safe_mix_enabled': dta_safe_mix_enabled,
                    'safe_mix_delta_clip': dta_safe_mix_delta_clip,
                    'safe_mix_phys_weight': dta_safe_mix_phys_weight,
                    'safe_mix_learned_weight': dta_safe_mix_learned_weight,
                    'safe_mix_gate_limit': dta_safe_mix_gate_limit,
                    'safe_mix_gate_bias': dta_safe_mix_gate_bias,
                    'router_fusion_enabled': dta_router_fusion_enabled,
                    'router_image_gate_limit': dta_router_image_gate_limit,
                    'router_patch_gate_limit': dta_router_patch_gate_limit,
                    'router_patch_size': dta_router_patch_size,
                    'router_image_bias': dta_router_image_bias,
                    'router_patch_bias': dta_router_patch_bias,
                    'feature_fusion_enabled': dta_feature_fusion_enabled,
                    'feature_fusion_strength': dta_feature_fusion_strength,
                    'feature_fusion_gate_limit': dta_feature_fusion_gate_limit,
                    'feature_fusion_gate_bias': dta_feature_fusion_gate_bias,
                }
            )
        else:
            kwargs['output_residual_scale'] = dta_output_residual_scale
        return ConvIRDTA(
            version,
            data,
            dta_variant=variant,
            **kwargs,
        )
    raise ValueError(f'Unsupported arch: {arch}')
