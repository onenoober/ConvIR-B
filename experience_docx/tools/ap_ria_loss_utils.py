"""Loss helpers for AP-RIA / AP-RSM-IA training.

These utilities intentionally avoid explicit E - A0 residual targets and GT-based
expert selection. Teacher outputs, when available, are converted into low-frequency
and detail/edge guidance targets plus GT-free confidence maps. The deployed model
does not consume teacher outputs.
"""

from __future__ import print_function

import torch
import torch.nn.functional as F


def avg_lowpass(x, kernel_size=9):
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def highpass(x, kernel_size=9):
    return x - avg_lowpass(x, kernel_size=kernel_size)


def gradient_magnitude(x):
    gray = x.mean(dim=1, keepdim=True)
    gx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    gy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def dark_channel(x, kernel_size=15):
    min_rgb = torch.min(x, dim=1, keepdim=True)[0]
    pad = kernel_size // 2
    return -F.max_pool2d(-min_rgb, kernel_size=kernel_size, stride=1, padding=pad)


def weighted_l1(pred, target, weight=None, eps=1e-6):
    loss = (pred - target).abs()
    if weight is None:
        return loss.mean()
    if weight.shape[-2:] != pred.shape[-2:]:
        weight = F.interpolate(weight, size=pred.shape[-2:], mode='bilinear', align_corners=False)
    while weight.dim() < loss.dim():
        weight = weight.unsqueeze(1)
    return (loss * weight).sum() / (weight.sum() * loss.shape[1] + eps)


def total_variation(x):
    if x is None:
        return x.new_tensor(0.0)
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def _teacher_self_risk(teacher, hazy=None):
    """GT-free per-pixel confidence prior from teacher output self-consistency.

    High confidence: not clipped/saturated, not unusually high-frequency, consistent
    with plausible haze-prior regions. This is intentionally simple and diagnostic.
    """
    # Range / saturation risk. Works best if tensors are in [0, 1]-like range.
    t_min = teacher.min(dim=1, keepdim=True)[0]
    t_max = teacher.max(dim=1, keepdim=True)[0]
    saturation = (t_max - t_min).clamp(0, 1)

    # Very high detail energy is risky in flat or bright regions.
    detail_energy = highpass(teacher, kernel_size=9).abs().mean(dim=1, keepdim=True)
    detail_norm = detail_energy / (detail_energy.detach().mean(dim=(-2, -1), keepdim=True) + 1e-6)
    detail_risk = torch.sigmoid(detail_norm - 2.0)

    conf = 1.0 - 0.35 * saturation - 0.35 * detail_risk
    conf = conf.clamp(0.05, 1.0)

    if hazy is not None:
        dc = dark_channel(hazy, kernel_size=15)
        bright = hazy.max(dim=1, keepdim=True)[0]
        # Bright low-dark-channel regions are often sky/flat risk; keep detail confidence lower.
        flat_bright_risk = (bright > 0.75).float() * (dc > 0.45).float()
        conf = (conf * (1.0 - 0.25 * flat_bright_risk)).clamp(0.05, 1.0)
    return conf


def build_teacher_guidance(
    teacher_outputs,
    hazy=None,
    lowpass_kernel=9,
    detail_kernel=9,
    agreement_sigma=0.08,
):
    """Build GT-free teacher guidance targets.

    Args:
        teacher_outputs: Tensor or list/tuple of tensors [B,3,H,W]. No A0 output is needed.
        hazy: optional hazy input for haze-prior confidence adjustment.
    Returns:
        dict with t_low, t_detail, c_low, c_detail.
    """
    if teacher_outputs is None:
        return None
    if torch.is_tensor(teacher_outputs):
        teachers = [teacher_outputs]
    else:
        teachers = list(teacher_outputs)
    if len(teachers) == 0:
        return None

    lows = [avg_lowpass(t, kernel_size=lowpass_kernel) for t in teachers]
    details = [highpass(t, kernel_size=detail_kernel) for t in teachers]

    if len(teachers) == 1:
        t_low = lows[0]
        t_detail = details[0]
        c_base = _teacher_self_risk(teachers[0], hazy=hazy)
        c_low = c_base
        c_detail = c_base * (1.0 - 0.35 * details[0].abs().mean(dim=1, keepdim=True).clamp(0, 1))
        c_detail = c_detail.clamp(0.05, 1.0)
        return {'t_low': t_low, 't_detail': t_detail, 'c_low': c_low, 'c_detail': c_detail}

    stack_low = torch.stack(lows, dim=0)
    stack_detail = torch.stack(details, dim=0)

    low_mean = stack_low.mean(dim=0)
    detail_mean = stack_detail.mean(dim=0)

    low_std = stack_low.std(dim=0, unbiased=False).mean(dim=1, keepdim=True)
    detail_std = stack_detail.std(dim=0, unbiased=False).mean(dim=1, keepdim=True)

    c_agree_low = torch.exp(-low_std / agreement_sigma).clamp(0.05, 1.0)
    c_agree_detail = torch.exp(-detail_std / agreement_sigma).clamp(0.05, 1.0)

    self_confs = [_teacher_self_risk(t, hazy=hazy) for t in teachers]
    c_self = torch.stack(self_confs, dim=0).mean(dim=0)

    c_low = (c_agree_low * c_self).clamp(0.05, 1.0)
    c_detail = (c_agree_detail * c_self).clamp(0.05, 1.0)

    return {'t_low': low_mean, 't_detail': detail_mean, 'c_low': c_low, 'c_detail': c_detail}


def ap_ria_loss(
    outputs,
    target,
    anchor_side=None,
    teacher_guidance=None,
    aux=None,
    weights=None,
):
    """Compute AP-RIA training loss.

    Args:
        outputs: model output list or final output tensor.
        target: clean GT tensor. Used only for normal supervised restoration loss.
        anchor_side: optional A0 side output for low-confidence preservation.
        teacher_guidance: optional dict from build_teacher_guidance().
        aux: optional dict returned by ConvIR_AP_RIA(return_aux=True).
        weights: optional dict; defaults are conservative.
    Returns:
        total_loss, stats dict.
    """
    if weights is None:
        weights = {}
    w_rec = float(weights.get('rec', 1.0))
    w_low = float(weights.get('teacher_low', 0.15))
    w_detail = float(weights.get('teacher_detail', 0.04))
    w_preserve = float(weights.get('preserve', 0.03))
    w_gate = float(weights.get('gate_prior', 0.0))
    w_smooth = float(weights.get('smooth', 0.001))

    output = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs

    losses = {}
    losses['rec_l1'] = F.l1_loss(output, target)

    total = w_rec * losses['rec_l1']

    if teacher_guidance is not None:
        t_low = teacher_guidance['t_low'].detach()
        t_detail = teacher_guidance['t_detail'].detach()
        c_low = teacher_guidance['c_low'].detach()
        c_detail = teacher_guidance['c_detail'].detach()

        pred_low = avg_lowpass(output, kernel_size=9)
        pred_detail = highpass(output, kernel_size=9)

        losses['teacher_low'] = weighted_l1(pred_low, t_low, c_low)
        losses['teacher_detail'] = weighted_l1(pred_detail, t_detail, c_detail)

        total = total + w_low * losses['teacher_low'] + w_detail * losses['teacher_detail']

        if anchor_side is not None and w_preserve > 0:
            conf = torch.max(c_low, c_detail)
            preserve_weight = (1.0 - conf).clamp(0.0, 1.0)
            losses['preserve_anchor'] = weighted_l1(output, anchor_side.detach(), preserve_weight)
            total = total + w_preserve * losses['preserve_anchor']

        if aux is not None and w_gate > 0 and 'g_low' in aux and 'g_detail' in aux:
            g_low = aux['g_low']
            g_detail = aux['g_detail']
            c_low_g = F.interpolate(c_low, size=g_low.shape[-2:], mode='bilinear', align_corners=False)
            c_detail_g = F.interpolate(c_detail, size=g_detail.shape[-2:], mode='bilinear', align_corners=False)
            losses['gate_prior'] = F.l1_loss(g_low, c_low_g) + F.l1_loss(g_detail, c_detail_g)
            total = total + w_gate * losses['gate_prior']

    if aux is not None and w_smooth > 0 and 'g_low' in aux and 'g_detail' in aux:
        losses['gate_smooth'] = total_variation(aux['g_low']) + total_variation(aux['g_detail'])
        total = total + w_smooth * losses['gate_smooth']

    stats = {}
    for k, v in losses.items():
        stats[k] = float(v.detach().cpu())
    stats['total'] = float(total.detach().cpu())
    return total, stats
