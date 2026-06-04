import os
import torch
from data import train_dataloader, valid_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torch.nn as nn

from warmup_scheduler import GradualWarmupScheduler


def _log_modulation_stats(model, args, epoch_idx, device):
    if args.mod_stats_freq <= 0 or epoch_idx % args.mod_stats_freq != 0:
        return
    if not hasattr(model, 'collect_modulation_stats'):
        return

    dataloader = valid_dataloader(
        args.data_dir,
        args.data,
        batch_size=1,
        num_workers=0,
        split_json=getattr(args, "dpga_valid_split_json", ""),
        split_name=getattr(args, "dpga_valid_split_name", ""),
    )
    sums = {}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if args.mod_stats_batches > 0 and batch_idx >= args.mod_stats_batches:
                break
            input_img = batch_data[0].to(device)
            batch_stats = model.collect_modulation_stats(input_img)
            for fam_name, fam_stats in batch_stats.items():
                sums.setdefault(fam_name, {})
                for key, value in fam_stats.items():
                    sums[fam_name][key] = sums[fam_name].get(key, 0.0) + value
            count += 1
    model.train()

    if count == 0:
        return
    for fam_name in sorted(sums):
        averaged = {key: value / count for key, value in sorted(sums[fam_name].items())}
        print(
            "MOD_STATS Epoch: %03d FAM: %s Samples: %d "
            "gamma_mean: %.8f gamma_std: %.8f gamma_min: %.8f gamma_max: %.8f "
            "gamma_abs_gt_0.5: %.8f beta_mean: %.8f beta_std: %.8f "
            "beta_min: %.8f beta_max: %.8f beta_abs_gt_0.1: %.8f" % (
                epoch_idx,
                fam_name,
                count,
                averaged.get('gamma_mean', 0.0),
                averaged.get('gamma_std', 0.0),
                averaged.get('gamma_min', 0.0),
                averaged.get('gamma_max', 0.0),
                averaged.get('gamma_abs_gt_0.5', 0.0),
                averaged.get('beta_mean', 0.0),
                averaged.get('beta_std', 0.0),
                averaged.get('beta_min', 0.0),
                averaged.get('beta_max', 0.0),
                averaged.get('beta_abs_gt_0.1', 0.0),
            )
        )


def _flatten_stat_dict(stats):
    flat = {}

    def visit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}{key.lower()}_", child)
        elif isinstance(value, (int, float)):
            flat[prefix[:-1]] = float(value)

    visit("", stats)
    return flat


def _pad_to_factor(input_img, factor=32):
    h, w = input_img.shape[2], input_img.shape[3]
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    padh = padded_h - h if h % factor != 0 else 0
    padw = padded_w - w if w % factor != 0 else 0
    if padh or padw:
        input_img = F.pad(input_img, (0, padw, 0, padh), 'reflect')
    return input_img


def _charbonnier_loss(pred, target, mask=None, eps=1e-3):
    loss = torch.sqrt((pred - target) * (pred - target) + eps * eps)
    if mask is None:
        return loss.mean()
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def _gradient_magnitude_1ch(x):
    grad_x = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    grad_y = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)


def _luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def _chroma(x):
    return x - _luma(x)


def _weighted_l1_loss(pred, target, weight=None):
    loss = (pred - target).abs()
    if weight is None:
        return loss.mean()
    if weight.shape[1] == 1 and pred.shape[1] != 1:
        weight = weight.expand(-1, pred.shape[1], -1, -1)
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


def _image_bucket_masks(bucket_label, like_tensor):
    if bucket_label is None:
        shape = (like_tensor.shape[0], 1, 1, 1)
        zeros = torch.zeros(shape, device=like_tensor.device, dtype=torch.bool)
        return zeros, zeros, zeros
    bucket = bucket_label.to(device=like_tensor.device).view(-1, 1, 1, 1)
    hard = bucket <= 0
    medium = bucket == 1
    easy = bucket >= 2
    return hard, medium, easy


def _tail_anchor_mask(input_img, anchor_img, label_img, args, bucket_label=None):
    gray = _luma(input_img.detach())
    grad = _gradient_magnitude_1ch(gray)
    max_rgb = input_img.detach().max(dim=1, keepdim=True).values
    min_rgb = input_img.detach().min(dim=1, keepdim=True).values
    saturation = max_rgb - min_rgb
    anchor_error = (anchor_img.detach() - label_img.detach()).abs().mean(dim=1, keepdim=True)

    high_anchor = anchor_error <= float(args.dpga_tc_anchor_error_threshold)
    bright_low_gradient = (gray >= 0.62) & (grad <= 0.035)
    low_saturation_bright = (gray >= 0.58) & (saturation <= 0.12)
    sky_bright_proxy = (gray >= 0.66) & (grad <= 0.05) & (saturation <= 0.20)
    if getattr(args, "dpga_tc_mask_mode", "legacy") == "legacy":
        mask = high_anchor | bright_low_gradient | low_saturation_bright | sky_bright_proxy
    else:
        hard_image, medium_image, easy_image = _image_bucket_masks(bucket_label, input_img)
        if bucket_label is None and getattr(args, "dpga_require_hard_labels", 0):
            raise ValueError("--dpga_tc_mask_mode hard_selective requires hard bucket labels")
        mask = (
            bright_low_gradient
            | low_saturation_bright
            | sky_bright_proxy
            | (easy_image & high_anchor)
        )
    mask = mask.to(dtype=input_img.dtype)
    hard_image, medium_image, easy_image = _image_bucket_masks(bucket_label, input_img)
    stats = {
        "mask_ratio": mask.mean().detach(),
        "high_anchor_ratio": high_anchor.to(dtype=input_img.dtype).mean().detach(),
        "bright_low_gradient_ratio": bright_low_gradient.to(dtype=input_img.dtype).mean().detach(),
        "low_saturation_bright_ratio": low_saturation_bright.to(dtype=input_img.dtype).mean().detach(),
        "sky_ratio": sky_bright_proxy.to(dtype=input_img.dtype).mean().detach(),
        "hard_image_ratio": hard_image.to(dtype=input_img.dtype).mean().detach(),
        "medium_image_ratio": medium_image.to(dtype=input_img.dtype).mean().detach(),
        "easy_image_ratio": easy_image.to(dtype=input_img.dtype).mean().detach(),
    }
    return mask, stats


def _total_variation_loss(delta):
    grad_x = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
    grad_y = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
    return grad_x + grad_y


def _reconstruction_loss(pred, target, args, weight=None):
    if getattr(args, "arch", "convir") == "dpga" and args.dpga_tc_rec_loss == "charbonnier":
        return _charbonnier_loss(pred, target, mask=weight)
    return _weighted_l1_loss(pred, target, weight=weight)


def _dpga_tail_control_enabled(args):
    return (
        getattr(args, "arch", "convir") == "dpga"
        and (
            args.dpga_tc_anchor_lambda > 0
            or args.dpga_tc_chroma_lambda > 0
            or args.dpga_tc_delta_lambda > 0
            or args.dpga_tc_delta_tv_lambda > 0
        )
    )


def _dpga_needs_anchor(args):
    return (
        getattr(args, "arch", "convir") == "dpga"
        and (
            _dpga_tail_control_enabled(args)
            or getattr(args, "dpga_hard_region_lambda", 0.0) > 0
        )
    )


def _dpga_anchor_forward(model, input_img, depth):
    if not hasattr(model, "set_dpga_runtime_config"):
        raise RuntimeError("DPGA tail-control anchor requires set_dpga_runtime_config().")
    active_adapters = set(getattr(model, "dpga_active_adapters", set()))
    scale_multiplier = float(getattr(model, "dpga_scale_multiplier", 1.0))
    hard_gate_mode = getattr(model, "dpga_hard_gate_mode", "off")
    adapter_scale_multipliers = dict(getattr(model, "dpga_adapter_scale_multipliers", {}))
    last_hard_gate = getattr(model, "_last_hard_gate", None)
    last_hard_gate_logits = getattr(model, "_last_hard_gate_logits", None)
    last_stats = {}
    for name, module in model.named_modules():
        if hasattr(module, "_last_stats"):
            last_stats[name] = dict(getattr(module, "_last_stats", {}))
    try:
        model.set_dpga_runtime_config(active_adapters="none", scale_multiplier=0.0)
        with torch.no_grad():
            anchor = model(input_img, depth)[2].detach()
    finally:
        model.set_dpga_runtime_config(
            active_adapters=active_adapters,
            scale_multiplier=scale_multiplier,
            hard_gate_mode=hard_gate_mode,
            shallow_scale_multiplier=adapter_scale_multipliers.get("shallow", 1.0),
            bottleneck_scale_multiplier=adapter_scale_multipliers.get("bottleneck", 1.0),
            skip_scale_multiplier=adapter_scale_multipliers.get("skip", 1.0),
        )
        model._last_hard_gate = last_hard_gate
        model._last_hard_gate_logits = last_hard_gate_logits
        for name, module in model.named_modules():
            if name in last_stats:
                module._last_stats = last_stats[name]
    return anchor


def _dpga_hard_region_weight(anchor, label_img, bucket_label, args):
    hard_sample_lambda = float(getattr(args, "dpga_hard_sample_lambda", 0.0))
    hard_region_lambda = float(getattr(args, "dpga_hard_region_lambda", 0.0))
    if hard_sample_lambda <= 0 and hard_region_lambda <= 0:
        return None, {}
    hard_image, _medium_image, _easy_image = _image_bucket_masks(bucket_label, label_img)
    if bucket_label is None and getattr(args, "dpga_require_hard_labels", 0):
        raise ValueError("DPGA hard reconstruction weighting requires hard bucket labels")
    anchor_error = (anchor.detach() - label_img.detach()).abs().mean(dim=1, keepdim=True)
    move = torch.relu(anchor_error - float(args.dpga_hard_region_error_threshold))
    norm = move.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    move = (move / norm).clamp(0, 1)
    hard_image = hard_image.to(dtype=label_img.dtype)
    weight = 1.0 + hard_sample_lambda * hard_image + hard_region_lambda * hard_image * move
    stats = {
        "hard_region_weight_mean": weight.mean().detach(),
        "hard_region_weight_max": weight.max().detach(),
        "hard_sample_weight_mean": (1.0 + hard_sample_lambda * hard_image).mean().detach(),
        "hard_region_move_ratio": (move > 0).to(dtype=label_img.dtype).mean().detach(),
    }
    return weight, stats


def _dpga_hard_gate_supervision(model, bucket_label, like_tensor, args):
    gate_lambda = float(getattr(args, "dpga_hard_gate_lambda", 0.0))
    if gate_lambda <= 0:
        return {}
    logits = getattr(model, "_last_hard_gate_logits", None)
    gate = getattr(model, "_last_hard_gate", None)
    if logits is None or gate is None:
        return {}
    if bucket_label is None:
        if getattr(args, "dpga_require_hard_labels", 0):
            raise ValueError("DPGA hard gate supervision requires hard bucket labels")
        return {}
    hard_image, medium_image, easy_image = _image_bucket_masks(bucket_label, like_tensor)
    target = (
        hard_image.to(dtype=logits.dtype) * float(args.dpga_hard_gate_hard_target)
        + medium_image.to(dtype=logits.dtype) * float(args.dpga_hard_gate_medium_target)
        + easy_image.to(dtype=logits.dtype) * float(args.dpga_hard_gate_easy_target)
    )
    if tuple(target.shape[-2:]) != tuple(logits.shape[-2:]):
        target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
    loss = F.binary_cross_entropy_with_logits(logits, target.expand_as(logits))
    stats = {
        "dpga_hard_gate_bce": loss,
        "dpga_hard_gate_mean": gate.detach().mean(),
        "dpga_hard_gate_target": target.detach().mean(),
        "dpga_hard_gate_hard_ratio": hard_image.to(dtype=like_tensor.dtype).mean().detach(),
        "dpga_hard_gate_easy_ratio": easy_image.to(dtype=like_tensor.dtype).mean().detach(),
    }
    return stats


def _dpga_tail_control_regularization(model, input_img, label_img, depth, pred_full, args, bucket_label=None):
    if not _dpga_needs_anchor(args):
        return {}, None, None

    anchor = _dpga_anchor_forward(model, input_img, depth)
    mask, mask_stats = _tail_anchor_mask(input_img, anchor, label_img, args, bucket_label=bucket_label)
    rec_weight, weight_stats = _dpga_hard_region_weight(anchor, label_img, bucket_label, args)
    if not _dpga_tail_control_enabled(args):
        reg = {}
        reg.update(mask_stats)
        reg.update(weight_stats)
        return reg, anchor, rec_weight
    delta = pred_full - anchor
    zero_delta = torch.zeros_like(delta)
    reg = {
        "dpga_tc_anchor": _charbonnier_loss(pred_full, anchor, mask=mask),
        "dpga_tc_chroma": _charbonnier_loss(_chroma(pred_full), _chroma(anchor), mask=mask),
        "dpga_tc_delta": _charbonnier_loss(delta, zero_delta),
        "dpga_tc_delta_tv": _total_variation_loss(delta),
    }
    reg.update(mask_stats)
    reg.update(weight_stats)
    return reg, anchor, rec_weight


def _log_apdr_stats(model, args, epoch_idx, device):
    if args.mod_stats_freq <= 0 or epoch_idx % args.mod_stats_freq != 0:
        return
    if not hasattr(model, 'collect_apdr_stats'):
        return

    dataloader = valid_dataloader(
        args.data_dir,
        args.data,
        batch_size=1,
        num_workers=0,
        split_json=getattr(args, "dpga_valid_split_json", ""),
        split_name=getattr(args, "dpga_valid_split_name", ""),
    )
    sums = {}
    count = 0
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                if args.mod_stats_batches > 0 and batch_idx >= args.mod_stats_batches:
                    break
                input_img = _pad_to_factor(batch_data[0].to(device))
                flat_stats = _flatten_stat_dict(model.collect_apdr_stats(input_img))
                for key, value in flat_stats.items():
                    sums[key] = sums.get(key, 0.0) + value
                count += 1
    finally:
        model.train(was_training)

    if count == 0 or not sums:
        return
    averaged = {key: value / count for key, value in sorted(sums.items())}
    detail = " ".join(f"{key}: {value:.8f}" for key, value in averaged.items())
    print(f"APDR_STATS Epoch: {epoch_idx:03d} Samples: {count} {detail}")


def _configure_train_scope(model, args):
    arch = getattr(args, "arch", "convir")
    if arch not in ("apdr", "dpga"):
        return list(model.parameters())

    if arch == "dpga":
        scope = getattr(args, "dpga_train_scope", "adapter_only")
        if scope == "all":
            print("DPGA_TRAIN_SCOPE all: all parameters trainable")
            return list(model.parameters())
        if scope not in ("adapter_only", "fusion_neighbor"):
            raise ValueError(f"Unsupported dpga_train_scope: {scope}")
        neighbor_prefixes = ("FAM1", "FAM2", "SCM1", "SCM2", "Convs.0", "Convs.1")
        dpga_params = []
        neighbor_params = []
        trainable = 0
        frozen = 0
        for name, param in model.named_parameters():
            is_dpga = name.startswith("DPGA_")
            is_neighbor = scope == "fusion_neighbor" and any(name.startswith(prefix) for prefix in neighbor_prefixes)
            param.requires_grad = is_dpga or is_neighbor
            if param.requires_grad:
                trainable += param.numel()
                if is_dpga:
                    dpga_params.append(param)
                else:
                    neighbor_params.append(param)
            else:
                frozen += param.numel()
        trainable_params = dpga_params + neighbor_params
        if not trainable_params:
            raise RuntimeError("No trainable parameters. Check --dpga_train_scope.")
        print(f"DPGA_TRAIN_SCOPE {scope}: trainable={trainable} frozen={frozen}")
        if scope == "fusion_neighbor":
            print(
                "DPGA_TRAIN_SCOPE fusion_neighbor groups: "
                f"dpga_params={sum(param.numel() for param in dpga_params)} "
                f"neighbor_params={sum(param.numel() for param in neighbor_params)} "
                f"neighbor_lr={args.dpga_neighbor_learning_rate}"
            )
            groups = [{"params": dpga_params}]
            if neighbor_params:
                groups.append({"params": neighbor_params, "lr": args.dpga_neighbor_learning_rate})
            return groups
        return trainable_params

    scope = getattr(args, "apdr_train_scope", "all")
    if scope == "all":
        print("APDR_TRAIN_SCOPE all: all parameters trainable")
        return list(model.parameters())

    if scope not in ("apdr_only", "apdr_residual_only"):
        raise ValueError(f"Unsupported apdr_train_scope: {scope}")

    active_prefixes = ("APDR_",)
    if hasattr(model, "active_apdr_prefixes"):
        active_prefixes = model.active_apdr_prefixes()

    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith("APDR_"):
            in_active_scale = any(name.startswith(prefix) for prefix in active_prefixes)
            if scope == "apdr_residual_only":
                param.requires_grad = in_active_scale and (
                    ".residual_body." in name or ".residual_head." in name
                )
            else:
                param.requires_grad = in_active_scale
        else:
            param.requires_grad = False
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Check --apdr_train_scope.")
    print(f"APDR_TRAIN_SCOPE {scope}: trainable={trainable} frozen={frozen}")
    return trainable_params


def _set_training_mode(model, args):
    if (
        getattr(args, "arch", "convir") == "dpga"
        and getattr(args, "dpga_train_scope", "adapter_only") != "all"
    ):
        model.eval()
        scope = getattr(args, "dpga_train_scope", "adapter_only")
        neighbor_prefixes = ("FAM1", "FAM2", "SCM1", "SCM2", "Convs.0", "Convs.1")
        for name, module in model.named_modules():
            is_dpga = name.startswith("DPGA_")
            is_neighbor = scope == "fusion_neighbor" and any(name.startswith(prefix) for prefix in neighbor_prefixes)
            if is_dpga or is_neighbor:
                module.train()
        return
    if (
        getattr(args, "arch", "convir") == "apdr"
        and getattr(args, "apdr_train_scope", "all") != "all"
    ):
        model.eval()
        for name, module in model.named_modules():
            if name.startswith("APDR_"):
                module.train()
        return
    model.train()


def _dpga_depth_cache_dir(args):
    if getattr(args, "arch", "convir") != "dpga":
        return ""
    depth_cache_dir = getattr(args, "dpga_depth_cache_dir", "")
    if not depth_cache_dir:
        raise ValueError("--dpga_depth_cache_dir is required when --arch dpga")
    return depth_cache_dir


def _split_batch(batch_data, device, args):
    if getattr(args, "arch", "convir") == "dpga":
        if len(batch_data) < 3:
            raise ValueError("DPGA dataloader must return input, label, depth prior.")
        input_img = batch_data[0].to(device)
        label_img = batch_data[1].to(device)
        depth = batch_data[2].to(device)
        bucket_label = None
        if len(batch_data) >= 4 and torch.is_tensor(batch_data[3]):
            bucket_label = batch_data[3].to(device)
        return input_img, label_img, depth, bucket_label
    input_img, label_img = batch_data[:2]
    return input_img.to(device), label_img.to(device), None, None


def _model_forward(model, input_img, depth, args):
    if getattr(args, "arch", "convir") == "dpga":
        return model(input_img, depth)
    return model(input_img)


def _log_dpga_stats(model, args, epoch_idx):
    if args.mod_stats_freq <= 0 or epoch_idx % args.mod_stats_freq != 0:
        return
    if getattr(args, "arch", "convir") != "dpga" or not hasattr(model, "collect_dpga_stats"):
        return
    stats = model.collect_dpga_stats()
    detail = []
    for name in sorted(stats):
        item = stats[name]
        if "scale" in item and "effective_scale" in item:
            detail.append(
                "%s_scale=%.8f %s_eff=%.8f"
                % (name, item["scale"], name, item["effective_scale"])
            )
        elif name == "DPGA_hard_gate":
            detail.append(
                "%s_mean=%.8f %s_min=%.8f %s_max=%.8f"
                % (
                    name,
                    item.get("gate_mean", 0.0),
                    name,
                    item.get("gate_min", 0.0),
                    name,
                    item.get("gate_max", 0.0),
                )
            )
    print(f"DPGA_STATS Epoch: {epoch_idx:03d} " + " ".join(detail))


def _train(model, args):
    if (
        getattr(args, "arch", "convir") == "apdr"
        and getattr(args, "apdr_loss_scales", "all") == "full_only"
        and getattr(args, "apdr_active_scales", "all") != "full"
    ):
        raise ValueError("--apdr_loss_scales full_only requires --apdr_active_scales full")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()

    trainable_params = _configure_train_scope(model, args)
    clip_params = [param for param in model.parameters() if param.requires_grad]
    _set_training_mode(model, args)
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.leaning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    dataloader = train_dataloader(
        args.data_dir,
        args.batch_size,
        args.num_worker,
        args.data,
        depth_cache_dir=_dpga_depth_cache_dir(args),
        depth_split=getattr(args, "dpga_train_depth_split", "train"),
        split_json=getattr(args, "dpga_train_split_json", ""),
        split_name=getattr(args, "dpga_train_split_name", ""),
        hard_sampler_json=getattr(args, "dpga_hard_sampler_json", ""),
        hard_sampler_split_name=getattr(args, "dpga_hard_sampler_split_name", ""),
        hard_sampler_seed=getattr(args, "dpga_hard_sampler_seed", 3407),
        hard_sampler_hard_ratio=getattr(args, "dpga_hard_sampler_hard_ratio", 1.0 / 3.0),
        hard_sampler_medium_ratio=getattr(args, "dpga_hard_sampler_medium_ratio", 1.0 / 3.0),
        hard_sampler_batches_per_epoch=getattr(args, "dpga_hard_sampler_batches_per_epoch", 0),
    )
    max_iter = len(dataloader)
    warmup_epochs=3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1

    end_epoch = args.stop_epoch if args.stop_epoch > 0 else args.num_epoch
    if end_epoch < epoch:
        raise ValueError(f'stop_epoch {end_epoch} is earlier than resume epoch {epoch}')

    for epoch_idx in range(epoch, end_epoch + 1):
        _set_training_mode(model, args)

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img, depth, bucket_label = _split_batch(batch_data, device, args)

            optimizer.zero_grad()
            pred_img = _model_forward(model, input_img, depth, args)
            if (
                getattr(args, "arch", "convir") == "apdr"
                and getattr(args, "apdr_loss_scales", "all") == "full_only"
            ):
                scale_pairs = [(pred_img[2], label_img)]
                apdr_targets = [label_img, label_img, label_img]
            else:
                label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
                label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
                scale_pairs = [
                    (pred_img[0], label_img4),
                    (pred_img[1], label_img2),
                    (pred_img[2], label_img),
                ]
                apdr_targets = [label_img4, label_img2, label_img]

            dpga_tc_reg = {}
            dpga_rec_weight = None
            if getattr(args, "arch", "convir") == "dpga":
                dpga_tc_reg, _anchor, dpga_rec_weight = _dpga_tail_control_regularization(
                    model,
                    input_img,
                    label_img,
                    depth,
                    pred_img[2],
                    args,
                    bucket_label=bucket_label,
                )
                dpga_gate_reg = _dpga_hard_gate_supervision(
                    model,
                    bucket_label,
                    label_img,
                    args,
                )
                dpga_tc_reg.update(dpga_gate_reg)

            weighted_scale_pairs = []
            for pred, target in scale_pairs:
                weight = None
                if dpga_rec_weight is not None:
                    weight = F.interpolate(
                        dpga_rec_weight,
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                weighted_scale_pairs.append((pred, target, weight))
            loss_content = sum(
                _reconstruction_loss(pred, target, args, weight=weight)
                for pred, target, weight in weighted_scale_pairs
            )
            loss_fft_terms = []
            for pred, target in scale_pairs:
                label_fft = torch.fft.fft2(target, dim=(-2,-1))
                label_fft = torch.stack((label_fft.real, label_fft.imag), -1)
                pred_fft = torch.fft.fft2(pred, dim=(-2,-1))
                pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
                loss_fft_terms.append(criterion(pred_fft, label_fft))
            loss_fft = sum(loss_fft_terms)

            fft_weight = args.dpga_tc_fft_lambda if getattr(args, "arch", "convir") == "dpga" else 0.1
            loss = loss_content + fft_weight * loss_fft
            apdr_reg = {}
            if hasattr(model, 'apdr_regularization'):
                apdr_reg = model.apdr_regularization()
            apdr_train_reg = {}
            if getattr(args, "arch", "convir") == "dpga":
                if _dpga_tail_control_enabled(args) and dpga_tc_reg:
                    loss = (
                        loss
                        + args.dpga_tc_anchor_lambda * dpga_tc_reg["dpga_tc_anchor"]
                        + args.dpga_tc_chroma_lambda * dpga_tc_reg["dpga_tc_chroma"]
                        + args.dpga_tc_delta_lambda * dpga_tc_reg["dpga_tc_delta"]
                        + args.dpga_tc_delta_tv_lambda * dpga_tc_reg["dpga_tc_delta_tv"]
                    )
                if "dpga_hard_gate_bce" in dpga_tc_reg:
                    loss = loss + args.dpga_hard_gate_lambda * dpga_tc_reg["dpga_hard_gate_bce"]
                fusion_delta_lambda = float(getattr(args, "dpga_fusion_delta_lambda", 0.0))
                if fusion_delta_lambda > 0 and hasattr(model, "dpga_fusion_delta_regularization"):
                    fusion_delta = model.dpga_fusion_delta_regularization()
                    if fusion_delta is not None:
                        dpga_tc_reg["dpga_fusion_delta_norm"] = fusion_delta
                        loss = loss + fusion_delta_lambda * fusion_delta
            if hasattr(model, 'apdr_training_regularization') and getattr(args, "arch", "convir") == "apdr":
                apdr_train_reg = model.apdr_training_regularization(
                    apdr_targets,
                    risk_temperature=args.apdr_risk_temperature,
                )
                loss = (
                    loss
                    + args.apdr_anchor_lambda * apdr_train_reg.get("apdr_anchor", 0.0)
                    + args.apdr_gate_lambda * apdr_train_reg.get("apdr_gate", 0.0)
                    + args.apdr_residual_lambda * apdr_train_reg.get("apdr_residual", 0.0)
                    + getattr(args, "apdr_delta_lambda", 0.0)
                    * apdr_train_reg.get("apdr_delta_supervision", 0.0)
                    + getattr(args, "apdr_gate_supervision_lambda", 0.0)
                    * apdr_train_reg.get("apdr_gate_supervision", 0.0)
                )
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip_norm)
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            if (iter_idx + 1) % args.print_freq == 0:
                apdr_detail = ""
                if apdr_train_reg:
                    apdr_detail = (
                        " APDR anchor: %.8f delta: %.8f gate_sup: %.8f "
                        "gate: %.8f residual: %.8f"
                    ) % (
                        apdr_train_reg.get("apdr_anchor", 0.0).detach().item(),
                        apdr_train_reg.get("apdr_delta_supervision", 0.0).detach().item(),
                        apdr_train_reg.get("apdr_gate_supervision", 0.0).detach().item(),
                        apdr_train_reg.get("apdr_gate", 0.0).detach().item(),
                        apdr_train_reg.get("apdr_residual", 0.0).detach().item(),
                    )
                dpga_tc_detail = ""
                if dpga_tc_reg:
                    zero = torch.zeros((), device=device)
                    dpga_tc_detail = (
                        " DPGA_TC anchor: %.8f chroma: %.8f delta: %.8f "
                        "delta_tv: %.8f mask: %.8f high_anchor: %.8f sky: %.8f "
                        "hard_img: %.8f easy_img: %.8f hard_weight: %.8f hard_weight_max: %.8f "
                        "hard_gate: %.8f hard_gate_bce: %.8f fusion_delta: %.8f"
                    ) % (
                        dpga_tc_reg.get("dpga_tc_anchor", zero).detach().item(),
                        dpga_tc_reg.get("dpga_tc_chroma", zero).detach().item(),
                        dpga_tc_reg.get("dpga_tc_delta", zero).detach().item(),
                        dpga_tc_reg.get("dpga_tc_delta_tv", zero).detach().item(),
                        dpga_tc_reg["mask_ratio"].detach().item(),
                        dpga_tc_reg["high_anchor_ratio"].detach().item(),
                        dpga_tc_reg["sky_ratio"].detach().item(),
                        dpga_tc_reg["hard_image_ratio"].detach().item(),
                        dpga_tc_reg["easy_image_ratio"].detach().item(),
                        dpga_tc_reg.get("hard_region_weight_mean", zero).detach().item(),
                        dpga_tc_reg.get("hard_region_weight_max", zero).detach().item(),
                        dpga_tc_reg.get("dpga_hard_gate_mean", zero).detach().item(),
                        dpga_tc_reg.get("dpga_hard_gate_bce", zero).detach().item(),
                        dpga_tc_reg.get("dpga_fusion_delta_norm", zero).detach().item(),
                    )
                print(("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f "
                       "Loss content: %7.4f Loss fft: %7.4f%s%s") % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter,
                    scheduler.get_lr()[0], iter_pixel_adder.average(),
                    iter_fft_adder.average(), apdr_detail, dpga_tc_detail))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            val = _valid(model, args, epoch_idx)
            _log_modulation_stats(model, args, epoch_idx, device)
            _log_apdr_stats(model, args, epoch_idx, device)
            _log_dpga_stats(model, args, epoch_idx)
            _set_training_mode(model, args)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)
            if val >= best_psnr:
                best_psnr = val
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
