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

    dataloader = valid_dataloader(args.data_dir, args.data, batch_size=1, num_workers=0)
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


def _log_apdr_stats(model, args, epoch_idx, device):
    if args.mod_stats_freq <= 0 or epoch_idx % args.mod_stats_freq != 0:
        return
    if not hasattr(model, 'collect_apdr_stats'):
        return

    dataloader = valid_dataloader(args.data_dir, args.data, batch_size=1, num_workers=0)
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
    if getattr(args, "arch", "convir") != "apdr":
        return list(model.parameters())

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
        getattr(args, "arch", "convir") == "apdr"
        and getattr(args, "apdr_train_scope", "all") != "all"
    ):
        model.eval()
        for name, module in model.named_modules():
            if name.startswith("APDR_"):
                module.train()
        return
    model.train()


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
    _set_training_mode(model, args)
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker, args.data)
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

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)
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

            loss_content = sum(criterion(pred, target) for pred, target in scale_pairs)
            loss_fft_terms = []
            for pred, target in scale_pairs:
                label_fft = torch.fft.fft2(target, dim=(-2,-1))
                label_fft = torch.stack((label_fft.real, label_fft.imag), -1)
                pred_fft = torch.fft.fft2(pred, dim=(-2,-1))
                pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
                loss_fft_terms.append(criterion(pred_fft, label_fft))
            loss_fft = sum(loss_fft_terms)

            loss = loss_content + 0.1 * loss_fft
            apdr_reg = {}
            if hasattr(model, 'apdr_regularization'):
                apdr_reg = model.apdr_regularization()
            apdr_train_reg = {}
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
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
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
                print(("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f "
                       "Loss content: %7.4f Loss fft: %7.4f%s") % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter,
                    scheduler.get_lr()[0], iter_pixel_adder.average(),
                    iter_fft_adder.average(), apdr_detail))
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
            _set_training_mode(model, args)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)
            if val >= best_psnr:
                best_psnr = val
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
