import os
import math
import torch
from data import train_dataloader, valid_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torch.nn as nn

from warmup_scheduler import GradualWarmupScheduler


def _fft_l1_loss(pred, target, criterion, return_band_stats=False):
    pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
    target_fft = torch.fft.fft2(target, dim=(-2, -1))
    loss = criterion(
        torch.stack((pred_fft.real, pred_fft.imag), -1),
        torch.stack((target_fft.real, target_fft.imag), -1),
    )
    if return_band_stats:
        return loss, _ortho_fft_full_band_diagnostics(pred_fft, target_fft)
    return loss


def _ortho_fft_full_band_diagnostics(pred_fft, target_fft):
    with torch.no_grad():
        height, width = pred_fft.shape[-2], pred_fft.shape[-1]
        diff = torch.fft.fftshift(
            (pred_fft.detach() - target_fft.detach()).abs() / math.sqrt(height * width),
            dim=(-2, -1),
        )
        yy = torch.linspace(-1.0, 1.0, height, device=diff.device, dtype=diff.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=diff.device, dtype=diff.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        radius = torch.sqrt(grid_x * grid_x + grid_y * grid_y)
        low = radius <= 0.20
        mid = (radius > 0.20) & (radius <= 0.55)
        high = radius > 0.55

        def masked_mean(mask):
            mask = mask.view(1, 1, height, width)
            denom = mask.sum().clamp_min(1).to(diff.dtype) * diff.shape[0] * diff.shape[1]
            return (diff * mask).sum() / denom

        return {
            "loss_fft_full_low": masked_mean(low).item(),
            "loss_fft_full_mid": masked_mean(mid).item(),
            "loss_fft_full_high": masked_mean(high).item(),
            "normalized_fft_l1_full_ortho": diff.mean().item(),
        }


def _add_loss_detail(iter_adders, epoch_adders, name, value):
    if name not in iter_adders:
        iter_adders[name] = Adder()
        epoch_adders[name] = Adder()
    value = float(value.detach().item() if torch.is_tensor(value) else value)
    iter_adders[name](value)
    epoch_adders[name](value)


def _flatten_stat_dict(stats):
    flat = {}

    def visit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "flags":
                    continue
                visit(f"{prefix}{key.lower()}_", child)
        elif isinstance(value, (int, float)):
            flat[prefix[:-1]] = float(value)

    visit("", stats)
    return flat


def _log_pfd_stats(model, args, epoch_idx, device):
    if args.mod_stats_freq <= 0 or epoch_idx % args.mod_stats_freq != 0:
        return
    if not hasattr(model, 'collect_pfd_stats'):
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
                input_img = batch_data[0].to(device)
                flat_stats = _flatten_stat_dict(model.collect_pfd_stats(input_img))
                for key, value in flat_stats.items():
                    sums[key] = sums.get(key, 0.0) + value
                count += 1
    finally:
        model.train(was_training)

    if count == 0 or not sums:
        return
    averaged = {key: value / count for key, value in sorted(sums.items())}
    detail = " ".join(f"{key}: {value:.8f}" for key, value in averaged.items())
    print(f"PFD_STATS Epoch: {epoch_idx:03d} Samples: {count} {detail}")


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


def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
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
    epoch_loss_detail_adders = {}
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    iter_loss_detail_adders = {}
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1

    end_epoch = args.stop_epoch if args.stop_epoch > 0 else args.num_epoch
    if end_epoch < epoch:
        raise ValueError(f'stop_epoch {end_epoch} is earlier than resume epoch {epoch}')

    for epoch_idx in range(epoch, end_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1+l2+l3

            f1 = _fft_l1_loss(pred_img[0], label_img4, criterion)
            f2 = _fft_l1_loss(pred_img[1], label_img2, criterion)
            f3, fft_band_details = _fft_l1_loss(
                pred_img[2], label_img, criterion, return_band_stats=True
            )
            loss_fft = f1+f2+f3

            loss_details = {
                "loss_content_quarter": l1,
                "loss_content_half": l2,
                "loss_content_full": l3,
                "loss_fft_quarter": f1,
                "loss_fft_half": f2,
                "loss_fft_full": f3,
            }
            loss_details.update(fft_band_details)
            for detail_name, detail_value in loss_details.items():
                _add_loss_detail(
                    iter_loss_detail_adders,
                    epoch_loss_detail_adders,
                    detail_name,
                    detail_value,
                )

            loss = loss_content + 0.1 * loss_fft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            if (iter_idx + 1) % args.print_freq == 0:
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, scheduler.get_lr()[0], iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                for detail_name, detail_adder in sorted(iter_loss_detail_adders.items()):
                    writer.add_scalar(
                        f"Loss detail/{detail_name}",
                        detail_adder.average(),
                        iter_idx + (epoch_idx - 1) * max_iter,
                    )
                
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
                for detail_adder in iter_loss_detail_adders.values():
                    detail_adder.reset()
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_detail_values = {
            name: adder.average()
            for name, adder in sorted(epoch_loss_detail_adders.items())
            if adder.count > 0
        }
        if epoch_detail_values:
            detail = " ".join(
                f"{name}: {value:.8f}" for name, value in epoch_detail_values.items()
            )
            print(f"EPOCH_LOSS_DETAIL Epoch: {epoch_idx:03d} {detail}")
            for detail_name, detail_value in epoch_detail_values.items():
                writer.add_scalar(f"Epoch loss detail/{detail_name}", detail_value, epoch_idx)
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        for detail_adder in epoch_loss_detail_adders.values():
            detail_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            val = _valid(model, args, epoch_idx)
            _log_modulation_stats(model, args, epoch_idx, device)
            _log_pfd_stats(model, args, epoch_idx, device)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)
            if val >= best_psnr:
                best_psnr = val
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
