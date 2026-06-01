import os
import torch
from data import train_dataloader, valid_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torch.nn as nn

from warmup_scheduler import GradualWarmupScheduler


def _per_image_l1(pred, target):
    return (pred - target).abs().flatten(1).mean(dim=1)


def _easy_rank_weights(per_image_loss):
    batch_size = per_image_loss.numel()
    if batch_size <= 1:
        return torch.ones_like(per_image_loss)
    order = torch.argsort(per_image_loss, descending=False)
    ranks = torch.empty_like(per_image_loss)
    ranks[order] = torch.arange(batch_size, device=per_image_loss.device, dtype=per_image_loss.dtype)
    return 1.0 - ranks / float(batch_size - 1)


def _hard_rank_weights(per_image_loss):
    batch_size = per_image_loss.numel()
    if batch_size <= 1:
        return torch.ones_like(per_image_loss)
    order = torch.argsort(per_image_loss, descending=True)
    ranks = torch.empty_like(per_image_loss)
    ranks[order] = torch.arange(batch_size, device=per_image_loss.device, dtype=per_image_loss.dtype)
    return 1.0 - ranks / float(batch_size - 1)


def _current_gate_lambda(args, epoch_idx):
    gate_lambda = getattr(args, 'gate_lambda', 0.0)
    if gate_lambda <= 0:
        return 0.0
    if epoch_idx <= getattr(args, 'gate_warmup_epochs', 5):
        return 0.0
    ramp_epochs = max(1, getattr(args, 'gate_ramp_epochs', 5))
    ramp = min(1.0, float(epoch_idx - getattr(args, 'gate_warmup_epochs', 5)) / float(ramp_epochs))
    return gate_lambda * ramp


def _current_hard_aux_lambda(args, epoch_idx):
    hard_aux_lambda = getattr(args, 'hard_aux_lambda', 0.0)
    if getattr(args, 'loss_mode', 'original') != 'hard_aux' or hard_aux_lambda <= 0:
        return 0.0
    if epoch_idx <= getattr(args, 'hard_aux_warmup_epochs', 3):
        return 0.0
    ramp_epochs = max(1, getattr(args, 'hard_aux_ramp_epochs', 5))
    ramp = min(1.0, float(epoch_idx - getattr(args, 'hard_aux_warmup_epochs', 3)) / float(ramp_epochs))
    return hard_aux_lambda * ramp


def _hard_aux_loss(restore_loss_per_image):
    hard_weight = _hard_rank_weights(restore_loss_per_image.detach())
    focus_weight = 1.0 + hard_weight
    focus_weight = focus_weight / focus_weight.mean().clamp_min(1e-6)
    return (focus_weight * restore_loss_per_image).mean()


def _log_modulation_stats(model, args, epoch_idx, device):
    mod_stats_freq = getattr(args, 'mod_stats_freq', 0)
    if mod_stats_freq <= 0 or epoch_idx % mod_stats_freq != 0:
        return
    if not hasattr(model, 'collect_modulation_stats'):
        return

    dataloader = valid_dataloader(args.data_dir, args.data, batch_size=1, num_workers=0)
    sums = {}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if getattr(args, 'mod_stats_batches', 64) > 0 and batch_idx >= getattr(args, 'mod_stats_batches', 64):
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
            "gamma_mean: %.8f gamma_abs_mean: %.8f gamma_std: %.8f "
            "gamma_min: %.8f gamma_max: %.8f gamma_abs_gt_0.5: %.8f "
            "gamma_abs_gt_0.05: %.8f gamma_abs_gt_0.10: %.8f gamma_abs_gt_0.09: %.8f "
            "gate_mean: %.8f gate_std: %.8f gate_min: %.8f gate_max: %.8f "
            "gamma_base_abs_mean: %.8f effective_gamma_abs_mean: %.8f "
            "beta_present: %.0f beta_mean: %.8f beta_abs_mean: %.8f beta_std: %.8f "
            "beta_min: %.8f beta_max: %.8f beta_abs_gt_0.1: %.8f "
            "beta_abs_gt_0.02: %.8f beta_abs_gt_0.05: %.8f" % (
                epoch_idx,
                fam_name,
                count,
                averaged.get('gamma_mean', 0.0),
                averaged.get('gamma_abs_mean', 0.0),
                averaged.get('gamma_std', 0.0),
                averaged.get('gamma_min', 0.0),
                averaged.get('gamma_max', 0.0),
                averaged.get('gamma_abs_gt_0.5', 0.0),
                averaged.get('gamma_abs_gt_0.05', 0.0),
                averaged.get('gamma_abs_gt_0.10', 0.0),
                averaged.get('gamma_abs_gt_0.09', 0.0),
                averaged.get('gate_mean', 0.0),
                averaged.get('gate_std', 0.0),
                averaged.get('gate_min', 0.0),
                averaged.get('gate_max', 0.0),
                averaged.get('gamma_base_abs_mean', 0.0),
                averaged.get('effective_gamma_abs_mean', 0.0),
                averaged.get('beta_present', 0.0),
                averaged.get('beta_mean', 0.0),
                averaged.get('beta_abs_mean', 0.0),
                averaged.get('beta_std', 0.0),
                averaged.get('beta_min', 0.0),
                averaged.get('beta_max', 0.0),
                averaged.get('beta_abs_gt_0.1', 0.0),
                averaged.get('beta_abs_gt_0.02', 0.0),
                averaged.get('beta_abs_gt_0.05', 0.0),
            )
        )


def _log_scm_stats(model, args, epoch_idx, device):
    scm_stats_freq = getattr(args, 'scm_stats_freq', 0)
    if scm_stats_freq <= 0 or epoch_idx % scm_stats_freq != 0:
        return
    if not hasattr(model, 'collect_scm_stats'):
        return

    dataloader = valid_dataloader(args.data_dir, args.data, batch_size=1, num_workers=0)
    sums = {}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if getattr(args, 'scm_stats_batches', 64) > 0 and batch_idx >= getattr(args, 'scm_stats_batches', 64):
                break
            input_img = batch_data[0].to(device)
            batch_stats = model.collect_scm_stats(input_img)
            for scm_name, scm_stats in batch_stats.items():
                sums.setdefault(scm_name, {})
                for key, value in scm_stats.items():
                    sums[scm_name][key] = sums[scm_name].get(key, 0.0) + value
            count += 1
    model.train()

    if count == 0:
        return
    for scm_name in sorted(sums):
        averaged = {key: value / count for key, value in sorted(sums[scm_name].items())}
        print(
            "SCM_STATS Epoch: %03d SCM: %s Samples: %d "
            "prior_min_mean: %.8f prior_max_mean: %.8f prior_dark_mean: %.8f "
            "prior_saturation_mean: %.8f prior_grad_mean: %.8f "
            "rgb_abs_mean: %.8f prior_branch_abs_mean: %.8f prior_to_rgb_abs_ratio: %.8f" % (
                epoch_idx,
                scm_name,
                count,
                averaged.get('prior_min_mean', 0.0),
                averaged.get('prior_max_mean', 0.0),
                averaged.get('prior_dark_mean', 0.0),
                averaged.get('prior_saturation_mean', 0.0),
                averaged.get('prior_grad_mean', 0.0),
                averaged.get('rgb_abs_mean', 0.0),
                averaged.get('prior_branch_abs_mean', 0.0),
                averaged.get('prior_to_rgb_abs_ratio', 0.0),
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
    epoch_gate_adder = Adder()
    epoch_hard_aux_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    iter_gate_adder = Adder()
    iter_hard_aux_adder = Adder()
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
            l1_per_image = _per_image_l1(pred_img[0], label_img4)
            l2_per_image = _per_image_l1(pred_img[1], label_img2)
            l3_per_image = _per_image_l1(pred_img[2], label_img)
            loss_content_per_image = l1_per_image + l2_per_image + l3_per_image
            loss_content = loss_content_per_image.mean()

            label_fft1 = torch.fft.fft2(label_img4, dim=(-2,-1))
            label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

            pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2,-1))
            pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

            label_fft2 = torch.fft.fft2(label_img2, dim=(-2,-1))
            label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

            pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2,-1))
            pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

            label_fft3 = torch.fft.fft2(label_img, dim=(-2,-1))
            label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

            pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2,-1))
            pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

            f1_per_image = _per_image_l1(pred_fft1, label_fft1)
            f2_per_image = _per_image_l1(pred_fft2, label_fft2)
            f3_per_image = _per_image_l1(pred_fft3, label_fft3)
            loss_fft_per_image = f1_per_image + f2_per_image + f3_per_image
            loss_fft = loss_fft_per_image.mean()

            restore_loss_per_image = loss_content_per_image + 0.1 * loss_fft_per_image
            loss_restore = restore_loss_per_image.mean()
            loss = loss_restore
            hard_aux_lambda = _current_hard_aux_lambda(args, epoch_idx)
            hard_aux_loss = None
            if hard_aux_lambda > 0:
                hard_aux_loss = _hard_aux_loss(restore_loss_per_image)
                loss = loss_restore + hard_aux_lambda * (hard_aux_loss - loss_restore)
            gate_lambda = _current_gate_lambda(args, epoch_idx)
            gate_loss = None
            if gate_lambda > 0 and hasattr(model, 'gate_budget_loss'):
                easy_weight = _easy_rank_weights(restore_loss_per_image.detach())
                gate_loss = model.gate_budget_loss(easy_weight)
                if gate_loss is not None:
                    loss = loss + gate_lambda * gate_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())
            gate_loss_value = 0.0 if gate_loss is None else gate_loss.item()
            hard_aux_loss_value = loss_restore.item() if hard_aux_loss is None else hard_aux_loss.item()
            iter_gate_adder(gate_loss_value)
            iter_hard_aux_adder(hard_aux_loss_value)

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())
            epoch_gate_adder(gate_loss_value)
            epoch_hard_aux_adder(hard_aux_loss_value)

            if (iter_idx + 1) % args.print_freq == 0:
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f Hard aux loss: %7.4f Hard aux lambda: %.6f Gate loss: %.6f Gate lambda: %.6f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, scheduler.get_lr()[0], iter_pixel_adder.average(),
                    iter_fft_adder.average(), iter_hard_aux_adder.average(), hard_aux_lambda, iter_gate_adder.average(), gate_lambda))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                writer.add_scalar('Hard Aux Loss', iter_hard_aux_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                writer.add_scalar('Gate Loss', iter_gate_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
                iter_gate_adder.reset()
                iter_hard_aux_adder.reset()
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f Epoch Hard Aux Loss: %7.4f Epoch Gate Loss: %.6f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average(), epoch_hard_aux_adder.average(), epoch_gate_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        epoch_gate_adder.reset()
        epoch_hard_aux_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            val = _valid(model, args, epoch_idx)
            _log_modulation_stats(model, args, epoch_idx, device)
            _log_scm_stats(model, args, epoch_idx, device)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)
            if val >= best_psnr:
                best_psnr = val
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
