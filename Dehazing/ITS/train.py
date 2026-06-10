import os
import torch
from data import train_dataloader, valid_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torch.nn as nn

from warmup_scheduler import GradualWarmupScheduler


def _unpack_train_batch(batch_data):
    if len(batch_data) == 3:
        input_img, label_img, depth = batch_data
        return input_img, label_img, depth
    input_img, label_img = batch_data
    return input_img, label_img, None


def _forward_model(model, input_img, depth, args):
    if getattr(args, 'arch', '') == 'dta':
        if depth is None and getattr(args, 'dta_require_depth', False):
            raise ValueError('DTA route requires depth but the dataloader returned no depth tensor.')
        return model(input_img, depth)
    return model(input_img)


def _apply_train_scope(model, args):
    train_scope = getattr(args, 'train_scope', 'all')
    arch = getattr(args, 'arch', 'official_convir')
    if arch != 'dta' or train_scope == 'all':
        for param in model.parameters():
            param.requires_grad = True
    else:
        if train_scope == 'adapter_only':
            prefixes = ('DTA.',)
        elif train_scope == 'adapter_neighbors':
            prefixes = ('DTA.', 'FAM1.', 'FAM2.', 'Convs.')
        else:
            raise ValueError(f'Unsupported train_scope: {train_scope}')
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(prefixes)

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(f'TRAIN_SCOPE scope={train_scope} arch={arch} trainable={trainable} total={total}')
    if trainable <= 0:
        raise ValueError(f'No trainable parameters for train_scope={train_scope}')


def _trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def _log_modulation_stats(model, args, epoch_idx, device):
    if args.mod_stats_freq <= 0 or epoch_idx % args.mod_stats_freq != 0:
        return
    if not hasattr(model, 'collect_modulation_stats'):
        return

    depth_cache_dir = args.dta_depth_cache_dir if getattr(args, 'arch', '') == 'dta' else ''
    dataloader = valid_dataloader(
        args.data_dir,
        args.data,
        batch_size=1,
        num_workers=0,
        depth_cache_dir=depth_cache_dir,
        depth_split=args.dta_eval_depth_split,
        split_json=args.split_json,
        split_name=args.split_name,
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


def _log_dta_stats(model, args, epoch_idx, device):
    if args.mod_stats_freq <= 0 or epoch_idx % args.mod_stats_freq != 0:
        return
    if not hasattr(model, 'collect_dta_stats'):
        return
    dataloader = valid_dataloader(
        args.data_dir,
        args.data,
        batch_size=1,
        num_workers=0,
        depth_cache_dir=args.dta_depth_cache_dir,
        depth_split=args.dta_eval_depth_split,
        split_json=args.split_json,
        split_name=args.split_name,
    )
    sums = {}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if args.mod_stats_batches > 0 and batch_idx >= args.mod_stats_batches:
                break
            input_img, _, depth = _unpack_train_batch(batch_data)
            input_img = input_img.to(device)
            depth = depth.to(device) if depth is not None else None
            batch_stats = model.collect_dta_stats(input_img, depth)
            for key, value in batch_stats.items():
                sums[key] = sums.get(key, 0.0) + value
            count += 1
    model.train()
    if count == 0:
        return
    averaged = {key: value / count for key, value in sorted(sums.items())}
    print(
        "DTA_STATS Epoch: %03d Samples: %d "
        "stage2_gate_mean: %.8f stage2_gate_max: %.8f "
        "stage3_gate_mean: %.8f stage3_gate_max: %.8f "
        "stage2_delta_abs_mean: %.8f stage3_delta_abs_mean: %.8f "
        "t_pred_mean: %.8f t_pred_std: %.8f" % (
            epoch_idx,
            count,
            averaged.get('stage2_gate_mean', 0.0),
            averaged.get('stage2_gate_max', 0.0),
            averaged.get('stage3_gate_mean', 0.0),
            averaged.get('stage3_gate_max', 0.0),
            averaged.get('stage2_delta_abs_mean', 0.0),
            averaged.get('stage3_delta_abs_mean', 0.0),
            averaged.get('t_pred_mean', 0.0),
            averaged.get('t_pred_std', 0.0),
        )
    )


def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()
    _apply_train_scope(model, args)

    learning_rate = getattr(args, 'learning_rate', getattr(args, 'leaning_rate', 1e-4))
    optimizer = torch.optim.Adam(_trainable_parameters(model), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    depth_cache_dir = args.dta_depth_cache_dir if getattr(args, 'arch', '') == 'dta' else ''
    dataloader = train_dataloader(
        args.data_dir,
        args.batch_size,
        args.num_worker,
        args.data,
        depth_cache_dir=depth_cache_dir,
        depth_split=args.dta_train_depth_split,
        split_json=args.split_json,
        split_name=args.split_name,
    )
    max_iter = len(dataloader)
    warmup_epochs=3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.num_epoch-warmup_epochs), eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
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
    epoch_dta_rank_adder = Adder()
    epoch_dta_tv_adder = Adder()
    epoch_dta_proxy_adder = Adder()
    iter_dta_rank_adder = Adder()
    iter_dta_tv_adder = Adder()
    iter_dta_proxy_adder = Adder()

    end_epoch = args.stop_epoch if args.stop_epoch > 0 else args.num_epoch
    if end_epoch < epoch:
        raise ValueError(f'stop_epoch {end_epoch} is earlier than resume epoch {epoch}')

    for epoch_idx in range(epoch, end_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img, depth = _unpack_train_batch(batch_data)
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            depth = depth.to(device) if depth is not None else None

            optimizer.zero_grad()
            pred_img = _forward_model(model, input_img, depth, args)
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1+l2+l3

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

            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1+f2+f3

            loss_dta_rank = input_img.new_zeros(())
            loss_dta_tv = input_img.new_zeros(())
            loss_dta_proxy = input_img.new_zeros(())
            if getattr(args, 'arch', '') == 'dta' and hasattr(model, 'dta_auxiliary_losses') and depth is not None:
                dta_losses = model.dta_auxiliary_losses(
                    rank_pairs=args.dta_rank_pairs,
                    min_depth_gap=args.dta_rank_min_depth_gap,
                )
                loss_dta_rank = dta_losses['rank']
                loss_dta_tv = dta_losses['tv']
                loss_dta_proxy = dta_losses['proxy']

            loss = (
                loss_content
                + 0.1 * loss_fft
                + args.dta_rank_weight * loss_dta_rank
                + args.dta_tv_weight * loss_dta_tv
                + args.dta_proxy_weight * loss_dta_proxy
            )
            loss.backward()
            grad_clip_norm = getattr(args, 'grad_clip_norm', 0.001)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())
            iter_dta_rank_adder(loss_dta_rank.item())
            iter_dta_tv_adder(loss_dta_tv.item())
            iter_dta_proxy_adder(loss_dta_proxy.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())
            epoch_dta_rank_adder(loss_dta_rank.item())
            epoch_dta_tv_adder(loss_dta_tv.item())
            epoch_dta_proxy_adder(loss_dta_proxy.item())

            if (iter_idx + 1) % args.print_freq == 0:
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f Loss dta_rank: %7.4f Loss dta_tv: %7.4f Loss dta_proxy: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, scheduler.get_lr()[0], iter_pixel_adder.average(),
                    iter_fft_adder.average(), iter_dta_rank_adder.average(), iter_dta_tv_adder.average(), iter_dta_proxy_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                writer.add_scalar('DTA Rank Loss', iter_dta_rank_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                writer.add_scalar('DTA TV Loss', iter_dta_tv_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
                iter_dta_rank_adder.reset()
                iter_dta_tv_adder.reset()
                iter_dta_proxy_adder.reset()
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f Epoch DTA Rank: %7.4f Epoch DTA TV: %7.4f Epoch DTA Proxy: %7.4f" % (
            epoch_idx,
            epoch_timer.toc(),
            epoch_pixel_adder.average(),
            epoch_fft_adder.average(),
            epoch_dta_rank_adder.average(),
            epoch_dta_tv_adder.average(),
            epoch_dta_proxy_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        epoch_dta_rank_adder.reset()
        epoch_dta_tv_adder.reset()
        epoch_dta_proxy_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            val = _valid(model, args, epoch_idx)
            _log_modulation_stats(model, args, epoch_idx, device)
            _log_dta_stats(model, args, epoch_idx, device)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)
            if val >= best_psnr:
                best_psnr = val
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
