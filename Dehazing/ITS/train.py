import os
import torch
from data import train_dataloader, valid_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torch.nn as nn

from warmup_scheduler import GradualWarmupScheduler


def _is_convir_wd(args):
    return getattr(args, 'arch', '') == 'convir_wd_lite'


def _is_wd_param(name):
    return name.startswith('WD_')


def _is_decoder_param(name):
    return (
        name.startswith('Decoder.')
        or name.startswith('Convs.')
        or name.startswith('ConvsOut.')
        or name.startswith('feat_extract.3.')
        or name.startswith('feat_extract.4.')
        or name.startswith('feat_extract.5.')
    )


def _configure_trainable_parameters(model, args, learning_rate):
    if not _is_convir_wd(args):
        total = sum(p.numel() for p in model.parameters())
        print(f'TRAINABLE_SCOPE official_all trainable={total} frozen=0')
        return [{'params': model.parameters(), 'lr': learning_rate, 'name': 'all'}]

    scope = getattr(args, 'convir_wd_train_scope', 'all')
    for _, param in model.named_parameters():
        param.requires_grad = False

    groups = []
    wd_params = []
    decoder_params = []
    other_params = []
    for name, param in model.named_parameters():
        if scope == 'all':
            param.requires_grad = True
            other_params.append(param)
        elif _is_wd_param(name):
            param.requires_grad = True
            wd_params.append(param)
        elif scope == 'wd_decoder' and _is_decoder_param(name):
            param.requires_grad = True
            decoder_params.append(param)

    if wd_params:
        groups.append({'params': wd_params, 'lr': learning_rate, 'name': 'WD'})
    if decoder_params:
        groups.append({
            'params': decoder_params,
            'lr': getattr(args, 'convir_wd_decoder_learning_rate', 1e-5),
            'name': 'decoder',
        })
    if other_params:
        groups.append({'params': other_params, 'lr': learning_rate, 'name': 'all'})

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    group_desc = ','.join(f"{group['name']}@{group['lr']}" for group in groups)
    print(f'TRAINABLE_SCOPE convir_wd scope={scope} trainable={trainable} frozen={frozen} groups={group_desc}')
    if trainable == 0:
        raise ValueError(f'No trainable parameters for convir_wd_train_scope={scope}')
    return groups


def _haar_dwt2(x):
    pad_h = x.shape[-2] % 2
    pad_w = x.shape[-1] % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 - x01 + x10 - x11)
    hl = 0.5 * (x00 + x01 - x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)
    return ll, lh, hl, hh


def _rgb_to_y(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _convir_wd_aux_loss(pred, label, criterion, args):
    pred_ll, pred_lh, pred_hl, pred_hh = _haar_dwt2(pred)
    label_ll, label_lh, label_hl, label_hh = _haar_dwt2(label)
    low = criterion(pred_ll, label_ll)
    high = (
        criterion(pred_lh, label_lh)
        + criterion(pred_hl, label_hl)
        + criterion(pred_hh, label_hh)
    ) / 3.0
    y_loss = criterion(_rgb_to_y(pred), _rgb_to_y(label))
    return (
        getattr(args, 'convir_wd_dwt_low_weight', 0.05) * low
        + getattr(args, 'convir_wd_dwt_high_weight', 0.01) * high
        + getattr(args, 'convir_wd_y_weight', 0.05) * y_loss
    )


def _log_modulation_stats(model, args, epoch_idx, device):
    if args.mod_stats_freq <= 0 or epoch_idx % args.mod_stats_freq != 0:
        return
    if hasattr(model, 'collect_wd_stats'):
        dataloader = valid_dataloader(args.data_dir, args.data, batch_size=1, num_workers=0)
        sums = {}
        count = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                if args.mod_stats_batches > 0 and batch_idx >= args.mod_stats_batches:
                    break
                input_img = batch_data[0].to(device)
                batch_stats = model.collect_wd_stats(input_img)
                for block_name, block_stats in batch_stats.items():
                    sums.setdefault(block_name, {})
                    for key, value in block_stats.items():
                        sums[block_name][key] = sums[block_name].get(key, 0.0) + value
                count += 1
        model.train()
        if count == 0:
            return
        for block_name in sorted(sums):
            averaged = {key: value / count for key, value in sorted(sums[block_name].items())}
            print(
                "WD_STATS Epoch: %03d Block: %s Samples: %d "
                "gamma_abs_mean: %.8f beta_abs_mean: %.8f" % (
                    epoch_idx,
                    block_name,
                    count,
                    averaged.get('gamma_abs_mean', 0.0),
                    averaged.get('beta_abs_mean', 0.0),
                )
            )
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

    learning_rate = getattr(args, 'learning_rate', getattr(args, 'leaning_rate', 1e-4))
    optimizer = torch.optim.Adam(
        _configure_trainable_parameters(model, args, learning_rate),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker, args.data)
    max_iter = len(dataloader)
    warmup_epochs=3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch-warmup_epochs, eta_min=1e-6)
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

            loss = loss_content + 0.1 * loss_fft
            if _is_convir_wd(args):
                loss = loss + _convir_wd_aux_loss(pred_img[2], label_img, criterion, args)
            loss.backward()
            grad_clip_norm = getattr(args, 'grad_clip_norm', 0.001)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip_norm,
                )
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
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)
            if val >= best_psnr:
                best_psnr = val
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
