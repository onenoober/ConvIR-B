import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f


def _unpack_valid_batch(data):
    if len(data) == 3:
        input_img, label_img, depth = data
        return input_img, label_img, depth
    input_img, label_img = data
    return input_img, label_img, None


def _forward_model(model, input_img, depth, args):
    if getattr(args, 'arch', '') == 'dta':
        if depth is None and getattr(args, 'dta_require_depth', False):
            raise ValueError('DTA validation requires depth but no depth tensor was returned.')
        return model(input_img, depth)
    return model(input_img)


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth_cache_dir = args.dta_depth_cache_dir if getattr(args, 'arch', '') == 'dta' else ''
    dataset = valid_dataloader(
        args.data_dir,
        args.data,
        batch_size=1,
        num_workers=0,
        depth_cache_dir=depth_cache_dir,
        depth_split=args.dta_eval_depth_split,
        split_json=args.split_json,
        split_name=args.split_name,
    )
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')
        factor = 32
        for idx, data in enumerate(dataset):
            input_img, label_img, depth = _unpack_valid_batch(data)
            input_img = input_img.to(device)
            depth = depth.to(device) if depth is not None else None

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')
            if depth is not None:
                depth = f.pad(depth, (0, padw, 0, padh), 'reflect')

            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = _forward_model(model, input_img, depth, args)[2]
            pred = pred[:,:,:h,:w]

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)

            psnr_adder(psnr)
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average()
