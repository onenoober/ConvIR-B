import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f


def _dpga_depth_cache_dir(args):
    if getattr(args, "arch", "convir") != "dpga":
        return ""
    depth_cache_dir = getattr(args, "dpga_depth_cache_dir", "")
    if not depth_cache_dir:
        raise ValueError("--dpga_depth_cache_dir is required when --arch dpga")
    return depth_cache_dir


def _forward(model, input_img, depth, args):
    if getattr(args, "arch", "convir") == "dpga":
        return model(input_img, depth)
    return model(input_img)


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_json = getattr(args, "dpga_valid_split_json", "")
    split_name = getattr(args, "dpga_valid_split_name", "")
    depth_split = getattr(args, "dpga_eval_depth_split", "test")
    if getattr(args, "arch", "convir") == "dpga" and split_json and split_name and depth_split == "test":
        depth_split = "train"
    dataset = valid_dataloader(
        args.data_dir,
        args.data,
        batch_size=1,
        num_workers=0,
        depth_cache_dir=_dpga_depth_cache_dir(args),
        depth_split=depth_split,
        split_json=split_json,
        split_name=split_name,
    )
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')
        factor = 32
        for idx, data in enumerate(dataset):
            if getattr(args, "arch", "convir") == "dpga":
                input_img, label_img, depth, _name = data
                depth = depth.to(device)
            else:
                input_img, label_img = data
                depth = None
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')
            if depth is not None:
                depth = f.pad(depth, (0, padw, 0, padh), 'reflect')

            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = _forward(model, input_img, depth, args)[2]
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
