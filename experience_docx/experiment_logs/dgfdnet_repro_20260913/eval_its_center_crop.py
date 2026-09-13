import argparse
import csv
import os
import sys
import time

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio

# The wrapper lives outside the official entrypoint directory.
sys.path.insert(0, os.getcwd())

from data import test_dataloader
from models.DGFDNet import build_net


def center_crop_to(image, height, width):
    _, _, current_height, current_width = image.shape
    if current_height < height or current_width < width:
        raise ValueError(
            f"Cannot center-crop label {(current_height, current_width)} "
            f"to {(height, width)}"
        )
    top = (current_height - height) // 2
    left = (current_width - width) // 2
    return image[:, :, top:top + height, left:left + width]


def evaluate(args):
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model = build_net().to(device)
    state_dict = torch.load(args.test_model, map_location="cpu")
    model.load_state_dict(state_dict["model"], strict=True)
    model.eval()

    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    total_psnr = 0.0
    total_ssim = 0.0
    total_time = 0.0

    with open(args.csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "input_h", "input_w", "gt_h", "gt_w", "psnr", "ssim", "seconds"])
        with torch.no_grad():
            for index, (input_img, label_img, name) in enumerate(dataloader, start=1):
                input_img = input_img.to(device)
                label_img = label_img.to(device)

                start = time.time()
                pred = model(input_img)
                elapsed = time.time() - start
                pred = pred[:, :, :input_img.shape[2], :input_img.shape[3]]
                label_img = center_crop_to(label_img, pred.shape[2], pred.shape[3])
                pred_clip = torch.clamp(pred, 0, 1)

                pred_numpy = pred_clip.squeeze(0).cpu().numpy()
                label_numpy = label_img.squeeze(0).cpu().numpy()
                psnr_value = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
                down_ratio = max(1, round(min(pred.shape[2:]) / 256))
                pred_small = F.adaptive_avg_pool2d(
                    pred_clip,
                    (int(pred.shape[2] / down_ratio), int(pred.shape[3] / down_ratio)),
                )
                label_small = F.adaptive_avg_pool2d(
                    label_img,
                    (int(pred.shape[2] / down_ratio), int(pred.shape[3] / down_ratio)),
                )
                ssim_value = float(
                    ssim(pred_small, label_small, data_range=1, size_average=False).item()
                )
                total_psnr += psnr_value
                total_ssim += ssim_value
                total_time += elapsed
                writer.writerow(
                    [
                        name[0],
                        input_img.shape[2],
                        input_img.shape[3],
                        label_img.shape[2],
                        label_img.shape[3],
                        f"{psnr_value:.8f}",
                        f"{ssim_value:.8f}",
                        f"{elapsed:.8f}",
                    ]
                )
                print(
                    f"{index} iter PSNR: {psnr_value:.4f} "
                    f"SSIM: {ssim_value:.6f} time: {elapsed:.6f}",
                    flush=True,
                )

    count = len(dataloader)
    print("==========================================================")
    print(f"The average PSNR is {total_psnr / count:.4f} dB")
    print(f"The average SSIM is {total_ssim / count:.6f}")
    print(f"Average time: {total_time / count:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--test_model", required=True)
    parser.add_argument("--csv_path", required=True)
    evaluate(parser.parse_args())
