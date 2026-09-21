"""Cloud-only, one-step Haze4K smoke for pinned upstream dehazing models."""

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d
from torchvision.transforms import functional as TF


AECR_COMMIT = "d046d0f5f849692ddb6eb6eb3d14f23c86ff5ca6"
FFA_COMMIT = "6651bcf693edd3ba2d097f728676942bd40e6e99"


class ModernFastDeconv(nn.Conv2d):
    """The upstream 3x3 FastDeconv with current torch.addmm calling convention."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 eps=1e-5, n_iter=5, momentum=0.1, sampling_stride=3):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.eps = eps
        self.n_iter = n_iter
        self.momentum = momentum
        self.sampling_stride = sampling_stride * stride
        self.num_features = kernel_size * kernel_size * in_channels
        self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer("running_deconv", torch.eye(self.num_features))

    def forward(self, x):
        if self.training:
            patches = F.unfold(x, self.kernel_size, self.dilation, self.padding,
                               self.sampling_stride).transpose(1, 2).reshape(-1, self.num_features)
            mean = patches.mean(0)
            centered = patches - mean
            eye = torch.eye(self.num_features, device=x.device, dtype=x.dtype)
            cov = torch.addmm(eye, centered.T, centered,
                              beta=self.eps, alpha=1.0 / centered.shape[0])
            norm = cov.norm()
            y, z = cov / norm, eye
            for _ in range(self.n_iter):
                t = 0.5 * (3.0 * eye - z @ y)
                y, z = y @ t, t @ z
            deconv = z / norm.sqrt()
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_deconv.lerp_(deconv.detach(), self.momentum)
        else:
            mean, deconv = self.running_mean, self.running_deconv
        weight = self.weight.flatten(1) @ deconv
        bias = self.bias - weight @ mean
        return F.conv2d(x, weight.view_as(self.weight), bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ModernDCN(nn.Module):
    """DCNv2's modulated convolution using torchvision's maintained CUDA op."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super().__init__()
        k = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        channels = deformable_groups * 3 * k * k
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, k, k))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.conv_offset_mask = nn.Conv2d(in_channels, channels, k, stride, padding)
        nn.init.uniform_(self.weight, -1.0 / math.sqrt(in_channels * k * k),
                         1.0 / math.sqrt(in_channels * k * k))
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)
        self.stride, self.padding, self.dilation = stride, padding, dilation

    def forward(self, x):
        a, b, mask = self.conv_offset_mask(x).chunk(3, dim=1)
        return deform_conv2d(x, torch.cat((a, b), dim=1), self.weight, self.bias,
                             stride=self.stride, padding=self.padding,
                             dilation=self.dilation, mask=mask.sigmoid())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_model(kind, source):
    if kind == "ffa":
        module = load_module("ffa_official_model", source / "net/models/FFA.py")
        return module.FFA(gps=3, blocks=19), FFA_COMMIT

    deconv = types.ModuleType("deconv")
    deconv.FastDeconv = ModernFastDeconv
    sys.modules["deconv"] = deconv
    package = types.ModuleType("DCNv2")
    package.__path__ = []
    sys.modules["DCNv2"] = package
    dcn = types.ModuleType("DCNv2.dcn_v2")
    dcn.DCN = ModernDCN
    sys.modules["DCNv2.dcn_v2"] = dcn
    module = load_module("aecr_official_model", source / "models/AECRNet.py")
    return module.Dehaze(3, 3), AECR_COMMIT


def pair_from_train(root):
    train = root / "train"
    inputs = next((train / name for name in ("IN", "haze", "hazy")
                   if (train / name).is_dir()), None)
    labels = next((train / name for name in ("GT", "gt")
                   if (train / name).is_dir()), None)
    if inputs is None or labels is None:
        raise FileNotFoundError(f"Haze4K train IN/GT not found under {train}")
    names = sorted(p.name for p in inputs.iterdir() if p.suffix.lower() in
                   (".png", ".jpg", ".jpeg", ".bmp"))
    paired = []
    for name in names:
        stem = Path(name).stem.split("_")[0]
        candidates = (name, stem + Path(name).suffix, stem + ".png")
        gt = next((labels / item for item in candidates if (labels / item).is_file()), None)
        if gt is not None:
            paired.append((inputs / name, gt))
    if not paired:
        raise ValueError("No Haze4K train pairs were found")
    return paired, len(names)


def image_pair(paths, kind, crop_size):
    with Image.open(paths[0]) as image, Image.open(paths[1]) as label:
        image, label = image.convert("RGB"), label.convert("RGB")
        if image.size != label.size or min(image.size) < crop_size:
            raise ValueError(f"Unaligned or undersized pair: {paths}")
        x = TF.to_tensor(TF.crop(image, 0, 0, crop_size, crop_size))[None]
        y = TF.to_tensor(TF.crop(label, 0, 0, crop_size, crop_size))[None]
    if kind == "ffa":
        x = TF.normalize(x, mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("aecr", "ffa"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--crop-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This repository requires cloud CUDA for model smoke tests")
    torch.manual_seed(3407)
    model, commit = make_model(args.model, args.source)
    actual = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"],
                                     text=True).strip()
    if actual != commit:
        raise RuntimeError(f"Upstream commit mismatch: {actual} != {commit}")
    pairs, hazy_count = pair_from_train(args.data)
    x, y = image_pair(pairs[0], args.model, args.crop_size)
    x, y = x.cuda(), y.cuda()
    model = model.cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    pred = model(x)
    if pred.shape != y.shape or not torch.isfinite(pred).all():
        raise RuntimeError("Invalid model output shape or non-finite pixels")
    loss = F.l1_loss(pred, y)
    contrast = None
    if args.model == "aecr":
        cr = load_module("aecr_official_cr", args.source / "models/CR.py")
        criterion = cr.ContrastLoss().cuda().eval()
        contrast = criterion(pred, y, x)
        loss = loss + 0.1 * contrast
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss")
    loss.backward()
    grad_count = sum(p.grad is not None and torch.isfinite(p.grad).all().item()
                     and p.grad.abs().sum().item() > 0 for p in model.parameters())
    if grad_count == 0:
        raise RuntimeError("No finite nonzero parameter gradients")
    optimizer.step()
    checkpoint = args.output / f"{args.model}_smoke.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite {checkpoint}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "source_commit": commit, "engineering_smoke_only": True}, checkpoint)
    restored, _ = make_model(args.model, args.source)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"],
                             strict=True)
    restored = restored.cuda().eval()
    with torch.no_grad():
        output = restored(x)
    if output.shape != y.shape or not torch.isfinite(output).all():
        raise RuntimeError("Restored checkpoint did not produce finite output")
    report = {
        "state": "ENGINEERING_SMOKE_PASS", "model": args.model,
        "upstream_commit": commit, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "source": str(args.source),
        "data": str(args.data), "train_hazy_count": hazy_count,
        "paired_train_count": len(pairs), "sample": pairs[0][0].name,
        "crop_size": args.crop_size, "output_shape": list(output.shape),
        "loss": float(loss.detach()), "contrast_loss": float(contrast.detach()) if contrast is not None else None,
        "nonzero_gradient_tensors": grad_count, "checkpoint": str(checkpoint),
        "locked_test_touched": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"{args.model.upper()}_SMOKE_OK")


if __name__ == "__main__":
    main()
