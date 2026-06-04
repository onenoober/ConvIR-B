import os
import numpy as np
import torch
import torch.nn.functional as torch_F
from PIL import Image as Image
from data import (
    PairCompose,
    PairRandomCrop,
    PairRandomHorizontalFilp,
    PairToTensor,
    TripleCompose,
    TripleRandomCrop,
    TripleRandomHorizontalFilp,
    TripleToTensor,
)
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

IMG_EXTENSIONS = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')


def _list_images(image_dir):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f'Image directory does not exist: {image_dir}')
    return sorted(
        name for name in os.listdir(image_dir)
        if name.lower().endswith(IMG_EXTENSIONS)
        and os.path.isfile(os.path.join(image_dir, name))
    )


def _first_existing_dir(root, names):
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f'None of the expected directories {names} exist under {root}'
    )


def _robust_normalize_np(values):
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    sample = values[finite]
    lo = float(np.min(sample))
    hi = float(np.max(sample))
    if hi - lo <= 1e-12:
        out = np.zeros_like(values, dtype=np.float32)
    else:
        out = np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[~finite] = 0.0
    return out


def depth_cache_path(cache_dir, split, image_name):
    return os.path.join(cache_dir, split, image_name.replace('/', '__') + '.npy')


def load_depth_prior(cache_dir, split, image_name, size):
    path = depth_cache_path(cache_dir, split, image_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing DPGA depth cache for {image_name}: {path}')
    depth = _robust_normalize_np(np.load(path).astype(np.float32))
    depth = torch.from_numpy(depth).view(1, depth.shape[0], depth.shape[1]).float()
    if tuple(depth.shape[-2:]) != tuple(size):
        depth = torch_F.interpolate(
            depth.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    return depth


def train_dataloader(
    path,
    batch_size=64,
    num_workers=0,
    data='ITS',
    use_transform=True,
    return_name=False,
    depth_cache_dir='',
    depth_split='train',
):
    image_dir = os.path.join(path, 'train')

    if data.lower() == 'real_haze':
        crop_size = [800,1184]
    else:
        crop_size = 256

    transform = None
    if use_transform:
        if depth_cache_dir:
            transform = TripleCompose(
                [
                    TripleRandomCrop(crop_size),
                    TripleRandomHorizontalFilp(),
                    TripleToTensor()
                ]
            )
        else:
            transform = PairCompose(
                [
                    PairRandomCrop(crop_size),
                    PairRandomHorizontalFilp(),
                    PairToTensor()
                ]
            )
    if depth_cache_dir:
        dataset = DepthPriorDeblurDataset(
            image_dir,
            data,
            depth_cache_dir,
            depth_split,
            transform=transform,
            return_name=return_name,
        )
    else:
        dataset = DeblurDataset(image_dir, data, transform=transform, return_name=return_name)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, data, batch_size=1, num_workers=0, depth_cache_dir='', depth_split='test'):
    image_dir = os.path.join(path, 'test')
    if depth_cache_dir:
        dataset = DepthPriorDeblurDataset(
            image_dir,
            data,
            depth_cache_dir,
            depth_split,
            is_test=True,
            return_name=True,
        )
    else:
        dataset = DeblurDataset(image_dir, data, is_test=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, data, batch_size=1, num_workers=0, depth_cache_dir='', depth_split='test'):
    image_dir = os.path.join(path, 'test')
    if depth_cache_dir:
        dataset = DepthPriorDeblurDataset(
            image_dir,
            data,
            depth_cache_dir,
            depth_split,
            return_name=True,
        )
    else:
        dataset = DeblurDataset(image_dir, data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, data, transform=None, is_test=False, return_name=False):
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.return_name = return_name
        self.data = data
        self.data_key = data.lower()

        if self.data_key == 'haze4k':
            self.input_dir = _first_existing_dir(image_dir, ('IN', 'haze', 'hazy'))
            self.label_dir = _first_existing_dir(image_dir, ('GT', 'gt'))
        elif self.data_key == 'real_haze':
            self.input_dir = _first_existing_dir(image_dir, ('hazy',))
            self.label_dir = _first_existing_dir(image_dir, ('GT', 'gt'))
        else:
            self.input_dir = _first_existing_dir(image_dir, ('hazy',))
            self.label_dir = _first_existing_dir(image_dir, ('gt',))

        self.image_list = _list_images(self.input_dir)

    def _label_path(self, image_name):
        candidates = []
        stem, ext = os.path.splitext(image_name)

        if self.data_key == 'its':
            candidates.append(f'{image_name.split("_")[0]}.png')
        elif self.data_key == 'haze4k':
            candidates.append(image_name)
            if '_' in stem:
                candidates.append(f'{stem.split("_")[0]}{ext}')
                candidates.append(f'{stem.split("_")[0]}.png')
        else:
            candidates.append(image_name)

        for candidate in candidates:
            path = os.path.join(self.label_dir, candidate)
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(
            f'No matching label for {image_name} in {self.label_dir}; '
            f'tried {candidates}'
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image = Image.open(os.path.join(self.input_dir, image_name)).convert('RGB')
        label = Image.open(self._label_path(image_name)).convert('RGB')

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test or self.return_name:
            name = self.image_list[idx]
            return image, label, name
        return image, label


class DepthPriorDeblurDataset(DeblurDataset):
    def __init__(
        self,
        image_dir,
        data,
        depth_cache_dir,
        depth_split,
        transform=None,
        is_test=False,
        return_name=False,
    ):
        super().__init__(image_dir, data, transform=transform, is_test=is_test, return_name=return_name)
        self.depth_cache_dir = depth_cache_dir
        self.depth_split = depth_split

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image = Image.open(os.path.join(self.input_dir, image_name)).convert('RGB')
        label = Image.open(self._label_path(image_name)).convert('RGB')
        depth = load_depth_prior(
            self.depth_cache_dir,
            self.depth_split,
            image_name,
            size=(image.size[1], image.size[0]),
        )

        if self.transform:
            image, label, depth = self.transform(image, label, depth)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
            depth = depth.float()

        if self.is_test or self.return_name:
            return image, label, depth, image_name
        return image, label, depth
