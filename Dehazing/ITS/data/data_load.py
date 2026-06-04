import os
import json
import math
import random
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
from torch.utils.data import Dataset, DataLoader, Sampler

IMG_EXTENSIONS = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')


def _list_images(image_dir):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f'Image directory does not exist: {image_dir}')
    return sorted(
        name for name in os.listdir(image_dir)
        if name.lower().endswith(IMG_EXTENSIONS)
        and os.path.isfile(os.path.join(image_dir, name))
    )


def _load_split_names(split_json, split_name):
    if not split_json or not split_name:
        return None
    with open(split_json, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if 'splits' in payload:
        splits = payload['splits']
    else:
        splits = payload
    if split_name not in splits:
        raise KeyError(f'Split {split_name!r} not found in {split_json}')
    names = splits[split_name]
    if not isinstance(names, list):
        raise TypeError(f'Split {split_name!r} must be a list of image names')
    return set(names)


def _load_json_payload(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _split_name_set(payload, split_name):
    if not split_name:
        return None
    splits = payload.get('splits', payload)
    names = splits.get(split_name)
    if names is None:
        raise KeyError(f'Split {split_name!r} not found in sampler metadata')
    return set(names)


def _normalize_bucket_label(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ('hard', 'bottom', 'bottom25', 'hard_bottom25', '0'):
            return 0
        if value in ('easy', 'top', 'top25', 'easy_top25', '2'):
            return 2
        return 1
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return 1
    return 0 if ivalue <= 0 else 2 if ivalue >= 2 else 1


def _quantile_cuts(values, hard_quantile=0.25, easy_quantile=0.75):
    if not values:
        return None, None
    ordered = sorted(values)

    def at_quantile(q):
        pos = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
        return ordered[pos]

    return at_quantile(hard_quantile), at_quantile(easy_quantile)


def _load_hard_bucket_labels(sampler_json, sampler_split_name, image_names):
    if not sampler_json:
        return None
    payload = _load_json_payload(sampler_json)
    allowed = _split_name_set(payload, sampler_split_name)
    rows = payload.get('rows', [])
    if not isinstance(rows, list):
        raise TypeError(f'Expected rows list in {sampler_json}')

    selected = []
    for row in rows:
        name = row.get('name') or row.get('image') or row.get('image_name')
        if not name or (allowed is not None and name not in allowed):
            continue
        selected.append(row)

    labels = {}
    for row in selected:
        name = row.get('name') or row.get('image') or row.get('image_name')
        for key in ('dpga_hard_bucket', 'a0_bucket', 'hard_bucket', 'bucket'):
            if key in row:
                labels[name] = _normalize_bucket_label(row[key])
                break

    if labels:
        missing = [name for name in image_names if name not in labels]
        if missing:
            raise KeyError(
                f'{len(missing)} selected images lack hard bucket metadata; first missing: {missing[:5]}'
            )
        return labels

    psnr_rows = []
    for row in selected:
        name = row.get('name') or row.get('image') or row.get('image_name')
        if 'a0_psnr' in row:
            psnr_rows.append((name, float(row['a0_psnr'])))
    hard_cut, easy_cut = _quantile_cuts([value for _name, value in psnr_rows])
    if hard_cut is None:
        raise ValueError(
            f'{sampler_json} must provide dpga_hard_bucket/a0_bucket or a0_psnr rows'
        )
    for name, value in psnr_rows:
        labels[name] = 0 if value <= hard_cut else 2 if value >= easy_cut else 1

    missing = [name for name in image_names if name not in labels]
    if missing:
        raise KeyError(
            f'{len(missing)} selected images lack hard bucket metadata; first missing: {missing[:5]}'
        )
    return labels


class HardAwareBatchSampler(Sampler):
    def __init__(
        self,
        bucket_labels,
        batch_size,
        seed=3407,
        hard_ratio=1.0 / 3.0,
        medium_ratio=1.0 / 3.0,
        batches_per_epoch=0,
    ):
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.bucket_labels = list(bucket_labels)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.hard_ratio = float(hard_ratio)
        self.medium_ratio = float(medium_ratio)
        self.batches_per_epoch = int(batches_per_epoch) if batches_per_epoch else math.ceil(
            len(self.bucket_labels) / self.batch_size
        )
        self._epoch = 0
        self._indices = {
            0: [idx for idx, label in enumerate(self.bucket_labels) if label == 0],
            1: [idx for idx, label in enumerate(self.bucket_labels) if label == 1],
            2: [idx for idx, label in enumerate(self.bucket_labels) if label == 2],
        }
        if not any(self._indices.values()):
            raise ValueError('hard-aware sampler received no indices')
        self._all = [idx for indices in self._indices.values() for idx in indices]

    def __len__(self):
        return self.batches_per_epoch

    def _sample_bucket(self, ng, bucket, count):
        source = self._indices.get(bucket) or self._all
        return [ng.choice(source) for _ in range(count)]

    def __iter__(self):
        ng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        hard_count = max(0, min(self.batch_size, round(self.batch_size * self.hard_ratio)))
        medium_count = max(
            0,
            min(self.batch_size - hard_count, round(self.batch_size * self.medium_ratio)),
        )
        easy_count = self.batch_size - hard_count - medium_count
        quotas = [(0, hard_count), (1, medium_count), (2, easy_count)]
        for _ in range(self.batches_per_epoch):
            batch = []
            for bucket, count in quotas:
                batch.extend(self._sample_bucket(ng, bucket, count))
            ng.shuffle(batch)
            yield batch


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
    split_json='',
    split_name='',
    hard_sampler_json='',
    hard_sampler_split_name='',
    hard_sampler_seed=3407,
    hard_sampler_hard_ratio=1.0 / 3.0,
    hard_sampler_medium_ratio=1.0 / 3.0,
    hard_sampler_batches_per_epoch=0,
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
        include_names = _load_split_names(split_json, split_name)
        bucket_labels = None
        if hard_sampler_json:
            selected_names = sorted(include_names) if include_names is not None else _list_images(image_dir)
            bucket_labels = _load_hard_bucket_labels(
                hard_sampler_json,
                hard_sampler_split_name or split_name,
                selected_names,
            )
        dataset = DepthPriorDeblurDataset(
            image_dir,
            data,
            depth_cache_dir,
            depth_split,
            transform=transform,
            return_name=return_name,
            include_names=include_names,
            hard_bucket_labels=bucket_labels,
            return_bucket_label=bool(bucket_labels) and not return_name,
        )
    else:
        dataset = DeblurDataset(
            image_dir,
            data,
            transform=transform,
            return_name=return_name,
            include_names=_load_split_names(split_json, split_name),
        )
    batch_sampler = None
    if hard_sampler_json:
        if not depth_cache_dir:
            raise ValueError('hard-aware sampler is only wired for DPGA depth-prior training')
        bucket_sequence = [
            dataset.hard_bucket_labels.get(name, 1)
            for name in dataset.image_list
        ]
        batch_sampler = HardAwareBatchSampler(
            bucket_sequence,
            batch_size=batch_size,
            seed=hard_sampler_seed,
            hard_ratio=hard_sampler_hard_ratio,
            medium_ratio=hard_sampler_medium_ratio,
            batches_per_epoch=hard_sampler_batches_per_epoch,
        )
    loader_kwargs = {
        'dataset': dataset,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if batch_sampler is not None:
        loader_kwargs['batch_sampler'] = batch_sample
    else:
        loader_kwargs['batch_size'] = batch_size
        loader_kwargs['shuffle'] = True
    dataloader = DataLoader(**loader_kwargs)
    return dataloader


def test_dataloader(
    path,
    data,
    batch_size=1,
    num_workers=0,
    depth_cache_dir='',
    depth_split='test',
    split_json='',
    split_name='',
):
    image_dir = os.path.join(path, 'test')
    if split_json and split_name:
        image_dir = os.path.join(path, 'train')
    if depth_cache_dir:
        dataset = DepthPriorDeblurDataset(
            image_dir,
            data,
            depth_cache_dir,
            depth_split,
            is_test=True,
            return_name=True,
            include_names=_load_split_names(split_json, split_name),
        )
    else:
        dataset = DeblurDataset(
            image_dir,
            data,
            is_test=True,
            include_names=_load_split_names(split_json, split_name),
        )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(
    path,
    data,
    batch_size=1,
    num_workers=0,
    depth_cache_dir='',
    depth_split='test',
    split_json='',
    split_name='',
):
    image_dir = os.path.join(path, 'test')
    if split_json and split_name:
        image_dir = os.path.join(path, 'train')
    if depth_cache_dir:
        dataset = DepthPriorDeblurDataset(
            image_dir,
            data,
            depth_cache_dir,
            depth_split,
            return_name=True,
            include_names=_load_split_names(split_json, split_name),
        )
    else:
        dataset = DeblurDataset(
            image_dir,
            data,
            include_names=_load_split_names(split_json, split_name),
        )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(
        self,
        image_dir,
        data,
        transform=None,
        is_test=False,
        return_name=False,
        include_names=None,
    ):
        self.image_dir = image_di
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
        if include_names is not None:
            missing = sorted(set(include_names).difference(self.image_list))
            if missing:
                raise FileNotFoundError(
                    f'{len(missing)} split images were not found in {self.input_dir}; '
                    f'first missing: {missing[:5]}'
                )
            self.image_list = [name for name in self.image_list if name in include_names]
        if not self.image_list:
            raise ValueError(f'No images selected from {self.input_dir}')

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
        include_names=None,
        hard_bucket_labels=None,
        return_bucket_label=False,
    ):
        super().__init__(
            image_dir,
            data,
            transform=transform,
            is_test=is_test,
            return_name=return_name,
            include_names=include_names,
        )
        self.depth_cache_dir = depth_cache_dir
        self.depth_split = depth_split
        self.hard_bucket_labels = hard_bucket_labels or {}
        self.return_bucket_label = return_bucket_label

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
        if self.return_bucket_label:
            bucket = self.hard_bucket_labels.get(image_name, 1)
            return image, label, depth, torch.tensor(bucket, dtype=torch.long)
        return image, label, depth
