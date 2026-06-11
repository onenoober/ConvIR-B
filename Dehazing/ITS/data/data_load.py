import json
import os
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
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


def train_dataloader(
    path,
    batch_size=64,
    num_workers=0,
    data='ITS',
    use_transform=True,
    depth_cache_dir='',
    depth_split='train',
    return_trans=False,
    return_meta=False,
    split_json='',
    split_name='',
):
    image_dir = os.path.join(path, 'train')

    if data.lower() == 'real_haze':
        crop_size = [800,1184]
    else:
        crop_size = 256

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(crop_size),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(
            image_dir,
            data,
            transform=transform,
            depth_cache_dir=depth_cache_dir,
            depth_split=depth_split,
            return_trans=return_trans,
            return_meta=return_meta,
            split_json=split_json,
            split_name=split_name,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(
    path,
    data,
    batch_size=1,
    num_workers=0,
    depth_cache_dir='',
    depth_split='test',
    root_split='test',
    return_trans=False,
    return_meta=False,
    split_json='',
    split_name='',
):
    image_dir = os.path.join(path, root_split)
    dataloader = DataLoader(
        DeblurDataset(
            image_dir,
            data,
            is_test=True,
            depth_cache_dir=depth_cache_dir,
            depth_split=depth_split,
            return_trans=return_trans,
            return_meta=return_meta,
            split_json=split_json,
            split_name=split_name,
        ),
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
    root_split='test',
    return_trans=False,
    return_meta=False,
    split_json='',
    split_name='',
):
    dataloader = DataLoader(
        DeblurDataset(
            os.path.join(path, root_split),
            data,
            depth_cache_dir=depth_cache_dir,
            depth_split=depth_split,
            return_trans=return_trans,
            return_meta=return_meta,
            split_json=split_json,
            split_name=split_name,
        ),
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
        depth_cache_dir='',
        depth_split='',
        return_trans=False,
        return_meta=False,
        split_json='',
        split_name='',
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.data = data
        self.data_key = data.lower()
        self.depth_cache_dir = depth_cache_dir
        self.depth_split = depth_split
        self.return_trans = return_trans
        self.return_meta = return_meta

        if self.data_key == 'haze4k':
            self.input_dir = _first_existing_dir(image_dir, ('IN', 'haze', 'hazy'))
            self.label_dir = _first_existing_dir(image_dir, ('GT', 'gt'))
            self.trans_dir = _first_existing_dir(image_dir, ('trans', 'Trans', 'transmission')) if return_trans else ''
        elif self.data_key == 'real_haze':
            self.input_dir = _first_existing_dir(image_dir, ('hazy',))
            self.label_dir = _first_existing_dir(image_dir, ('GT', 'gt'))
            self.trans_dir = ''
        else:
            self.input_dir = _first_existing_dir(image_dir, ('hazy',))
            self.label_dir = _first_existing_dir(image_dir, ('gt',))
            self.trans_dir = ''

        self.image_list = _list_images(self.input_dir)
        if split_json or split_name:
            self.image_list = self._filter_split(self.image_list, split_json, split_name)

    def _filter_split(self, image_list, split_json, split_name):
        if not split_json or not split_name:
            return image_list
        with open(split_json, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        splits = payload.get('splits', payload)
        if split_name not in splits:
            raise KeyError(f'Split {split_name} not found in {split_json}')
        wanted = splits[split_name]
        if wanted and isinstance(wanted[0], dict):
            wanted = [row.get('name') or row.get('image') for row in wanted]
        wanted = {os.path.basename(name) for name in wanted if name}
        filtered = [name for name in image_list if name in wanted]
        if not filtered:
            raise ValueError(f'Split {split_name} from {split_json} matched no images')
        return filtered

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

    def _trans_path(self, image_name):
        if not self.trans_dir:
            return ''
        candidates = []
        stem, ext = os.path.splitext(image_name)
        candidates.append(image_name)
        if '_' in stem:
            base = stem.split('_')[0]
            candidates.append(f'{base}{ext}')
            candidates.append(f'{base}.png')

        for candidate in candidates:
            path = os.path.join(self.trans_dir, candidate)
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(
            f'No matching transmission map for {image_name} in {self.trans_dir}; '
            f'tried {candidates}'
        )

    def _depth_path(self, image_name):
        if not self.depth_cache_dir:
            return ''
        split = self.depth_split or ('test' if self.is_test else 'train')
        candidates = [
            os.path.join(self.depth_cache_dir, split, image_name.replace('/', '__') + '.npy'),
            os.path.join(self.depth_cache_dir, split, image_name + '.npy'),
            os.path.join(self.depth_cache_dir, image_name.replace('/', '__') + '.npy'),
            os.path.join(self.depth_cache_dir, image_name + '.npy'),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f'Missing depth cache for {image_name}; tried {candidates}'
        )

    def _load_depth(self, image_name, size):
        depth_path = self._depth_path(image_name)
        if not depth_path:
            return None
        depth = np.load(depth_path).astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if depth.ndim == 3:
            depth = depth.squeeze()
        depth_img = Image.fromarray(depth, mode='F')
        if depth_img.size != size:
            depth_img = depth_img.resize(size, resample=Image.BICUBIC)
        return depth_img

    def _load_trans(self, image_name, size):
        trans_path = self._trans_path(image_name)
        if not trans_path:
            return None
        trans = Image.open(trans_path).convert('L')
        if trans.size != size:
            trans = trans.resize(size, resample=Image.BICUBIC)
        return trans

    @staticmethod
    def _airlight_from_name(image_name):
        stem = os.path.splitext(os.path.basename(image_name))[0]
        parts = stem.split('_')
        if len(parts) >= 2:
            try:
                return np.float32(float(parts[1]))
            except ValueError:
                pass
        return np.float32(1.0)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image = Image.open(os.path.join(self.input_dir, image_name)).convert('RGB')
        label = Image.open(self._label_path(image_name)).convert('RGB')
        depth = self._load_depth(image_name, image.size)
        trans = self._load_trans(image_name, image.size) if self.return_trans else None

        if self.transform:
            if depth is None and trans is None:
                image, label = self.transform(image, label)
            elif trans is None:
                image, label, depth = self.transform(image, label, depth)
            else:
                image, label, depth, trans = self.transform(image, label, depth, trans)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
            if depth is not None:
                depth = F.to_tensor(depth).float()
            if trans is not None:
                trans = F.to_tensor(trans).float().clamp(0.0, 1.0)
        payload = [image, label]
        if depth is not None:
            payload.append(depth)
        if trans is not None:
            payload.append(trans)
        if self.return_meta:
            payload.append(self._airlight_from_name(image_name))
        if self.is_test:
            name = self.image_list[idx]
            payload.append(name)
        return tuple(payload)
