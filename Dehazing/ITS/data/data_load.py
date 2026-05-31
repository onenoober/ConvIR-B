import os
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


def train_dataloader(path, batch_size=64, num_workers=0, data='ITS', use_transform=True):
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
        DeblurDataset(image_dir, data, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, data, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, data, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, data, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test'), data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, data, transform=None, is_test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
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
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label
