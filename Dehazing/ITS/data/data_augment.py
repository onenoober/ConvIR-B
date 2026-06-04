import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)


class TripleRandomCrop(transforms.RandomCrop):
    def __call__(self, image, label, prior):
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)
            prior = F.pad(prior, self.padding, self.fill, self.padding_mode)

        if self.pad_if_needed and image.size[0] < self.size[1]:
            pad = (self.size[1] - image.size[0], 0)
            image = F.pad(image, pad, self.fill, self.padding_mode)
            label = F.pad(label, pad, self.fill, self.padding_mode)
            prior = F.pad(prior, pad, self.fill, self.padding_mode)
        if self.pad_if_needed and image.size[1] < self.size[0]:
            pad = (0, self.size[0] - image.size[1])
            image = F.pad(image, pad, self.fill, self.padding_mode)
            label = F.pad(label, pad, self.fill, self.padding_mode)
            prior = F.pad(prior, pad, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)
        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w), F.crop(prior, i, j, h, w)


class TripleCompose(transforms.Compose):
    def __call__(self, image, label, prior):
        for transform in self.transforms:
            image, label, prior = transform(image, label, prior)
        return image, label, prior


class TripleRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, image, label, prior):
        if random.random() < self.p:
            return F.hflip(image), F.hflip(label), F.hflip(prior)
        return image, label, prior


class TripleToTensor(transforms.ToTensor):
    def __call__(self, image, label, prior):
        if not isinstance(prior, torch.Tensor):
            prior = F.to_tensor(prior)
        return F.to_tensor(image), F.to_tensor(label), prior.float()
