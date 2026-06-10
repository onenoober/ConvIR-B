import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label, depth=None):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)
            if depth is not None:
                depth = F.pad(depth, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
            if depth is not None:
                depth = F.pad(depth, (self.size[1] - depth.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            if depth is not None:
                depth = F.pad(depth, (0, self.size[0] - depth.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        image = F.crop(image, i, j, h, w)
        label = F.crop(label, i, j, h, w)
        if depth is None:
            return image, label
        depth = F.crop(depth, i, j, h, w)
        return image, label, depth


class PairCompose(transforms.Compose):
    def __call__(self, image, label, depth=None):
        for t in self.transforms:
            if depth is None:
                image, label = t(image, label)
            else:
                image, label, depth = t(image, label, depth)
        if depth is None:
            return image, label
        return image, label, depth


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label, depth=None):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img = F.hflip(img)
            label = F.hflip(label)
            if depth is not None:
                depth = F.hflip(depth)
        if depth is None:
            return img, label
        return img, label, depth


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label, depth=None):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        pic = F.to_tensor(pic)
        label = F.to_tensor(label)
        if depth is None:
            return pic, label
        depth = F.to_tensor(depth).float()
        return pic, label, depth
