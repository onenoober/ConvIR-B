import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


def _pack_outputs(image, label, depth=None, trans=None):
    if depth is None and trans is None:
        return image, label
    if trans is None:
        return image, label, depth
    return image, label, depth, trans


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label, depth=None, trans=None):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)
            if depth is not None:
                depth = F.pad(depth, self.padding, self.fill, self.padding_mode)
            if trans is not None:
                trans = F.pad(trans, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
            if depth is not None:
                depth = F.pad(depth, (self.size[1] - depth.size[0], 0), self.fill, self.padding_mode)
            if trans is not None:
                trans = F.pad(trans, (self.size[1] - trans.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            if depth is not None:
                depth = F.pad(depth, (0, self.size[0] - depth.size[1]), self.fill, self.padding_mode)
            if trans is not None:
                trans = F.pad(trans, (0, self.size[0] - trans.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        image = F.crop(image, i, j, h, w)
        label = F.crop(label, i, j, h, w)
        if depth is not None:
            depth = F.crop(depth, i, j, h, w)
        if trans is not None:
            trans = F.crop(trans, i, j, h, w)
        return _pack_outputs(image, label, depth, trans)


class PairCompose(transforms.Compose):
    def __call__(self, image, label, depth=None, trans=None):
        for t in self.transforms:
            if depth is None and trans is None:
                image, label = t(image, label)
            elif trans is None:
                image, label, depth = t(image, label, depth)
            else:
                image, label, depth, trans = t(image, label, depth, trans)
        return _pack_outputs(image, label, depth, trans)


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label, depth=None, trans=None):
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
            if trans is not None:
                trans = F.hflip(trans)
        return _pack_outputs(img, label, depth, trans)


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label, depth=None, trans=None):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        pic = F.to_tensor(pic)
        label = F.to_tensor(label)
        if depth is not None:
            depth = F.to_tensor(depth).float()
        if trans is not None:
            trans = F.to_tensor(trans).float()
        return _pack_outputs(pic, label, depth, trans)
