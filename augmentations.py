"""
This module contains multiple functions that do the actual augmentation

All functions shall take the observations as NUMPY array in the form (Batch x color x height x width) and return the
augmented images in the same format. However it can be that height and width are not the same.
"""

import numpy as np
import torch
import torchvision.transforms as T
from torch import nn
import kornia
import torchvision.transforms.functional as fn
from torchvision.transforms.functional import InterpolationMode


def rand_conv(images: torch.Tensor):
    """
    Taken from https://github.com/rraileanu/auto-drac/blob/master/data_augs.py
    """
    _device = images.device
    n_batch, n_channels, img_h, img_w = images.shape
    aug_images = torch.empty_like(images)

    # initialize random convolution
    conv_layer = nn.Conv2d(n_channels, n_channels, kernel_size=3, bias=False, padding=1).to(_device)
    for param in conv_layer.parameters():
        param.requires_grad_(False)

    for i in range(n_batch):
        torch.nn.init.xavier_normal_(conv_layer.weight.data)
        aug_images[i] = conv_layer(images[i])

    return torch.clip(aug_images, images.min(), images.max())



def random_translate(images: torch.tensor, size=65, h1s=None, w1s=None) -> torch.Tensor:
    """
    # Taken from https://github.com/MishaLaskin/rad
    # Augmentation that zooms out of the image and shifts the environment randomly around
    :param images: images in the form [B, C, H, W]
    :param size: size of the returned images
    :param h1s:
    :param w1s:
    """
    # Taken from https://github.com/MishaLaskin/rad
    n, c, h, w = images.shape
    assert size >= h and size >= w
    outs = torch.zeros((n, c, size, size), dtype=images.dtype, device=images.device)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, images, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img

    # Resize the image
    return fn.resize(outs, size=[h, w], interpolation=InterpolationMode.BILINEAR, antialias=False)


def random_crop(images: torch.tensor, crop=50) -> torch.tensor:
    """
    # Taken from https://github.com/MishaLaskin/rad
    Randomly crops out an image (of size crop x crop) of the original image and scales it back up to the
    original shape
    :param images: images in the form [B, C, H, W]
    :param crop: size of the window cropped out of the image
    """
    n, c, h, w = images.shape
    crop_max = h - crop + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = torch.empty((n, c, crop, crop), dtype=images.dtype, device=images.device)
    for i, (img, w11, h11) in enumerate(zip(images, w1, h1)):
        cropped[i] = img[:, h11:h11 + crop, w11:w11 + crop]

    # scale the image back up to its original shape
    cropped = fn.resize(cropped, size=[h, w], interpolation=InterpolationMode.NEAREST)
    return cropped


def random_crop2(images: torch.tensor, padding=12) -> torch.tensor:
    """
    Augmentation that adds a padding of x pixels and then randomly crops the image back to original form
    Taken from https://github.com/rraileanu/auto-drac/blob/master/data_augs.py
    """
    trans = nn.Sequential(
        nn.ReplicationPad2d(padding),
        kornia.augmentation.RandomCrop((images.shape[-2], images.shape[-1]))
    )
    return trans(images)


def random_cutout(images: torch.tensor, min_cut=10, max_cut=25) -> torch.tensor:
    """
    # Taken from https://github.com/MishaLaskin/rad
    :param images: images in the form [B, C, H, W]
    :param min_cut: smallest cutout possible
    :param max_cut: largest cutout possible
    """
    n, c, h, w = images.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)

    cutouts = torch.empty((n, c, h, w), dtype=images.dtype, device=images.device)
    for i, (img, w11, h11) in enumerate(zip(images, w1, h1)):
        cut_img = img.clone()
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
        cutouts[i] = cut_img
    return cutouts


def gaussian_blur(images: torch.tensor, kernel_size: int = 3, sigma: float = .5) -> torch.tensor:
    transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    blur_img = transform(images)

    # make sure we dont have illegal pixel values
    return torch.clip(blur_img, torch.min(images), torch.max(images))


def random_noise(images: torch.tensor, strength=0.05) -> torch.tensor:
    noise = torch.normal(0, strength, size=images.shape, device=images.device)
    # make sure we dont have illegal pixel values (i.e.: 255.3 or 1.1)
    return torch.clip(images + noise, torch.min(images), torch.max(images))


def identity(images: any) -> any:
    return images


aug_map = {
    "identity": identity,
    "conv": rand_conv,
    "translate": random_translate,
    "crop1": random_crop,
    "crop2": random_crop2,
    "cutout": random_cutout,
    "blur": gaussian_blur,
    "noise": random_noise,
}
