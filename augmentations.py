"""
This module contains multiple functions that do the actual augmentation

All functions shall take the observations as NUMPY array in the form (Batch x color x height x width) and return the
augmented images in the same format. However it can be that height and width are not the same.
"""

import numpy as np
import torch
import torchvision.transforms as T


def random_translate(images: np.ndarray, size, return_random_idxs=False, h1s=None, w1s=None):
    """
    # Taken from https://github.com/MishaLaskin/rad
    :param images: images in the form [B, C, H, W]
    :param size: size of the returned images
    :param return_random_idxs: if true, this function will return the translation coordinates for each image
    :param h1s:
    :param w1s:
    """
    # Taken from https://github.com/MishaLaskin/rad
    n, c, h, w = images.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=images.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, images, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_idxs:  # So can do the same to another set of images.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs


def random_flip(images: np.ndarray, p=.5):
    """
    # Taken from https://github.com/MishaLaskin/rad
    :param images: images in the form [B, C, H, W]
    :param p: probability of flipping
    """
    images = torch.from_numpy(images)
    bs, channels, h, w = images.shape

    images = images

    flipped_images = images.flip([3])

    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1]  # // 3
    images = images.view(*flipped_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)

    mask = mask.type(images.dtype)
    mask = mask[:, :, None, None]

    out = mask * flipped_images + (1 - mask) * images

    out = out.view([bs, -1, h, w])
    return out.numpy()


def random_crop(images: np.ndarray, out):
    """
    # Taken from https://github.com/MishaLaskin/rad
    :param images: images in the form [B, C, H, W]
    :param out: output shape of the image
    """
    n, c, h, w = images.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=images.dtype)
    for i, (img, w11, h11) in enumerate(zip(images, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped


def random_cutout(images: np.ndarray, min_cut, max_cut):
    """
    # Taken from https://github.com/MishaLaskin/rad
    :param images: images in the form [B, C, H, W]
    :param min_cut: smallest cutout possible
    :param max_cut: largest cutout possible
    """
    n, c, h, w = images.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)

    cutouts = np.empty((n, c, h, w), dtype=images.dtype)
    for i, (img, w11, h11) in enumerate(zip(images, w1, h1)):
        cut_img = img.copy()
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
        # print(img[:, h11:h11 + h11, w11:w11 + w11].shape)
        cutouts[i] = cut_img
    return cutouts


def random_rotation(images: np.ndarray, device, p=.5):
    # images: [B, C, H, W]
    images = torch.from_numpy(images)
    bs, channels, h, w = images.shape

    images = images.to(device)

    rot90_images = images.rot90(1, [2, 3])
    rot180_images = images.rot90(2, [2, 3])
    rot270_images = images.rot90(3, [2, 3])

    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    rnd_rot = np.random.randint(1, 4, size=(images.shape[0],))
    mask = rnd <= p
    mask = rnd_rot * mask
    mask = torch.from_numpy(mask).to(device)

    frames = images.shape[1]
    masks = [torch.zeros_like(mask) for _ in range(4)]
    for i, m in enumerate(masks):
        m[torch.where(mask == i)] = 1
        m = m[:, None] * torch.ones([1, frames]).type(mask.dtype).type(images.dtype).to(device)
        m = m[:, :, None, None]
        masks[i] = m

    out = masks[0] * images + masks[1] * rot90_images + masks[2] * rot180_images + masks[3] * rot270_images

    out = out.view([bs, -1, h, w])
    return out.numpy()


def gaussian_blur(images: np.ndarray, kernel_size: int = 3, sigma: float = .5):
    img_tensor = torch.from_numpy(images)
    transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    blur_img = transform(img_tensor).numpy()
    # make sure we dont have illegal pixel values
    return np.array(np.clip(blur_img, np.min(images), np.max(images)), dtype=images.dtype)


def random_noise(images: np.ndarray, strength=0.05):
    noise = np.random.normal(0, strength, size=images.shape)
    # make sure we dont have illegal pixel values (i.e.: 255.3 or 1.1)
    return np.array(np.clip(images + noise, np.min(images), np.max(images)), dtype=images.dtype)


def identity(images: np.ndarray):
    return images
