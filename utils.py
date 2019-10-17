import numpy as np


def symetrize(img, N):
    img_pad = np.pad(img, ((N, N), (N, N)), 'symmetric')
    return img_pad


def add_gaussian_noise(im, sigma):
    im_h, im_w = im.shape
    im = im + (sigma * np.random.randn(im_h, im_w)).astype(np.int)
    im = np.clip(im, 0, 255, out=None)
    im = im.astype(np.uint8)
    return im
