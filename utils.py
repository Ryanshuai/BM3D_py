import numpy as np


def symetrize(img, N):
    img_pad = np.pad(img, ((N, N), (N, N)), 'symmetric')
    return img_pad

