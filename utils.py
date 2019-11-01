import numpy as np


def symetrize(img, N):
    img_pad = np.pad(img, ((N, N), (N, N)), 'symmetric')
    return img_pad


def add_gaussian_noise(im, sigma, seed=None):
    if seed is not None:
        np.random.seed(seed)
    im = im + (sigma * np.random.randn(*im.shape)).astype(np.int)
    im = np.clip(im, 0., 255., out=None)
    im = im.astype(np.uint8)
    return im

def ind_initialize(max_size, N, step):
    ind = range(N, max_size - N, step)
    if ind[-1] < max_size - N - 1:
        ind = np.append(ind, np.array([max_size - N - 1]), axis=0)
    return ind


def get_kaiserWindow(kHW):
    k = np.kaiser(kHW, 2)
    k_2d = k[:, np.newaxis] @ k[np.newaxis, :]
    return k_2d


def get_coef(kHW):
    coef_norm = np.zeros(kHW * kHW)
    coef_norm_inv = np.zeros(kHW * kHW)
    coef = 0.5 / ((float)(kHW))
    for i in range(kHW):
        for j in range(kHW):
            if i == 0 and j == 0:
                coef_norm[i * kHW + j] = 0.5 * coef
                coef_norm_inv[i * kHW + j] = 2.0
            elif i * j == 0:
                coef_norm[i * kHW + j] = 0.7071067811865475 * coef
                coef_norm_inv[i * kHW + j] = 1.414213562373095
            else:
                coef_norm[i * kHW + j] = 1.0 * coef
                coef_norm_inv[i * kHW + j] = 1.0

    return coef_norm, coef_norm_inv


def sd_weighting(group_3D):
    N = group_3D.size

    mean = np.sum(group_3D)
    std = np.sum(group_3D * group_3D)

    res = (std - mean * mean / N) / (N - 1)
    weight = 1.0 / np.sqrt(res) if res > 0. else 0.
    return weight


if __name__ == '__main__':
    kaiser = get_kaiserWindow(12)
    print()
