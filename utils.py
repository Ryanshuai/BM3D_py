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


def ind_initialize(max_size, N, step):
    ind = range(N, max_size - N, step)
    if ind[-1] < max_size - N - 1:
        ind = np.append(ind, np.array([max_size - N - 1]), axis=0)
    return ind


def get_kaiserWindow(kHW):
    kaiserWindow = np.zeros(kHW * kHW)
    if kHW == 8:
        kaiserWindow[0 + kHW * 0] = 0.1924
        kaiserWindow[0 + kHW * 1] = 0.2989
        kaiserWindow[0 + kHW * 2] = 0.3846
        kaiserWindow[0 + kHW * 3] = 0.4325
        kaiserWindow[1 + kHW * 0] = 0.2989
        kaiserWindow[1 + kHW * 1] = 0.4642
        kaiserWindow[1 + kHW * 2] = 0.5974
        kaiserWindow[1 + kHW * 3] = 0.6717
        kaiserWindow[2 + kHW * 0] = 0.3846
        kaiserWindow[2 + kHW * 1] = 0.5974
        kaiserWindow[2 + kHW * 2] = 0.7688
        kaiserWindow[2 + kHW * 3] = 0.8644
        kaiserWindow[3 + kHW * 0] = 0.4325
        kaiserWindow[3 + kHW * 1] = 0.6717
        kaiserWindow[3 + kHW * 2] = 0.8644
        kaiserWindow[3 + kHW * 3] = 0.9718

        for i in range(kHW // 2):
            for j in range(kHW // 2, kHW):
                kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)]

        for i in range(kHW // 2, kHW):
            for j in range(kHW // 2):
                kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j]

        for i in range(kHW // 2, kHW):
            for j in range(kHW // 2, kHW):
                kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * (kHW - j - 1)]

    elif kHW == 12:
        kaiserWindow[0 + kHW * 0] = 0.1924
        kaiserWindow[0 + kHW * 1] = 0.2615
        kaiserWindow[0 + kHW * 2] = 0.3251
        kaiserWindow[0 + kHW * 3] = 0.3782
        kaiserWindow[0 + kHW * 4] = 0.4163
        kaiserWindow[0 + kHW * 5] = 0.4362
        kaiserWindow[1 + kHW * 0] = 0.2615
        kaiserWindow[1 + kHW * 1] = 0.3554
        kaiserWindow[1 + kHW * 2] = 0.4419
        kaiserWindow[1 + kHW * 3] = 0.5139
        kaiserWindow[1 + kHW * 4] = 0.5657
        kaiserWindow[1 + kHW * 5] = 0.5927
        kaiserWindow[2 + kHW * 0] = 0.3251
        kaiserWindow[2 + kHW * 1] = 0.4419
        kaiserWindow[2 + kHW * 2] = 0.5494
        kaiserWindow[2 + kHW * 3] = 0.6390
        kaiserWindow[2 + kHW * 4] = 0.7033
        kaiserWindow[2 + kHW * 5] = 0.7369
        kaiserWindow[3 + kHW * 0] = 0.3782
        kaiserWindow[3 + kHW * 1] = 0.5139
        kaiserWindow[3 + kHW * 2] = 0.6390
        kaiserWindow[3 + kHW * 3] = 0.7433
        kaiserWindow[3 + kHW * 4] = 0.8181
        kaiserWindow[3 + kHW * 5] = 0.8572
        kaiserWindow[4 + kHW * 0] = 0.4163
        kaiserWindow[4 + kHW * 1] = 0.5657
        kaiserWindow[4 + kHW * 2] = 0.7033
        kaiserWindow[4 + kHW * 3] = 0.8181
        kaiserWindow[4 + kHW * 4] = 0.9005
        kaiserWindow[4 + kHW * 5] = 0.9435
        kaiserWindow[5 + kHW * 0] = 0.4362
        kaiserWindow[5 + kHW * 1] = 0.5927
        kaiserWindow[5 + kHW * 2] = 0.7369
        kaiserWindow[5 + kHW * 3] = 0.8572
        kaiserWindow[5 + kHW * 4] = 0.9435
        kaiserWindow[5 + kHW * 5] = 0.9885

        for i in range(kHW // 2):
            for j in range(kHW // 2, kHW):
                kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)]

        for i in range(kHW // 2, kHW):
            for j in range(kHW // 2):
                kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j]

        for i in range(kHW // 2, kHW):
            for j in range(kHW // 2, kHW):
                kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * (kHW - j - 1)]

    else:
        for k in range(kHW * kHW):
            kaiserWindow[k] = 1.0

    kaiserWindow = kaiserWindow.reshape((kHW, kHW))
    # kaiserWindow = np.ones((kHW, kHW))
    return kaiserWindow


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