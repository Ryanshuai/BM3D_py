import math
import numpy as np
from test_history.bior_2d_forward_test1 import original_bior_2d_forward, bior15_coef


def bior_2d_forward(img):
    assert img.shape[0] == img.shape[1]
    N = img.shape[0]
    iter_max = int(math.log2(N))

    for iter in range(iter_max):
        coeffs2 = pywt.dwt2(img[:N, :N], 'bior1.5', mode='periodic')
        LL, (LH, HL, HH) = coeffs2
        img[:N//2, :N//2] = LL[2: -2, 2: -2]
        img[N//2:N, N//2:N] = HH[2: -2, 2: -2]
        img[:N//2, N//2:N] = -HL[2: -2, 2: -2]
        img[N//2:N, :N//2] = -LH[2: -2, 2: -2]
        N //= 2
    return img


if __name__ == '__main__':
    import cv2
    import pywt
    import matplotlib.pyplot as plt

    # input
    img = pywt.data.camera()
    img = img.astype(np.float64)

    # original way
    original_bior_img = original_bior_2d_forward(img)

    # my way
    bior_img = bior_2d_forward(img.copy())


    a, b = 0, 4
    c, d = 0, 4
    print('original_bior_img\n', original_bior_img[a:b, c:d].astype(np.int))
    print('bior_img\n', bior_img[a:b, c:d].astype(np.int))

    # print('max original_bior_img', np.max(original_bior_img))
    # print('min original_bior_img', np.min(original_bior_img))
    #
    # print('max bior_img', np.max(bior_img))
    # print('min bior_img', np.min(bior_img))
    diff = original_bior_img - bior_img
    print('sum of diff', np.sum(np.abs(diff)))
    print('max of diff', np.max(np.abs(diff)))
    cv2.imshow('original_bior_img', original_bior_img)
    cv2.imshow('bior_img', bior_img)
    cv2.imshow('diff', diff)
    cv2.waitKey()
