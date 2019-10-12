import math
import pywt
import numpy as np


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


def bior_2d_reverse(bior_img):
    pass # TODO