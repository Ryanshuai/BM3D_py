import math
import numpy as np
from history_of_test.bior_2d_forward_test1 import original_bior_2d_forward, bior15_coef


def bior_2d_forward(img):
    assert img.shape[0] == img.shape[1]
    N = img.shape[0]
    iter_max = int(math.log2(N))
    bior_img = np.zeros_like(img)

    for iter in range(iter_max):
        N //= 2
        if N == 256:
            coeffs2 = pywt.dwt2(img, 'bior1.5', mode='periodization')
        else:
            coeffs2 = pywt.dwt2(img, 'bior1.5', mode='zero')
            coeffs2 = pywt.dwt2(img, 'bior1.5', mode='constant')
            coeffs2 = pywt.dwt2(img, 'bior1.5', mode='symmetric')
            coeffs2 = pywt.dwt2(img, 'bior1.5', mode='reflect')
            coeffs2 = pywt.dwt2(img, 'bior1.5', mode='periodic')
            coeffs2 = pywt.dwt2(img, 'bior1.5', mode='smooth')
            coeffs2 = pywt.dwt2(img, 'bior1.5', mode='antisymmetric')
            coeffs2 = pywt.dwt2(img, 'bior1.5', mode='antireflect')
            coeffs2 = pywt.dwt2(img, 'bior1.5', mode='periodization')

        LL, (LH, HL, HH) = coeffs2
        img = LL
        bior_img[N:2 * N, N:2 * N] = HH
        bior_img[:N, N:2 * N] = -HL
        bior_img[N:2 * N, :N] = -LH

    return bior_img


if __name__ == '__main__':
    import cv2
    import pywt
    import matplotlib.pyplot as plt

    # input
    img = pywt.data.camera()
    img = img.astype(np.float64)

    # original way
    img_flat = img.flatten()
    N = int(math.sqrt(len(img_flat)))
    lpd, hpd, lpr, hpr = bior15_coef()
    original_bior_2d_forward(img_flat, N, lpd, hpd)
    original_bior_img = img_flat.reshape(N, N)

    # my way
    bior_img = bior_2d_forward(img)


    diff = original_bior_img - bior_img
    print('sum diff of LH', np.sum(np.abs(diff)))
    print('max diff of HH', np.max(np.abs(diff)))

    cv2.imshow('original_bior_img', original_bior_img)
    cv2.imshow('bior_img', bior_img)
    cv2.imshow('diff', diff)
    cv2.waitKey()
