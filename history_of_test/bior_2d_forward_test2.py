import math
import numpy as np


def isPowerOfTwo(n: int):
    return bool(n & (n - 1) == 0)


def bior_2d_forward(output, N, lpd, hpd):
    assert isPowerOfTwo(N)
    iter_max = int(math.log2(N))
    N_1 = N
    N_2 = N // 2
    S_1 = len(lpd)
    S_2 = S_1 // 2 - 1
    tmp = np.zeros(N_1 + 2 * S_2)
    ind_per = np.zeros(N_1 + 2 * S_2, dtype=np.int)
    per_ext_ind(ind_per, N_1, S_2)

    for iter in range(0, iter_max):
        for i in range(N_1):
            for j in range(len(tmp)):
                tmp[j] = output[i * N + ind_per[j]]

            for j in range(N_2):
                v_l = 0.
                v_h = 0.
                for k in range(S_1):
                    v_l += tmp[k + j * 2] * lpd[k]
                    v_h += tmp[k + j * 2] * hpd[k]
                output[i * N + j] = v_l
                output[i * N + j + N_2] = v_h

        for j in range(N_1):
            for i in range(len(tmp)):
                tmp[i] = output[j + ind_per[i] * N]
            for i in range(N_2):
                v_l = 0.
                v_h = 0.
                for k in range(S_1):
                    v_l += tmp[k + i * 2] * lpd[k]
                    v_h += tmp[k + i * 2] * hpd[k]
                output[j + i * N] = v_l
                output[j + (i + N_2) * N] = v_h

        N_1 //= 2
        N_2 //= 2


def bior15_coef():
    lpd = np.zeros(10)
    hpd = np.zeros(10)
    lpr = np.zeros(10)
    hpr = np.zeros(10)

    coef_norm = 1. / (math.sqrt(2.) * 128.)
    sqrt2_inv = 1. / math.sqrt(2.)

    lpd[0] = 3.
    lpd[1] = -3.
    lpd[2] = -22.
    lpd[3] = 22.
    lpd[4] = 128.
    lpd[5] = 128.
    lpd[6] = 22.
    lpd[7] = -22.
    lpd[8] = -3.
    lpd[9] = 3.

    hpd[4] = -sqrt2_inv
    hpd[5] = sqrt2_inv

    lpr[4] = sqrt2_inv
    lpr[5] = sqrt2_inv

    hpr[0] = 3.
    hpr[1] = 3.
    hpr[2] = -22.
    hpr[3] = -22.
    hpr[4] = 128.
    hpr[5] = -128.
    hpr[6] = 22.
    hpr[7] = 22.
    hpr[8] = -3.
    hpr[9] = -3.

    for k in range(10):
        lpd[k] *= coef_norm
        hpr[k] *= coef_norm
    return lpd, hpd, lpr, hpr


def per_ext_ind(ind_per, N, L):
    for k in range(N):
        ind_per[k + L] = k
    ind1 = N - L
    while ind1 < 0:
        ind1 += N
    ind2 = 0
    k = 0
    while k < L:
        ind_per[k] = ind1
        ind_per[k + L + N] = ind2
        ind1 = ind1 + 1 if ind1 < N - 1 else 0
        ind2 = ind2 + 1 if ind2 < N - 1 else 0
        k += 1


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
    bior_2d_forward(img_flat, N, lpd, hpd)
    img_ = img_flat.reshape(N, N)

    # my way
    coeffs2 = pywt.dwt2(img, 'bior1.5')
    LL, (LH, HL, HH) = coeffs2

    # test diff
    HH_ = HH[2: -2, 2: -2]
    HL_ = HL[2: -2, 2: -2]
    LH_ = LH[2: -2, 2: -2]





    all_HL = np.zeros((512, 512), dtype=np.float64)
    all_HL[256:, 256:] = HH[2: -2, 2: -2]
    all_HL[:256, 256:] = HL[2: -2, 2: -2]
    all_HL[256:, :256] = LH[2: -2, 2: -2]
    cv2.imshow('all_HL', all_HL)

    print(img_.shape)
    cv2.imshow('forwarc', img_)
    cv2.waitKey()


