import numpy as np
import math


def bior_2d_forward(input, output, N, d_i, r_i, d_o, lpd, hpd):
    for i in range(N):
        for j in range(N):
            output[i * N + j + d_o] = input[i * r_i + j + d_i]

    iter_max = int(math.log2(N))
    N_1 = N
    N_2 = N // 2
    S_1 = len(lpd)
    S_2 = S_1 // 2 - 1
    tmp = np.zeros(N_1 + 2 * S_2)
    ind_per = np.zeros(N_1 + 2 * S_2)

    for iter in range(0, iter_max):
        per_ext_ind(ind_per, N_1, S_2)
        for i in range(N_1):
            for j in range(len(tmp)):
                tmp[j] = output[d_o + i * N + ind_per[j]]

            for j in range(N_2):
                v_l = 0.
                v_h = 0.
                for k in range(S_1):
                    v_l += tmp[k + j * 2] * lpd[k]
                    v_h += tmp[k + j * 2] * hpd[k]
                output[d_o + i * N + j] = v_l
                output[d_o + i * N + j + N_2] = v_h

        for j in range(N_1):
            for i in range(len(tmp)):
                tmp[i] = output[d_o + j + ind_per[i] * N]
            for i in range(N_2):
                v_l = 0.
                v_h = 0.
                for k in range(S_1):
                    v_l += tmp[k + i * 2] * lpd[k]
                    v_h += tmp[k + i * 2] * hpd[k]
                output[d_o + j + i * N] = v_l
                output[d_o + j + (i + N_2) * N] = v_h

        N_1 /= 2
        N_2 /= 2


def bior15_coef():
    lp1 = np.zeros(10)
    hp1 = np.zeros(10)
    lp2 = np.zeros(10)
    hp2 = np.zeros(10)

    coef_norm = 1. / (math.sqrt(2.) * 128.)
    sqrt2_inv = 1. / math.sqrt(2.)

    lp1[0] = 3.
    lp1[1] = -3.
    lp1[2] = -22.
    lp1[3] = 22.
    lp1[4] = 128.
    lp1[5] = 128.
    lp1[6] = 22.
    lp1[7] = -22.
    lp1[8] = -3.
    lp1[9] = 3.

    hp1[4] = -sqrt2_inv
    hp1[5] = sqrt2_inv

    lp2[4] = sqrt2_inv
    lp2[5] = sqrt2_inv

    hp2[0] = 3.
    hp2[1] = 3.
    hp2[2] = -22.
    hp2[3] = -22.
    hp2[4] = 128.
    hp2[5] = -128.
    hp2[6] = 22.
    hp2[7] = 22.
    hp2[8] = -3.
    hp2[9] = -3.

    for k in range(10):
        lp1[k] *= coef_norm
        hp2[k] *= coef_norm
    return lp1, hp1, lp2, hp2


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
