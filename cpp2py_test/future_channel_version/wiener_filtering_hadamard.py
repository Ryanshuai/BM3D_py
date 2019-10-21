import numpy as np
from scipy.linalg import hadamard
import math


def wiener_filtering_hadamard(group_3D_img, group_3D_est, nSx_r, kWien, chnls, sigma_table, weight_table, doWeight):
    kWien_2 = kWien * kWien
    coef = 1.0 / nSx_r
    for c in range(chnls):
        weight_table[c] = 0.
    for n in range(kWien_2 * chnls):
        hadamard_transform(group_3D_img, nSx_r, n * nSx_r)
        hadamard_transform(group_3D_est, nSx_r, n * nSx_r)

    for c in range(chnls):
        dc = nSx_r * kWien_2  # diff from original definition
        group_3D_img_c = group_3D_img[c * dc: (c + 1) * dc]
        group_3D_est_c = group_3D_est[c * dc: (c + 1) * dc]
        value = np.power(group_3D_est_c, 2) * coef
        value /= (value + sigma_table[c] * sigma_table[c])
        group_3D_est[c * dc: (c + 1) * dc] = group_3D_img_c * value * coef
        weight_table[c] += sum(value)

    for n in range(kWien_2 * chnls):
        hadamard_transform(group_3D_est, nSx_r, n * nSx_r)

    if doWeight:
        for c in range(chnls):
            weight_table[c] = 1. / (sigma_table[c] * sigma_table[c] * weight_table[c]) if weight_table[c] > 0. else 1.


def hadamard_transform(vec, n, start):
    h_mat = hadamard(n)
    v = vec[start: start + n]
    v_h = np.matmul(v, h_mat)
    vec[start: start + n] = v_h
