import numpy as np
from scipy.linalg import hadamard
import math


def ht_filtering_hadamard(group_3D, nSx_r, kHard, chnls, sigma_table, lambdaHard3D, doWeight):
    kHard_2 = kHard * kHard
    weight_table = np.array([0.]*chnls)
    coef_norm = math.sqrt(nSx_r)
    coef = 1.0 / nSx_r
    for n in range(kHard_2*chnls):
        hadamard_transform(group_3D, nSx_r, n * nSx_r)

    for c in range(chnls):
        dc = nSx_r * kHard_2
        T = lambdaHard3D * sigma_table[c] * coef_norm
        group_3D_c = group_3D[c * dc: (c + 1) * dc]
        group_3D[c * dc: (c + 1) * dc] = np.where(group_3D_c > T, group_3D_c, 0)
        T_3D = np.where(group_3D_c > T, 1, 0)
        weight_table[c] = sum(T_3D)

    for n in range(kHard_2*chnls):
        hadamard_transform(group_3D, nSx_r, n * nSx_r)
    for k in range(group_3D.size):
        group_3D[k] *= coef
    if doWeight:
        for c in range(chnls):
            weight_table[c] = 1. / (sigma_table[c] * sigma_table[c] * weight_table[c]) if weight_table[c] > 0. else 1.

    return group_3D, weight_table


def hadamard_transform(vec, n, start):
    h_mat = hadamard(n)
    v = vec[start: start+n]
    v_h = np.matmul(v, h_mat)
    vec[start: start+n] = v_h

