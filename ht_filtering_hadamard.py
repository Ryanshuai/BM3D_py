import numpy as np
from scipy.linalg import hadamard
import math


def ht_filtering_hadamard(group_3D, sigma, lambdaHard3D, doWeight):  # group_3D shape=(n*n, nSx_r)
    nSx_r = group_3D.shape[-1]
    coef_norm = math.sqrt(nSx_r)
    coef = 1.0 / nSx_r

    hadamard_transform(group_3D)

    T = lambdaHard3D * sigma * coef_norm
    group_3D = np.where(group_3D > T, group_3D, 0)
    T_3D = np.where(group_3D > T, 1, 0)
    weight = sum(T_3D)

    hadamard_transform(group_3D)

    for k in range(group_3D.size):
        group_3D[k] *= coef
    if doWeight:
        weight = 1. / (sigma * sigma * weight) if weight > 0. else 1.

    return group_3D, weight


def hadamard_transform(vec):
    n = len(vec)
    h_mat = hadamard(n)
    v_h = vec @ h_mat
    return v_h
