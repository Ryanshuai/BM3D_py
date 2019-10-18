import numpy as np
from scipy.linalg import hadamard
import math


def ht_filtering_hadamard(group_3D, sigma, lambdaHard3D, doWeight):  # group_3D shape=(n*n, nSx_r)
    nSx_r = group_3D.shape[-1]
    coef_norm = math.sqrt(nSx_r)
    coef = 1.0 / nSx_r

    group_3D_h = hadamard_transform(group_3D)

    T = lambdaHard3D * sigma * coef_norm
    T_3D = np.where(np.abs(group_3D_h) > T, 1, 0)
    weight = np.sum(T_3D)
    group_3D_h = np.where(np.abs(group_3D_h) > T, group_3D_h, 0.)

    group_3D = hadamard_transform(group_3D_h)

    group_3D *= coef
    if doWeight:
        weight = 1. / (sigma * sigma * weight) if weight > 0. else 1.

    return group_3D, weight


def hadamard_transform(vec):
    n = vec.shape[-1]
    h_mat = hadamard(n)
    v_h = vec @ h_mat
    return v_h
