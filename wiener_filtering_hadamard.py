import numpy as np
from scipy.linalg import hadamard


def wiener_filtering_hadamard(group_3D_img, group_3D_est, sigma, doWeight):
    assert group_3D_img.shape == group_3D_est.shape
    nSx_r = group_3D_img.shape[-1]
    coef = 1.0 / nSx_r

    group_3D_img_h = hadamard_transform(group_3D_img)  # along nSx_r axis
    group_3D_est_h = hadamard_transform(group_3D_est)

    value = np.power(group_3D_est_h, 2) * coef
    value /= (value + sigma * sigma)
    group_3D_est_h = group_3D_img_h * value * coef
    weight = np.sum(value)

    group_3D_est = hadamard_transform(group_3D_est_h)

    if doWeight:
        weight = 1. / (sigma * sigma * weight) if weight > 0. else 1.

    return group_3D_est, weight


def hadamard_transform(vec):
    n = vec.shape[-1]
    h_mat = hadamard(n).astype(np.float64)
    v_h = vec @ h_mat
    return v_h
