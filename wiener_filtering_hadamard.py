import numpy as np
from scipy.linalg import hadamard


def wiener_filtering_hadamard(group_3D_img, group_3D_est, sigma, doWeight):
    assert group_3D_img.shape == group_3D_est.shap
    nSx_r = group_3D_img.shape[0]  # TODO 具体取哪个轴要和之前的代码对应
    coef = 1.0 / nSx_r

    hadamard_transform(group_3D_img)  # along nSx_r axis
    hadamard_transform(group_3D_est)

    value = np.power(group_3D_est, 2) * coef
    value /= (value + sigma * sigma)
    group_3D_est = group_3D_img * value * coef
    weight = sum(value)

    hadamard_transform(group_3D_est)

    if doWeight:
        weight = 1. / (sigma * sigma * weight) if weight > 0. else 1.

    return group_3D_est, weight


def hadamard_transform(vec):
    n = len(vec)
    h_mat = hadamard(n)
    v_h = vec @ h_mat
    return v_h
