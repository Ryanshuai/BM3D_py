import numpy as np


def sd_weighting(group_3D):
    N = group_3D.size

    mean = np.sum(group_3D)
    std = np.sum(group_3D * group_3D)

    res = (std - mean * mean / N) / (N - 1)
    weight = 1.0 / np.sqrt(res) if res > 0. else 0.
    return weight
