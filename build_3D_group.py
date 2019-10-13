import numpy as np


def my_build_3D_group(table_2D, i_r, patch_table_k_r, nSx_r):
    chnls, nHard, width, kHard_2 = table_2D.shape
    nHard = (nHard - 1) // 2
    group_3D = np.zeros((chnls, nSx_r, kHard_2))
    for n in range(nSx_r):
        group_3D[:, n, :] = table_2D[:, nHard - i_r, patch_table_k_r[n], :]
    group_3D = group_3D.transpose((0, 2, 1))
    return group_3D
