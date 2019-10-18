import numpy as np


def build_3D_group(fre_all_patches, N__ni_nj, nSx_r):
    _, _, k, k_ = fre_all_patches.shape
    assert k == k_
    group_3D = np.zeros((nSx_r, k, k))
    for n in range(nSx_r):
        ni, nj = N__ni_nj[n]
        group_3D[n, :, :] = fre_all_patches[ni, nj]
    group_3D = group_3D.transpose((1, 2, 0))
    return group_3D  # shape=(k, k, nSx_r)
