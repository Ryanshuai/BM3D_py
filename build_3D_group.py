import numpy as np


def build_3D_group(fre_all_patches, N__ni_nj, nSx_r):
    """
    :stack frequency patches into a 3D block
    :param fre_all_patches: all frequency patches
    :param N__ni_nj: the position of the N most similar patches
    :param nSx_r: how many similar patches according to threshold
    :return: the 3D block
    """
    _, _, k, k_ = fre_all_patches.shape
    assert k == k_
    group_3D = np.zeros((nSx_r, k, k))
    for n in range(nSx_r):
        ni, nj = N__ni_nj[n]
        group_3D[n, :, :] = fre_all_patches[ni, nj]
    group_3D = group_3D.transpose((1, 2, 0))
    return group_3D  # shape=(k, k, nSx_r)
