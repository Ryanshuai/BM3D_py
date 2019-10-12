import numpy as np
import cv2


def original_build_3D_group(table_2D, i_r):
    group_3D = np.zeros((chnls * nSx_r * kHard_2))
    for c in range(chnls):
        for n in range(nSx_r):
            ind = patch_table_k_r[n] + (nHard - i_r) * width
            # ind = (nHard - i_r) * width
            for k in range(kHard_2):
                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] = \
                    table_2D[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width]
    return group_3D


def my_build_3D_group(table_2D, i_r, patch_table_k_r, nSx_r):
    chnls, nHard, width, kHard_2 = table_2D.shape
    nHard = (nHard - 1) // 2
    group_3D = np.zeros((chnls, nSx_r, kHard_2))
    for n in range(nSx_r):
        group_3D[:, n, :] = table_2D[:, nHard - i_r, patch_table_k_r[n], :]
    group_3D = group_3D.transpose((0, 2, 1))
    return group_3D


if __name__ == '__main__':
    nHard = 8
    width = 4
    chnls = 3
    kHard_2 = 16

    nSx_r = 2

    patch_table_k_r = np.array([0, 3])

    # table_2D = np.random.randint(0, 20, size=((2 * nHard + 1) * width * chnls * kHard_2))
    table_2D = np.arange((2 * nHard + 1) * width * chnls * kHard_2)
    table_2D = table_2D.reshape((chnls, (2 * nHard + 1), width, kHard_2))
    print(table_2D)

    i_r = 0
    group_3D_orginal = original_build_3D_group(table_2D.flatten(), i_r)
    group_3D_my = my_build_3D_group(table_2D, i_r, patch_table_k_r, nSx_r)
    print(group_3D_my.astype(np.int))
    print(group_3D_orginal - group_3D_my.flatten())
