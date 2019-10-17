import numpy as np

from ind_initialize import ind_initialize
from preProcess import preProcess
from precompute_BM import precompute_BM
from bior_2d import bior_2d_forward
from image_to_patches import image2patches
from build_3D_group import build_3D_group
from ht_filtering_hadamard import ht_filtering_hadamard
from sd_weighting import sd_weighting
from bior_2d import bior_2d_reverse


def bm3d_1st_step(sigma, img_noisy, nHard, kHard, NHard, pHard, useSD, tau_2D):
    height, width = img_noisy.shape[0], img_noisy.shape[1]

    lambdaHard3D = 2.7  # ! Threshold for Hard Thresholding
    tauMatch = 3 * (2500 if sigma < 35 else 5000)  # ! threshold determinates similarity between patches

    row_ind = ind_initialize(height - kHard + 1, nHard, pHard)
    column_ind = ind_initialize(width - kHard + 1, nHard, pHard)

    kaiserWindow, coef_norm, coef_norm_inv = preProcess(kHard)
    ri_rj_N__ni_nj, threshold_count = precompute_BM(img_noisy, kHW=kHard, NHW=NHard, nHW=nHard, tauMatch=tauMatch)
    group_len = int(np.sum(threshold_count))
    group_3D_table = np.zeros((group_len, kHard, kHard))
    weight_table = np.zeros((height, width))

    all_patches = image2patches(img_noisy, k=kHard, p=pHard)  # i_j_ipatch_jpatch__v
    if tau_2D == 'DCT':
        pass
        # fre_all_patches = dct_2d_forward(all_patches)  # TODO
    else:  # 'BIOR'
        fre_all_patches = bior_2d_forward(all_patches)
    fre_all_patches = fre_all_patches.reshape((height-kHard+1, height-kHard+1, kHard, kHard))

    acc_pointer = 0
    for i_r in row_ind:
        for j_r in column_ind:
            nSx_r = threshold_count[i_r, j_r]
            group_3D = build_3D_group(fre_all_patches, ri_rj_N__ni_nj[i_r, j_r], nSx_r)
            group_3D = group_3D.reshape(kHard * kHard, nSx_r)
            group_3D, weight = ht_filtering_hadamard(group_3D, sigma, lambdaHard3D, not useSD)
            group_3D = group_3D.reshape(kHard, kHard, nSx_r)
            group_3D = group_3D.transpose((2, 0, 1))
            group_3D_table[acc_pointer:acc_pointer + nSx_r] = group_3D
            acc_pointer += nSx_r

            if useSD:
                weight = sd_weighting(group_3D)

            weight_table[i_r, j_r] = weight

    if tau_2D == 'DCT':
        pass
        # dct_2d_reverse(group_3D_table)  # TODO
    else:  # 'BIOR'
        bior_2d_reverse(group_3D_table)

    group_3D_table *= kaiserWindow

    numerator = np.zeros_like(img_noisy, dtype=np.float)
    denominator = np.zeros_like(img_noisy, dtype=np.float)
    acc_pointer = 0
    for i_r in row_ind:
        for j_r in column_ind:
            nSx_r = threshold_count[i_r, j_r]
            Pnear = ri_rj_N__ni_nj[i_r, j_r]
            group_3D = group_3D_table[acc_pointer:acc_pointer + nSx_r]
            acc_pointer += nSx_r
            weight = weight_table[i_r, j_r]
            for n in nSx_r:
                P_r = Pnear[n]
                patch = group_3D[n]
                Inear = P_r // weight
                Jnear = P_r % weight

                numerator[Inear:Inear+kHard, Jnear:Jnear+kHard] += patch * weight
                denominator[Inear:Inear+kHard, Jnear:Jnear+kHard] += kaiserWindow * weight

    img_basic= numerator / denominator
    return img_basic