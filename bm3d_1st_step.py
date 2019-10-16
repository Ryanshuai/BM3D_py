import numpy as np
from preProcess import preProcess
from precompute_BM import precompute_BM
from bior_2d import bior_2d_forward
from image_to_patches import image2patches
from build_3D_group import build_3D_group
from ht_filtering_hadamard import ht_filtering_hadamard


def bm3d_1st_step(sigma, img_noisy, nHard, kHard, NHard, pHard, useSD, color_space, tau_2D):
    height, width = img_noisy.shape[0], img_noisy.shape[1]
    chnls = 1 if img_noisy.dim() < 3 else img_noisy.shape[2]

    sigma_table = np.zeros((chnls))
    estimate_sigma

    lambdaHard3D = 2.7  # ! Threshold for Hard Thresholding
    tauMatch = (4 - chnls) * (
        2500 if sigma_table[0] < 35 else 5000)  # ! threshold used to determinate similarity between patches

    row_ind = ind_initialize()
    column_ind = ind_initialize()
    group_3D_table = np.zeros((nSx_r, kHard, kHard))

    aiser_window, coef_norm, coef_norm_inv = preProcess(kHard)
    Pr_N__Pnear, threshold_count = precompute_BM(img_noisy, kHW=kHard, NHW=NHard, nHW=nHard, tauMatch=tauMatch)

    all_patches = image2patches(img_noisy, k=kHard, p=pHard)  # i_j_ipatch_jpatch__v
    if tau_2D == 'DCT':
        fre_all_patches = 1#TODO
    else:   # 'BIOR'
        fre_all_patches = bior_2d_forward(all_patches)

    for i_r in row_ind:
        for j_r in column_ind:
            k_r = i_r * width + j_r
            nSx_r = threshold_count[k_r]
            group_3D = build_3D_group(fre_all_patches, Pr_N__Pnear[k_r], nSx_r)
            ht_filtering_hadamard(group_3D, nSx_r, kHard, chnls, sigma_table, lambdaHard3D, not useSD)

            if useSD:
                sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table)





