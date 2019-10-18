import numpy as np
import cv2

from utils import ind_initialize, get_kaiserWindow, get_coef, sd_weighting
from precompute_BM import precompute_BM
from bior_2d import bior_2d_forward, bior_2d_reverse
from image_to_patches import image2patches
from build_3D_group import build_3D_group
from wiener_filtering_hadamard import wiener_filtering_hadamard


def bm3d_2nd_step(sigma, img_noisy, img_basic, nWien, kWien, NWien, pWien, useSD, tau_2D):
    height, width = img_noisy.shape[0], img_noisy.shape[1]

    tauMatch = 400 if sigma < 35 else 3500  # ! threshold determinates similarity between patches

    row_ind = ind_initialize(height - kWien + 1, nWien, pWien)
    column_ind = ind_initialize(width - kWien + 1, nWien, pWien)

    kaiserWindow = get_kaiserWindow(kWien)
    ri_rj_N__ni_nj, threshold_count = precompute_BM(img_noisy, kHW=kWien, NHW=NWien, nHW=nWien, tauMatch=tauMatch)
    group_len = int(np.sum(threshold_count))
    group_3D_table = np.zeros((group_len, kWien, kWien))
    weight_table = np.ones((height, width))

    noisy_patches = image2patches(img_noisy, k=kWien, p=pWien)  # i_j_ipatch_jpatch__v
    basic_patches = image2patches(img_basic, k=kWien, p=pWien)  # i_j_ipatch_jpatch__v
    if tau_2D == 'DCT':
        pass
        # fre_all_patches = dct_2d_forward(all_patches)  # TODO
    else:  # 'BIOR'
        fre_noisy_patches = bior_2d_forward(noisy_patches)
        fre_basic_patches = bior_2d_forward(basic_patches)

    fre_noisy_patches = fre_noisy_patches.reshape((height - kWien + 1, height - kWien + 1, kWien, kWien))
    fre_basic_patches = fre_basic_patches.reshape((height - kWien + 1, height - kWien + 1, kWien, kWien))

    acc_pointer = 0
    for i_r in row_ind:
        for j_r in column_ind:
            nSx_r = threshold_count[i_r, j_r]
            group_3D_img = build_3D_group(fre_noisy_patches, ri_rj_N__ni_nj[i_r, j_r], nSx_r)
            group_3D_est = build_3D_group(fre_basic_patches, ri_rj_N__ni_nj[i_r, j_r], nSx_r)
            group_3D_img = group_3D_img.reshape(kWien * kWien, nSx_r)
            group_3D_est = group_3D_est.reshape(kWien * kWien, nSx_r)

            group_3D, weight = wiener_filtering_hadamard(group_3D_img, group_3D_est, sigma, not useSD)

            group_3D = group_3D.reshape(kWien, kWien, nSx_r)
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
        group_3D_table = bior_2d_reverse(group_3D_table)

    # for i in range(1000):
    #     patch = group_3D_table[i]
    #     print(i, '----------------------------')
    #     print(patch)
    #     cv2.imshow('', patch.astype(np.uint8))
    #     cv2.waitKey()

    group_3D_table *= kaiserWindow

    numerator = np.zeros_like(img_noisy, dtype=np.float)
    denominator = np.zeros_like(img_noisy, dtype=np.float)
    acc_pointer = 0
    for i_r in row_ind:
        for j_r in column_ind:
            nSx_r = threshold_count[i_r, j_r]
            N_ni_nj = ri_rj_N__ni_nj[i_r, j_r]
            group_3D = group_3D_table[acc_pointer:acc_pointer + nSx_r]
            acc_pointer += nSx_r
            weight = weight_table[i_r, j_r]
            for n in range(nSx_r):
                ni, nj = N_ni_nj[n]
                patch = group_3D[n]

                numerator[ni:ni + kWien, nj:nj + kWien] += patch * weight
                denominator[ni:ni + kWien, nj:nj + kWien] += kaiserWindow * weight

    img_denoised = numerator / denominator
    img_denoised = img_denoised.astype(np.uint8)
    return img_denoised
