import numpy as np
import cv2
import math
from scipy.linalg import hadamard

from utils import ind_initialize, get_kaiserWindow, sd_weighting
from precompute_BM import precompute_BM
from bior_2d import bior_2d_forward, bior_2d_reverse
from dct_2d import dct_2d_forward, dct_2d_reverse
from image_to_patches import image2patches
from build_3D_group import build_3D_group


def ht_filtering_hadamard(group_3D, sigma, lambdaHard3D, doWeight):  # group_3D shape=(n*n, nSx_r)
    nSx_r = group_3D.shape[-1]
    coef_norm = math.sqrt(nSx_r)
    coef = 1.0 / nSx_r

    group_3D_h = hadamard_transform(group_3D)

    T = lambdaHard3D * sigma * coef_norm
    T_3D = np.where(np.abs(group_3D_h) > T, 1, 0)
    weight = np.sum(T_3D)
    # group_3D_h = np.where(np.abs(group_3D_h) > T, group_3D_h, 0.)

    group_3D = hadamard_transform(group_3D_h)

    group_3D *= coef
    if doWeight:
        weight = 1. / (sigma * sigma * weight) if weight > 0. else 1.

    return group_3D, weight


def hadamard_transform(vec):
    n = vec.shape[-1]
    h_mat = hadamard(n).astype(np.float64)
    v_h = vec @ h_mat
    return v_h


def bm3d_1st_step(sigma, img_noisy, nHard, kHard, NHard, pHard, lambdaHard3D, tauMatch, useSD, tau_2D):
    height, width = img_noisy.shape[0], img_noisy.shape[1]

    row_ind = ind_initialize(height - kHard + 1, nHard, pHard)
    column_ind = ind_initialize(width - kHard + 1, nHard, pHard)

    kaiserWindow = get_kaiserWindow(kHard)
    ri_rj_N__ni_nj, threshold_count = precompute_BM(img_noisy, kHW=kHard, NHW=NHard, nHW=nHard, tauMatch=tauMatch)
    group_len = int(np.sum(threshold_count))
    group_3D_table = np.zeros((group_len, kHard, kHard))
    weight_table = np.ones((height, width))

    all_patches = image2patches(img_noisy, k=kHard, p=pHard)  # i_j_ipatch_jpatch__v
    if tau_2D == 'DCT':
        fre_all_patches = dct_2d_forward(all_patches)
    else:  # 'BIOR'
        fre_all_patches = bior_2d_forward(all_patches)
    fre_all_patches = fre_all_patches.reshape((height - kHard + 1, height - kHard + 1, kHard, kHard))

    acc_pointer = 0
    for i_r in row_ind:
        for j_r in column_ind:
            nSx_r = threshold_count[i_r, j_r]
            group_3D = build_3D_group(fre_all_patches, ri_rj_N__ni_nj[i_r, j_r], nSx_r)
            group_3D, weight = ht_filtering_hadamard(group_3D, sigma, lambdaHard3D, not useSD)
            group_3D = group_3D.transpose((2, 0, 1))
            group_3D_table[acc_pointer:acc_pointer + nSx_r] = group_3D
            acc_pointer += nSx_r

            if useSD:
                weight = sd_weighting(group_3D)

            weight_table[i_r, j_r] = weight

    if tau_2D == 'DCT':
        group_3D_table = dct_2d_reverse(group_3D_table)
    else:  # 'BIOR'
        group_3D_table = bior_2d_reverse(group_3D_table)

    # group_3D_table = np.maximum(group_3D_table, 0)
    # for i in range(1000):
    #     patch = group_3D_table[i]
    #     print(i, '----------------------------')
    #     print(patch)
    #     print(np.min(patch))
    #     print(np.max(patch))
    #     print(np.sum(patch))
    #     cv2.imshow('', patch.astype(np.uint8))
    #     cv2.waitKey()

    group_3D_table *= kaiserWindow

    numerator = np.zeros_like(img_noisy, dtype=np.float64)
    denominator = np.zeros_like(img_noisy, dtype=np.float64)
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

                numerator[ni:ni + kHard, nj:nj + kHard] += patch * weight
                denominator[ni:ni + kHard, nj:nj + kHard] += kaiserWindow * weight

    img_basic = numerator / denominator
    return img_basic


if __name__ == '__main__':
    from utils import add_gaussian_noise, symetrize

    # <hyper parameter> -------------------------------------------------------------------------------
    sigma = 20

    nHard = 16
    kHard = 8
    NHard = 16
    pHard = 3
    lambdaHard3D = 2.7  # ! Threshold for Hard Thresholding
    tauMatchHard = 2500 if sigma < 35 else 5000  # ! threshold determinates similarity between patches
    useSD_h = False
    tau_2D_hard = 'BIOR'
    # <\ hyper parameter> -----------------------------------------------------------------------------

    img = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (128, 128))
    img_noisy = add_gaussian_noise(img, sigma)

    img_noisy_p = symetrize(img_noisy, nHard)
    img_basic = bm3d_1st_step(sigma, img_noisy_p, nHard, kHard, NHard, pHard, lambdaHard3D, tauMatchHard, useSD_h,
                              tau_2D_hard)
    img_basic = img_basic[nHard: -nHard, nHard: -nHard]

    diff = np.abs(img_basic - img_noisy)
    print('sum of diff', np.sum(diff))
    print('max of diff', np.max(diff))
    cv2.imshow('diff', diff)
    cv2.waitKey()
