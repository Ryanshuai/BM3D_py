import numpy as np
from preProcess import preProcess
from precompute_bm import precompute_BM
from bior_2d import bior_2d_forward


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

    aiser_window, coef_norm, coef_norm_inv = preProcess(kHard)
    patch_table = precompute_BM()

    for i_r in row_ind:
        if tau_2D == 'DCT':
            dct_2d_process(img_noisy)
        elif tau_2D == 'BIOR':
            bior_2d_process(img_noisy)

        for j_r in column_ind:
            k_r = i_r * width + j_r
