import numpy as np


def precompute_BM(img, kHW, NHW, nHW, tauMatch):
    img = img.astype(np.float64)
    height, width = img.shape
    Ns = 2 * nHW + 1
    threshold = tauMatch * kHW * kHW
    sum_table = np.ones((Ns, Ns, height, width)) * 2 * threshold  # di, dj, ph, pw
    row_add_mat, column_add_mat = get_add_patch_matrix(height, width, nHW, kHW)
    diff_margin = np.pad(np.ones((height - 2 * nHW, width - 2 * nHW)), nHW, 'constant', constant_values=0.)
    sum_margin = (1 - diff_margin) * 2 * threshold

    for di in range(-nHW, nHW + 1):
        for dj in range(-nHW, nHW + 1):
            t_img = translation_2d_mat(img, right=-dj, down=-di)
            diff_table_2 = (img - t_img) * (img - t_img) * diff_margin

            sum_diff_2 = row_add_mat @ diff_table_2 @ column_add_mat
            sum_table[di + nHW, dj + nHW] = np.maximum(sum_diff_2, sum_margin)  # sum_table (2n+1, 2n+1, height, width)

    sum_table = sum_table.reshape((Ns * Ns, height * width))  # di_dj, ph_pw
    sum_table_T = sum_table.transpose((1, 0))  # ph_pw__di_dj
    argsort = np.argpartition(sum_table_T, range(NHW))[:, :NHW]
    argsort[:, 0] = (Ns * Ns - 1) // 2
    argsort_di = argsort // Ns - nHW
    argsort_dj = argsort % Ns - nHW
    near_pi = argsort_di.reshape((height, width, -1)) + np.arange(height)[:, np.newaxis, np.newaxis]
    near_pj = argsort_dj.reshape((height, width, -1)) + np.arange(width)[np.newaxis, :, np.newaxis]
    ri_rj_N__ni_nj = np.concatenate((near_pi[:, :, :, np.newaxis], near_pj[:, :, :, np.newaxis]), axis=-1)

    sum_filter = np.where(sum_table_T < threshold, 1, 0)
    threshold_count = np.sum(sum_filter, axis=1)
    threshold_count = closest_power_of_2(threshold_count, max_=NHW)
    threshold_count = threshold_count.reshape((height, width))

    return ri_rj_N__ni_nj, threshold_count


def get_add_patch_matrix(h, w, nHW, kHW):
    row_add = np.eye(h - 2 * nHW)
    row_add = np.pad(row_add, nHW, 'constant')
    row_add_mat = row_add.copy()
    for k in range(1, kHW):
        row_add_mat += translation_2d_mat(row_add, right=k, down=0)

    column_add = np.eye(w - 2 * nHW)
    column_add = np.pad(column_add, nHW, 'constant')
    column_add_mat = column_add.copy()
    for k in range(1, kHW):
        column_add_mat += translation_2d_mat(column_add, right=0, down=k)

    return row_add_mat, column_add_mat


def translation_2d_mat(mat, right, down):
    mat = np.roll(mat, right, axis=1)
    mat = np.roll(mat, down, axis=0)
    return mat


def closest_power_of_2(M, max_):
    M = np.where(max_ < M, max_, M)
    while max_ > 1:
        M = np.where((max_ // 2 < M) * (M < max_), max_ // 2, M)
        max_ //= 2
    return M


if __name__ == '__main__':
    import os
    import cv2
    from utils import add_gaussian_noise, symetrize

    # <hyper parameter>
    # ref_i, ref_j = 196, 142
    ref_i, ref_j = 164, 135
    # ref_i, ref_j = 271, 206

    kHW = 8
    NHW = 3
    nHW = 16
    tauMatch = 2500
    # <hyper parameter \>

    im = cv2.imread('test_data/image/Cameraman.png', cv2.IMREAD_GRAYSCALE)
    im = im[100:, :]
    ref_i, ref_j = 64, 135
    im_noisy = add_gaussian_noise(im, 10, seed=1)

    img_noisy_p = symetrize(im_noisy, nHW)
    near_pij, threshold_count = precompute_BM(img_noisy_p, kHW=kHW, NHW=NHW, nHW=nHW, tauMatch=tauMatch)

    im = cv2.cvtColor(img_noisy_p, cv2.COLOR_GRAY2RGB)
    # <draw search area>
    points_list = [(ref_j - nHW, ref_i - nHW), (ref_j + nHW, ref_i - nHW), (ref_j - nHW, ref_i + nHW),
                   (ref_j + nHW, ref_i + nHW)]
    for point in points_list:
        cv2.circle(im, point, 0, (0, 0, 255), 1)
    # <draw search area \>

    # <draw reference patch>
    cv2.rectangle(im, (ref_j, ref_i), (ref_j + kHW, ref_i + kHW), color=(255, 0, 0), thickness=1)
    # <draw reference patch \>

    # <draw similar patches>
    count = threshold_count[ref_i, ref_j]
    for i, Pnear in enumerate(near_pij[ref_i, ref_j]):
        if i == 0:
            continue
        if i > count:
            break
        y, x = Pnear
        cv2.rectangle(im, (x, y), (x + kHW, y + kHW), color=(0, 255, 0), thickness=1)
    # <draw similar patches \>

    # cv2.imshow('im', im)
    # cv2.waitKey()
    cv2.imwrite('BM_real_im_test.png', im)
