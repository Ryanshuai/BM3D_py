import numpy as np


def precompute_BM(img, kHW, NHW, nHW, tauMatch):
    img = img.astype(np.int)
    height, width = img.shape
    Ns = 2 * nHW + 1
    threshold = tauMatch * kHW * kHW
    sum_table = np.ones((Ns * Ns, height, width), dtype=np.int) * 2 * threshold  # di*width+dj, ph, pw
    add_mat = get_add_patch_matrix(width, nHW, kHW)
    diff_margin = np.pad(np.ones((height - 2 * nHW, width - 2 * nHW)), ((nHW, nHW), (nHW, nHW)), 'constant',
                         constant_values=(0, 0)).astype(np.uint8)
    sum_margin = (1 - diff_margin) * 2 * threshold

    for di in range(-nHW, nHW + 1):
        for dj in range(-nHW, nHW + 1):
            ddk = (di + nHW) * Ns + dj + nHW
            t_img = translation_2d_mat(img, right=-dj, down=-di)
            diff_table_2 = (img - t_img) * (img - t_img) * diff_margin

            sum_diff_2 = np.matmul(np.matmul(add_mat, diff_table_2), add_mat.T)
            sum_table[ddk] = np.maximum(sum_diff_2, sum_margin)

    sum_table = sum_table.reshape((Ns * Ns, height * width))  # di_dj, ph_pw
    sum_table_T = sum_table.transpose((1, 0))  # ph_pw__di_dj

    argsort = np.argpartition(sum_table_T, range(NHW))[:, :NHW]
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


def get_add_patch_matrix(n, nHW, kHW):
    """
    :param n: len of mat
    :param nHW: len of search area
    :param kHW: len of patch
    :return: manipulate mat
    """
    mat = np.eye(n - 2 * nHW)
    mat = np.pad(mat, nHW, 'constant')
    res_mat = mat.copy()
    for k in range(1, kHW):
        res_mat += translation_2d_mat(mat, right=k, down=0)
    return res_mat


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
