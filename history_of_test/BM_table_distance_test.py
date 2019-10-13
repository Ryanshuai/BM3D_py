import numpy as np


def precompute_BM(sum_table, width, height, kHW, NHW, nHW, pHW, tauMatch, ind_i, ind_j):
    Ns = 2 * nHW + 1
    threshold = tauMatch * kHW * kHW

    table_distance = np.empty(shape=[0, 5], dtype=np.int)

    k_r = ind_i * width + ind_j
    for dj in range(-nHW, nHW + 1):
        for di in range(nHW + 1):
            if sum_table[dj + nHW + di * Ns][k_r] < threshold:

                pair = np.array([[sum_table[dj + nHW + di * Ns][k_r], k_r + di * width + dj]], dtype=np.int)
                pair_ = np.array([[sum_table[dj + nHW + di * Ns][k_r], ind_i, ind_j, di, dj]], dtype=np.int)

                Ns = 2 * nHW + 1
                Ns_ = nHW + 1
                sum_table_ = sum_table.reshape((Ns_, Ns, height, width))
                sum_table_ = sum_table_.transpose((2, 3, 0, 1))                # print('sum_table_value: ', sum_table[dj + nHW + di * Ns][k_r])
                # print('di: ', di, 'dj:', dj)
                # print('pi: ', ind_i, 'pj: ', ind_j)
                table_distance = np.append(table_distance, pair_, axis=0)

                # table_distance = np.append(table_distance, pair, axis=0)

        for di in range(-nHW, 0):
            if sum_table[-dj + nHW + (-di) * Ns][k_r] < threshold:

                pair = np.array(
                    [[sum_table[-dj + nHW + (-di) * Ns][k_r + di * width + dj], k_r + di * width + dj]], dtype=np.int)
                pair_ = np.array(
                    [[sum_table[-dj + nHW + (-di) * Ns][k_r + di * width + dj], ind_i, ind_j, di, dj]], dtype=np.int)
                table_distance = np.append(table_distance, pair_, axis=0)
    return table_distance


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
        res_mat += transport_2d_mat(mat, right=k, down=0)
    return res_mat


def my_precompute_BM(img, width, height, kHW, NHW, nHW, pHW, tauMatch):
    Ns = 2 * nHW + 1
    Ns_ = 2 * nHW + 1
    Ns_ = nHW + 1
    threshold = tauMatch * kHW * kHW
    sum_table = np.ones((Ns * Ns_, height, width), dtype=np.int) * 2 * threshold  # di*width+dj, ph, pw

    add_mat = get_add_patch_matrix(width, nHW, kHW)

    diff_margin = np.pad(np.ones((height - 2 * nHW, width - 2 * nHW)), ((nHW, nHW), (nHW, nHW)), 'constant',
                         constant_values=(0, 0)).astype(np.uint8)
    sum_margin = (1 - diff_margin) * 2 * threshold

    # for di in range(-nHW, nHW + 1):
    for di in range(0, nHW + 1):
        for dj in range(-nHW, nHW + 1):
            ddk = di * Ns + dj + nHW
            t_img = transport_2d_mat(img, right=-dj, down=-di)
            diff_table = (img - t_img) * (img - t_img) * diff_margin

            sum_t = np.matmul(np.matmul(add_mat, diff_table), add_mat.T)
            sum_table[ddk] = np.maximum(sum_t, sum_margin)

    sum_table_ = sum_table.reshape((Ns * Ns_, height * width))  # di_dj, ph_pw
    sum_table_ = sum_table_.transpose((1, 0))  # ph_pw, di_dj
    sum_filter = np.where(sum_table_ < threshold, 1, 0)
    argsort = np.argpartition(sum_table_, (0, NHW))  # pah_paw --> pbh_pbw
    # argsort = argsort.reshape((height, width, Ns_, Ns))

    # argsort = argsort[:, :NHW]
    # argsort = argsort - np.arange(argsort.shape[1]).reshape(1, argsort.shape[1])  # pah_paw --> pbh_pbw

    # return argsort.transpose((1, 0))
    return sum_table_.reshape((height, width, Ns_, Ns))


def transport_2d_mat(mat, right, down):
    rows, cols = mat.shape
    t_M = np.float32([[1, 0, right], [0, 1, down]])
    t_img = cv2.warpAffine(mat, t_M, (cols, rows))
    return t_img


def ind_initialize(max_size, N, step):
    ind_set = np.empty(shape=[0], dtype=np.int)
    ind = N
    while (ind < max_size - N):
        ind_set = np.append(ind_set, np.array([ind]), axis=0)
        ind += step
    if ind_set[-1] < max_size - N - 1:
        ind_set = np.append(ind_set, np.array([max_size - N - 1]), axis=0)
    return ind_set


if __name__ == '__main__':
    import cv2
    from history_of_test.BM_diff_sum_test import precompute_BM_sum_table

    img = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    wh = 10
    img = img[:wh, :wh]
    height, width = img.shape[0], img.shape[1]

    im_flat = img.flatten()

    kHW = 1
    NHW = 9
    nHW = 2
    pHW = 1
    tauMatch = 4000

    sum_table = precompute_BM_sum_table(img, width, height, kHW, NHW, nHW, pHW, tauMatch)

    print(sum_table)
    row_ind = ind_initialize(height - kHW + 1, nHW, pHW)
    column_ind = ind_initialize(width - kHW + 1, nHW, pHW)

    for ind_i in row_ind:
        for ind_j in column_ind:
            table_distance = precompute_BM(sum_table, width, height, kHW, NHW, nHW, pHW, tauMatch, ind_i, ind_j)
            print(table_distance)
            break
        break
    td = my_precompute_BM(img, width, height, kHW, NHW, nHW, pHW, tauMatch)
    print(td[2, 2])


