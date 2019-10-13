import numpy as np


def precompute_BM(img, width, height, kHW, NHW, nHW, pHW, tauMatch):
    Ns = 2 * nHW + 1
    threshold = tauMatch * kHW * kHW
    diff_table = np.zeros(width * height, dtype=np.int)
    sum_table = np.ones(((nHW + 1) * Ns, width * height), dtype=np.int) * 2 * threshold
    row_ind = ind_initialize(height - kHW + 1, nHW, pHW)
    column_ind = ind_initialize(width - kHW + 1, nHW, pHW)

    for di in range(nHW + 1):
        for dj in range(Ns):
            dk = int(di * width + dj) - int(nHW)
            ddk = di * Ns + dj
            for i in range(nHW, height - nHW):
                k = i * width + nHW
                for j in range(nHW, width - nHW):
                    diff_table[k] = (img[k + dk] - img[k]) * (img[k + dk] - img[k])
                    k += 1
            dn = nHW * width + nHW
            value = 0.0
            for p in range(kHW):
                pq = p * width + dn
                for q in range(kHW):
                    value += diff_table[pq]
                    pq += 1
            sum_table[ddk][dn] = value

            for j in range(nHW + 1, width - nHW):
                ind = nHW * width + j - 1
                sum = sum_table[ddk][ind]
                for p in range(kHW):
                    sum += diff_table[ind + p * width + kHW] - diff_table[ind + p * width]
                sum_table[ddk][ind + 1] = sum

            for i in range(nHW + 1, height - nHW):
                ind = (i - 1) * width + nHW
                sum = sum_table[ddk][ind]
                for q in range(kHW):
                    sum += diff_table[ind + kHW * width + q] - diff_table[ind + q]
                sum_table[ddk][ind + width] = sum

                k = i * width + nHW + 1
                pq = (i + kHW - 1) * width + kHW - 1 + nHW + 1
                for j in range(nHW + 1, width - nHW):
                    sum_table[ddk][k] = \
                        sum_table[ddk][k - 1] \
                        + sum_table[ddk][k - width] \
                        - sum_table[ddk][k - 1 - width] \
                        + diff_table[pq] \
                        - diff_table[pq - kHW] \
                        - diff_table[pq - kHW * width] \
                        + diff_table[pq - kHW - kHW * width]
                    k += 1
                    pq += 1

    table_distance = np.empty(shape=[0, 2], dtype=np.int)
    patch_table = np.zeros((width * height, NHW), dtype=np.int)
    for ind_i in row_ind:
        for ind_j in column_ind:
            k_r = ind_i * width + ind_j

            for dj in range(-nHW, nHW + 1):
                for di in range(nHW + 1):
                    if sum_table[dj + nHW + di * Ns][k_r] < threshold:
                        pair = np.array([[sum_table[dj + nHW + di * Ns][k_r], k_r + di * width + dj]], dtype=np.int)
                        table_distance = np.append(table_distance, pair, axis=0)

                for di in range(-nHW, 0):
                    if sum_table[-dj + nHW + (-di) * Ns][k_r] < threshold:
                        pair = np.array(
                            [[sum_table[-dj + nHW + (-di) * Ns][k_r + di * width + dj], k_r + di * width + dj]],
                            dtype=np.int)
                        table_distance = np.append(table_distance, pair, axis=0)
            return table_distance

            nSx_r = closest_power_of_2(len(table_distance) * 2) if NHW > len(
                table_distance) * 2 else NHW

            if nSx_r == 1 and len(table_distance) * 2 == 0:
                print('problem size')
                pair = np.array([[0, k_r]], dtype=np.int)
                table_distance = np.append(table_distance, pair, axis=0)

            # partial_sort(table_distance.begin(), table_distance.begin() + nSx_r,
            #                               table_distance.end(), ComparaisonFirst);
            sorted(table_distance, key=lambda x: x[0], )

            for n in range(nSx_r):
                patch_table[k_r][n] = table_distance[n][1]

            if nSx_r == 1:
                patch_table[k_r][0] = table_distance[0][1]

    return patch_table


def closest_power_of_2(n):
    r = 1
    while (r * 2 <= n):
        r *= 2
    return r


def ind_initialize(max_size, N, step):
    ind_set = np.empty(shape=[0], dtype=np.int)
    ind = N
    while (ind < max_size - N):
        ind_set = np.append(ind_set, np.array([ind]), axis=0)
        ind += step
    if ind_set[-1] < max_size - N - 1:
        ind_set = np.append(ind_set, np.array([max_size - N - 1]), axis=0)
    return ind_set


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

    row_ind = ind_initialize(height - kHW + 1, nHW, pHW)
    column_ind = ind_initialize(width - kHW + 1, nHW, pHW)
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
    argsort = argsort[:, :NHW]
    argsort = argsort - np.arange(argsort.shape[1]).reshape(1, argsort.shape[1])  # pah_paw --> pbh_pbw

    # return argsort.transpose((1, 0))
    return argsort


def transport_2d_mat(mat, right, down):
    rows, cols = mat.shape
    t_M = np.float32([[1, 0, right], [0, 1, down]])
    t_img = cv2.warpAffine(mat, t_M, (cols, rows))
    return t_img


if __name__ == '__main__':
    import cv2

    im = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    wh = 20
    im = im[:wh, :wh]

    im = np.zeros_like(im)
    height, width = im.shape[0], im.shape[1]

    im_flat = im.flatten()

    # a = precompute_BM(im, width, height, kHW=8, NHW=16, nHW=16, pHW=3, tauMatch=40)
    aaa = precompute_BM(im_flat, width, height, kHW=1, NHW=9, nHW=2, pHW=1, tauMatch=4000)
    bbb = my_precompute_BM(im, width, height, kHW=1, NHW=9, nHW=2, pHW=1, tauMatch=4000)

    print(aaa)
    # for b in bbb:
    #     print(b)

    # for a, b in zip(aaa, bbb):
    #     # print(a.reshape(wh, wh))
    #     # print(b.reshape(wh, wh))
    #     print('----------------')
    #     print(a)
    #     print(b)
    #
    # diff = aaa - bbb
    # for line in diff:
    #     print(line.reshape(wh, wh))

