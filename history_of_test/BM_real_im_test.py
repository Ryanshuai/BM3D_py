import numpy as np
import cv2


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


def my_precompute_BM(img, kHW, NHW, nHW, tauMatch):
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
            t_img = transport_2d_mat(img, right=-dj, down=-di)
            diff_table = (img - t_img) * (img - t_img) * diff_margin

            sum_t = np.matmul(np.matmul(add_mat, diff_table), add_mat.T)
            sum_table[ddk] = np.maximum(sum_t, sum_margin)

    sum_table = sum_table.reshape((Ns * Ns, height * width))  # di_dj, ph_pw
    sum_table_T = sum_table.transpose((1, 0))  # ph_pw__di_dj
    # print(sum_table_T[22].reshape(Ns, Ns))
    argsort = np.argsort(sum_table_T, axis=1)
    argsort_di = argsort // (Ns) - nHW
    argsort_dj = argsort % (Ns) - nHW
    Pr_S__Vnear = argsort_di * width + argsort_dj
    Pr_S__Pnear = Pr_S__Vnear + np.arange(Pr_S__Vnear.shape[0]).reshape((Pr_S__Vnear.shape[0], 1))
    Pr_N__Pnear = Pr_S__Pnear[:, :NHW]
    # for test
    # nn = 22
    # for ag, di, dj, posr, pr in zip(argsort[nn], argsort_di[nn], argsort_dj[nn], Pr_S__Vnear[nn], Pr_S__Pnear[nn]):
    #     print(ag, '\t', di, '\t', dj, '\t', posr, '\t', pr)
    # for test
    sum_filter = np.where(sum_table_T < threshold, 1, 0)
    threshold_count = np.sum(sum_filter, axis=1)

    return Pr_N__Pnear, argsort_di, argsort_dj, threshold_count


def transport_2d_mat(mat, right, down):
    rows, cols = mat.shape
    t_M = np.float32([[1, 0, right], [0, 1, down]])
    t_img = cv2.warpAffine(mat, t_M, (cols, rows))
    return t_img


if __name__ == '__main__':
    im = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    im_w = im.shape[1]

    kHW, NHW, nHW, tauMatch = 1, 5, 5, 1000
    Pr_N__Pnear, argsort_di, argsort_dj, threshold_count = my_precompute_BM(im, kHW=kHW, NHW=NHW, nHW=nHW, tauMatch=tauMatch)
    ref_y, ref_x = 100, 100
    Pr = ref_y * im_w + ref_x

    # print(im[ref_y - nHW:ref_y + nHW + 1, ref_x - nHW:ref_x + nHW + 1])

    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im = cv2.rectangle(im, (ref_x, ref_y), (ref_x + kHW, ref_y + kHW), color=(255, 0, 0), thickness=1)
    im = cv2.rectangle(im, (ref_x - nHW, ref_y - nHW), (ref_x + nHW, ref_y + nHW), color=(0, 0, 255), thickness=1)

    count = threshold_count[Pr]
    for i, Pnear in enumerate(Pr_N__Pnear[Pr]):
        di = argsort_di[Pr][i]
        dj = argsort_dj[Pr][i]
        print(di, dj)
        if i > count:
            break
        y = Pnear // im_w
        x = Pnear % im_w
        # print(y, x)
        im = cv2.rectangle(im, (x, y), (x + kHW, y + kHW), color=(0, 255, 0), thickness=1)

    cv2.imshow('im', im)
    cv2.waitKey()
