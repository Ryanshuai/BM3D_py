import numpy as np
import math
import pywt
import matplotlib.pyplot as plt
import cv2


def transport_2d_mat(mat, right, down):
    rows, cols = mat.shape
    t_M = np.float32([[1, 0, right], [0, 1, down]])
    t_img = cv2.warpAffine(mat, t_M, (cols, rows))
    return t_img


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
    print(res_mat)
    return res_mat


def precompute_BM(img, width, height, kHW, NHW, nHW, pHW, tauMatch):
    Ns = 2 * nHW + 1
    diff_table = np.zeros(width * height, dtype=np.int)

    for di in range(nHW + 1):
        for dj in range(Ns):
            dk = int(di * width + dj) - int(nHW)
            for i in range(nHW, height - nHW):
                k = i * width + nHW
                for j in range(nHW, width - nHW):
                    diff_table[k] = (img[k + dk] - img[k]) * (img[k + dk] - img[k])
                    k += 1
            return diff_table


def my2_precompute_BM(img, width, height, kHW, NHW, nHW, pHW, tauMatch):
    margin = np.pad(np.ones((height - 2 * nHW, width - 2 * nHW)), ((nHW, nHW), (nHW, nHW)), 'constant',
                    constant_values=(0, 0)).astype(np.uint8)
    for di in range(0, nHW + 1):
        for dj in range(-nHW, nHW + 1):
            t_img = transport_2d_mat(img, right=-dj, down=-di)
            diff_table = (t_img - img) * (t_img - img) * margin
            return diff_table


if __name__ == '__main__':
    import cv2

    im = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    wh = 10
    im = im[:wh, :wh]
    # im = np.ones_like(im)
    # im = np.arange(wh * wh).reshape(wh, wh).astype(np.uint8)
    print(im)

    # im = np.zeros_like(im)
    height, width = im.shape[0], im.shape[1]

    im_flat = im.flatten()

    # a = precompute_BM(im, width, height, kHW=8, NHW=16, nHW=16, pHW=3, tauMatch=40)
    aaa = precompute_BM(im_flat, width, height, kHW=1, NHW=25, nHW=2, pHW=1, tauMatch=4000)
    aaa = aaa.reshape(wh, wh)
    bbb = my2_precompute_BM(im, width, height, kHW=1, NHW=3, nHW=2, pHW=1, tauMatch=4000)

    # print(aaa)
    # print(bbb)
    # diff = aaa.reshape(wh, wh) - bbb
    # print(diff)

    add_mat = get_add_patch_matrix(10, 2, 2)

    aaa_sum = np.matmul(np.matmul(add_mat, aaa), add_mat.T)
    print(aaa_sum)
    bbb_sum = np.matmul(np.matmul(add_mat, bbb), add_mat.T)
    print(bbb_sum)
    print(aaa_sum-bbb_sum)
