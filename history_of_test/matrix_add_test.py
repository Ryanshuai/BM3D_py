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
    return res_mat


if __name__ == '__main__':
    import cv2

    diff_table = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 4, 16, 0, 1, 1, 4, 0, 0],
                           [0, 0, 9, 16, 0, 1, 1, 1, 0, 0],
                           [0, 0, 1, 36, 16, 16, 9, 0, 0, 0],
                           [0, 0, 0, 4, 1, 4, 16, 1, 0, 0],
                           [0, 0, 1, 9, 16, 1, 1, 0, 0, 0],
                           [0, 0, 9, 1, 0, 1, 1, 4, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    wh = 10

    nHW = 2
    kHW = 1

    # im = np.zeros_like(im)
    height, width = diff_table.shape[0], diff_table.shape[1]
    add_mat = get_add_patch_matrix(width, nHW, kHW)

    summ = np.matmul(np.matmul(add_mat, diff_table), add_mat.T)
    print(summ)
    print(summ-diff_table)
