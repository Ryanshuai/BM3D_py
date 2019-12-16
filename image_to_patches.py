import numpy as np


def image2patches(im, patch_h, patch_w):
    im_h, im_w = im.shape[0], im.shape[1]
    patch_table = np.zeros((im_h - patch_h + 1, im_w - patch_w + 1, patch_h, patch_w), dtype=np.float64)
    for i in range(im_h - patch_h + 1):
        for j in range(im_w - patch_w + 1):
            patch_table[i][j] = im[i:i + patch_h, j:j + patch_w]

    return patch_table


if __name__ == '__main__':
    import cv2
    from utils import ind_initialize

    im = cv2.imread('test_data/image/Cameraman.png', cv2.IMREAD_GRAYSCALE)
    height, width = im.shape[0], im.shape[1]

    k = 8
    n = 16
    p = 3

    # row_ind = ind_initialize(height - k + 1, n, p)
    # column_ind = ind_initialize(width - k + 1, n, p)
    # im = cv2.resize(im, (512, 512))
    # for i in range(100):
    #     im[i, i] = 0
    #     im[i, i + 50] = 0
    #     im[i, i + 100] = 0
    #     im[i, i + 150] = 0
    # cv2.imshow('im', im)

    res = image2patches(im, 8, 8)
    print()
    # res = image2patches_v1(im, k, 0)
    # for i in range(res.shape[0]):
    #     for j in range(res.shape[1]):
    #         cv2.imshow('patches', res[i, j].astype(np.uint8))
    #         cv2.waitKey(100)
