import numpy as np
import cv2


def transport_2d_mat(mat, right, down):
    rows, cols = mat.shape
    t_M = np.float32([[1, 0, right], [0, 1, down]])
    t_img = cv2.warpAffine(mat, t_M, (cols, rows))
    return t_img


img = np.array([[12, 9, 8, 10, 11, 9, 17, 13, 17],
                [10, 9, 11, 13, 11, 11, 11, 16, 106],
                [11, 12, 12, 12, 11, 12, 11, 69, 181],
                [12, 13, 12, 12, 11, 12, 82, 168, 60],
                [11, 11, 10, 9, 10, 69, 182, 67, 14],
                [11, 10, 10, 10, 71, 200, 81, 15, 12],
                [12, 12, 12, 58, 204, 91, 17, 12, 14],
                [11, 11, 46, 201, 106, 18, 14, 16, 15],
                [10, 34, 185, 122, 23, 10, 14, 17, 16]])

di = -1
dj = -1
filt = np.array([[10, 69], [71, 200]])
for i in range(img.shape[0]-1):
    for j in range(img.shape[1]-1):
        a = img[i, j] - 10
        b = img[i, j+dj] - 69
        c = img[i+di, j] - 71
        d = img[i+di, j+dj] - 200

        print(a*a+b*b+c*c+d*d, i, j)
        if a*a+b*b+c*c+d*d == 48:
            print(i, j)





# di = -1
# dj = -1
#
# img = img.astype(np.uint8)
# t_img = transport_2d_mat(img, right=-dj, down=-di)
# diff_table = (img - t_img) * (img - t_img)
#
# print(diff_table)