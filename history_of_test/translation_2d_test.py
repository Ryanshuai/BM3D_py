import numpy as np
import cv2


def translation_2d_mat_cv2(mat, right, down):
    mat = mat.astype(np.uint8)
    rows, cols = mat.shape
    t_M = np.float32([[1, 0, right], [0, 1, down]])
    t_img = cv2.warpAffine(mat, t_M, (cols, rows))
    return t_img


def translation_2d_mat(mat, right, down):
    mat = np.roll(mat, right, axis=1)
    mat = np.roll(mat, down, axis=0)
    return mat


if __name__ == '__main__':
    mat = np.random.randint(0, 20, (5, 5))
    mat_t_cv2 = translation_2d_mat_cv2(mat, 2, 1)
    mat_t_np = translation_2d_mat(mat, 2, 1)
    print(mat_t_cv2)
    print(mat_t_np)
    print(mat_t_np - mat_t_cv2)
