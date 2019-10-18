import numpy as np
import cv2
import math
import pywt


def matlab_bior15(mat):
    t_mat = np.array([[0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274,
                       0.353553390593274, 0.353553390593274, 0.353553390593274],
                      [0.219417649252501, 0.449283757993216, 0.449283757993216, 0.219417649252501, -0.219417649252501,
                       -0.449283757993216, -0.449283757993216, -0.219417649252501],
                      [0.569359398342846, 0.402347308162278, -0.402347308162278, - 0.569359398342846,
                       - 0.083506045090284, 0.083506045090284, -0.083506045090284, 0.083506045090284],
                      [-0.083506045090284, 0.083506045090284, -0.083506045090284, 0.083506045090284, 0.569359398342846,
                       0.402347308162278, -0.402347308162278, -0.569359398342846],
                      [0.707106781186547, -0.707106781186547, 0., 0., 0., 0., 0., 0],
                      [0., 0., 0.707106781186547, -0.707106781186547, 0., 0., 0., 0],
                      [0., 0., 0., 0., 0.707106781186547, -0.707106781186547, 0., 0],
                      [0., 0., 0., 0., 0., 0., 0.707106781186547, -0.707106781186547]])

    return mat @ t_mat


def my_bior15(img):
    iter_max = int(math.log2(img.shape[-1]))

    coeffs = pywt.wavedec2(img, 'bior1.5', level=iter_max, mode='periodization')
    wave_im = np.zeros_like(img, dtype=np.float64)

    N = 1
    wave_im[..., :N, :N] = coeffs[0]
    for i in range(1, iter_max + 1):
        wave_im[..., N:2 * N, N:2 * N] = coeffs[i][2]
        wave_im[..., 0:N, N: 2 * N] = -coeffs[i][1]
        wave_im[..., N: 2 * N, 0:N] = -coeffs[i][0]
        N *= 2
    return wave_im


if __name__ == '__main__':
    im = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    im = im[0:8, 0:8]
    im = np.eye(8)
    print(im)
    print(np.sum(im[0]))
    print(np.sum(im[0])*0.353553390593274)

    matlab_res = matlab_bior15(im)
    my_res = my_bior15(im)

    matlab_res = np.around(matlab_res, decimals=2)
    my_res = np.around(my_res, decimals=2)
    print(matlab_res)
    print(my_res)
