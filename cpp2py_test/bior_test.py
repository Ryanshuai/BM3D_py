import math
import pywt
import numpy as np


def bior_2d_forward(img):
    assert img.shape[-1] == img.shape[-2]
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


def bior_2d_reverse(bior_img):
    assert bior_img.shape[-1] == bior_img.shape[-2]
    iter_max = int(math.log2(bior_img.shape[-1]))

    N = 1
    rec_coeffs = [bior_img[..., 0:1, 0:1]]
    for i in range(iter_max):
        LL = bior_img[..., N:2 * N, N:2 * N]
        HL = -bior_img[..., 0:N, N: 2 * N]
        LH = -bior_img[..., N: 2 * N, 0:N]
        t = (LH, HL, LL)
        rec_coeffs.append(t)
        N *= 2

    rec_im = pywt.waverec2(rec_coeffs, 'bior1.5', mode='periodization')
    return rec_im


if __name__ == '__main__':
    import cv2

    img = cv2.imread('Untitled 2.png', cv2.IMREAD_GRAYSCALE)
    bior_img = bior_2d_forward(img)

    img_ = bior_2d_reverse(bior_img)


    diff = np.abs(img - img_)
    print(np.max(diff))
    print(np.sum(diff))

    cv2.imshow('', img_.astype(np.uint8))
    cv2.waitKey()
