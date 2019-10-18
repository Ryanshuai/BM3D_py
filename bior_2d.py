import math
import pywt
import numpy as np


def bior_2d_forward(img):
    assert img.shape[-1] == img.shape[-2]
    N = img.shape[-1]
    iter_max = int(math.log2(N))

    fre_img = (img.copy()).astype(np.float64)
    for iter in range(iter_max):
        coeffs2 = pywt.dwt2(fre_img[..., :N, :N], 'bior1.5', mode='periodization')
        LL, (LH, HL, HH) = coeffs2
        fre_img[..., :N // 2, :N // 2] = LL
        fre_img[..., N // 2:N, N // 2:N] = HH
        fre_img[..., :N // 2, N // 2:N] = -HL
        fre_img[..., N // 2:N, :N // 2] = -LH
        N //= 2
    return fre_img


def bior_2d_reverse(bior_img):
    assert bior_img.shape[-1] == bior_img.shape[-2]
    N = bior_img.shape[-1]
    iter_max = int(math.log2(N))
    img = bior_img.copy()

    N = 2
    for iter in range(iter_max - 1):
        LL = img[..., :N, :N]
        HH = img[..., N:2 * N, N:2 * N]
        HL = -img[..., :N, N:2 * N]
        LH = -img[..., N:2 * N, :N]
        coeffs = LL, (LH, HL, HH)
        N *= 2
        img[..., :N, :N] = pywt.idwt2(coeffs, 'bior1.5', mode='periodization')

    return img


if __name__ == '__main__':
    import cv2

    img = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    img_ = img.copy()
    fre_img = bior_2d_forward(img)

    img = bior_2d_reverse(fre_img)

    diff = np.abs(img - img_)
    print(np.max(diff))
    print(np.sum(diff))

    cv2.imshow('', img)
    cv2.waitKey()