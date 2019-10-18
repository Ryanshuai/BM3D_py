import math
import pywt
import numpy as np


def bior_2d_forward(img):
    assert img.shape[-1] == img.shape[-2]
    N = img.shape[-1]
    iter_max = int(math.log2(N))

    fre_img = (img.copy()).astype(np.float64)
    for iter in range(iter_max):
        coeffs2 = pywt.dwt2(fre_img[..., :N, :N], 'bior1.5', mode='periodic')
        LL, (LH, HL, HH) = coeffs2
        fre_img[..., :N // 2, :N // 2] = LL[..., 2: -2, 2: -2]
        fre_img[..., N // 2:N, N // 2:N] = HH[..., 2: -2, 2: -2]
        fre_img[..., :N // 2, N // 2:N] = -HL[..., 2: -2, 2: -2]
        fre_img[..., N // 2:N, :N // 2] = -LH[..., 2: -2, 2: -2]
        N //= 2
    return fre_img


def bior_2d_reverse(bior_img):
    assert bior_img.shape[-1] == bior_img.shape[-2]
    N = bior_img.shape[-1]
    iter_max = int(math.log2(N))
    img = bior_img.copy()

    N = 2
    for iter in range(iter_max-1):
        LL = np.pad(img[..., :N, :N], (2, 2), 'wrap')
        HH = np.pad(img[..., N:2 * N, N:2 * N], (2, 2), 'wrap')
        HL = np.pad(-img[..., :N, N:2 * N], (2, 2), 'wrap')
        LH = np.pad(-img[..., N:2 * N, :N], (2, 2), 'wrap')
        coeffs = LL, (LH, HL, HH)
        N *= 2
        img[..., :N, :N] = pywt.idwt2(coeffs, 'bior1.5', mode='periodic')

    return img.astype(np.uint8)


if __name__ == '__main__':
    import cv2

    img = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    # img = img[:64, :64]
    bior_2d_forward(img)

    bior_2d_reverse(img)
    cv2.imshow('', img)
    cv2.waitKey()
