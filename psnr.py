import numpy as np
import math


def compute_psnr(img1, img2, useFloat=False):
    PIXEL_MAX = 255.0
    if useFloat:
        img1= img1.astype(np.float64)
        img2= img2.astype(np.float64)
        PIXEL_MAX = 1.

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    import cv2

    img1 = cv2.imread('Cameraman256.png')
    img2 = cv2.imread('img_basic.png')

    img1 = cv2.imread('matlab_official_compare/Cameraman256.png', True)
    img2 = cv2.imread('matlab_official_compare/y_basic.png', True)

    psnr = compute_psnr(img1, img2)

    print(psnr)
