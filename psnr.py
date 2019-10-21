import numpy as np
import math


def compute_psnr(img1, img2, PIXEL_MAX=255.0):
    img1= img1.astype(np.float64)
    img2= img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    import cv2

    # img1 = cv2.imread('Cameraman256.png')
    # img2 = cv2.imread('img_basic.png')

    img0 = cv2.imread('matlab_official_compare/Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread('matlab_official_compare/y_basic.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('matlab_official_compare/y_final.png', cv2.IMREAD_GRAYSCALE)

    psnr1 = compute_psnr(img0, img1)
    psnr2 = compute_psnr(img0, img2)

    print('img and img_basic PSNR: ', psnr1)
    print('img and img_denoised PSNR: ', psnr2)
