import numpy as np
import math


def compute_psnr(img1, img2):
    img1 = img1.astype(np.float64) / 255.
    img2 = img2.astype(np.float64) / 255.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return 10 * math.log10(1. / mse)


if __name__ == '__main__':
    import cv2

    img0 = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    imgn = cv2.imread('mage_noise.png', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread('y_basic.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('y_final.png', cv2.IMREAD_GRAYSCALE)

    # img0 = cv2.imread('matlab_official_compare/Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    imgn = cv2.imread('matlab_official_compare/image_noise.png', cv2.IMREAD_GRAYSCALE)
    # img1 = cv2.imread('matlab_official_compare/y_basic.png', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('matlab_official_compare/y_final.png', cv2.IMREAD_GRAYSCALE)

    psnrn = compute_psnr(img0, imgn)
    psnr1 = compute_psnr(img0, img1)
    psnr2 = compute_psnr(img0, img2)

    print('img and img_noise PSNR: ', psnrn)
    print('img and img_basic PSNR: ', psnr1)
    print('img and img_denoised PSNR: ', psnr2)
