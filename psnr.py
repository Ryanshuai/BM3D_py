import math

import numpy as np


def compute_psnr(img1, img2):
    img1 = img1.astype(np.float64) / 255.
    img2 = img2.astype(np.float64) / 255.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return 10 * math.log10(1. / mse)


if __name__ == '__main__':
    import cv2

    im_dir = 'test_data/image'
    im_s2_dir = 'result_compare/sigma2'
    im_s5_dir = 'result_compare/sigma5'
    im_s10_dir = 'result_compare/sigma10'

    im_name = 'Alley.png'

    im_path = 'test_data/image/Alley.png'
    im_1st_path = 'result_compare/sigma2/Alley_s2_cpp_1st_P39.781.png'
    im_2nd_path = 'result_compare/sigma2/Alley_s2_cpp_2nd_P44.582.png'
    im_1st_path = 'result_compare/sigma2/Alley_s2_py_1st_P44.369.png'
    im_2nd_path = 'result_compare/sigma2/Alley_s2_py_2nd_P44.62.png'

    img0 = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(im_1st_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(im_2nd_path, cv2.IMREAD_GRAYSCALE)

    psnr1 = compute_psnr(img0, img1)
    psnr2 = compute_psnr(img0, img2)

    print('img and img_basic PSNR: ', psnr1)
    print('img and img_denoised PSNR: ', psnr2)
