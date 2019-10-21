import cv2
import numpy as np

from utils import add_gaussian_noise, symetrize
from bm3d_1st_step import bm3d_1st_step
from bm3d_2nd_step import bm3d_2nd_step
from psnr import compute_psnr

# <hyper parameter> -------------------------------------------------------------------------------
sigma = 20

nHard = 16
kHard = 8
NHard = 16
pHard = 3
lambdaHard3D = 2.7  # ! Threshold for Hard Thresholding
tauMatchHard = 2500 if sigma < 35 else 5000  # ! threshold determinates similarity between patches
useSD_h = False
tau_2D_hard = 'BIOR'

nWien = 16
kWien = 8
NWien = 32
pWien = 3
tauMatchWien = 400 if sigma < 35 else 3500  # ! threshold determinates similarity between patches
useSD_w = True
tau_2D_wien = 'DCT'
# <\ hyper parameter> -----------------------------------------------------------------------------


img = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (128, 128))
# img_noisy = add_gaussian_noise(img, sigma)

img_noisy = cv2.imread('image_noise.png', cv2.IMREAD_GRAYSCALE)

img_noisy_p = symetrize(img_noisy, nHard)
img_basic = bm3d_1st_step(sigma, img_noisy_p, nHard, kHard, NHard, pHard, lambdaHard3D, tauMatchHard, useSD_h,
                          tau_2D_hard)
img_basic = img_basic[nHard: -nHard, nHard: -nHard]

cv2.imwrite('y_basic.png', img_basic.astype(np.uint8))

img_basic_p = symetrize(img_basic, nWien)
img_noisy_p = symetrize(img_noisy, nWien)
img_denoised = bm3d_2nd_step(sigma, img_noisy_p, img_basic_p, nWien, kWien, NWien, pWien, tauMatchWien, useSD_w,
                             tau_2D_wien)
img_denoised = img_denoised[nWien: -nWien, nWien: -nWien]

cv2.imwrite('y_final.png', img_denoised.astype(np.uint8))

psnr_1st = compute_psnr(img, img_basic)
psnr_2st = compute_psnr(img, img_denoised)
print('img and img_basic PSNR: ', psnr_1st)
print('img and img_denoised PSNR: ', psnr_2st)
