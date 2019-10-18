import cv2

from utils import add_gaussian_noise, symetrize
from bm3d_1st_step import bm3d_1st_step
from bm3d_2nd_step import bm3d_2nd_step

# <hyper parameter> -------------------------------------------------------------------------------
sigma = 25

nHard = 16
kHard = 8
NHard = 16
pHard = 3
useSD_h = False
tau_2D_hard = 'BIOR'

nWien = 16
kWien = 8
NWien = 16
pWien = 3
useSD_w = True
tau_2D_wien = 'DCT'
# <\ hyper parameter> -----------------------------------------------------------------------------


img = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))
img_noisy = add_gaussian_noise(img, sigma)
# img_noisy = cv2.imread('matlab_official_compare/noisy_image.png', cv2.IMREAD_GRAYSCALE)

img_noisy_p = symetrize(img_noisy, nHard)
img_basic = bm3d_1st_step(sigma, img_noisy_p, nHard, kHard, NHard, pHard, useSD_h, tau_2D_hard)
img_basic = img_basic[nHard: -nHard, nHard: -nHard]

cv2.imwrite('img_basic.png', img_basic)

img_basic_p = symetrize(img_basic, nWien)
img_noisy_p = symetrize(img_noisy, nWien)
img_denoised = bm3d_2nd_step(sigma, img_noisy_p, img_basic_p, nWien, kWien, NWien, pWien, useSD_w, tau_2D_wien)
img_denoised = img_denoised[nWien: -nWien, nWien: -nWien]

cv2.imwrite('img_denoised.png', img_denoised)
