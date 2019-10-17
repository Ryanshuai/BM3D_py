import cv2

from utils import add_gaussian_noise, symetrize
from bm3d_1st_step import bm3d_1st_step

# <hyper parameter> -------------------------------------------------------------------------------
sigma = 10.

nHard = 16
kHard = 8
NHard = 16
pHard = 3
useSD = True
tau_2D = ?

nWiener = 16
kWiener = 8
NWiener = 16
pWiener = 3

# <\ hyper parameter> -----------------------------------------------------------------------------


img = cv2.imread('Cameraman256.png')
img_noisy = add_gaussian_noise(img, sigma)

img_noisy = symetrize(img_noisy, nHard)

bm3d_1st_step(sigma, img_noisy, nHard, kHard, NHard, pHard, useSD, tau_2D)




