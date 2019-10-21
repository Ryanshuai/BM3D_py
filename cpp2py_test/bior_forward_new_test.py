import pywt
import cv2
import numpy as np
import math

from cpp2py_test.bior_2d_forward_test1 import original_bior_2d_forward

im = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
# im = im[:8, :8]

#  <original result show>
wave_im_original = original_bior_2d_forward(img=im.astype(np.float64))
# cv2.imshow('wave_im_original', wave_im_original.astype(np.uint8))
#  </ original result show>

im_h = im.shape[0]
iter_max = int(math.log2(im_h))

coeffs = pywt.wavedec2(im, 'bior1.5', level=iter_max, mode='periodization')
wave_im = np.zeros_like(im, dtype=np.float64)

N = 1
wave_im[..., :N, :N] = coeffs[0]
for i in range(1, iter_max + 1):
    wave_im[..., N:2 * N, N:2 * N] = coeffs[i][2]
    wave_im[..., 0:N, N: 2 * N] = -coeffs[i][1]
    wave_im[..., N: 2 * N, 0:N] = -coeffs[i][0]
    N *= 2

# cv2.imshow('wave_im', wave_im.astype(np.uint8))
# diff = np.abs(wave_im - wave_im_original)
# cv2.imshow('diff', diff)
# print(np.max(diff))
# print(np.sum(diff))
# cv2.waitKey()

wave_im = wave_im
N = 1
rec_coeffs = [wave_im[..., 0:1, 0:1]]
for i in range(iter_max):
    LL = wave_im[..., N:2 * N, N:2 * N]
    HL = -wave_im[..., 0:N, N: 2 * N]
    LH = -wave_im[..., N: 2 * N, 0:N]
    t = (LH, HL, LL)
    rec_coeffs.append(t)
    N *= 2

rec_im = pywt.waverec2(rec_coeffs, 'bior1.5', mode='periodization')

print(rec_im)
diff = np.abs(rec_im - im)
print(np.max(diff))
print(np.sum(diff))

cv2.imshow('rec_im', rec_im.astype(np.uint8))
cv2.waitKey()
