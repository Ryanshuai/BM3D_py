import cv2
import numpy as np
import pywt

data = np.array([[1, 2], [3, 4]], dtype=np.float64)

coeffs = pywt.dwt2(data, 'bior1.5', mode='periodic')
data_recover = pywt.idwt2(coeffs, 'bior1.5', mode='periodic')

print(data_recover)

img = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
img = img[0:64, 0:64]
img = img.astype(np.float64)

coeffs = pywt.dwt2(img, 'bior1.5', mode='periodic')
img_recover = pywt.idwt2(coeffs, 'bior1.5', mode='periodic')
img_recover = img_recover.astype(np.uint8)

diff = img - img_recover
print(diff)
print('max diff', np.max(np.abs(diff)))
print('sum diff', np.sum(np.abs(diff)))

cv2.imshow('recover', img_recover)
cv2.waitKey()
