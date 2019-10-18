import numpy as np
from scipy.fftpack import dct, idct

img = np.array(
    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])

dct_img = dct(dct(img, axis=0), axis=1)
print(dct_img.astype(np.int))

recovered_img = idct(idct(dct_img, axis=1), axis=0)
print(recovered_img)


if __name__ == '__main__':
    import cv2

    cv2.imread('Cameraman256.png')
