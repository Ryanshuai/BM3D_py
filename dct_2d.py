from scipy.fftpack import dct, idct


def dct_2d_forward(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def dct_2d_reverse(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


if __name__ == '__main__':
    import cv2
    import numpy as np

    im = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    im = im.astype(np.float64)
    fre_im = dct_2d_forward(im)
    im_ = dct_2d_reverse(fre_im)
    im_ = im_.astype(np.uint8)

    diff = np.abs(im - im_)
    print(np.max(diff))
    print(np.sum(diff))

    cv2.imshow('im_', im_)
    cv2.waitKey()
