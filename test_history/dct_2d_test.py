from scipy.fftpack import dct, idct


def dct_2d_forward(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def dct_2d_reverse(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


if __name__ == '__main__':
    import numpy as np

    im = np.ones((3, 2, 5, 5), dtype=np.uint64)

    fre_im = dct_2d_forward(im)
    print(fre_im)
    im_ = dct_2d_reverse(fre_im)

    print(im_)
