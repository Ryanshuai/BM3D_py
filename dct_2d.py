from scipy.fftpack import dct, idct


def end_T(tensor):
    """
    :transpose the last two axes
    :param tensor:
    :return:
    """
    axes = list(range(tensor.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    tensor = tensor.transpose(axes)
    return tensor


def dct_2d_forward(block):
    """
    :forward 2d Discrete Cosine Transformation
    :param tensor:
    :return:
    """
    block = end_T(block)
    block = dct(block, norm='ortho')
    block = end_T(block)
    block = dct(block, norm='ortho')
    return block


def dct_2d_reverse(block):
    """
    :reverse 2d Discrete Cosine Transformation
    :param tensor:
    :return:
    """
    block = end_T(block)
    block = idct(block, norm='ortho')
    block = end_T(block)
    block = idct(block, norm='ortho')
    return block


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
