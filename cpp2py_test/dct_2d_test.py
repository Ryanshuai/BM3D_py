from scipy.fftpack import dct, idct


def end_T(tensor):
    axes = list(range(tensor.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    tensor = tensor.transpose(axes)
    return tensor


def dct_2d_forward(block):
    block = end_T(block)
    block = dct(block, norm='ortho')
    block = end_T(block)
    block = dct(block, norm='ortho')
    return block


def dct_2d_reverse(block):
    block = end_T(block)
    block = idct(block, norm='ortho')
    block = end_T(block)
    block = idct(block, norm='ortho')
    return block


if __name__ == '__main__':
    import numpy as np

    im = np.ones((3, 2, 5, 6), dtype=np.uint64)

    fre_im = dct_2d_forward(im)
    print(fre_im)
    im_ = dct_2d_reverse(fre_im)

    print(im_)
