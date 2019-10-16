import numpy as np


def image2patches(im, k):
    '''
    :param im:
    :param k: patch size
    :return: all patches in image
    '''
    assert im.ndim == 2
    assert im.shape[0] == im.shape[1]
    im_h = im.shape[0]



im_s = 5
k = 3

A = np.arange(im_s*im_s).reshape((im_s, im_s))

T = np.zeros(im_s, (im_s-k+1)*k)
for i in range(k):
    T[i, i] = 1


