import numpy as np


def get_transport_mat(im_s, k):
    temp = np.zeros((im_s, (im_s - k + 1) * k), dtype=np.int)
    for i in range(k):
        temp[i, i] = 1
    Trans = temp.copy()
    for i in range(1, im_s - k + 1):
        dT = np.roll(temp, i, axis=0)
        dT = np.roll(dT, i * k, axis=1)
        Trans += dT
    return Trans


def image2patches(im, k, p):
    '''
    :param im:
    :param k: patch size
    :param p: step TODO
    :return:
    '''
    assert im.ndim == 2
    assert im.shape[0] == im.shape[1]
    im_s = im.shape[0]

    Trans = get_transport_mat(im_s, k)
    repetition = Trans.T @ im @ Trans
    print(repetition)
    repetition = repetition.reshape((im_s - k + 1, k, im_s - k + 1, k))
    repetition = repetition.transpose((0, 2, 1, 3))
    return repetition


if __name__ == '__main__':
    import cv2

    im = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    for i in range(100):
        im[i, i] = 0
        im[i, i + 50] = 0
        im[i, i + 100] = 0
        im[i, i + 150] = 0
    cv2.imshow('im', im)

    k = 8
    res = image2patches(im, k)
    # for i in range(res.shape[0]):
    #     for j in range(res.shape[1]):
    #         cv2.imshow('patches', res[i, j].astype(np.uint8))
    #         cv2.waitKey(100)
