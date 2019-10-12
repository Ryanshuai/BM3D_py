import numpy as np
import math
import pywt
import matplotlib.pyplot as plt
import cv2



def bior_2d_forward(input, output, N, d_i, r_i, d_o):
    for i in range(N):
        for j in range(N):
            aaa = i * r_i + j + d_i
            print(aaa)
            a = input[i * r_i + j + d_i]
            input[i * r_i + j + d_i] = 0
            output[i * N + j + d_o] = a

            cv2.imshow('output', output[0:N*N].reshape(N, N))
            cv2.imshow('input', input.reshape(256, 256))
            cv2.waitKey(100)


if __name__ == '__main__':
    im = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    for i in range(im.shape[0]):
        im[i, i] = 0
        im[i, im.shape[0]-i-1] = 0

    im = im[:, :]
    width = im.shape[0]
    k = 16
    kWien_2 = k*k
    chnls = 1
    nWien = 8
    output = np.zeros((2 * nWien + 1) * width * chnls * kWien_2)
    im = im.flatten()
    bior_2d_forward(im, output, k, 0, width, 0)
    # cv2.imshow('output', output)
    # cv2.imshow('im', im)
    # cv2.waitKey()

