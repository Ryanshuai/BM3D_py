import numpy as np
import cv2


def symetrize(img, width, height, chnls, N):
    w = width + 2 * N
    h = height + 2 * N
    img_sym = np.zeros((w * h * chnls))

    for c in range(chnls):
        dc = c * width * height
        dc_2 = c * w * h + N * w + N

        # ! Center of the image
        for i in range(height):
            for j in range(width):
                img_sym[dc_2 + i * w + j] = img[dc]
                dc += 1

        dc_2 = c * w * h
        for j in range(w):
            for i in range(N):
                img_sym[dc_2 + i * w] = img_sym[dc_2 + (2 * N - i - 1) * w]
                img_sym[dc_2 + (h - i - 1) * w] = img_sym[dc_2 + (h - 2 * N + i) * w]
            dc_2 += 1

        dc_2 = c * w * h
        for i in range(h):
            di = dc_2 + i * w
            for j in range(N):
                img_sym[di + j] = img_sym[di + 2 * N - j - 1]
                img_sym[di + w - j - 1] = img_sym[di + w - 2 * N + j]

    return img_sym


if __name__ == '__main__':
    img = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[0], img.shape[1]
    chnls = 1
    N = 10

    img_flat = img.flatten()
    img_sym = symetrize(img_flat, width, height, chnls, N)
    img_sym = img_sym.reshape((width + 2 * N, height + 2 * N)).astype(np.uint8)

    img_pad = np.pad(img, ((N, N), (N, N)), 'symmetric')

    diff = img_pad-img_sym
    print(np.max(np.abs(diff)))
    cv2.imshow('diff', np.abs(diff))
    cv2.waitKey()

