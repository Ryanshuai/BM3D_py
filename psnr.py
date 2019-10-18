import numpy
import math


def compute_psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    import cv2

    img1 = cv2.imread('Cameraman256.png')
    img2 = cv2.imread('img_basic.png')

    psnr = compute_psnr(img1, img2)

    print(psnr)
