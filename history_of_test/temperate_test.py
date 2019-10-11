import numpy as np
import math
import pywt
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    # mat = np.arange(1, 65).reshape(8, 8)
    mat = np.ones((8, 8))
    # mat = np.ones((4, 4))
    # mat = np.ones((2, 2))
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    # original = cv2.imread('Cameraman256.png')
    original = pywt.data.camera()

    coeffs2 = pywt.dwt2(original, 'bior1.5')
    LL, (LH, HL, HH) = coeffs2
    print(LL.shape)
    print(LH.shape)
    print(HL.shape)
    print(HH.shape)
    plt.imshow(original)
    plt.colorbar(shrink=0.8)
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

