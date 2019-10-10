import numpy as np
import math

def bior15_coef():
    lp1 = np.zeros(10)
    hp1 = np.zeros(10)
    lp2 = np.zeros(10)
    hp2 = np.zeros(10)

    coef_norm = 1. / (math.sqrt(2.) * 128.)
    sqrt2_inv = 1. / math.sqrt(2.)

    lp1[0] =  3.
    lp1[1] = -3.
    lp1[2] = -22.
    lp1[3] =  22.
    lp1[4] =  128.
    lp1[5] =  128.
    lp1[6] =  22.
    lp1[7] = -22.
    lp1[8] = -3.
    lp1[9] =  3.

    hp1[4] = -sqrt2_inv
    hp1[5] =  sqrt2_inv

    lp2[4] = sqrt2_inv
    lp2[5] = sqrt2_inv

    hp2[0] =  3.
    hp2[1] =  3.
    hp2[2] = -22.
    hp2[3] = -22.
    hp2[4] =  128.
    hp2[5] = -128.
    hp2[6] =  22.
    hp2[7] =  22.
    hp2[8] = -3.
    hp2[9] = -3.

    for k in range(10):
        lp1[k] *= coef_norm
        hp2[k] *= coef_norm
    return lp1, hp1, lp2, hp2


if __name__ == '__main__':

    a, b, c, d = bior15_coef()
    print(a)
    print(b)
    print(c)
    print(d)
