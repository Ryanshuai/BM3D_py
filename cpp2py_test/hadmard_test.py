import numpy as np
from scipy.linalg import hadamard


def hadamard_transform(vec, N, start):
    tmp = np.zeros_like(vec)
    if N == 1:
        return
    elif N == 2:
        a = vec[start + 0]
        b = vec[start + 1]
        vec[start + 0] = a + b
        vec[start + 1] = a - b
    else:
        n = N // 2
        for k in range(n):
            a = vec[start + 2 * k]
            b = vec[start + 2 * k + 1]
            vec[start + k] = a + b
            tmp[k] = a - b
        for k in range(n):
            vec[start + n + k] = tmp[k]

        hadamard_transform(vec, n, start)
        hadamard_transform(vec, n, start + n)


def hadamard_transform_(vec):
    n = vec.shape[-1]
    h_mat = hadamard(n)
    v_h = vec @ h_mat
    return v_h


if __name__ == '__main__':
    vec = np.array([11, 3, 23, 37])
    # vec = np.array([11, 3, 23, 37, 5, 7, 13, 29, ])
    vec = np.array([11, 3, 23, 37, 5, 7, 13, 29, 11, 3, 23, 37, 5, 7, 13, 29])
    # vec = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    N = len(vec)
    start = 0

    v = vec.copy()
    v_h = hadamard_transform_(v)
    print('my hadamard transform')
    print(v_h)
    vv = hadamard_transform_(v_h)
    print('my hadamard inverse transform')
    print(vv)

    v = vec.copy()
    hadamard_transform(v, N, start)
    print('original hadamard transform')
    print(v)
    hadamard_transform(v, N, start)
    print('original hadamard inverse transform')
    print(v)

