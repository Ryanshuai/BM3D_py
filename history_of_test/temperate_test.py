import numpy as np



def hadamard_transform(vec, tmp, N, D):
    if (N == 1):
        return
    elif N == 2:
        a = vec[D + 0]
        b = vec[D + 1]
        vec[D + 0] = a + b
        vec[D + 1] = a - b
    else:
        n = N / 2
        for k in range(n):
            a = vec[D + 2 * k]
            b = vec[D + 2 * k + 1]
            vec[D + k] = a + b
            tmp[k] = a - b
        for k in range(n):
            vec[D + n + k] = tmp[k]

        hadamard_transform(vec, tmp, n, D)
        hadamard_transform(vec, tmp, n, D + n)

