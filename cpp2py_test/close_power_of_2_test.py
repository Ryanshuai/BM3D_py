import numpy as np


def my_closest_power_of_2(M, max):
    M = np.where(max < M, max, M)
    while max > 1:
        M = np.where((max // 2 < M) * (M < max), max // 2, M)
        max //= 2
    return M


def closest_power_of_2(n):
    r = 1
    while r * 2 <= n:
        r *= 2
    return r


if __name__ == '__main__':
    for i in range(20):
        print(i, closest_power_of_2(i))
    n = np.arange(60)
    res_n = my_closest_power_of_2(n, 16)
    for i, v in enumerate(res_n):
        print(i, '-->', v)
