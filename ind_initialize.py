import numpy as np


def ind_initialize(max_size, N, step):
    ind = range(N, max_size-N, step)
    if ind[-1] < max_size - N - 1:
        ind = np.append(ind, np.array([max_size - N - 1]), axis=0)
    return ind
