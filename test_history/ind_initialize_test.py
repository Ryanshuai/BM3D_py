import numpy as np


def ind_initialize(max_size, N, step):
    ind_set = np.empty(shape=[0], dtype=np.int)
    ind = N
    while (ind < max_size - N):
        ind_set = np.append(ind_set, np.array([ind]), axis=0)
        ind += step
    if ind_set[-1] < max_size - N - 1:
        ind_set = np.append(ind_set, np.array([max_size - N - 1]), axis=0)
    return ind_set


def my_ind(max_size, N, step):
    ind = range(N, max_size-N, step)
    if ind[-1] < max_size - N - 1:
        ind = np.append(ind, np.array([max_size - N - 1]), axis=0)
    return ind


if __name__ == '__main__':
    max_size = 100
    N = 5
    for step in range(1, 20):
        ind1 = ind_initialize(max_size, N, step)
        ind2 = my_ind(max_size, N, step)

        print(all(ind1==ind2))
    # for i in ind1:
    #     print(i)
    # print('-----------------------')
    # for i in ind2:
    #     print(i)