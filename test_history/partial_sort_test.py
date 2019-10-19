import numpy as np

mat = np.array([4, 3, 32, 72, 13, 11, 6, 70, 88, 5, 49, 6, 1, 0, 50, 20, 100])
# print(mat)
# sort = np.partition(mat, (0, 2))
# print(sort)

n = 10
index = np.argpartition(mat, range(n))
# index = index[:n]
for i, idx in enumerate(index):
    print(i, 'idx:', idx, 'value:', mat[idx])


# test_array = np.array([1, 16, 4, 100, 9, 9, 4, 121, 0, 36, 100, 1, 0, 16, 16, 0, 16, 64, 49, 121, 9, 49, 100, 100, 25])
# # argsort = np.argpartition(test_array, (0, 24))
# argsort = np.argsort(test_array)
# print(argsort)
# for arg in argsort:
#     print(test_array[arg])


test_array = np.array([[1, 5, 3, 2], [11, 33, 4, 16]])
# argsort = np.argpartition(test_array, (0, 24))
argsort = np.argsort(test_array, axis=1)
for i in range(len(argsort)):
    for arg in argsort[i]:
        print(test_array[i, arg])
