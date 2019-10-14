import  numpy as np


mat = np.array([[4, 3, 2, 13, 11], [5, 7, 6, 1, 0]])
# print(mat)
# sort = np.partition(mat, (0, 2))
# print(sort)

arg = np.argpartition(mat, (0, 2))
# print(arg)
arg = np.argpartition(mat, (0, 3))
# print(arg)
# argsort = mat[arg]
# print(argsort)

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
