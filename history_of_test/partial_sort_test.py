import  numpy as np


mat = np.array([[4, 3, 2, 13, 11], [5, 7, 6, 1, 0]])
print(mat)
# sort = np.partition(mat, (0, 2))
# print(sort)

arg = np.argpartition(mat, (0, 2))
print(arg)
arg = np.argpartition(mat, (0, 3))
print(arg)
# argsort = mat[arg]
# print(argsort)
