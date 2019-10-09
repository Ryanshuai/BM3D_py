import numpy as np

# a = np.empty(shape=[0])
# b = np.array([1])
# c = np.array([2])
#
# ab = np.append(a, b, axis=0)
# print(ab)
#
# abc = np.append(ab, c, axis=0)
# print(abc)


patch_table = np.zeros(shape=[10, 100])

# bbb = np.append(patch_table[3], [[555], axis=0)
patch_table[3] = np.array([5,5])
print(patch_table)
