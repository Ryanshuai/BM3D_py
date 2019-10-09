import numpy as np


chnls = 2
nSx_r = 5
kHard_2 = 4

weight_table = np.array([0.]*chnls)
group_3D = np.array([1, 2, 3, 1, 5, 6, 3, 2, 7, 9, 1, 2, 3, 1, 5, 6, 3, 2, 7, 9, 1, 2, 3, 1, 5, 1, 3, 2, 7, 1, 1, 2, 3, 1, 5, 6, 3, 2, 7, 9])

Ts = [1, 2]
for c in range(chnls):
    dc = c * nSx_r * kHard_2
    T = Ts[c]
    for k in range(kHard_2 * nSx_r):
        if abs(group_3D[k + dc]) > T:
            weight_table[c] += 1
        else:
            group_3D[k + dc] = 0.


print(group_3D)
print(weight_table)


weight_table = np.array([0.]*chnls)
group_3D = np.array([1, 2, 3, 1, 5, 6, 3, 2, 7, 9, 1, 2, 3, 1, 5, 6, 3, 2, 7, 9, 1, 2, 3, 1, 5, 1, 3, 2, 7, 1, 1, 2, 3, 1, 5, 6, 3, 2, 7, 9])

Ts = [1, 2]
for c in range(chnls):
    dc = nSx_r * kHard_2
    T = Ts[c]
    group_3D_c = group_3D[c*dc: (c+1)*dc]
    group_3D[c*dc: (c+1)*dc] = np.where(group_3D_c > T, group_3D_c, 0)
    T_3D = np.where(group_3D_c > T, 1, 0)
    weight_table[c] = sum(T_3D)

print(group_3D)
print(weight_table)