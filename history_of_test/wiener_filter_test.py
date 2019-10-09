import numpy as np


chnls = 2
nSx_r = 5
kWien_2 = 4
coef = 1.0 / nSx_r

sigma_table = np.array([1, 2])
weight_table = np.array([0.]*chnls)
group_3D_img = np.array([1, 2, 3, 1, 5, 6, 3, 2, 7, 9, 1, 2, 3, 1, 5, 6, 3, 2, 7, 9, 1, 2, 3, 1, 5, 1, 3, 2, 7, 1, 1, 2, 3, 1, 5, 6, 3, 2, 7, 9])
group_3D_est = np.array([0, 2, 3, 0, 5, 6, 3, 2, 7, 9, 0, 2, 3, 0, 5, 6, 3, 2, 7, 9, 0, 0, 3, 0, 5, 0, 3, 0, 7, 0, 0, 0, 3, 0, 5, 6, 3, 0, 7, 9])

for c in range(chnls):
    dc = c * nSx_r * kWien_2
    for k in range(kWien_2 * nSx_r):
        value = group_3D_est[dc + k] * group_3D_est[dc + k] * coef
        value /= (value + sigma_table[c] * sigma_table[c])
        group_3D_est[k + dc] = group_3D_img[k + dc] * value * coef
        weight_table[c] += value

# print(group_3D_img)
print(group_3D_est)
print(weight_table)
#############----------------------------------------------------------------------------------------------------------

sigma_table = np.array([1, 2])
weight_table = np.array([0.]*chnls)
group_3D_img = np.array([1, 2, 3, 1, 5, 6, 3, 2, 7, 9, 1, 2, 3, 1, 5, 6, 3, 2, 7, 9, 1, 2, 3, 1, 5, 1, 3, 2, 7, 1, 1, 2, 3, 1, 5, 6, 3, 2, 7, 9])
group_3D_est = np.array([0, 2, 3, 0, 5, 6, 3, 2, 7, 9, 0, 2, 3, 0, 5, 6, 3, 2, 7, 9, 0, 0, 3, 0, 5, 0, 3, 0, 7, 0, 0, 0, 3, 0, 5, 6, 3, 0, 7, 9])

for c in range(chnls):
    dc = nSx_r * kWien_2  # diff from original definition
    group_3D_img_c = group_3D_img[c*dc: (c+1)*dc]
    group_3D_est_c = group_3D_est[c*dc: (c+1)*dc]
    value = np.power(group_3D_est_c, 2) * coef
    value /= (value + sigma_table[c] * sigma_table[c])
    group_3D_est[c*dc: (c+1)*dc] = group_3D_img_c * value * coef
    weight_table[c] += sum(value)

# print(group_3D_img)
print(group_3D_est)
print(weight_table)
