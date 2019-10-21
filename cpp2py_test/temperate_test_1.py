import numpy as np

b_dim = 5
a =  list(range(b_dim))
a[-1], a[-2] = a[-2], a[-1]
print(a)
# t_axis = tuple(())
# print(t_axis)