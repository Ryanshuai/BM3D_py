import numpy as np

height = 10
h = np.arange(height)[:, np.newaxis]
print(h)

width = 5
w = np.arange(width)[np.newaxis, :]
print(w)

