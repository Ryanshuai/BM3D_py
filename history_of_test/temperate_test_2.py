import numpy as np


argsort = np.array([[1, 2, 3], [3, 4, 1]])


argsort_p2p = argsort + np.arange(argsort.shape[0]).reshape((argsort.shape[0], 1))

print(argsort_p2p)
