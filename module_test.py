import cv2
import numpy as np
import precompute_bm as p


im = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
im = im[:64][:64]
height, width = im.shape[0], im.shape[1]

im = im.reshape(height*width)

a = p.precompute_BM(im, width, height, kHW=8, NHW=16, nHW=16, pHW=3, tauMatch=400)
print(a)




