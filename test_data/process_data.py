import cv2
import os

from utils import add_gaussian_noise

input_path = 'original_image'

for im_name in os.listdir(input_path):
    im_path = os.path.join(input_path, im_name)
    im = cv2.imread(im_path)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    h = im.shape[0]
    w = im.shape[1]
    if w != h:
        if w > h:
            im = im[:, -h:]
        else:
            im = im[-w:, :]

    if min(h, w) > 256:
        im = cv2.resize(im, (256, 256))

    os.makedirs('image', exist_ok=True)
    cv2.imwrite('image/' + im_name[:-4] + '.png', im)

    for sigma in [2, 5, 10, 20, 30, 40, 60, 80, 100]:
        os.makedirs('sigma' + str(sigma), exist_ok=True)
        im_noise = add_gaussian_noise(im, sigma, seed=0)
        cv2.imwrite('sigma' + str(sigma) + '/' + im_name[:-4] + '.png', im_noise)
