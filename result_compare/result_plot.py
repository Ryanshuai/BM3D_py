import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from psnr import compute_psnr


im_dir = '../test_data/image'
sigma_list = [2, 5, 10, 20, 30, 40, 60, 80, 100]


for i, im_name in enumerate(os.listdir(im_dir)):
    im = cv2.imread(os.path.join(im_dir, im_name))

    f_p_c_1 = list()
    u_p_c_1 = list()
    f_p_c_2 = list()
    u_p_c_2 = list()

    f_p_p_1 = list()
    u_p_p_1 = list()
    f_p_p_2 = list()
    u_p_p_2 = list()

    for sigma in sigma_list:
        res_dir = 'sigma' + str(sigma)
        for res_im_name in os.listdir(res_dir):
            if im_name[:-4] in res_im_name:
                if 'cpp_1st' in res_im_name:
                    P_str = res_im_name.split('_')[-1][1:-4]
                    float_psnr_cpp_1st = float(P_str)
                    f_p_c_1.append(float_psnr_cpp_1st)

                    im_res = cv2.imread(os.path.join(res_dir ,res_im_name))
                    uint8_psnr_cpp_1st = compute_psnr(im, im_res)
                    u_p_c_1.append(uint8_psnr_cpp_1st)
                elif 'cpp_2nd' in res_im_name:
                    P_str = res_im_name.split('_')[-1][1:-4]
                    float_psnr_cpp_2nd = float(P_str)
                    f_p_c_2.append(float_psnr_cpp_2nd)

                    im_res = cv2.imread(os.path.join(res_dir ,res_im_name))
                    uint8_psnr_cpp_2nd = compute_psnr(im, im_res)
                    u_p_c_2.append(uint8_psnr_cpp_2nd)
                elif 'py_1st' in res_im_name:
                    P_str = res_im_name.split('_')[-1][1:-4]
                    float_psnr_py_1st = float(P_str)
                    f_p_p_1.append(float_psnr_py_1st)

                    im_res = cv2.imread(os.path.join(res_dir ,res_im_name))
                    uint8_psnr_py_1st = compute_psnr(im, im_res)
                    u_p_p_1.append(uint8_psnr_py_1st)
                elif 'py_2nd' in res_im_name:
                    P_str = res_im_name.split('_')[-1][1:-4]
                    float_psnr_py_2nd = float(P_str)
                    f_p_p_2.append(float_psnr_py_2nd)

                    im_res = cv2.imread(os.path.join(res_dir ,res_im_name))
                    uint8_psnr_py_2nd = compute_psnr(im, im_res)
                    u_p_p_2.append(uint8_psnr_py_2nd)

    x1 = [2, 5, 10, 20, 30, 40, 60, 80, 100]
    l1 = plt.plot(x1, u_p_c_2, 'g--', label='cpp_version')
    for a, b in zip(x1, u_p_c_2):
        plt.text(a, b, '%.2f' % b, ha='center', va='top', fontsize=8, color='g')

    l2 = plt.plot(x1, u_p_p_2, 'r--', label='py_version')
    for a, b in zip(x1, u_p_p_2):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=8, color='r')
    # l3 = plt.plot(x1, y3, 'b--', label='type3')
    # plt.plot(x1, y1, 'ro-', x2, y2, 'g+-', x3, y3, 'b^-')
    # plt.title('The Lasers in Three Conditions')
    plt.xlabel('sigma')
    plt.ylabel('psnr')
    plt.legend()
    plt.savefig(im_name)
    plt.show()

    if i > 0:
        break


