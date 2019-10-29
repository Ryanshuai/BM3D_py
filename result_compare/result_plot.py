import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from psnr import compute_psnr


def compare_psnr(im_name):
    sigma_list = [2, 5, 10, 20, 30, 40, 60, 80, 100]

    im_dir = '../test_data/image'
    py_res_dir = 'python'
    cpp_res_dir = 'cpp'

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
        for cpp_res_im_name in os.listdir(cpp_res_dir):
            if im_name[:-4] + '_s' + str(sigma) + '_' in cpp_res_im_name:
                if 'cpp_1st' in cpp_res_im_name:
                    P_str = cpp_res_im_name.split('_')[-1][1:-4]
                    float_psnr_cpp_1st = float(P_str)
                    f_p_c_1.append(float_psnr_cpp_1st)

                    im_res = cv2.imread(os.path.join(cpp_res_dir, cpp_res_im_name))
                    uint8_psnr_cpp_1st = compute_psnr(im, im_res)
                    u_p_c_1.append(uint8_psnr_cpp_1st)
                elif 'cpp_2nd' in cpp_res_im_name:
                    P_str = cpp_res_im_name.split('_')[-1][1:-4]
                    float_psnr_cpp_2nd = float(P_str)
                    f_p_c_2.append(float_psnr_cpp_2nd)

                    im_res = cv2.imread(os.path.join(cpp_res_dir, cpp_res_im_name))
                    uint8_psnr_cpp_2nd = compute_psnr(im, im_res)
                    u_p_c_2.append(uint8_psnr_cpp_2nd)
        for py_res_im_name in os.listdir(py_res_dir):
            if im_name[:-4] + '_s' + str(sigma) + '_' in py_res_im_name:
                if 'py_1st' in py_res_im_name:
                    P_str = py_res_im_name.split('_')[-1][1:-4]
                    float_psnr_py_1st = float(P_str)
                    f_p_p_1.append(float_psnr_py_1st)

                    im_res = cv2.imread(os.path.join(py_res_dir, py_res_im_name))
                    uint8_psnr_py_1st = compute_psnr(im, im_res)
                    u_p_p_1.append(uint8_psnr_py_1st)
                elif 'py_2nd' in py_res_im_name:
                    P_str = py_res_im_name.split('_')[-1][1:-4]
                    float_psnr_py_2nd = float(P_str)
                    f_p_p_2.append(float_psnr_py_2nd)

                    im_res = cv2.imread(os.path.join(py_res_dir, py_res_im_name))
                    uint8_psnr_py_2nd = compute_psnr(im, im_res)
                    u_p_p_2.append(uint8_psnr_py_2nd)

    l1 = plt.plot(sigma_list, u_p_c_2, 'g--', label='cpp_version')
    l2 = plt.plot(sigma_list, u_p_p_2, 'b--', label='py_version')

    for x, cpp, py in zip(sigma_list, u_p_c_2, u_p_p_2):
        plt.text(x, cpp, 'cpp' + '%.3f' % cpp, ha='center', va='top', fontsize=10, color='g')
        plt.text(x, py, 'py' + '%.3f' % py, ha='center', va='bottom', fontsize=10, color='b')
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.title(im_name)
    plt.xlabel('sigma')
    plt.ylabel('psnr')
    plt.legend()
    plt.savefig(im_name)
    plt.show()


if __name__ == '__main__':
    im_name_list = ['Alley.png', 'Baboon.png', 'barbara.png', 'boat.png', 'Book.png', 'Building1.png', 'Building2.png',
                    'Cameraman.png', 'Computer.png', 'couple.png', 'Dice.png', 'F16.png', 'fingerprint.png',
                    'Flowers1.png', 'Flowers2.png', 'Gardens.png', 'Girl.png', 'Hallway.png', 'hill.png', 'house.png',
                    'Lena.png', 'man.png', 'Man1.png', 'Man2.png', 'montage.png', 'pentagon.png', 'peppers.png',
                    'Plaza.png', 'Statue.png', 'Street1.png', 'Street2.png', 'Traffic.png', 'Trees.png',
                    'Valldemossa.png', 'Yard.png']

    for im_name in im_name_list:
        compare_psnr(im_name)
