import math
import pywt
import numpy as np


#  * @brief Compute a full 2D Bior 1.5 spline wavelet (normalized)
#  *
#  * @param input: vector on which the transform will be applied;
#  * @param output: will contain the result;
#  * @param N: size of the 2D patch (N x N) on which the 2D transform
#  *           is applied. Must be a power of 2;
#  * @param d_i: for convenience. Shift for input to access to the patch;
#  * @param r_i: for convenience. input(i, j) = input[d_i + i * r_i + j];
#  * @param d_o: for convenience. Shift for output;
#  * @param lpd: low frequencies coefficients for the forward Bior 1.5;
#  * @param hpd: high frequencies coefficients for the forward Bior 1.5.
#  *
#  * @return none.
#  **/
# void bior_2d_forward(
#     vector<float> const& input
# ,   vector<float> &output
# ,   const unsigned N
# ,   const unsigned d_i
# ,   const unsigned r_i
# ,   const unsigned d_o
# ,   vector<float> const& lpd
# ,   vector<float> const& hpd
# ){
#     //! Initializing output
#     for (unsigned i = 0; i < N; i++)
#         for (unsigned j = 0; j < N; j++)
#             output[i * N + j + d_o] = input[i * r_i + j + d_i];

# table_2D_img((2 * nWien + 1) * width * chnls * kWien_2, 0.0f)



def bior_2d_forward(img):
    assert img.shape[0] == img.shape[1]
    N = img.shape[0]
    iter_max = int(math.log2(N))

    for iter in range(iter_max):
        coeffs2 = pywt.dwt2(img[:N, :N], 'bior1.5', mode='periodic')
        LL, (LH, HL, HH) = coeffs2
        img[:N//2, :N//2] = LL[2: -2, 2: -2]
        img[N//2:N, N//2:N] = HH[2: -2, 2: -2]
        img[:N//2, N//2:N] = -HL[2: -2, 2: -2]
        img[N//2:N, :N//2] = -LH[2: -2, 2: -2]
        N //= 2
    return img


def bior_2d_reverse(bior_img):
    pass # TODO