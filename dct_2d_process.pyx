import numpy as np


cdef void dct_2d_process(
        np.ndarray DCT_table_2D,
        np.ndarray img,
        np.ndarray coef_norm,
        const unsigned nHW,
        const unsigned width,
        const unsigned height,
        const unsigned chnls,
        const unsigned kHW,
        const unsigned i_r,
        const unsigned step,
        const unsigned i_min,
        const unsigned i_max,):
    cdef np.ndarray vec = np.zeros(shape=())
    cdef np.ndarray dct = np.zeros(shape=())

    cdef const unsigned kHW_2 = kHW * kHW
    cdef const unsigned size = chnls * kHW_2 * width * (2 * nHW + 1)

    cdef int c, i, j, p, q
    cdef const unsigned dc, dc_p, ds

    if i_r == i_min or i_r == i_max:
        for c in range(chnls):
            dc = c * width * height
            dc_p = c * kHW_2 * width * (2 * nHW + 1)
            for i in range(2 * nHW + 1):
                for j in range(width - kHW):
                    for p in range(kHW):
                        for q in range(kHW):
                            vec[p * kHW + q + dc_p + (i * width + j) * kHW_2] = \
                                img[dc + (i_r + i - nHW + p) * width + j + q]

        #TODO fftwf

        for c in range(chnls):
            dc = c * kHW_2 * width * (2 * nHW + 1)
            dc_p = c * kHW_2 * width * (2 * nHW + 1)
            for i in range(2 * nHW + 1):
                for j in range(width - kHW):
                    for k in range(kHW_2):
                        DCT_table_2D[dc + (i * width + j) * kHW_2 + k] = \
                                dct[dc_p + (i * width + j) * kHW_2 + k] * coef_norm[k]

    else:
        ds = step * width * kHW_2
        for c in range(chnls):
            dc = c * width * (2 * nHW + 1) * kHW_2
            for i in range(2 * nHW + 1 - step):
                for j in range(width - kHW):
                    for k in range(kHW_2):
                        DCT_table_2D[k + (i * width + j) * kHW_2 + dc] = \
                            DCT_table_2D[k + (i * width + j) * kHW_2 + dc + ds]

        for c in range(chnls):
            dc = c * width * height
            dc_p = c * kHW_2 * width * step
            for i in range(step):
                for j in range(width-kHW):
                    for p in range(kHW):
                        for q in range(kHW):
                            vec[p * kHW + q + dc_p + (i * width + j) * kHW_2] = \
                                        img[(p + i + 2 * nHW + 1 - step + i_r - nHW)* width + j + q + dc]

        #TODO fftwf

        for c in range(chnls):
            dc = c * kHW_2 * width * (2 * nHW + 1)
            dc_p = c * kHW_2 * width * step
            for i in range(step):
                for j in range(width-kHW):
                    for k in range(kHW_2):
                        DCT_table_2D[dc + ((i + 2 * nHW + 1 - step) * width + j) * kHW_2 + k] = \
                            dct[dc_p + (i * width + j) * kHW_2 + k] * coef_norm[k]
