import numpy as np
cimport numpy as np

def precompute_BM(
        np.ndarray img,
        const unsigned width,
        const unsigned height,
        const unsigned kHW,
        const unsigned NHW,
        const unsigned nHW,
        const unsigned pHW,
        const float    tauMatch):
    cdef unsigned Ns = 2 * nHW + 1
    cdef float threshold = tauMatch * kHW * kHW
    cdef np.ndarray diff_table = np.zeros(width * height, dtype=np.int)
    cdef np.ndarray sum_table = np.ones(((nHW + 1) * Ns, width * height), dtype=np.int) * 2 * threshold
    cdef row_ind = ind_initialize(height - kHW + 1, nHW, pHW)
    cdef column_ind = ind_initialize(width - kHW + 1, nHW, pHW)

    cdef int di, dj, dk, i, j, p, q
    cdef unsigned ddk, k, dn, pq, ind
    cdef float value, sum
    # for di in range(nHW + 1):
    for di in range(nHW):
        for dj in range(Ns):
            dk = int(di * width + dj) - int(nHW)
            ddk = di * Ns + dj
            for i in range(nHW, height - nHW):
                k = i * width + nHW
                for j in range(nHW, width - nHW):
                    k += 1
                    diff_table[k] = (img[k + dk] - img[k]) * (img[k + dk] - img[k])
            dn = nHW * width + nHW
            value = 0.0
            for p in range(kHW):
                pq = p * width + dn
                for q in range(kHW):
                    pq += 1
                    value += diff_table[pq]
            sum_table[ddk][dn] = value

            for j in range(nHW + 1, width - nHW):
                ind = nHW * width + j - 1
                sum = sum_table[ddk][ind]
                for p in range(kHW):
                    sum += diff_table[ind + p * width + kHW] - diff_table[ind + p * width]
                sum_table[ddk][ind + 1] = sum

            for i in range(nHW + 1, height - nHW):
                ind = (i - 1) * width + nHW
                sum = sum_table[ddk][ind]
                for q in range(kHW):
                    sum += diff_table[ind + kHW * width + q] - diff_table[ind + q]
                sum_table[ddk][ind + width] = sum

                k = i * width + nHW + 1
                pq = (i + kHW - 1) * width + kHW - 1 + nHW + 1
                for j in range(nHW + 1, width - nHW):
                    k += 1
                    pq += 1
                    sum_table[ddk][k] = \
                        sum_table[ddk][k - 1] \
                        + sum_table[ddk][k - width] \
                        - sum_table[ddk][k - 1 - width] \
                        + diff_table[pq] \
                        - diff_table[pq - kHW] \
                        - diff_table[pq - kHW * width] \
                        + diff_table[pq - kHW - kHW * width]

    cdef np.ndarray table_distance = np.empty(shape=[0, 2], dtype=np.int)
    cdef np.ndarray pair
    cdef np.ndarray patch_table
    cdef int ind_i, ind_j, n
    cdef unsigned k_r, nSx_r
    for ind_i in row_ind:
        for ind_j in column_ind:
            k_r = ind_i * width + ind_j

            for dj in range(-nHW, nHW + 1):
                for di in range(nHW + 1):
                    if sum_table[dj + nHW + di * Ns][k_r] < threshold:
                        pair = np.array([[sum_table[dj + nHW + di * Ns][k_r], k_r + di * width + dj]], dtype=np.int)
                        table_distance = np.append(table_distance, pair, axis=0)

                for di in range(-nHW, 0):
                    if sum_table[-dj + nHW + (-di) * Ns][k_r] < threshold:
                        pair = np.array(
                            [[sum_table[-dj + nHW + (-di) * Ns][k_r + di * width + dj], k_r + di * width + dj]],
                            dtype=np.int)
                        table_distance = np.append(table_distance, pair, axis=0)

            nSx_r = closest_power_of_2(len(table_distance) * 2) if NHW > len(
                table_distance) * 2 else NHW

            if nSx_r == 1 and len(table_distance) * 2 == 0:
                print('problem size')
                pair = np.array([[0, k_r]], dtype=np.int)
                table_distance = np.append(table_distance, pair, axis=0)

            #partial_sort(table_distance.begin(), table_distance.begin() + nSx_r,
            #                               table_distance.end(), ComparaisonFirst);
            sorted(table_distance, key=lambda x: x[0],)

            try:
                patch_table[0][0]
            except NameError:
                patch_table = np.zeros((width * height, nSx_r), dtype=np.int)

            for n in range(nSx_r):
                patch_table[k_r][n] = table_distance[n][1]

            if nSx_r == 1:
                patch_table[k_r][0] = table_distance[0][1]

    return patch_table

cdef int closest_power_of_2(const unsigned n):
    cdef unsigned r = 1
    while (r * 2 <= n):
        r *= 2
    return r

cdef np.ndarray ind_initialize(
        const unsigned max_size,
        const unsigned N,
        const unsigned step):
    cdef np.ndarray ind_set = np.empty(shape=[0], dtype=np.int)
    cdef unsigned ind = N
    while (ind < max_size - N):
        ind_set = np.append(ind_set, np.array([ind]), axis=0)
        ind += step
    if ind_set[-1] < max_size - N - 1:
        ind_set = np.append(ind_set, np.array([max_size - N - 1]), axis=0)
    return ind_set
