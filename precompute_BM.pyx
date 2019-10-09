import numpy as np

cdef np.ndarray precompute_BM(
        np.ndarray img,
        const unsigned width,
        const unsigned height,
        const unsigned kHW,
        const unsigned NHW,
        const unsigned nHW,
        const unsigned pHW,
        const float    tauMatch):
    cdef const unsigned Ns = 2 * nHW + 1
    cdef const float threshold = tauMatch * kHW * kHW
    cdef np.ndarray diff_table = np.zeros(width * height, dtype=np.int)
    cdef np.ndarray sum_table = np.ones(((nHW + 1) * Ns, width * height), dtype=np.int) * 2 * threshold
    cdef row_ind = ind_initialize(height - kHW + 1, nHW, pHW)
    cdef column_ind = ind_initialize(width - kHW + 1, nHW, pHW)

    cdef int di
    for di in range(nHW + 1):
        cdef int dj
        for dj in range(Ns):
            cdef const int dk = int(di * width + dj) - int(nHW)
            cdef const unsigned ddk = di * Ns + dj

            cdef int i
            for i in range(height - nHW):
                cdef unsigned k = i * width + nHW

                cdef int j
                for j in range(nHW, width - nHW):
                    k += 1
                    diff_table[k] = (img[k + dk] - img[k]) * (img[k + dk] - img[k])
            cdef const unsigned dn = nHW * width + nHW
            cdef float value = 0.0

            cdef int p
            for p in range(kHW):
                cdef unsigned pq = p * width + dn
                cdef int q
                for q in range(kHW):
                    pq += 1
                    value += diff_table[pq]
            sum_table[ddk][dn] = value

            cdef int j
            for j in range(nHW + 1, width - nHW):
                cdef const unsigned ind = nHW * width + j - 1
                cdef float sum = sum_table[ddk][ind]
                cdef int p
                for p in range(kHW):
                    sum += diff_table[ind + p * width + kHW] - diff_table[ind + p * width]
                sum_table[ddk][ind + 1] = sum

            cdef int i
            for i in range(nHW + 1, height - nHW):
                cdef const unsigned ind = (i - 1) * width + nHW
                cdef float sum = sum_table[ddk][ind]
                cdef int q
                for q in range(kHW):
                    sum += diff_table[ind + kHW * width + q] - diff_table[ind + q]
                sum_table[ddk][ind + width] = sum

                cdef unsigned k = i * width + nHW + 1
                cdef unsigned pq = (i + kHW - 1) * width + kHW - 1 + nHW + 1
                cdef int j
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
    cdef int ind_i
    for ind_i in row_ind:
        cdef int ind_j
        for ind_j in column_ind:
            cdef const unsigned k_r = ind_i * width + ind_j

            for dj in range(-nHW, nHW + 1):
                for di in range(nHW + 1):
                    if sum_table[dj + nHW + di * Ns][k_r] < threshold:
                        cdef np.ndarray pair = np.array([[sum_table[dj + nHW + di * Ns][k_r], k_r + di * width + dj]],
                                                        dtype=np.int)
                        table_distance = np.append(table_distance, pair, axis=0)

                for di in range(-nHW, 0):
                    if sum_table[-dj + nHW + (-di) * Ns][k_r] < threshold:
                        cdef np.ndarray pair = np.array(
                            [[sum_table[-dj + nHW + (-di) * Ns][k_r + di * width + dj], k_r + di * width + dj]],
                            dtype=np.int)
                        table_distance = np.append(table_distance, pair, axis=0)

            cdef const unsigned nSx_r = closest_power_of_2(len(table_distance) * 2) if NHW > len(
                table_distance) * 2 else NHW

            if nSx_r == 1 and len(table_distance) * 2 == 0:
                print('problem size')
                cdef np.ndarray pair = np.array([[0, k_r]], dtype=np.int)
                table_distance = np.append(table_distance, pair, axis=0)

            #partial_sort(table_distance.begin(), table_distance.begin() + nSx_r,
            #                               table_distance.end(), ComparaisonFirst);
            table_distance.sort(key=lambda x: x[0])

            try:
                patch_table
            except NameError:
                cdef np.ndarray patch_table = np.zeros((width * height, nSx_r), dtype=np.int)

            cdef int n
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
