

def bior_2d_inverse(group_3D_table, kHW, lpr, hpr):
    kHW_2 = kHW * kHW
    N = group_3D_table.size() / kHW_2

    for n in range(N):
        bior_2d_inverse_(group_3D_table, kHW, n * kHW_2, lpr, hpr)


cdef void bior_2d_inverse_(
    vector<float> &signal
,   const unsigned N
,   const unsigned d_s
,   vector<float> const& lpr
,   vector<float> const& hpr
){
    //! Initialization
    const unsigned iter_max = log2(N);
    unsigned N_1 = 2;
    unsigned N_2 = 1;
    const unsigned S_1 = lpr.size();
    const unsigned S_2 = S_1 / 2 - 1;

    for (unsigned iter = 0; iter < iter_max; iter++)
    {

        vector<float> tmp(N_1 + S_2 * N_1);
        vector<unsigned> ind_per(N_1 + 2 * S_2 * N_2);
        per_ext_ind(ind_per, N_1, S_2 * N_2);

        //! Implementing column filtering
        for (unsigned j = 0; j < N_1; j++)
        {
            //! Periodic extension of the signal in column
            for (unsigned i = 0; i < tmp.size(); i++)
                tmp[i] = signal[d_s + j + ind_per[i] * N];

            //! Low and High frequencies filtering
            for (unsigned i = 0; i < N_2; i++)
            {
                float v_l = 0.0f, v_h = 0.0f;
                for (unsigned k = 0; k < S_1; k++)
                {
                    v_l += lpr[k] * tmp[k * N_2 + i];
                    v_h += hpr[k] * tmp[k * N_2 + i];
                }

                signal[d_s + i * 2 * N + j] = v_h;
                signal[d_s + (i * 2 + 1) * N + j] = v_l;
            }
        }

        //! Implementing row filtering
        for (unsigned i = 0; i < N_1; i++)
        {
            //! Periodic extension of the signal in row
            for (unsigned j = 0; j < tmp.size(); j++)
                tmp[j] = signal[d_s + i * N + ind_per[j]];

            //! Low and High frequencies filtering
            for (unsigned j = 0; j < N_2; j++)
            {
                float v_l = 0.0f, v_h = 0.0f;
                for (unsigned k = 0; k < S_1; k++)
                {
                    v_l += lpr[k] * tmp[k * N_2 + j];
                    v_h += hpr[k] * tmp[k * N_2 + j];
                }

                signal[d_s + i * N + j * 2] = v_h;
                signal[d_s + i * N + j * 2 + 1] = v_l;
            }
        }

        //! Sizes update
        N_1 *= 2;
        N_2 *= 2;
    }
}

