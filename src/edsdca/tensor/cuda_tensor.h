#ifndef __CUDA_TENSOR_H
#define __CUDA_TENSOR_H

namespace edsdca {
namespace tensor {

class Helper {
    unsigned long translate_to_col_idx(unsigned long m, unsigned long n, unsigned long n_rows) {
        return n * n_rows + m;
    }

    usigned long translate_to_row_idx(unsigned long m, unsigned long n, unsigned long n_cols) {
        return m * n_cols + n;
    }
};

} // tensor namespace
} // edsdca namespace

#endif // __CUDA_TENSOR_H
