#ifndef __CUDA_MATRIX_H
#define __CUDA_MATRIX_H

#include <iostream>
#include "stdlib.h"

#include "cuda_tensor.h"

namespace edsdca {
namespace tensor {

/**
 * Wrapper for cuBLAS data structures. This class stores matrices in
 * column-major order although the interface is row-major. In other words, use
 * the matrices as you would any standard C-style matrix, but know they are
 * actually implemented in column-major.
 */
template <typename Dtype>
class CudaMatrix2d {

    public:
        const unsigned long m;
        const unsigned long n;
        const bool row_major;

        Matrix2d(unsigned long m, unsigned long n, bool row_major) : 
            m(m), n(n), row_major(row_major) {
            this->elements_ = static_cast<Dtype*>(malloc(sizeof(Dtype)*m*n));
        }

        /**
         * Default initialization of a @p Matrix2d object is column-major
         */
        Matrix2d(unsigned long m, unsigned long n) : Matrix2d(m, n, false) {}

        ~Matrix2d() {
            free (this->elements_);
        }

        const Dtype operator()(long m, long n) const {
            return this->elements_[translate_to_col_idx(m, n, this->n)];
        }

        void set(const unsigned long m, const unsigned long n, const Dtype val) {
            this->elements_[translate_to_col_idx(m, n, this->n)] = val;
        }

        const Dtype get(const unsigned long m, const unsigned long n) const {
            return this->elements_[translate_to_col_idx(m, n, this->n)];
        }


    private:
        Dtype* elements_;
};

#endif //__CUDA_MATRIX_H
