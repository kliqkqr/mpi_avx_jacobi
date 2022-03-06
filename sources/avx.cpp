#include "matrix.h"
#include "vector.h"
#include <immintrin.h>

float avx_sum_m256(__m256& a) {
    __m128 high = _mm256_extractf128_ps(a, 1);
    __m128 sum  = _mm256_castps256_ps128(a);
    sum  = _mm_add_ps(sum, high);

    high = _mm_movehl_ps(sum, sum);
    sum  = _mm_add_ps(sum, high);

    high = _mm_shuffle_ps(sum, sum, 0x1);
    sum  = _mm_add_ps(sum, high);

    return _mm_cvtss_f32(sum);
}

float avx_dot_product_no_align_m256(float a[], float b[], int size) {
    int max_index = size / 8;
    int rest = size % 8;
    float sum = 0;

    for (int i = 0; i < max_index; i += 1) {
        int offset = i * 8;

//        int _0 = i, _1 = i + 1, _2 = i + 2, _3 = i + 3, _4 = i + 4, _5 = i + 5, _6 = i + 6, _7 = i + 7;
//        __m256 v_a   = _mm256_set_ps(a[_0], a[_1], a[_2], a[_3], a[_4], a[_5], a[_6], a[_7]);
//        __m256 v_b   = _mm256_set_ps(b[_0], b[_1], b[_2], b[_3], b[_4], b[_5], b[_6], b[_7]);
//        __m256 v_mul = _mm256_mul_ps(v_a, v_b);

        __m256 v_a   = _mm256_loadu_ps(&a[offset]);
        __m256 v_b   = _mm256_loadu_ps(&b[offset]);
        __m256 v_mul = _mm256_mul_ps(v_a, v_b);

        sum += avx_sum_m256(v_mul);
    }

    if (rest != 0) {
        // TODO: maybe optimize for rest < 4 in m128

        int offset = max_index * 8;

        float r_a[8];
        float r_b[8];

        for (int i = 0; i < rest; i += 1) {
            r_a[i] = a[offset + i];
            r_b[i] = b[offset + i];
        }

        for (int i = rest; i < 8; i += 1) {
            r_a[i] = 0;
            r_a[i] = 0;
        }

//        int _0 = max_index    , _1 = max_index + 1, _2 = max_index + 2, _3 = max_index + 3,
//            _4 = max_index + 4, _5 = max_index + 5, _6 = max_index + 6, _7 = max_index + 7;

        __m256 v_a   = _mm256_loadu_ps(r_a);
        __m256 v_b   = _mm256_loadu_ps(r_b);
        __m256 v_mul = _mm256_mul_ps(v_a, v_b);

        sum += avx_sum_m256(v_mul);
    }

    return sum;
}


//Vector<float> jacobi_avx_no_align_f(const Matrix<float>& matrix, const Vector<float>& vector, const int iterations) {
//    assert(matrix.height() == vector.height());
//    assert(matrix.width()  == vector.height());
//    assert(vector.width()  == 1);
//
//    auto a = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
//}