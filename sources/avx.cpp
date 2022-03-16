#include <immintrin.h>

#include "matrix.h"
#include "vector.h"

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

        __m256 v_a   = _mm256_loadu_ps(r_a);
        __m256 v_b   = _mm256_loadu_ps(r_b);
        __m256 v_mul = _mm256_mul_ps(v_a, v_b);

        sum += avx_sum_m256(v_mul);
    }

    return sum;
}

float avx_dot_product_align_m256(float a[], float b[], int size) {
    int max_index = size / 8;
    int rest = size % 8;
    float sum = 0;

    for (int i = 0; i < max_index; i += 1) {
        int offset = i * 8;

        __m256 v_a   = _mm256_load_ps(&a[offset]);
        __m256 v_b   = _mm256_load_ps(&b[offset]);
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

        __m256 v_a   = _mm256_load_ps(r_a);
        __m256 v_b   = _mm256_load_ps(r_b);
        __m256 v_mul = _mm256_mul_ps(v_a, v_b);

        sum += avx_sum_m256(v_mul);
    }

    return sum;
}

