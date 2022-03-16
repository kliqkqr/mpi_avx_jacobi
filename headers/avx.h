#ifndef MPI_AVX_JACOBI_AVX_H
#define MPI_AVX_JACOBI_AVX_H

#include <iostream>

#include "util.h"

float avx_dot_product_no_align_m256(float a[], float b[], int size);

template <typename A>
A* avx_allocate_aligned_blocks(const size_t elem_count, const size_t block_elem_count, const size_t align_size) {
    const size_t elem_size        = sizeof(A);
    const size_t block_size       = block_elem_count * elem_size;
//    const size_t block_size_align = block_size % align_size == 0 ? block_size : align_size * ((block_size / align_size) + 1);
    const size_t block_size_align = ceil_to_multiple(block_size, align_size);
    const size_t block_full_count = elem_count / block_elem_count;
    const size_t block_rest       = elem_count % block_elem_count;
    const size_t size             = block_size_align * block_full_count + block_rest * elem_size;

    return (A*)_mm_malloc(size, align_size);
}

template <typename A>
A* avx_allocate_aligned_blocks_clear(const size_t elem_count, const size_t block_elem_count, const size_t align_size, const uint8_t clear) {
    const size_t elem_size        = sizeof(A);
    const size_t block_size       = block_elem_count * elem_size;
//    const size_t block_size_align = block_size % align_size == 0 ? block_size : align_size * ((block_size / align_size) + 1);
    const size_t block_size_align = ceil_to_multiple(block_size, align_size);
    const size_t block_full_count = elem_count / block_elem_count;
    const size_t block_rest       = elem_count % block_elem_count;
    const size_t size             = block_size_align * block_full_count + block_rest * elem_size;

    uint8_t* buffer = (uint8_t*)_mm_malloc(size, align_size);
    for (int i = 0; i < size; i += 1) {
        buffer[i] = clear;
    }

    return (A*)buffer;
}

template <typename A>
A* avx_aligned_blocks_ptr_to_index(const A* const buffer, const size_t elem_index, const size_t block_elem_count, const size_t align_size) {
    const size_t elem_size        = sizeof(A);
    const size_t block_size       = block_elem_count * elem_size;
//    const size_t block_size_align = block_size % align_size == 0 ? block_size : align_size * ((block_size / align_size) + 1);
    const size_t block_size_align = ceil_to_multiple(block_size, align_size);
    const size_t block_index      = elem_index / block_elem_count;
    const size_t block_elem_index = elem_index % block_elem_count;
    const size_t byte_index       = block_index * block_size_align + block_elem_index * elem_size;

    const uint8_t* byte_buffer = (uint8_t*)buffer;
    return (A*)(&byte_buffer[byte_index]);
}

template <typename A>
A* avx_aligned_blocks_ptr_to_index_raw(const A* const buffer, const size_t elem_index, const size_t block_elem_count, const size_t block_size_align) {
    const size_t elem_size  = sizeof(A);
    const size_t block_index  = elem_index / block_elem_count;
    const size_t block_elem_index = elem_index % block_elem_count;
    const size_t byte_index = block_index * block_size_align + block_elem_index * elem_size;

    const uint8_t* byte_buffer = (uint8_t*)buffer;
    return (A*)(&byte_buffer[byte_index]);
}

#endif //MPI_AVX_JACOBI_AVX_H
