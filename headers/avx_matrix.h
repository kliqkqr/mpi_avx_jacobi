#ifndef MPI_AVX_JACOBI_AVX_MATRIX_H
#define MPI_AVX_JACOBI_AVX_MATRIX_H

#include <memory>
#include <functional>
#include <iostream>
#include <cassert>
#include <string>

#include "avx.h"
#include "util.h"

template <typename A>
class Matrix;

template <typename A>
class AvxMatrix {
    int _width;
    int _height;
    A*  _buffer;

    int _align_width;
    int _block_elem_count;
    int _block_align_size;
    int _align_size;

    [[nodiscard]] int _index(int y, int x) const;
    [[nodiscard]] A* _pointer(int y, int x) const;

    void _init_fields(int height, int width, int block_element_count, int align_size);

public:
    AvxMatrix();
    AvxMatrix(int height, int width, A default_value, int block_element_count, int align_size);
    AvxMatrix(int height, int width, A* values, int block_element_count, int align_size);
    AvxMatrix(int height, int width, std::function<A(int, int)> func, int block_element_count, int align_size);
    AvxMatrix(int height, int width, uint8_t* buffer, int align_width, int block_element_count, int block_align_size, int align_size);

    [[nodiscard]] A get(int y, int x) const;

    [[nodiscard]] int height() const;
    [[nodiscard]] int width() const;
    [[nodiscard]] int align_width() const;
    [[nodiscard]] int block_elem_count() const;
    [[nodiscard]] int block_align_size() const;
    [[nodiscard]] int align_size() const;

    [[nodiscard]] Matrix<A> to_matrix(bool remove_align) const;
    [[nodiscard]] uint8_t* to_byte_buffer() const;
    [[nodiscard]] size_t byte_buffer_size() const;

    void print_meta() const;

    void buffer_free();
};

template <typename A>
int AvxMatrix<A>::_index(int y, int x) const {
    return (y * this->_align_width) + x;
}

template <typename A>
A* AvxMatrix<A>::_pointer(int y, int x) const {
    const int index = this->_index(y, x);
    return avx_aligned_blocks_ptr_to_index_raw(this->_buffer, index, this->_block_elem_count, this->_block_align_size);
}

template <typename A>
void AvxMatrix<A>::_init_fields(int height, int width, int block_element_count, int align_size) {
    // allign_size = n^2
    assert(align_size > 0 && (align_size & (align_size - 1)) == 0);

    this->_height = height;
    this->_width  = width;

    this->_block_elem_count = block_element_count;
    this->_align_size       = align_size;

    const int block_size    = block_element_count * sizeof(A);
    this->_block_align_size = ceil_to_multiple(block_size, align_size);
    this->_align_width      = ceil_to_multiple(this->_width, this->_block_elem_count);
}

template  <typename A>
AvxMatrix<A>::AvxMatrix() {
    this->_width  = 0;
    this->_height = 0;
    this->_buffer = nullptr;

    this->_align_width      = 0;
    this->_block_elem_count = 0;
    this->_block_align_size = 0;
    this->_align_size       = 0;
}

template <typename A>
AvxMatrix<A>::AvxMatrix(int height, int width, A default_value, int block_element_count, int align_size) {
    this->_init_fields(height, width, block_element_count, align_size);

    const int elem_count = this->_height * this->_align_width;
    this->_buffer = avx_allocate_aligned_blocks<A>(elem_count, this->_block_elem_count, this->_align_size);

    for (int y = 0; y < this->_height; y += 1) {
        for (int x = 0; x < this->_align_width; x += 1) {
            A* pointer = this->_pointer(y, x);

            if (x < this->_width) {
                *pointer = default_value;
            }
            else {
                *pointer = 0;
            }
        }
    }
}

template <typename A>
AvxMatrix<A>::AvxMatrix(int height, int width, A* values, int block_element_count, int align_size) {
    this->_init_fields(height, width, block_element_count, align_size);

    const int elem_count = this->_height * this->_align_width;
    this->_buffer = avx_allocate_aligned_blocks<A>(elem_count, this->_block_elem_count, this->_align_size);

    for (int y = 0; y < this->_height; y += 1) {
        for (int x = 0; x < this->_align_width; x += 1) {
            A* pointer = this->_pointer(y, x);

            if (x < this->_width) {
                const int index = (y * width) + x;
                *pointer = values[index];
            }
            else {
                *pointer = 0;
            }
        }
    }
}

template <typename A>
AvxMatrix<A>::AvxMatrix(int height, int width, std::function<A(int, int)> func, int block_element_count, int align_size) {
    this->_init_fields(height, width, block_element_count, align_size);

    const int elem_count = this->_height * this->_align_width;
    this->_buffer = avx_allocate_aligned_blocks<A>(elem_count, this->_block_elem_count, this->_align_size);

    for (int y = 0; y < this->_height; y += 1) {
        for (int x = 0; x < this->_align_width; x += 1) {
            A* pointer = this->_pointer(y, x);

            if (x < this->_width) {
                *pointer = func(y, x);
            }
            else {
                *pointer = 0;
            }
        }
    }
}

template <typename A>
AvxMatrix<A>::AvxMatrix(int height, int width, uint8_t* buffer, int align_width, int block_element_count, int block_align_size, int align_size) {
    this->_height           = height;
    this->_width            = width;
    this->_buffer           = (A*)buffer;
    this->_align_width      = align_width;
    this->_block_elem_count = block_element_count;
    this->_block_align_size = block_align_size;
    this->_align_size       = align_size;
}

template <typename A>
A AvxMatrix<A>::get(int y, int x) const {
    A* pointer = this->_pointer(y, x);
    return *pointer;
}

template <typename A>
int AvxMatrix<A>::height() const {
    return this->_height;
}

template <typename A>
int AvxMatrix<A>::width() const {
    return this->_width;
}

template <typename A>
int AvxMatrix<A>::align_width() const {
    return this->_align_width;
}

template <typename A>
int AvxMatrix<A>::block_elem_count() const {
    return this->_block_elem_count;
}

template <typename A>
int AvxMatrix<A>::block_align_size() const {
    return this->_block_align_size;
}

template <typename A>
int AvxMatrix<A>::align_size() const {
    return this->_align_size;
}

template <typename A>
Matrix<A> AvxMatrix<A>::to_matrix(bool remove_align) const {
    std::function<A(int, int)> func = [&] (int y, int x) {
        return this->get(y, x);
    };

    const int width = remove_align ? this->_width : this->_align_width;
    return Matrix<A>(this->_height, width, func);
}

template <typename A>
uint8_t* AvxMatrix<A>::to_byte_buffer() const {
    return (uint8_t*)this->_buffer;
}

template <typename A>
size_t AvxMatrix<A>::byte_buffer_size() const {
    const size_t size = this->_block_align_size * (this->_align_width / this->_block_elem_count) * this->_height;
    return size;
}

template <typename A>
void AvxMatrix<A>::buffer_free() {
    _mm_free(this->_buffer);
}

template <typename A>
void AvxMatrix<A>::print_meta() const {
//    int _width;
//    int _height;
//    A*  _buffer;
//
//    int _align_width;
//    int _block_elem_count;
//    int _block_align_size;
//    int _align_size;

    std::cout << "height=" << this->_height << "; width=" << this->_width << "; align_width=" << this->_align_width << ";" << std::endl;
    std::cout << "block_elem_count=" << this->_block_elem_count << "; block_align_size=" << this->_block_align_size << "; align_size=" << this->_align_size << ";" << std::endl;
}

#endif //MPI_AVX_JACOBI_AVX_MATRIX_H
