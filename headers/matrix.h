#ifndef MPI_AVX_JACOBI_MATRIX_H
#define MPI_AVX_JACOBI_MATRIX_H

#include <memory>
#include <functional>
#include <iostream>

template <typename A>
class Matrix {
    int _height;
    int _width;
    std::unique_ptr<A[]> _buffer;

    [[nodiscard]] int _index(int y, int x) const;

public:
    Matrix();
    Matrix(int height, int width, A default_value);
    Matrix(int height, int width, A values[]);
    Matrix(int height, int width, std::function<A(int, int)> func);

    A get(int y, int x) const;
    void set(int y, int x, A value);

    int height() const;
    int width()  const;

    void print() const;

    static void swap(Matrix<A>* first, Matrix<A>* second);
};

template <typename A>
int Matrix<A>::_index(int y, int x) const {
    return (y * this->_width) + x;
}

template <typename A>
Matrix<A>::Matrix() {
    this->_height = 0;
    this->_width  = 0;
    this->_buffer = std::make_unique<A[]>(0);
}

template <typename A>
Matrix<A>::Matrix(int height, int width, A default_value)  {
    const int size = height * width;

    this->_height = height;
    this->_width  = width;
    this->_buffer = std::make_unique<A[]>(size);

    for (int i = 0; i < size; i += 1) {
        this->_buffer[i] = default_value;
    }
}

template <typename A>
Matrix<A>::Matrix(int height, int width, A values[]) {
    const int size = height * width;

    this->_height = height;
    this->_width  = width;
    this->_buffer = std::make_unique<A[]>(size);

    for (int i = 0; i < size; i += 1) {
        this->_buffer[i] = values[i];
    }
}

template <typename A>
Matrix<A>::Matrix(int height, int width, std::function<A(int, int)> func) {
    const int size = height * width;

    this->_height = height;
    this->_width  = width;
    this->_buffer = std::make_unique<A[]>(size);

    for (int y = 0; y < height; y += 1) {
        for (int x = 0; x < width; x += 1) {
            const int index = this->_index(y, x);
            this->_buffer[index] = func(y, x);
        }
    }
}

template <typename A>
A Matrix<A>::get(int y, int x) const {
    assert(y < this->_height);
    assert(x < this->_width);

    const int index = this->_index(y, x);
    return this->_buffer[index];
}

template <typename A>
void Matrix<A>::set(int y, int x, A value) {
    assert(y < this->_height);
    assert(x < this->_width);

    const int index = this->_index(y, x);
    this->_buffer[index] = value;
}

template <typename A>
int Matrix<A>::height() const {
    return this->_height;
}

template <typename A>
int Matrix<A>::width() const {
    return this->_width;
}

template <typename A>
void Matrix<A>::print() const {
    for (int y = 0; y < this->height(); y += 1) {
        std::cout << "| ";

        for (int x = 0; x < this->width(); x += 1) {
            std::cout << this->get(y, x);

            if (x != this->width() - 1) {
                std::cout << ", ";
            }
        }

        std::cout << " |" << std::endl;
    }
}

template <typename A>
void Matrix<A>::swap(Matrix<A>* first, Matrix<A>* second) {
    const int first_height = first->_height;
    const int first_width  = first->_width;

    first->_height = second->_height;
    first->_width  = second->_width;

    second->_height = first_height;
    second->_width  = first_width;

    first->_buffer.swap(second->_buffer);
}

#endif //MPI_AVX_JACOBI_MATRIX_H
