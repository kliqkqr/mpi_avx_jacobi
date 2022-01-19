#ifndef MPI_AVX_JACOBI_MATRIX_H
#define MPI_AVX_JACOBI_MATRIX_H

#include <memory>
#include <functional>
#include <iostream>
#include <cassert>
#include <string>

template <typename A> class Vector;

template <typename A>
class Matrix {
    int _height;
    int _width;
    std::unique_ptr<A[]> _buffer;

    [[nodiscard]] int _index(int y, int x) const;

public:
    Matrix();
    Matrix(int height, int width, A default_value);
    Matrix(int height, int width, A* values);
    Matrix(int height, int width, std::function<A(int, int)> func);

    A get(int y, int x) const;
    void set(int y, int x, A value);
    void set_row(int y, A* values);

    void set_dimensions(int height, int width);

    std::unique_ptr<A[]>& get_buffer();

    int height() const;
    int width()  const;

    [[nodiscard]] Matrix<double> to_double() const;
    [[nodiscard]] A* copy_buffer() const;

    std::string to_display() const;
    void print() const;

    static void swap(Matrix<A>* first, Matrix<A>* second);

    Vector<A> operator*(const Vector<A>& vector) const;
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
Matrix<A>::Matrix(int height, int width, A* values) {
    this->_height = height;
    this->_width  = width;
    this->_buffer = std::unique_ptr<A[]>(values);
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
void Matrix<A>::set_row(int y, A* values) {
    assert(y < this->height());

    for (int x = 0; x < this->width(); x += 1) {
        this->set(y, x, values[x]);
    }
}

template <typename A>
void Matrix<A>::set_dimensions(int height, int width) {
    this->_height = height;
    this->_width  = width;
}

template <typename A>
std::unique_ptr<A[]>& Matrix<A>::get_buffer() {
    return this->_buffer;
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
Matrix<double> Matrix<A>::to_double() const {
    const int size = this->height() * this->width();
    double* buffer = (double*)std::malloc(sizeof(double) * size);

    for (int i = 0; i < size; i += 1) {
        buffer[i] = (double)this->_buffer[i];
    }

    return Matrix<double>(this->height(), this->width(), buffer);
}

template <typename A>
A* Matrix<A>::copy_buffer() const {
    const int size = this->height() * this->width();

    A* buffer = (A*)std::malloc(sizeof(A) * size);
    assert(buffer != nullptr);

    for (int i = 0; i < size; i += 1) {
        buffer[i] = this->_buffer[i];
    }

    return buffer;
}

template <typename A>
std::string Matrix<A>::to_display() const {
    int max_length = 0;
    for (int y = 0; y < this->height(); y += 1) {

        for (int x = 0; x < this->width(); x += 1) {
            A value = this->get(y, x);
            std::string string = std::to_string(value);

            max_length = std::max(max_length, (int) string.length());
        }
    }

    std::string display = std::string("");
    for (int y = 0; y < this->height(); y += 1) {
        display = display.append("| ");

        for (int x = 0; x < this->width(); x += 1) {
            std::string string = std::to_string(this->get(y, x));
            string = string.insert(0, max_length - string.length(), ' ');

            display = display.append(string);

            if (x != this->width() - 1) {
                display = display.append(" ");
            }
        }

        display = display.append(" |\n");
    }

    return display;
}

template <typename A>
void Matrix<A>::print() const {
    std::cout << this->to_display() << std::flush;
}

template <typename A>
void Matrix<A>::swap(Matrix<A>* first, Matrix<A>* second) {
    const int first_height = first->height();
    const int first_width  = first->width();

    first->_height = second->height();
    first->_width  = second->width();

    second->_height = first_height;
    second->_width  = first_width;

    first->_buffer.swap(second->_buffer);
}

template <typename A>
Vector<A> Matrix<A>::operator*(const Vector<A>& vector) const {
    assert(this->width() == vector.height());

    Vector<A> result = Vector(this->width(), (A)0);

    for (int y = 0; y < this->height(); y += 1) {
        int sum = 0;
        for (int x = 0; x < this->width(); x += 1) {
            sum += this->get(y, x) * vector.get(x);
        }

        result.set(y, sum);
    }

    return result;
}

#endif //MPI_AVX_JACOBI_MATRIX_H
