#ifndef MPI_AVX_JACOBI_VECTOR_H
#define MPI_AVX_JACOBI_VECTOR_H

#include <memory>
#include <functional>
#include <iostream>
#include <cassert>
#include <cmath>

template <typename A> class Matrix;

template <typename A>
class Vector {
    int _height;
    std::unique_ptr<A[]> _buffer;

public:
    Vector();
    Vector(int height, A default_value);
    Vector(int height, A* values);
    Vector(int height, std::function<A(int)> func);

    A get(int y) const;
    void set(int y, A value);

    std::unique_ptr<A[]>& get_buffer();

    int height() const;
    int width()  const;
    double length() const;
    double distance(const Vector<A>& vector) const;

    [[nodiscard]] Vector<double> to_double() const;
    [[nodiscard]] Vector<float> to_float() const;
    Matrix<A> to_matrix() const;
    Vector<A> to_slice(int y, int height) const;

    A* copy_buffer() const;

    std::string to_display() const;
    void print() const;

    static void swap(Vector<A>* first, Vector<A>* second);

    Vector<A> operator-(const Vector<A>& vector) const;
};

template <typename A>
Vector<A>::Vector() {
    this->_height = 0;
    this->_buffer = std::make_unique<A[]>(0);
}

template <typename A>
Vector<A>::Vector(int height, A default_value) {
    this->_height = height;
    this->_buffer = std::make_unique<A[]>(height);

    for (int i = 0; i < height; i += 1) {
        this->_buffer[i] = default_value;
    }
}

template <typename A>
Vector<A>::Vector(int height, A *values) {
    this->_height = height;
    this->_buffer = std::unique_ptr<A[]>(values);
}

template <typename A>
Vector<A>::Vector(int height, std::function<A(int)> func) {
    this->_height = height;
    this->_buffer = std::make_unique<A[]>(height);

    for (int i = 0; i < height; i += 1) {
        this->_buffer[i] = func(i);
    }
}

template <typename A>
A Vector<A>::get(int y) const {
    assert(y < this->_height);

    return this->_buffer[y];
}

template <typename A>
void Vector<A>::set(int y, A value) {
    assert(y < this->_height);

    this->_buffer[y] = value;
}

template <typename A>
std::unique_ptr<A[]>& Vector<A>::get_buffer() {
    return this->_buffer;
}

template <typename A>
int Vector<A>::height() const {
    return this->_height;
}

template <typename A>
int Vector<A>::width() const {
    return 1;
}

template <typename A>
double Vector<A>::length() const {
    A sum;

    for (int y = 0; y < this->height(); y += 1) {
        A value = this->get(y);
        sum += value * value;
    }

    return std::sqrt((double)sum);
}

template <typename A>
double Vector<A>::distance(const Vector<A>& vector) const {
    return (*this - vector).length();
}

template <typename A>
Vector<double> Vector<A>::to_double() const {
    double* buffer = (double*)std::malloc(sizeof(double) * this->height());

    assert(buffer != nullptr);

    for (int y = 0; y < this->height(); y += 1) {
        buffer[y] = (double)this->get(y);
    }

    return Vector<double>(this->height(), buffer);
}

template <typename A>
Vector<float> Vector<A>::to_float() const {
    float* buffer = (float*)std::malloc(sizeof(float) * this->height());

    assert(buffer != nullptr);

    for (int y = 0; y < this->height(); y += 1) {
        buffer[y] = (float)this->get(y);
    }

    return Vector<float>(this->height(), buffer);
}

template <typename A>
Matrix<A> Vector<A>::to_matrix() const {
    A* buffer = (A*)std::malloc(sizeof(A) * this->height());

    assert(buffer != nullptr);

    for (int y = 0; y < this->height(); y += 1) {
        buffer[y] = this->get(y);
    }

    return Matrix<A>(this->height(), 1, buffer);
}

template <typename A>
Vector<A> Vector<A>::to_slice(int y, int height) const {
    assert(y + height <= this->height());
    assert(y >= 0 && height >= 0);

    A* buffer = (A*)std::malloc(sizeof(A) * height);
    assert(buffer != nullptr);

    for (int i = 0; i < height; i += 1) {
        buffer[i] = this->get(i + y);
    }

    return Vector<A>(height, buffer);
}

template <typename A>
A* Vector<A>::copy_buffer() const {
    A* buffer = (A*)std::malloc(sizeof(A) * this->height());

    assert(buffer != nullptr);

    for (int y = 0; y < this->height(); y += 1) {
        buffer[y] = this->get(y);
    }

    return buffer;
}

// TODO: optimize
template <typename A>
std::string Vector<A>::to_display() const {
    return this->to_matrix().to_display();
}

// TODO: optimize
template <typename A>
void Vector<A>::print() const {
    this->to_matrix().print();
}

template <typename A>
void Vector<A>::swap(Vector<A> *first, Vector<A> *second) {
    const int first_height = first->height();

    first->_height  = second->height();
    second->_height = first_height;

    first->_buffer.swap(second->_buffer);
}

template <typename A>
Vector<A> Vector<A>::operator-(const Vector<A> &vector) const {
    assert(this->height() == vector.height());

    Vector<A> result = Vector<A>(this->height(), (A)0);

    for (int y = 0; y < this->height(); y += 1) {
        result.set(y, this->get(y) - vector.get(y));
    }

    return result;
}

#endif //MPI_AVX_JACOBI_VECTOR_H
