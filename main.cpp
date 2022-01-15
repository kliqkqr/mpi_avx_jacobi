#include <cassert>
#include <iostream>

#include "matrix.h"

double next_value(const Matrix<double>& matrix, const Matrix<double>& vector, const Matrix<double>& last_result, int index) {
    const double scalar = 1.0 / matrix.get(index, index);
    double sum = 0.0;

    for (int y = 0; y < vector.height(); y += 1) {
        if (y == index) {
            continue;
        }

        sum += matrix.get(index, y) * last_result.get(y, 0);
    }

    return scalar * (vector.get(index, 0) - sum);
}

Matrix<double> jacobi(Matrix<double>& matrix, Matrix<double>& vector, int iterations) {
    assert(matrix.height() == vector.height());
    assert(matrix.width()  == vector.height());
    assert(vector.width()  == 1);

    const int height = matrix.height();
    Matrix<double> last_result = Matrix<double>(height, 1, 1.0);
    Matrix<double> result      = Matrix<double>(height, 1, 1.0);

    for (int i = 0; i < iterations; i += 1) {
        for (int y = 0; y < height; y += 1) {
            const double value = next_value(matrix, vector, last_result, y);
            result.set(y, 0, value);
        }

        Matrix<double>::swap(&last_result, &result);
    }

    return last_result;
}

int main() {
    double a[] = { 10, -1,  2,  0,
                   -1, 11, -1,  3,
                    2, -1, 10, -1,
                    0,  3, -1,  8 };
    double b[] = { 6, 25, -11, 15 };
    Matrix<double> matrix = Matrix(4, 4, a);
    Matrix<double> vector = Matrix(4, 1, b);

    matrix.print();
    vector.print();

    Matrix<double> result = jacobi(matrix, vector, 20);

    result.print();
}
