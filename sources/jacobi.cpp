#include "jacobi.h"

#include <thread>
#include <chrono>

double next_value(const Matrix<double>& matrix, const Vector<double>& vector, const Vector<double>& last_result, int index) {
    const double scalar = 1.0 / matrix.get(index, index);
    double sum = 0.0;

    for (int y = 0; y < vector.height(); y += 1) {
        if (y == index) {
            continue;
        }

        sum += matrix.get(index, y) * last_result.get(y);
    }

    return scalar * (vector.get(index) - sum);
}

Vector<double> jacobi(const Matrix<double>& matrix, const Vector<double>& vector, const int iterations) {
    assert(matrix.height() == vector.height());
    assert(matrix.width()  == vector.height());
    assert(vector.width()  == 1);

    const int height = matrix.height();
    Vector<double> last_result = Vector<double>(height, 1.0);
    Vector<double> result      = Vector<double>(height, 1.0);

    for (int i = 0; i < iterations; i += 1) {
        for (int y = 0; y < height; y += 1) {
            const double value = next_value(matrix, vector, last_result, y);
            result.set(y, value);
        }

        Vector<double>::swap(&last_result, &result);
    }

    return last_result;
}

Vector<double> jacobi_mpi_ring(const Matrix<double>& matrix, const Vector<double>& vector, const int iterations, const MPI& mpi, const int offset, const int count) {
    assert(matrix.height() == vector.height());
    assert(matrix.width()  == vector.height());
    assert(vector.width()  == 1);

    const int height = matrix.height();
    Vector<double> last_result = Vector<double>(height, 1.0);
    Vector<double> result      = Vector<double>(height, 1.0);

    for (int i = 0; i < iterations; i += 1) {
        for (int y = offset; y < offset + count; y += 1) {
            const double value = next_value(matrix, vector, last_result, y);
            result.set(y, value);
        }

        Vector<double>::swap(&last_result, &result);
        mpi.fold_vector_ring(&last_result, offset, count);
    }

    return last_result;
}