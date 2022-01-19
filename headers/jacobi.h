#ifndef MPI_AVX_JACOBI_JACOBI_H
#define MPI_AVX_JACOBI_JACOBI_H

#include "matrix.h"
#include "vector.h"
#include "mpi.h"

double next_value(const Matrix<double>& matrix, const Vector<double>& vector, const Vector<double>& last_result, int index);

Vector<double> jacobi(const Matrix<double>& matrix, const Vector<double>& vector, int iterations);
Vector<double> jacobi_mpi_ring(const Matrix<double>& matrix, const Vector<double>& vector, const int iterations, const MPI& mpi, const int offset, const int count);

#endif //MPI_AVX_JACOBI_JACOBI_H
