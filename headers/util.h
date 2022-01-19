#ifndef MPI_AVX_JACOBI_UTIL_H
#define MPI_AVX_JACOBI_UTIL_H

#include <utility>
#include <cmath>

std::pair<int, int> distribute_even(int height, int rank, int size);

#endif //MPI_AVX_JACOBI_UTIL_H
