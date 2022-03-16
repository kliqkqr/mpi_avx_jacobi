#ifndef MPI_AVX_JACOBI_UTIL_H
#define MPI_AVX_JACOBI_UTIL_H

#include <utility>
#include <cmath>
#include <cassert>

std::pair<int, int> distribute_even(int height, int rank, int size);

size_t ceil_to_multiple(size_t num, size_t factor);
size_t ceil_to_power_of_2(size_t num, size_t factor);

#endif //MPI_AVX_JACOBI_UTIL_H
