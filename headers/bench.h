#ifndef MPI_AVX_JACOBI_BENCH_H
#define MPI_AVX_JACOBI_BENCH_H

#include <functional>
#include <chrono>

int benchmark(const std::function<void()>& func);

#endif //MPI_AVX_JACOBI_BENCH_H
