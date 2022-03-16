#include <cassert>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <utility>
#include <immintrin.h>

#include "avx.h"
#include "vector.h"
#include "matrix.h"
#include "avx_matrix.h"
#include "generator.h"
#include "jacobi.h"
#include "mpi.h"
#include "util.h"
#include "bench.h"

const int SIZE  = 8;
const int RANGE = 100;
const int ITERATIONS = 50;

int old_main(int argc, char** argv) {
    MPI mpi = MPI(&argc, &argv);

    Matrix<double> matrix;
    Vector<double> vector, result, solution;

    if (mpi.rank() == 0) {
        Generator generator = Generator();

        matrix = generator.random_diag_dom_int_matrix(SIZE, -RANGE, RANGE).to_double();
        vector = generator.random_int_vector(SIZE, -RANGE, RANGE).to_double();
        result = matrix * vector;
    }

    mpi.sync_matrix(&matrix, 0);
    mpi.sync_vector(&vector, 0);
    mpi.sync_vector(&result, 0);

    auto offset_count = distribute_even(vector.height(), mpi.rank(), mpi.size());
    const int offset = offset_count.first;
    const int count  = offset_count.second;

    auto func = [&] () {
        solution = jacobi_mpi_allgatherv(matrix, result, ITERATIONS, mpi, offset, count);
    };

    int duration = benchmark(func);

    if (mpi.rank() == 0) {
        std::cout << "duration: " << duration << std::endl;

        auto distance = vector.distance(solution);

        std::cout << "distance: " << distance << std::endl;

        std::cout << "matrix" << std::endl;
        matrix.print();

        std::cout << "vector" << std::endl;
        vector.print();

        std::cout << "solution" << std::endl;
        solution.print();
    }

    return 0;
}

int test_avx_matrix_mpi(int* argc, char*** argv) {
    auto mpi = MPI(argc, argv);

    AvxMatrix<double> matrix;

    if (mpi.rank() == 0) {
        auto gen = Generator();
        matrix = gen.random_diag_dom_int_matrix(SIZE, -RANGE, RANGE).to_double().to_avx_matrix(8, 32);
    }

    mpi.sync_avx_matrix(&matrix, 0);

    std::this_thread::sleep_for(std::chrono::milliseconds(1000 * mpi.rank()));

    std::cout << mpi.rank() << std::endl;
    matrix.to_matrix(false).print();
    std::cout << std::endl;

    return 0;
}

int main(int argc, char** argv) {
    return test_avx_matrix_mpi(&argc, &argv);
}