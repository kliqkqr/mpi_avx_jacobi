#include <cassert>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>

#include "vector.h"
#include "matrix.h"
#include "generator.h"
#include "jacobi.h"
#include "mpi.h"

std::pair<int, int> offset_count(int height, int rank, int size) {
    const int count_max = height % size;
    const int count_min = size - count_max;
    const int size_min  = height / size;
    const int size_max  = size_min + 1;

    const int offset = std::min(count_max, rank) * size_max + std::max(0, rank - count_max) * size_min;
    const int count  = rank < count_max ? size_max : size_min;

    return std::pair<int, int>(offset, count);
}

const int SIZE  = 20;
const int RANGE = 10;
const int ITERATIONS = 50;

int main(int argc, char** argv) {
    MPI mpi = MPI(&argc, &argv);
    Generator generator = Generator();

    Matrix<double> matrix;
    Vector<double> vector, result, solution;

    if (mpi.rank() == 0) {
        matrix = generator.random_diag_dom_int_matrix(SIZE, SIZE, -RANGE, RANGE).to_double();
        vector = generator.random_int_vector(SIZE, -RANGE, RANGE).to_double();
        result = matrix * vector;
    }

    mpi.sync_matrix(&matrix, 0);
    mpi.sync_vector(&vector, 0);
    mpi.sync_vector(&result, 0);

    auto pair = offset_count(vector.height(), mpi.rank(), mpi.size());
    const int offset = pair.first;
    const int count  = pair.second;

    solution = jacobi_mpi_ring(matrix, result, ITERATIONS, mpi, offset, count);

    if (mpi.rank() == 0) {
        auto v = jacobi(matrix, result, ITERATIONS);

        std::cout << "vector" << std::endl;
        vector.print();

        std::cout << "solution" << std::endl;
        solution.print();
    }
}