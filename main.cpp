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
#include "util.h"

const int SIZE  = 20;
const int RANGE = 10;
const int ITERATIONS = 50;

int main(int argc, char** argv) {
    MPI mpi = MPI(&argc, &argv);

    Matrix<double> matrix;
    Vector<double> vector, result, solution;

    if (mpi.rank() == 0) {
        Generator generator = Generator();

        matrix = generator.random_diag_dom_int_matrix(SIZE, SIZE, -RANGE, RANGE).to_double();
        vector = generator.random_int_vector(SIZE, -RANGE, RANGE).to_double();
        result = matrix * vector;
    }

    mpi.sync_matrix(&matrix, 0);
    mpi.sync_vector(&vector, 0);
    mpi.sync_vector(&result, 0);

    auto offset_count = distribute_even(vector.height(), mpi.rank(), mpi.size());
    const int offset = offset_count.first;
    const int count  = offset_count.second;

    solution = jacobi_mpi_allgatherv(matrix, result, ITERATIONS, mpi, offset, count);

    if (mpi.rank() == 0) {
        std::cout << "vector" << std::endl;
        vector.print();

        std::cout << "solution" << std::endl;
        solution.print();
    }
}