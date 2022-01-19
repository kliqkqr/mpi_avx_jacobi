#ifndef MPI_AVX_JACOBI_GENERATOR_H
#define MPI_AVX_JACOBI_GENERATOR_H

#include <random>

#include "matrix.h"
#include "vector.h"

class Generator {
    std::mt19937 _rng;

public:
    Generator();

    int random_int(int min, int max);
    int random_int(std::uniform_int_distribution<std::mt19937::result_type> dist);
    int* random_int_array(int size, int min, int max);

    Vector<int> random_int_vector(int height, int min, int max);
    Matrix<int> random_diag_dom_int_matrix(int height, int width, int min, int max);
};


#endif //MPI_AVX_JACOBI_GENERATOR_H
