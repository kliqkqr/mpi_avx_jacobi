#ifndef MPI_AVX_JACOBI_MPI_H
#define MPI_AVX_JACOBI_MPI_H

#include <MPI/Include/mpi.h>

#include "vector.h"
#include "matrix.h"

enum MPITag {
    MPITag_Double,
    MPITag_Vector,
    MPITag_Matrix_Dimensions,
    MPITag_Matrix_Buffer,
};

class MPI {
    bool _init;
    int _rank;
    int _size;

public:
    MPI();
    MPI(int* argc, char*** argv);

    ~MPI();

    void init(int* argc, char*** argv);

    [[nodiscard]] bool is_init() const;
    [[nodiscard]] int rank() const;
    [[nodiscard]] int size() const;

    void send_double(double* buffer, int count, int dest) const;
    void recv_double(double* buffer, int count, int source) const;

    void send_vector(Vector<double>* vector, int dest) const;
    void recv_vector(Vector<double>* vector, int source) const;
    void sync_vector(Vector<double>* vector, int source) const;

    void send_matrix(Matrix<double>* matrix, int dest) const;
    void recv_matrix(Matrix<double>* matrix, int source) const;
    void sync_matrix(Matrix<double>* matrix, int source) const;

    void fold_vector_ring(Vector<double>* vector, int offset, int count) const;
};

#endif //MPI_AVX_JACOBI_MPI_H
