#ifndef MPI_AVX_JACOBI_MPI_H
#define MPI_AVX_JACOBI_MPI_H

#include <MPI/Include/mpi.h>

#include "avx.h"
#include "avx_matrix.h"
#include "vector.h"
#include "matrix.h"
#include "util.h"

enum MPITag {
    MPITag_Double,
    MPITag_Vector,
    MPITag_Matrix_Dimensions,
    MPITag_Matrix_Buffer,
    MPITag_Byte,
    MPITag_AvxMatrix_Meta,
    MPITag_AvxMatrix_Buffer
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

    void wait(MPI_Request* request, MPI_Status* status) const;

//    void send_byte(uint8_t* buffer, int count, int dest) const;
//    void recv_byte(uint8_t* buffer, int count, int source) const;
//
//    void send_double(double* buffer, int count, int dest) const;
//    void recv_double(double* buffer, int count, int source) const;

    void send_vector(Vector<double>* vector, int dest) const;
    void recv_vector(Vector<double>* vector, int source) const;

    void isend_vector(Vector<double>* vector, int dest, MPI_Request* request) const;
    void irecv_vector(Vector<double>* vector, int source, MPI_Request* request) const;

    void bcast_vector(Vector<double>* vector, int root) const;

    void sync_vector(Vector<double>* vector, int source) const;

    void send_matrix(Matrix<double>* matrix, int dest) const;
    void recv_matrix(Matrix<double>* matrix, int source) const;
    void sync_matrix(Matrix<double>* matrix, int source) const;

    template <typename A>
    void send_avx_matrix(AvxMatrix<A>* matrix, int dest) const;

    template <typename A>
    void recv_avx_matrix(AvxMatrix<A>* matrix, int source) const;

    template <typename A>
    void sync_avx_matrix(AvxMatrix<A>* matrix, int source) const;

    void fold_vector_ring(Vector<double>* vector, int offset, int count) const;
    void fold_vector_iring(Vector<double>* vector, int offset, int count) const;
    void fold_vector_gatherv(Vector<double>* vector, int offset, int count, int root) const;
    void fold_vector_allgatherv(Vector<double>* vector, int offset, int count) const;
};

#endif //MPI_AVX_JACOBI_MPI_H
