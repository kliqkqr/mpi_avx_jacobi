#ifndef MPI_AVX_JACOBI_MPI_H
#define MPI_AVX_JACOBI_MPI_H

#include <MPI/Include/mpi.h>

#include "avx.h"
#include "avx_matrix.h"
#include "vector.h"
#include "matrix.h"
#include "util.h"

enum MPITag {
//    MPITag_Double,
    MPITag_Vector,
    MPITag_Matrix_Dimensions,
    MPITag_Matrix_Buffer,
//    MPITag_Byte,
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

template <typename A>
void MPI::send_avx_matrix(AvxMatrix<A>* matrix, int dest) const {
    if (!this->is_init()) {
        return;
    }

    const int width            = matrix->width();
    const int height           = matrix->height();
    const int align_width      = matrix->align_width();
    const int block_elem_count = matrix->block_elem_count();
    const int block_align_size = matrix->block_align_size();
    const int align_size       = matrix->align_size();

    const int meta[6] = {
            width, height,
            align_width, block_elem_count, block_align_size, align_size
    };
    MPI_Send(meta, 6, MPI_INT, dest, MPITag_AvxMatrix_Meta, MPI_COMM_WORLD);

    uint8_t* buffer = matrix->to_byte_buffer();
    const size_t   size   = matrix->byte_buffer_size();
    MPI_Send(buffer, size, MPI_BYTE, dest, MPITag_AvxMatrix_Buffer, MPI_COMM_WORLD);
}

template <typename A>
void MPI::recv_avx_matrix(AvxMatrix<A>* matrix, int source) const {
    if (!this->is_init()) {
        return;
    }

    int meta[6];
    MPI_Recv(meta, 6, MPI_INT, source, MPITag_AvxMatrix_Meta, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    const int width            = meta[0];
    const int height           = meta[1];
    const int align_width      = meta[2];
    const int block_elem_count = meta[3];
    const int block_align_size = meta[4];
    const int align_size       = meta[5];

    const bool create_new = width            != matrix->width()  ||
                            height           != matrix->height() ||
                            align_width      != matrix->align_width() ||
                            block_elem_count != matrix->block_elem_count() ||
                            block_align_size != matrix->block_align_size() ||
                            align_size       != matrix->align_size();

    if (create_new) {
        const int elem_count = height * align_width;
        uint8_t* buffer      = (uint8_t*)avx_allocate_aligned_blocks<A>(elem_count, block_elem_count, align_size);
        const int size       = block_align_size * align_width * height;

        MPI_Recv(buffer, size, MPI_BYTE, source, MPITag_AvxMatrix_Buffer, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        matrix->buffer_free();
        *matrix = AvxMatrix<A>(height, width, buffer, align_width, block_elem_count, block_align_size, align_size);
        return;
    }

    uint8_t* buffer = matrix->to_byte_buffer();
    const int size  = (int)matrix->byte_buffer_size();

    MPI_Recv(buffer, size, MPI_BYTE, source, MPITag_AvxMatrix_Buffer, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

template <typename A>
void MPI::sync_avx_matrix(AvxMatrix<A>* matrix, int source) const {
    if (!this->is_init()) {
        return;
    }

    if (!(source < this->size() && source >= 0)) {
        return;
    }

    // TODO: Broadcast
    if (this->rank() == source) {
        for (int i = 0; i < this->size(); i += 1) {
            if (i != source) {
                this->send_avx_matrix(matrix, i);
            }
        }

        return;
    }

    this->recv_avx_matrix(matrix, source);
}

#endif //MPI_AVX_JACOBI_MPI_H