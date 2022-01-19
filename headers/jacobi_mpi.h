#ifndef MPI_AVX_JACOBI_JACOBI_MPI_H
#define MPI_AVX_JACOBI_JACOBI_MPI_H

//#include <iostream>
//#include <cassert>
//
//#include <MPI/Include/mpi.h>
//
//#include "matrix.h"
//#include "vector.h"
//
//std::pair<int, int> mpi_setup(int argc, char** argv) {
//    MPI_Init(&argc, &argv);
//
//    int rank_res, size_res, rank, size;
//
//    rank_res = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    size_res = MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//    return std::pair<int, int>(rank, size);
//}
//
//void mpi_finalize() {
//    MPI_Finalize();
//}
//
//void mpi_send_double_vector(const Vector<double>& vector, int dest, int tag, MPI_Comm comm) {
//    const int height = vector.height();
//    MPI_Send(&height, 1, MPI_INT, dest, tag, comm);
//
//    const double* buffer = vector.copy_buffer();
//    MPI_Send(buffer, height, MPI_DOUBLE, dest, tag, comm);
//}
//
//void mpi_recv_double_vector(Vector<double>* vector, int source, int tag, MPI_Comm comm) {
//    MPI_Status status;
//    int height;
//
//    MPI_Recv(&height, 1, MPI_INT, source, tag, comm, &status);
//
//    double* buffer = (double*)std::malloc(sizeof(double) * height);
//    assert(buffer != nullptr);
//
//    MPI_Recv(buffer, height, MPI_DOUBLE, source, tag, comm, &status);
//
//    *vector = Vector<double>(height, buffer);
//}
//
//void mpi_send_double_matrix(const Matrix<double>& matrix, int dest, int tag, MPI_Comm comm) {
//    const int height = matrix.height();
//    const int width  = matrix.width();
//    const int size   = height * width;
//    const int int_buffer[2] = { height, width };
//
//    MPI_Send(int_buffer, 2, MPI_INT, dest, tag, comm);
//
//    const double* double_buffer = matrix.copy_buffer();
//
//    MPI_Send(double_buffer, size, MPI_DOUBLE, dest, tag, comm);
//}
//
//void mpi_recv_double_matrix(Matrix<double>* matrix, int source, int tag, MPI_Comm comm) {
//    MPI_Status status;
//    int height, width, size;
//    int int_buffer[2] = { -1, -1 };
//
//    MPI_Recv(int_buffer, 2, MPI_INT, source, tag, comm, &status);
//
//    height = int_buffer[0];
//    width  = int_buffer[1];
//    size   = height * width;
//
//    double* double_buffer = (double*)std::malloc(sizeof(double) * size);
//    assert(double_buffer != nullptr);
//
//    MPI_Recv(double_buffer, size, MPI_DOUBLE, source, tag, comm, &status);
//
//    *matrix = Matrix<double>(height, width, double_buffer);
//}

#endif //MPI_AVX_JACOBI_JACOBI_MPI_H
