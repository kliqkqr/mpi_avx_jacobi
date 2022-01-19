#include "mpi.h"

MPI::MPI() {
    this->_init = false;
    this->_rank = -1;
    this->_size = -1;
}

MPI::MPI(int* argc, char*** argv) {
    MPI_Init(argc, argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &this->_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->_size);

    this->_init = true;
}

MPI::~MPI() {
    MPI_Finalize();
}

void MPI::init(int* argc, char*** argv) {
    MPI_Init(argc, argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &this->_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->_size);

    this->_init = true;
}

bool MPI::is_init() const {
    return this->_init;
}

int MPI::rank() const {
    return this->_rank;
}

int MPI::size() const {
    return this->_size;
}

void MPI::wait(MPI_Request* request, MPI_Status* status) const {
    MPI_Wait(request, status);
}

void MPI::send_double(double *buffer, int count, int dest) const {
    if (!this->is_init()) {
        return;
    }

    MPI_Send(buffer, count, MPI_DOUBLE, dest, MPITag_Double, MPI_COMM_WORLD);
}

void MPI::recv_double(double* buffer, int count, int source) const {
    if (!this->is_init()) {
        return;
    }

    MPI_Recv(buffer, count, MPI_DOUBLE, source, MPITag_Double, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void MPI::send_vector(Vector<double>* vector, int dest) const {
    if (!this->is_init()) {
        return;
    }

    const int     height = vector->height();
    const double* buffer = vector->get_buffer().get();
    MPI_Send(buffer, height, MPI_DOUBLE, dest, MPITag_Vector, MPI_COMM_WORLD);
}

// TODO: use recv_double
void MPI::recv_vector(Vector<double> *vector, int source) const {
    if (!this->is_init()) {
        return;
    }

    MPI_Status status;
    MPI_Probe(source, MPITag_Vector, MPI_COMM_WORLD, &status);

    int height;
    MPI_Get_count(&status, MPI_DOUBLE, &height);

    if (height == vector->height()) {
        double* buffer = vector->get_buffer().get();
        MPI_Recv(buffer, height, MPI_DOUBLE, source, MPITag_Vector, MPI_COMM_WORLD, &status);
        return;
    }

    double* buffer = (double*)std::malloc(sizeof(double) * height);
    MPI_Recv(buffer, height, MPI_DOUBLE, source, MPITag_Vector, MPI_COMM_WORLD, &status);

    vector->get_buffer().reset();
    *vector = Vector<double>(height, buffer);
}

void MPI::isend_vector(Vector<double> *vector, int dest, MPI_Request *request) const {
    if (!this->is_init()) {
        return;
    }

    const int     height = vector->height();
    const double* buffer = vector->get_buffer().get();
    MPI_Isend(buffer, height, MPI_DOUBLE, dest, MPITag_Vector, MPI_COMM_WORLD, request);
}

// TODO: height change handling
void MPI::irecv_vector(Vector<double> *vector, int source, MPI_Request* request) const {
    if (!this->is_init()) {
        return;
    }

    const int     height = vector->height();
          double* buffer = vector->get_buffer().get();
    MPI_Irecv(buffer, height, MPI_DOUBLE, source, MPITag_Vector, MPI_COMM_WORLD, request);
}

// TODO: height change handling
void MPI::bcast_vector(Vector<double> *vector, int root) const {
    if (!this->is_init()) {
        return;
    }

    const int     height = vector->height();
          double* buffer = vector->get_buffer().get();
    MPI_Bcast(buffer, height, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

void MPI::sync_vector(Vector<double> *vector, int source) const {
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
                this->send_vector(vector, i);
            }
        }

        return;
    }

    this->recv_vector(vector, source);
}

void MPI::send_matrix(Matrix<double>* matrix, int dest) const {
    if (!this->is_init()) {
        return;
    }

    const int height = matrix->height();
    const int width  = matrix->width();
    const int size   = height * width;

    const int dimensions[] = { height, width };
    MPI_Send(dimensions, 2, MPI_INT, dest, MPITag_Matrix_Dimensions, MPI_COMM_WORLD);

    const double* buffer = matrix->get_buffer().get();
    MPI_Send(buffer, size, MPI_DOUBLE, dest, MPITag_Matrix_Buffer, MPI_COMM_WORLD);
}

void MPI::recv_matrix(Matrix<double>* matrix, int source) const {
    if (!this->is_init()) {
        return;
    }

    int dimensions[2];
    MPI_Recv(dimensions, 2, MPI_INT, source, MPITag_Matrix_Dimensions, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    const int height = dimensions[0];
    const int width  = dimensions[1];
    const int size   = height * width;

    if (size == matrix->height() * matrix->width()) {
        double* buffer = matrix->get_buffer().get();
        MPI_Recv(buffer, size, MPI_DOUBLE, source, MPITag_Matrix_Buffer, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        matrix->set_dimensions(height, width);
        return;
    }

    double* buffer = (double*)std::malloc(sizeof(double) * size);
    MPI_Recv(buffer, size, MPI_DOUBLE, source, MPITag_Matrix_Buffer, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    matrix->get_buffer().reset();
    *matrix = Matrix<double>(height, width, buffer);
}

void MPI::sync_matrix(Matrix<double> *matrix, int source) const {
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
                this->send_matrix(matrix, i);
            }
        }

        return;
    }

    this->recv_matrix(matrix, source);
}

// TODO: maybe sendrecv
void MPI::fold_vector_ring(Vector<double>* vector, int offset, int count) const {
    if (!this->is_init()) {
        return;
    }

    const int dest   = (this->rank() + 1) % this->size();
    const int source = (((this->rank() - 1) % this->size()) + this->size()) % this->size();

    Vector<double> slice = vector->to_slice(offset, count);

    for (int i = 0; i < this->size(); i += 1) {
        this->send_vector(vector, dest);
        this->recv_vector(vector, source);

        for (int y = 0; y < count; y += 1) {
            vector->set(y + offset, slice.get(y));
        }
    }
}

// TODO: maybe Isendrecv
void MPI::fold_vector_iring(Vector<double> *vector, int offset, int count) const {
    if (!this->is_init()) {
        return;
    }

    const int dest   = (this->rank() + 1) % this->size();
    const int source = (((this->rank() - 1) % this->size()) + this->size()) % this->size();

    Vector<double> slice = vector->to_slice(offset, count);

    for (int i = 0; i < this->size(); i += 1) {
        MPI_Request send_request, recv_request;
        this->isend_vector(vector, dest,   &send_request);
        this->irecv_vector(vector, source, &recv_request);

        MPI_Status send_status, recv_status;
        this->wait(&send_request, &send_status);
        this->wait(&recv_request, &recv_status);

        for (int y = 0; y < count; y += 1) {
            vector->set(y + offset, slice.get(y));
        }
    }
}

void MPI::fold_vector_gatherv(Vector<double> *vector, int offset, int count, int root) const {
    if (!this->is_init()) {
        return;
    }

    // TODO: vector->as_slice(..) - implement cheaper conversion
    Vector<double> slice = vector->to_slice(offset, count);
    double* send_buffer = slice.get_buffer().get();
    double* recv_buffer = vector->get_buffer().get();

    Vector<int> displacements = Vector<int>(this->size(), 0);
    Vector<int> recv_counts   = Vector<int>(this->size(), 0);
    if (this->rank() == root) {
        for (int i = 0; i < this->size(); i += 1) {
            auto offset_count = distribute_even(vector->height(), i, this->size());
            displacements.set(i, offset_count.first);
            recv_counts.set(i, offset_count.second);
        }
    }

    // TODO: replace with this->gatherv_vector
    MPI_Gatherv(send_buffer, count, MPI_DOUBLE, recv_buffer, recv_counts.get_buffer().get(), displacements.get_buffer().get(), MPI_DOUBLE, root, MPI_COMM_WORLD);

    this->bcast_vector(vector, root);
}

void MPI::fold_vector_allgatherv(Vector<double> *vector, int offset, int count) const {
    if (!this->is_init()) {
        return;
    }

    // TODO: vector->as_slice(..) - implement cheaper conversion
    Vector<double> slice = vector->to_slice(offset, count);
    double* send_buffer = slice.get_buffer().get();
    double* recv_buffer = vector->get_buffer().get();

    Vector<int> displacements = Vector<int>(this->size(), 0);
    Vector<int> recv_counts   = Vector<int>(this->size(), 0);
    for (int i = 0; i < this->size(); i += 1) {
        auto offset_count = distribute_even(vector->height(), i, this->size());
        displacements.set(i, offset_count.first);
        recv_counts.set(i, offset_count.second);
    }

    MPI_Allgatherv(send_buffer, count, MPI_DOUBLE, recv_buffer, recv_counts.get_buffer().get(), displacements.get_buffer().get(), MPI_DOUBLE, MPI_COMM_WORLD);
}