#include "generator.h"

Generator::Generator() {
    std::random_device device = std::random_device();

    this->_rng = std::mt19937(device());
}

int Generator::random_int(int min, int max) {
    auto dist = std::uniform_int_distribution<std::mt19937::result_type>(min, max);

    return dist(this->_rng);
}

int Generator::random_int(std::uniform_int_distribution<std::mt19937::result_type> dist) {
    return dist(this->_rng);
}

int* Generator::random_int_array(int size, int min, int max) {
    int* array = (int*)std::malloc(sizeof(int) * size);

    if (array == nullptr) {
        return nullptr;
    }

    auto dist = std::uniform_int_distribution<std::mt19937::result_type>(min, max);

    for (int i = 0; i < size; i += 1) {
        array[i] = this->random_int(dist);
    }

    return array;
}

Matrix<int> Generator::random_diag_dom_int_matrix(int size, int min, int max) {
    Matrix<int> matrix = Matrix<int>(size, size, 0);

    for (int y = 0; y < size; y += 1) {
        int* row = this->random_int_array(size, min, max);

        assert(row != nullptr);

        int abs_sum = 1;
        for (int x = 0; x < size; x += 1) {
            if (x == y) {
                continue;
            }

            abs_sum += std::abs(row[x]);
        }

        if (this->random_int(0, 1)) {
            row[y] = -abs_sum;
        }
        else {
            row[y] = abs_sum;
        }

        matrix.set_row(y, row);
    }

    return matrix;
}

Vector<int> Generator::random_int_vector(int height, int min, int max) {
    int* column = this->random_int_array(height, min, max);

    assert(column != nullptr);

    return Vector<int>(height, column);
}