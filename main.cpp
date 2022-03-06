#include <cassert>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <utility>

#include "avx.h"
#include "vector.h"
#include "matrix.h"
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

float dot_product(float a[], float b[], int size) {
    float sum = 0;

    for (int i = 0; i < size; i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

std::vector<float*> gen_arr_float(int count, int size) {
    auto result = std::vector<float*>();
    auto gen    = Generator();

    for (int i = 0; i < count; i += 1) {
        auto arr = gen.random_int_vector(size, 0, 10).to_float().copy_buffer();
        result.push_back(arr);
    }

    return result;
}


void calc_dot_product_avx(const std::vector<float*>& a, const std::vector<float*>& b, int size, float* result) {
    for (int i = 0; i < a.size(); i += 1) {
        result[i] = avx_dot_product_no_align_m256(a[i], b[i], size);
    }
}

void calc_dot_product_sim(const std::vector<float*>& a, const std::vector<float*>& b, int size, float* result) {
    for (int i = 0; i < a.size(); i += 1) {
        result[i] = dot_product(a[i], b[i], size);
    }
}

#define SIZE 65000

int main() {
    auto a = gen_arr_float(SIZE, SIZE);

    auto r_sim = (float*)malloc(sizeof(float) * SIZE);
    auto r_avx = (float*)malloc(sizeof(float) * SIZE);

    auto t_sim = benchmark([&]() { calc_dot_product_sim(a, a, SIZE, r_sim); });
    auto t_avx = benchmark([&]() { calc_dot_product_avx(a, a, SIZE, r_avx); });

    std::cout << "t_sim: " << t_sim << "; t_avx: " << t_avx << "; ratio: " << (float)t_sim / (float)t_avx << std::endl;

    float s_sim = 0, s_avx = 0;
    for (int i = 0; i < SIZE; i += 1) {
        s_sim += r_sim[i];
        s_avx += r_avx[i];
    }

    std::cout << "s_sim: " << s_sim << "; s_avx: " << s_avx << std::endl;

    return 0;
}