cmake_minimum_required(VERSION 3.21)
project(mpi_avx_jacobi)

set(CMAKE_CXX_STANDARD 20)

include_directories(headers)

add_executable(mpi_avx_jacobi main.cpp headers/matrix.h)
