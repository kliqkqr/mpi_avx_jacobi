cmake_minimum_required(VERSION 3.21)
project(mpi_avx_jacobi)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXE_LINKER_FLAGS "-static")

include_directories(headers "c:\\Program Files (x86)\\Microsoft SDKs")

link_directories("c:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Lib\\x86" "c:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Lib\\x64")

add_executable(mpi_avx_jacobi main.cpp headers/matrix.h headers/generator.h sources/generator.cpp headers/vector.h headers/jacobi.h sources/jacobi.cpp headers/jacobi_mpi.h headers/mpi.h sources/mpi.cpp)

target_link_libraries(mpi_avx_jacobi msmpi.lib msmpifec.lib msmpifmc.lib)
#  msmpifes.lib msmpifms.lib