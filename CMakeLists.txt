cmake_minimum_required(VERSION 3.7)
project(mpi_avx_jacobi)

set(CMAKE_CXX_STANDARD 17)

if (UNIX)
    include_directories(headers)
else()
    set(CMAKE_EXE_LINKER_FLAGS "-static")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx")

    include_directories(headers "c:\\Program Files (x86)\\Microsoft SDKs")

    link_directories("c:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Lib\\x86" "c:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Lib\\x64")
endif()

#set(CMAKE_EXE_LINKER_FLAGS "-static")
#
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx")
#
#include_directories(headers "c:\\Program Files (x86)\\Microsoft SDKs")
#
#link_directories("c:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Lib\\x86" "c:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Lib\\x64")

add_executable(mpi_avx_jacobi main.cpp headers/matrix.h headers/generator.h sources/generator.cpp headers/vector.h headers/jacobi.h sources/jacobi.cpp headers/mpi.h sources/mpi.cpp headers/util.h sources/util.cpp headers/bench.h sources/bench.cpp headers/avx.h sources/avx.cpp headers/avx_matrix.h)

if (NOT UNIX)
    target_link_libraries(mpi_avx_jacobi msmpi.lib msmpifec.lib msmpifmc.lib) #  msmpifes.lib msmpifms.lib
endif()

#target_link_libraries(mpi_avx_jacobi msmpi.lib msmpifec.lib msmpifmc.lib) #  msmpifes.lib msmpifms.lib