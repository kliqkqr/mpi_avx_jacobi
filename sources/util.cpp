#include "util.h"

std::pair<int, int> distribute_even(int height, int rank, int size) {
    const int count_max = height % size;
    const int size_min  = height / size;
    const int size_max  = size_min + 1;

    const int offset = std::min(count_max, rank) * size_max + std::max(0, rank - count_max) * size_min;
    const int count  = rank < count_max ? size_max : size_min;

    return std::pair<int, int>(offset, count);
}