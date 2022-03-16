def foo(elem_size, elem_count, block_elem_count, align_size):
    block_size = block_elem_count * elem_size
    block_size_align = block_size if block_size % align_size == 0 else align_size * ((block_size // align_size) + 1)

    return block_size, block_size_align


def round_up(n, f):
    assert f > 0

    rem = n % f

    if rem == 0:
        return n

    return n + f - rem

def round_up_2(n, f):
    assert f > 0

    return (n + f - 1) & -f