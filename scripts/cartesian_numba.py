import numba as nb
import numpy as np
from typing import List


@nb.njit(nb.types.Tuple((nb.int32[:, :],
                         nb.int32[:]))(nb.int32[:],
                                       nb.int32[:],
                                       nb.int64, nb.int64))
def cproduct(sizes: np.ndarray, current_tuple: np.ndarray, start_idx: int, end_idx: int):
    """Generates ids tuples from start_id to end_id"""
    assert len(sizes) >= 2
    assert start_idx < end_idx

    tuples = np.zeros((end_idx - start_idx, len(sizes)), dtype=np.int32)
    tuple_idx = 0
    # stores the current combination
    current_tuple = current_tuple.copy()
    while tuple_idx < end_idx - start_idx:
        tuples[tuple_idx] = current_tuple
        current_tuple[0] += 1
        # using a condition here instead of including this in the inner loop
        # to gain a bit of speed: this is going to be tested each iteration,
        # and starting a loop to have it end right away is a bit silly
        if current_tuple[0] == sizes[0]:
            # the reset to 0 and subsequent increment amount to carrying
            # the number to the higher "power"
            current_tuple[0] = 0
            current_tuple[1] += 1
            for i in range(1, len(sizes) - 1):
                if current_tuple[i] == sizes[i]:
                    # same as before, but in a loop, since this is going
                    # to get called less often
                    current_tuple[i + 1] += 1
                    current_tuple[i] = 0
                else:
                    break
        tuple_idx += 1
    return tuples, current_tuple


def chunked_cartesian_product_ids(sizes: List[int], chunk_size: int):
    """Just generates chunks of the cartesian product of the ids of each
    input arrays (thus, we just need their sizes here, not the actual arrays)"""
    prod = np.prod(sizes)

    # putting the largest number at the front to more efficiently make use
    # of the cproduct numba function
    sizes = np.array(sizes, dtype=np.int32)
    sorted_idx = np.argsort(sizes)[::-1]
    sizes = sizes[sorted_idx]
    if chunk_size > prod:
        chunk_bounds = (np.array([0, prod])).astype(np.int64)
    else:
        num_chunks = np.maximum(np.ceil(prod / chunk_size), 2).astype(np.int32)
        chunk_bounds = (np.arange(num_chunks + 1) * chunk_size).astype(np.int64)
        chunk_bounds[-1] = prod
    current_tuple = np.zeros(len(sizes), dtype=np.int32)
    for start_idx, end_idx in zip(chunk_bounds[:-1], chunk_bounds[1:]):
        tuples, current_tuple = cproduct(sizes, current_tuple, start_idx, end_idx)
        # re-arrange columns to match the original order of the sizes list
        # before yielding
        yield tuples[:, np.argsort(sorted_idx)]


def chunked_cartesian_product(*arrays, chunk_size=2 ** 25):
    """Returns chunks of the full cartesian product, with arrays of shape
    (chunk_size, n_arrays). The last chunk will obviously have the size of the
    remainder"""
    array_lengths = [len(array) for array in arrays]
    for array_ids_chunk in chunked_cartesian_product_ids(array_lengths, chunk_size):
        slices_lists = [arrays[i][array_ids_chunk[:, i]] for i in range(len(arrays))]
        yield np.vstack(slices_lists).swapaxes(0,1)


def cartesian_product(*arrays):
    """Actual cartesian product, not chunked, still fast"""
    total_prod = np.prod([len(array) for array in arrays])
    return next(chunked_cartesian_product(*arrays, total_prod))


a = np.arange(0, 3)
b = np.arange(8, 10)
c = np.arange(13, 16)
for cartesian_tuples in chunked_cartesian_product(*[a, b, c], chunk_size=5):
    print(cartesian_tuples)
