from typing import List

import numba as nb
import numpy as np

LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')


@nb.njit
def fast_factorial(n):
    """Fast factorial is computed using a simple lookup table"""
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]


@nb.njit(nb.int32(nb.int32, nb.int32))
def binom(n, k):
    return fast_factorial(n) / (fast_factorial(k) * fast_factorial(n - k))


@nb.njit(nb.types.Tuple((nb.int32[:, :],
                         nb.int32[:]))(nb.int32[:],
                                       nb.int32[:],
                                       nb.int64, nb.int64))
def cproduct(sizes: np.ndarray, current_tuple: np.ndarray, start_idx: int, end_idx: int):
    assert len(sizes) >= 2
    assert start_idx < end_idx

    tuples = np.zeros((end_idx - start_idx, len(sizes)), dtype=np.int32)
    tuple_idx = 0
    current_tuple = current_tuple.copy()
    while tuple_idx < end_idx - start_idx:
        tuples[tuple_idx] = current_tuple
        current_tuple[0] += 1
        if current_tuple[0] == sizes[0]:
            current_tuple[0] = 0
            current_tuple[1] += 1
            for i in range(1, len(sizes) - 1):
                if current_tuple[i] == sizes[i]:
                    current_tuple[i + 1] += 1
                    current_tuple[i] = 0
                else:
                    break
        tuple_idx += 1
    return tuples, current_tuple


def chunked_cartesian_product(sizes: List[int], chunk_size: int):
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


def cartesian_product(sizes: List[int]):
    return next(chunked_cartesian_product(sizes, np.prod(sizes)))


