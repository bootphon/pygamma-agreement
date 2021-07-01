# The MIT License (MIT)

# Copyright (c) 2020-2021 CoML

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Rachid RIAD, Hadrien TITEUX, Léopold FAVRE

from typing import List

import numba as nb
import numpy as np


@nb.njit(nb.types.Tuple((nb.int32[:, :],
                         nb.int32[:]))(nb.int32[:],
                                       nb.int32[:],
                                       nb.int64, nb.int64))
def cproduct(sizes: np.ndarray, current_tuple: np.ndarray, start_idx: int, end_idx: int):
    """
    A numba njitted function that builds chunks of the cartesian product.
    It works by enumerating all possible combinations of the indices of
    each input sets, starting at the `current_tuple` element, and stopping
    after enumerating `start_idx - end_idx` elements.

    Parameters
    ----------
    sizes: np.ndarray of shape (N,)
        Cardinals of the sets
    current_tuple: np.ndarray of shape (N,)
        First cartesian combination to be enumerated
    start_idx:
        Index of `current_tuple` in the global cartesian product
    end_idx
        Index of last cartesian combination to be enumerated (not included in
        the returned tuples)

    Returns
    -------
    np.ndarray of shape (end_ix - start_idx, N)
    np.ndarray of shape (N,)

    """
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
    """Computes (fast) the cartesian product for the all the possible
    indices for the sets sets whose cardinals are defined in the list `sizes`,
    in chunks of size `chunk_size`.

    Parameters
    ----------
    sizes: List[int]D
        List of cardinals for the sets for which the function will produce the
        cartesian product
    chunk_size: int
        Size of the cartesian product chunks

    Returns
    -------
    np.ndarray of shape (chunk_size, len(sizes))
    """
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
    """Regular cartesian product"""
    return next(chunked_cartesian_product(sizes, np.prod(sizes)))
