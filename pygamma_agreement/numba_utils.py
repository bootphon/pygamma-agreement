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

import numba as nb
import numpy as np


@nb.njit(nb.float32(nb.types.string, nb.types.string))
def levenshtein(str1: str, str2: str):
    n1, n2 = len(str1) + 1, len(str2) + 1
    matrix_lev = np.empty((n1, n2), dtype=np.int16)
    for i in range(1, n1):
        matrix_lev[i, 0] = i
    for j in range(1, n2):
        matrix_lev[0, j] = j
    for j in range(1, n2):
        for i in range(1, n1):
            cost = int(str1[i-1] != str2[j-1])
            matrix_lev[i, j] = np.min(np.array([matrix_lev[i-1, j] + 1,
                                                matrix_lev[i, j-1] + 1,
                                                matrix_lev[i-1, j-1] + cost]))
    return matrix_lev[-1, -1]


@nb.njit(nb.int16[:, ::1](nb.int16[:, ::1], nb.int64), parallel=True)
def extend_right_alignments(arr: np.ndarray, n: int):
    i, j = arr.shape
    new_array = np.empty((i, j + n), dtype=np.int16)
    new_array[:, :j] = arr
    return new_array


@nb.njit(nb.float32[:](nb.float32[:], nb.int64), parallel=True)
def extend_right_disorders(arr: np.ndarray, n: int):
    new_array = np.empty(len(arr) + n, dtype=np.float32)
    new_array[:len(arr)] = arr
    return new_array

@nb.njit()
def iter_tuples(sizes: np.ndarray):
    """
    Iterates over all the arrays of {0..sizes[0]-1} * {0..size[1]-1} * ... * {0..size[n-1]-1}
    """
    nb_annotators = len(sizes)
    current = np.zeros(nb_annotators, dtype=np.int16)
    while True:
        yield current
        for i in range(nb_annotators):
            current[i] += 1
            if current[i] < sizes[i]:
                break
            current[i] = 0
        else:
            return






