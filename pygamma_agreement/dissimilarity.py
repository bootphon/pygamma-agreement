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
"""
##########
Dissimilarity
##########

"""
import abc
import time
from typing import List, TYPE_CHECKING, Tuple, Union, Callable
from abc import ABCMeta

from pyannote.core import Segment
from Levenshtein import distance as lev

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from sortedcontainers import SortedSet
from .numba_utils import iter_tuples

if TYPE_CHECKING:
    from .continuum import Continuum
    from .alignment import Alignment

dissimilarity_dec = nb.cfunc(nb.float32(nb.float32[:], nb.float32[:]), nopython=True)


class AbstractDissimilarity(metaclass=ABCMeta):
    def __init__(self, delta_empty: float = 1.0):
        self.delta_empty = np.float32(delta_empty)

    @property
    @abc.abstractmethod
    def d_mat(self):
        raise NotImplemented()

    @staticmethod
    def build_arrays_continuum(continuum: 'Continuum') -> nb.typed.List:
        """
        Builds the compact, array-shaped representation of a continuum.
        """
        categories = continuum.categories

        unit_arrays = nb.typed.List()
        for annotator_id, (annotator, units) in enumerate(continuum._annotations.items()):
            # dim x : segment
            # dim y : (start, end, dur) / annotation
            unit_array = np.zeros((len(units), 4), dtype=np.float32)
            for unit_id, unit in enumerate(units):
                unit_array[unit_id][0] = unit.segment.start
                unit_array[unit_id][1] = unit.segment.end
                unit_array[unit_id][2] = unit.segment.duration
                unit_array[unit_id][3] = categories.index(unit.annotation)
            unit_arrays.append(unit_array)
        return unit_arrays

    @staticmethod
    def build_arrays_alignment(alignment: 'Alignment') -> np.ndarray:
        """
        Builds the compact, array-shaped representation of an alignment.
        """
        categories = alignment.categories
        nb_unitary_alignments = len(alignment.unitary_alignments)
        annotators = alignment.annotators
        nb_annotators = len(annotators)
        alignment_array = np.full((nb_unitary_alignments, nb_annotators, 4),
                                  fill_value=-1, dtype=np.float32)
        for i, unitary_alignment in enumerate(alignment.unitary_alignments):
            for annotator, unit in unitary_alignment.n_tuple:
                if unit is not None:
                    annotator_i = annotators.index(annotator)
                    alignment_array[i, annotator_i, 0] = unit.segment.start
                    alignment_array[i, annotator_i, 1] = unit.segment.end
                    alignment_array[i, annotator_i, 2] = unit.segment.duration
                    alignment_array[i, annotator_i, 3] = categories.index(unit.annotation)
        return alignment_array

    @staticmethod
    @nb.njit(nb.float32[:](nb.float32[:, :, ::1],
                           nb.types.FunctionType(nb.float32(nb.float32[:],
                                                            nb.float32[:])),
                           nb.float32))
    def compute_alignment_disorders(alignment_array: np.ndarray, d_mat, delta_empty: float):
        nb_alignments, nb_annotators, _ = alignment_array.shape
        res = np.zeros(nb_alignments, dtype=np.float32)
        c2n = nb_annotators * (nb_annotators - 1) // 2
        for unitary_alignment_i in range(nb_alignments):
            unitary_alignment = alignment_array[unitary_alignment_i]
            for i in range(nb_annotators):
                for j in range(i):
                    if unitary_alignment[i, 0] == -1 or unitary_alignment[j, 0] == -1:
                        res[unitary_alignment_i] += delta_empty
                    else:
                        res[unitary_alignment_i] += d_mat(unitary_alignment[i], unitary_alignment[j])
        res /= c2n
        return res

    @staticmethod
    @nb.njit(nb.types.Tuple((nb.float32[:], nb.int16[:, :]))(nb.types.ListType(nb.float32[:, ::1]),
                                                             nb.types.FunctionType(nb.float32(nb.float32[:],
                                                                                              nb.float32[:])),
                                                             nb.float32))
    def get_all_valid_alignments(unit_arrays: nb.typed.List,
                                 d_mat,
                                 delta_empty: float):
        chunk_size = (10**6) // 8
        nb_annotators = len(unit_arrays)
        c2n = (nb_annotators * (nb_annotators - 1) // 2)
        criterium = c2n * delta_empty * nb_annotators

        sizes_with_null = np.zeros(nb_annotators).astype(np.int16)
        sizes = np.zeros(nb_annotators).astype(np.int16)
        for annotator_id in range(nb_annotators):
            sizes[annotator_id] = len(unit_arrays[annotator_id])
            sizes_with_null[annotator_id] = len(unit_arrays[annotator_id]) + 1

        # PRECOMPUTATION OF ALL INTER-ANNOTATOR COUPLES OF UNITS
        precomputation = nb.typed.List([nb.typed.List([np.zeros((1, 1), dtype=np.float32) for _ in range(i)])
                                        for i in range(nb_annotators)])
        for annotator_a in range(nb_annotators):
            for annotator_b in range(annotator_a):
                nb_annot_a, nb_annot_b = sizes[annotator_a], sizes[annotator_b]
                matrix = np.full((nb_annot_a + 1, nb_annot_b + 1), fill_value=delta_empty, dtype=np.float32)
                for annot_a in range(nb_annot_a):
                    for annot_b in range(nb_annot_b):
                        matrix[annot_a, annot_b] = d_mat(unit_arrays[annotator_a][annot_a],
                                                         unit_arrays[annotator_b][annot_b])
                precomputation[annotator_a][annotator_b] = matrix

        disorders = np.zeros(chunk_size, dtype=np.float32)
        alignments = np.zeros((chunk_size, nb_annotators), dtype=np.int16)
        i_chosen = 0
        for unitary_alignment in iter_tuples(sizes_with_null):
            # for each tuple (corresponding to a unitary alignment), compute disorder
            disorder = 0
            for annot_a in range(nb_annotators):
                for annot_b in range(annot_a):
                    # this block looks a bit slow (because of all the variables
                    # declarations) but should be fairly sped up automatically
                    # by the LLVM optimization pass
                    unit_a_id, unit_b_id = unitary_alignment[annot_a], unitary_alignment[annot_b]
                    disorder += precomputation[annot_a][annot_b][unit_a_id, unit_b_id]
            if disorder <= criterium:
                disorders[i_chosen] = disorder
                alignments[i_chosen] = unitary_alignment
                i_chosen += 1
                if i_chosen == chunk_size:
                    disorders = np.concatenate((disorders, np.zeros(chunk_size // 2, dtype=np.float32)))
                    alignments = np.concatenate((alignments, np.zeros((chunk_size // 2, nb_annotators), dtype=np.int16)))
                    chunk_size += chunk_size // 2
        disorders, alignments = disorders[:i_chosen], alignments[:i_chosen]
        disorders /= c2n
        return disorders, alignments

    @abc.abstractmethod
    def d(self, unit1: 'Unit', unit2: 'Unit'):
        raise NotImplemented()

    def valid_alignments(self, continuum: 'Continuum') -> Tuple[np.ndarray, np.ndarray]:
        units_array = self.build_arrays_continuum(continuum)
        disorders, alignments = self.get_all_valid_alignments(units_array, self.d_mat, self.delta_empty)
        return disorders, alignments

    def compute_disorder(self, alignment: 'Alignment') -> np.array:
        alignment_arrays = self.build_arrays_alignment(alignment)
        return self.compute_alignment_disorders(alignment_arrays, self.d_mat, self.delta_empty)


class PositionalDissimilarity(AbstractDissimilarity):
    def __init__(self, delta_empty: float = 1.0):
        super().__init__(delta_empty)

    @property
    def d_mat(self):
        delta_empty = self.delta_empty

        @dissimilarity_dec
        def d_mat(unit1: np.ndarray, unit2: np.ndarray):
            dist = ((np.abs(unit1[0] - unit2[0]) + np.abs(unit1[1] - unit2[1])) /
                    (unit1[2] + unit2[2]))
            return dist * dist * delta_empty
        return d_mat

    def d(self, unit1: 'Unit', unit2: 'Unit'):
        pos = ((abs(unit1.segment.start - unit2.segment.start) + abs(unit1.segment.end - unit2.segment.end)) /
               (unit1.segment.duration + unit2.segment.duration))
        return pos * pos * self.delta_empty


class CategoricalDissimilarity(AbstractDissimilarity, metaclass=abc.ABCMeta):
    pass


class PrecomputedCategoricalDissimilarity(CategoricalDissimilarity, metaclass=abc.ABCMeta):
    def __init__(self, categories: SortedSet, delta_empty: float = 1.0):
        from .continuum import Unit
        nb_categories = len(categories)
        self.categories = categories
        self.matrix = np.zeros((nb_categories, nb_categories), dtype=np.float32)
        matrix = self.matrix
        max_val = 1.0
        for i in range(nb_categories):
            for j in range(i):
                dist_cat = self.cat_dissim_func(categories[i], categories[j])
                max_val = max(max_val, dist_cat)
                matrix[i, j] = dist_cat
                matrix[j, i] = dist_cat
        matrix /= max_val
        super().__init__(delta_empty)

    @abc.abstractmethod
    def cat_dissim_func(self, str1: str, str2: str):
        raise NotImplemented()

    @property
    def d_mat(self):
        matrix = self.matrix

        @dissimilarity_dec
        def d_mat(unit1: np.ndarray, unit2: np.ndarray):
            return matrix[np.int8(unit1[3]), np.int8(unit2[3])]
        return d_mat

    def d(self, unit1: 'Unit', unit2: 'Unit'):
        return self.matrix[self.categories.index(unit1.annotation), self.categories.index(unit2.annotation)]


class MatrixCategoricalDissimilarity(CategoricalDissimilarity):
    def __init__(self, categories: SortedSet, matrix: np.ndarray, delta_empty: float = 1.0):
        assert matrix.shape == (len(categories), len(categories)), \
            "Provided categorical dissimilarity matrix's shape doesn't match number of categories."
        super().__init__(delta_empty)
        self.matrix = matrix
        self.categories = categories

    @property
    def d_mat(self):
        matrix = self.matrix

        @dissimilarity_dec
        def d_mat(unit1: np.ndarray, unit2: np.ndarray):
            return matrix[unit1[3], unit2[3]]
        return d_mat

    def d(self, unit1: 'Unit', unit2: 'Unit'):
        return self.matrix[self.categories.index(unit1.annotation), self.categories.index(unit2.annotation)]


class AbsoluteCategoricalDissimilarity(CategoricalDissimilarity):
    def __init__(self, delta_empty: float = 1.0):
        super().__init__(delta_empty)

    @property
    def d_mat(self):

        @dissimilarity_dec
        def d_mat(unit1: np.ndarray, unit2: np.ndarray):
            return 0 if unit1[3] == unit2[3] else 1
        return d_mat

    def d(self, unit1: 'Unit', unit2: 'Unit'):
        return float(unit1.annotation != unit2.annotation)


class LevenshteinCategoricalDissimilarity(PrecomputedCategoricalDissimilarity):
    def __init__(self, categories: SortedSet, delta_empty: float = 1.0):
        super().__init__(categories, delta_empty)

    @staticmethod
    @nb.cfunc(nb.float32(nb.types.string, nb.types.string))
    def levenshtein(str1: str, str2: str):
        n1, n2 = len(str1) + 1, len(str2) + 1
        matrix_lev = np.zeros((n1, n2), dtype=np.int16)
        for i in range(1, n1):
            matrix_lev[i, 0] = i
        for j in range(1, n2):
            matrix_lev[0, j] = j
        for j in range(1, n2):
            for i in range(1, n1):
                cost = int(str1[i - 1] != str2[j - 1])
                matrix_lev[i, j] = np.min(np.array([matrix_lev[i - 1, j] + 1,
                                                    matrix_lev[i, j - 1] + 1,
                                                    matrix_lev[i - 1, j - 1] + cost]))
        return matrix_lev[-1, -1]

    def cat_dissim_func(self, str1: str, str2: str):
        return self.levenshtein(str1, str2)


class OrdinalCategoricalDissimilarity(PrecomputedCategoricalDissimilarity):
    def __init__(self, categories: SortedSet, delta_empty: float = 1.0):
        try:
            np.array(list(categories), dtype=np.float32)
        except ValueError:
            raise ValueError("Cannot-use ordinal dissimilarity on non-numeric categories.")

        super().__init__(categories, delta_empty)

    def cat_dissim_func(self, str1: str, str2: str):
        return str(float(str1) - float(str2))


class CombinedCategoricalDissimilarity(AbstractDissimilarity):
    def __init__(self,
                 alpha: float = 1,
                 beta: float = 1,
                 delta_empty: float = 1.0,
                 cat_dissim: CategoricalDissimilarity = None):
        self.positional_dissim = PositionalDissimilarity(delta_empty)
        if cat_dissim is None:
            cat_dissim = AbsoluteCategoricalDissimilarity()
        self.categorical_dissim = cat_dissim

        alphabeta = np.array([alpha, beta], dtype=np.float32)
        self._alphabeta = alphabeta

        super().__init__(delta_empty)

    @property
    def d_mat(self):
        pos = self.positional_dissim.d_mat
        cat = self.categorical_dissim.d_mat
        alphabeta = self._alphabeta

        @dissimilarity_dec
        def d_mat(unit1: np.ndarray, unit2: np.ndarray):
            return (alphabeta[0] * pos(unit1, unit2) +
                    alphabeta[1] * cat(unit1, unit2))

        return d_mat


    @property
    def alpha(self) -> float:
        return self._alphabeta[0]

    @property
    def beta(self) -> float:
        return self._alphabeta[1]

    @alpha.setter
    def alpha(self, alpha: float):
        self._alphabeta[0] = alpha

    @beta.setter
    def beta(self, beta):
        self._alphabeta[1] = beta

    def d(self, unit1: 'Unit', unit2: 'Unit'):
        return self.alpha * self.positional_dissim.d(unit1, unit2) + \
               self.beta * self.categorical_dissim.d(unit1, unit2)


