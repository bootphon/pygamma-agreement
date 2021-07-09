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
import time
from typing import List, TYPE_CHECKING, Tuple, Union, Callable

import pyannote.core
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



class AbstractDissimilarity:
    def __init__(self, d_mat, delta_empty: float = 1.0):
        self.delta_empty = np.float32(delta_empty)
        self.d_mat = d_mat

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

    def build_arrays_alignment(self, alignment: 'Alignment') -> List[np.ndarray]:
        """
        Builds the compact, array-shaped representation of an alignment.
        """
        raise NotImplemented()

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

    def d(self, unit1: 'Unit', unit2: 'Unit'):
        unit1 = np.array([unit1.segment.start, unit1.segment.end, unit1.segment.end - unit1.segment.start])
        unit2 = np.array([unit2.segment.start, unit2.segment.end, unit2.segment.end - unit2.segment.start])

        return self.d_mat(unit1, unit2)

    @staticmethod
    def alignments_disorders(*args, **kwargs):
        """Computes the disorder for a batch of unitary alignments."""
        raise NotImplemented()

    def __call__(self, continuum: 'Continuum') -> Tuple[np.ndarray, np.ndarray]:
        units_array = self.build_arrays_continuum(continuum)
        disorders, alignments = self.get_all_valid_alignments(units_array, self.d_mat, self.delta_empty)
        return disorders, alignments


class PositionalDissimilarity(AbstractDissimilarity):
    def __init__(self, delta_empty: float = 1.0):

        @dissimilarity_dec
        def d_mat(unit1: np.ndarray, unit2: np.ndarray):
            dist = ((np.abs(unit1[0] - unit2[0]) + np.abs(unit1[1] - unit2[1])) /
                    (unit1[2] + unit2[2]))
            return dist * dist * delta_empty

        super().__init__(d_mat, delta_empty)


class CategoricalDissimilarity(AbstractDissimilarity):
    def __init__(self, delta_empty: float = 1.0, method: str = 'absolute'):

        if method == 'absolute':
            @dissimilarity_dec
            def d_mat(unit1, unit2):
                return 0.0 if unit1[3] == unit2[3] else 1.0
        elif method == 'ordinal':
            raise NotImplemented
        elif method == 'levenshtein':
            raise NotImplemented
        else:
            raise ValueError(f"Categorical dissimilarity method '{method}' unknown.")

        super().__init__(d_mat, delta_empty)


class CombinedCategoricalDissimilarity(AbstractDissimilarity):
    def __init__(self,
                 alpha: float = 1,
                 beta: float = 1,
                 delta_empty: float = 1.0,
                 cat_method: str = 'absolute',
                 precompile=True):
        self.positional_dissim = PositionalDissimilarity(delta_empty)
        self.categorical_dissim = CategoricalDissimilarity(delta_empty, method='absolute')

        self._alpha = nb.float32(alpha)
        self._beta = nb.float32(beta)
        pos = self.positional_dissim.d_mat
        cat = self.categorical_dissim.d_mat

        @dissimilarity_dec
        def d_mat(unit1: np.ndarray, unit2: np.ndarray):
            return (alpha * pos(unit1, unit2) +
                    beta * cat(unit1, unit2))

        super().__init__(d_mat, delta_empty)

    def __reset_dmat(self):
        alpha = self.alpha
        beta = self.beta
        pos = self.positional_dissim.d_mat
        cat = self.categorical_dissim.d_mat

        @dissimilarity_dec
        def d_mat(unit1: np.ndarray, unit2: np.ndarray):
            return (alpha * pos(unit1, unit2) +
                    beta * cat(unit1, unit2))
        self.d_mat = d_mat

    @property
    def alpha(self) -> float:
        return float(self._alpha)

    @property
    def beta(self) -> float:
        return float(self._beta)

    @alpha.setter
    def alpha(self, alpha: float):
        self._alpha = nb.float32(alpha)
        self.__reset_dmat()

    @beta.setter
    def beta(self, beta):
        self._beta = nb.float32(beta)
        self.__reset_dmat()



