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
from typing import List, TYPE_CHECKING, Tuple, Union, Callable

from Levenshtein import distance as lev

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from sortedcontainers import SortedSet
from .numba_utils import next_tuple

if TYPE_CHECKING:
    from .continuum import Continuum
    from .alignment import Alignment

class AbstractDissimilarity:
    def __init__(self, delta_empty: float = 1.0, d_mat=None):
        self.delta_empty = np.float32(delta_empty)
        self.d_mat = d_mat

    @staticmethod
    def build_arrays_continuum(continuum: 'Continuum') -> Tuple[nb.typed.List, nb.typed.List]:
        """
        Builds the compact, array-shaped representation of a continuum.
        """
        positions_arrays = nb.typed.List()
        category_arrays = nb.typed.List()
        for annotator_id, (annotator, units) in enumerate(continuum._annotations.items()):
            # dim x : segment
            # dim y : (start, end, dur) / annotation
            unit_dists_array = np.zeros((len(units) + 1, 3)).astype(np.float32)
            unit_cat_array = np.zeros(len(units) + 1, dtype=str).astype(np.str)
            for unit_id, unit in enumerate(units):
                unit_dists_array[unit_id][0] = unit.segment.start
                unit_dists_array[unit_id][1] = unit.segment.end
                unit_dists_array[unit_id][2] = unit.segment.duration
                unit_cat_array[unit_id] = unit.annotation
            unit_dists_array[-1, :] = np.array([np.NaN for _ in range(3)])
            unit_cat_array[-1] = 0
            positions_arrays.append(unit_dists_array)
            category_arrays.append(unit_cat_array)
        return positions_arrays, category_arrays

    def build_arrays_alignment(self, alignment: 'Alignment') -> List[np.ndarray]:
        """
        Builds the compact, array-shaped representation of an alignment.
        """
        raise NotImplemented()

    @staticmethod
    @nb.njit
    def get_all_valid_alignments(position_arrays, category_arrays, d_mat, delta_empty):
        sizes = np.zeros(len(position_arrays)).astype(np.int32)
        for annotator_id in range(len(position_arrays)):
            sizes[annotator_id] = len(position_arrays[annotator_id])
        disorders = nb.typed.List([np.float64(0) for _ in range(0)])
        alignments = nb.typed.List([sizes for _ in range(0)])

        unitary_alignment = np.zeros(len(position_arrays)).astype(np.int32)
        while True:
            # for each tuple (corresponding to a unitary alignment), compute disorder
            disorder = 0
            for annot_a in np.arange(len(unitary_alignment)):
                for annot_b in np.arange(annot_a + 1, len(unitary_alignment)):
                    # this block looks a bit slow (because of all the variables
                    # declarations) but should be fairly sped up automatically
                    # by the LLVM optimization pass
                    unit_a_id, unit_b_id = unitary_alignment[annot_a], unitary_alignment[annot_b]
                    disorder += d_mat(position_arrays[annot_a][unit_a_id], category_arrays[annot_a][unit_a_id],
                                      position_arrays[annot_b][unit_b_id], category_arrays[annot_b][unit_b_id])
            disorder /= (len(unitary_alignment) * (len(unitary_alignment) - 1) // 2)
            if disorder <= len(unitary_alignment) * delta_empty:
                disorders.append(disorder)
                alignments.append(unitary_alignment.copy())
            next_tuple(unitary_alignment, sizes)
            if not np.any(unitary_alignment):
                break

        return disorders, alignments

    def d(self, unit1: 'Unit', unit2: 'Unit'):
        unit1_pos = np.array([unit1.segment.start, unit1.segment.end, unit1.segment.end - unit1.segment.start])
        unit2_pos = np.array([unit2.segment.start, unit2.segment.end, unit2.segment.end - unit2.segment.start])
        unit1_cat = unit1.annotation
        unit2_cat = unit2.annotation
        return self.d_mat(unit1_pos, unit1_cat, unit2_pos, unit2_cat)

    @staticmethod
    def alignments_disorders(*args, **kwargs):
        """Computes the disorder for a batch of unitary alignments."""
        raise NotImplemented()

    def __call__(self, continuum: 'Continuum') -> Tuple[np.ndarray, np.ndarray]:
        position_arrays, category_arrays = self.build_arrays_continuum(continuum)
        disorders, alignments = self.get_all_valid_alignments(position_arrays,
                                                              category_arrays,
                                                              self.d_mat,
                                                              self.delta_empty)
        return np.array(disorders), np.array(alignments)


class PositionalDissimilarity(AbstractDissimilarity):
    def __init__(self, delta_empty: float = 1.0):

        @nb.njit
        def d_mat(unit1_pos: np.ndarray, unit1_cat: str, unit2_pos: np.ndarray, unit2_cat: str):
            if np.isnan(unit1_pos[0]) or np.isnan(unit2_pos[0]):
                return delta_empty
            dist = ((np.abs(unit1_pos[0] - unit2_pos[0]) + np.abs(unit1_pos[1] - unit2_pos[1])) /
                    (unit1_pos[2] + unit2_pos[2]))
            return dist * dist * delta_empty

        super().__init__(delta_empty, d_mat)


class CategoricalDissimilarity(AbstractDissimilarity):
    def __init__(self, delta_empty: float = 1.0, method: str = 'absolute'):

        if method == 'absolute':
            @nb.njit
            def d_mat(unit1_pos, unit1_cat, unit2_pos, unit2_cat):
                if np.isnan(unit1_pos[0]) or np.isnan(unit2_pos[0]):
                    return delta_empty
                return 0.0 if unit1_cat == unit2_cat else 1.0
        elif method == 'ordinal':
            raise NotImplemented
        elif method == 'levenshtein':
            raise NotImplemented
        else:
            raise ValueError(f"Categorical dissimilarity method '{method}' unknown.")

        super().__init__(delta_empty, d_mat)


class CombinedCategoricalDissimilarity(AbstractDissimilarity):
    def __init__(self, alpha: float = 1, beta: float = 1, delta_empty: float = 1.0, cat_method: str = 'absolute'):
        self.positional_dissim = PositionalDissimilarity(delta_empty)
        self.categorical_dissim = CategoricalDissimilarity(delta_empty, method='absolute')

        super().__init__(delta_empty, None)

        self._alpha = alpha
        self._beta = beta

        self.__reset_dmat()

    def __reset_dmat(self):
        delta_empty = self.delta_empty
        alpha = self.alpha
        beta = self.beta
        pos = self.positional_dissim.d_mat
        cat = self.categorical_dissim.d_mat

        @nb.njit
        def d_mat(unit1_pos: np.ndarray, unit1_cat: str, unit2_pos: np.ndarray, unit2_cat: str):
            if np.isnan(unit1_pos[0]) or np.isnan(unit2_pos[0]):
                return delta_empty
            return (alpha * pos(unit1_pos, unit1_cat, unit2_pos, unit2_cat) +
                    beta * cat(unit1_pos, unit1_cat, unit2_pos, unit2_cat))

        self.d_mat = d_mat

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        self.__reset_dmat()

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        self.__reset_dmat()



