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
from .numba_utils import iter_tuples

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
        categories = continuum.categories

        positions_arrays = nb.typed.List()
        category_arrays = nb.typed.List()
        for annotator_id, (annotator, units) in enumerate(continuum._annotations.items()):
            # dim x : segment
            # dim y : (start, end, dur) / annotation
            unit_dists_array = np.zeros((len(units) + 1, 3), dtype=np.float64)
            unit_cat_array = np.zeros(len(units) + 1, dtype=np.int32)
            for unit_id, unit in enumerate(units):
                unit_dists_array[unit_id][0] = unit.segment.start
                unit_dists_array[unit_id][1] = unit.segment.end
                unit_dists_array[unit_id][2] = unit.segment.duration
                unit_cat_array[unit_id] = categories.index(unit.annotation)
            unit_dists_array[-1, :] = np.array([-1 for _ in range(3)])
            unit_cat_array[-1] = -1
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
    def get_all_valid_alignments(position_arrays: nb.typed.List, category_arrays: nb.typed.List,
                                 d_mat,
                                 delta_empty: float):
        chunk_size = (10**6) // 8
        nb_annotators = len(position_arrays)
        c2n = (nb_annotators * (nb_annotators - 1) // 2)
        criterium = c2n * delta_empty * nb_annotators

        sizes = np.zeros(nb_annotators).astype(np.int32)
        for annotator_id in range(nb_annotators):
            sizes[annotator_id] = len(position_arrays[annotator_id])

        disorders = np.zeros(chunk_size, dtype=np.float64)
        alignments = np.zeros((chunk_size, nb_annotators), dtype=np.int32)

        i_chosen = 0
        for unitary_alignment in iter_tuples(sizes):
            # for each tuple (corresponding to a unitary alignment), compute disorder
            disorder = 0
            for annot_a in range(nb_annotators):
                for annot_b in range(annot_a + 1, nb_annotators):
                    # this block looks a bit slow (because of all the variables
                    # declarations) but should be fairly sped up automatically
                    # by the LLVM optimization pass
                    unit_a_id, unit_b_id = unitary_alignment[annot_a], unitary_alignment[annot_b]
                    pos_a, cat_a = position_arrays[annot_a][unit_a_id], category_arrays[annot_a][unit_a_id]
                    pos_b, cat_b = position_arrays[annot_b][unit_b_id], category_arrays[annot_b][unit_b_id]
                    if cat_a == -1 or cat_b == -1:
                        disorder += delta_empty
                    else:
                        disorder += d_mat(pos_a, cat_a, pos_b, cat_b)
            if disorder <= criterium:
                disorders[i_chosen] = disorder
                alignments[i_chosen] = unitary_alignment
                i_chosen += 1
                if i_chosen == chunk_size:
                    disorders = np.concatenate((disorders, np.zeros(chunk_size // 2, dtype=np.float64)))
                    alignments = np.concatenate((alignments, np.zeros((chunk_size // 2, nb_annotators), dtype=np.int32)))
                    chunk_size += chunk_size // 2

        disorders, alignments = disorders[:i_chosen], alignments[:i_chosen]
        disorders /= c2n
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
        def d_mat(unit1_pos: np.ndarray, unit1_cat: int, unit2_pos: np.ndarray, unit2_cat: int):
            dist = ((np.abs(unit1_pos[0] - unit2_pos[0]) + np.abs(unit1_pos[1] - unit2_pos[1])) /
                    (unit1_pos[2] + unit2_pos[2]))
            return dist * dist * delta_empty

        super().__init__(delta_empty, d_mat)


class CategoricalDissimilarity(AbstractDissimilarity):
    def __init__(self, delta_empty: float = 1.0, method: str = 'absolute'):

        if method == 'absolute':
            @nb.njit
            def d_mat(unit1_pos, unit1_cat, unit2_pos, unit2_cat):
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
        def d_mat(unit1_pos: np.ndarray, unit1_cat: int, unit2_pos: np.ndarray, unit2_cat: int):
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



