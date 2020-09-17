#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CoML

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
# Rachid RIAD & Hadrien TITEUX
"""
##########
Dissimilarity
##########

"""
from typing import List, Optional, TYPE_CHECKING

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from sortedcontainers import SortedSet

from pygamma.numba_utils import binom

if TYPE_CHECKING:
    from .continuum import Continuum


class AbstractDissimilarity:

    def __init__(self, delta_empty: float):
        self.delta_empty = np.float32(delta_empty)

    @staticmethod
    def tuples_disorders(*args, **kwargs):
        raise NotImplemented()

    def __call__(self, *args) -> np.ndarray:
        raise NotImplemented()


class CategoricalDissimilarity(AbstractDissimilarity):
    """Categorical Dissimilarity

    Parameters
    ----------
    list_categories :
        list of categories
    categorical_dissimilarity_matrix :
        Dissimilarity matrix to compute
    delta_empty :
        empty dissimilarity value
    """

    def __init__(self,
                 list_categories: List[str],
                 categorical_dissimilarity_matrix: Optional[np.ndarray] = None,
                 delta_empty: float = 1):
        super().__init__(delta_empty)

        self.categories = SortedSet(list_categories)
        assert len(list_categories) == len(self.categories)
        self.categories_nb = len(self.categories)
        # TODO: make sure that the categorical dissim matrix matches the categories order
        self.cat_matrix = categorical_dissimilarity_matrix
        if self.cat_matrix is None:
            # building the default dissimilarity matrix
            self.cat_matrix = (np.ones((self.categories_nb, self.categories_nb))
                               - np.eye(self.categories_nb))
        else:
            # sanity checks on the categorical_dissimilarity_matrix
            assert isinstance(self.cat_matrix, np.ndarray)
            assert np.all(self.cat_matrix <= 1)
            assert np.all(0 <= self.cat_matrix)
            assert np.all(self.cat_matrix ==
                          categorical_dissimilarity_matrix.T)
        self.cat_matrix = self.cat_matrix.astype(np.float32)

    def build_categories_arrays(self, continuum: 'Continuum'):
        categories_arrays = nb.typed.List()
        for annotator_id, (annotator, units) in enumerate(continuum._annotations.items()):
            cat_array = np.zeros(len(units) + 1).astype(np.int16)
            for unit_id, (segment, unit) in enumerate(units.items()):
                if unit.annotation is None:
                    raise ValueError(f"In segment {segment} for annotator {annotator}: annotation cannot be None")
                try:
                    cat_array[unit_id] = self.categories.index(unit.annotation)
                except ValueError:
                    raise ValueError(f"In segment {segment} for annotator {annotator}: "
                                     f"annotation of category {unit.category} is not in set {set(self.categories)} "
                                     f"of allowed categories")
            cat_array[-1] = -1
            categories_arrays.append(cat_array)
        return categories_arrays

    def plot_categorical_dissimilarity_matrix(self):
        fig, ax = plt.subplots()
        im = plt.imshow(
            self.cat_matrix,
            extent=[0, self.categories_nb, 0, self.categories_nb])
        ax.figure.colorbar(im, ax=ax)
        plt.xticks([el + 0.5 for el in range(self.categories_nb)],
                   self.categories)
        plt.yticks([el + 0.5 for el in range(self.categories_nb)],
                   self.categories[::-1])
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor")
        ax.xaxis.set_ticks_position('top')
        plt.show()

    def __call__(self, *args, **kwargs) -> np.ndarray:
        raise NotImplemented()


class PositionalDissimilarity(AbstractDissimilarity):
    """Positional Dissimilarity

    Parameters
    ----------
    delta_empty : float
        empty dissimilarity value
    """

    def build_positions_arrays(self, continuum: 'Continuum'):
        positions_arrays = nb.typed.List()
        for annotator_id, (annotator, units) in enumerate(continuum._annotations.items()):
            # dim x : segment
            # dim y : (start, end, dur)
            unit_dists_array = np.zeros((len(units) + 1, 3)).astype(np.float32)
            for unit_id, (segment, unit) in enumerate(units.items()):
                unit_dists_array[unit_id][0] = segment.start
                unit_dists_array[unit_id][1] = segment.end
                unit_dists_array[unit_id][2] = segment.duration
            unit_dists_array[-1, :] = np.array([np.NaN for _ in range(3)])
            positions_arrays.append(unit_dists_array)
        return positions_arrays

    @staticmethod
    @nb.njit(nb.float32[:](nb.int32[:, :],
                           nb.types.ListType(nb.float32[:, ::1]),
                           nb.float32))
    def alignments_disorders(units_tuples_ids: np.ndarray,
                             units_positions: List[np.ndarray],
                             delta_empty: float = 1.0):
        disorders = np.zeros(len(units_tuples_ids), dtype=np.float32)
        annotator_id = np.arange(units_tuples_ids.shape[1]).astype(np.int16)
        for tuple_id in np.arange(len(units_tuples_ids)):
            # for each tuple (corresponding to a unitary alignment), compute disorder
            for annot_a in annotator_id:
                for annot_b in annotator_id[annot_a + 1:]:
                    # this block looks a bit slow (because of all the variables
                    # declarations) but should be fairly sped up automatically
                    # by the LLVM optimization pass
                    unit_a_id, unit_b_id = units_tuples_ids[tuple_id, annot_a], units_tuples_ids[tuple_id, annot_b]
                    unit_a, unit_b = units_positions[annot_a][unit_a_id], units_positions[annot_b][unit_b_id]
                    if np.isnan(unit_a[0]) or np.isnan(unit_b[0]):
                        disorders[tuple_id] += delta_empty
                    else:
                        ends_diff = np.abs(unit_a[1] - unit_b[1])
                        starts_diff = np.abs(unit_a[0] - unit_b[0])
                        # unit[2] is the duration
                        distance_pos = (starts_diff + ends_diff) / (unit_a[2] + unit_b[2])
                        distance_pos = distance_pos * distance_pos * delta_empty
                        disorders[tuple_id] += distance_pos

        disorders = disorders / binom(units_tuples_ids.shape[1], 2)

        return disorders

    def __call__(self, positional_arrays: np.ndarray,
                 units_positions: List[np.ndarray]) -> np.ndarray:
        return self.alignments_disorders(units_tuples_ids=positional_arrays,
                                         units_positions=units_positions)


class CombinedCategoricalDissimilarity(AbstractDissimilarity):
    """Combined Dissimilarity

    Parameters
    ----------
    list_categories :
        list of categories
    categorical_dissimilarity_matrix :
        Dissimilarity matrix to compute
    delta_empty :
        empty dissimilarity value
    alpha:
        coefficient weighting the positional dissimilarity value
    beta:
        coefficient weighting the categorical dissimilarity value
    """

    def __init__(self,
                 list_categories: List[str],
                 alpha: int = 3,
                 beta: int = 1,
                 delta_empty: float = 1,
                 categorical_dissimilarity_matrix=None):
        super().__init__(delta_empty)
        self.categorical_dissim = CategoricalDissimilarity(
            list_categories,
            categorical_dissimilarity_matrix,
            delta_empty)
        self.positional_dissim = PositionalDissimilarity(delta_empty)
        self.alpha = np.float32(alpha)
        self.beta = np.float32(beta)

    def build_positions_arrays(self, continuum: 'Continuum'):
        return self.positional_dissim.build_positions_arrays(continuum)

    def build_categories_arrays(self, continuum: 'Continuum'):
        return self.categorical_dissim.build_categories_arrays(continuum)

    @staticmethod
    @nb.njit(nb.float32[:](nb.int32[:, :],
                           nb.types.ListType(nb.float32[:, ::1]),
                           nb.types.ListType(nb.int16[::1]),
                           nb.float32,
                           nb.float32[:, :],
                           nb.float32,
                           nb.float32))
    def alignments_disorders(units_tuples_ids: np.ndarray,
                             units_positions: List[np.ndarray],
                             units_categories: List[np.ndarray],
                             delta_empty: float,
                             cat_matrix: np.ndarray,
                             alpha: float,
                             beta: float):
        disorders = np.zeros(len(units_tuples_ids), dtype=np.float32)
        annotator_id = np.arange(units_tuples_ids.shape[1]).astype(np.int16)
        for tuple_id in np.arange(len(units_tuples_ids)):
            # for each tuple (corresponding to a unitary alignment), compute disorder
            for annot_a in annotator_id:
                for annot_b in annotator_id[annot_a + 1:]:
                    # this block looks a bit slow (because of all the variables
                    # declarations) but should be fairly sped up automatically
                    # by the LLVM optimization pass
                    unit_a_id, unit_b_id = units_tuples_ids[tuple_id, annot_a], units_tuples_ids[tuple_id, annot_b]
                    unit_a, unit_b = units_positions[annot_a][unit_a_id], units_positions[annot_b][unit_b_id]
                    cat_a, cat_b = units_categories[annot_a][unit_a_id], units_categories[annot_b][unit_b_id]
                    if np.isnan(unit_a[0]) or np.isnan(unit_b[0]):
                        disorders[tuple_id] += delta_empty
                    else:
                        ends_diff = np.abs(unit_a[1] - unit_b[1])
                        starts_diff = np.abs(unit_a[0] - unit_b[0])
                        # unit[2] is the duration
                        distance_pos = (starts_diff + ends_diff) / (unit_a[2] + unit_b[2])
                        distance_pos = distance_pos * distance_pos * delta_empty
                        distance_cat = cat_matrix[cat_a, cat_b] * delta_empty
                        disorders[tuple_id] += distance_pos * alpha + distance_cat * beta

        disorders = disorders / binom(units_tuples_ids.shape[1], 2)

        return disorders

    def __call__(self, units_tuples: np.ndarray,
                 units_positions: List[np.ndarray],
                 units_categories: List[np.ndarray]) -> np.ndarray:
        return self.alignments_disorders(units_tuples_ids=units_tuples,
                                         units_positions=units_positions,
                                         units_categories=units_categories,
                                         delta_empty=self.delta_empty,
                                         cat_matrix=self.categorical_dissim.cat_matrix,
                                         alpha=self.alpha,
                                         beta=self.beta)
