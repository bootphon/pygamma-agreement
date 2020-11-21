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
from typing import List, Optional, TYPE_CHECKING, Tuple, Union, Iterable

import numba as nb
import numpy as np
from matplotlib import pyplot as plt

from .numba_utils import binom

if TYPE_CHECKING:
    from .continuum import Continuum
    from .alignment import Alignment


@nb.njit(nb.float32(nb.float32[:], nb.float32[:], nb.float32))
def positional_dissim(unit_a: np.ndarray,
                      unit_b: np.ndarray,
                      delta_empty: float):
    ends_diff = np.abs(unit_a[1] - unit_b[1])
    starts_diff = np.abs(unit_a[0] - unit_b[0])
    # unit[2] is the duration
    distance_pos = (starts_diff + ends_diff) / (unit_a[2] + unit_b[2])
    return distance_pos * distance_pos * delta_empty


class AbstractDissimilarity:

    def __init__(self, delta_empty: float = 1):
        self.delta_empty = np.float32(delta_empty)

    def build_arrays_continuum(self, continuum: 'Continuum') -> List[np.ndarray]:
        """Builds the compact, array-shaped representation of a continuum"""
        raise NotImplemented()

    def build_arrays_alignment(self, alignment: 'Alignment') -> List[np.ndarray]:
        """Builds the compact, array-shaped representation of an alignment"""
        raise NotImplemented()

    def build_args(self, resource: Union['Alignment', 'Continuum']) -> Tuple:
        """Computes a compact, array-shaped representation of units
        needed for fast computation of inter-units disorders
        computed and set when compute_disorder is called"""
        from .continuum import Continuum
        from .alignment import Alignment
        if isinstance(resource, Continuum):
            return self.build_arrays_continuum(resource),
        elif isinstance(resource, Alignment):
            return self.build_arrays_alignment(resource),

    @staticmethod
    def tuples_disorders(*args, **kwargs):
        raise NotImplemented()

    def __call__(self, *args) -> np.ndarray:
        raise NotImplemented()


class CategoricalDissimilarity(AbstractDissimilarity):
    """Categorical Dissimilarity

    Parameters
    ----------
    categories : iterable of str
        iterable of N categories
    cat_dissimilarity_matrix : optional, (N,N) numpy array
        Dissimilarity values between categories. Has to be symetrical
        with an empty diagonal. Defaults to setting all dissimilarities to 1.
    delta_empty : optional, float
        empty dissimilarity value. Defaults to 1.
    """

    def __init__(self,
                 categories: Iterable[str],
                 cat_dissimilarity_matrix: Optional[np.ndarray] = None,
                 delta_empty: float = 1):
        super().__init__(delta_empty)

        categories = list(categories)
        self.categories = set(categories)
        self.categories_dict = {cat: i for i, cat in enumerate(categories)}
        assert len(categories) == len(self.categories)
        self.categories_nb = len(self.categories)
        # TODO: make sure that the categorical dissim matrix matches the categories order
        self.cat_matrix = cat_dissimilarity_matrix
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
                          cat_dissimilarity_matrix.T)
        self.cat_matrix = self.cat_matrix.astype(np.float32)

    def build_arrays_continuum(self, continuum: 'Continuum'):
        categories_arrays = nb.typed.List()
        for annotator_id, (annotator, units) in enumerate(continuum._annotations.items()):
            cat_array = np.zeros(len(units) + 1).astype(np.int16)
            for unit_id, unit in enumerate(units):
                if unit.annotation is None:
                    raise ValueError(f"In segment {unit.segment} for annotator "
                                     f"{annotator}: annotation cannot be None")
                try:
                    cat_array[unit_id] = self.categories_dict[unit.annotation]
                except ValueError:
                    raise ValueError(
                        f"In segment {unit.segment} for annotator {annotator}: "
                        f"annotation of category {unit.category} is not in "
                        f"set {set(self.categories)} of allowed categories")
            cat_array[-1] = -1
            categories_arrays.append(cat_array)
        return categories_arrays

    def build_arrays_alignment(self, alignment: 'Alignment'):
        cat_arrays = nb.typed.List()
        for _ in range(alignment.num_annotators):
            unit_dists_array = np.zeros(alignment.num_alignments)
            cat_arrays.append(unit_dists_array.astype(np.int16))
        for unit_id, unit_align in enumerate(alignment.unitary_alignments):
            for annot_id, (annotator, unit) in enumerate(unit_align.n_tuple):
                if unit is None:
                    cat_arrays[annot_id][unit_id] = -1
                    continue

                if unit.annotation is None:
                    raise ValueError(f"In unit {unit} in unitary alignment "
                                     f"{unit_align}: annotation cannot be None")
                try:
                    cat_arrays[annot_id][unit_id] = self.categories_dict[unit.annotation]
                except ValueError:
                    raise ValueError(f"In unit {unit} for annotator {annotator}"
                                     f"in unitary alignment {unit_align}: "
                                     f"annotation of category {unit.category} "
                                     f"is not in set {set(self.categories)} "
                                     f"of allowed categories")
        return cat_arrays

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

    @staticmethod
    @nb.njit(nb.float32(nb.int32[:, :],
                        nb.types.ListType(nb.float32[:, ::1]),
                        nb.types.ListType(nb.int16[::1]),
                        nb.float32,
                        nb.float32[:, :]))
    def alignments_disorder(units_tuples_ids: np.ndarray,
                            units_positions: List[np.ndarray],
                            units_categories: List[np.ndarray],
                            delta_empty: np.float32,
                            cat_matrix: np.ndarray):
        disorder, weight_tot = 0.0, 0.0
        annotator_id = np.arange(units_tuples_ids.shape[1]).astype(np.int16)
        couples_count = binom(units_tuples_ids.shape[1], 2)
        for tuple_id in np.arange(len(units_tuples_ids)):
            real_couples_count = 0
            couple_id = 0
            # weight and categorical dissim vectors for all categories
            weights_confidence = np.zeros(couples_count, dtype=np.float32)
            dissim = np.zeros(couples_count, dtype=np.float32)
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
                        distance_cat = delta_empty
                        distance_pos = delta_empty
                    else:
                        real_couples_count += 1
                        distance_pos = positional_dissim(unit_a, unit_b, delta_empty)
                        distance_cat = cat_matrix[cat_a, cat_b] * delta_empty
                    dissim[couple_id] = distance_cat
                    weights_confidence[couple_id] = max(0, 1 - distance_pos)
                    couple_id += 1
            weight_base = 1 / (real_couples_count - 1)
            weight_tot += weights_confidence.sum() * weight_base
            disorder += (weights_confidence * dissim).sum() * weight_base

        return disorder / weight_tot

    def __call__(self, units_tuples: np.ndarray,
                 units_positions: List[np.ndarray],
                 units_categories: List[np.ndarray]) -> np.ndarray:
        return self.alignments_disorder(units_tuples_ids=units_tuples,
                                        units_positions=units_positions,
                                        units_categories=units_categories,
                                        delta_empty=self.delta_empty,
                                        cat_matrix=self.cat_matrix)


class PositionalDissimilarity(AbstractDissimilarity):
    """Positional Dissimilarity

    Parameters
    ----------
    delta_empty : float
        empty dissimilarity value
    """

    def build_arrays_continuum(self, continuum: 'Continuum'):
        positions_arrays = nb.typed.List()
        for annotator_id, (annotator, units) in enumerate(continuum._annotations.items()):
            # dim x : segment
            # dim y : (start, end, dur)
            unit_dists_array = np.zeros((len(units) + 1, 3)).astype(np.float32)
            for unit_id, unit in enumerate(units):
                unit_dists_array[unit_id][0] = unit.segment.start
                unit_dists_array[unit_id][1] = unit.segment.end
                unit_dists_array[unit_id][2] = unit.segment.duration
            unit_dists_array[-1, :] = np.array([np.NaN for _ in range(3)])
            positions_arrays.append(unit_dists_array)
        return positions_arrays

    def build_arrays_alignment(self, alignment: 'Alignment'):
        positions_arrays = nb.typed.List()
        null_unit_arr = np.array([np.NaN] * 3, dtype=np.float32)

        for _ in range(alignment.num_annotators):
            unit_dists_array = np.zeros((alignment.num_alignments, 3),
                                        dtype=np.float32)
            positions_arrays.append(unit_dists_array)
        for unit_id, unit_align in enumerate(alignment.unitary_alignments):
            # dim x : segment
            # dim y : (start, end, dur)
            for annot_it, (_, unit) in enumerate(unit_align.n_tuple):
                if unit is None:
                    positions_arrays[annot_it][unit_id] = null_unit_arr
                    continue

                positions_arrays[annot_it][unit_id][0] = unit.segment.start
                positions_arrays[annot_it][unit_id][1] = unit.segment.end
                positions_arrays[annot_it][unit_id][2] = unit.segment.duration
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
                        distance_pos = positional_dissim(unit_a, unit_b, delta_empty)
                        disorders[tuple_id] += distance_pos

        disorders = disorders / binom(units_tuples_ids.shape[1], 2)

        return disorders

    def __call__(self, units_tuples_ids: np.ndarray,
                 units_positions: List[np.ndarray]) -> np.ndarray:
        return self.alignments_disorders(units_tuples_ids=units_tuples_ids,
                                         units_positions=units_positions,
                                         delta_empty=self.delta_empty)


class CombinedCategoricalDissimilarity(AbstractDissimilarity):
    def __init__(self,
                 categories: List[str],
                 alpha: float = 3,
                 beta: float = 1,
                 delta_empty: float = 1,
                 cat_dissimilarity_matrix=None):
        super().__init__(delta_empty)
        assert alpha >= 0
        assert beta >= 0
        self.categorical_dissim = CategoricalDissimilarity(
            categories,
            cat_dissimilarity_matrix,
            delta_empty)
        self.positional_dissim = PositionalDissimilarity(delta_empty)
        self.alpha = np.float32(alpha)
        self.beta = np.float32(beta)

    """Combined Dissimilarity

    Parameters
    ----------
    categories : list of str
        list of N categories
    cat_dissimilarity_matrix : optional, (N,N) numpy array
        Dissimilarity values between categories. Has to be symetrical 
        with an empty diagonal. Defaults to setting all dissimilarities to 1.
    delta_empty : optional, float
        empty dissimilarity value. Defaults to 1.
    alpha: optional float
        coefficient weighting the positional dissimilarity value.
        Defaults to 1.
    beta: optional float
        coefficient weighting the categorical dissimilarity value.
        Defaults to 1.
    """

    def build_args(self, resource: Union['Alignment', 'Continuum']) -> Tuple:
        from .continuum import Continuum
        from .alignment import Alignment
        if isinstance(resource, Continuum):
            return (self.positional_dissim.build_arrays_continuum(resource),
                    self.categorical_dissim.build_arrays_continuum(resource))
        elif isinstance(resource, Alignment):
            return (self.positional_dissim.build_arrays_alignment(resource),
                    self.categorical_dissim.build_arrays_alignment(resource))

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
                        distance_pos = positional_dissim(unit_a, unit_b, delta_empty)
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
