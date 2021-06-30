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

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from sortedcontainers import SortedSet

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


cat_arguments = {}
try:
    def cat_levenshtein(cat1: str, cat2: str) -> float:
        import Levenshtein
        return Levenshtein.distance(cat1, cat2) / max(len(cat1), len(cat2))
    cat_arguments["levenshtein"] = cat_levenshtein
except ImportError:
    pass


def cat_default(cat1: str, cat2: str) -> float:
    return float(cat1 != cat2)


cat_arguments["default"] = cat_default


def cat_ord(cat1: str, cat2: str) -> float:
    if not (cat1.isnumeric() and cat2.isnumeric()):
        raise ValueError("Error : tried to compute ordinal categorical dissimilarity"
                         f"but categories are non-numeric (category {cat1} or {cat2})")
    return abs(float(int(cat1) - int(cat2)))


cat_arguments["ordinal"] = cat_ord


class AbstractDissimilarity:

    def __init__(self, delta_empty: float = 1):
        self.delta_empty = np.float32(delta_empty)

    def build_arrays_continuum(self, continuum: 'Continuum') -> List[np.ndarray]:
        """
        Builds the compact, array-shaped representation of a continuum.
        """
        raise NotImplemented()

    def build_arrays_alignment(self, alignment: 'Alignment') -> List[np.ndarray]:
        """
        Builds the compact, array-shaped representation of an alignment.
        """
        raise NotImplemented()

    def d(self, unit_a: 'Unit', unit_b: 'Unit') -> float:
        """
        Returns the disorder between two unit objects, Depending on the type of dissimilaty.
        If unit_a or unit_b is the empty unit, delta_empty is returned.
        d(unit_a, unit_b) = d(unit_b, unit_a) is always True.
        """

    def build_args(self, resource: Union['Alignment', 'Continuum']) -> Tuple:
        """
        Computes a compact, array-shaped representation of units
        needed for fast computation of inter-units disorders
        computed, and set when compute_disorder is called.
        """
        from .continuum import Continuum
        from .alignment import Alignment
        if isinstance(resource, Continuum):
            return self.build_arrays_continuum(resource),
        elif isinstance(resource, Alignment):
            return self.build_arrays_alignment(resource),

    @staticmethod
    def alignments_disorders(*args, **kwargs):
        """Computes the disorder for a batch of unitary alignments."""
        raise NotImplemented()

    def __call__(self, *args) -> np.ndarray:
        raise NotImplemented()


class CategoricalDissimilarity(AbstractDissimilarity):
    """Categorical Dissimilarity

    Parameters
    ----------
    categories : iterable of str
        iterable of N categories
    cat_dissimilarity_matrix : optional, f: (str,str) -> float function representing
        the matrix containing the values between categories. Has to be symetrical (f(x, y) = f(y, x))
        with an empty diagonal (f(x, x) = 0). Defaults to setting all dissimilarities to 1.
        OR
        N,N matrix (np.array) containing dissimilarity values between categories (works best with ordinal
        categories, since categories are indexed in alphabetical order).
    delta_empty : optional, float
        empty dissimilarity value. Defaults to 1.
    """

    def __init__(self,
                 categories: SortedSet,
                 cat_dissimilarity_matrix: Union[Callable[[str, str], float], np.ndarray] = cat_default,
                 delta_empty: float = 1):
        super().__init__(delta_empty)

        self.categories = categories
        self.categories_nb = len(self.categories)

        if isinstance(cat_dissimilarity_matrix, np.ndarray):
            self.cat_matrix = cat_dissimilarity_matrix
        else:
            self.cat_matrix = np.zeros((len(categories), len(categories)))
            max_val = 1.0
            for id1, cat1 in enumerate(categories):
                for id2, cat2 in enumerate(categories):
                    elem = cat_dissimilarity_matrix(cat1, cat2)
                    if elem > max_val:
                        max_val = elem
                    self.cat_matrix[id1, id2] = elem
            self.cat_matrix /= max_val
        # sanity checks on the categorical_dissimilarity_matrix
        assert isinstance(self.cat_matrix, np.ndarray)
        assert np.all(self.cat_matrix <= 1)
        assert np.all(0 <= self.cat_matrix)
        self.cat_matrix = self.cat_matrix.astype(np.float32)

    def build_arrays_continuum(self, continuum: 'Continuum'):
        """
        The continuum's matrix representation for categorical dissimilarity is :
        M(i, j) = id of the category of unit j of annotator i.
        Annotators' order is alphabetical, and units' order is implemented by Unit.__lt__().
        """
        categories_arrays = nb.typed.List()
        for annotator_id, (annotator, units) in enumerate(continuum._annotations.items()):
            cat_array = np.zeros(len(units) + 1).astype(np.int16)
            for unit_id, unit in enumerate(units):
                if unit.annotation is None:
                    raise ValueError(f"In segment {unit.segment} for annotator "
                                     f"{annotator}: annotation cannot be None")
                try:
                    cat_array[unit_id] = self.categories.index(unit.annotation)
                except ValueError:
                    raise ValueError(
                        f"In segment {unit.segment} for annotator {annotator}: "
                        f"annotation of category {unit.annotation} is not in "
                        f"set {self.categories} of allowed categories")
            cat_array[-1] = -1  # We add an empty unit at the end of each line so that the carthesian product uses it
            categories_arrays.append(cat_array)
        return categories_arrays

    def build_arrays_alignment(self, alignment: 'Alignment'):
        """
        The continuum's matrix representation for categorical dissimilarity is :
        M(i, j) = id of the category of the unit from annotator i in unitary alignment j.
        Annotators' order is alphabetical.
        id -1 corresponds to the empty unit.
        """
        cat_arrays = nb.typed.List()
        for _ in range(alignment.num_annotators):
            unit_dists_array = np.zeros(alignment.num_unitary_alignments)
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
                    cat_arrays[annot_id][unit_id] = self.categories.index(unit.annotation)
                except ValueError:
                    raise ValueError(f"In unit {unit} for annotator {annotator}"
                                     f"in unitary alignment {unit_align}: "
                                     f"annotation of category {unit.category} "
                                     f"is not in set {self.categories} "
                                     f"of allowed categories")
        return cat_arrays

    def d(self, unit_a: 'Unit', unit_b: 'Unit'):
        if unit_a is None or unit_b is None:
            return self.delta_empty
        return self.cat_matrix[self.categories.index(unit_a.annotation),
                               self.categories.index(unit_b.annotation)] * self.delta_empty

    def plot_categorical_dissimilarity_matrix(self):
        fig, ax = plt.subplots()
        im = plt.imshow(
            self.cat_matrix,
            extent=[0, self.categories_nb, 0, self.categories_nb])
        ax.figure.colorbar(im, ax=ax)
        categories_indexedlist = [None for _ in range(self.categories_nb)]
        for (id_cat, cat) in enumerate(self.categories):
            categories_indexedlist[id_cat] = cat
        plt.xticks([el + 0.5 for el in range(self.categories_nb)],
                   categories_indexedlist)
        plt.yticks([el + 0.5 for el in range(self.categories_nb)],
                   reversed(categories_indexedlist))
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor")
        ax.xaxis.set_ticks_position('top')
        plt.show()

    def build_args(self, resource: Union['Alignment', 'Continuum']) -> Tuple:
        from .continuum import Continuum
        from .alignment import Alignment
        if isinstance(resource, Continuum):
            return self.build_arrays_continuum(resource),
        elif isinstance(resource, Alignment):
            return self.build_arrays_alignment(resource),


    @staticmethod
    @nb.njit(nb.float32[:](nb.int32[:, :],
                           nb.types.ListType(nb.int16[::1]),
                           nb.float32,
                           nb.float32[:, :]))
    def alignments_disorder(units_tuples_ids: np.ndarray,
                            units_categories: List[np.ndarray],
                            delta_empty: np.float32,
                            cat_matrix: np.ndarray):
        """
        Computes the categorical disorder of a unitary alignment (4.6.1)

        Parameters
        ----------
        units_tuples_ids :
            array of the units tuple in the alignment [(ua1, empty, uc1), (ua2, ub2, empty)...] where ub5 for instance
            is the unit 5 of annotator b
        units_categories :
            array given by CategoricalDissimilarity.build_array_alignment()
        delta_empty : float
            empty dissimilarity value.
        cat_matrix :
            Matrix of the dissimilarity categories (CategoricalDissimilarity.cat_matrix)
        """
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
                    unit_a, unit_b = units_categories[annot_a][unit_a_id], units_categories[annot_b][unit_b_id]
                    if unit_a == -1 or unit_b == -1:
                        disorders[tuple_id] += delta_empty
                    else:
                        distance_pos = cat_matrix[unit_a, unit_b] * delta_empty
                        disorders[tuple_id] += distance_pos

        disorders /= (units_tuples_ids.shape[1] * (units_tuples_ids.shape[1] - 1) // 2)  # averaging by C^2_n = n(n-1)/2
        return disorders

    def __call__(self, units_tuples: np.ndarray,
                 units_categories: List[np.ndarray]) -> np.ndarray:
        return self.alignments_disorder(units_tuples_ids=units_tuples,
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
    def __init__(self, delta_empty: float = 1):
        super().__init__(delta_empty=delta_empty)

    def build_args(self, resource: Union['Alignment', 'Continuum']) -> Tuple:
        from .continuum import Continuum
        from .alignment import Alignment
        if isinstance(resource, Continuum):
            return self.build_arrays_continuum(resource),
        elif isinstance(resource, Alignment):
            return self.build_arrays_alignment(resource),

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
            unit_dists_array = np.zeros((alignment.num_unitary_alignments, 3),
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

    def d(self, unit_a: 'Unit', unit_b: 'Unit') -> float:
        if unit_a is None or unit_b is None:
            return self.delta_empty
        return ((abs(unit_a.segment.start - unit_b.segment.start) + abs(unit_a.segment.end - unit_b.segment.end)) /
                (unit_a.segment.end - unit_a.segment.start + unit_b.segment.end - unit_b.segment.start)) ** 2 *\
            self.delta_empty



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

        # averaging by C^2_n = n(n-1)/2
        disorders /= (units_tuples_ids.shape[1] * (units_tuples_ids.shape[1] - 1) // 2)

        return disorders

    def __call__(self, units_tuples_ids: np.ndarray,
                 units_positions: List[np.ndarray]) -> np.ndarray:
        return self.alignments_disorders(units_tuples_ids=units_tuples_ids,
                                         units_positions=units_positions,
                                         delta_empty=self.delta_empty)


class CombinedCategoricalDissimilarity(AbstractDissimilarity):
    def __init__(self,
                 categories: SortedSet,
                 alpha: float = 1,
                 beta: float = 1,
                 delta_empty: float = 1,
                 cat_dissimilarity_matrix: Union[Callable[[str, str], float], np.ndarray] = cat_default):
        """
        Combined categorical dissimilarity constructor.

        Parameters
        ----------
        categories : list of str
            list of N categories
        cat_dissimilarity_matrix : optional, (str,str) -> float function or np.ndarray
            Function :math:`f` that gives the 'distance' between categories. Has to be symetrical
            (:math:`f(x, y) = f(y, x)`) with an empty diagonal (:math:`f(x, x) = 0`).
            OR
            N,N matrix (np.array) containing dissimilarity values between categories (works best with ordinal
            categories, since categories are indexed in alphabetical order).
        delta_empty : optional, float
            empty dissimilarity value. Defaults to 1.
        alpha: optional float
            coefficient weighting the positional dissimilarity value.
            Defaults to 1.
        beta: optional float
            coefficient weighting the categorical dissimilarity value.
            Defaults to 1.
        """
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
        """
        Computes the disorder of each unitary alignment provided.
        Parameters
        """
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
                    if cat_a == -1 or cat_b == -1:
                        disorders[tuple_id] += delta_empty
                    else:
                        disorders[tuple_id] += \
                            positional_dissim(unit_a, unit_b, delta_empty) * alpha \
                            + cat_matrix[cat_a, cat_b] * delta_empty * beta
        disorders /= (units_tuples_ids.shape[1] * (units_tuples_ids.shape[1] - 1) // 2)  # averaging by C^2_n = n(n-1)/2
        return disorders

    def d(self, unit_a: 'Unit', unit_b: 'Unit') -> float:
        if unit_a is None or unit_b is None:
            return self.delta_empty
        return (self.alpha * self.positional_dissim.d(unit_a, unit_b) +
                self.beta * self.categorical_dissim.d(unit_a, unit_b))

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