#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2019 CNRS

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
# Rachid RIAD
"""
##########
Alignement and disorder
##########

"""
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from functools import lru_cache
from itertools import product
from typing import List, Tuple

import cvxpy as cp
import numpy as np
from pyannote.core import Segment
from scipy.special import binom

from pygamma.continuum import Continuum, Annotator
from pygamma.dissimilarity import AbstractDissimilarity, AbstractCombinedDissimilarity


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class SetPartitionError(Error):
    """Exception raised for errors in the partition of units of continuum.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class AbstractAlignment(metaclass=ABCMeta):

    def __init__(
            self,
            continuum: Continuum,
            combined_dissimilarity: AbstractCombinedDissimilarity,
    ):

        self.continuum = continuum
        self.combined_dissim = combined_dissimilarity

    @property
    @abstractmethod
    def disorder(self) -> float:
        raise NotImplemented()


class UnitaryAlignment(AbstractAlignment):
    """Unitary Alignment
    Parameters
    ----------
    continuum :
        Continuum where the unitary alignment is from
    n_tuple :
        n-tuple where n is the number of annotators of the continuum
        This is a list of (annotator, segment) couples
    combined_dissimilarity :
        combined_dissimilarity
    """

    def __init__(
            self,
            continuum: Continuum,
            n_tuple: List[Tuple[Annotator, Segment]],
            combined_dissimilarity,
    ):
        super().__init__(continuum, combined_dissimilarity)
        self.n_tuple = n_tuple
        assert len(n_tuple) == len(self.continuum)

    @property
    @lru_cache(maxsize=None)
    def disorder(self):
        """Compute the disorder for the unitary alignement
        >>> unitary_alignement.compute_disorder() = ...
        Based on formula (6) of the original paper
        Note:
        unit is the equivalent of segment in pyannote
        """
        disorder = 0.0
        num_couples = 0
        for idx, (annotator_u, unit_u) in enumerate(self.n_tuple):
            for (annotator_v, unit_v) in self.n_tuple[idx + 1:]:
                # This is not as the paper formula (6)...
                # for (annotator_v, unit_v) in self.n_tuple:
                if unit_u is None or unit_v is None:
                    disorder += self.combined_dissim.DELTA_EMPTY
                else:
                    disorder += self.combined_dissim[
                        (unit_u, unit_v),
                        (self.continuum[annotator_u][unit_u],
                         self.continuum[annotator_v][unit_v])
                    ]
                num_couples += 1
        disorder = disorder / binom(len(self.n_tuple), 2)
        assert num_couples == binom(len(self.n_tuple), 2)
        return disorder


class UnitaryAlignmentBatch(AbstractAlignment):

    def __init__(
            self,
            continuum: Continuum,
            tuples_list: List[List[Tuple[Annotator, Segment]]],
            combined_dissimilarity,
    ):
        super().__init__(continuum, combined_dissimilarity)
        self.tuples_list = tuples_list

    def disorder(self) -> np.ndarray:
        # building batch units lists
        timings_tuples_list = []
        annots_tuples_list = []
        tuples_idx_bounds = list()
        tuples_num_couples = list()
        counter = 0
        for tuple_idx, n_tuple in enumerate(self.tuples_list):
            # all tuples in tuples_data are not None
            tuples_data = [(unit, self.continuum[annotator][unit])
                           for annotator, unit in n_tuple if unit is not None]
            idx_start = counter
            tuples_num_couples.append(binom(len(n_tuple), 2))
            for idx, (unit_u, annot_u) in enumerate(tuples_data):
                for (unit_v, annot_v) in tuples_data[idx + 1:]:
                    timings_tuples_list.append((unit_u, unit_v))
                    annots_tuples_list.append((annot_u, annot_v))
                    counter += 1
            idx_end = counter
            tuples_idx_bounds.append((idx_start, idx_end))
        dissimilarities = self.combined_dissim.batch_compute((timings_tuples_list,
                                                              annots_tuples_list))
        tuples_disorders = []
        for idx, (start, end) in enumerate(tuples_idx_bounds):
            tuple_dissims = dissimilarities[start:end]
            # calculating the number of empty dissimilarities
            empty_count = tuples_num_couples[idx] - (end - start)
            # computing the tuple's total disorder using the dissimilarities
            # and the empty couples count
            tuple_disorder = (tuple_dissims.sum()
                              + empty_count * self.combined_dissim.DELTA_EMPTY)
            tuple_disorder = tuple_disorder / tuples_num_couples[idx]
            tuples_disorders.append(tuple_disorder)

        return np.array(tuples_disorders)


class Alignment(AbstractAlignment):
    """Alignment
    Parameters
    ----------
    continuum :
        Continuum where the unitary alignment is from
    set_unitary_alignments :
        set of unitary alignments that make a partition of the set of
        units/segments
    combined_dissimilarity :
        combined_dissimilarity
    """

    def __init__(
            self,
            continuum: Continuum,
            set_unitary_alignments: List[UnitaryAlignment],
            combined_dissimilarity: AbstractCombinedDissimilarity,
    ):
        super().__init__(continuum, combined_dissimilarity)
        self.set_unitary_alignments = set_unitary_alignments

        # set partition tests for the unitary alignments
        for annotator in self.continuum.iterannotators():
            for unit in self.continuum[annotator].itersegments():
                found = 0
                for unitary_alignment in self.set_unitary_alignments:
                    if [annotator, unit] in unitary_alignment.n_tuple:
                        found += 1
                if found == 0:
                    raise SetPartitionError(
                        '{} {} not in the set of unitary alignments'.format(
                            annotator, unit))
                elif found > 1:
                    raise SetPartitionError('{} {} assigned twice'.format(
                        annotator, unit))

    @property
    def num_alignments(self):
        return len(self.set_unitary_alignments)

    @property
    @lru_cache(maxsize=None)
    def disorder(self):
        """Compute the disorder as the mean of its child unitary alignments
        >>> unitary_alignment.compute_disorder() = ...
        Based on formula (6) of the original paper
        Note:
        unit is the equivalent of segment in pyannote
        """
        disorder = 0.0
        for unitary_alignment in self.set_unitary_alignments:
            disorder += unitary_alignment.disorder
        return disorder / self.num_alignments


class BestAlignment(AbstractAlignment):
    """Alignement
    Parameters
    ----------
    continuum :
        Continuum where the unitary alignment is from
    combined_dissimilarity :
        combined_dissimilarity
    """

    def __init__(
            self,
            continuum: Continuum,
            combined_dissimilarity: AbstractCombinedDissimilarity,
    ):
        super().__init__(continuum, combined_dissimilarity)

        self.set_unitary_alignments = self.get_unitary_alignments_best()

        # set partition tests for the unitary alignements
        for annotator in self.continuum.iterannotators():
            for unit in self.continuum[annotator].itersegments():
                found = 0
                for unitary_alignment in self.set_unitary_alignments:
                    if [annotator, unit] in unitary_alignment.n_tuple:
                        found += 1
                if found == 0:
                    raise SetPartitionError(
                        '{} {} not in the set of unitary alignements'.format(
                            annotator, unit))
                elif found > 1:
                    raise SetPartitionError('{} {} assigned twice'.format(
                        annotator, unit))

    @property
    def num_alignements(self):
        return len(self.set_unitary_alignments)

    def get_unitary_alignments_best(self):
        set_of_possible_segments = []
        for annotator in self.continuum.iterannotators():
            annot_segments = []
            for segment in self.continuum[annotator].itersegments():
                annot_segments.append((annotator, segment))
            annot_segments.append((annotator, None))
            set_of_possible_segments.append(annot_segments)

        set_of_possible_tuples = list(product(*set_of_possible_segments))
        # computing all disorders for alignments in one big batch
        unit_batch = UnitaryAlignmentBatch(self.continuum,
                                           set_of_possible_tuples,
                                           self.combined_dissim)
        tuples_disorders = unit_batch.disorder()
        # Property section 5.1.1 to reduce initial complexity
        set_of_possible_unitary_alignments = []
        for idx, n_tuple in enumerate(set_of_possible_tuples):
            # Property section 5.1.1 to reduce initial complexity
            disorder = tuples_disorders[idx]
            if disorder < len(self.continuum) * (
                    self.combined_dissim.DELTA_EMPTY):
                unitary_alignment = UnitaryAlignment(self.continuum,
                                                     n_tuple,
                                                     self.combined_dissim)
                set_of_possible_unitary_alignments.append(unitary_alignment)

        # Definition of the integer linear program
        num_possible_unitary_alignments = len(
            set_of_possible_unitary_alignments)
        # x is the alignment variable: contains the best alignment once the
        # problem has been solved
        x = cp.Variable(shape=num_possible_unitary_alignments, boolean=True)
        d = np.array([
            unitary_alignment.disorder
            for unitary_alignment in set_of_possible_unitary_alignments
        ])

        # Constraints matrix
        A = np.zeros((self.continuum.num_units,
                      num_possible_unitary_alignments))

        curr_idx = 0
        # fill unitary alignments matching with units
        for annotator in self.continuum.iterannotators():
            for unit in list(self.continuum[annotator].itersegments()):
                for idx_ua, unitary_alignment in enumerate(
                        set_of_possible_unitary_alignments):
                    if [annotator, unit] in unitary_alignment.n_tuple:
                        A[curr_idx][idx_ua] = 1
                curr_idx += 1
        obj = cp.Minimize(d.T * x)
        constraints = [cp.matmul(A, x) == 1]
        prob = cp.Problem(obj, constraints)

        # we don't actually need the optimal disorder value to build the
        # best alignment
        optimal_disorder = prob.solve()
        set_unitary_alignments = []

        # compare with 0.9 as cvxpy returns 1.000 or small values 10e-14
        for idx, chosen_unitary_alignment in enumerate(list(x.value > 0.9)):
            if chosen_unitary_alignment:
                set_unitary_alignments.append(
                    set_of_possible_unitary_alignments[idx])
        return set_unitary_alignments

    @property
    @lru_cache(maxsize=None)
    def disorder(self):
        """Compute the disorder for the unitary alignement
        >>> unitary_alignment.compute_disorder() = ...
        Based on formula (6) of the original paper
        Note:
        unit is the equivalent of segment in pyannote
        """
        disorder = 0.0
        for unitary_alignment in self.set_unitary_alignments:
            disorder += unitary_alignment.disorder
        return disorder / self.num_alignements
