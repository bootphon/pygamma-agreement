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
from itertools import product
from typing import List, Tuple
import itertools

import cvxpy as cp
import numpy as np
from multiprocess.pool import Pool
from pyannote.core import Segment
from scipy.special import binom
from tornado.process import cpu_count

from pygamma.continuum import Continuum, Annotator
from pygamma.dissimilarity import AbstractCombinedDissimilarity, AbstractDissimilarity


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
            dissimilarity: AbstractDissimilarity,
    ):
        self.continuum = continuum
        self.combined_dissim = dissimilarity

    @property
    @abstractmethod
    def disorder(self) -> float:
        """Compute the disorder for the alignement

        >>> aligment.disorder()
        ... 0.123

        """

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
    dissimilarity :
        combined_dissimilarity
    """

    def __init__(
            self,
            continuum: Continuum,
            n_tuple: List[Tuple[Annotator, Segment]],
            dissimilarity,
    ):
        super().__init__(continuum, dissimilarity)
        self.n_tuple = n_tuple
        assert len(n_tuple) == len(self.continuum)

    @property
    def disorder(self):
        disorder = 0.0
        num_couples = 0
        for idx, (annotator_u, seg_u) in enumerate(self.n_tuple):
            for (annotator_v, seg_v) in self.n_tuple[idx + 1:]:
                # This is not as the paper formula (6)...
                # for (annotator_v, unit_v) in self.n_tuple:
                if seg_u is None or seg_v is None:
                    disorder += self.combined_dissim.DELTA_EMPTY
                else:
                    disorder += self.combined_dissim[
                        (seg_u, self.continuum[annotator_u][seg_u]),
                        (seg_v, self.continuum[annotator_v][seg_v]),
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
            dissimilarity,
    ):
        super().__init__(continuum, dissimilarity)
        self.tuples_list = tuples_list

    @property
    def disorder(self) -> np.ndarray:
        # building batch units list
        units_list = []
        tuples_idx_bounds = list()
        units_bounds_idx = list()
        counter = 0
        for tuple_idx, n_tuple in enumerate(self.tuples_list):
            # all tuples in tuples_data are not None
            tuples_data = [(segment, self.continuum[annotator][segment])
                           for annotator, segment in n_tuple
                           if segment is not None]
            idx_start = counter
            units_bounds_idx.append(binom(len(n_tuple), 2))
            for idx, (seg_u, annot_u) in enumerate(tuples_data):
                for (seg_v, annot_v) in tuples_data[idx + 1:]:
                    units_list.append(((seg_u, annot_u),
                                       (seg_v, annot_v)))
                    counter += 1
            idx_end = counter
            tuples_idx_bounds.append((idx_start, idx_end))
        dissimilarities = self.combined_dissim.batch_compute(units_list)
        tuples_disorders = []
        for idx, (start, end) in enumerate(tuples_idx_bounds):
            tuple_dissims = dissimilarities[start:end]
            # calculating the number of empty dissimilarities
            empty_count = units_bounds_idx[idx] - (end - start)
            # computing the tuple's total disorder using the dissimilarities
            # and the empty couples count
            tuple_disorder = (tuple_dissims.sum()
                              + empty_count * self.combined_dissim.DELTA_EMPTY)
            tuple_disorder = tuple_disorder / units_bounds_idx[idx]
            tuples_disorders.append(tuple_disorder)

        return np.array(tuples_disorders)


def get_disorder(unitary_alignment: UnitaryAlignment):
    return unitary_alignment, unitary_alignment.disorder

def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk

class Alignment(AbstractAlignment):
    """Alignment

    Parameters
    ----------
    continuum :
        Continuum where the alignment is from
    set_unitary_alignments :
        set of unitary alignments that make a partition of the set of
        units/segments
    dissimilarity :
        Combined dissimilarity measure used to compute this alignment's
        disorder.
    """

    NB_UNITS_PER_BATCH = 50000

    @staticmethod
    def batch_runner(possible_segments: List[List],
                     continuum: Continuum,
                     dissimilarity: AbstractDissimilarity):



        return possible_unitary_alignments, alignments_disorders

    @classmethod
    def get_best_alignment(cls,
                           continuum: Continuum,
                           dissimilarity: AbstractDissimilarity
                           ) -> 'Alignment':
        """Alignment

        Parameters
        ----------
        continuum :
            Continuum where the best alignment is computed from
        dissimilarity :
            Dissimilarity measure used to compute the disorder
        """

        possible_segments = []
        for annotator in continuum.iterannotators():
            annot_segments = []
            for segment in continuum[annotator].itersegments():
                annot_segments.append((annotator, segment))
            annot_segments.append((annotator, None))
            possible_segments.append(annot_segments)

        def unit_alignment_generator():
            for batch in grouper(2**18, product(*possible_segments)):
                yield UnitaryAlignmentBatch(continuum, batch, dissimilarity)

        with Pool(cpu_count()) as pool:
            results = pool.imap_unordered(lambda alignment: (alignment,
                                                             alignment.disorder),
                                          unit_alignment_generator(),
                                          chunksize=1)

            possible_unitary_alignments = []
            alignments_disorders = []
            for unitary_alignment, disorders in results:
                for idx, n_tuple in enumerate(unitary_alignment.tuples_list):
                    disorder = disorders[idx]
                    if disorder < len(continuum) * dissimilarity.DELTA_EMPTY:
                        unitary_alignment = UnitaryAlignment(continuum,
                                                             n_tuple,
                                                             dissimilarity)
                        possible_unitary_alignments.append(unitary_alignment)
                        alignments_disorders.append(disorder)

        # Definition of the integer linear program
        num_possible_unitary_alignments = len(possible_unitary_alignments)
        # x is the alignment variable: contains the best alignment once the
        # problem has been solved
        x = cp.Variable(shape=num_possible_unitary_alignments, boolean=True)
        d = np.array(alignments_disorders)

        # Constraints matrix
        A = np.zeros((continuum.num_units, num_possible_unitary_alignments))

        curr_idx = 0
        # fill unitary alignments matching with units
        for annotator in continuum.iterannotators():
            for unit in list(continuum[annotator].itersegments()):
                for idx_ua, unitary_alignment in enumerate(
                        possible_unitary_alignments):
                    if (annotator, unit) in unitary_alignment.n_tuple:
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
                    possible_unitary_alignments[idx])

        return cls(continuum, set_unitary_alignments, dissimilarity)

    def __init__(
            self,
            continuum: Continuum,
            set_unitary_alignments: List[UnitaryAlignment],
            dissimilarity: AbstractDissimilarity,
    ):
        super().__init__(continuum, dissimilarity)
        self.set_unitary_alignments = set_unitary_alignments

        # set partition tests for the unitary alignments
        for annotator in self.continuum.iterannotators():
            for unit in self.continuum[annotator].itersegments():
                found = 0
                for unitary_alignment in self.set_unitary_alignments:
                    if (annotator, unit) in unitary_alignment.n_tuple:
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
    def disorder(self):
        disorder = 0.0
        for unitary_alignment in self.set_unitary_alignments:
            disorder += unitary_alignment.disorder
        return disorder / self.num_alignments
