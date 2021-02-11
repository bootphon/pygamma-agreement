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
Alignement and disorder
##########

"""
from abc import ABCMeta, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Union
from typing import Tuple, Optional, Iterable

import numpy as np

from .dissimilarity import AbstractDissimilarity

if TYPE_CHECKING:
    from .continuum import Continuum, Annotator

UnitsTuple = Tuple[Tuple['Annotator', 'Unit']]


class SetPartitionError(Exception):
    """Exception raised for errors in the partition of units of continuum.

    Attributes:
        message -- explanation of the error
    """


class AbstractAlignment(metaclass=ABCMeta):

    @property
    @abstractmethod
    def disorder(self) -> float:
        """Return the disorder for an alignment, if it has already been computed
        beforehand

        >>> aligment.disorder
        ... 0.123

        """

    @abstractmethod
    def compute_disorder(self, dissimilarity: AbstractDissimilarity) -> float:
        """Compute the disorder for that alignment"""


class UnitaryAlignment:
    """Unitary Alignment

    Parameters
    ----------
    n_tuple :
        n-tuple where n is the number of annotators of the continuum
        This is a list of (annotator, segment) couples
    """

    def __init__(self, n_tuple: UnitsTuple):
        assert len(n_tuple) >= 2
        self._n_tuple = n_tuple
        self._disorder: Optional[float] = None

    @property
    def n_tuple(self):
        return self._n_tuple

    @n_tuple.setter
    def n_tuple(self, n_tuple: UnitsTuple):
        self._n_tuple = n_tuple
        self._disorder = None

    @property
    def disorder(self) -> float:
        # TODO : doc
        if self._disorder is None:
            raise ValueError("Disorder hasn't been computed. "
                             "Call `compute_disorder()` first to compute it.")
        else:
            return self._disorder

    @disorder.setter
    def disorder(self, value: float):
        self._disorder = value

    def compute_disorder(self, dissimilarity: AbstractDissimilarity):
        # TODO : doc
        # building a fake one-element alignment to compute the dissim
        fake_alignment = Alignment([self])
        self._disorder = fake_alignment.compute_disorder(dissimilarity)
        return self._disorder


class Alignment(AbstractAlignment):
    """Alignment

    Parameters
    ----------
    unitary_alignments :
        set of unitary alignments that make a partition of the set of
        units/segments
    continuum : optional Continuum
        Continuum where the alignment is from
    check_validity: bool
        Check the validity of that Alignment against the specified continuum
    """

    def __init__(self,
                 unitary_alignments: Iterable[UnitaryAlignment],
                 continuum: Optional['Continuum'] = None,
                 check_validity: bool = False
                 ):
        self.unitary_alignments = list(unitary_alignments)
        self.continuum = continuum
        self._disorder: Optional[float] = None

        if not check_validity:
            return
        else:
            self.check()

    def __getitem__(self, *keys: Union[int, Tuple[int, 'Annotator']]):
        # TODO : test and document
        if len(keys) == 1:
            idx = keys[0]
            return self.unitary_alignments[idx]
        elif len(keys) == 2:
            idx, annotator = keys
            annotator_idx = self.annotators.index(annotator)
            return self.unitary_alignments[idx].n_tuple[annotator_idx]
        else:
            raise KeyError("Invalid number of items in key")

    @property
    def annotators(self):
        return [annotator for annotator, _
                in self.unitary_alignments[0].n_tuple]

    @property
    def num_alignments(self):
        return len(self.unitary_alignments)

    @property
    def num_annotators(self):
        return len(self.unitary_alignments[0].n_tuple)

    @property
    def disorder(self):
        # TODO : doc
        if self._disorder is None:
            self._disorder = (sum(u_align.disorder for u_align
                                  in self.unitary_alignments)
                              / self.num_alignments)
        return self._disorder

    def compute_disorder(self, dissimilarity: AbstractDissimilarity):
        # TODO : doc
        disorder_args = dissimilarity.build_args(self)
        unit_ids = np.arange(self.num_alignments, dtype=np.int32)
        unit_ids = np.vstack([unit_ids] * self.num_annotators)
        unit_ids = unit_ids.swapaxes(0, 1)
        disorders = dissimilarity(unit_ids, *disorder_args)
        for i, disorder in enumerate(disorders):
            self.unitary_alignments[i].disorder = disorder
        self._disorder = (dissimilarity(unit_ids, *disorder_args).sum()
                          / self.num_alignments)
        return self._disorder

    def check(self, continuum: Optional['Continuum'] = None):
        """
        Checks that an alignment is a valid partition of a Continuum. That is,
        that all annotations from the referenced continuum *can be found*
        in the alignment and can be found *only once*. Empty units are not
        taken into account.

        Parameters
        ----------
        continuum: optional Continuum
            Continuum to check the alignment against. If none is specified,
            will try to use the one set at instanciation.

        Raises
        -------
        ValueError, SetPartitionError
        """
        if continuum is None:
            if self.continuum is None:
                raise ValueError("No continuum was set")
            continuum = self.continuum

        # simple check: verify that all unitary alignments have the same length
        first_len = len(self.unitary_alignments[0].n_tuple)
        for unit_align in self.unitary_alignments:
            if len(unit_align.n_tuple) != first_len:
                raise ValueError(
                    f"Unitary alignments {self.unitary_alignments[0]} and"
                    f"{unit_align} don't have the same amount of units tuples")

        # set partition tests for the unitary alignments
        continuum_tuples = set()
        for annotator, unit in continuum:
            continuum_tuples.add((annotator, unit))

        alignment_tuples = list()
        for unitary_alignment in self.unitary_alignments:
            for (annotator, unit) in unitary_alignment.n_tuple:
                if unit is None:
                    continue
                alignment_tuples.append((annotator, unit))

        # let's first look for missing ones, then for repeated assignments
        missing_tuples = continuum_tuples - set(alignment_tuples)
        if missing_tuples:
            repeated_tuples_str = ', '.join(f"{annotator}->{unit}"
                                            for annotator, unit in missing_tuples)

            raise SetPartitionError(f'{repeated_tuples_str} '
                                    f'not in the set of unitary alignments')

        tuples_counts = Counter(alignment_tuples)
        repeated_tuples = {tup for tup, count in tuples_counts.items() if count > 1}
        if repeated_tuples:
            repeated_tuples_str = ', '.join(f"{annotator}->{unit}"
                                            for annotator, unit in repeated_tuples)

            raise SetPartitionError(f'{repeated_tuples_str} '
                                    f'are found more than once in the set '
                                    f'of unitary alignments')

    def _repr_png_(self):
        """IPython notebook support

        See also
        --------
        :mod:`pygamma_agreement.notebook`
        """

        from .notebook import repr_alignment
        return repr_alignment(self)
