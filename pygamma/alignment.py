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
from typing import List, Tuple, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .continuum import Continuum, Unit, Annotator


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

    @property
    @abstractmethod
    def disorder(self) -> float:
        """Compute the disorder for the alignement

        >>> aligment.disorder
        ... 0.123

        """

        raise NotImplemented()


class UnitaryAlignment(AbstractAlignment):
    """Unitary Alignment

    Parameters
    ----------
    n_tuple :
        n-tuple where n is the number of annotators of the continuum
        This is a list of (annotator, segment) couples
    dissimilarity :
        combined_dissimilarity
    """

    def __init__(self, n_tuple: Tuple[Tuple['Annotator', Optional['Unit']]]):
        self.n_tuple = n_tuple
        self._disorder: Optional[float] = None

    @property
    def disorder(self) -> float:
        return self._disorder


class Alignment(AbstractAlignment):
    """Alignment

    Parameters
    ----------
    continuum :
        Continuum where the alignment is from
    set_unitary_alignments :
        set of unitary alignments that make a partition of the set of
        units/segments
    """

    def __init__(self,
                 continuum: 'Continuum',
                 set_unitary_alignments: List[UnitaryAlignment]):
        self.set_unitary_alignments = set_unitary_alignments
        self.continuum = continuum

        # set partition tests for the unitary alignments
        # TODO : this has to be seriously sped up
        for annotator, units in self.continuum:
            for unit in units.values():
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
        return sum(u_align.disorder for u_align
                   in self.set_unitary_alignments) / self.num_alignments
