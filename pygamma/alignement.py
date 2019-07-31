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

import numpy as np
from scipy.special import binom

from matplotlib import pyplot as plt
from itertools import product


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


class Unitary_Alignement(object):
    """Unitary Alignement
    Parameters
    ----------
    continuum :
        Continuum where the unitary alignement is from
    n_tuple :
        n-tuple where n is the number of categories of the continuum
        The tuple is composed of (annotator, segment) couples
    combined_dissimilarity :
        combined_dissimilarity
    """

    def __init__(
            self,
            continuum,
            n_tuple,
            combined_dissimilarity,
    ):

        super(Unitary_Alignement, self).__init__()
        self.continuum = continuum
        self.n_tuple = n_tuple
        assert len(n_tuple) == len(self.continuum)
        self.combined_dissimilarity = combined_dissimilarity

    @property
    def disorder(self):
        """Compute the disorder for the unitary alignement
        >>> unitary_alignement.compute_disorder() = ...
        Based on formula (6) of the original paper
        Note:
        unit is the equivalent of segment in pyannote
        """
        disorder = 0.0
        for idx, (annotator_u, unit_u) in enumerate(self.n_tuple):
            for (annotator_v, unit_v) in self.n_tuple[idx + 1:]:
                if unit_u is None:
                    disorder += self.combined_dissimilarity[
                        [unit_v], [self.continuum[annotator_v][unit_v]]]
                elif unit_v is None:
                    disorder += self.combined_dissimilarity[
                        [unit_u], [self.continuum[annotator_u][unit_u]]]
                else:
                    disorder += self.combined_dissimilarity[[unit_u, unit_v], [
                        self.continuum[annotator_u][unit_u], self.continuum[
                            annotator_v][unit_v]
                    ]]
        disorder /= binom(len(self.n_tuple), 2)
        return disorder


class Alignement(object):
    """Alignement
    Parameters
    ----------
    continuum :
        Continuum where the unitary alignement is from
    set_unitary_alignements :
        set of unitary alignements that make a partition of the set of
        units/segments
    combined_dissimilarity :
        combined_dissimilarity
    """

    def __init__(
            self,
            continuum,
            set_unitary_alignements,
            combined_dissimilarity,
    ):

        super(Alignement, self).__init__()
        self.continuum = continuum
        self.set_unitary_alignements = set_unitary_alignements
        self.combined_dissimilarity = combined_dissimilarity

        # set partition tests for the unitary alignements
        for annotator in self.continuum.iterannotators():
            for unit in self.continuum[annotator].itersegments():
                found = 0
                for unitary_alignement in self.set_unitary_alignements:
                    if [annotator, unit] in unitary_alignement.n_tuple:
                        found += 1
                if found == 0:
                    raise SetPartitionError(
                        '{} {} not in the set of unitary alignements'.format(
                            annotator, unit))
                elif found > 1:
                    raise SetPartitionError('{} {} assigned twice'.format(
                        annotator, unit))

    @property
    def disorder(self):
        """Compute the disorder for the unitary alignement
        >>> unitary_alignement.compute_disorder() = ...
        Based on formula (6) of the original paper
        Note:
        unit is the equivalent of segment in pyannote
        """
        disorder = 0.0
        for unitary_alignement in self.set_unitary_alignements:
            disorder += unitary_alignement.disorder
        return disorder


class Best_Alignement(object):
    """Alignement
    Parameters
    ----------
    continuum :
        Continuum where the unitary alignement is from
    combined_dissimilarity :
        combined_dissimilarity
    """

    def __init__(
            self,
            continuum,
            combined_dissimilarity,
    ):

        super(Alignement, self).__init__()
        self.continuum = continuum
        self.set_unitary_alignements = self.get_unitary_alignements_best()
        self.combined_dissimilarity = combined_dissimilarity

        # set partition tests for the unitary alignements
        for annotator in self.continuum.iterannotators():
            for unit in self.continuum[annotator].itersegments():
                found = 0
                for unitary_alignement in self.set_unitary_alignements:
                    if [annotator, unit] in unitary_alignement.n_tuple:
                        found += 1
                if found == 0:
                    raise SetPartitionError(
                        '{} {} not in the set of unitary alignements'.format(
                            annotator, unit))
                elif found > 1:
                    raise SetPartitionError('{} {} assigned twice'.format(
                        annotator, unit))

    def get_unitary_alignements_best(self):
        set_of_possible_segments = []
        for annotator in continuum.iterannotators():
            set_of_possible_segments.append(
                [[el, annotator]
                 for el in list(continuum[annotator].itersegments()) + [None]])

        set_of_possible_tuples = list(product(*set_of_possible_segments))
        # Property section 5.1.1 to reduce initial complexity
        set_of_possible_unitary_alignements = []
        for n_tuple in set_of_possible_tuples:
            unitary_alignement = Unitary_Alignement(
                self.continuum, n_tuple, self.combined_dissimilarity)

            # Property section 5.1.1 to reduce initial complexity
            disorder = unitary_alignement.disorder
            if disorder < self.combined_dissimilarity.DELTA_EMPTY:
                set_of_possible_unitary_alignements.append(unitary_alignement)
        return set_of_possible_unitary_alignements

    @property
    def disorder(self):
        """Compute the disorder for the unitary alignement
        >>> unitary_alignement.compute_disorder() = ...
        Based on formula (6) of the original paper
        Note:
        unit is the equivalent of segment in pyannote
        """
        disorder = 0.0
        for unitary_alignement in self.set_unitary_alignements:
            disorder += unitary_alignement.disorder
        return disorder
