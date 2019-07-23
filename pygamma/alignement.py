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
