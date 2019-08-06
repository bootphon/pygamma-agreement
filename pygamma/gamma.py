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
Gamma Agreement
##########

"""

import numpy as np

from pygamma.continuum import Continuum, Corpus


class Gamma_Agreement(object):
    """Gamma Agreement
    Parameters
    ----------
    continuum :
        Continuum where the alignement is from
    alignement :
        Alignement to evaluate
    strategy :
        Strategy to compute the Expected Disorder (`single` or `multi`)
        Default is `single` and stands for single continuum
        `multi` option requires a corpus object and computes it from several
        continuua
    confidence_level :
        confidence level to obtain to the expected disorder value
        Default: 0.95
    corpus (Optional):
        Corpus where the Continuum is from
    """

    def __init__(self,
                 continuum,
                 alignement,
                 corpus=Corpus(),
                 confidence_level=0.95,
                 number_sample=30):

        super(Unitary_Alignement, self).__init__()
        self.continuum = continuum
        self.alignement = alignement
        self.confidence_level = confidence_level
        assert confidence_level in (0.9, 0.95, 0.98, 0.99)

    @property
    def value(self):
        """Compute the disorder for the unitary alignement
        >>> unitary_alignement.compute_disorder() = ...
        Based on formula (6) of the original paper
        Note:
        unit is the equivalent of segment in pyannote
        """
        disorder = 0.0
        for idx, (annotator_u, unit_u) in enumerate(self.n_tuple):
            for (annotator_v, unit_v) in self.n_tuple[idx + 1:]:
                if unit_u is None and unit_v is None:
                    return self.combined_dissimilarity.DELTA_EMPTY
                elif unit_u is None:
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

    def sample_from_single_continuum(self):
        """Generate a new random alignement from a single continuum
        Strategy from figure 12
        >>> gamma_agreement.sample_from_single_continuum() = random_continuum
        """
        return self.continuum
