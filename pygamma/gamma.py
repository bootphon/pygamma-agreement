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
from pyannote.core import Segment, Timeline, Annotation

from pygamma.continuum import Continuum, Corpus
from pygamma.alignement import Unitary_Alignement, Alignement, Best_Alignement


class Gamma_Agreement(object):
    """Gamma Agreement
    Parameters
    ----------
    continuum :
        Continuum where the alignement is from
    alignement :
        Alignement to evaluate
    dissimilarity :
        Dissimilarity to be used to estimate the disorder and gamma agreement
    strategy :
        Strategy to compute the Expected Disorder (`single` or `multi`)
        Default is `single` and stands for single continuum
        `multi` option requires a corpus object and computes it from several
        continuua
    number_samples :
        Number of samples drawn to estimate the expected disorder
    confidence_level :
        confidence level to obtain to the expected disorder value
        Default: 0.95
    corpus (Optional):
        Corpus where the Continuum is from
    type_pivot (Optional):
        If text input would be nice to have only pivot with integer
        Does not change anything in practice
    """

    def __init__(self,
                 continuum,
                 alignement,
                 dissimilarity,
                 strategy='single',
                 corpus=Corpus(),
                 confidence_level=0.95,
                 number_samples=30,
                 type_pivot='float_pivot'):

        super(Gamma_Agreement, self).__init__()
        self.continuum = continuum
        self.alignement = alignement
        self.confidence_level = confidence_level
        self.type_pivot = type_pivot
        self.corpus = corpus
        self.strategy = strategy
        self.dissimilarity = dissimilarity
        assert self.strategy in ('single', 'multi')
        if self.strategy is 'multi':
            assert self.corpus, 'Should be provided with a corpus object'
        self.number_samples = number_samples
        assert self.confidence_level in (0.9, 0.95, 0.98, 0.99)
        assert self.type_pivot in ('float_pivot', 'int_pivot')
        self.last_unit_start = 0.0
        for annotator in self.continuum.iterannotators():
            for unit in self.continuum[annotator].itersegments():
                if unit.start > self.last_unit_start:
                    self.last_unit_start = unit.start

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

    def sample_annotation_from_single_continuum(self):
        """Generate a new random annotation from a single continuum
        Strategy from figure 12
        >>> gamma_agreement.sample_from_single_continuum() = random_annotation
        """
        if self.type_pivot == 'float_pivot':
            pivot = np.random.uniform(self.continuum.avg_length_unit,
                                      self.last_unit_start)
        else:
            pivot = np.random.randint(self.continuum.avg_length_unit,
                                      self.last_unit_start)
        sampled_annotation = Annotation()
        annotator = np.random.choice(list(self.continuum.iterannotators()))
        annotation = self.continuum[annotator]
        for unit in annotation.itersegments():
            if pivot - unit.start < 0:
                sampled_annotation[Segment(
                    unit.start - pivot, unit.end - pivot)] = annotation[unit]
            else:
                sampled_annotation[Segment(
                    unit.start + pivot, unit.end + pivot)] = annotation[unit]
        return sampled_annotation

    def sample_annotation_from_corpus(self):
        """Generate a new random annotation from a corpus
        Strategy from figure 13
        >>> gamma_agreement.sample_annotation_from_corpus() = random_annotation
        """
        # pivot = 6
        if type_pivot == float:
            pivot = np.random.uniform(self.continuum.avg_length_unit, 2)
        sampled_annotation = Annotation()

        for unit in annotation.itersegments():
            if pivot - unit.start < 0:
                sampled_annotation[Segment(
                    unit.start - pivot, unit.end - pivot)] = annotation[unit]
            else:
                sampled_annotation[Segment(
                    unit.start + pivot, unit.end + pivot)] = annotation[unit]
        return sampled_annotation

    def compute_chance_disorder_values(self):
        """Compute the chance disorder values
        """
        chance_disorder_values = []
        for _ in range(self.number_samples):
            sampled_continuum = Continuum()
            for idx in range(len(self.continuum)):
                if self.strategy is 'single':
                    sampled_continuum['Sampled_annotation {}'.format(
                        idx)] = self.sample_annotation_from_single_continuum()
                if self.strategy is 'multi':
                    sampled_continuum['Sampled_annotation {}'.format(
                        idx)] = self.sample_annotation_from_corpus()
            best_seq_alignement = Best_Alignement(sampled_continuum,
                                                  self.dissimilarity)
            chance_disorder_values.append(best_seq_alignement.disorder)

        return chance_disorder_values
