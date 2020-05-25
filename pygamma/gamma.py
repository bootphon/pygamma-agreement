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
Gamma Agreement
##########

"""

import csv
import logging
from collections import defaultdict
from typing import Iterable, Tuple

import numpy as np
from pyannote.core import Annotation, Segment

from .alignment import Alignment
from .continuum import Continuum, Corpus
from .dissimilarity import AbstractDissimilarity
from .dissimilarity import SequenceDissimilarity, CategoricalDissimilarity


class GammaAgreement:
    """Gamma Agreement
    Parameters
    ----------
    continuum :
        Continuum where the alignment is from
    alignment :
        alignment to evaluate
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
                 continuum: Continuum,
                 alignment: Alignment,
                 dissimilarity: AbstractDissimilarity,
                 strategy: str = 'single',
                 corpus: Corpus = Corpus(),  # TODO : move this to __init__
                 confidence_level: float = 0.95,
                 number_samples: int = 30,
                 type_pivot: str = 'float_pivot'):

        self.continuum = continuum
        self.alignment = alignment
        self.confidence_level = confidence_level
        self.type_pivot = type_pivot
        self.corpus = corpus
        self.strategy = strategy
        self.dissimilarity = dissimilarity
        assert self.strategy in ('single', 'multi')
        if self.strategy is 'multi':
            assert self.corpus, 'Should be provided with a corpus object'
        self.number_samples = number_samples
        # TODO : maybe change this to "low, high, higher, maximum" or smthg
        assert self.confidence_level in (0.9, 0.95, 0.98, 0.99)
        assert self.type_pivot in ('float_pivot', 'int_pivot')
        self.last_unit_start = 0.0
        for annotator in self.continuum.iterannotators():
            for unit in self.continuum[annotator].itersegments():
                if unit.start > self.last_unit_start:
                    self.last_unit_start = unit.start

    def sample_annotation_from_single_continuum(self):
        """Generate a new random annotation from a single continuum
        Strategy from figure 12

        >>> gamma_agreement.sample_from_single_continuum()
        ... <pyannote.core.annotation.Annotation at 0x7f5527a19588>
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
                new_segment = Segment(unit.start - pivot, unit.end - pivot)
            else:
                new_segment = Segment(unit.start + pivot, unit.end + pivot)
            sampled_annotation[new_segment] = annotation[unit]
        return sampled_annotation

    def sample_annotation_from_corpus(self):
        """Generate a new random annotation from a corpus
        Strategy from figure 13

        >>> gamma_agreement.sample_annotation_from_corpus()
        ... <pyannote.core.annotation.Annotation at 0x7f5527a19588>
        """
        # TODO: doesn't seem finished
        # pivot = 6
        if self.type_pivot == "float_pivot":
            pivot = np.random.uniform(self.continuum.avg_length_unit, 2)
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
            best_seq_alignment = Alignment.get_best_alignment(
                sampled_continuum,
                self.dissimilarity)
            chance_disorder_values.append(best_seq_alignment.disorder)

        return chance_disorder_values

    def get_gamma(self) -> float:
        """Compute gamma value"""
        chance_disorders = self.compute_chance_disorder_values()
        best_seq_alignment_disorder = self.alignment.disorder
        return 1 - (best_seq_alignment_disorder / np.mean(chance_disorders))


def compute_gamma(units: Iterable[Tuple[str, str, float, float]],
                  dissimilarity: str = "categorical",) -> float:
    assert dissimilarity in ("categorical", "sequence")
    annotations = defaultdict(Annotation)
    categories = set()
    symbols = set()

    for row in units:
        if dissimilarity == "categorical":
            categories.add(row[1])
        else:
            symbols.update(set(row[1]))
        annotations[row[0]][Segment(int(row[2]), int(row[3]))] = row[1]

    continuum = Continuum()
    for annotator, annotation in annotations.items():
        continuum[annotator] = annotation

    if dissimilarity == "categorical":
        dissim = CategoricalDissimilarity("", list(categories))
    else:
        dissim = SequenceDissimilarity("", list(symbols))

    logging.info("Computing best alignment...")
    best_alignment = Alignment.get_best_alignment(continuum, dissim)
    gamma = GammaAgreement(continuum, best_alignment, dissim)
    logging.info("Computing Gamma Agreement for that alignment...")
    return gamma.get_gamma()
