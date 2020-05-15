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
# Rachid RIAD
"""
##########
Continuum and corpus
##########

"""

from functools import lru_cache
from typing import Optional, Hashable, Dict

import numpy as np
from pyannote.core import Annotation
from sortedcontainers import SortedDict

# defining Annotator type
Annotator = Hashable


class Continuum:
    """Continuum

    Parameters
    ----------
    uri : string, optional
        name of annotated resource (e.g. audio or video file)
    modality : string, optional
        name of annotated modality
    """

    def __init__(self, uri: Optional[str] = None,
                 modality: Optional[str] = None):

        self._uri = uri
        self.modality = modality

        # sorted dictionary
        # keys: annotators
        # values: {annotator: annotations} dictionary
        self._annotators = SortedDict()

        # values: {annotator: num_units} dictionary
        self._annotators_num_units = SortedDict()

    def _get_uri(self):
        return self._uri

    def __len__(self):
        """Number of annotators in the continuum

        >>> len(continuum)  # continuum contains 3 annotators
        ... 3
        """
        return len(self._annotators)

    def __bool__(self):
        """Truthiness, basically tests for emptiness

        >>> if continuum:
        ...    # continuum is not empty
        ... else:
        ...    # continuum is empty
        """
        return len(self._annotators) > 0

    @property
    def num_units(self):
        """Number of units"""
        num_units = 0
        for annotator in self._annotators:
            num_units += len(self[annotator])
        return num_units

    @property
    def avg_num_annotations_per_annotator(self):
        """Average number of annotated segments per annotator"""
        return self.num_units / len(self)

    @property
    def max_num_annotations_per_annotator(self):
        """The maximum number of annotated segments an annotator has
        in this continuum"""
        max_num_annotations_per_annotator = 0
        for annotator in self._annotators:
            max_num_annotations_per_annotator = np.max(
                [max_num_annotations_per_annotator,
                 len(self[annotator])])
        return max_num_annotations_per_annotator

    @property
    def avg_length_unit(self):
        """Mean of the annotated segments' durations"""
        total_length_unit = 0
        for annotator in self.iterannotators():
            for unit in self[annotator].itersegments():
                total_length_unit += unit.duration
        return total_length_unit / self.num_units

    def __setitem__(self, annotator: Annotator, annotation: Annotation):
        """Add new or update existing Annotation

        >>> continuum[annotator] = Annotation

        If Annotator does not exist, it is added.
        If Annotator already exists, it is updated.

        Note
        ----
        If `Annotation` is empty, it does nothing.
        """

        # do not add empty annotation
        if not annotation:
            return

        self._annotators[annotator] = annotation
        self._annotators_num_units[annotator] = len(annotation)

    def __getitem__(self, annotator: Annotator):
        """Get annotation object

        >>> annotation = continuum[annotator]
        """

        return self._annotators[annotator]

    def iterannotators(self):
        """Iterate over annotator (in chronological order)

        >>> for annotator in annotation.iterannotators():
        ...     # do something with the annotator
        """
        return iter(self._annotators)


class Corpus:
    """Corpus

    Parameters
    ----------
    uri : string, optional
        name of annotated resource (e.g. audio or video file)
    modality : string, optional
        name of annotated modality
    """
    
    def __init__(self, uri=None, modality=None):

        self._uri = uri
        self.modality = modality

        # sorted dictionary
        # keys: annotators
        # values: {annotator: annotations} dictionary
        self._annotators: Dict[Annotator, Annotation] = SortedDict()

        # sorted dictionary
        # keys: file_name_annotated
        # values: {file_name_annotated: continuum} dictionary
        self._continuua: Dict[str, Continuum] = SortedDict()

    def _get_uri(self):
        return self._uri

    def __len__(self):
        """Number of annotators

        >>> len(corpus)  # corpus contains 2 annotated files
        2
        """
        return len(self._continuua)

    def __bool__(self):
        """Emptiness

        >>> if corpus:
        ...    # corpus is not empty
        ... else:
        ...    # corpus is empty
        """
        return len(self._continuua) > 0

    @property
    @lru_cache(maxsize=None)
    def num_units(self):
        """Number of units across continua"""
        num_units = 0
        for continuum in self._continuua:
            num_units += self._continuua[continuum].num_units
        return num_units

    @property
    @lru_cache(maxsize=None)
    def avg_num_annotations_per_annotator(self):
        return self.num_units / len(self)

    def __setitem__(self, file_name_annotated: str,
                    continuum: Continuum):
        """Add new or update existing Continuum

        >>> corpus[file_name_annotated] = Continuum
        If file_name_annotated does not exist, it is added.
        If file_name_annotated already exists, it is updated.

        Note
        ----
        If `Continuum` is empty, it does nothing.
        """

        # do not add empty annotation
        if not continuum:
            return

        self._continuua[file_name_annotated] = continuum

    def __getitem__(self, file_name_annotated: str):
        """Get continuum object

        >>> continuum = corpus[file_name_annotated]
        """

        return self._continuua[file_name_annotated]

    def itercontinuua(self):
        """Iterate over continuum (in chronological order)

        >>> for continuum in corpus.itercontinuua():
        ...     # do something with the continuum
        """
        return iter(self._continuua)
