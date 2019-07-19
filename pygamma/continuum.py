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
Annotation
##########

"""

import itertools
import numpy as np

from sortedcontainers import SortedDict

from pyannote.core import Segment, Timeline, Annotation


class Continuum(object):
    """Continuum
    Parameters
    ----------
    uri : string, optional
        name of annotated resource (e.g. audio or video file)
    modality : string, optional
        name of annotated modality
    Returns
    -------
    continuum : Continuum
        New continuum
    """

    def __init__(self, uri=None, modality=None):

        super(Continuum, self).__init__()

        self._uri = uri
        self.modality = modality

        # sorted dictionary
        # keys: annotators
        # values: {annotator: annotations} dictionary
        self._annotators = SortedDict()

        # values: {annotator: num_units} dictionary
        self._annotators_num_units = SortedDict()

        # timeline meant to store all annotated segments
        self._timeline = None
        self._timelineNeedsUpdate = True

    def _get_uri(self):
        return self._uri

    def __len__(self):
        """Number of annotators
        >>> len(continuum)  # continuum contains 3 annotators
        3
        """
        return len(self._annotators)

    def __bool__(self):
        """Emptiness
        >>> if continuum:
        ...    # continuum is not empty
        ... else:
        ...    # continuum is empty
        """
        return len(self._annotators) > 0

    @property
    def num_units(self):
        """Emptiness
        >>> if continuum:
        ...    # continuum is empty
        ... else:
        ...    # continuum is not empty
        """
        num_units = 0
        for annotator in self._annotators:
            num_units += len(self[annotator])
        return num_units

    @property
    def avg_num_annotations_per_annotator(self):
        return self.num_units / len(self)

    def __setitem__(self, annotator, annotation):
        """Add new or update existing Annotation
        >>> continuum[Annotator] = Annotation
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

    # annotation = continuum[annotator]
    def __getitem__(self, annotator):
        """Get annotation object
        >>> annotation = continuum[annotator]
        """

        return self._annotators[annotator]

    def iterannotators(self):
        """Iterate over segments (in chronological order)
        >>> for segment in annotation.iterannotators():
        ...     # do something with the annotator
        """
        return iter(self._annotators)
