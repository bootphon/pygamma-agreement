"""Test of the module pygamma.continuum"""

import tempfile
import numpy as np
from pygamma.continuum import Continuum
from pyannote.core import Annotation, Segment

import pytest


def test_continuum_init():
    continuum = Continuum()
    annotation = Annotation()
    assert len(continuum) == 0
    annotation[Segment(1, 5)] = 'Carol'
    annotation[Segment(6, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Carol'
    annotation[Segment(7, 20)] = 'Alice'
    continuum['liza'] = annotation
    annotation = Annotation()
    annotation[Segment(2, 6)] = 'Carol'
    annotation[Segment(7, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Alice'
    annotation[Segment(8, 10)] = 'Alice'
    annotation[Segment(7, 19)] = 'Carol'
    continuum['pierrot'] = annotation
    assert continuum
    assert len(continuum) == 2
    assert continuum.num_units == 9
    assert continuum['pierrot'] == annotation
    assert continuum.avg_num_annotations_per_annotator == 4.5
    assert list(continuum.iterannotators) == ['liza', 'pierrot']
