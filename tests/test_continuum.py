"""Test of the module pygamma.continuum"""

import tempfile
import numpy as np
from pygamma.continuum import Continuum, Corpus
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


def test_corpus_init():
    continuum = Continuum()
    annotation = Annotation()
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
    corpus = Corpus()
    assert len(corpus) == 0
    corpus['first_file.wav'] = continuum
    continuum = Continuum()
    annotation = Annotation()
    annotation[Segment(1, 5)] = 'Carol'
    annotation[Segment(6, 8)] = 'Bob'
    continuum['liza'] = annotation
    annotation = Annotation()
    annotation[Segment(2, 6)] = 'Carol'
    annotation[Segment(8, 10)] = 'Alice'
    annotation[Segment(7, 19)] = 'Carol'
    continuum['pierrot'] = annotation
    corpus['long_first_file.wav'] = continuum
    assert corpus
    assert len(corpus) == 2
    assert corpus.num_units == 14
    assert corpus['long_first_file.wav'] == continuum
    assert corpus.avg_num_annotations_per_annotator == 7
