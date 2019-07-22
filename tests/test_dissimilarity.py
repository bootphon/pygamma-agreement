"""Test of the module pygamma.continuum"""

import tempfile
import numpy as np
from pygamma.continuum import Continuum
from pygamma.dissimilarity import Categorical_Dissimilarity
from pygamma.dissimilarity import Positional_Dissimilarity

from pyannote.core import Annotation, Segment

import pytest


def test_categorical_dissimilarity():
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
    annotation[Segment(7, 19)] = 'Jeremy'
    continuum['pierrot'] = annotation
    categories = ['Carol', 'Bob', 'Alice', 'Jeremy']
    cat = np.array([[0, 0.5, 0.3, 0.7], [0.5, 0., 0.6, 0.4],
                    [0.3, 0.6, 0., 0.7], [0.7, 0.4, 0.7, 0.]])

    cat_dis = Categorical_Dissimilarity(
        'diarization',
        list_categories=categories,
        categorical_dissimlarity_matrix=cat,
        DELTA_EMPTY=0.5)

    assert cat_dis[['Carol', 'Carol']] == 0.
    assert cat_dis[['Carol', 'Jeremy']] == 0.35


def test_positional_dissimilarity():
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
    annotation[Segment(7, 19)] = 'Jeremy'
    continuum['pierrot'] = annotation
    categories = ['Carol', 'Bob', 'Alice', 'Jeremy']

    pos_dis = Positional_Dissimilarity('diarization', DELTA_EMPTY=0.5)

    list_dis = []
    for liza_unit in continuum['liza'].itertracks():
        for pierrot_unit in continuum['pierrot'].itertracks():
            list_dis.append(pos_dis[[liza_unit, pierrot_unit]])
    assert list_dis == [
        0.03125, 1.62, 0.78125, 2.0, 2.88, 0.5, 0.05555555555555555,
        0.36734693877551017, 0.5, 2.0, 0.6245674740484429, 0.36734693877551017,
        0.0008, 0.26888888888888884, 0.06786703601108032, 2.4200000000000004,
        2.2959183673469385, 0.05555555555555555, 1.125, 0.0
    ]
    assert pos_dis[[liza_unit,
                    pierrot_unit]] == pos_dis[[pierrot_unit, liza_unit]]
    assert pos_dis[[liza_unit, liza_unit]] == 0
    assert pos_dis[[liza_unit]] == 1 * 0.5
