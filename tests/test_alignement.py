"""Test of the module pygamma.alignement"""

import tempfile
import numpy as np
from pygamma.continuum import Continuum
from pygamma.dissimilarity import Categorical_Dissimilarity
from pygamma.dissimilarity import Positional_Dissimilarity
from pygamma.dissimilarity import Combined_Dissimilarity
from pygamma.alignement import Unitary_Alignement

from pyannote.core import Annotation, Segment

import pytest


def test_unitary_alignement():
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
    annotation = Annotation()

    annotation[Segment(1, 6)] = 'Carol'
    annotation[Segment(8, 10)] = 'Alice'
    annotation[Segment(7, 19)] = 'Jeremy'
    annotation[Segment(19, 20)] = 'Alice'

    continuum['hadrien'] = annotation

    categories = ['Carol', 'Bob', 'Alice', 'Jeremy']
    cat = np.array([[0, 0.5, 0.3, 0.7], [0.5, 0., 0.6, 0.4],
                    [0.3, 0.6, 0., 0.7], [0.7, 0.4, 0.7, 0.]])
    combi_dis = Combined_Dissimilarity(
        'diarization',
        list_categories=categories,
        DELTA_EMPTY=0.5,
        categorical_dissimlarity_matrix=cat)
    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(12, 18)], ['hadrien', None])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)

    assert unitary_alignement.disorder == 0.3833333333333333
