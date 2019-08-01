"""Test of the module pygamma.alignement"""

import tempfile
import numpy as np
from pygamma.continuum import Continuum
from pygamma.dissimilarity import Categorical_Dissimilarity
from pygamma.dissimilarity import Positional_Dissimilarity
from pygamma.dissimilarity import Combined_Categorical_Dissimilarity
from pygamma.alignement import Unitary_Alignement, Alignement, Best_Alignement

from pygamma.alignement import SetPartitionError

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
    combi_dis = Combined_Categorical_Dissimilarity(
        'diarization',
        list_categories=categories,
        DELTA_EMPTY=0.5,
        categorical_dissimilarity_matrix=cat)
    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(12, 18)], ['hadrien', None])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)

    assert unitary_alignement.disorder == pytest.approx(
        0.3833333333333333, 0.001)


def test_alignement():
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
    combi_dis = Combined_Categorical_Dissimilarity(
        'diarization',
        list_categories=categories,
        DELTA_EMPTY=0.5,
        categorical_dissimilarity_matrix=cat)
    set_unitary_alignements = []

    n_tuple = (['liza', Segment(1, 5)], ['pierrot', Segment(2, 6)],
               ['hadrien', Segment(1, 6)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(6, 8)], ['pierrot', Segment(7, 8)],
               ['hadrien', Segment(8, 10)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(7, 20)], ['pierrot', Segment(7, 19)],
               ['hadrien', Segment(7, 19)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(12, 18)], ['hadrien', None])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', None], ['pierrot', Segment(8, 10)],
               ['hadrien', Segment(19, 20)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    alignement = Alignement(continuum, set_unitary_alignements, combi_dis)

    assert alignement.disorder == 2.2346731718898387


def test_wrong_set_unitary_alignement():
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
    combi_dis = Combined_Categorical_Dissimilarity(
        'diarization',
        list_categories=categories,
        DELTA_EMPTY=0.5,
        categorical_dissimilarity_matrix=cat)
    set_unitary_alignements = []

    n_tuple = (['liza', Segment(1, 5)], ['pierrot', Segment(2, 6)],
               ['hadrien', Segment(1, 6)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(6, 8)], ['pierrot', Segment(7, 8)],
               ['hadrien', Segment(8, 10)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(7, 20)], ['pierrot', Segment(7, 19)],
               ['hadrien', Segment(7, 19)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(12, 18)], ['hadrien', None])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    # Error is here: segments used twice for liza
    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(8, 10)],
               ['hadrien', Segment(19, 20)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)
    with pytest.raises(SetPartitionError):
        alignement = Alignement(continuum, set_unitary_alignements, combi_dis)

    set_unitary_alignements = []

    n_tuple = (['liza', Segment(1, 5)], ['pierrot', Segment(2, 6)],
               ['hadrien', Segment(1, 6)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(6, 8)], ['pierrot', Segment(7, 8)],
               ['hadrien', Segment(8, 10)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(7, 20)], ['pierrot', Segment(7, 19)],
               ['hadrien', Segment(7, 19)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(12, 18)], ['hadrien', None])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    # Error is here: segment from hadrien never mentionned
    n_tuple = (['liza', None], ['pierrot', Segment(8, 10)], ['hadrien', None])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)
    with pytest.raises(SetPartitionError):
        alignement = Alignement(continuum, set_unitary_alignements, combi_dis)


def test_best_alignement():
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
    combi_dis = Combined_Categorical_Dissimilarity(
        'diarization',
        list_categories=categories,
        DELTA_EMPTY=0.5,
        categorical_dissimilarity_matrix=cat)
    set_unitary_alignements = []

    n_tuple = (['liza', Segment(1, 5)], ['pierrot', Segment(2, 6)],
               ['hadrien', Segment(1, 6)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(6, 8)], ['pierrot', Segment(7, 8)],
               ['hadrien', Segment(8, 10)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(7, 20)], ['pierrot', Segment(7, 19)],
               ['hadrien', Segment(7, 19)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(12, 18)], ['hadrien', None])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', None], ['pierrot', Segment(8, 10)],
               ['hadrien', Segment(19, 20)])
    unitary_alignement = Unitary_Alignement(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    alignement = Alignement(continuum, set_unitary_alignements, combi_dis)

    best_alignement = Best_Alignement(continuum, combi_dis)

    assert best_alignement.disorder == pytest.approx(0.32092361020850013,
                                                     0.001)
    assert best_alignement.disorder < alignement.disorder
