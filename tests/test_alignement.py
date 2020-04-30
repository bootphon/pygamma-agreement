"""Test of the module pygamma.alignement"""

import tempfile
import numpy as np
from pygamma.continuum import Continuum
from pygamma.dissimilarity import CategoricalDissimilarity
from pygamma.dissimilarity import PositionalDissimilarity
from pygamma.dissimilarity import CombinedCategoricalDissimilarity
from pygamma.dissimilarity import CombinedSequenceDissimilarity
from pygamma.alignment import UnitaryAlignment, Alignment, BestAlignment

from pygamma.alignment import SetPartitionError

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
    combi_dis = CombinedCategoricalDissimilarity(
        'diarization',
        list_categories=categories,
        DELTA_EMPTY=0.5,
        categorical_dissimilarity_matrix=cat)
    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(12, 18)], ['hadrien', None])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)

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
    combi_dis = CombinedCategoricalDissimilarity(
        'diarization',
        list_categories=categories,
        DELTA_EMPTY=0.5,
        categorical_dissimilarity_matrix=cat)
    set_unitary_alignements = []

    n_tuple = (['liza', Segment(1, 5)], ['pierrot', Segment(2, 6)],
               ['hadrien', Segment(1, 6)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(6, 8)], ['pierrot', Segment(7, 8)],
               ['hadrien', Segment(8, 10)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(7, 20)], ['pierrot', Segment(7, 19)],
               ['hadrien', Segment(7, 19)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(12, 18)], ['hadrien', None])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', None], ['pierrot', Segment(8, 10)],
               ['hadrien', Segment(19, 20)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    alignement = Alignment(continuum, set_unitary_alignements, combi_dis)

    assert alignement.disorder == pytest.approx(5.35015024691358, 0.001)


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
    combi_dis = CombinedCategoricalDissimilarity(
        'diarization',
        list_categories=categories,
        DELTA_EMPTY=0.5,
        categorical_dissimilarity_matrix=cat)
    set_unitary_alignements = []

    n_tuple = (['liza', Segment(1, 5)], ['pierrot', Segment(2, 6)],
               ['hadrien', Segment(1, 6)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(6, 8)], ['pierrot', Segment(7, 8)],
               ['hadrien', Segment(8, 10)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(7, 20)], ['pierrot', Segment(7, 19)],
               ['hadrien', Segment(7, 19)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(12, 18)], ['hadrien', None])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    # Error is here: segments used twice for liza
    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(8, 10)],
               ['hadrien', Segment(19, 20)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)
    with pytest.raises(SetPartitionError):
        alignement = Alignment(continuum, set_unitary_alignements, combi_dis)

    set_unitary_alignements = []

    n_tuple = (['liza', Segment(1, 5)], ['pierrot', Segment(2, 6)],
               ['hadrien', Segment(1, 6)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(6, 8)], ['pierrot', Segment(7, 8)],
               ['hadrien', Segment(8, 10)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(7, 20)], ['pierrot', Segment(7, 19)],
               ['hadrien', Segment(7, 19)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(12, 18)], ['hadrien', None])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    # Error is here: segment from hadrien never mentionned
    n_tuple = (['liza', None], ['pierrot', Segment(8, 10)], ['hadrien', None])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)
    with pytest.raises(SetPartitionError):
        alignement = Alignment(continuum, set_unitary_alignements, combi_dis)


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
    combi_dis = CombinedCategoricalDissimilarity(
        'diarization',
        list_categories=categories,
        DELTA_EMPTY=0.5,
        categorical_dissimilarity_matrix=cat)
    set_unitary_alignements = []

    n_tuple = (['liza', Segment(1, 5)],
               ['pierrot', Segment(2, 6)],
               ['hadrien', Segment(1, 6)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(6, 8)],
               ['pierrot', Segment(7, 8)],
               ['hadrien', Segment(8, 10)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(7, 20)],
               ['pierrot', Segment(7, 19)],
               ['hadrien', Segment(7, 19)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(12, 18)],
               ['pierrot', Segment(12, 18)],
               ['hadrien', None])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', None], ['pierrot', Segment(8, 10)],
               ['hadrien', Segment(19, 20)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    alignement = Alignment(continuum, set_unitary_alignements, combi_dis)

    best_alignement = BestAlignment(continuum, combi_dis)

    assert best_alignement.disorder == pytest.approx(0.31401409465020574,
                                                     0.001)
    assert best_alignement.disorder < alignement.disorder


def test_best_alignement_sequence():
    continuum = Continuum()
    annotation = Annotation()
    annotation[Segment(1, 5)] = ('a', 'c', 'd')
    annotation[Segment(6, 8)] = ('a', 'b', 'b', 'd')
    annotation[Segment(12, 18)] = ('a', 'b', 'c')
    annotation[Segment(7, 20)] = ('a')
    continuum['liza'] = annotation
    annotation = Annotation()
    annotation[Segment(2, 6)] = ('a', 'b')
    annotation[Segment(7, 8)] = ('a', 'b', 'c')
    annotation[Segment(12, 18)] = ('a', 'b', 'b', 'b', 'b')
    annotation[Segment(8, 10)] = ('a')
    annotation[Segment(7, 19)] = ('a', 'b', 'b')
    continuum['pierrot'] = annotation
    annotation = Annotation()

    annotation[Segment(1, 6)] = ('a', 'b')
    annotation[Segment(8, 10)] = ('a', 'b', 'b')
    annotation[Segment(7, 19)] = ('a', )
    annotation[Segment(19, 20)] = ('a', 'b', 'c', 'c', 'd')

    continuum['hadrien'] = annotation

    symbols = ['a', 'b', 'c', 'd']
    cat = np.array([[0, 0.5, 0.3, 0.7], [0.5, 0., 0.6, 0.4],
                    [0.3, 0.6, 0., 0.7], [0.7, 0.4, 0.7, 0.]])

    combi_dis = CombinedSequenceDissimilarity(
        'SR',
        list_admitted_symbols=symbols,
        DELTA_EMPTY=0.5,
        symbol_dissimilarity_matrix=cat)

    set_unitary_alignements = []

    n_tuple = (['liza', Segment(1, 5)], ['pierrot', Segment(2, 6)],
               ['hadrien', Segment(1, 6)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(6, 8)], ['pierrot', Segment(7, 8)],
               ['hadrien', Segment(8, 10)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(7, 20)], ['pierrot', Segment(7, 19)],
               ['hadrien', Segment(7, 19)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(12, 18)], ['hadrien', None])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    n_tuple = (['liza', None], ['pierrot', Segment(8, 10)],
               ['hadrien', Segment(19, 20)])
    unitary_alignement = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignements.append(unitary_alignement)

    alignement = Alignment(continuum, set_unitary_alignements, combi_dis)

    best_alignement = BestAlignment(continuum, combi_dis)
    assert best_alignement.disorder == pytest.approx(0.3738289094650206, 0.001)
    assert best_alignement.disorder < alignement.disorder
