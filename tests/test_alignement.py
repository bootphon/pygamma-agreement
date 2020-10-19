"""Test of the module pygamma.alignment"""

import numpy as np
import pytest
from pyannote.core import Annotation, Segment

from pygamma.alignment import SetPartitionError
from pygamma.alignment import UnitaryAlignment, Alignment
from pygamma.continuum import Continuum, Unit
from pygamma.dissimilarity import CombinedCategoricalDissimilarity


@pytest.mark.skip(reason="Needs support for None units in alignments")
def test_unitary_alignment():
    continuum = Continuum()
    annotation = Annotation()
    annotation[Segment(1, 5)] = 'Carol'
    annotation[Segment(6, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Carol'
    annotation[Segment(7, 20)] = 'Alice'
    continuum.add_annotation("liza", annotation)
    annotation = Annotation()
    annotation[Segment(2, 6)] = 'Carol'
    annotation[Segment(7, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Alice'
    annotation[Segment(8, 10)] = 'Alice'
    annotation[Segment(7, 19)] = 'Jeremy'
    continuum.add_annotation("pierrot", annotation)
    annotation = Annotation()

    annotation[Segment(1, 6)] = 'Carol'
    annotation[Segment(8, 10)] = 'Alice'
    annotation[Segment(7, 19)] = 'Jeremy'
    annotation[Segment(19, 20)] = 'Alice'

    continuum.add_annotation("hadrien", annotation)

    categories = ['Carol', 'Bob', 'Alice', 'Jeremy']
    cat = np.array([[0, 0.5, 0.3, 0.7], [0.5, 0., 0.6, 0.4],
                    [0.3, 0.6, 0., 0.7], [0.7, 0.4, 0.7, 0.]])
    combi_dis = CombinedCategoricalDissimilarity(
        list_categories=categories,
        delta_empty=0.5,
        categorical_dissimilarity_matrix=cat,
        beta=3)
    n_tuple = (('liza', Unit(Segment(12, 18))),
               ('pierrot', Unit(Segment(12, 18))),
               ('hadrien', None))
    unitary_alignment = UnitaryAlignment(n_tuple)

    assert unitary_alignment.compute_disorder(combi_dis) == pytest.approx(
        0.3833333333333333, 0.001)

@pytest.mark.skip(reason="Needs support for None units in alignments")
def test_alignment():
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
    set_unitary_alignments = []

    n_tuple = (('liza', Segment(1, 5)),
               ('pierrot', Segment(2, 6)),
               ('hadrien', Segment(1, 6)))
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (('liza', Segment(6, 8)),
               ('pierrot', Segment(7, 8)),
               ('hadrien', Segment(8, 10)))
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (('liza', Segment(7, 20)),
               ('pierrot', Segment(7, 19)),
               ('hadrien', Segment(7, 19)))
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (('liza', Segment(12, 18)),
               ('pierrot', Segment(12, 18)),
               ('hadrien', None))
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (('liza', None),
               ('pierrot', Segment(8, 10)),
               ('hadrien', Segment(19, 20)))
    unitary_alignment = UnitaryAlignment(continuum)
    set_unitary_alignments.append(unitary_alignment)

    alignment = Alignment(continuum, set_unitary_alignments, combi_dis)

    assert alignment.disorder == pytest.approx(5.35015024691358, 0.001)

@pytest.mark.skip(reason="Needs update")
def test_wrong_set_unitary_alignment():
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
    set_unitary_alignments = []

    n_tuple = (['liza', Segment(1, 5)], ['pierrot', Segment(2, 6)],
               ['hadrien', Segment(1, 6)])
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (['liza', Segment(6, 8)], ['pierrot', Segment(7, 8)],
               ['hadrien', Segment(8, 10)])
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (['liza', Segment(7, 20)], ['pierrot', Segment(7, 19)],
               ['hadrien', Segment(7, 19)])
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(12, 18)], ['hadrien', None])
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    # Error is here: segments used twice for liza
    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(8, 10)],
               ['hadrien', Segment(19, 20)])
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)
    with pytest.raises(SetPartitionError):
        alignment = Alignment(continuum, set_unitary_alignments, combi_dis)

    set_unitary_alignments = []

    n_tuple = (['liza', Segment(1, 5)], ['pierrot', Segment(2, 6)],
               ['hadrien', Segment(1, 6)])
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (['liza', Segment(6, 8)], ['pierrot', Segment(7, 8)],
               ['hadrien', Segment(8, 10)])
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (['liza', Segment(7, 20)], ['pierrot', Segment(7, 19)],
               ['hadrien', Segment(7, 19)])
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (['liza', Segment(12, 18)], ['pierrot',
                                           Segment(12, 18)], ['hadrien', None])
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    # Error is here: segment from hadrien never mentionned
    n_tuple = (['liza', None], ['pierrot', Segment(8, 10)], ['hadrien', None])
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)
    with pytest.raises(SetPartitionError):
        alignment = Alignment(continuum, set_unitary_alignments, combi_dis)


@pytest.mark.skip(reason="Needs support for None units in alignments")
def test_best_alignment():
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
    set_unitary_alignments = []

    n_tuple = (('liza', Segment(1, 5)),
               ('pierrot', Segment(2, 6)),
               ('hadrien', Segment(1, 6)))
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (('liza', Segment(6, 8)),
               ('pierrot', Segment(7, 8)),
               ('hadrien', Segment(8, 10)))
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (('liza', Segment(7, 20)),
               ('pierrot', Segment(7, 19)),
               ('hadrien', Segment(7, 19)))
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (('liza', Segment(12, 18)),
               ('pierrot', Segment(12, 18)),
               ('hadrien', None))
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = (('liza', None),
               ('pierrot', Segment(8, 10)),
               ('hadrien', Segment(19, 20)))
    unitary_alignment = UnitaryAlignment(continuum, n_tuple, combi_dis)
    set_unitary_alignments.append(unitary_alignment)

    alignment = Alignment(continuum, set_unitary_alignments, combi_dis)

    best_alignment = Alignment.get_best_alignment(continuum, combi_dis)

    assert best_alignment.disorder == pytest.approx(0.31401409465020574,
                                                    0.001)
    assert best_alignment.disorder < alignment.disorder


# TODO test continuum for best alignment