"""Test of the Alignment class pygamma_agreement.alignment"""

import numpy as np
import pytest
from pyannote.core import Annotation, Segment

from pygamma_agreement.alignment import SetPartitionError
from pygamma_agreement.alignment import UnitaryAlignment, Alignment
from pygamma_agreement.continuum import Continuum, Unit
from pygamma_agreement.dissimilarity import CombinedCategoricalDissimilarity, PrecomputedCategoricalDissimilarity
from sortedcontainers import SortedSet

def test_alignment_checking():
    continuum = Continuum()
    annotation = Annotation()
    annotation[Segment(1, 5)] = 'Carol'
    annotation[Segment(6, 8)] = 'Bob'
    continuum.add_annotation('liza', annotation)

    annotation = Annotation()
    annotation[Segment(2, 6)] = 'Carol'
    continuum.add_annotation('pierrot', annotation)

    # checking valid alignment
    alignment = Alignment([], continuum=continuum)

    n_tuple = [('liza', Unit(Segment(1, 5), 'Carol')),
               ('pierrot', Unit(Segment(2, 6), 'Carol'))]
    unitary_alignment = UnitaryAlignment(n_tuple)
    assert unitary_alignment.nb_units == 2
    alignment.unitary_alignments.append(unitary_alignment)

    # checking missing
    with pytest.raises(SetPartitionError):
        alignment.check(continuum)

    n_tuple = [('liza', Unit(Segment(6, 8), 'Bob')),
               ('pierrot', None)]
    unitary_alignment = UnitaryAlignment(n_tuple)
    assert unitary_alignment.nb_units == 1
    alignment.unitary_alignments.append(unitary_alignment)

    # checking valid alignment
    alignment.check(continuum)

    # checking with extra tuple
    n_tuple = [('liza', Unit(Segment(6, 8), 'Bob')),
               ('pierrot', None)]
    unitary_alignment = UnitaryAlignment(n_tuple)
    assert unitary_alignment.nb_units == 1
    alignment.unitary_alignments.append(unitary_alignment)

    # checking missing
    with pytest.raises(SetPartitionError):
        alignment.check(continuum)


def test_unitary_alignment():
    categories = SortedSet(['Carol', 'Bob', 'Alice', 'Jeremy'])
    cat = np.array([[0, 0.5, 0.3, 0.7],
                    [0.5, 0., 0.6, 0.4],
                    [0.3, 0.6, 0., 0.7],
                    [0.7, 0.4, 0.7, 0.]])
    combi_dis = CombinedCategoricalDissimilarity(
        delta_empty=0.5,
        cat_dissim=PrecomputedCategoricalDissimilarity(categories, cat, delta_empty=0.5),
        alpha=1)
    n_tuple = [('liza', Unit(Segment(12, 18), "Carol")),
               ('pierrot', Unit(Segment(12, 18), "Alice")),
               ('hadrien', None)]
    unitary_alignment = UnitaryAlignment(n_tuple)
    assert unitary_alignment.nb_units == 2

    assert (unitary_alignment.compute_disorder(combi_dis)
           ==
           pytest.approx(0.574999988079071, 0.001))


def test_alignment():
    continuum = Continuum()
    annotation = Annotation()
    annotation[Segment(1, 5)] = 'Carol'
    annotation[Segment(6, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Carol'
    annotation[Segment(7, 20)] = 'Alice'
    continuum.add_annotation('liza', annotation)

    annotation = Annotation()
    annotation[Segment(2, 6)] = 'Carol'
    annotation[Segment(7, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Alice'
    annotation[Segment(8, 10)] = 'Alice'
    annotation[Segment(7, 19)] = 'Jeremy'
    continuum.add_annotation('pierrot', annotation)
    annotation = Annotation()

    annotation[Segment(1, 6)] = 'Carol'
    annotation[Segment(8, 10)] = 'Alice'
    annotation[Segment(7, 19)] = 'Jeremy'
    annotation[Segment(19, 20)] = 'Alice'

    continuum.add_annotation('hadrien', annotation)

    categories = SortedSet(['Carol', 'Bob', 'Alice', 'Jeremy'])
    cat = np.array([[0, 0.5, 0.3, 0.7],
                    [0.5, 0., 0.6, 0.4],
                    [0.3, 0.6, 0., 0.7],
                    [0.7, 0.4, 0.7, 0.]])
    combi_dis = CombinedCategoricalDissimilarity(
        delta_empty=0.5,
        alpha=3,
        cat_dissim=PrecomputedCategoricalDissimilarity(categories, cat, delta_empty=0.5))
    set_unitary_alignments = []

    n_tuple = [('liza', Unit(Segment(1, 5), 'Carol')),
               ('pierrot', Unit(Segment(2, 6), 'Carol')),
               ('hadrien', Unit(Segment(1, 6), 'Carol'))]
    unitary_alignment = UnitaryAlignment(n_tuple)
    assert unitary_alignment.nb_units == 3
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = [('liza', Unit(Segment(6, 8), 'Bob')),
               ('pierrot', Unit(Segment(7, 8), 'Bob')),
               ('hadrien', Unit(Segment(8, 10), 'Alice'))]
    unitary_alignment = UnitaryAlignment(n_tuple)
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = [('liza', Unit(Segment(7, 20), 'Alice')),
               ('pierrot', Unit(Segment(7, 19), 'Jeremy')),
               ('hadrien', Unit(Segment(7, 19), 'Jeremy'))]
    unitary_alignment = UnitaryAlignment(n_tuple)
    assert unitary_alignment.nb_units == 3
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = [('liza', Unit(Segment(12, 18), 'Carol')),
               ('pierrot', Unit(Segment(12, 18), 'Alice')),
               ('hadrien', None)]
    unitary_alignment = UnitaryAlignment(n_tuple)
    assert unitary_alignment.nb_units == 2
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = [('liza', None),
               ('pierrot', Unit(Segment(8, 10), 'Alice')),
               ('hadrien', Unit(Segment(19, 20), 'Alice'))]
    unitary_alignment = UnitaryAlignment(n_tuple)
    assert unitary_alignment.nb_units == 2
    set_unitary_alignments.append(unitary_alignment)

    alignment = Alignment(set_unitary_alignments,
                          continuum=continuum,
                          check_validity=True)


    assert (alignment.compute_disorder(combi_dis)
            ==
            pytest.approx(6.1655581547663765, 0.001))


def test_best_alignment():
    continuum = Continuum()
    annotation = Annotation()
    annotation[Segment(1, 5)] = 'Carol'
    annotation[Segment(6, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Carol'
    annotation[Segment(7, 20)] = 'Alice'
    continuum.add_annotation('liza', annotation)

    annotation = Annotation()
    annotation[Segment(2, 6)] = 'Carol'
    annotation[Segment(7, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Alice'
    annotation[Segment(8, 10)] = 'Alice'
    annotation[Segment(7, 19)] = 'Jeremy'
    continuum.add_annotation('pierrot', annotation)
    annotation = Annotation()

    annotation[Segment(1, 6)] = 'Carol'
    annotation[Segment(8, 10)] = 'Alice'
    annotation[Segment(7, 19)] = 'Jeremy'
    annotation[Segment(19, 20)] = 'Alice'

    continuum.add_annotation('hadrien', annotation)

    categories = SortedSet(['Carol', 'Bob', 'Alice', 'Jeremy'])
    cat = np.array([[0, 0.5, 0.3, 0.7],
                    [0.5, 0., 0.6, 0.4],
                    [0.3, 0.6, 0., 0.7],
                    [0.7, 0.4, 0.7, 0.]])
    combi_dis = CombinedCategoricalDissimilarity(
        delta_empty=0.5,
        alpha=3,
        cat_dissim=PrecomputedCategoricalDissimilarity(categories, cat, delta_empty=0.5))
    set_unitary_alignments = []

    n_tuple = [('liza', Unit(Segment(1, 5), 'Carol')),
               ('pierrot', Unit(Segment(2, 6), 'Carol')),
               ('hadrien', Unit(Segment(1, 6), 'Carol'))]
    unitary_alignment = UnitaryAlignment(n_tuple)
    assert unitary_alignment.nb_units == 3
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = [('liza', Unit(Segment(6, 8), 'Bob')),
               ('pierrot', Unit(Segment(7, 8), 'Bob')),
               ('hadrien', Unit(Segment(8, 10), 'Alice'))]
    unitary_alignment = UnitaryAlignment(n_tuple)
    assert unitary_alignment.nb_units == 3
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = [('liza', Unit(Segment(7, 20), 'Alice')),
               ('pierrot', Unit(Segment(7, 19), 'Jeremy')),
               ('hadrien', Unit(Segment(7, 19), 'Jeremy'))]
    unitary_alignment = UnitaryAlignment(n_tuple)
    assert unitary_alignment.nb_units == 3
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = [('liza', Unit(Segment(12, 18), 'Carol')),
               ('pierrot', Unit(Segment(12, 18), 'Alice')),
               ('hadrien', None)]
    unitary_alignment = UnitaryAlignment(n_tuple)
    assert unitary_alignment.nb_units == 2
    set_unitary_alignments.append(unitary_alignment)

    n_tuple = [('liza', None),
               ('pierrot', Unit(Segment(8, 10), 'Alice')),
               ('hadrien', Unit(Segment(19, 20), 'Alice'))]
    unitary_alignment = UnitaryAlignment(n_tuple)
    assert unitary_alignment.nb_units == 2
    set_unitary_alignments.append(unitary_alignment)

    alignment = Alignment(set_unitary_alignments,
                          continuum=continuum,
                          check_validity=True)

    best_alignment = continuum.get_best_alignment(combi_dis)

    assert best_alignment.disorder == pytest.approx(0.43478875578596043,
                                                    0.001)
    assert best_alignment.disorder < alignment.compute_disorder(combi_dis)