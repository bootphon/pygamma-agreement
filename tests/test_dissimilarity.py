"""Test of the module pygamma_agreement.dissimilarity"""

import numpy as np
import pytest
from pyannote.core import Annotation, Segment

from pygamma_agreement.alignment import (UnitaryAlignment)
from pygamma_agreement.continuum import Continuum, Unit
from pygamma_agreement.dissimilarity import (PositionalDissimilarity,
                                   CombinedCategoricalDissimilarity)


def test_categorical_dissimilarity():
    categories = ['A', 'B', 'C', 'D']
    cat = np.array([[0, 0.5, 0.3, 0.7], [0.5, 0., 0.6, 0.4],
                    [0.3, 0.6, 0., 0.7], [0.7, 0.4, 0.7, 0.]])

    cat_dis = CombinedCategoricalDissimilarity(
        categories=categories,
        cat_dissimilarity_matrix=cat,
        delta_empty=0.5,
        alpha=0.0,
        beta=1.0)
    fake_seg = Segment(0, 1)
    unit_alignment = UnitaryAlignment((("Carol", Unit(fake_seg, "A")),
                                       ("John", Unit(fake_seg, "D"))))
    assert unit_alignment.compute_disorder(cat_dis) == np.float32(0.35)

    unit_alignment = UnitaryAlignment((("Carol", Unit(fake_seg, "A")),
                                       ("John", Unit(fake_seg, "A"))))
    assert unit_alignment.compute_disorder(cat_dis) == np.float32(0.0)

    unit_alignment_a = UnitaryAlignment((("Carol", Unit(fake_seg, "A")),
                                         ("John", Unit(fake_seg, "B"))))
    unit_alignment_b = UnitaryAlignment((("Carol", Unit(fake_seg, "B")),
                                         ("John", Unit(fake_seg, "A"))))
    assert (unit_alignment_a.compute_disorder(cat_dis)
            ==
            unit_alignment_b.compute_disorder(cat_dis))


def test_positional_dissimilarity():
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

    pos_dis = PositionalDissimilarity(delta_empty=0.5)

    list_dis = []
    for liza_unit in continuum['liza']:
        for pierrot_unit in continuum['pierrot']:
            unit_alignment = UnitaryAlignment((("liza", liza_unit),
                                               ("pierrot", pierrot_unit)))
            list_dis.append(unit_alignment.compute_disorder(pos_dis))
    assert list_dis == pytest.approx([
        0.03125, 1.62, 0.78125, 2.0, 2.88, 0.5, 0.05555555555555555,
        0.36734693877551017, 0.5, 2.0, 0.6245674740484429, 0.36734693877551017,
        0.0008, 0.26888888888888884, 0.06786703601108032, 2.4200000000000004,
        2.2959183673469385, 0.05555555555555555, 1.125, 0.0
    ], 0.001)

    liza_unit = Unit(Segment(45, 77), "")
    pierrot_unit = Unit(Segment(16, 64), "")
    unit_alignment_a = UnitaryAlignment((("liza", liza_unit),
                                         ("pierrot", pierrot_unit)))
    unit_alignment_b = UnitaryAlignment((("pierrot", pierrot_unit),
                                         ("liza", liza_unit)))
    assert (unit_alignment_a.compute_disorder(pos_dis)
            ==
            unit_alignment_b.compute_disorder(pos_dis))

    unit_alignment = UnitaryAlignment((("liza", liza_unit),
                                       ("pierrot", liza_unit)))
    assert unit_alignment.compute_disorder(pos_dis) == np.float32(0.0)


def test_positional_dissimilarity_figure10():
    pos_dis = PositionalDissimilarity(delta_empty=1.0)
    segments = {
        (Segment(4, 14), Segment(40, 44)): 22.2,
        (Segment(4, 14), Segment(4, 14)): 0.,
        (Segment(4, 14), Segment(20, 25)): 3.2,
        (Segment(4, 14), Segment(14, 24)): 1.,
        (Segment(20, 30), Segment(20, 25)): 0.11,
        (Segment(20, 25), Segment(14, 24)): 0.22,
    }
    for (seg_a, seg_b), value in segments.items():
        unit_alignment = UnitaryAlignment((("liza", Unit(seg_a)),
                                           ("pierrot", Unit(seg_b))))
        assert unit_alignment.compute_disorder(pos_dis) == pytest.approx(value, 0.1)


def test_positional_dissimilarity_figure20_scale_effect():
    pos_dis = PositionalDissimilarity(delta_empty=1.0)
    unit_align_a = UnitaryAlignment((("pierrot", Unit(Segment(0, 7))),
                                     ("liza", Unit(Segment(0, 10)))))
    unit_align_b = UnitaryAlignment((("pierrot", Unit(Segment(0, 21))),
                                     ("liza", Unit(Segment(0, 30)))))

    assert (unit_align_a.compute_disorder(pos_dis)
            ==
            unit_align_b.compute_disorder(pos_dis))


def test_combi_categorical_dissimilarity():
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
    categories = ['Carol', 'Bob', 'Alice', 'Jeremy']

    cat = np.array([[0, 0.5, 0.3, 0.7],
                    [0.5, 0., 0.6, 0.4],
                    [0.3, 0.6, 0., 0.7],
                    [0.7, 0.4, 0.7, 0.]])
    combi_dis = CombinedCategoricalDissimilarity(
        categories=categories,
        delta_empty=0.5,
        cat_dissimilarity_matrix=cat,
        alpha=3, beta=1)
    list_dis = []
    for liza_unit in continuum['liza']:
        for pierrot_unit in continuum['pierrot']:
            unit_alignment = UnitaryAlignment((("liza", liza_unit),
                                               ("pierrot", pierrot_unit)))
            list_dis.append(unit_alignment.compute_disorder(combi_dis))
    print(len(list_dis))
    assert list_dis == pytest.approx([
        0.09375, 5.11, 2.69375, 6.15, 8.790000000000001, 1.75,
        0.16666666666666666, 1.3020408163265305, 1.8, 6.3, 2.0237024221453286,
        1.4020408163265305, 0.3524, 0.8066666666666665, 0.20360110803324097,
        7.260000000000002, 7.137755102040815, 0.5166666666666666, 3.525, 0.15
    ], 0.001)

    unit_align_a = UnitaryAlignment((("liza", Unit(Segment(1, 5), "Carol")),
                                     ("pierrot", Unit(Segment(7, 19), "Jeremy"))))
    unit_align_b = UnitaryAlignment((("pierrot", Unit(Segment(7, 19), "Jeremy")),
                                     ("liza", Unit(Segment(1, 5), "Carol")),))
    assert (unit_align_a.compute_disorder(combi_dis)
            ==
            unit_align_b.compute_disorder(combi_dis))

    same_align = UnitaryAlignment((("liza", Unit(Segment(1, 5), "Carol")),
                                   ("pierrot", Unit(Segment(1, 5), "Carol"))))

    assert same_align.compute_disorder(combi_dis) == np.float32(0.0)
