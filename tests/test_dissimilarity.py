"""Test of the module pygamma.dissimilarity"""

import tempfile
import numpy as np
from pygamma.continuum import Continuum
from pygamma.dissimilarity import (CategoricalDissimilarity,
                                   SequenceDissimilarity,
                                   PositionalDissimilarity,
                                   CombinedCategoricalDissimilarity,
                                   CombinedSequenceDissimilarity)

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

    cat_dis = CategoricalDissimilarity(
        'diarization',
        list_categories=categories,
        categorical_dissimilarity_matrix=cat,
        DELTA_EMPTY=0.5)

    assert cat_dis[('Carol', 'Carol')] == 0.
    assert cat_dis[('Carol', 'Jeremy')] == 0.35
    assert cat_dis[('Carol', 'Jeremy')] == cat_dis[('Jeremy', 'Carol')]


def test_sequence_dissimilarity():
    continuum = Continuum()
    annotation = Annotation()
    annotation[Segment(1, 5)] = ('a', 'b', 'b')
    annotation[Segment(6, 8)] = ('a', 'b')
    annotation[Segment(12, 18)] = ('a', 'c', 'c', 'c')
    annotation[Segment(7, 20)] = ('b', 'b', 'c', 'c', 'a')
    continuum['liza'] = annotation
    annotation = Annotation()
    annotation[Segment(2, 6)] = ('a', 'b', 'b')
    annotation[Segment(7, 8)] = ('a')
    annotation[Segment(12, 18)] = ('b', 'c', 'b', 'c')
    annotation[Segment(8, 10)] = ('a', 'a')
    annotation[Segment(7, 19)] = ('b', 'b', 'a', 'c', 'a')
    continuum['pierrot'] = annotation
    annotation = Annotation()

    annotation[Segment(1, 6)] = ('a', 'b', 'b')
    annotation[Segment(8, 10)] = ('a', 'b')
    annotation[Segment(7, 19)] = ('a', 'c', 'b', 'c')
    annotation[Segment(19, 20)] = ('a', 'c')

    continuum['hadrien'] = annotation
    symbols = ['a', 'b', 'c', 'd']
    cat = np.array([[0, 0.5, 0.3, 0.7], [0.5, 0., 0.6, 0.4],
                    [0.3, 0.6, 0., 0.7], [0.7, 0.4, 0.7, 0.]])

    seq_dis = SequenceDissimilarity('SR', admitted_symbols=symbols,
                                    symbol_dissimilarity_matrix=cat)

    assert seq_dis[(['a', 'c', 'a', 'c'], ['a', 'c', 'a', 'c'])] == 0.0
    assert seq_dis[(['a', 'c', 'a', 'c'], ['a', 'c', 'b', 'c'])] == 0.125
    assert (seq_dis[(['a', 'c', 'a', 'c'], ['a', 'c', 'b', 'c'])] ==
            seq_dis[(['a', 'c', 'b', 'c'], ['a', 'c', 'a', 'c'])])


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

    pos_dis = PositionalDissimilarity('diarization', DELTA_EMPTY=0.5)

    list_dis = []
    for liza_unit in continuum['liza'].itersegments():
        for pierrot_unit in continuum['pierrot'].itersegments():
            list_dis.append(pos_dis[(liza_unit, pierrot_unit)])
    assert list_dis == pytest.approx([
        0.03125, 1.62, 0.78125, 2.0, 2.88, 0.5, 0.05555555555555555,
        0.36734693877551017, 0.5, 2.0, 0.6245674740484429, 0.36734693877551017,
        0.0008, 0.26888888888888884, 0.06786703601108032, 2.4200000000000004,
        2.2959183673469385, 0.05555555555555555, 1.125, 0.0
    ], 0.001)
    assert pos_dis[(liza_unit, pierrot_unit)] == pos_dis[(pierrot_unit,
                                                          liza_unit)]
    assert pos_dis[(liza_unit, liza_unit)] == 0
    assert pos_dis[(liza_unit, )] == 1 * 0.5


def test_positional_dissimilarity_figure10():
    pos_dis = PositionalDissimilarity('diarization', DELTA_EMPTY=1.0)
    assert pos_dis[(Segment(4, 14), Segment(40, 44))] == pytest.approx(22.2, 0.1)
    assert pos_dis[(Segment(4, 14), Segment(4, 14))] == pytest.approx(0., 0.1)
    assert pos_dis[(Segment(4, 14), Segment(20, 25))] == pytest.approx(3.2, 0.1)
    assert pos_dis[(Segment(4, 14), Segment(14, 24))] == pytest.approx(1., 0.1)
    assert pos_dis[(Segment(20, 30), Segment(20, 25))] == pytest.approx(0.11, 0.1)
    assert pos_dis[(Segment(20, 25), Segment(14, 24))] == pytest.approx(0.22, 0.1)


def test_positional_dissimilarity_figure20_scale_effect():
    pos_dis = PositionalDissimilarity('diarization', DELTA_EMPTY=1.0)
    assert pos_dis[(Segment(0, 7), Segment(0, 10))] == pos_dis[(Segment(0, 21), Segment(0, 30))]


def test_combi_categorical_dissimilarity():
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
    combi_dis = CombinedCategoricalDissimilarity(
        'diarization',
        list_categories=categories,
        DELTA_EMPTY=0.5,
        categorical_dissimilarity_matrix=cat)
    list_dis = []
    for liza_unit in continuum['liza'].itersegments():
        for pierrot_unit in continuum['pierrot'].itersegments():
            list_dis.append(combi_dis[(liza_unit, pierrot_unit),
                                      (continuum['liza'][liza_unit],
                                       continuum['pierrot'][pierrot_unit])])
    assert list_dis == pytest.approx([
        0.09375, 5.11, 2.69375, 6.15, 8.790000000000001, 1.75,
        0.16666666666666666, 1.3020408163265305, 1.8, 6.3, 2.0237024221453286,
        1.4020408163265305, 0.3524, 0.8066666666666665, 0.20360110803324097,
        7.260000000000002, 7.137755102040815, 0.5166666666666666, 3.525, 0.15
    ], 0.001)
    assert combi_dis[(liza_unit, pierrot_unit), (
        continuum['liza'][liza_unit],
        continuum['pierrot'][pierrot_unit])] == combi_dis[(
                                                              pierrot_unit, liza_unit), (continuum['pierrot'][pierrot_unit],
                                                                                         continuum['liza'][liza_unit])]
    assert combi_dis[(liza_unit,
                      liza_unit), (continuum['liza'][liza_unit],
                                   continuum['liza'][liza_unit])] == 0
    assert combi_dis[(liza_unit, ), (
        continuum['liza'][liza_unit], )] == 1 * 0.5


def test_combi_sequence_dissimilarity():
    continuum = Continuum()
    annotation = Annotation()
    annotation[Segment(1, 5)] = ('a', 'b', 'b')
    annotation[Segment(6, 8)] = ('a', 'b')
    annotation[Segment(12, 18)] = ('a', 'c', 'c', 'c')
    annotation[Segment(7, 20)] = ('b', 'b', 'c', 'c', 'a')
    continuum['liza'] = annotation
    annotation = Annotation()
    annotation[Segment(2, 6)] = ('a', 'b', 'b')
    annotation[Segment(7, 8)] = ('a')
    annotation[Segment(12, 18)] = ('b', 'c', 'b', 'c')
    annotation[Segment(8, 10)] = ('a', 'a')
    annotation[Segment(7, 19)] = ('b', 'b', 'a', 'c', 'a')
    continuum['pierrot'] = annotation
    annotation = Annotation()

    annotation[Segment(1, 6)] = ('a', 'b', 'b')
    annotation[Segment(8, 10)] = ('a', 'b')
    annotation[Segment(7, 19)] = ('a', 'c', 'b', 'c')
    annotation[Segment(19, 20)] = ('a', 'c')

    continuum['hadrien'] = annotation

    symbols = ['a', 'b', 'c', 'd']
    cat = np.array([[0, 0.5, 0.3, 0.7], [0.5, 0., 0.6, 0.4],
                    [0.3, 0.6, 0., 0.7], [0.7, 0.4, 0.7, 0.]])
    DELTA_EMPTY = 0.5
    alpha = 1
    beta = 1
    combi_dis = CombinedSequenceDissimilarity(
        'SR',
        list_admitted_symbols=symbols,
        DELTA_EMPTY=0.5,
        symbol_dissimilarity_matrix=cat)
    list_dis = []
    for liza_unit in continuum['liza'].itersegments():
        for pierrot_unit in continuum['pierrot'].itersegments():
            list_dis.append(combi_dis[(liza_unit, pierrot_unit), (
                continuum['liza'][liza_unit],
                continuum['pierrot'][pierrot_unit])])
    assert list_dis == pytest.approx([
        0.09375, 5.193333333333333, 2.64375, 6.25, 8.877500000000001,
        1.6666666666666667, 0.41666666666666663, 1.4520408163265306, 1.625,
        6.2875, 2.1737024221453285, 1.5020408163265304, 0.0324,
        1.1366666666666665, 0.393601108033241, 7.535000000000002,
        7.262755102040815, 0.3766666666666667, 3.6625, 0.1375
    ], 0.001)
    assert combi_dis[(liza_unit, pierrot_unit), (
        continuum['liza'][liza_unit],
        continuum['pierrot'][pierrot_unit])] == combi_dis[(
                                                              pierrot_unit, liza_unit), (continuum['pierrot'][pierrot_unit],
                                                                                         continuum['liza'][liza_unit])]
    assert combi_dis[(liza_unit,
                      liza_unit), (continuum['liza'][liza_unit],
                                   continuum['liza'][liza_unit])] == 0
    assert combi_dis[(liza_unit, ), (
        continuum['liza'][liza_unit], )] == DELTA_EMPTY
