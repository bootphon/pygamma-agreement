"""Tests for the gamma computations"""
from pathlib import Path

import numpy as np
import pytest

from pygamma_agreement.continuum import Continuum
from pygamma_agreement.dissimilarity import (CombinedCategoricalDissimilarity,
                                             PositionalSporadicDissimilarity,
                                             NumericalCategoricalDissimilarity,
                                             LevenshteinCategoricalDissimilarity,
                                             OrdinalCategoricalDissimilarity)
from pygamma_agreement.sampler import ShuffleContinuumSampler
from pyannote.core import Segment


def test_gamma_2by1000():
    np.random.seed(4772)
    continuum = Continuum.from_csv(Path("tests/data/2by1000.csv"))
    dissim = CombinedCategoricalDissimilarity(delta_empty=1,
                                              alpha=3,
                                              beta=1)

    gamma_results = continuum.compute_gamma(dissim)
    assert len(gamma_results.best_alignment.unitary_alignments) == 1085

    # Gamma:
    assert 0.39 <= gamma_results.gamma <= 0.42
    # Gamma-cat:
    assert 0.38 <= gamma_results.gamma_cat <= 0.42
    # Gamma_k's
    assert 0.21 <= gamma_results.gamma_k('Adj') <= 0.25
    assert 0.36 <= gamma_results.gamma_k('Noun') <= 0.39
    assert 0.31 <= gamma_results.gamma_k('Prep') <= 0.35
    assert 0.11 <= gamma_results.gamma_k('Verb') <= 0.15


def test_gamma_3by100():
    np.random.seed(4772)
    continuum = Continuum.from_csv(Path("tests/data/3by100.csv"))
    dissim = CombinedCategoricalDissimilarity(delta_empty=1,
                                              alpha=3,
                                              beta=1)

    gamma_results = continuum.compute_gamma(dissim)
    assert len(gamma_results.best_alignment.unitary_alignments) == 127
    # Gamma
    assert 0.79 <= gamma_results.gamma <= 0.81
    # Gamma-cat:
    assert 0.89 <= gamma_results.gamma_cat <= 0.91
    # Gamma_k's
    assert 0.81 <= gamma_results.gamma_k('Adj') <= 0.83
    assert 0.88 <= gamma_results.gamma_k('Noun') <= 0.90
    assert 0.69 <= gamma_results.gamma_k('Verb') <= 0.71
    assert 0.96 <= gamma_results.gamma_k('Prep')


def test_gamma_alexpaulsuzan():
    np.random.seed(4772)
    continuum = Continuum.from_csv(Path("tests/data/AlexPaulSuzan.csv"))
    dissim = CombinedCategoricalDissimilarity(delta_empty=1,
                                              alpha=3,
                                              beta=1)
    sampler = ShuffleContinuumSampler()
    gamma_results = continuum.compute_gamma(dissim, sampler=sampler, precision_level=0.05)
    assert len(gamma_results.best_alignment.unitary_alignments) == 6

    assert gamma_results.best_alignment.disorder == pytest.approx(0.96, 0.01)
    # Gamma:
    assert 0.44 <= gamma_results.gamma <= 0.47
    # Gamma-cat:
    assert 0.38 <= gamma_results.gamma_cat <= 0.42
    # Gamma-k's
    gamma_ks = {'1': 1, '2': 0, '3': 0, '4': 0, '5': 1, '6': 0.37, '7': 0}
    for category, gk in gamma_ks.items():
        if gk != 0:
            print(category)
            assert gk - 0.01 <= gamma_results.gamma_k(category) <= gk + 0.01
        else:
            assert gamma_results.gamma_k(category) <= 0
    # assert gamma_results.gamma_k('7') is np.NaN


def test_gamma_alexpaulsuzan_otherdissims():
    np.random.seed(4772)
    continuum = Continuum.from_csv(Path("tests/data/AlexPaulSuzan.csv"))
    dissimilarity = PositionalSporadicDissimilarity()

    gamma_results = continuum.compute_gamma(dissimilarity=dissimilarity, precision_level=0.05)
    gamma = gamma_results.gamma
    with pytest.raises(Exception):
        gamma_cat = gamma_results.gamma_cat

    dissimilarity = NumericalCategoricalDissimilarity(continuum.categories)

    gamma_results = continuum.compute_gamma(dissimilarity=dissimilarity, precision_level=0.05)

    gamma = gamma_results.gamma
    with pytest.raises(Exception):
        gamma_cat = gamma_results.gamma_cat

    dissimilarity = LevenshteinCategoricalDissimilarity(continuum.categories)
    best_alignment = continuum.get_best_alignment(dissimilarity)

    dissimilarity = OrdinalCategoricalDissimilarity(continuum.categories)
    best_alignment = continuum.get_best_alignment(dissimilarity)

def test_too_many_unitary_alignments():
    a = [['annotator_1', 1240, 1269, 'label_1'],
         ['annotator_1', 1270, 1273, 'label_2'],
         ['annotator_1', 1274, 1275, 'label_1'],
         ['annotator_1', 1280, 1294, 'label_1'],
         ['annotator_1', 1295, 1308, 'label_1'],
         ['annotator_1', 1309, 1322, 'label_2'],
         ['annotator_1', 1350, 1362, 'label_2'],
         ['annotator_1', 1363, 1376, 'label_1'],
         ['annotator_1', 1377, 1380, 'label_2'],
         ['annotator_1', 1381, 1385, 'label_1'],
         ['annotator_1', 1414, 1423, 'label_3'],
         ['annotator_1', 1424, 1454, 'label_1'],
         ['annotator_1', 1428, 1437, 'label_3'],
         ['annotator_1', 1442, 1454, 'label_3'],
         ['annotator_2', 1240, 1269, 'label_1'],
         ['annotator_2', 1270, 1273, 'label_2'],
         ['annotator_2', 1274, 1275, 'label_1'],
         ['annotator_2', 1280, 1294, 'label_1'],
         ['annotator_2', 1295, 1308, 'label_1'],
         ['annotator_2', 1309, 1322, 'label_2'],
         ['annotator_2', 1350, 1362, 'label_2'],
         ['annotator_2', 1363, 1366, 'label_2'],
         ['annotator_2', 1363, 1376, 'label_1'],
         ['annotator_2', 1377, 1380, 'label_2'],
         ['annotator_2', 1381, 1385, 'label_1'],
         ['annotator_2', 1414, 1423, 'label_3'],
         ['annotator_2', 1424, 1454, 'label_1'],
         ['annotator_2', 1428, 1437, 'label_3'],
         ['annotator_2', 1442, 1454, 'label_3'],
         ['annotator_3', 1240, 1269, 'label_1'],
         ['annotator_3', 1270, 1273, 'label_2'],
         ['annotator_3', 1274, 1275, 'label_1'],
         ['annotator_3', 1280, 1294, 'label_1'],
         ['annotator_3', 1295, 1308, 'label_1'],
         ['annotator_3', 1309, 1322, 'label_2'],
         ['annotator_3', 1323, 1343, 'label_1'],
         ['annotator_3', 1350, 1362, 'label_2'],
         ['annotator_3', 1363, 1376, 'label_1'],
         ['annotator_3', 1377, 1380, 'label_2'],
         ['annotator_3', 1381, 1385, 'label_1'],
         ['annotator_3', 1414, 1423, 'label_3'],
         ['annotator_3', 1424, 1437, 'label_1'],
         ['annotator_3', 1428, 1437, 'label_3'],
         ['annotator_3', 1442, 1454, 'label_2'],
         ['annotator_3', 1442, 1454, 'label_1'],
         ['annotator_4', 1240, 1269, 'label_1'],
         ['annotator_4', 1270, 1273, 'label_2'],
         ['annotator_4', 1274, 1275, 'label_1'],
         ['annotator_4', 1280, 1294, 'label_1'],
         ['annotator_4', 1295, 1308, 'label_1'],
         ['annotator_4', 1309, 1322, 'label_2'],
         ['annotator_4', 1350, 1362, 'label_2'],
         ['annotator_4', 1363, 1376, 'label_1'],
         ['annotator_4', 1377, 1380, 'label_2'],
         ['annotator_4', 1381, 1385, 'label_1'],
         ['annotator_4', 1414, 1423, 'label_3'],
         ['annotator_4', 1428, 1437, 'label_3'],
         ['annotator_4', 1442, 1454, 'label_3'],
         ['annotator_5', 1240, 1269, 'label_1'],
         ['annotator_5', 1270, 1273, 'label_2'],
         ['annotator_5', 1274, 1275, 'label_1'],
         ['annotator_5', 1280, 1294, 'label_1'],
         ['annotator_5', 1295, 1308, 'label_1'],
         ['annotator_5', 1309, 1322, 'label_2'],
         ['annotator_5', 1350, 1362, 'label_2'],
         ['annotator_5', 1363, 1376, 'label_1'],
         ['annotator_5', 1377, 1380, 'label_2'],
         ['annotator_5', 1381, 1385, 'label_1'],
         ['annotator_5', 1414, 1423, 'label_3'],
         ['annotator_5', 1424, 1437, 'label_1'],
         ['annotator_5', 1428, 1437, 'label_3'],
         ['annotator_5', 1442, 1454, 'label_3'],
         ['annotator_5', 1442, 1454, 'label_1']]

    continuum = Continuum()

    for ann in a:
        continuum.add(ann[0], Segment(ann[1], ann[2]), ann[3])

    diss = CombinedCategoricalDissimilarity()

    continuum.compute_gamma(diss)
