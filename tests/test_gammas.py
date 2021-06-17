"""Tests for the gamma computations"""
from pathlib import Path

import numpy as np
import pytest

from pygamma_agreement.continuum import Continuum
from pygamma_agreement.dissimilarity import CombinedCategoricalDissimilarity


def test_gamma_2by1000():
    np.random.seed(4772)
    continuum = Continuum.from_csv(Path("tests/data/2by1000.csv"))
    dissim = CombinedCategoricalDissimilarity(continuum.categories,
                                              delta_empty=1,
                                              alpha=3,
                                              beta=1)

    gamma_results = continuum.compute_gamma(dissim, precision_level=0.5)
    assert len(gamma_results.best_alignment.unitary_alignments) == 1085

    # Gamma:
    assert 0.38 <= gamma_results.gamma <= 0.41
    # Gamma-cat:
    assert 0.38 <= gamma_results.gamma_cat <= 0.42
    # Gamma_k's
    assert 0.21 <= gamma_results.gamma_k('Adj') <= 0.25
    assert 0.33 <= gamma_results.gamma_k('Noun') <= 0.37
    assert 0.31 <= gamma_results.gamma_k('Prep') <= 0.35
    assert 0.11 <= gamma_results.gamma_k('Verb') <= 0.15

def test_gamma_3by100():
    np.random.seed(4772)
    continuum = Continuum.from_csv(Path("tests/data/3by100.csv"))
    dissim = CombinedCategoricalDissimilarity(continuum.categories,
                                              delta_empty=1,
                                              alpha=3,
                                              beta=1)

    gamma_results = continuum.compute_gamma(dissim, precision_level=0.5)
    assert len(gamma_results.best_alignment.unitary_alignments) == 127
    # Gamma
    assert 0.79 <= gamma_results.gamma <= 0.81
    # Gamma-cat:
    assert 0.96 <= gamma_results.gamma_cat
    # Gamma_k's
    assert 0.96 <= gamma_results.gamma_k('Adj')
    assert 0.94 <= gamma_results.gamma_k('Noun')
    assert 0.86 <= gamma_results.gamma_k('Verb') <= 0.90
    assert 0.96 <= gamma_results.gamma_k('Prep')


def test_gamma_alexpaulsuzan():
    np.random.seed(4772)
    continuum = Continuum.from_csv(Path("tests/data/AlexPaulSuzan.csv"))

    gamma_results = continuum.compute_gamma(precision_level=0.01)
    assert len(gamma_results.best_alignment.unitary_alignments) == 7

    assert gamma_results.best_alignment.disorder == pytest.approx(0.96, 0.01)
    # Gamma:
    assert 0.44 <= gamma_results.gamma <= 0.47
    # Gamma-cat:
    assert 0.67 <= gamma_results.gamma_cat <= 0.70
    # Gamma-k's
    gamma_ks = {'1': 1, '2': 0, '3': 0, '4': 0, '5': 1, '6': 1}
    for i in range(1, 7):
        assert gamma_ks[str(i)] == gamma_results.gamma_k(str(i))
    # assert gamma_results.gamma_k('7') is np.NaN
