"""Tests for the gamma computations"""
from pathlib import Path

import pytest

from pygamma_agreement.continuum import Continuum
from pygamma_agreement.dissimilarity import CombinedCategoricalDissimilarity


def test_gamma_2by1000():
    continuum = Continuum.from_csv(Path("tests/data/2by1000.csv"))
    dissim = CombinedCategoricalDissimilarity(continuum.categories,
                                              delta_empty=1,
                                              alpha=3,
                                              beta=1)

    gamma_results = continuum.compute_gamma(dissim, precision_level=0.5)
    assert len(gamma_results.best_alignment.unitary_alignments) == 1085
    assert 0.45 <= gamma_results.gamma <= 0.49


def test_gamma_3by100():
    continuum = Continuum.from_csv(Path("tests/data/3by100.csv"))
    dissim = CombinedCategoricalDissimilarity(continuum.categories,
                                              delta_empty=1,
                                              alpha=3,
                                              beta=1)

    gamma_results = continuum.compute_gamma(dissim, precision_level=0.5)
    assert len(gamma_results.best_alignment.unitary_alignments) == 127
    assert 0.82 <= gamma_results.gamma <= 0.85


def test_gamma_alexpaulsuzan():
    continuum = Continuum.from_csv(Path("tests/data/AlexPaulSuzan.csv"))
    dissim = CombinedCategoricalDissimilarity(continuum.categories,
                                              delta_empty=1,
                                              alpha=3,
                                              beta=1)

    gamma_results = continuum.compute_gamma(dissim, precision_level=0.01)
    assert len(gamma_results.best_alignment.unitary_alignments) == 7
    assert gamma_results.best_alignment.disorder == pytest.approx(0.96, 0.01)
    assert 0.47 <= gamma_results.gamma <= 0.49
