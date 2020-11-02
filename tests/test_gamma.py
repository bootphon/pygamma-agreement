"""Tests for the gamma computations"""
from pathlib import Path

import pytest

from pygamma_agreement.continuum import Continuum
from pygamma_agreement.dissimilarity import CombinedCategoricalDissimilarity


def test_gamma_2by1000():
    continuum = Continuum.from_csv(Path("tests/data/2by1000.csv"))
    dissim = CombinedCategoricalDissimilarity(list(continuum.categories),
                                              delta_empty=1,
                                              alpha=3,
                                              beta=1)

    best_alignment = continuum.get_best_alignment(dissim)
    gamma_results = continuum.compute_gamma(dissim, precision_level=0.5)
    assert len(best_alignment.unitary_alignments) == 1085
    assert 0.20 <= gamma_results.gamma <= 0.27


def test_gamma_3by100():
    continuum = Continuum.from_csv(Path("tests/data/3by100.csv"))
    dissim = CombinedCategoricalDissimilarity(list(continuum.categories),
                                              delta_empty=1,
                                              alpha=3,
                                              beta=1)

    best_alignment = continuum.get_best_alignment(dissim)
    gamma_results = continuum.compute_gamma(dissim, precision_level=0.5)
    assert len(best_alignment.unitary_alignments) == 127
    assert 0.68 <= gamma_results.gamma <= 0.70


def test_gamma_alexpaulsuzan():
    continuum = Continuum.from_csv(Path("tests/data/AlexPaulSuzan.csv"))
    dissim = CombinedCategoricalDissimilarity(list(continuum.categories),
                                              delta_empty=1,
                                              alpha=3,
                                              beta=1)

    best_alignment = continuum.get_best_alignment(dissim)
    gamma_results = continuum.compute_gamma(dissim, precision_level=0.01)
    assert len(best_alignment.unitary_alignments) == 7
    assert best_alignment.disorder == pytest.approx(0.731, 0.01)
    assert 0.28 <= gamma_results.gamma <= 0.31
