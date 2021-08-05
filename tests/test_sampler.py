"""Test for the different continuum samplers"""
from pathlib import Path
import numpy as np
from pygamma_agreement.continuum import Continuum

from pygamma_agreement.dissimilarity import CombinedCategoricalDissimilarity
from pygamma_agreement.sampler import ShuffleContinuumSampler, StatisticalContinuumSampler
from sortedcontainers import SortedSet


def test_mathet_sampler():
    np.random.seed(4778)
    continuum = Continuum.from_csv(Path("tests/data/AlexPaulSuzan.csv"))

    sampler = ShuffleContinuumSampler(pivot_type='float_pivot')
    sampler.init_sampling(continuum,
                          ground_truth_annotators=SortedSet(("Paul", "Suzan")))
    new_continuum = sampler.sample_from_continuum
    assert new_continuum.categories.issubset(continuum.categories)
    assert len(new_continuum.annotators) == 2
    dissim = CombinedCategoricalDissimilarity(delta_empty=1,
                                              alpha=3,
                                              beta=1)
    gamma_results = new_continuum.compute_gamma(dissim,
                                                sampler=ShuffleContinuumSampler(),
                                                precision_level=0.05)
    assert gamma_results.gamma < 0.2
    assert gamma_results.gamma_cat < 0.2


def test_statistical_sampler():
    np.random.seed(4778)
    continuum = Continuum.from_csv(Path("tests/data/AlexPaulSuzan.csv"))
    sampler = StatisticalContinuumSampler()
    sampler.init_sampling(continuum)

    new_continuum = sampler.sample_from_continuum
    assert len(new_continuum.annotators) == 3
    assert abs(continuum.avg_length_unit - new_continuum.avg_length_unit) <= 3
    assert abs(continuum.avg_num_annotations_per_annotator - new_continuum.avg_num_annotations_per_annotator) <= 3
    assert new_continuum.categories.issubset(continuum.categories)
    dissim = CombinedCategoricalDissimilarity(delta_empty=1,
                                              alpha=3,
                                              beta=1)

    gamma_results = new_continuum.compute_gamma(dissim,
                                                precision_level=0.05)
    assert gamma_results.gamma < 0.1
    assert gamma_results.gamma_cat < 0.1

    gamma_results = continuum.compute_gamma(dissim, sampler=sampler, precision_level=0.05)
    # Gamma:
    assert 0.41 <= gamma_results.gamma <= 0.42
    # Gamma-cat:
    assert 0.61 <= gamma_results.gamma_cat <= 0.64


def test_statistical_sampler_manual():
    np.random.seed(7455)
    sampler = StatisticalContinuumSampler()
    sampler.init_sampling_custom(avg_duration=10, avg_gap=5, avg_num_units_per_annotator=30,
                                 annotators=['Martin', 'Martino', 'Martine'],
                                 std_duration=3, std_gap=5, std_num_units_per_annotator=5,
                                 categories=np.array(['Verb', 'Noun', 'Prep', 'Adj']),
                                 categories_weight=np.array([0.1, 0.4, 0.2, 0.3]))
    continuum = sampler.sample_from_continuum
    gamma_results = continuum.compute_gamma()
    gamma = gamma_results.gamma
    gamma_cat = gamma_results.gamma_cat
    for category in ['Verb', 'Noun', 'Prep', 'Adj']:
        gamma_k = gamma_results.gamma_k(category)




