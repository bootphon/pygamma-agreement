"""Test for the different continuum samplers"""
from pathlib import Path

from pygamma_agreement.continuum import Continuum
from pygamma_agreement.dissimilarity import CombinedCategoricalDissimilarity
from pygamma_agreement.cst import CorpusShufflingTool, random_reference
from pygamma_agreement.cat_dissim import cat_ord
from pygamma_agreement.sampler import *
from pygamma_agreement.notebook import continuum_png
from sortedcontainers import SortedSet

def test_mathet_sampler():
    np.random.seed(4778)
    continuum = Continuum.from_csv(Path("tests/data/AlexPaulSuzan.csv"))

    sampler = MathetContinuumSampler(continuum,
                                     ground_truth_annotators=SortedSet(("Paul", "Suzan")),
                                     pivot_type='float_pivot')
    new_continuum = sampler.sample_from_continuum
    assert new_continuum.categories.issubset(continuum.categories)
    assert len(new_continuum.annotators) == 2
    dissim = CombinedCategoricalDissimilarity(continuum.categories,
                                              delta_empty=1,
                                              alpha=3,
                                              beta=1)
    gamma_results = new_continuum.compute_gamma(dissim, precision_level=0.01)
    assert gamma_results.gamma < 0.1
    assert gamma_results.gamma_cat < 0.1

def test_statistical_sampler():
    np.random.seed(4778)
    continuum = Continuum.from_csv(Path("tests/data/AlexPaulSuzan.csv"))
    sampler = StatisticalContinuumSampler(continuum)

    new_continuum = sampler.sample_from_continuum
    assert len(new_continuum.annotators) == 3
    assert abs(continuum.avg_length_unit - new_continuum.avg_length_unit) <= 3
    assert abs(continuum.avg_num_annotations_per_annotator - new_continuum.avg_num_annotations_per_annotator) <= 3
    assert new_continuum.categories.issubset(continuum.categories)
    dissim = CombinedCategoricalDissimilarity(continuum.categories,
                                              delta_empty=1,
                                              alpha=3,
                                              beta=1)

    gamma_results = new_continuum.compute_gamma(dissim, precision_level=0.01)
    assert gamma_results.gamma < 0.1
    assert gamma_results.gamma_cat < 0.1

    gamma_results = continuum.compute_gamma(dissim, sampler=sampler, precision_level=0.01)
    # Gamma:
    assert 0.41 <= gamma_results.gamma <= 0.44
    # Gamma-cat:
    assert 0.61 <= gamma_results.gamma_cat <= 0.64




