"""Tests for the CST & generation"""
from pathlib import Path
import random

from pygamma_agreement.continuum import Continuum
from pygamma_agreement.dissimilarity import CombinedCategoricalDissimilarity
from pygamma_agreement.cst import CorpusShufflingTool, random_reference
from sortedcontainers import SortedSet

def test_random_reference():
    categories = SortedSet(("aa", "ab", "ba", "bb"))
    continuum = random_reference("Martino", 200, 40, 10, 1,
                                 categories,
                                 seed=random.randint(0, 10000),
                                 overlapping=False)
    assert continuum.categories == categories
    assert continuum.num_annotators == 1
    assert continuum.annotators[0] == "Martino"

    # this assertion isnt deterministic but probability of passing/failing by chance is like 6.25e-05
    assert abs(continuum.avg_length_unit - 10) < 20

    assert continuum.avg_num_annotations_per_annotator == 40
    assert continuum.num_units == 40

def test_cst():
    """
    This test only checks deterministic possibilities. A tool for benchmarking gamma
    using the CST is also provided for more experimental tests.
    """
    continuum = Continuum.from_csv(Path("tests/data/AlexPaulSuzan.csv"))
    categories = continuum.categories
    dissim = CombinedCategoricalDissimilarity(list(categories),
                                              delta_empty=1,
                                              alpha=3,
                                              beta=1)
    cst_alex = CorpusShufflingTool(0.0, continuum)  # alex is reference

    # A shuffle with magnitude 0.0 just copies the reference.
    shuffled = cst_alex.corpus_shuffle(["Martino", "Martingale"],
                                       shift=True,
                                       false_positive=True,
                                       false_negative=True,
                                       category=True,
                                       splits=True)
    shuffled_with_reference = cst_alex.corpus_shuffle(["Martino", "Martingale"],
                                                      shift=True,
                                                      false_positive=True,
                                                      false_negative=True,
                                                      category=True,
                                                      splits=True,
                                                      add_reference=True)
    assert shuffled_with_reference.annotators == SortedSet(("Alex", "Martingale", "Martino"))
    assert shuffled.annotators == SortedSet(("Martingale", "Martino"))
    assert shuffled_with_reference.categories.issubset(categories)
    assert shuffled.categories.issubset(categories)
    assert shuffled_with_reference.compute_gamma(dissim).gamma > 0.99
    assert shuffled.compute_gamma(dissim).gamma > 0.99

    cst_alex.magnitude = 1.0
    shuffled = cst_alex.corpus_shuffle(["Martino", "Martingale"],
                                       shift=True,
                                       false_positive=True,
                                       false_negative=True,
                                       category=True,
                                       splits=True)
    shuffled_with_reference = cst_alex.corpus_shuffle(["Martino", "Martingale"],
                                                      shift=True,
                                                      false_positive=True,
                                                      false_negative=True,
                                                      category=True,
                                                      splits=True)
    assert shuffled.compute_gamma(dissim).gamma < 0.01
    assert shuffled_with_reference.compute_gamma(dissim).gamma < 0.01






