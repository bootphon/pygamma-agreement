"""Tests for the CST & generation"""
import random
from pathlib import Path

from pygamma_agreement.continuum import Continuum
from pygamma_agreement.dissimilarity import CombinedCategoricalDissimilarity
from pygamma_agreement.cst import CorpusShufflingTool, random_reference
from pygamma_agreement.cat_dissim import cat_ord
from pygamma_agreement.notebook import continuum_png
from sortedcontainers import SortedSet


def test_random_reference():
    categories = SortedSet(("aa", "ab", "ba", "bb"))
    for _ in range(10):  # we do it a certain number of time to be sure no chance happened
        continuum = random_reference("Martino", 200, 40, 10, 1,
                                     categories,
                                     overlapping=False)
        assert continuum.categories == categories
        assert continuum.num_annotators == 1
        assert continuum.annotators[0] == "Martino"

        # this assertion isnt deterministic but probability of passing/failing by chance is like 6.25e-05
        assert abs(continuum.avg_length_unit - 10) < 20

        assert continuum.avg_num_annotations_per_annotator == 40
        assert continuum.num_units == 40


def test_cst_0():
    """
    This test only checks deterministic possibilities. A tool for benchmarking gamma
    using the CST is also provided for more experimental tests.
    """
    continuum = Continuum.from_csv(Path("tests/data/annotation_paul_suzann_alex.csv"))
    categories = continuum.categories
    dissim = CombinedCategoricalDissimilarity(categories,
                                              delta_empty=1,
                                              alpha=3,
                                              beta=1,
                                              cat_dissimilarity_matrix=cat_ord)
    cst_alex = CorpusShufflingTool(0.0, continuum)  # alex is reference

    # A shuffle with magnitude 0.0 just copies the reference.
    shuffled = cst_alex.corpus_shuffle(["Martino", "Martingale"],
                                       shift=True,
                                       false_pos=True,
                                       false_neg=True,
                                       cat_shuffle=True,
                                       split=True)
    shuffled_with_reference = cst_alex.corpus_shuffle(["Martino", "Martingale"],
                                                      shift=True,
                                                      false_pos=True,
                                                      false_neg=True,
                                                      cat_shuffle=True,
                                                      split=True,
                                                      include_ref=True)
    assert shuffled_with_reference.annotators == SortedSet(("Alex", "Martingale", "Martino"))
    assert shuffled.annotators == SortedSet(("Martingale", "Martino"))
    assert shuffled_with_reference.categories.issubset(categories)
    assert shuffled.categories.issubset(categories)
    assert shuffled_with_reference.compute_gamma(dissim).gamma > 0.9
    assert shuffled.compute_gamma(dissim).gamma > 0.9


def test_cst_1():
    continuum = Continuum.from_csv(Path("tests/data/annotation_paul_suzann_alex.csv"))
    categories = continuum.categories
    dissim = CombinedCategoricalDissimilarity(categories,
                                              delta_empty=1,
                                              alpha=3,
                                              beta=1,
                                              cat_dissimilarity_matrix=cat_ord)
    cst_alex = CorpusShufflingTool(1.0, continuum)  # alex is reference
    shuffled = cst_alex.corpus_shuffle(["Martino", "Martingale"],
                                       shift=True,
                                       false_pos=True,
                                       false_neg=True,
                                       cat_shuffle=True,
                                       split=True)
    shuffled_with_reference = cst_alex.corpus_shuffle(["Martino", "Martingale"],
                                                      shift=True,
                                                      false_pos=True,
                                                      false_neg=True,
                                                      cat_shuffle=True,
                                                      split=True,
                                                      include_ref=True)
    assert shuffled.compute_gamma(dissim).gamma < 0.2
    assert shuffled_with_reference.compute_gamma(dissim).gamma < 0.2


def test_cst_cat():
    continuum = Continuum.from_csv(Path("tests/data/annotation_paul_suzann_alex.csv"))
    categories = continuum.categories
    dissim = CombinedCategoricalDissimilarity(categories,
                                              delta_empty=1,
                                              alpha=3,
                                              beta=3,
                                              cat_dissimilarity_matrix=cat_ord)
    cst_alex = CorpusShufflingTool(1.0, continuum)  # alex is reference
    # We test the category shuffle independently for the special options.
    shuffled_cat = cst_alex.corpus_from_reference(["martino", "Martingale", "Martine"])
    cst_alex.category_shuffle(shuffled_cat, overlapping_fun=cat_ord, prevalence=True)
    # This reference doesn't have enough categories for the gamma to go lower.
    assert shuffled_cat.compute_gamma(dissim).gamma < 0.6

    # Now we generate a reference with A LOT of categories and close segments:
    continuum_martino = random_reference("Martino", 200, 40, 20, 3,
                                         40,  # 40 categories
                                         overlapping=False)
    cst_lots_of_cat = CorpusShufflingTool(0.9, continuum_martino)
    shuffled_cat = cst_lots_of_cat.corpus_from_reference(["martino", "Martingale", "Martine"])
    cst_lots_of_cat.category_shuffle(shuffled_cat, overlapping_fun=cat_ord, prevalence=True)
    dissim = CombinedCategoricalDissimilarity(continuum_martino.categories,
                                              delta_empty=1,
                                              alpha=3,
                                              beta=3,  # higher beta should make the gamma fall a lot since categories
                                              cat_dissimilarity_matrix=cat_ord)  # are now a mess
    assert shuffled_cat.compute_gamma(dissim).gamma < 0.6







