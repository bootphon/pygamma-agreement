import pygamma_agreement as pa
from pyannote.core import Segment


def test_errors_continuum():
    continuum = pa.Continuum()
    dissim = pa.CombinedCategoricalDissimilarity(continuum.categories, alpha=3, beta=2, delta_empty=1.0,
                                                 cat_dissimilarity_matrix=pa.dissimilarity.cat_levenshtein)
    # 0 annotators
    try:
        best_alignment = continuum.get_best_alignment(dissim)
    except AssertionError:
        best_alignment = None
    assert best_alignment is None

    continuum.add('Martin', Segment(0, 10), '15')
    # 2 annotators, 1 annotation
    try:
        best_alignment = continuum.get_best_alignment(dissim)
    except AssertionError:
        best_alignment = None
    assert best_alignment is None

    continuum.add_annotator('Martino')
    # dissim without categories
    try:
        best_alignment = continuum.get_best_alignment(dissim)
    except ValueError:
        best_alignment = None
    assert best_alignment is None

    dissim = pa.CombinedCategoricalDissimilarity(continuum.categories, alpha=3, beta=2, delta_empty=1.0,
                                                 cat_dissimilarity_matrix=pa.dissimilarity.cat_levenshtein)

    best_alignment = continuum.get_best_alignment(dissim)
    only_unit_align = best_alignment.unitary_alignments[0]
    assert only_unit_align.n_tuple == [('Martin', pa.Unit(Segment(0, 10), '15')), ('Martino', None)]

    gamma_results = continuum.compute_gamma(dissim)
    gamma, gamma_cat, gamma_k = gamma_results.gamma, gamma_results.gamma_cat, gamma_results.gamma_k('15')
    assert gamma <= 0 and gamma_cat == 1 and gamma_k == 1

    gamma_results = continuum.compute_gamma(dissim, sampler=pa.ShuffleContinuumSampler())
    gamma, gamma_cat, gamma_k = gamma_results.gamma, gamma_results.gamma_cat, gamma_results.gamma_k('15')
    assert gamma <= 0 and gamma_cat == 1 and gamma_k == 1

    unit_martin = next(continuum.iter_annotator('Martin'))
    assert unit_martin is not None
    try:
        continuum.remove('Martino', unit_martin)
        assert False
    except KeyError:
        pass

    continuum.remove('Martin', unit_martin)

    # Gamma - no annotations
    dissim = pa.CombinedCategoricalDissimilarity(continuum.categories, alpha=3, beta=2, delta_empty=1.0,
                                                 cat_dissimilarity_matrix=pa.dissimilarity.cat_levenshtein)
    try:
        gamma_results = continuum.compute_gamma(dissim)
        exit(1)
        gamma, gamma_cat = gamma_results.gamma, gamma_results.gamma_cat
    except AssertionError:
        pass




