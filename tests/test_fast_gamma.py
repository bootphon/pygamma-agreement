from pygamma_agreement import (Continuum,
                               CombinedCategoricalDissimilarity,
                               StatisticalContinuumSampler,
                               CorpusShufflingTool)
import numpy as np


def test_fast_gamma():
    sampler = StatisticalContinuumSampler()
    sampler.init_sampling_custom(annotators=['Ref'],
                                 avg_num_units_per_annotator=200, std_num_units_per_annotator=0,
                                 avg_duration=80, std_duration=20,
                                 avg_gap=80, std_gap=20,
                                 categories=np.array([str(i) for i in range(500)]))
    continuum = sampler.sample_from_continuum
    cst = CorpusShufflingTool(0.3, continuum)

    cont_cst = cst.corpus_shuffle(["Martino", "Martingale"],
                                  shift=True,
                                  false_pos=True,
                                  false_neg=True,
                                  cat_shuffle=True,
                                  split=True)

    dissim = CombinedCategoricalDissimilarity(cont_cst.categories, alpha=3, beta=1)
    cont_cst.compute_gamma(dissim, fast=True)


