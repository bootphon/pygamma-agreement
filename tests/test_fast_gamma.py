from pygamma_agreement import (Continuum,
                               CombinedCategoricalDissimilarity,
                               StatisticalContinuumSampler,
                               CorpusShufflingTool)
import numpy as np


def test_fast_gamma():
    np.random.seed(4556)

    sampler = StatisticalContinuumSampler()
    sampler.init_sampling_custom(annotators=['Ref'],
                                 avg_num_units_per_annotator=0, std_num_units_per_annotator=0,
                                 avg_duration=80, std_duration=20,
                                 avg_gap=80, std_gap=20,
                                 categories=np.array([str(i) for i in range(4)]))
    sample = sampler.sample_from_continuum
    cst = CorpusShufflingTool(0.3, sample)
    dissim = CombinedCategoricalDissimilarity(alpha=3, beta=1)

    for nb_annotator in [2, 3, 4]:
        max_nb_annot = (2000**2)**(1/nb_annotator)

        for nb_annotation in [max_nb_annot // 2, max_nb_annot]:

            sampler._avg_nb_units_per_annotator = nb_annotation
            sample = sampler.sample_from_continuum
            cst = CorpusShufflingTool(0.3, sample)
            dissim = CombinedCategoricalDissimilarity(alpha=3, beta=1)

            cont_cst = cst.corpus_shuffle(["Martino", "Martingale"],
                                          shift=True,
                                          false_neg=True,
                                          cat_shuffle=True)

            seed = np.random.randint(0, 10000)

            np.random.seed(seed)
            gamma = cont_cst.compute_gamma(dissim, fast=False).gamma

            np.random.seed(seed)
            gamma_fast = cont_cst.compute_gamma(dissim, fast=True).gamma

            assert abs(gamma - gamma_fast) < 0.0001


