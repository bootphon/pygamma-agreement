import random

import numpy.random

from .continuum import Continuum
from .cat_dissim import cat_default
from typing import Union, Iterable, List, Callable
import numpy as np
from pyannote.core import Segment

class ContinuumGenerator:
    SHUFFLE = 0
    TOTAL = 0
    """
    Class for generating random continuua, using exponential distribution.
    
    """
    def __init__(self,
                 annotators: Union[int, Iterable[str]],
                 duration: float,
                 avg_gap: float,
                 avg_duration: float,
                 categories: Union[int, Iterable[str]],
                 seed: int = 4772,):
        if isinstance(annotators, int):
            self.seed: int = seed

            self.annotators: List[str]
            if isinstance(annotators, int):
                self.annotators = [f"annotator_{annotator}" for annotator in range(annotators)]
            else:
                self.annotators = sorted(set(annotators))

            self.nb_annotators: int = len(self.annotators)
            self.avg_duration: float = avg_duration
            self.avg_gap: float = avg_gap
            self.duration: float = duration

            self.categories: Iterable[str]
            if isinstance(categories, int):
                self.categories = [str(i) for i in range(categories)]
            else:
                self.categories = sorted(set(categories))

    def random_continuum_total(self):
        """
        Generates a totally random continuum, which means this algorithm does not try
        to make any similarities between annotators.
        """
        np.random.seed(self.seed)
        continuum = Continuum()
        for annotator in self.annotators:
            last_t = 0.0
            while last_t < self.duration:
                # Exponential distribution value for next unit
                last_t += np.random.exponential(self.avg_gap)
                # random category (equiprobable)
                category = np.random.choice(self.categories)
                continuum.add(annotator,
                              # Exponential distribution value for duration of unit
                              Segment(last_t, last_t + np.random.exponential(self.avg_duration)),
                              category)
        return continuum

    def random_continuum_tweak(self,
                               variation_pos: float,
                               variation_cat: float,
                               cat_dissim_matrix: Callable[[str, str], float] = cat_default):
        """
        This algorithm generates a random continuum by inserting small dissimilarities between annotations, by tweaking
        the start/end of generated normal distribution, and categories using geometrical distribution.

        Parameters:
        - variation_pos:
            standard deviation of the times delimiting segments of annotators. The lower it is, the higher the chance
            will be that annotators agree on the delimitations of annotations.
        - variation_cat:
            success probability of annotating with the next closest category. The higher it is, the higher
            the chance will be that annotators will agree on categories.
        """
        # Retain a list of categories sorted by dissimilarity with the key category.
        cat_dissim_table = {category: sorted(self.categories,
                                             key=(lambda x: cat_dissim_matrix(category, x)))
                            for category in self.categories}

        np.random.seed(self.seed)
        continuum = Continuum()
        last_t = 0.0
        while last_t < self.duration:
            # Exponential distribution value for next unit
            last_t += np.random.exponential(self.avg_gap)
            # random category (equiprobable)
            category = np.random.choice(self.categories)
            for annotator in self.annotators:
                variation = numpy.random.normal(variation_pos)
                category = np.random.choice(self.categories)
                continuum.add(annotator,
                              # Exponential distribution value for duration of unit
                              Segment(last_t + variation,
                                      last_t + variation + np.random.exponential(self.avg_duration)),
                              # Geometrical distribution for difference between
                              cat_dissim_table[category][min(np.random.geometric(variation_cat) - 1,
                                                         len(self.categories))])
        return continuum













