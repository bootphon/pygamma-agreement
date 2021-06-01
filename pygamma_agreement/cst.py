import random

import numpy.random
import logging

from .continuum import Continuum, Unit
from typing import Union, Iterable, List, Callable, Set, Tuple
from sortedcontainers import SortedDict, SortedSet
import numpy as np
from pyannote.core import Segment


class CorpusShufflingTool:
    """
    Corpus shuffling tool as detailed in section 6.3 of @gamma-paper
    (https://www.aclweb.org/anthology/J15-3003.pdf#page=30).
    Beware that the reference continuum is a copy of the given continuum.
    """
    SHIFT_FACTOR = 0.5  # must be < 1 or there might be problems of empty ranges
    SPLIT_FACTOR = 5
    FALSE_POS_FACTOR = 5

    def __init__(self,
                 magnitude: float,
                 reference_continuum: Continuum,
                 seed: int = 4772,
                 categories: Iterable[str] = None):
        """
        Parameters
        ----------
        magnitude:
            magnitude m of the cst (cf @gamma-paper)
        reference_continuum:
            this continuum will be copied, and will serve as reference for the tweaks made by the corpus shuffling tool.
        categories:
            this is used to consider additionnal categories when shuffling the corpus, in the eventuality that the
            reference continuum does not contain any unit of a possible category.
        """
        self._seed: int = seed
        self.magnitude: float = magnitude
        self._reference_continuum: Continuum = reference_continuum
        self._categories = self._reference_continuum.categories.union(categories)

    def corpus_shuffle(self,
                       new_annotators: Union[int, Iterable[str]],
                       reference_annotator: str = None,
                       include_reference=False):
        """
        Generates a shuffled corpus with the provided (or generated) reference annotation set,
        using the method described in 6.3 of @gamma-paper, https://www.aclweb.org/anthology/J15-3003.pdf#page=30
        """
        continuum = Continuum()
        reference_annotators = self._reference_continuum.annotators

        if reference_annotator is None:
            reference_annotator = next(iter(reference_annotators))
            if len(reference_annotators) > 1:
                logging.warning("CST was given a multi-annotator reference, but reference annotator was not\n"
                                "specified. A default annotator (1st in alphabetical) was used.")
        else:
            assert (reference_annotator in reference_annotators)

        units = self._reference_continuum[reference_annotator]
        if include_reference:
            continuum[reference_annotator] = units

        if isinstance(new_annotators, int):
            new_annotators = (f"annotator_cst_{i}" for i in range(new_annotators))

        shift_max = self.magnitude * self.SHIFT_FACTOR
        bounds_inf, bounds_sup = (next(iter(units)).segment.start, next(reversed(units)).segment.end)
        avg_dur_false_pos = np.average([unit.segment.end - unit.segment.start for unit in units])
        for new_annotator in new_annotators:
            assert new_annotator not in reference_annotators
            continuum.add_annotator(new_annotator)
            for unit in units:
                # false negatives
                if np.random.uniform() < self.magnitude:
                    continue
                # category TODO
                category = unit.annotation
                # positions
                len_unit = unit.segment.end - unit.segment.start
                continuum.add(new_annotator,
                              Segment(unit.segment.start + np.random.uniform(-shift_max*len_unit/2,
                                                                             shift_max*len_unit/2),
                                      unit.segment.end + np.random.uniform(-shift_max*len_unit/2,
                                                                           shift_max*len_unit/2)),
                              category)
            # false positives
            for _ in range(int(self.magnitude * self.FALSE_POS_FACTOR)):
                # a random unit is generated from a (all random) central point, duration, and category
                category = np.random.choice(self._categories)
                center = np.random.uniform(bounds_inf, bounds_sup)
                duration = np.random.exponential(avg_dur_false_pos)
                continuum.add(new_annotator,
                              Segment(center - duration/2, center + duration/2),
                              annotation=category)
            # splits
            new_units = continuum[new_annotator]
            if len(new_units) > 0:
                for _ in range(int(self.magnitude * self.SPLIT_FACTOR)):
                    to_split = new_units.pop(numpy.random.randint(0, len(new_units)))
                    cut = numpy.random.uniform(to_split.segment.start, to_split.segment.end)
                    new_units.add(Unit(Segment(cut, to_split.segment.end), to_split.annotation))
                    new_units.add(Unit(Segment(to_split.segment.start, cut), to_split.annotation))
                    del to_split
        return continuum


def random_reference(reference_annotator: str,
                     duration: float,
                     nb_unit: int,
                     avg_unit_duration: float,
                     categories: Union[int, Iterable[str]],
                     seed: int = 4772):
    # TODO: option pour désactiver l'overlap
    """
    Generates a random reference annotation set using some sort of poisson
    distributed points on the timeline. avg_gap is the gap between STARTS of
    segments to ensure free overlap is possible.
    """
    if isinstance(categories, int):
        categories = (f"cat_{i}" for i in range(categories))
    categories = SortedSet(categories)

    continuum = Continuum()
    np.random.seed(seed)
    last_t = 0.0
    for _ in range(nb_unit):
        # Exponential distribution value for next unit
        center = np.random.uniform(0, duration)
        unit_dur = np.random.exponential(avg_unit_duration)
        # random category (equiprobable)
        category = np.random.choice(categories)
        continuum.add(reference_annotator, Segment(center - unit_dur/2, center + unit_dur/2), annotation=category)
    return continuum








