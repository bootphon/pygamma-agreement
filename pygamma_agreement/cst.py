import math
import random

import numpy.random
import logging

from .continuum import Continuum, Unit
from typing import Union, Iterable, List, Callable, Set, Tuple
from sortedcontainers import SortedDict, SortedSet
from multiprocessing import pool
import numpy as np
from pyannote.core import Segment


class CorpusShufflingTool:
    """
    Corpus shuffling tool as detailed in section 6.3 of @gamma-paper
    (https://www.aclweb.org/anthology/J15-3003.pdf#page=30).
    Beware that the reference continuum is a copy of the given continuum.
    """
    SHIFT_FACTOR = 2
    SPLIT_FACTOR = 1
    FALSE_POS_FACTOR = 1

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
        assert len(reference_continuum.annotators) == 1
        self._reference_continuum: Continuum = reference_continuum
        self._categories = self._reference_continuum.categories.union(categories)

    def corpus_from_reference(self, new_annotators: Union[int, Iterable[str]]):
        continuum = Continuum()
        annotator_set = iter(self._reference_continuum.annotators)
        if isinstance(new_annotators, int):
            new_annotators = SortedSet(f"annotator_{i}" for i in range(new_annotators))
        else:
            new_annotators = SortedSet(new_annotators)
        for unit in self._reference_continuum[next(annotator_set)]:
            for new_annotator in new_annotators:
                continuum.add(new_annotator,
                              Segment(unit.segment.start, unit.segment.end),
                              unit.annotation)
        if next(annotator_set, None) is not None:
            logging.warning("Warning : a reference continuum with multiple annotators was given to the CST, so"
                            "its first annotator in alphabetical order was picked.")
        return continuum

    def shift_shuffle(self, continuum: Continuum):
        """
        Tweaks the continuum by shifting the ends of each segment, with uniformly distributed values
        of bounds proportionnal to the magnitude of the CST and the length of the segment.
        """
        shift_max = self.magnitude * self.SHIFT_FACTOR * \
            self._reference_continuum.avg_length_unit
        for annotator in continuum.annotators:
            for unit in continuum[annotator]:
                continuum[annotator].remove(unit)
                len_unit = unit.segment.end - unit.segment.start
                start_seg, end_seg = 0.0, 0.0
                while start_seg >= end_seg:
                    start_seg = unit.segment.start + np.random.uniform(-shift_max, shift_max)
                    end_seg = unit.segment.end + np.random.uniform(-shift_max, shift_max)
                continuum.add(annotator, Segment(start_seg, end_seg), unit.annotation)

    def false_neg_shuffle(self, continuum: Continuum):
        """
        Tweaks the continuum by randomly removing units ("false negatives").
        Every unit (for each annotator) have a probability equal to the magnitude of being removed.
        If this probability is one, a single random unit (for each annotator) will be left alone.
        """
        for annotator in continuum.annotators:
            security = np.random.choice(continuum[annotator])
            # security : if an annotator doesnt have any annotations gamma cant be computed.
            for unit in continuum[annotator]:
                if np.random.uniform() < self.magnitude:
                    continuum[annotator].remove(unit)
            if len(continuum[annotator]) == 0:
                continuum[annotator].add(security)

    def false_pos_shuffle(self, continuum: Continuum):
        """
        Tweaks the continuum by randomly adding "false positive" units.
        The number of added units per annotator is constant & proportionnal to the magnitude of the CST.
        The chosen category is random and depends on the probability of occurence of the category in the reference.
        The length of the segment is random (normal distribution) based on the average and standard deviation
        of those of the reference.
        """
        ref_units = self._reference_continuum[next(iter(self._reference_continuum.annotators))]
        avg_dur = np.average(unit.segment.end - unit.segment.start for unit in ref_units)
        var_dur = np.std(unit.segment.end - unit.segment.start for unit in ref_units)
        category_weights = self._reference_continuum.category_weights
        bounds_inf, bounds_sup = (next(iter(ref_units)).segment.start, next(reversed(ref_units)).segment.end)
        for annotator in continuum.annotators:
            for _ in range(int(self.magnitude * self.FALSE_POS_FACTOR * len(self._reference_continuum))):
                # a random unit is generated from a (all random) central point, duration, and category
                category = np.random.choice(category_weights.keys(), p=category_weights.values())
                center = np.random.uniform(bounds_inf, bounds_sup)
                duration = np.random.normal(avg_dur, var_dur)
                continuum.add(annotator,
                              Segment(center - duration / 2, center + duration / 2),
                              annotation=category)
    def category_shuffle(self, continuum: Continuum):
        #todo
        pass

    def splits_shuffle(self, continuum: Continuum):
        """
        Tweak the continuum by randomly splitting segments.
        Number of splits per annotator is constant & proportionnal to the magnitude of the CST
        and the number of units in the reference.
        A splitted segment can be re-splitted.
        """
        for annotator in continuum.annotators:
            units = continuum[annotator]
            for _ in range(int(self.magnitude * self.SPLIT_FACTOR * len(units))):
                to_split = units.pop(numpy.random.randint(0, len(units)))
                cut = numpy.random.uniform(to_split.segment.start, to_split.segment.end)
                units.add(Unit(Segment(cut, to_split.segment.end), to_split.annotation))
                units.add(Unit(Segment(to_split.segment.start, cut), to_split.annotation))
                del to_split



    def corpus_shuffle(self,
                       annotators: Union[int, Iterable[str]],
                       shift=False,
                       false_positive=False,
                       false_negative=False,
                       category=False,
                       splits=False,
                       include_reference=False) -> Continuum:
        """
        Generates a shuffled corpus with the provided (or generated) reference annotation set,
        using the method described in 6.3 of @gamma-paper, https://www.aclweb.org/anthology/J15-3003.pdf#page=30,
        and missing elements described in another article : https://hal.archives-ouvertes.fr/hal-00769639/
        """
        continuum = self.corpus_from_reference(annotators)
        if shift:
            self.shift_shuffle(continuum)
        if false_positive:
            self.false_pos_shuffle(continuum)
        if false_negative:
            self.false_neg_shuffle(continuum)
        if category:
            self.category_shuffle(continuum)
        if splits:
            self.splits_shuffle(continuum)
        if include_reference:
            ref_annotator = next(iter(self._reference_continuum.annotators))
            assert ref_annotator not in continuum.annotators, "Reference annotator can't be included as " \
                                                              "an annotator with the same name is in the " \
                                                              "generated corpus."
            for unit in self._reference_continuum[next(iter(self._reference_continuum.annotators))]:
                continuum.add(ref_annotator, unit.segment, unit.annotation)
        return continuum

def random_reference(reference_annotator: str,
                     duration: float,
                     nb_unit: int,
                     avg_unit_duration: float,
                     std_unit_duration: float,
                     categories: Union[int, Iterable[str]],
                     seed: int = 4772,
                     overlapping: bool = True):
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
    points = np.random.uniform(0, duration, nb_unit)
    points.sort()
    last_end = -math.inf
    for point in points:
        # Exponential distribution value for next unit
        unit_dur = abs(np.random.normal(avg_unit_duration, std_unit_duration))
        # random category (equiprobable) todo: parameter->prob. of each category
        category = np.random.choice(categories)
        if overlapping:
            continuum.add(reference_annotator,
                          Segment(point - unit_dur/2, point + unit_dur/2),
                          annotation=category)
        else:
            continuum.add(reference_annotator,
                          Segment(max(point - unit_dur, last_end), point),
                          annotation=category)
            last_end = point
    return continuum








