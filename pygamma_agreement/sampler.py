#!usr/bin/env python
from abc import ABCMeta, abstractmethod
from typing import Optional, Literal
from sortedcontainers import SortedSet
import numpy as np
from pyannote.core import Segment
from .continuum import Continuum, Unit, Annotator

PivotType = Literal["float_pivot", "int_pivot"]


class AbstractContinuumSampler(metaclass=ABCMeta):
    """
    Tool for generating sampled continuua from a reference continuum.
    Used to compute the "expected disorder" when calculating the gamma,
    using particular sampling techniques.
    """
    _reference_continuum: Continuum
    _ground_truth_annotators: SortedSet

    def __init__(self, reference_continuum: Continuum,
                 ground_truth_annotators: Optional[SortedSet] = None):
        self._reference_continuum = reference_continuum
        if ground_truth_annotators is None:
            self._ground_truth_annotators = self._reference_continuum.annotators
        else:
            assert self._reference_continuum.annotators.issuperset(ground_truth_annotators),\
                   "Can't sample from ground truth annotators not in the reference continuum."
            self._ground_truth_annotators = ground_truth_annotators

    @property
    @abstractmethod
    def sample_from_continuum(self) -> Continuum:
        """
        Returns a shuffled continuum based on the reference.
        Every data in the generated sample must be a new object.
        """
        pass


class MathetContinuumSampler(AbstractContinuumSampler):
    """
    This continuum sampler uses the methods used in gamma-software, ie those described in
    gamma-paper : https://www.aclweb.org/anthology/J15-3003.pdf, section 5.2.
    and implemented in the GammaSoftware.
    We found some unexplained specificities, such as a minimum distance between pivots, and chose
    to add them to our implementation so our results correspond to their program.
    """
    _pivot_type: PivotType
    _min_dist_between_pivots: bool

    def __init__(self, reference_continuum: Continuum,
                 ground_truth_annotators: Optional[SortedSet] = None,
                 pivot_type: PivotType = 'int_pivot',
                 min_dist_between_pivots: bool = True):
        super().__init__(reference_continuum, ground_truth_annotators)
        self._pivot_type = pivot_type
        self._min_dist_between_pivots = min_dist_between_pivots

    @property
    def sample_from_continuum(self) -> Continuum:
        assert self._pivot_type in ('float_pivot', 'int_pivot')
        continuum = self._reference_continuum
        min_dist_between_pivots = continuum.avg_length_unit / 2
        bound_inf, bound_sup = continuum.bounds
        new_continuum = Continuum()
        annotators = self._ground_truth_annotators
        pivots = []
        for idx in range(len(annotators)):
            if self._pivot_type == 'float_pivot':
                pivot: float = np.random.uniform(bound_inf, bound_sup)
                if self._min_dist_between_pivots:
                    # While the pivot is closer than min_dist to a precedent pivot, pick another one
                    # (takes wrapping of continuum into consideration).
                    while any(map((lambda x: abs(x - pivot) < min_dist_between_pivots or
                                   abs(x - (pivot - bound_sup)) < min_dist_between_pivots),
                                  pivots)):
                        pivot = np.random.uniform(bound_inf, bound_sup)
            else:
                pivot: int = np.random.randint(np.floor(bound_inf), np.ceil(bound_sup))
                if self._min_dist_between_pivots:
                    while any(map((lambda x: abs(x - pivot) < min_dist_between_pivots or
                                   abs(x - (pivot - bound_sup)) < min_dist_between_pivots),
                                  pivots)):
                        pivot = np.random.randint(np.floor(bound_inf), np.ceil(bound_sup))
            pivots.append(pivot)

            rnd_annotator = np.random.choice(annotators)
            units = continuum[rnd_annotator]
            sampled_annotation = SortedSet()
            for unit in units:
                if unit.segment.start + pivot > bound_sup:
                    new_segment = Segment(unit.segment.start + pivot + bound_inf - bound_sup,
                                          unit.segment.end + pivot + bound_inf - bound_sup)
                else:
                    new_segment = Segment(unit.segment.start + pivot,
                                          unit.segment.end + pivot)
                sampled_annotation.add(Unit(new_segment, unit.annotation))
            new_continuum[f'Sampled_annotation {idx}'] = sampled_annotation
        return new_continuum


class StatisticalContinuumSampler(AbstractContinuumSampler):

    _avg_nb_units_per_annotator: float
    _std_nb_units_per_annotator: float
    _avg_gap: float
    _std_gap: float
    _avg_unit_duration: float
    _std_unit_duration: float
    _categories: np.ndarray
    _categories_weight: np.ndarray

    def _set_gap_information(self):
        gaps = []
        current_annotator = None
        last_unit = None
        for annotator, unit in self._reference_continuum:
            if annotator != current_annotator:
                current_annotator = annotator
            else:
                gaps.append(unit.segment.start - last_unit.segment.end)
            last_unit = unit
        self._avg_gap = float(np.mean(gaps))
        self._std_gap = float(np.std(gaps))

    def _set_nb_units_information(self):
        nb_units = [len(annotations) for annotator, annotations in self._reference_continuum._annotations.items()]
        self._avg_nb_units_per_annotator = float(np.mean(nb_units))
        self._std_nb_units_per_annotator = float(np.std(nb_units))

    def _set_duration_information(self):
        durations = [unit.segment.duration for _, unit in self._reference_continuum]
        self._avg_unit_duration = float(np.mean(durations))
        self._std_unit_duration = float(np.std(durations))

    def _set_categories_information(self):
        categories_set = self._reference_continuum.categories
        self._categories = np.array(categories_set)
        self._categories_weight = np.zeros(len(categories_set))
        for _, unit in self._reference_continuum:
            self._categories_weight[categories_set.index(unit.annotation)] += 1
        self._categories_weight /= self._reference_continuum.num_units

    def __init__(self, continuum: Continuum):
        super().__init__(continuum)
        self._set_gap_information()
        self._set_duration_information()
        self._set_categories_information()
        self._set_nb_units_information()

    @property
    def sample_from_continuum(self) -> Continuum:
        new_continnum = Continuum()
        for annotator in self._reference_continuum.annotators:
            last_point = 0
            nb_units = int(np.random.normal(self._avg_nb_units_per_annotator, self._std_nb_units_per_annotator))
            for _ in range(nb_units):
                gap = np.random.normal(self._avg_gap, self._std_gap)
                length = abs(np.random.normal(self._avg_unit_duration, self._std_unit_duration))
                category = np.random.choice(self._categories, p=self._categories_weight)
                start = last_point + gap
                end = start + length
                new_continnum.add(annotator, Segment(start, end), category)
                last_point = end
        return new_continnum



