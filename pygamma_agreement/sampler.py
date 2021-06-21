#!usr/bin/env python
from abc import ABCMeta, abstractmethod
from typing import Optional, Iterable, List
from typing_extensions import Literal
from sortedcontainers import SortedSet, SortedList
import numpy as np
from pyannote.core import Segment
from .continuum import Continuum, Unit

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


class ShuffleContinuumSampler(AbstractContinuumSampler):
    """
    This continuum sampler uses the methods used in gamma-software, ie those described in
    gamma-paper : https://www.aclweb.org/anthology/J15-3003.pdf, section 5.2.
    and implemented in the GammaSoftware.
    We found some unexplained specificities, such as a minimum distance between pivots, and chose
    to add them to our implementation so our results correspond to their program's. They can be disabled
    at construction.
    """
    _pivot_type: PivotType
    _min_dist_between_pivots: bool

    def __init__(self, reference_continuum: Continuum,
                 ground_truth_annotators: Optional[SortedSet] = None,
                 pivot_type: PivotType = 'int_pivot'):
        """
        Parameters
        ----------
        reference_continuum: Continuum
            the continuum that will be shuffled into the samples
        ground_truth_annotators: SortedSet of str, optional
            the set of annotators (from the reference) that will be considered for sampling
        pivot_type: 'int_pivot' or 'float_pivot'
            the java implementation by Mathet et Al. uses a integer pivoting for shuffling, we judged it unclear that
            the method descibed in the paper was typed this way so we left the option to generate pivots using floats.
        """
        super().__init__(reference_continuum, ground_truth_annotators)
        self._pivot_type = pivot_type

    @staticmethod
    def __remove_pivot_segment__(pivot: float, segments: List[Segment], dist: float) -> List[Segment]:
        new_segments = []
        while len(segments) > 0:
            segment = segments.pop()
            if segment.start > pivot - dist:
                if segment.end < pivot + dist:
                    continue
                else:
                    new_segments.append(Segment(pivot + dist, segment.end))
            else:
                if segment.end > pivot + dist:
                    new_segments.append(Segment(segment.start, pivot - dist))
                    new_segments.append(Segment(pivot + dist, segment.end))
                else:
                    new_segments.append(Segment(segment.start, pivot - dist))
        return new_segments

    def __random_from_segments__(self, segments: List[Segment]) -> float:
        segments = np.array(segments)
        weights = np.array(list(segment.end - segment.start for segment in segments))
        weights /= np.sum(weights)
        segment = np.random.choice(np.array(segments), p=weights)
        if self._pivot_type == 'int_pivot':
            return int(np.random.uniform(segment.start, segment.end))
        else:
            return np.random.uniform(segment.start, segment.end)


    @property
    def sample_from_continuum(self) -> Continuum:
        assert self._pivot_type in ('float_pivot', 'int_pivot')
        continuum = self._reference_continuum
        min_dist_between_pivots = continuum.avg_length_unit / 2
        bound_inf, bound_sup = continuum.bounds
        new_continuum = Continuum()
        annotators = self._ground_truth_annotators
        segments_available = [Segment(bound_inf, bound_sup)]
        for idx in range(len(annotators)):
            if len(segments_available) != 0:
                pivot: float = self.__random_from_segments__(segments_available)
                segments_available = self.__remove_pivot_segment__(pivot, segments_available, min_dist_between_pivots)
            else:
                pivot = np.random.uniform(bound_inf, bound_sup)
            rnd_annotator = np.random.choice(annotators)
            units = continuum[rnd_annotator]
            new_annotator = f'Sampled_annotation {idx}'
            for unit in units:
                if unit.segment.start + pivot > bound_sup:
                    new_continuum.add(new_annotator,
                                      Segment(unit.segment.start + pivot + bound_inf - bound_sup,
                                              unit.segment.end + pivot + bound_inf - bound_sup),
                                      unit.annotation)
                else:
                    new_continuum.add(new_annotator,
                                      Segment(unit.segment.start + pivot,
                                              unit.segment.end + pivot),
                                      unit.annotation)
        return new_continuum


class StatisticalContinuumSampler(AbstractContinuumSampler):
    """
    This sampler creates continua using the average and standard deviation of :

    * The number of annotations per annotator
    * The gap between two of an annotator's annotations
    * The duration of the annotations' segments
    The sample is thus created by computing normal distributions using these parameters.

    It also requires the probability of occurence of each annotations category.
    """

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
        for annotation_set in self._reference_continuum._annotations.values():
            if annotation_set[0].segment.start > 0:
                gaps.append(annotation_set[0].segment.start)
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

    def __init__(self, continuum: Continuum = None,
                 annotators: Iterable[str] = None,
                 avg_num_units_per_annotator: float = None, std_num_units_per_annotator: float = None,
                 avg_gap: float = None, std_gap: float = None,
                 avg_duration: float = None, std_duration: float = None,
                 categories: np.ndarray = None, categories_weight: np.ndarray = None
                 ):
        """

        Parameters
        ----------
        continuum: Continuum, optional
            The continuum that will be analysed to obtain all statistical values needed for sample generation.
            If not set, it is required that all the parameters are provided. If set, the other parameters will be
            ignored.
        annotators:
            the annotators that will be involved in the samples
        avg_num_units_per_annotator: float, optional
            average number of units per annotator
        std_num_units_per_annotator: float, optional
            standard deviation of the number of units per annotator
        avg_gap: float, optional
            average gap between two of an annotator's annotations
        std_gap: float, optional
            standard deviation of the gap between two of an annotator's annotations
        avg_duration: float, optional
            average duration of an annotation
        std_duration: float, optional
            standard deviation of the duration of an annotation
        categories: np.array[str, 1d]
            The possible categories of the annotations
        categories_weight: np.array[float, 1d]
            The probability of occurence of each category. Can raise errors if len(categories) != len(category_weights)
            and category_weights.sum() != 1.0.
        """
        if continuum is None:
            reference_dummy = Continuum()
            for annotator in annotators:
                reference_dummy.add_annotator(annotator)
            super().__init__(reference_dummy)
            self._avg_nb_units_per_annotator = avg_num_units_per_annotator
            self._std_nb_units_per_annotator = std_num_units_per_annotator
            self._avg_gap = avg_gap
            self._std_gap = std_gap
            self._categories = categories
            self._categories_weight = categories_weight
            self._avg_unit_duration = avg_duration
            self._std_unit_duration = std_duration
        else:
            super().__init__(continuum)
            self._set_gap_information()
            self._set_duration_information()
            self._set_categories_information()
            self._set_nb_units_information()

    @property
    def sample_from_continuum(self) -> Continuum:
        new_continnum = Continuum()
        for annotator in self._ground_truth_annotators:
            last_point = 0
            nb_units = max(1, int(np.random.normal(self._avg_nb_units_per_annotator, self._std_nb_units_per_annotator)))
            for _ in range(nb_units):
                gap = np.random.normal(self._avg_gap, self._std_gap)
                length = abs(np.random.normal(self._avg_unit_duration, self._std_unit_duration))
                category = np.random.choice(self._categories, p=self._categories_weight)
                start = last_point + gap
                end = start + length
                new_continnum.add(annotator, Segment(start, end), category)
                last_point = end
        return new_continnum



