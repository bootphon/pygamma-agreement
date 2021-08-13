# The MIT License (MIT)

# Copyright (c) 2020-2021 CoML

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Rachid RIAD, Hadrien TITEUX, Léopold FAVRE

from abc import ABCMeta, abstractmethod
from typing import Optional, Iterable, List

import numpy as np
from pyannote.core import Segment
from sortedcontainers import SortedSet
from typing_extensions import Literal

from .continuum import Continuum, Annotator

PivotType = Literal["float_pivot", "int_pivot"]


class AbstractContinuumSampler(metaclass=ABCMeta):
    """
    Tool for generating sampled continuua from a reference continuum.
    Used to compute the "expected disorder" when calculating the gamma,
    using particular sampling techniques.
    Must be initalized (with self.init_sampling for instance)
    """
    _reference_continuum: Optional[Continuum]
    _ground_truth_annotators: Optional[SortedSet]

    def __init__(self):
        """Super constructor, sets everything to None since a call to init_sampling to set
         parameters is mandatory."""
        self._reference_continuum = None
        self._ground_truth_annotators = None

    def init_sampling(self, reference_continuum: Continuum,
                      ground_truth_annotators: Optional[Iterable['Annotator']] = None):
        """
        Parameters
        ----------
        reference_continuum: Continuum
            the continuum that will be shuffled into the samples
        ground_truth_annotators: iterable of str, optional
            the set of annotators (from the reference) that will be considered for sampling
        """
        assert reference_continuum, "Cannot initialize sampling with an empty reference continuum."
        self._reference_continuum = reference_continuum
        if ground_truth_annotators is None:
            self._ground_truth_annotators = self._reference_continuum.annotators
        else:
            assert self._reference_continuum.annotators.issuperset(ground_truth_annotators), \
                "Can't sample from ground truth annotators not in the reference continuum."
            self._ground_truth_annotators = SortedSet(ground_truth_annotators)

    def _has_been_init(self):
        assert self._reference_continuum is not None, \
            "Sampler hasnt been initialized. Call 'sampler.init_sampling' before 'sampler.sample_from_continuum'."


    @property
    @abstractmethod
    def sample_from_continuum(self) -> Continuum:
        """
        Returns a shuffled continuum based on the reference.
        Everything in the generated sample is at least a copy.

        Raises
        ------
        ValueError:
            if `init_sampling` or another initalization method hasn't been called before.
        """
        pass


class ShuffleContinuumSampler(AbstractContinuumSampler):
    """
    This continuum sampler uses the methods used in gamma-software, ie those described in
    gamma-paper : https://www.aclweb.org/anthology/J15-3003.pdf, section 5.2.
    and implemented in the GammaSoftware.
    """
    _pivot_type: PivotType

    def __init__(self, pivot_type: PivotType = 'int_pivot'):
        """This constructor allows to set the pivot type to int or float. Defaults to
        int to match the java implementation."""
        super().__init__()
        self._pivot_type = pivot_type

    def init_sampling(self, reference_continuum: Continuum,
                      ground_truth_annotators: Optional[Iterable['Annotator']] = None):
        """
        Parameters
        ----------
        reference_continuum: Continuum
            the continuum that will be shuffled into the samples
        ground_truth_annotators: iterable of str, optional
            the set of annotators (from the reference) that will be considered for sampling
        """
        super().init_sampling(reference_continuum, ground_truth_annotators)

    @staticmethod
    def _remove_pivot_segment(pivot: float, segments: List[Segment], dist: float) -> List[Segment]:
        """
        Returns a copy of the given list of segments, minus the segment delimited by [pivot - dist, pivot + dist].
        """
        new_segments = []
        while len(segments) > 0:
            segment = segments.pop()
            if segment.start >= pivot - dist:
                if segment.end <= pivot + dist:
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

    def _random_from_segments(self, segments: List[Segment]) -> float:
        """
        Returns a random value from the provided list of segments, by randomly choosing
        a segment (weighted by its length) and then using uniform distribution in it.
        """
        segments = np.array(segments)
        weights = np.array(list(segment.end - segment.start for segment in segments))
        weights /= np.sum(weights)
        try:
            segment = np.random.choice(np.array(segments), p=weights)
        except ValueError:
            return 1
        if self._pivot_type == 'int_pivot':
            return int(np.random.uniform(segment.start, segment.end))
        else:
            return np.random.uniform(segment.start, segment.end)

    @property
    def sample_from_continuum(self) -> Continuum:
        self._has_been_init()
        assert self._pivot_type in ('float_pivot', 'int_pivot')
        continuum = self._reference_continuum
        min_dist_between_pivots = continuum.avg_length_unit / 2
        bound_inf, bound_sup = continuum.bounds
        new_continuum = continuum.copy_flush()
        annotators = self._ground_truth_annotators
        while not new_continuum:  # Simple check to prevent returning an empty continuum.
            segments_available = [Segment(bound_inf, bound_sup)]
            for idx in range(len(annotators)):
                if len(segments_available) != 0:
                    pivot: float = self._random_from_segments(segments_available)
                    segments_available = self._remove_pivot_segment(pivot, segments_available, min_dist_between_pivots)
                else:
                    pivot = np.random.uniform(bound_inf, bound_sup)
                rnd_annotator = np.random.choice(annotators)
                new_annotator = f'Sampled_annotation {idx}'
                new_continuum.add_annotator(new_annotator)
                for unit in continuum.iter_annotator(rnd_annotator):
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

    - The number of annotations per annotator
    - The gap between two of an annotator's annotations
    - The duration of the annotations' segments
    The sample is thus created by computing normal distributions using these parameters.

    It also requires the probability of occurence of each annotations category. You can either initalize sampling with
    custom values or with a reference continuum.
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
        # To prevent glitching continua with 1 unit
        gaps = [0]
        current_annotator = None
        last_unit = None
        for annotator, unit in self._reference_continuum:
            if annotator != current_annotator:
                current_annotator = annotator
            else:
                gaps.append(unit.segment.start - last_unit.segment.end)
            last_unit = unit
        for annotation_set in self._reference_continuum._annotations.values():
            if len(annotation_set) == 0:
                continue
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

    def init_sampling_custom(self, annotators: Iterable[str],
                             avg_num_units_per_annotator: float, std_num_units_per_annotator: float,
                             avg_gap: float, std_gap: float,
                             avg_duration: float, std_duration: float,
                             categories: Iterable[str], categories_weight: Iterable[float] = None):
        """

        Parameters
        ----------
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
        categories_weight: np.array[float, 1d], optional
            The probability of occurence of each category. Can raise errors if len(categories) != len(category_weights)
            and category_weights.sum() != 1.0. If not set, every category is equiprobable.
        """
        reference_dummy = Continuum()
        for annotator in annotators:
            reference_dummy.add(annotator, Segment(0, 10), "Dummy")
        super().init_sampling(reference_dummy)
        self._avg_nb_units_per_annotator = avg_num_units_per_annotator
        self._std_nb_units_per_annotator = std_num_units_per_annotator
        self._avg_gap = avg_gap
        self._std_gap = std_gap
        self._categories = np.array(categories)
        self._categories_weight = None
        if categories_weight is not None:
            self._categories_weight = np.array(categories_weight)
            if len(self._categories) != len(self._categories_weight):
                raise ValueError("categories and categories_weight have different sizes.")
        self._avg_unit_duration = avg_duration
        self._std_unit_duration = std_duration

    def init_sampling(self, reference_continuum: Continuum,
                      ground_truth_annotators: Optional[Iterable['Annotator']] = None):
        """
        Sets the sampling parameters using statistical values obtained from the reference continuum.

        Parameters
        ----------
        reference_continuum: Continuum
            the continuum that will be shuffled into the samples
        ground_truth_annotators: iterable of str, optional
            the set of annotators (from the reference) that will be considered for sampling
        """
        super().init_sampling(reference_continuum, ground_truth_annotators)
        self._set_gap_information()
        self._set_duration_information()
        self._set_categories_information()
        self._set_nb_units_information()

    @property
    def sample_from_continuum(self) -> Continuum:
        self._has_been_init()
        new_continnum = self._reference_continuum.copy_flush()
        for annotator in self._ground_truth_annotators:
            new_continnum.add_annotator(annotator)
            last_point = 0
            nb_units = abs(int(np.random.normal(self._avg_nb_units_per_annotator, self._std_nb_units_per_annotator)))
            if not new_continnum:
                nb_units = max(1, nb_units)
            for _ in range(nb_units):
                gap = np.random.normal(self._avg_gap, self._std_gap)
                length = abs(np.random.normal(self._avg_unit_duration, self._std_unit_duration))
                while length == 0:
                    length = abs(np.random.normal(self._avg_unit_duration, self._std_unit_duration))
                category = np.random.choice(self._categories, p=self._categories_weight)
                start = last_point + gap
                end = start + length
                new_continnum.add(annotator, Segment(start, end), category)
                last_point = end
        return new_continnum
