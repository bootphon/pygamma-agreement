#!usr/bin/env python
import abc
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Optional, List, Literal
from sortedcontainers import SortedSet
import random
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
    _ground_truth_annotators: SortedSet[Annotator]

    def __init__(self, reference_continuum: Continuum,
                 ground_truth_annotators: Optional[SortedSet[Annotator]] = None):
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
    def __init__(self, reference_continuum: Continuum,
                 ground_truth_annotators: Optional[SortedSet    [Annotator]] = None,
                 pivot_type: PivotType = 'int_pivot'):
        super().__init__(reference_continuum, ground_truth_annotators)
        self._pivot_type = pivot_type

    @property
    def sample_from_continuum(self) -> Continuum:
        assert self._pivot_type in ('float_pivot', 'int_pivot')
        continuum = self._reference_continuum
        min_dist_between_pivots = continuum.avg_length_unit / 2
        if self._pivot_type == 'int_pivot':
            min_dist_between_pivots = (continuum.avg_length_unit / 2)
        bound_inf, bound_sup = continuum.bounds
        new_continuum = Continuum()
        annotators = self._ground_truth_annotators
        # TODO: why not sample from the whole continuum?
        # TODO : shouldn't the sampled annotators nb be equal to the annotators amount?
        pivots = []
        for idx in range(continuum.num_annotators):
            if self._pivot_type == 'float_pivot':
                pivot: float = random.uniform(bound_inf, bound_sup)
                if min_dist_between_pivots is not None:
                    # While the pivot is closer than min_dist to a precedent pivot, pick another one
                    # (takes wrapping of continuum into consideration).
                    while any(map((lambda x: abs(x - pivot) < min_dist_between_pivots or
                                   abs(x - (pivot - bound_sup)) < min_dist_between_pivots),
                                  pivots)):
                        pivot = random.uniform(bound_inf, bound_sup)
            else:
                pivot: int = random.randint(np.floor(bound_inf), np.ceil(bound_sup))
                if min_dist_between_pivots is not None:
                    while any(map((lambda x: abs(x - pivot) < min_dist_between_pivots or
                                   abs(x - (pivot - bound_sup)) < min_dist_between_pivots),
                                  pivots)):
                        pivot = random.randint(np.floor(bound_inf), np.ceil(bound_sup))
            pivots.append(pivot)

            rnd_annotator = random.choice(list(annotators))
            units = continuum._annotations[rnd_annotator]
            sampled_annotation = SortedSet()
            for unit in units:
                if unit.segment.start + pivot > bound_sup:
                    new_segment = Segment(unit.segment.start + pivot + bound_inf - bound_sup,
                                          unit.segment.end + pivot + bound_inf - bound_sup)
                else:
                    new_segment = Segment(unit.segment.start + pivot,
                                          unit.segment.end + pivot)
                sampled_annotation.add(Unit(new_segment, unit.annotation))
            new_continuum._annotations[f'Sampled_annotation {idx}'] = sampled_annotation
        return new_continuum
