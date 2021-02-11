#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CoML

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
# Rachid RIAD & Hadrien TITEUX
"""
##########
Continuum and corpus
##########

"""
import csv
import logging
import random
from copy import deepcopy
from functools import total_ordering
from pathlib import Path
from typing import Optional, Tuple, List, Union, Set, Iterable, TYPE_CHECKING, Dict

import cvxpy as cp
import numpy as np
from dataclasses import dataclass
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm
from sortedcontainers import SortedDict, SortedSet
from typing_extensions import Literal

from .dissimilarity import AbstractDissimilarity
from .numba_utils import chunked_cartesian_product

if TYPE_CHECKING:
    from .alignment import UnitaryAlignment, Alignment

CHUNK_SIZE = 2 ** 25

# defining Annotator type
Annotator = str
PivotType = Literal["float_pivot", "int_pivot"]
PrecisionLevel = Literal["high", "medium", "low"]

# percentages for the precision
PRECISION_LEVEL = {
    "high": 0.01,
    "medium": 0.02,
    "low": 0.1
}


@total_ordering
@dataclass(frozen=True, eq=True)
class Unit:
    """
    Represents an annotated unit, e.g., a time segment and (optionally)
    a text annotation. Can be sorted or used in a set. If two units share
    the same time segment, they're sorted alphabetically using their
    annotation. The `None` annotation is first in the "alphabet"
    """
    segment: Segment
    annotation: Optional[str] = None

    def __lt__(self, other: 'Unit'):
        if self.segment == other.segment:
            if self.annotation is None:
                return True
            elif other.annotation is None:
                return False
            else:
                return self.annotation < other.annotation
        else:
            return self.segment < other.segment


class Continuum:
    """Continuum

    Parameters
    ----------
    uri : string, optional
        name of annotated resource (e.g. audio or video file)
    """

    @classmethod
    def from_csv(cls,
                 path: Union[str, Path],
                 discard_invalid_rows=True,
                 delimiter: str = ","):
        """
        Load annotations from a CSV file , with structure
        annotator, category, segment_start, segment_end.

        .. warning::

            The CSV file shouldn't have any header

        Parameters
        ----------
        path: path or str
            Path to the CSV file storing annotations
        discard_invalid_rows: bool
            Path: if a row contains invalid annotations, discard it)
        delimiter: str, default ","
            CSV delimiter

        Returns
        -------
        continuum : Continuum
            New continuum object loaded from the CSV

        """
        if isinstance(path, str):
            path = Path(path)

        continuum = cls()
        with open(path) as csv_file:
            reader = csv.reader(csv_file, delimiter=delimiter)
            for row in reader:
                seg = Segment(float(row[2]), float(row[3]))
                try:
                    continuum.add(row[0], seg, row[1])
                except ValueError as e:
                    if discard_invalid_rows:
                        print(f"Discarded invalid segment : {str(e)}")
                    else:
                        raise e

        return continuum

    @classmethod
    def from_rttm(cls, path: Union[str, Path]) -> 'Continuum':
        """
        Load annotations from a RTTM file. The file name field will be used
        as an annotation's annotator

        Parameters
        ----------
        path: Path or str
            Path to the CSV file storing annotations

        Returns
        -------
        continuum : Continuum
            New continuum object loaded from the RTTM file
        """
        annotations = load_rttm(str(path))
        continuum = cls()
        for uri, annot in annotations.items():
            continuum.add_annotation(uri, annot)
        return continuum

    @classmethod
    def sample_from_continuum(cls, continuum: 'Continuum',
                              pivot_type: PivotType = "float_pivot",
                              ground_truth_annotators: Optional[List[Annotator]] = None) -> 'Continuum':
        """Generate a new random annotation from a single continuum
                Strategy from figure 12

                >>> continuum.sample_from_continuum()
                ... <pygamma_agreement.continuum.Continuum at 0x7f5527a19588>
                """
        assert pivot_type in ('float_pivot', 'int_pivot')

        last_start_time = max(unit.segment.start for _, unit in continuum)
        new_continuum = Continuum()
        if ground_truth_annotators is not None:
            assert set(continuum.annotators).issuperset(set(ground_truth_annotators))
            annotators = ground_truth_annotators
        else:
            annotators = continuum.annotators

        # TODO: why not sample from the whole continuum?
        # TODO : shouldn't the sampled annotators nb be equal to the annotators amount?
        for idx in range(continuum.num_annotators):
            if pivot_type == 'float_pivot':
                pivot = random.uniform(continuum.avg_length_unit, last_start_time)
            else:
                pivot = random.randint(np.floor(continuum.avg_length_unit),
                                       np.ceil(last_start_time))

            rnd_annotator = random.choice(annotators)
            units = continuum._annotations[rnd_annotator]
            sampled_annotation = SortedSet()
            for unit in units:
                if pivot < unit.segment.start:
                    new_segment = Segment(unit.segment.start - pivot,
                                          unit.segment.end - pivot)
                else:
                    new_segment = Segment(unit.segment.start + pivot,
                                          unit.segment.end + pivot)
                sampled_annotation.add(Unit(new_segment, unit.annotation))
            new_continuum._annotations[f'Sampled_annotation {idx}'] = sampled_annotation
        return new_continuum

    def __init__(self, uri: Optional[str] = None):
        self.uri = uri
        # Structure {annotator -> SortedSet[Unit]}
        self._annotations: Dict[Annotator, Set[Unit]] = SortedDict()

        # these are instanciated when compute_disorder is called
        self._chosen_alignments: Optional[np.ndarray] = None
        self._alignments_disorders: Optional[np.ndarray] = None

    def copy(self) -> 'Continuum':
        """
        Makes a copy of the current continuum.

        Returns
        -------
        continuum: Continuum
        """
        continuum = Continuum(self.uri)
        continuum._annotations = deepcopy(self._annotations)
        return continuum

    def __bool__(self):
        """Truthiness, basically tests for emptiness

        >>> if continuum:
        ...    # continuum is not empty
        ... else:
        ...    # continuum is empty
        """
        return len(self._annotations) > 0

    def __len__(self):
        return len(self._annotations)

    @property
    def num_units(self) -> int:
        """Number of units"""
        return sum(len(units) for units in self._annotations.values())

    @property
    def categories(self) -> Set[str]:
        return set(unit.annotation for _, unit in self
                   if unit.annotation is not None)

    @property
    def num_annotators(self) -> int:
        """Number of annotators"""
        return len(self._annotations)

    @property
    def avg_num_annotations_per_annotator(self):
        """Average number of annotated segments per annotator"""
        return self.num_units / self.num_annotators

    @property
    def max_num_annotations_per_annotator(self):
        """The maximum number of annotated segments an annotator has
        in this continuum"""
        max_num_annotations_per_annotator = 0
        for annotator in self._annotations:
            max_num_annotations_per_annotator = np.max(
                [max_num_annotations_per_annotator,
                 len(self[annotator])])
        return max_num_annotations_per_annotator

    @property
    def avg_length_unit(self) -> float:
        """Mean of the annotated segments' durations"""
        return sum(unit.segment.duration for _, unit in self) / self.num_units

    def add(self, annotator: Annotator, segment: Segment, annotation: Optional[str] = None):
        """
        Add a segment to the continuum

        Parameters
        ----------
        annotator: str
            The annotator that produced the added annotation
        segment: `pyannote.core.Segment`
            The segment for that annotation
        annotation: optional str
            That segment's annotation, if any.
        """
        if segment.duration == 0.0:
            raise ValueError("Tried adding segment of duration 0.0")

        if annotator not in self._annotations:
            self._annotations[annotator] = SortedSet()

        self._annotations[annotator].add(Unit(segment, annotation))

        # units array has to be updated, nullifying
        if self._alignments_disorders is not None:
            self._chosen_alignments = None
            self._alignments_disorders = None

    def add_annotation(self, annotator: Annotator, annotation: Annotation):
        """
        Add a full pyannote annotation to the continuum.

        Parameters
        ----------
        annotator: str
            A string id for the annotator who produced that annotation.
        annotation: :class:`pyannote.core.Annotation`
            A pyannote `Annotation` object. If a label is present for a given
            segment, it will be considered as that label's annotation.
        """
        for segment, _, label in annotation.itertracks(yield_label=True):
            self.add(annotator, segment, label)

    def add_timeline(self, annotator: Annotator, timeline: Timeline):
        """
        Add a full pyannote timeline to the continuum.

        Parameters
        ----------
        annotator: str
            A string id for the annotator who produced that timeline.
        timeline: `pyannote.core.Timeline`
            A pyannote `Annotation` object. No annotation will be attached to
            segments.
        """
        for segment in timeline:
            self.add(annotator, segment)

    def add_textgrid(self,
                     annotator: Annotator,
                     tg_path: Union[str, Path],
                     selected_tiers: Optional[List[str]] = None,
                     use_tier_as_annotation: bool = False):
        """
        Add a textgrid file's content to the Continuum

        Parameters
        ----------
        annotator: str
            A string id for the annotator who produced that TextGrid.
        tg_path: `Path` or str
            Path to the textgrid file.
        selected_tiers: optional list of str
            If set, will drop tiers that are not contained in this list.
        use_tier_as_annotation: optional bool
            If True, the annotation for each non-empty interval will be the name
            of its parent Tier.
        """
        from textgrid import TextGrid, IntervalTier
        tg = TextGrid.fromFile(str(tg_path))
        for tier_name in tg.getNames():
            if selected_tiers is not None and tier_name not in selected_tiers:
                continue
            tier: IntervalTier = tg.getFirst(tier_name)
            for interval in tier:
                if not interval.mark:
                    continue

                if use_tier_as_annotation:
                    self.add(annotator,
                             Segment(interval.minTime, interval.maxTime),
                             tier_name)
                else:
                    self.add(annotator,
                             Segment(interval.minTime, interval.maxTime),
                             interval.mark)

    def add_elan(self,
                 annotator: Annotator,
                 eaf_path: Union[str, Path],
                 selected_tiers: Optional[List[str]] = None,
                 use_tier_as_annotation: bool = False):
        """
        Add an Elan (.eaf) file's content to the Continuum

        Parameters
        ----------
        annotator: str
            A string id for the annotator who produced that ELAN file.
        eaf_path: `Path` or str
            Path to the .eaf (ELAN) file.
        selected_tiers: optional list of str
            If set, will drop tiers that are not contained in this list.
        use_tier_as_annotation: optional bool
            If True, the annotation for each non-empty interval will be the name
            of its parent Tier.
        """
        from pympi import Eaf
        eaf = Eaf(eaf_path)
        for tier_name in eaf.get_tier_names():
            if selected_tiers is not None and tier_name not in selected_tiers:
                continue
            for start, end, value in eaf.get_annotation_data_for_tier(tier_name):
                if use_tier_as_annotation:
                    self.add(annotator, Segment(start, end), tier_name)
                else:
                    self.add(annotator, Segment(start, end), value)

    def merge(self, continuum: 'Continuum', in_place: bool = False) \
            -> Optional['Continuum']:
        """
        Merge two Continuua together. Units from the same annotators
        are also merged together.

        Parameters
        ----------
        continuum: Continuum
            other continuum to merge the current one with.
        in_place: optional bool
            If set to true, the merge is done in place, and the current
            continuum (self) is the one being modified.

        Returns
        -------
        continuum: optional Continuum
            Only returned if "in_place" is false

        """
        current_cont = self if in_place else self.copy()
        for annotator, unit in continuum:
            current_cont.add(annotator, unit.segment, unit.annotation)
        if not in_place:
            return current_cont

    def __add__(self, other: 'Continuum'):
        """
        Same as a "not-in-place" merge.

        Parameters
        ----------
        other: Continuum

        Returns
        -------
        continuum: Continuum

        See also
        --------
        :meth:`pygamma_agreement.Continuum.merge`
        """
        return self.merge(other, in_place=False)

    def __getitem__(self, *keys: Union[Annotator, Tuple[Annotator, int]]) \
            -> Union[SortedSet, Unit]:
        """Get annotation object

        >>> annotation = continuum[annotator]
        """
        if len(keys) == 1:
            annotator = keys[0]
            return self._annotations[annotator]
        elif len(keys) == 2 and isinstance(keys[1], int):
            annotator, idx = keys
            return self._annotations[annotator][idx]

    def __iter__(self) -> Iterable[Tuple[Annotator, Unit]]:
        for annotator, annotations in self._annotations.items():
            for unit in annotations:
                yield annotator, unit

    @property
    def annotators(self):
        """List all annotators in the Continuum

        >>> continuum.annotators:
        ... ["annotator_a", "annotator_b", "annot_ref"]
        """
        return list(self._annotations.keys())

    def iterunits(self, annotator: str):
        # TODO: implem and doc
        """Iterate over units (in chronological and alphabetical order
        if annotations are present)

        >>> for unit in continuum.iterunits("Max"):
        ...     # do something with the unit
        """
        return iter(self._annotations)

    def compute_disorders(self, dissimilarity: AbstractDissimilarity):
        assert isinstance(dissimilarity, AbstractDissimilarity)
        assert len(self.annotators) >= 2

        disorder_args = dissimilarity.build_args(self)

        nb_unit_per_annot = [len(arr) + 1 for arr in self._annotations.values()]
        all_disorders = []
        all_valid_tuples = []
        for tuples_batch in chunked_cartesian_product(nb_unit_per_annot, CHUNK_SIZE):
            batch_disorders = dissimilarity(tuples_batch, *disorder_args)

            # Property section 5.1.1 to reduce initial complexity
            valid_disorders_ids, = np.where(batch_disorders < self.num_annotators * dissimilarity.delta_empty)
            all_disorders.append(batch_disorders[valid_disorders_ids])
            all_valid_tuples.append(tuples_batch[valid_disorders_ids])

        disorders = np.concatenate(all_disorders)
        possible_unitary_alignments = np.concatenate(all_valid_tuples)

        # Definition of the integer linear program
        num_possible_unitary_alignements = len(disorders)
        x = cp.Variable(shape=num_possible_unitary_alignements, boolean=True)

        true_units_ids = []
        num_units = 0
        for units in self._annotations.values():
            true_units_ids.append(np.arange(num_units, num_units + len(units)).astype(np.int32))
            num_units += len(units)

        # Constraints matrix
        A = np.zeros((num_units, num_possible_unitary_alignements))

        for p_id, unit_ids_tuple in enumerate(possible_unitary_alignments):
            for annotator_id, unit_id in enumerate(unit_ids_tuple):
                if unit_id != len(true_units_ids[annotator_id]):
                    A[true_units_ids[annotator_id][unit_id], p_id] = 1

        obj = cp.Minimize(disorders.T @ x)
        constraints = [cp.matmul(A, x) == 1]
        prob = cp.Problem(obj, constraints)

        # we don't actually care about the optimal loss value
        optimal_value = prob.solve()

        # compare with 0.9 as cvxpy returns 1.000 or small values i.e. 10e-14
        chosen_alignments_ids, = np.where(x.value > 0.9)
        self._chosen_alignments = possible_unitary_alignments[chosen_alignments_ids]
        self._alignments_disorders = disorders[chosen_alignments_ids]
        return self._alignments_disorders.sum() / len(self._alignments_disorders)

    def get_best_alignment(self, dissimilarity: Optional['AbstractDissimilarity'] = None):
        if self._chosen_alignments is None or self._alignments_disorders is None:
            if dissimilarity is not None:
                self.compute_disorders(dissimilarity)
            else:
                raise ValueError("Best alignment disorder hasn't been computed, "
                                 "a the dissimilarity argument is required")

        from .alignment import UnitaryAlignment, Alignment

        set_unitary_alignements = []
        for alignment_id, alignment in enumerate(self._chosen_alignments):
            u_align_tuple = []
            for annotator_id, unit_id in enumerate(alignment):
                annotator, units = self._annotations.peekitem(annotator_id)
                try:
                    unit = units[unit_id]
                    u_align_tuple.append((annotator, unit))
                except IndexError:  # it's a "null unit"
                    u_align_tuple.append((annotator, None))
            unitary_alignment = UnitaryAlignment(tuple(u_align_tuple))
            unitary_alignment.disorder = self._alignments_disorders[alignment_id]
            set_unitary_alignements.append(unitary_alignment)
        return Alignment(set_unitary_alignements, continuum=self, check_validity=True)

    def compute_gamma(self,
                      dissimilarity: 'AbstractDissimilarity',
                      n_samples: int = 30,
                      precision_level: Optional[Union[float, PrecisionLevel]] = None,
                      ground_truth_annotators: Optional[List[Annotator]] = None,
                      sampling_strategy: str = "single",
                      pivot_type: PivotType = "float_pivot",
                      random_seed: Optional[float] = 4577
                      ) -> 'GammaResults':
        """

        Parameters
        ----------
        dissimilarity: AbstractDissimilarity
            dissimilarity instance. Used to compute the disorder between units.
        n_samples: optional int
            number of random continuum sampled from this continuum  used to
            estimate the gamma measure
        precision_level: optional float or "high", "medium", "low"
            error percentage of the gamma estimation. If a literal
            precision level is passed (e.g. "medium"), the corresponding numerical
            value will be used (high: 1%, medium: 2%, low : 5%)
        ground_truth_annotators:
            if set, the random continuua will only be sampled from these
            annotators. This should be used when you want to compare a prediction
            against some ground truth annotation.
        pivot_type: 'float_pivot' or 'int_pivot'
            pivot type to be used when sampling continuua
        random_seed: optional float, int or str
            random seed used to set up the random state before sampling the
            random continuua

        Returns
        -------

        """
        assert sampling_strategy in ("single", "multi")
        if sampling_strategy == "multi":
            raise NotImplemented("Multi-continuum sampling strategy is not "
                                 "supported for now")

        if random_seed is not None:
            random.seed(random_seed)

        chance_disorders = []
        for _ in range(n_samples):
            sampled_continuum = Continuum.sample_from_continuum(self, pivot_type, ground_truth_annotators)
            sample_disorder = sampled_continuum.compute_disorders(dissimilarity)
            chance_disorders.append(sample_disorder)

        if precision_level is not None:
            if isinstance(precision_level, str):
                precision_level = PRECISION_LEVEL[precision_level]
            assert 0 < precision_level < 1.0

            # taken from subsection 5.3 of the original paper
            # confidence at 95%, i.e., 1.96
            variation_coeff = np.std(chance_disorders) / np.mean(chance_disorders)
            confidence = 1.96
            required_samples = np.ceil((variation_coeff * confidence / precision_level) ** 2).astype(np.int32)
            logging.debug(f"Number of required samples for confidence {precision_level}: {required_samples}")
            if required_samples > n_samples:
                for _ in range(required_samples - n_samples):
                    sampled_continuum = Continuum.sample_from_continuum(self, pivot_type, ground_truth_annotators)
                    sample_disorder = sampled_continuum.compute_disorders(dissimilarity)
                    chance_disorders.append(sample_disorder)

        best_alignment = self.get_best_alignment(dissimilarity)

        return GammaResults(
            best_alignment=best_alignment,
            pivot_type=pivot_type,
            n_samples=n_samples,
            chance_disorders=np.array(chance_disorders),
            precision_level=precision_level
        )

    def compute_gamma_cat(self):
        raise NotImplemented()

    def to_csv(self, path: Union[str, Path], delimiter=","):
        if isinstance(path, str):
            path = Path(path)
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=delimiter)
            for annotator, unit in self:
                writer.writerow([annotator, unit.annotation,
                                 unit.segment.start, unit.segment.end])

    def _repr_png_(self):
        """IPython notebook support

        See also
        --------
        :mod:`pygamma_agreement.notebook`
        """

        from .notebook import repr_continuum
        return repr_continuum(self)


@dataclass
class GammaResults:
    """
    Gamma results object. Stores information about a gamma measure computation.
    """
    best_alignment: 'Alignment'
    pivot_type: PivotType
    n_samples: int
    chance_disorders: np.ndarray
    precision_level: Optional[float] = None

    @property
    def alignments_nb(self):
        return len(self.best_alignment.unitary_alignments)

    @property
    def observed_agreement(self) -> float:
        """Returns the disorder of the computed best alignment, i.e, the
        observed agreement."""
        return self.best_alignment.disorder

    @property
    def expected_disagreement(self) -> float:
        """Returns the expected disagreement for computed random samples, i.e.,
        the mean of the sampled continuua's disorders"""
        return self.chance_disorders.mean()

    @property
    def approx_gamma_range(self):
        """Returns a tuple of the expected boundaries of the computed gamma,
         obtained using the expected disagreement and the precision level"""
        if self.precision_level is None:
            raise ValueError("No precision level has been set, cannot compute"
                             "the gamma boundaries")
        return (1 - self.observed_agreement / (self.expected_disagreement *
                                               (1 - self.precision_level)),
                1 - self.observed_agreement / (self.expected_disagreement *
                                               (1 + self.precision_level)))

    @property
    def gamma(self):
        """Returns the gamma value"""
        return 1 - self.observed_agreement / self.expected_disagreement
