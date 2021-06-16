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
import os
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
from multiprocessing import Pool

from .dissimilarity import AbstractDissimilarity
from .numba_utils import chunked_cartesian_product

if TYPE_CHECKING:
    from .alignment import UnitaryAlignment, Alignment
    from .sampler import AbstractContinuumSampler, MathetContinuumSampler

CHUNK_SIZE = (10**5) // os.cpu_count()

# defining Annotator type
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
    """
    Representation of a continuum, i.e a set of annotated segments by multiple annotators.
    It is implemented as a dictionnary of sets (all sorted) :

    ``{'annotator1': {unit1, ...}, ...}``
    """
    uri: str
    _annotations: SortedDict

    def __init__(self, uri: Optional[str] = None):
        """
        Default constructor.

        Parameters
        ----------
        uri: optional str
            name of annotated resource (e.g. audio or video file)
        """
        self.uri = uri
        # Structure {annotator -> SortedSet}
        self._annotations: SortedDict = SortedDict()

    @classmethod
    def from_csv(cls,
                 path: Union[str, Path],
                 discard_invalid_rows=True,
                 delimiter: str = ","):
        """
        Load annotations from a CSV file , with structure
        annotator, category, segment_start, segment_end.

        .. warning::

            The CSV file mustn't have any header

        Parameters
        ----------
        path: Path or str
            Path to the CSV file storing annotations
        discard_invalid_rows: bool
            If set, every invalid row is ignored when parsing the file.
        delimiter: str
            CSV columns delimiter. Defaults to ','

        Returns
        -------
        Continuum:
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
        return all(len(annotations) > 0 for annotations in self._annotations.values())

    def __len__(self):
        return len(self._annotations)

    @property
    def num_units(self) -> int:
        """Total number of units in the continuum."""
        return sum(len(units) for units in self._annotations.values())

    @property
    def categories(self) -> SortedSet:
        """
        Returns the (alphabetically) sorted set of all the continuum's annotations's categories.
        """
        return SortedSet(unit.annotation for _, unit in self
                         if unit.annotation is not None)

    @property
    def category_weights(self) -> SortedDict:
        """
        Returns a dictionnary where the keys are the categories in the continuum, and a key's value
        is the proportion of occurence of the category in the continuum.
        """
        weights = SortedDict()
        nb_units = 0
        for _, unit in self:
            nb_units += 1
            if unit.annotation not in weights:
                weights[unit.annotation] = 1
            else:
                weights[unit.annotation] += 1
        for annotation in weights.keys():
            weights[annotation] /= nb_units
        return weights

    @property
    def bounds(self) -> Tuple[float, float]:
        """Start of 'smallest' annotation and end of 'largest' annotation."""
        bounds_inf = min(next(iter(annotation_set)).segment.start for annotation_set in self._annotations.values())
        bounds_sup = max(next(reversed(annotation_set)).segment.end for annotation_set in self._annotations.values())
        return bounds_inf, bounds_sup

    @property
    def bounds_narrow(self):
        """End of 'smallest' annotation and start of 'largest' annotation."""
        bounds_inf = min(next(iter(annotation_set)).segment.end for annotation_set in self._annotations.values())
        bounds_sup = max(next(reversed(annotation_set)).segment.start for annotation_set in self._annotations.values())
        return bounds_inf, bounds_sup



    @property
    def num_annotators(self) -> int:
        """Number of annotators"""
        return len(self._annotations)

    @property
    def avg_num_annotations_per_annotator(self) -> float:
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

    def add_annotator(self,  annotator: str):
        """
        Adds the annotator to the set, with no annotated segment. Does nothing if already present.
        """
        if annotator not in self._annotations:
            self._annotations[annotator] = SortedSet()

    def add(self, annotator: str, segment: Segment, annotation: Optional[str] = None):
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

    def add_annotation(self, annotator: str, annotation: Annotation):
        """
        Add a full pyannote annotation to the continuum.

        Parameters
        ----------
        annotator: str
            A string id for the annotator who produced that annotation.
        annotation: pyannote.core.Annotation
            A pyannote `Annotation` object. If a label is present for a given
            segment, it will be considered as that label's annotation.
        """
        for segment, _, label in annotation.itertracks(yield_label=True):
            self.add(annotator, segment, label)

    def add_timeline(self, annotator: str, timeline: Timeline):
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
                     annotator: str,
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
                 annotator: str,
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

    def merge(self, continuum: 'Continuum', in_place: bool = False) -> Optional['Continuum']:
        """
        Merge two Continuua together. Units from the same annotators
        are also merged together (with the usual order of units).

        :rtype: Continuum, optional

        Parameters
        ----------
        continuum: Continuum
            other continuum to merge into the current one.
        in_place: bool
            If set to true, the merge is done in place, and the current
            continuum (self) is the one being modified. A new continuum
            resulting in the merge is returned otherwise.
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
            the continuum to merge into `self`
        """
        return self.merge(other, in_place=False)

    def __setitem__(self, annotator: str, units: SortedSet):
        """
        Overwrites the annotators's set of units with the given one.
        >>> units = SortedSet((Unit(segment=Segment(12.5, 13.0), annotation='Verb')))
        >>> continuum['Victor'] = units
        >>> continuum['Victor']
        SortedSet([Unit(segment=<Segment(12.5, 13.0)>, annotation='Verb')]

        Parameters
        ----------
        annotator: str
            Any annotator, existing or not in the continuum.
        units: SortedSet of Unit
            A SortedSet of units, that will become the annotator's.
        """
        self._annotations[annotator] = units

    def __getitem__(self, keys: Union[str, Tuple[str, int]]) -> Union[SortedSet, Unit]:
        """Get a set of annotations from an annotator or a specific annotation.

        >>> continuum['Alex']
        SortedSet([Unit(segment=<Segment(2, 9)>, annotation='1'), Unit(segment=<Segment(11, 17)>, ...
        >>> continuum['Alex', 0]
        Unit(segment=<Segment(2, 9)>, annotation='1')

        Parameters
        ----------
        keys: Annotator or Annotator,int


        Raises
        ------
        KeyError
        """
        try:
            if isinstance(keys, str):
                return self._annotations[keys]
            else:
                annotator, idx = keys
                try:
                    return self._annotations[annotator][idx]
                except IndexError:
                    raise IndexError(f'index {idx} of annotations by {annotator} is out of range')
        except KeyError:
            raise KeyError('key must be either Annotator (from the continuum) or (Annotator, int)')

    def __iter__(self) -> Iterable[Tuple[str, Unit]]:
        for annotator, annotations in self._annotations.items():
            for unit in annotations:
                yield annotator, unit

    @property
    def annotators(self) -> SortedSet:
        """Returns a sorted set of the annotators in the Continuum

        >>> self.annotators:
        ... SortedSet(["annotator_a", "annotator_b", "annot_ref"])
        """
        return SortedSet(self._annotations.keys())

    def iterunits(self, annotator: str):
        """Iterate over units from the given annotator
        (in chronological and alphabetical order if annotations are present)

        >>> for unit in self.iterunits("Max"):
        ...     # do something with the unit
        """
        return iter(self._annotations[annotator])

    def get_best_alignment(self, dissimilarity: AbstractDissimilarity) -> 'Alignment':
        """
        Returns the best alignment of the continuum for the given dissimilarity. This alignment comes
        with the associated disorder, so you can obtain it in constant time with alignment.disorder.
        Beware that the computational complexity of the algorithm is very high
        :math:`(O(p_1 \\times p_2 \\times ... \\times p_n)` where :math:`p_i` is the number
        of annotations of annotator :math:`i`).

        Parameters
        ----------
        dissimilarity: AbstractDissimilarity
            the dissimilarity that will be used to compute unit-to-unit disorder.
        """
        assert len(self.annotators) >= 2, "Disorder cannot be computed with less than two annotators."

        disorder_args = dissimilarity.build_args(self)

        nb_unit_per_annot = []
        for annotator, arr in self._annotations.items():
            assert len(arr) > 0, f"Disorder cannot be computed because annotator {annotator} has no annotations."
            nb_unit_per_annot.append(len(arr) + 1)

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
        prob.solve()

        # compare with 0.9 as cvxpy returns 1.000 or small values i.e. 10e-14
        chosen_alignments_ids, = np.where(x.value > 0.9)
        chosen_alignments: np.ndarray = possible_unitary_alignments[chosen_alignments_ids]
        alignments_disorders: np.ndarray = disorders[chosen_alignments_ids]

        from .alignment import UnitaryAlignment, Alignment

        set_unitary_alignements = []
        for alignment_id, alignment in enumerate(chosen_alignments):
            u_align_tuple = []
            for annotator_id, unit_id in enumerate(alignment):
                annotator, units = self._annotations.peekitem(annotator_id)
                try:
                    unit = units[unit_id]
                    u_align_tuple.append((annotator, unit))
                except IndexError:  # it's a "null unit"
                    u_align_tuple.append((annotator, None))
            unitary_alignment = UnitaryAlignment(list(u_align_tuple))
            unitary_alignment.disorder = alignments_disorders[alignment_id]
            set_unitary_alignements.append(unitary_alignment)
        return Alignment(set_unitary_alignements,
                         continuum=self,
                         # Validity of results from get_best_alignments have been thoroughly tested :
                         check_validity=False,
                         disorder=alignments_disorders.sum() / self.avg_num_annotations_per_annotator)

    def compute_gamma(self,
                      dissimilarity: 'AbstractDissimilarity',
                      n_samples: int = 30,
                      precision_level: Optional[Union[float, PrecisionLevel]] = None,
                      ground_truth_annotators: Optional[SortedSet] = None,
                      sampler: 'AbstractContinuumSampler' = None,
                      random_seed: Optional[int] = 4577,
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
        ground_truth_annotators: SortedSet of str
            if set, the random continuua will only be sampled from these
            annotators. This should be used when you want to compare a prediction
            against some ground truth annotation.
        sampler: AbstractContinuumSampler
            Sampler object, which implements a sampling strategy for creating random continuua used
            to calculate the expected disorder. If not set, defaults to the same one as in the Gamma Software.
        random_seed: optional int
            random seed used to set up the random state before sampling the
            random continuua
        """
        from .sampler import MathetContinuumSampler
        if sampler is None:
            sampler = MathetContinuumSampler(self, ground_truth_annotators)

        if random_seed is not None:
            np.random.seed(random_seed)

        # Multiprocessed computation of sample disorder
        p = Pool()
        result_pool = [
            # Step one : computing the disorders of a batch of random samples from the continuum (done in parallel)
            p.apply_async(_compute_best_alignment_job,
                          (sampler.sample_from_continuum,
                           dissimilarity,)
                          )
            for _ in range(n_samples)]
        chance_best_alignments: List[Alignment] = []
        chance_disorders: List[float] = []
        logging.info(f"Starting computation for a batch of {n_samples} random samples...")
        for i, result in enumerate(result_pool):
            chance_best_alignments.append(result.get())
            logging.info(f"finished computation of random sample dissimilarity {i + 1}/{n_samples}")
            chance_disorders.append(chance_best_alignments[-1].disorder)
        logging.info("done.")

        if precision_level is not None:
            if isinstance(precision_level, str):
                precision_level = PRECISION_LEVEL[precision_level]
            assert 0 < precision_level < 1.0
            # If the variation of the disorders of the samples si too high, others are generated.
            # taken from subsection 5.3 of the original paper
            # confidence at 95%, i.e., 1.96
            variation_coeff = np.std(chance_disorders) / np.mean(chance_disorders)
            confidence = 1.96
            required_samples = np.ceil((variation_coeff * confidence / precision_level) ** 2).astype(np.int32)
            if required_samples > n_samples:
                result_pool = [
                    p.apply_async(_compute_best_alignment_job,
                                  (sampler.sample_from_continuum,
                                   dissimilarity,)
                                  )
                    for _ in range(required_samples - n_samples)
                ]
                logging.info(f"Computing second batch of {required_samples - n_samples} "
                             f"because variation was too high.")
                for i, result in enumerate(result_pool):
                    chance_best_alignments.append(result.get())
                    logging.info(f"finished computation of additionnal random sample dissimilarity "
                                 f"{i + 1}/{required_samples - n_samples}")
                logging.info("done.")

        p.close()
        # Step 2: find the best alignment of the continuum from there, gamma is rapidly calculed (cf GammaResults class)
        best_alignment = self.get_best_alignment(dissimilarity)

        return GammaResults(
            best_alignment=best_alignment,
            chance_alignments=chance_best_alignments,
            precision_level=precision_level,
            dissimilarity=dissimilarity
        )

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
    Gamma results object. Stores the informations about a gamma measure computation,
    used for getting the values of measures from the gamma family (gamma, gamma-cat and gamma-k).
    """
    best_alignment: 'Alignment'
    chance_alignments: List['Alignment']
    dissimilarity: AbstractDissimilarity
    precision_level: Optional[float] = None

    @property
    def n_samples(self):
        return len(self.chance_alignments)

    @property
    def alignments_nb(self):
        return len(self.best_alignment.unitary_alignments)

    @property
    def observed_disorder(self) -> float:
        """Returns the disorder of the computed best alignment, i.e, the
        observed agreement."""
        return self.best_alignment.disorder

    @property
    def observed_cat_disorder(self) -> float:
        return self.best_alignment.gamma_k_disorder(self.dissimilarity, None)

    def observed_k_disorder(self, category: str) -> float:
        return self.best_alignment.gamma_k_disorder(self.dissimilarity, category)

    @property
    def expected_disorder(self) -> float:
        """Returns the expected disagreement for computed random samples, i.e.,
        the mean of the sampled continuua's disorders"""
        return float(np.mean([align.disorder for align in self.chance_alignments]))

    @property
    def expected_cat_disorder(self) -> float:
        """
        Returns the expected disagreement (as defined for gamma-cat) using the same random samples
        as for gamma (the mean of the sampled continuua's gamma-cat disorders)
        """
        return float(np.mean(list(filter((lambda x: x is not np.NaN),
                                         (align.gamma_k_disorder(self.dissimilarity, None)
                                          for align in self.chance_alignments)))))

    def expected_k_disorder(self, category: str) -> float:
        """
        Returns the expected disagreement (as defined for gamma-k) using the same random samples
        as for gamma (the mean of the sampled continuua's gamma-k disorders)
        """
        return float(np.mean(list(filter((lambda x: x is not np.NaN),
                                         (align.gamma_k_disorder(self.dissimilarity, category)
                                          for align in self.chance_alignments)))))

    @property
    def approx_gamma_range(self):
        """Returns a tuple of the expected boundaries of the computed gamma,
         obtained using the expected disagreement and the precision level"""
        if self.precision_level is None:
            raise ValueError("No precision level has been set, cannot compute"
                             "the gamma boundaries")
        return (1 - self.observed_disorder / (self.expected_disorder *
                (1 - self.precision_level)),
                1 - self.observed_disorder / (self.expected_disorder *
                (1 + self.precision_level)))

    @property
    def gamma(self) -> float:
        """Returns the gamma value"""
        return max(1 - self.observed_disorder / self.expected_disorder, 0.0)

    @property
    def gamma_cat(self) -> float:
        """Returns the gamma-cat value"""
        return max(1 - self.observed_cat_disorder / self.expected_cat_disorder, 0.0)

    def gamma_k(self, category: str) -> float:
        """Returns the gamma-k value"""
        return max(1 - self.observed_k_disorder(category) / self.expected_k_disorder(category), 0.0)


def _compute_best_alignment_job(continuum: Continuum, dissimilarity: AbstractDissimilarity):
    """
    Function used to launch a multiprocessed job for calculating the best aligment of a continuum
    using the given dissimilarity.
    """
    return continuum.get_best_alignment(dissimilarity)
