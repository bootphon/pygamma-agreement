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
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union, Set, Iterable

import cvxpy as cp
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pyannote.core import Annotation, Segment, Timeline
from sortedcontainers import SortedDict
from tqdm import tqdm

from .alignment import UnitaryAlignment, Alignment
from .dissimilarity import AbstractDissimilarity, PositionalDissimilarity, CombinedCategoricalDissimilarity
from .numba_utils import chunked_cartesian_product

CHUNK_SIZE = 2 ** 25

# defining Annotator type
Annotator = str


@dataclass
class Unit:
    segment: Segment
    annotation: Optional[str] = None


class Continuum:
    """Continuum

    Parameters
    ----------
    uri : string, optional
        name of annotated resource (e.g. audio or video file)
    modality : string, optional
        name of annotated modality
    """

    @classmethod
    def from_rttm(cls, df: pd.DataFrame):
        raise NotImplemented()

    @classmethod
    def from_csv(cls, path: Union[str, Path], discard_invalid_rows=True):
        if isinstance(path, str):
            path = Path(path)

        continuum = cls()
        with open(path) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                seg = Segment(float(row[4]), float(row[5]))
                try:
                    continuum.add(row[1], seg, row[2])
                except ValueError as e:
                    if discard_invalid_rows:
                        print(f"Discarded invalid segment : {str(e)}")
                    else:
                        raise e

        return continuum

    @classmethod
    def sample_from_continuum(cls, continuum: 'Continuum',
                              pivot_type: str = "float_pivot") -> 'Continuum':
        assert pivot_type in ('float_pivot', 'int_pivot')
        """Generate a new random annotation from a single continuum
                Strategy from figure 12

                >>> gamma_agreement.sample_from_single_continuum()
                ... <pyannote.core.annotation.Annotation at 0x7f5527a19588>
                """
        last_start_time = max(unit.segment.start for unit in continuum.iterunits())
        if pivot_type == 'float_pivot':
            pivot = np.random.uniform(continuum.avg_length_unit, last_start_time)
        else:
            pivot = np.random.randint(continuum.avg_length_unit, last_start_time)
        new_continuum = Continuum()
        # TODO: why not sample from the whole continuum?
        for idx in range(continuum.num_annotators):
            rnd_annotator = np.random.choice(continuum.annotators)
            units = continuum._annotations[rnd_annotator]
            sampled_annotation = SortedDict()
            for segment, unit in units.items():
                if pivot - segment.start < 0:
                    new_segment = Segment(segment.start - pivot, segment.end - pivot)
                else:
                    new_segment = Segment(segment.start + pivot, segment.end + pivot)
                sampled_annotation[new_segment] = Unit(new_segment, unit.annotation)
            new_continuum._annotations[f'Sampled_annotation {idx}'] = sampled_annotation
        return new_continuum

    @classmethod
    def sample_from_corpus(cls, corpus: 'Corpus',
                           pivot_type: str = "float_pivot") -> 'Continuum':
        assert pivot_type in ('float_pivot', 'int_pivot')
        # TODO

    def __init__(self, uri: Optional[str] = None):
        self.uri = uri
        # Structure {annotator -> { segment -> unit}}
        self._annotations = SortedDict()
        # a compact, array-shaped representation of each annotators' units
        # needed for fast computation of inter-units disorders
        # computed and set when compute_disorder is called
        self._positions_arrays: List[np.ndarray] = None
        self._categories_arrays: List[np.ndarray] = None

        # these are instanciated when compute_disorder is called
        self._chosen_alignments: np.ndarray = None
        self._alignments_disorders: np.ndarray = None

    def __bool__(self):
        """Truthiness, basically tests for emptiness

        >>> if continuum:
        ...    # continuum is not empty
        ... else:
        ...    # continuum is empty
        """
        return len(self._annotations) > 0

    @property
    def num_units(self) -> int:
        """Number of units"""
        return sum(len(units) for units in self._annotations.values())

    @property
    def categories(self) -> Set[str]:
        return set(unit.annotation for unit in self.iterunits()
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
    def avg_length_unit(self):
        """Mean of the annotated segments' durations"""
        return sum(unit.segment.duration for unit in self.iterunits()) / self.num_units

    def add(self, annotator: Annotator, segment: Segment, annotation: Optional[str] = None):
        if segment.duration == 0.0: # TODO: use pyannote segment precision?
            raise ValueError("Added segment of duration 0.0")

        if annotator not in self._annotations:
            self._annotations[annotator] = SortedDict()

        self._annotations[annotator][segment] = Unit(segment, annotation)

        # units array has to be updated, nullifying
        if self._positions_arrays is not None:
            self._positions_arrays = None
            self._categories_arrays = None

    def add_annotation(self, annotator: Annotator, annotation: Annotation):
        for label in annotation.labels():
            for segment in annotation.label_timeline(label):
                self.add(annotator, segment, label)

    def add_timeline(self, annotator: Annotator, timeline: Timeline):
        for segment in timeline:
            self.add(annotator, segment)

    def __getitem__(self, keys: Union[Annotator, Tuple[Annotator, Segment]]):
        """Get annotation object

        >>> annotation = continuum[annotator]
        """
        if len(keys) == 1:
            annotator = keys[0]
            return self._annotations[annotator]
        elif len(keys) == 2 and isinstance(keys[2], Segment):
            annotator, segment = keys
            return self._annotations[annotator][segment]

    def __iter__(self) -> Iterable[Tuple[Annotator, SortedDict]]:
        return iter(self._annotations.items())

    @property
    def annotators(self):
        """List all annotators in the Continuum
        # TODO: doc example
        """
        return list(self._annotations.keys())

    def iterunits(self):
        """Iterate over units (in chronological order)

        >>> for annotator in annotation.iterannotators():
        ...     # do something with the annotator
        """
        for units in self._annotations.values():
            yield from units.values()

    def compute_disorder(self, dissimilarity: AbstractDissimilarity):
        assert isinstance(dissimilarity, AbstractDissimilarity)
        if isinstance(dissimilarity, PositionalDissimilarity):
            self._positions_arrays = dissimilarity.build_positions_arrays(self)
            disorder_args = (self._positions_arrays, )
        elif isinstance(dissimilarity, CombinedCategoricalDissimilarity):
            self._positions_arrays = dissimilarity.build_positions_arrays(self)
            self._categories_arrays = dissimilarity.build_categories_arrays(self)
            disorder_args = (self._positions_arrays, self._categories_arrays)
        else:
            raise TypeError(f"dissimilarity argument is of unexpected type "
                            f"{type(dissimilarity)}")

        nb_unit_per_annot = [len(arr) + 1 for arr in self._annotations.values()]
        number_tuples = np.product(nb_unit_per_annot)
        all_disorders = []
        all_valid_tuples = []
        num_chunks = (number_tuples // CHUNK_SIZE) + 1
        for tuples_batch in tqdm(chunked_cartesian_product(nb_unit_per_annot, CHUNK_SIZE),
                                 total=num_chunks):
            disorders = dissimilarity(tuples_batch, *disorder_args)

            # Property section 5.1.1 to reduce initial complexity
            valid_disorders_ids = np.where(disorders < self.num_annotators * dissimilarity.delta_empty)
            all_disorders.append(disorders[valid_disorders_ids])
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
        print(A.shape)

        for p_id, unit_ids_tuple in enumerate(possible_unitary_alignments):
            for annotator_id, unit_id in enumerate(unit_ids_tuple):
                if unit_id != len(true_units_ids[annotator_id]):
                   A[true_units_ids[annotator_id][unit_id], p_id] = 1

        obj = cp.Minimize(disorders.T * x)
        constraints = [cp.matmul(A, x) == 1]
        prob = cp.Problem(obj, constraints)

        # we don't actually care about the optimal disorder value
        optimal_disorder = prob.solve()

        # compare with 0.9 as cvxpy returns 1.000 or small values i.e. 10e-14
        chosen_alignments_ids,  = np.where(x.value > 0.9)
        self._chosen_alignments = possible_unitary_alignments[chosen_alignments_ids]
        self._alignments_disorders = disorders[chosen_alignments_ids]
        return self._alignments_disorders.sum() / len(self._alignments_disorders)

    def get_best_alignment(self, dissimilarity: Optional[AbstractDissimilarity] = None):
        if self._chosen_alignments is None or self._alignments_disorders is None:
            if dissimilarity is not None:
                self.compute_disorder(dissimilarity)
            else:
                raise ValueError("Best alignment disorder hasn't been computed, "
                                 "a the dissimilarity argument is required")


        set_unitary_alignements = []
        for alignment_id, alignment in enumerate(self._chosen_alignments):
            u_align_tuple = []
            for annotator_id, unit_id in enumerate(alignment):
                annotator, units = self._annotations.peekitem(annotator_id)
                try:
                    _, unit = units.peekitem(unit_id)
                    u_align_tuple.append((annotator, unit))
                except IndexError: # it's a "null unit"
                    u_align_tuple.append((annotator, None))
            unitary_alignment = UnitaryAlignment(tuple(u_align_tuple))
            unitary_alignment._disorder = self._alignments_disorders[alignment_id]
            set_unitary_alignements.append(unitary_alignment)
        return Alignment(self, set_unitary_alignements)

    def compute_gamma(self,
                      dissimilarity: AbstractDissimilarity,
                      strategy: str = "single",
                      pivot_type: str = "float_pivot",
                      number_samples: int = 30,
                      corpus: Optional['Corpus'] = None):
        assert strategy in ("single", "multi")
        chance_disorders = []
        for _ in range(number_samples):
            if strategy == "single":
                sampled_continuum = Continuum.sample_from_continuum(self, pivot_type)
            elif strategy == "multi":
                assert corpus is not None
                sampled_continuum = Continuum.sample_from_corpus(corpus, pivot_type)
            chance_disorders.append(sampled_continuum.compute_disorder(dissimilarity))

        best_align_disorder = self.compute_disorder(dissimilarity)
        return 1 - (best_align_disorder / np.mean(chance_disorders))



class Corpus:
    """Corpus

    Parameters
    ----------
    uri : string, optional
        name of annotated resource (e.g. audio or video file)
    modality : string, optional
        name of annotated modality
    """

    def __init__(self, uri=None, modality=None):

        self._uri = uri
        self.modality = modality

        # sorted dictionary
        # keys: annotators
        # values: {annotator: annotations} dictionary
        self._annotators: Dict[Annotator, Annotation] = SortedDict()

        # sorted dictionary
        # keys: file_name_annotated
        # values: {file_name_annotated: continuum} dictionary
        self._continuua: Dict[str, Continuum] = SortedDict()

    def _get_uri(self):
        return self._uri

    def __len__(self):
        """Number of annotators

        >>> len(corpus)  # corpus contains 2 annotated files
        2
        """
        return len(self._continuua)

    def __bool__(self):
        """Emptiness

        >>> if corpus:
        ...    # corpus is not empty
        ... else:
        ...    # corpus is empty
        """
        return len(self._continuua) > 0

    @property
    @lru_cache(maxsize=None)
    def num_units(self):
        """Number of units across continua"""
        num_units = 0
        for continuum in self._continuua:
            num_units += self._continuua[continuum].num_units
        return num_units

    @property
    @lru_cache(maxsize=None)
    def avg_num_annotations_per_annotator(self):
        return self.num_units / len(self)

    def __setitem__(self, file_name_annotated: str,
                    continuum: Continuum):
        """Add new or update existing Continuum

        >>> corpus[file_name_annotated] = Continuum
        If file_name_annotated does not exist, it is added.
        If file_name_annotated already exists, it is updated.

        Note
        ----
        If `Continuum` is empty, it does nothing.
        """

        # do not add empty annotation
        if not continuum:
            return

        self._continuua[file_name_annotated] = continuum

    def __getitem__(self, file_name_annotated: str):
        """Get continuum object

        >>> continuum = corpus[file_name_annotated]
        """

        return self._continuua[file_name_annotated]

    def itercontinuua(self):
        """Iterate over continuum (in chronological order)

        >>> for continuum in corpus.itercontinuua():
        ...     # do something with the continuum
        """
        return iter(self._continuua)
