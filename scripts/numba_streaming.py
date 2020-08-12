import csv
import time
from collections import defaultdict
from typing import List, Dict, Tuple

import cvxpy as cp
import numba as nb
import numpy as np
import pyannote.core as pa
from dataclasses import dataclass, field
from sortedcontainers import SortedDict
import itertools
from tqdm import tqdm
from multiprocess import pool


class Timer:

    def start(self):
        self.start_time = time.time()
        self.lap_time = self.start_time

    def lap(self):
        last_time = self.lap_time
        self.lap_time = time.time()
        print("Done in ", self.lap_time - last_time)

    def total(self):
        print("All done in ", time.time() - self.start_time)


@dataclass
class Unit:
    segment: pa.Segment
    annot: str


@dataclass
class UnitaryAlignment:
    units: Tuple[Tuple[str, Unit]]


@dataclass
class Alignment:
    unitary_alignments: List[UnitaryAlignment] = field(default_factory=list)


LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')


@nb.njit
def fast_factorial(n):
    """Fast factorial is computed using a simple lookup table"""
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]


@nb.njit(nb.types.Tuple((nb.int32[:, :],
                         nb.int32[:]))(nb.int32[:],
                                       nb.int32[:],
                                       nb.int64, nb.int64))
def cproduct(sizes: np.ndarray, current_tuple: np.ndarray, start_idx: int, end_idx: int):
    assert len(sizes) >= 2
    assert start_idx < end_idx

    tuples = np.zeros((end_idx - start_idx, len(sizes)), dtype=np.int32)
    tuple_idx = 0
    current_tuple = current_tuple.copy()
    while tuple_idx < end_idx - start_idx:
        tuples[tuple_idx] = current_tuple
        current_tuple[0] += 1
        if current_tuple[0] == sizes[0]:
            current_tuple[0] = 0
            current_tuple[1] += 1
            for i in range(1, len(sizes) - 1):
                if current_tuple[i] == sizes[i]:
                    current_tuple[i + 1] += 1
                    current_tuple[i] = 0
                else:
                    break
        tuple_idx += 1
    return tuples, current_tuple


def chunked_cartesian_product(sizes: List[int], chunk_size: int):
    prod = np.prod(sizes)

    # putting the largest number at the front to more efficiently make use
    # of the cproduct numba function
    sizes = np.array(sizes, dtype=np.int32)
    sorted_idx = np.argsort(sizes)[::-1]
    sizes = sizes[sorted_idx]
    if chunk_size > prod:
        chunk_bounds = (np.array([0, prod])).astype(np.int64)
    else:
        num_chunks = np.maximum(np.ceil(prod / chunk_size), 2).astype(np.int32)
        chunk_bounds = (np.arange(num_chunks + 1) * chunk_size).astype(np.int64)
        chunk_bounds[-1] = prod
    current_tuple = np.zeros(len(sizes), dtype=np.int32)
    for start_idx, end_idx in zip(chunk_bounds[:-1], chunk_bounds[1:]):
        tuples, current_tuple = cproduct(sizes, current_tuple, start_idx, end_idx)
        # re-arrange columns to match the original order of the sizes list
        # before yielding
        yield tuples[:, np.argsort(sorted_idx)]


def cartesian_product(sizes: List[int]):
    return next(chunked_cartesian_product(sizes, np.prod(sizes)))


@nb.njit(nb.int32(nb.int32, nb.int32))
def binom(n, k):
    return fast_factorial(n) / (fast_factorial(k) * fast_factorial(n - k))


@nb.njit(nb.float32[:](nb.int32[:, :],
                       nb.types.ListType(nb.float32[:, ::1]),
                       nb.float32,
                       nb.float32[:, :],
                       nb.float32,
                       nb.float32))
def alignments_disorders(units_tuples_ids: np.ndarray,
                         units_data: List[np.ndarray],
                         delta_empty: float = 1.0,
                         cat_matrix: np.ndarray = None,
                         alpha: float = 3.0,
                         beta: float = 1.0):
    disorders = np.zeros(len(units_tuples_ids), dtype=np.float32)
    categories = [unit_data[:, 3].astype(np.int32) for unit_data in units_data]
    annot_id = np.arange(units_tuples_ids.shape[1]).astype(np.int32)
    for tuple_id in np.arange(len(units_tuples_ids)):
        #  for each tuple (corresponding to a unitary alignment), compute disorder
        for annot_a in annot_id:
            for annot_b in annot_id[annot_a + 1:]:
                # this block looks a bit slow (because of all the variables
                # declarations) but should be fairly sped up automatically
                # by the LLVM optimization pass
                unit_a_id, unit_b_id = units_tuples_ids[tuple_id, annot_a], units_tuples_ids[tuple_id, annot_b]
                unit_a, unit_b = units_data[annot_a][unit_a_id], units_data[annot_b][unit_b_id]
                cat_a, cat_b = categories[annot_a][unit_a_id], categories[annot_b][unit_b_id]
                if np.isnan(unit_a[0]) or np.isnan(unit_b[0]):
                    disorders[tuple_id] += delta_empty
                else:
                    ends_diff = np.abs(unit_a[1] - unit_b[1])
                    starts_diff = np.abs(unit_a[0] - unit_b[0])
                    # unit[2] is the duration
                    distance_pos = (starts_diff + ends_diff) / (unit_a[2] + unit_b[2])
                    distance_pos = distance_pos * distance_pos * delta_empty
                    distance_cat = cat_matrix[cat_a, cat_b] * delta_empty
                    disorders[tuple_id] += distance_pos * alpha + distance_cat * beta
        # print(disorders[tuple_id])

    disorders = disorders / binom(units_tuples_ids.shape[1], 2)

    return disorders


timer = Timer()
timer.start()
print("Loading csv")
annotations = defaultdict(SortedDict)
categories = set()
with open("DATA/AlexPaulSuzan.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        seg = pa.Segment(float(row[4]), float(row[5]))
        # discarding "empty" segments
        if seg.duration == 0.0:
            continue
        categories.add(row[2])
        annotations[row[1]][seg] = row[2]

DELTA_EMPTY = 1.0
# size of each unit tuple chunk (in number of cells)
CHUNK_SIZE = 2 ** 25
categories_dict = {cat: i for i, cat in enumerate(categories)}

cat_matrix = (np.ones((4,4)) - np.eye(4)).astype(np.float32)

timer.lap()
print("Converting csv data to arrays")

annotators_arrays: List[np.ndarray] = nb.typed.List()
annotator_ids: Dict[int, str] = dict()
unit_id = 0
for annotator_id, (annotator, segments) in enumerate(annotations.items()):
    # dim x : segment
    # dim y : (start, end, dur, cat)
    annotator_ids[annotator_id] = annotator
    annot_array = np.zeros((len(segments) + 1, 5)).astype(np.float32)
    for seg_id, (segment, cat) in enumerate(segments.items()):
        annot_array[seg_id][0] = segment.start
        annot_array[seg_id][1] = segment.end
        annot_array[seg_id][2] = segment.duration
        annot_array[seg_id][3] = categories_dict[cat]
        annot_array[seg_id][4] = unit_id
        unit_id += 1

    # # computing distance for each segment
    # annot_array[:,2] = annot_array[:, 1] - annot_array[:, 0]
    # annot_array = annot_array[np.where(annot_array[:,2] != 0.0)]
    annot_array[-1, :] = np.array([np.NaN for _ in range(5)])
    annotators_arrays.append(annot_array)

timer.lap()

nb_unit_per_annot = [len(arr) for arr in annotators_arrays]
number_tuples = np.product(nb_unit_per_annot)
print(f"Computing disorders for {number_tuples} alignments")
all_disorders = []
all_valid_tuples = []
num_chunks = (number_tuples // CHUNK_SIZE) + 1
for tuples_batch in tqdm(chunked_cartesian_product(nb_unit_per_annot, CHUNK_SIZE),
                         total=num_chunks):
    print(tuples_batch.shape)
    batch_disorders = alignments_disorders(tuples_batch,
                                     annotators_arrays,
                                     delta_empty=np.float32(DELTA_EMPTY),
                                     cat_matrix=cat_matrix,
                                     alpha=np.float32(1.0),
                                     beta=np.float32(1.0))

    valid_disorders_ids,  = np.where(batch_disorders < len(annotators_arrays) * DELTA_EMPTY)
    all_disorders.append(batch_disorders[valid_disorders_ids])
    all_valid_tuples.append(tuples_batch[valid_disorders_ids])

disorders = np.concatenate(all_disorders)
possible_unitary_alignments = np.concatenate(all_valid_tuples)

timer.lap()
print("Computing min disorder using the linear solver")
print(f"There are  {len(possible_unitary_alignments)} possible alignments")

print(disorders.mean())

# Definition of the integer linear program
num_possible_unitary_alignements = len(disorders)
x = cp.Variable(shape=num_possible_unitary_alignements, boolean=True)

num_units = sum([len(arr) - 1 for arr in annotators_arrays])
# Constraints matrix
A = np.zeros((num_units, num_possible_unitary_alignements))
print(A.shape)


for p_id, unit_ids_tuple in enumerate(possible_unitary_alignments):
    for annot_id, unit_id in enumerate(unit_ids_tuple):
        true_unit_id = annotators_arrays[annot_id][unit_id, 4]
        if not np.isnan(true_unit_id):
            A[int(true_unit_id), p_id] = 1

obj = cp.Minimize(disorders.T * x)
constraints = [cp.matmul(A, x) == 1]
prob = cp.Problem(obj, constraints)

optimal_disorder = prob.solve()
print(f"Optimal disorder is {optimal_disorder}")
# set_unitary_alignements = []
timer.lap()
print("Building best aligment")

# compare with 0.9 as cvxpy returns 1.000 or small values i.e. 10e-14
#print(np.where(x.value > 0.9))
chosen_alignments = possible_unitary_alignments[np.where(x.value > 0.9)[0]]
print("disorder:", disorders[np.where(x.value > 0.9)].sum() / len(chosen_alignments))
alignment = Alignment()
for idx, chosen_unitary_alignement in enumerate(chosen_alignments):
    u_align_tuple = tuple(("annotator_a", Unit(pa.Segment(0, 1), "C"))
                          for _ in chosen_unitary_alignement)
    alignment.unitary_alignments.append(UnitaryAlignment(u_align_tuple))

timer.lap()
timer.total()
