import csv
from collections import defaultdict

import numpy as np
from pyannote.core import Segment, Annotation
from tqdm import tqdm

from pygamma.continuum import Continuum
from pygamma.dissimilarity import CombinedCategoricalDissimilarity
from pygamma.alignment import Alignment
import time

annotations = defaultdict(Annotation)
categories = set()

with open("DATA/2by1000.txt") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        seg = Segment(float(row[4]), float(row[5]))
        # discarding "empty" segments
        if seg.duration == 0.0:
            continue
        categories.add(row[2])
        annotations[row[1]][seg] = row[2]

continuum = Continuum()
for annotator, annotation in annotations.items():
    continuum[annotator] = annotation

cat = np.array([[0, 0.5, 0.3, 0.7], [0.5, 0., 0.6, 0.4],
                [0.3, 0.6, 0., 0.7], [0.7, 0.4, 0.7, 0.]])

combi_dis = CombinedCategoricalDissimilarity(
    'diarization',
    categories=list(categories),
    DELTA_EMPTY=0.5,
    cat_dissimilarity_matrix=cat)

start_t = time.time()
best_alignement = Alignment.get_best_alignment(continuum, combi_dis)
end_t = time.time()
print(best_alignement.disorder)
print(f"Took {end_t - start_t}s")
