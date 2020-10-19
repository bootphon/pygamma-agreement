import logging
from pathlib import Path
import numpy as np
import time

from pyannote.core import Annotation, Segment
from pygamma import Continuum, CombinedCategoricalDissimilarity, UnitaryAlignment

continuum = Continuum()
annotation = Annotation()
annotation[Segment(1, 5)] = 'Carol'
annotation[Segment(6, 8)] = 'Bob'
annotation[Segment(12, 18)] = 'Carol'
annotation[Segment(7, 20)] = 'Alice'
continuum.add_annotation('liza', annotation)
annotation = Annotation()
annotation[Segment(2, 6)] = 'Carol'
annotation[Segment(7, 8)] = 'Bob'
annotation[Segment(12, 18)] = 'Alice'
annotation[Segment(8, 10)] = 'Alice'
annotation[Segment(7, 19)] = 'Jeremy'
continuum.add_annotation('pierrot', annotation)
categories = ['Carol', 'Bob', 'Alice', 'Jeremy']

cat = np.array([[0, 0.5, 0.3, 0.7],
                [0.5, 0., 0.6, 0.4],
                [0.3, 0.6, 0., 0.7],
                [0.7, 0.4, 0.7, 0.]])
combi_dis = CombinedCategoricalDissimilarity(
    list_categories=categories,
    delta_empty=0.5,
    categorical_dissimilarity_matrix=cat,
    alpha=3, beta=1)
list_dis = []
for liza_unit in continuum['liza'].values():
    for pierrot_unit in continuum['pierrot'].values():
        unit_alignment = UnitaryAlignment((("liza", liza_unit),
                                           ("pierrot", pierrot_unit)))
        list_dis.append(unit_alignment.compute_disorder(combi_dis))
print(list_dis)