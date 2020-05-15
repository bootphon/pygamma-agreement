import argparse
import logging
from pathlib import Path
import csv
from collections import defaultdict
from pyannote.core import Annotation, Segment

from .alignment import Alignment
from .continuum import Continuum
from .dissimilarity import SequenceDissimilarity, CategoricalDissimilarity
from .gamma import GammaAgreement

def pygamma_cmd():
    # TODO : define row structure in help
    # TODO : talk about the best way for sequence dissim to be implemented (input-wise)
    # TODO : figure out hte best way to save or display gamma output for users
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_csv", type=Path,
                           help="Path to an input csv file")
    argparser.add_argument("-d", "--dissimilarity",
                           choices=["categorical", "sequence"],
                           help="Type of dissimilarity that is to be used")

    args = argparser.parse_args()
    annotations = defaultdict(Annotation)
    categories = set()
    symbols = set()
    with open(args.input_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if args.dissimilarity == "categorical":
                categories.add(row[1])
            else:
                symbols.update(set(row[1]))
            annotations[row[0]][Segment(int(row[2]), int(row[3]))] = row[1]

    continuum = Continuum()
    for annotator, annotation in annotations.items():
        continuum[annotator] = annotation

    if args.dissimilarity == "categorical":
        dissim = CategoricalDissimilarity("", list(categories))
    else:
        dissim = SequenceDissimilarity("", list(symbols))

    logging.info("Computing best alignment...")
    best_alignment = Alignment.get_best_alignment(continuum, dissim)
    gamma = GammaAgreement(continuum, best_alignment, dissim)
    logging.info("Computing Gamma Agreement for that alignment...")
    gamma.get_gamma()


