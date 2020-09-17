import argparse
import csv
from pathlib import Path

from .gamma import compute_gamma


def pygamma_cmd():
    # TODO : detail row structure in help
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_csv", type=Path,
                           help="Path to an input csv file")
    argparser.add_argument("-a", "--alpha",
                           default=2, type=float,
                           help="Alpha coefficient (positional dissimilarity ponderation)")
    argparser.add_argument("-b", "--beta",
                           default=1, type=float,
                           help="Beta coefficient (categorical dissimilarity ponderation)")

    args = argparser.parse_args()
    units_tuples = []
    with open(args.input_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            units_tuples.append((row[0], row[1], float(row[2]), float[row[3]]))

    compute_gamma(units_tuples, args.dissimilarity)

