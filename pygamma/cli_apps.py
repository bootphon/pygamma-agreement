import argparse
import csv
from pathlib import Path

from .gamma import compute_gamma


def pygamma_cmd():
    # TODO : detail row structure in help
    # TODO : talk about the best way for sequence dissim to be implemented (input-wise)
    # TODO : figure out hte best way to save or display gamma output for users
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_csv", type=Path,
                           help="Path to an input csv file")
    argparser.add_argument("-d", "--dissimilarity",
                           choices=["categorical", "sequence"],
                           help="Type of dissimilarity that is to be used")

    args = argparser.parse_args()
    units_tuples = []
    with open(args.input_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            units_tuples.append((row[0], row[1], float(row[2]), float[row[3]]))

    compute_gamma(units_tuples, args.dissimilarity)

