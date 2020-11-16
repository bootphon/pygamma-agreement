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


import argparse
import csv
import logging
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Dict, List

from pygamma_agreement import Continuum, CombinedCategoricalDissimilarity


class RawAndDefaultArgumentFormatter(RawTextHelpFormatter,
                                     ArgumentDefaultsHelpFormatter):
    pass


argparser = argparse.ArgumentParser(
    formatter_class=RawAndDefaultArgumentFormatter,
    description="""
    A command-line tool to compute the gamma-agreement for 
    CSV or RTTM files. It can compute the gamma for one or more CSV files.
    The expected format of input files is a 4-columns CSV, 
    with the following structure: 
    
    annotator_id, annotation, segment_start, segment_end
    
    E.G.:
    
    annotator_1, Marvin, 11.3, 15.6
    annotator_1, Maureen, 20, 25.7
    annotator_2, Marvin, 10, 26.3
    ...
    
    It can also read RTTM files if you set the --format option to "rttm".
    
    """)
argparser.add_argument("input_csv", type=Path, nargs="+",
                       help="Path to an input csv file(s) or directory(es)")
argparser.add_argument("-d", "--delimiter",
                       default=",", type=str,
                       help="Column delimiter used for input and output csv")
argparser.add_argument("-f", "--format", type=str, choices=["rttm", "csv"],
                       default="csv",
                       help="Path to the output csv report")
argparser.add_argument("-o", "--output_csv", type=Path,
                       help="Path to the output csv report")
argparser.add_argument("-a", "--alpha",
                       default=2, type=float,
                       help="Alpha coefficient (positional dissimilarity ponderation)")
argparser.add_argument("-b", "--beta",
                       default=1, type=float,
                       help="Beta coefficient (categorical dissimilarity ponderation)")
argparser.add_argument("-p", "--precision_level",
                       default=0.05, type=float,
                       help="Precision level used for the gamma computation. "
                            "This is a percentage, lower means more precision. "
                            "A value under 0.10 is advised.")
argparser.add_argument("-n", "--n_samples",
                       default=30, type=int,
                       help="Number of random continuua to be sampled for the "
                            "gamma computation.")
argparser.add_argument("-v", "--verbose",
                       action="store_true",
                       help="Verbose mode")


def pygamma_cmd():
    args = argparser.parse_args()
    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.ERROR)

    input_files: List[Path] = []
    results: Dict[Path, float] = {}
    for input_csv in args.input_csv:
        input_csv: Path

        if input_csv.is_dir():
            logging.info(f"Loading csv files in folder {input_csv}")
            for file_path in input_csv.iterdir():
                input_files.append(file_path)

        elif input_csv.is_file():
            logging.info(f"Loading CSV file {input_csv}")
            input_files.append(input_csv)
        else:
            logging.error(f"Input CSV file or folder '{input_csv}' couldn't be found")

    logging.info(f"Found {len(input_files)} csv files.")

    for file_path in input_files:
        if args.format == "csv":
            continuum = Continuum.from_csv(file_path, delimiter=args.delimiter)
        else:
            continuum = Continuum.from_rttm(file_path)

        dissim = CombinedCategoricalDissimilarity(continuum.categories,
                                                  alpha=args.alpha,
                                                  beta=args.beta)
        gamma = continuum.compute_gamma(dissimilarity=dissim,
                                        precision_level=args.precision_level,
                                        n_samples=args.n_samples)
        results[file_path] = gamma.gamma
        print(f"{file_path} : {gamma.gamma}")

    if args.output_csv is not None:
        with open(args.output_csv, "w") as output_csv:
            writer = csv.writer(output_csv, delimiter=args.delimiter)
            writer.writerows(list(results.items()))
