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
from pathlib import Path
from typing import Dict

from pygamma import Continuum

argparser = argparse.ArgumentParser(
    description="A command-line tool to compute the gamma-agreement for "
                "CSV or RTTM files.")
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
argparser.add_argument("-v", "--verbose",
                       action="store_true",
                       help="Verbose mode")


def pygamma_cmd():
    # TODO : detail row structure in help
    # TODO : support for folder-wise gamma computation
    # TODO: add support for positional only dissim and categorical only
    args = argparser.parse_args()
    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.ERROR)

    results: Dict[Path, float] = {}

    for input_csv in args.input_csv:
        input_csv: Path

        if input_csv.is_dir():
            logging.info(f"Loading csv files in folder {input_csv}")
            for csv_path in input_csv.iterdir():
                continuum = Continuum.from_csv(csv_path)
                gamma = continuum.compute_gamma()
                results[csv_path] = gamma
                print(f"{csv_path} : {gamma}")

        elif input_csv.is_file():
            logging.info(f"Loading CSV file {input_csv}")
            continuum = Continuum.from_csv(path=input_csv)
            gamma = continuum.compute_gamma()
            results[input_csv] = gamma
            print(gamma)
        else:
            logging.error("Input CSV file or folder couldn't be found")

    if args.output_csv is not None:
        with open(args.output_csv, "w") as output_csv:
            writer = csv.writer(output_csv, delimiter=args.delimiter)
            writer.writerows(list(results.items()))
