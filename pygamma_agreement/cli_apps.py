# The MIT License (MIT)

# Copyright (c) 2020-2021 CoML

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
# Rachid RIAD, Hadrien TITEUX, Léopold FAVRE


import argparse
import csv
import logging
import os
import time
import json
import numpy as np
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Dict, List

from pygamma_agreement import (Continuum,
                               LevenshteinCategoricalDissimilarity,
                               NumericalCategoricalDissimilarity,
                               ShuffleContinuumSampler,
                               CombinedCategoricalDissimilarity)

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
argparser.add_argument("-s", "--separator",
                       default=",", type=str,
                       help="Column delimiter used for input and output csv")
argparser.add_argument("--seed",
                       default=None, type=int,
                       help="random seed")
argparser.add_argument("-f", "--format", type=str, choices=["rttm", "csv"],
                       default="csv",
                       help="Format of the input file")

output = argparser.add_mutually_exclusive_group()
output.add_argument("-o", "--output-csv", type=Path,
                    help="Path to the output csv report")
output.add_argument("-j", "--output-json", type=Path,
                    help="Path to the output json report")

argparser.add_argument("-e", "--empty-delta",
                       default=1, type=float,
                       help="Delta empty coefficient (empty alignment tolerance)")
argparser.add_argument("-a", "--alpha",
                       default=1, type=float,
                       help="Alpha coefficient (positional dissimilarity ponderation)")
argparser.add_argument("-b", "--beta",
                       default=1, type=float,
                       help="Beta coefficient (categorical dissimilarity ponderation)")
argparser.add_argument("-p", "--precision-level",
                       default=0.05, type=float,
                       help="Precision level used for the gamma computation. \n"
                            "This is a percentage, lower means more precision. \n"
                            "A value under 0.10 is advised.")
argparser.add_argument("-n", "--n-samples",
                       default=30, type=int,
                       help="Number of random continuua to be sampled for the \n"
                            "gamma computation. Warning : additionnal continuua \n"
                            "will be sampled if precision level is not satisfied.\n")
argparser.add_argument("-d", "--cat-dissim", type=str, choices={"absolute", "numerical", "levenshtein"},
                       default="absolute",
                       help="Categorical dissimilarity to use for measuring \n"
                            "inter-annotation disorder. The default one gives 1.0 \n"
                            "if annotation have different categories, 0.0 otherwise")
argparser.add_argument("-v", "--verbose",
                       action="store_true",
                       help="Logs progress of the algorithm")
argparser.add_argument("-c", "--gamma-cat",
                       action="store_true",
                       help="Outputs the gamma-cat in addition to the gamma-agreement")
argparser.add_argument("-k", "--gamma-k",
                       action="store_true",
                       help="Outputs the gamma-k's every inputs' categories")
argparser.add_argument("-m", "--mathet-sampler",
                       action="store_true",
                       help="Set the expected dissimilarity sampler to the one \n"
                            "chosen by Mathet et Al.")


def pygamma_cmd():
    args = argparser.parse_args()
    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.ERROR)

    input_files: List[Path] = []
    results: List = []
    for input_csv in args.input_csv:
        input_csv: Path

        if input_csv.is_dir():
            logging.info(f"Loadinmodeg csv files in folder {input_csv}")
            for file_path in input_csv.iterdir():
                input_files.append(file_path)

        elif input_csv.is_file():
            logging.info(f"Loading CSV file {input_csv}")
            input_files.append(input_csv)
        else:
            logging.error(f"Input CSV file or folder '{input_csv}' couldn't be found")

    logging.info(f"Found {len(input_files)} csv files.")

    json_dict = {}
    if args.seed is not None:
        np.random.seed(args.seed)

    for file_path in input_files:
        start = time.time()
        if args.format == "csv":
            continuum = Continuum.from_csv(file_path, delimiter=args.separator)
        else:
            continuum = Continuum.from_rttm(file_path)
        logging.info(f"Finished loading continuum from {os.path.basename(file_path)} in {(time.time() - start) * 1000} ms")
        start = time.time()

        cat_dissim = None
        if args.cat_dissim == "levenshtein":
            cat_dissim = LevenshteinCategoricalDissimilarity(continuum.categories)
        elif args.cat_dissim == "ordinal":
            cat_dissim = NumericalCategoricalDissimilarity(continuum.categories)

        dissim = CombinedCategoricalDissimilarity(alpha=args.alpha,
                                                  beta=args.beta,
                                                  delta_empty=args.empty_delta,
                                                  cat_dissim=cat_dissim)
        logging.info(f"Finished loading dissimilarity object in {(time.time() - start) * 1000} ms")
        start = time.time()

        sampler = None
        if args.mathet_sampler:
            sampler = ShuffleContinuumSampler()

        gamma = continuum.compute_gamma(dissimilarity=dissim,
                                        precision_level=args.precision_level,
                                        fast=True,
                                        sampler=sampler,
                                        n_samples=args.n_samples)
        logging.info(f"Finished computing best alignment & gamma in {(time.time() - start) * 1000} ms")
        # start = time.time()

        result_list = [file_path]
        if args.output_csv is None and args.output_json is None:
            print(f"{file_path}")
            print(f"gamma={gamma.gamma}")
            if args.gamma_cat:
                print(f"gamma-cat={gamma.gamma_cat}")
            if args.gamma_k:
                for category in continuum.categories:
                    print(f"gamma-k('{category}')={gamma.gamma_k(category)}")
        else:
            result_list.append(gamma.gamma)
            if args.gamma_cat:
                result_list.append(gamma.gamma_cat)
            if args.gamma_k:
                result_list.append({category: gamma.gamma_k(category) for category in continuum.categories})
        results.append(result_list)

    labels = ['filename', 'gamma']
    if args.gamma_cat:
        labels.append('gamma-cat')
    if args.gamma_k:
        labels.append("gamma-k")
    if args.output_csv is not None:
        with open(args.output_csv, "w") as output_csv:
            writer = csv.writer(output_csv, delimiter=args.separator)
            writer.writerow(labels)
            writer.writerows(results)
    elif args.output_json is not None:
        for result in results:
            json_dict[str(result[0])] = {label: result for (label, result) in zip(labels[1:], result[1:])}
        with open(args.output_json, "w") as output_json:
            json.dump(json_dict, output_json, indent=4)


