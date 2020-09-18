import argparse
import csv
import logging
from pathlib import Path
from typing import Dict

from pygamma import Continuum

argparser = argparse.ArgumentParser()
argparser.add_argument("input_csv", type=Path,
                       help="Path to an input csv file or directory")
argparser.add_argument("-d", "--delimiter",
                       default=",", type=str,
                       help="Column delimiter used for input and output csv")
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
    input_csv: Path = args.input_csv
    results: Dict[Path, float] = {}

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
