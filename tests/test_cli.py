"""Tests for the command line interface"""

import ast
import os
import re
import json
import tempfile as tf
import csv
from typing import Dict


def test_cli_print():
    pygamma_cmd = "python -c'from pygamma_agreement.cli_apps import pygamma_cmd;pygamma_cmd()'"
    input_file = "tests/data/AlexPaulSuzan.csv"
    with tf.NamedTemporaryFile('w+', delete=True) as f:
        assert os.system(f"{pygamma_cmd} {input_file} {input_file} --seed 4772"
                         f" -k -c --alpha 3 -m -p 0.05 > {f.name}") == 0
        lines = filter(lambda line: line != "Long-step dual simplex will be used",  # Necessity bc of GLPK bug
                       f.read().splitlines())
        for _ in range(2):
            filename = next(lines)
            assert filename == input_file
            gamma = next(lines).split(sep='=')
            assert gamma[0] == 'gamma'
            assert 0.43 <= float(gamma[1]) <= 0.47
            gamma_cat = next(lines).split('=')
            assert gamma_cat[0] == 'gamma-cat'
            assert 0.66 <= float(gamma_cat[1]) <= 0.70
            for category, gk in {'1': 1, '2': 0, '3': 0, '4': 0, '5': 1, '6': 1, '7': 0}.items():
                gamma_k = re.split("\\('|'\\)=", next(lines))
                assert gamma_k[0] == 'gamma-k'
                assert gamma_k[1] == category
                if gk != 0:
                    assert float(gamma_k[2]) - 0.2 <= gk <= float(gamma_k[2]) + 0.2
                else:
                    assert gk <= 0


def test_cli_csv():
    pygamma_cmd = "python -c'from pygamma_agreement.cli_apps import pygamma_cmd;pygamma_cmd()'"
    input_file = "tests/data/AlexPaulSuzan.csv"
    with tf.NamedTemporaryFile('w+', delete=True) as f:
        assert os.system(f"{pygamma_cmd} {input_file} {input_file} --seed 4772 "
                         f"-k -c --alpha 3 -m -p 0.05 -o {f.name}") == 0
        reader = iter(row for row in csv.reader(f))
        assert next(reader) == ['filename', 'gamma', 'gamma-cat', "gamma-k"]
        for _ in range(2):
            values = next(reader)
            filename, gamma, gamma_cat, gamma_k = values
            gamma, gamma_cat = float(gamma), float(gamma_cat)
            gamma_k = ast.literal_eval(gamma_k)

            assert 0.43 <= gamma <= 0.47
            assert 0.66 <= gamma_cat <= 0.70
            for category, gk in {'1': 1, '2': 0, '3': 0, '4': 0, '5': 1, '6': 1, '7': 0}.items():
                if gk != 0:
                    assert gk - 0.2 <= gamma_k[category] <= gk + 0.2
                else:
                    assert gk <= 0


def test_cli_json():
    pygamma_cmd = "python -c'from pygamma_agreement.cli_apps import pygamma_cmd;pygamma_cmd()'"
    input_file = "tests/data/AlexPaulSuzan.csv"
    with tf.NamedTemporaryFile('w+', delete=True) as f:
        assert os.system(f"{pygamma_cmd} {input_file} {input_file}  --seed 4772"
                         f" -k -c --alpha 3 -m -p 0.05 -j {f.name}") == 0
        json_data: Dict = json.load(f)
        assert len(json_data) == 1
        for file, data in json_data.items():
            assert file == input_file
            assert 0.43 <= data['gamma'] <= 0.47
            assert 0.66 <= data['gamma-cat'] <= 0.70
            for category, gk in {'1': 1, '2': 0, '3': 0, '4': 0, '5': 1, '6': 1, '7': 0}.items():
                if gk != 0:
                    assert gk - 0.2 <= data['gamma-k'][category] <= gk + 0.2
                else:
                    assert gk <= 0







