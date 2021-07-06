#!usr/bin/env python

import csv
from pygamma_agreement.continuum import Continuum
from pygamma_agreement.dissimilarity import CombinedCategoricalDissimilarity
from colorama import Fore
from time import time
from datetime import date
from subprocess import check_output

try:
    with open("benchmark/records.csv", 'r') as f:
        records = list(csv.reader(f))
except FileNotFoundError:
    records = []

# Line:
# Time, Date, Branch, gamma, gamma-cat

start = time()
continuum = Continuum.from_csv("tests/data/3by100.csv")
dissim = CombinedCategoricalDissimilarity(continuum.categories, alpha=3, beta=2)
gamma_results = continuum.compute_gamma(precision_level=0.05)
gamma = gamma_results.gamma
gamma_cat = gamma_results.gamma_cat
score = time() - start

if len(records) == 0 or score <= float(records[-1][0])*0.99:  # record is only registered if lowered by 1%
    print(Fore.GREEN + f"new record : {score} seconds !")
    if len(records) > 0:
        print(Fore.GREEN + f"(former record was {records[-1][0]} seconds).")
    branch = check_output("git branch --show-current", shell=True).decode('UTF-8').rstrip('\n')
    records.append([score, date.today().strftime("%d/%m/%Y"), branch, gamma, gamma_cat])
    with open("benchmark/records.csv", 'w+') as f:
        csv.writer(f).writerows(records)

else:
    print(Fore.RED + f"No new record.")
