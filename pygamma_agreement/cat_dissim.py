#!usr/bin/env python
"""
(str, str) -> float default functions to use in a categorical dissimilarity between units.
may give results > 1 but will be normalized.
"""
# TODO ajouter aux dépendances
from Levenshtein import distance as lev

dict = {}

def cat_levenshtein(cat1: str, cat2: str):
    return lev(cat1, cat2)
dict["levenshtein"] = cat_levenshtein

def cat_default(cat1: str, cat2: str):
    return cat1 != cat2
dict["default"] = cat_default

ord_val_max = 1
def cat_ord(cat1: str, cat2: str):
    # TODO : entre 0 et 1
    assert cat1.isnumeric() and cat2.isnumeric()
    return abs(int(cat1) - int(cat1))
dict["ordinal"] = cat_ord