#!usr/bin/env python
"""
(str, str) -> float default functions to use in a categorical dissimilarity between units.
may give results > 1 but will be normalized.
"""
from Levenshtein import distance as lev

arguments = {}


def cat_levenshtein(cat1: str, cat2: str):
    return lev(cat1, cat2)


def cat_default(cat1: str, cat2: str):
    return cat1 != cat2


def cat_ord(cat1: str, cat2: str):
    if not (cat1.isnumeric() and cat2.isnumeric()):
        raise ValueError("Error : tried to compute ordinal categorical dissimilarity"
                         f"but categories are non-numeric (category {cat1} or {cat2})")
    return abs(int(cat1) - int(cat2))


arguments["default"] = cat_default
arguments["levenshtein"] = cat_levenshtein
arguments["ordinal"] = cat_ord
