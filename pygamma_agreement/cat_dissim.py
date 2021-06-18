#!usr/bin/env python
"""
(str, str) -> float default functions to use in a categorical dissimilarity between units.
may give results > 1 but will be normalized.
"""

arguments = {}

try:
    import Levenshtein

    def cat_levenshtein(cat1: str, cat2: str) -> float:
        return Levenshtein.distance(cat1, cat2) / max(len(cat1), len(cat2))

    arguments["levenshtein"] = cat_levenshtein
except ImportError:
    pass


def cat_default(cat1: str, cat2: str) -> float:
    return float(cat1 != cat2)


arguments["default"] = cat_default


def cat_ord(cat1: str, cat2: str) -> float:
    if not (cat1.isnumeric() and cat2.isnumeric()):
        raise ValueError("Error : tried to compute ordinal categorical dissimilarity"
                         f"but categories are non-numeric (category {cat1} or {cat2})")
    return abs(float(int(cat1) - int(cat2)))


arguments["ordinal"] = cat_ord
