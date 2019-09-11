#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2019 CNRS

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
# Rachid RIAD
"""
##########
Dissimilarity
##########

"""

import numpy as np
from matplotlib import pyplot as plt

from similarity.weighted_levenshtein import WeightedLevenshtein
from similarity.weighted_levenshtein import CharacterSubstitutionInterface
from similarity.weighted_levenshtein import CharacterInsDelInterface

from functools import lru_cache


class Categorical_Dissimilarity(object):
    """Categorical Dissimilarity
    Parameters
    ----------
    annotation_task :
        task to be annotated
    list_categories :
        list of categories
    categorical_dissimilarity_matrix :
        Dissimilarity matrix to compute
    DELTA_EMPTY :
        empty dissimilarity value
    function_cat :
        Function to adjust dissimilarity based on categorical matrix values
    Returns
    -------
    dissimilarity : Dissimilarity
        Dissimilarity
    """

    def __init__(self,
                 annotation_task,
                 list_categories,
                 categorical_dissimilarity_matrix=None,
                 DELTA_EMPTY=1,
                 function_cat=lambda x: x):

        super(Categorical_Dissimilarity, self).__init__()
        self.annotation_task = annotation_task
        self.list_categories = list_categories
        assert len(list_categories) == len(set(list_categories))
        self.num_categories = len(self.list_categories)
        self.dict_list_categories = dictionary = dict(
            zip(self.list_categories, list(range(self.num_categories))))
        cat_dissimilarity_matrix = categorical_dissimilarity_matrix
        self.categorical_dissimilarity_matrix = cat_dissimilarity_matrix
        if self.categorical_dissimilarity_matrix is None:
            self.categorical_dissimilarity_matrix = np.ones(
                (self.num_categories, self.num_categories)) - np.eye(
                    self.num_categories)
        else:
            assert type(self.categorical_dissimilarity_matrix) == np.ndarray
            assert np.all(self.categorical_dissimilarity_matrix <= 1)
            assert np.all(0 <= self.categorical_dissimilarity_matrix)
            assert np.all(self.categorical_dissimilarity_matrix ==
                          categorical_dissimilarity_matrix.T)
        self.function_cat = function_cat
        self.DELTA_EMPTY = DELTA_EMPTY

    def plot_categorical_dissimilarity_matrix(self):
        fig, ax = plt.subplots()
        im = plt.imshow(
            self.categorical_dissimilarity_matrix,
            extent=[0, self.num_categories, 0, self.num_categories])
        ax.figure.colorbar(im, ax=ax)
        plt.xticks([el + 0.5 for el in range(self.num_categories)],
                   self.list_categories)
        plt.yticks([el + 0.5 for el in range(self.num_categories)],
                   self.list_categories[::-1])
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor")
        ax.xaxis.set_ticks_position('top')
        plt.show()

    @lru_cache(maxsize=None)
    def __getitem__(self, units):
        # assert type(units) == list
        if len(units) < 2:
            return self.DELTA_EMPTY
        else:
            assert units[0] in self.list_categories
            assert units[1] in self.list_categories
            cat_dis = self.categorical_dissimilarity_matrix[
                self.dict_list_categories[units[0]]][self.dict_list_categories[
                    units[1]]]
            return self.function_cat(cat_dis) * self.DELTA_EMPTY


class Sequence_Dissimilarity(object):
    """Sequence Dissimilarity
    Parameters
    ----------
    annotation_task :
        task to be annotated
    list_admitted_symbols :
        list of admitted symbols in the sequence
    categorical_symbol_dissimlarity_matrix :
        Dissimilarity matrix to compute between symbols
    DELTA_EMPTY :
        empty dissimilarity value
    function_cat :
        Function to adjust dissimilarity based on categorical matrix values
    Returns
    -------
    dissimilarity : Dissimilarity
        Dissimilarity
    """

    def __init__(self,
                 annotation_task,
                 list_admitted_symbols,
                 symbol_dissimlarity_matrix=None,
                 DELTA_EMPTY=1,
                 function_cat=lambda x: x):

        super(Sequence_Dissimilarity, self).__init__()
        self.annotation_task = annotation_task
        self.list_admitted_symbols = list_admitted_symbols
        assert len(list_admitted_symbols) == len(set(list_admitted_symbols))
        self.num_symbols = len(self.list_admitted_symbols)
        self.dict_list_symbols = dictionary = dict(
            zip(self.list_admitted_symbols, list(range(self.num_symbols))))
        self.symbol_dissimlarity_matrix = symbol_dissimlarity_matrix
        if self.symbol_dissimlarity_matrix is None:
            self.symbol_dissimlarity_matrix = np.ones(
                (self.num_symbols, self.num_symbols)) - np.eye(
                    self.num_symbols)
        else:
            assert type(self.symbol_dissimlarity_matrix) == np.ndarray
            assert np.all(self.symbol_dissimlarity_matrix <= 1)
            assert np.all(0 <= self.symbol_dissimlarity_matrix)
            assert np.all(self.symbol_dissimlarity_matrix ==
                          symbol_dissimlarity_matrix.T)
        self.function_cat = function_cat
        self.DELTA_EMPTY = DELTA_EMPTY

    def plot_symbol_dissimilarity_matrix(self):
        fig, ax = plt.subplots()
        im = plt.imshow(
            self.symbol_dissimlarity_matrix,
            extent=[0, self.num_symbols, 0, self.num_symbols])
        ax.figure.colorbar(im, ax=ax)
        plt.xticks([el + 0.5 for el in range(self.num_symbols)],
                   self.list_admitted_symbols)
        plt.yticks([el + 0.5 for el in range(self.num_symbols)],
                   self.list_admitted_symbols[::-1])
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor")
        ax.xaxis.set_ticks_position('top')
        plt.show()

    @lru_cache(maxsize=None)
    def __getitem__(self, units):
        class CharacterSubstitution(CharacterSubstitutionInterface):
            def __init__(self, list_admitted_symbols, dict_list_symbols,
                         symbol_dissimlarity_matrix):

                super(CharacterSubstitution, self).__init__()
                self.list_admitted_symbols = list_admitted_symbols
                self.dict_list_symbols = dict_list_symbols
                self.symbol_dissimlarity_matrix = symbol_dissimlarity_matrix

            def cost(self, c0, c1):
                return self.symbol_dissimlarity_matrix[self.dict_list_symbols[
                    c0]][self.dict_list_symbols[c1]]

        weighted_levenshtein = WeightedLevenshtein(
            CharacterSubstitution(self.list_admitted_symbols,
                                  self.dict_list_symbols,
                                  self.symbol_dissimlarity_matrix))

        # assert type(units) == list
        if len(units) < 2:
            return self.DELTA_EMPTY
        else:
            for symbol in units[0]:
                assert symbol in self.list_admitted_symbols
            for symbol in units[1]:
                assert symbol in self.list_admitted_symbols
            return self.function_cat(
                weighted_levenshtein.distance(units[0], units[1]) / max(
                    len(units[0]), len(units[1]))) * self.DELTA_EMPTY


class Positional_Dissimilarity(object):
    """Positional Dissimilarity
    Parameters
    ----------
    annotation_task :
        task to be annotated
    DELTA_EMPTY :
        empty dissimilarity value
    function_distance :
        position function difference
    Returns
    -------
    dissimilarity : Dissimilarity
        Dissimilarity
    """

    def __init__(self, annotation_task, DELTA_EMPTY=1, function_distance=None):

        super(Positional_Dissimilarity, self).__init__()
        self.annotation_task = annotation_task
        self.DELTA_EMPTY = DELTA_EMPTY
        self.function_distance = function_distance

    @lru_cache(maxsize=None)
    def __getitem__(self, units):
        # assert type(units) == list
        if len(units) < 2:
            return self.DELTA_EMPTY
        else:
            if self.function_distance is None:
                # triple indexing to tracks in pyannote
                # DANGER if the api breaks
                distance_pos = (np.abs(units[0][0] - units[1][0]) +
                                np.abs(units[0][1] - units[1][1]))
                distance_pos /= ((
                    units[0][1] - units[0][0] + units[1][1] - units[1][0]))
                distance_pos = distance_pos * distance_pos * self.DELTA_EMPTY
                return distance_pos


class Combined_Categorical_Dissimilarity(object):
    """Combined Dissimilarity
    Parameters
    ----------
    annotation_task :
        task to be annotated
    list_categories :
        list of categories
    categorical_dissimilarity_matrix :
        Dissimilarity matrix to compute
    function_cat :
        Function to adjust dissimilarity based on categorical matrix values
    DELTA_EMPTY :
        empty dissimilarity value
    f_dis :
        position function difference
    Returns
    -------
    dissimilarity : Dissimilarity
        Dissimilarity
    """

    def __init__(self,
                 annotation_task,
                 list_categories,
                 alpha=3,
                 beta=1,
                 DELTA_EMPTY=1,
                 function_distance=None,
                 categorical_dissimilarity_matrix=None,
                 function_cat=lambda x: x):

        super(Combined_Categorical_Dissimilarity, self).__init__()
        self.annotation_task = annotation_task
        self.function_distance = function_distance
        self.annotation_task = annotation_task
        self.list_categories = list_categories
        self.num_categories = len(self.list_categories)

        self.alpha = alpha
        self.beta = beta

        self.dict_list_categories = dictionary = dict(
            zip(self.list_categories, list(range(self.num_categories))))
        cat_dissimilarity_matrix = categorical_dissimilarity_matrix
        self.categorical_dissimilarity_matrix = cat_dissimilarity_matrix

        self.function_cat = function_cat
        self.function_distance = function_distance
        self.DELTA_EMPTY = DELTA_EMPTY

        self.positional_dissimilarity = Positional_Dissimilarity(
            annotation_task=annotation_task,
            DELTA_EMPTY=DELTA_EMPTY,
            function_distance=function_distance)

        self.categorical_dissimlarity = Categorical_Dissimilarity(
            annotation_task=annotation_task,
            list_categories=list_categories,
            categorical_dissimilarity_matrix=categorical_dissimilarity_matrix,
            DELTA_EMPTY=DELTA_EMPTY,
            function_cat=function_cat)

    @lru_cache(maxsize=None)
    def __getitem__(self, units):
        timing_units, categorical_units = units
        assert len(timing_units) == len(categorical_units)
        if len(timing_units) < 2:
            return self.DELTA_EMPTY
        else:
            dis = self.alpha * self.positional_dissimilarity[timing_units]
            dis += self.beta * self.categorical_dissimlarity[categorical_units]
            return dis


class Combined_Sequence_Dissimilarity(object):
    """Combined Sequence Dissimilarity
    Parameters
    ----------
    annotation_task :
        task to be annotated
    list_admitted_symbols :
        list of admitted symbols in the sequence
    symbol_dissimlarity_matrix :
        Dissimilarity matrix to compute between symbols
    function_cat :
        Function to adjust dissimilarity based on categorical matrix values
    DELTA_EMPTY :
        empty dissimilarity value
    f_dis :
        position function difference
    Returns
    -------
    dissimilarity : Dissimilarity
        Dissimilarity
    """

    def __init__(self,
                 annotation_task,
                 list_admitted_symbols,
                 alpha=3,
                 beta=1,
                 DELTA_EMPTY=1,
                 function_distance=None,
                 symbol_dissimlarity_matrix=None,
                 function_cat=lambda x: x):

        super(Combined_Sequence_Dissimilarity, self).__init__()
        self.annotation_task = annotation_task
        self.function_distance = function_distance
        self.annotation_task = annotation_task
        self.list_admitted_symbols = list_admitted_symbols

        self.alpha = alpha
        self.beta = beta

        self.num_symbols = len(self.list_admitted_symbols)
        self.dict_list_symbols = dictionary = dict(
            zip(self.list_admitted_symbols, list(range(self.num_symbols))))
        dissimilarity_matrix = symbol_dissimlarity_matrix
        self.symbol_dissimlarity_matrix = dissimilarity_matrix

        self.function_cat = function_cat
        self.function_distance = function_distance
        self.DELTA_EMPTY = DELTA_EMPTY

        self.positional_dissimilarity = Positional_Dissimilarity(
            annotation_task=annotation_task,
            DELTA_EMPTY=DELTA_EMPTY,
            function_distance=function_distance)

        self.sequence_dissimlarity = Sequence_Dissimilarity(
            annotation_task=annotation_task,
            list_admitted_symbols=self.list_admitted_symbols,
            symbol_dissimlarity_matrix=self.symbol_dissimlarity_matrix,
            DELTA_EMPTY=DELTA_EMPTY,
            function_cat=function_cat)

    @lru_cache(maxsize=None)
    def __getitem__(self, units):
        timing_units, categorical_units = units
        assert len(timing_units) == len(categorical_units)
        if len(timing_units) < 2:
            return self.DELTA_EMPTY
        else:
            dis = self.alpha * self.positional_dissimilarity[timing_units]
            dis += self.beta * self.sequence_dissimlarity[categorical_units]
            return dis
