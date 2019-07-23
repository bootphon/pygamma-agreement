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


class Categorical_Dissimilarity(object):
    """Categorical Dissimilarity
    Parameters
    ----------
    annotation_task :
        task to be annotated
    list_categories :
        list of categories
    categorical_dissimlarity_matrix :
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
                 categorical_dissimlarity_matrix=None,
                 DELTA_EMPTY=1,
                 function_cat=lambda x: x):

        super(Categorical_Dissimilarity, self).__init__()
        self.annotation_task = annotation_task
        self.list_categories = list_categories
        assert len(list_categories) == len(set(list_categories))
        self.num_categories = len(self.list_categories)
        self.dict_list_categories = dictionary = dict(
            zip(self.list_categories, list(range(self.num_categories))))
        self.categorical_dissimlarity_matrix = categorical_dissimlarity_matrix
        if self.categorical_dissimlarity_matrix is None:
            self.categorical_dissimlarity_matrix = np.ones(
                (self.num_categories, self.num_categories)) - np.eye(
                    self.num_categories)
        else:
            assert type(self.categorical_dissimlarity_matrix) == np.ndarray
            assert np.all(self.categorical_dissimlarity_matrix <= 1)
            assert np.all(0 <= self.categorical_dissimlarity_matrix)
            assert np.all(self.categorical_dissimlarity_matrix ==
                          categorical_dissimlarity_matrix.T)
        self.function_cat = function_cat
        self.DELTA_EMPTY = DELTA_EMPTY

    def plot_categorical_dissimilarity_matrix(self):
        fig, ax = plt.subplots()
        im = plt.imshow(
            self.categorical_dissimlarity_matrix,
            extent=[0, self.num_categories, 0, self.num_categories])
        ax.figure.colorbar(im, ax=ax)
        plt.xticks([el + 0.5 for el in range(self.num_categories)],
                   self.list_categories)
        plt.yticks([el + 0.5 for el in range(self.num_categories)],
                   self.list_categories)
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor")
        plt.show()

    def __getitem__(self, units):
        assert type(units) == list
        if len(units) < 2:
            return self.DELTA_EMPTY
        else:
            assert units[0] in self.list_categories
            assert units[1] in self.list_categories
            return self.function_cat(
                self.
                categorical_dissimlarity_matrix[self.
                                                dict_list_categories[units[0]]]
                [self.dict_list_categories[units[1]]]) * self.DELTA_EMPTY


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

    def __getitem__(self, units):
        assert type(units) == list
        if len(units) < 2:
            return self.DELTA_EMPTY
        else:
            if self.function_distance is None:
                # triple indexing to tracks in pyannote
                # DANGER if the api breaks
                distance_pos = (np.abs(units[0][0] - units[1][0]) +
                                np.abs(units[0][1] - units[1][1]))
                distance_pos /= (units[0][1] - units[0][0] +
                                 units[1][1] - units[1][0])
                distance_pos = distance_pos * distance_pos * self.DELTA_EMPTY
                return distance_pos


class Combined_Dissimilarity(object):
    """Combined Dissimilarity
    Parameters
    ----------
    annotation_task :
        task to be annotated
    num_categories :
        number of categories
    categorical_dissimlarity_matrix :
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
                 alpha=1,
                 beta=1,
                 DELTA_EMPTY=1,
                 function_distance=None,
                 categorical_dissimlarity_matrix=None,
                 function_cat=lambda x: x):

        super(Combined_Dissimilarity, self).__init__()
        self.annotation_task = annotation_task
        self.function_distance = function_distance
        self.annotation_task = annotation_task
        self.list_categories = list_categories
        self.num_categories = len(self.list_categories)

        self.alpha = alpha
        self.beta = beta

        self.dict_list_categories = dictionary = dict(
            zip(self.list_categories, list(range(self.num_categories))))
        self.categorical_dissimlarity_matrix = categorical_dissimlarity_matrix

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
            categorical_dissimlarity_matrix=categorical_dissimlarity_matrix,
            DELTA_EMPTY=DELTA_EMPTY,
            function_cat=function_cat)

    def __getitem__(self, units):
        timing_units, categorical_units = units
        assert len(timing_units) == len(categorical_units)
        if len(timing_units) < 2:
            return self.DELTA_EMPTY
        else:
            dis = self.alpha * self.positional_dissimilarity[timing_units]
            dis += self.beta * self.categorical_dissimlarity[categorical_units]
            return dis
