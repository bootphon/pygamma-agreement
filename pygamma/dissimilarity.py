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


class Categorical_Dissimilarity(object):
    """Dissimilarity
    Parameters
    ----------
    annotation_task :
        task to be annotated
    num_categories :
        number of categories
    categorical_dissimlarity_matrix :
        Dissimilarity matrix to compute
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
        self.num_categories = len(self.list_categories)
        self.categorical_dissimlarity_matrix = categorical_dissimlarity_matrix
        if self.categorical_dissimlarity_matrix is None:
            self.categorical_dissimlarity_matrix = np.ones(
                (self.num_categories, self.num_categories)) - np.eye(
                    self.num_categories)

    def plot_categorical_dissimlarity_matrix(self):
        fig, ax = plt.subplots()
        im = plt.imshow(
            cat, extent=[0, self.num_categories, 0, self.num_categories])
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


class Combined_Dissimilarity(object):
    """Dissimilarity
    Parameters
    ----------
    categorical_dissimlarity : Dissimilarity function to use between units
    Returns
    -------
    dissimilarity : Dissimilarity
        Dissimilarity
    """
