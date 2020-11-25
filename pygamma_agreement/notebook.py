#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CoML

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
# Hadrien TITEUX & Rachid RIAD
# Inspired by code from pyannote.core, Hervé BREDIN - http://herve.niderb.fr

"""
#############
Visualization
#############
"""
from typing import Iterable, Dict, Optional, Tuple, Hashable, Union

from pyannote.core import Timeline, Segment

try:
    from IPython.core.pylabtools import print_figure
except Exception as e:
    pass

try:
    from matplotlib.markers import MarkerStyle
    from matplotlib.cm import get_cmap
except Exception as e:
    pass
import numpy as np
from itertools import cycle, product, groupby

from .alignment import Alignment
from .continuum import Continuum

LabelStyle = Tuple[str, int, Tuple[float, float, float]]
Label = Hashable


class Notebook:

    def __init__(self):
        self._crop: Optional[Segment] = None
        self._width: Optional[int] = None
        self._style: Dict[Optional[Label], LabelStyle] = {}
        self._style_generator: Iterable[LabelStyle] = None
        self.reset()

    def reset(self):
        line_width = [3, 1]
        line_style = ['solid', 'dashed', 'dotted']

        cm = get_cmap('Set1')
        colors = [cm(1. * i / 8) for i in range(9)]

        self._style_generator = cycle(product(line_style, line_width, colors))
        self._style: Dict[Optional[Label], LabelStyle] = {
            None: ('solid', 1, (0.0, 0.0, 0.0))
        }
        del self.crop
        del self.width

    @property
    def crop(self):
        """The crop property."""
        return self._crop

    @crop.setter
    def crop(self, segment: Segment):
        self._crop = segment

    @crop.deleter
    def crop(self):
        self._crop = None

    @property
    def width(self):
        """The width property"""
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value

    @width.deleter
    def width(self):
        self._width = 20

    def __getitem__(self, label: Label) -> LabelStyle:
        if label not in self._style:
            self._style[label] = next(self._style_generator)
        return self._style[label]

    def setup(self, ax=None, ylim=(0, 1), yaxis=False, time=True):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        ax.set_xlim(self.crop)
        if time:
            ax.set_xlabel('Time')
        else:
            ax.set_xticklabels([])
        ax.set_ylim(ylim)
        ax.axes.get_yaxis().set_visible(yaxis)
        return ax

    def draw_segment(self, ax, segment: Segment, y, annotator=None, boundaries=True):

        # do nothing if segment is empty
        if not segment:
            return

        linestyle, linewidth, color = self[annotator]

        # draw segment
        ax.hlines(y, segment.start, segment.end, color,
                  linewidth=linewidth, linestyle=linestyle, label=annotator)
        if boundaries:
            ax.vlines(segment.start, y + 0.05, y - 0.05,
                      color, linewidth=1, linestyle='solid')
            ax.vlines(segment.end, y + 0.05, y - 0.05,
                      color, linewidth=1, linestyle='solid')

        if annotator is None:
            return

    def draw_centered_text(self, ax, x: float, y: float, text: str):
        ax.text(x, y, text,
                horizontalalignment='center',
                fontsize=10, color='black',
                )

    def draw_centered_segment(self, ax, length: float, center: float, y,
                              annotator=None, boundaries=True, text=None):
        segment = Segment(center - (length / 2), center + (length / 2))
        self.draw_segment(ax, segment, y, annotator, boundaries)
        if text is not None:
            self.draw_centered_text(ax, center, y, text)

    def draw_vline(self, ax, x: float, y_lim: Tuple[float, float]):
        ax.vlines(x, y_lim[0], y_lim[1],
                  "black", linewidth=1, linestyle='dotted')

    def draw_empty_unit(self, ax, x, y, label=None):
        linestyle, linewidth, color = self[label]
        ax.scatter(x, y, color=color, marker=MarkerStyle(marker="X"))

    def draw_legend_from_labels(self, ax):
        H, L = ax.get_legend_handles_labels()

        # corner case when no segment is visible
        if not H:
            return

        # this gets exactly one legend handle and one legend label per label
        # (avoids repeated legends for repeated tracks with same label)
        HL = groupby(sorted(zip(H, L), key=lambda h_l: h_l[1]),
                     key=lambda h_l: h_l[1])
        H, L = zip(*list((next(h_l)[0], l) for l, h_l in HL))
        ax.legend(H, L, bbox_to_anchor=(0, 1), loc=3,
                  ncol=5, borderaxespad=0., frameon=False)

    def get_y(self, segments: Iterable[Segment]) -> np.ndarray:
        """

        Parameters
        ----------
        segments : Iterable
            `Segment` iterable (sorted)

        Returns
        -------
        y : np.array
            y coordinates of each segment

        """

        # up_to stores the largest end time
        # displayed in each line (at the current iteration)
        # (at the beginning, there is only one empty line)
        up_to = [-np.inf]

        # y[k] indicates on which line to display kth segment
        y = []

        for segment in segments:
            # so far, we do not know which line to use
            found = False
            # try each line until we find one that is ok
            for i, u in enumerate(up_to):
                # if segment starts after the previous one
                # on the same line, then we add it to the line
                if segment.start >= u:
                    found = True
                    y.append(i)
                    up_to[i] = segment.end
                    break
            # in case we went out of lines, create a new one
            if not found:
                y.append(len(up_to))
                up_to.append(segment.end)

        # from line numbers to actual y coordinates
        y = 1. - 1. / (len(up_to) + 1) * (1 + np.array(y))
        return y

    def __call__(self, resource: Union[Alignment, Continuum],
                 time: bool = True,
                 legend: bool = True):

        if isinstance(resource, Alignment):
            self.plot_alignment(resource, time=time)

        elif isinstance(resource, Continuum):
            self.plot_continuum(resource, time=time)

    def plot_alignment(self, alignment: Alignment, ax=None, time=True, legend=True):
        self.crop = Segment(0, alignment.num_alignments)

        ax = self.setup(ax=ax, time=time)
        ax.set_xlabel('Alignments')

        all_segs = []
        for unit_alignment in alignment.unitary_alignments:
            for annot, unit in unit_alignment.n_tuple:
                if unit is None:
                    continue
                all_segs.append(unit.segment)
        max_duration = max(seg.duration for seg in all_segs)

        for align_id, unit_alignment in enumerate(alignment.unitary_alignments):
            unit_align_center = (align_id * 2 + 1) / 2
            self.draw_vline(ax, unit_align_center, (0, 1))
            for annot_id, (annotator, unit) in enumerate(unit_alignment.n_tuple):
                y = (annot_id + 1) / (alignment.num_annotators + 1)

                if unit is None:
                    self.draw_empty_unit(ax, unit_align_center, y, annotator)
                else:
                    normed_len = unit.segment.duration / max_duration
                    self.draw_centered_segment(ax,
                                               length=normed_len,
                                               center=unit_align_center,
                                               y=y,
                                               annotator=annotator,
                                               # text=unit.annotation
                                               )
        if legend:
            self.draw_legend_from_labels(ax)

    def plot_continuum(self, continuum: Continuum, ax=None, time=True,
                       legend=True):
        self.crop = Timeline([unit.segment for (_, unit) in continuum]).extent()
        self.setup(ax, ylim=(0, continuum.num_annotators))

        for annot_id, annotator in enumerate(continuum.annotators):
            units_tl = Timeline([unit.segment for unit in continuum[annotator]])
            for segment, y in zip(units_tl, self.get_y(units_tl)):
                self.draw_segment(ax, segment, y + annot_id,
                                  annotator=annotator)
        if legend:
            self.draw_legend_from_labels(ax)


notebook = Notebook()


def repr_alignment(alignment: Alignment):
    """Get `png` data for `Alignment`"""
    import matplotlib.pyplot as plt
    figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = (notebook.width, 1)
    fig, ax = plt.subplots()
    notebook.plot_alignment(alignment, ax=ax)
    data = print_figure(fig, 'png')
    plt.close(fig)
    plt.rcParams['figure.figsize'] = figsize
    return data


def repr_continuum(continuum: Continuum):
    """Get `png` data for `annotation`"""
    import matplotlib.pyplot as plt
    figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = (notebook.width, 2)
    fig, ax = plt.subplots()
    notebook.plot_continuum(continuum, ax=ax)
    data = print_figure(fig, 'png')
    plt.close(fig)
    plt.rcParams['figure.figsize'] = figsize
    return data
