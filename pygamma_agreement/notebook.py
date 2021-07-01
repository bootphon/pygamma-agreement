# The MIT License (MIT)

# Copyright (c) 2020-2021 CoML

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
from typing import Iterable, Dict, Optional, Tuple, Hashable, Union, Iterator

from pyannote.core import Timeline, Segment

try:
    from IPython.core.pylabtools import print_figure
    from IPython.core.display import display_png
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
        self._style_generator: Iterator[LabelStyle] = iter([])
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

    def draw_segment(self, ax, segment: Segment, y, annotator=None, boundaries=True, text=None):

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
        if text is not None:
            self.draw_centered_text(ax, segment.start + (segment.end - segment.start)/2, y, text)

        if annotator is None:
            return

    def draw_centered_text(self, ax, x: float, y: float, text: str):
        ax.text(x, y-0.07, text,
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

    def link_segments(self, ax, segment1: Segment, y1: float, segment2: Segment, y2: float):
        x = [(segment1.end + segment1.start) / 2, (segment2.end + segment2.start) / 2]
        y = [y1, y2]
        ax.plot(x, y, color='black', linestyle='dotted', linewidth=1)



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
            self.plot_continuum(resource)

    def plot_alignment(self, alignment: Alignment, ax=None, time=True, legend=True, labelled=True):
        if alignment.continuum is not None:
            self.plot_alignment_continuum(alignment, ax, time, legend, labelled)
            return
        self.crop = Segment(0, alignment.num_unitary_alignments)

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
                    if labelled:
                        text = unit.annotation
                    else:
                        text = None
                    self.draw_centered_segment(ax,
                                               length=normed_len,
                                               center=unit_align_center,
                                               y=y,
                                               annotator=annotator,
                                               text=text
                                               )
        if legend:
            self.draw_legend_from_labels(ax)

    def plot_alignment_continuum(self, alignment: Alignment, ax=None, time=True, legend=True, labelled=True):
        assert alignment.continuum is not None
        y_annotator_unit = self.plot_continuum(alignment.continuum, ax=ax, legend=legend, labelled=labelled)
        for unitary_alignment in alignment:
            annotations = iter(unitary_alignment.n_tuple)
            last_unit = None
            while last_unit is None:
                last_annotator, last_unit = next(annotations)
            for (annotator, unit) in annotations:
                if unit is None:
                    continue
                self.link_segments(ax,
                                   last_unit.segment, y_annotator_unit[last_annotator][last_unit],
                                   unit.segment, y_annotator_unit[annotator][unit])
                last_annotator, last_unit = annotator, unit

    def plot_continuum(self, continuum: Continuum, ax=None,  # time=True,
                       legend=True, labelled=True):
        self.crop = Segment(continuum.bound_inf, continuum.bound_sup)
        self.setup(ax, ylim=(0, continuum.num_annotators))
        y_annotator_unit = {}
        for annot_id, annotator in enumerate(continuum.annotators):
            y_annotator_unit[annotator] = {}
            units_tl = continuum[annotator]
            for unit, y in zip(units_tl, self.get_y(unit.segment for unit in units_tl)):
                y_annotator_unit[annotator][unit] = y + annot_id
                if labelled:
                    text = unit.annotation
                else:
                    text = None
                self.draw_segment(ax, unit.segment, y + annot_id,
                                  annotator=annotator, text=text)
        if legend:
            self.draw_legend_from_labels(ax)
        return y_annotator_unit


notebook = Notebook()


def repr_alignment(alignment: Alignment, labelled=True):
    """Get `png` data for `Alignment`"""
    import matplotlib.pyplot as plt
    figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = (notebook.width, 2)
    fig, ax = plt.subplots()
    notebook.plot_alignment(alignment, ax=ax, labelled=labelled)
    data = print_figure(fig, 'png')
    plt.close(fig)
    plt.rcParams['figure.figsize'] = figsize
    return data


def repr_continuum(continuum: Continuum, labelled=True):
    """Get `png` data for `Continuum`"""
    import matplotlib.pyplot as plt
    figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = (notebook.width, 2)
    fig, ax = plt.subplots()
    notebook.plot_continuum(continuum, ax=ax, labelled=labelled)
    data = print_figure(fig, 'png')
    plt.close(fig)
    plt.rcParams['figure.figsize'] = figsize
    return data

def show_continuum(continuum: Continuum, labelled=True):
    import matplotlib.pyplot as plt
    figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = (notebook.width, 2)
    fig, ax = plt.subplots()
    notebook.plot_continuum(continuum, ax=ax, labelled=labelled)
    fig.show()
    plt.rcParams['figure.figsize'] = figsize


def show_alignment(alignment: Alignment, labelled=True):
    import matplotlib.pyplot as plt
    figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = (notebook.width, 2)
    fig, ax = plt.subplots()
    notebook.plot_alignment_continuum(alignment, ax=ax, labelled=labelled)
    fig.show()
    plt.rcParams['figure.figsize'] = figsize
