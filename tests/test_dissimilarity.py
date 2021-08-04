"""Test of the module pygamma_agreement.dissimilarity"""

import numpy as np
import pytest
from pyannote.core import Annotation, Segment
from sortedcontainers import SortedSet

from pygamma_agreement.alignment import (UnitaryAlignment)
from pygamma_agreement.continuum import Continuum, Unit
from pygamma_agreement.dissimilarity import (PositionalSporadicDissimilarity,
                                             CombinedCategoricalDissimilarity,
                                             PrecomputedCategoricalDissimilarity,
                                             AbstractDissimilarity,
                                             CategoricalDissimilarity,
                                             LambdaCategoricalDissimilarity)


def test_categorical_dissimilarity():
    categories = SortedSet(['A', 'B', 'C', 'D'])
    cat = np.array([[0, 0.5, 0.3, 0.7], [0.5, 0., 0.6, 0.4],
                    [0.3, 0.6, 0., 0.7], [0.7, 0.4, 0.7, 0.]])

    cat_dis = CombinedCategoricalDissimilarity(
        cat_dissim=PrecomputedCategoricalDissimilarity(categories, cat, delta_empty=0.5),
        delta_empty=0.5,
        alpha=0.0,
        beta=1.0)
    fake_seg = Segment(0, 1)
    unit_alignment = UnitaryAlignment([("Carol", Unit(fake_seg, "A")),
                                       ("John", Unit(fake_seg, "D"))])
    assert unit_alignment.compute_disorder(cat_dis) == np.float32(0.35)

    unit_alignment = UnitaryAlignment([("Carol", Unit(fake_seg, "A")),
                                       ("John", Unit(fake_seg, "A"))])
    assert unit_alignment.compute_disorder(cat_dis) == np.float32(0.0)

    unit_alignment_a = UnitaryAlignment([("Carol", Unit(fake_seg, "A")),
                                         ("John", Unit(fake_seg, "B"))])
    unit_alignment_b = UnitaryAlignment([("Carol", Unit(fake_seg, "B")),
                                         ("John", Unit(fake_seg, "A"))])
    assert (unit_alignment_a.compute_disorder(cat_dis)
            ==
            unit_alignment_b.compute_disorder(cat_dis))


def test_positional_dissimilarity():
    continuum = Continuum()
    annotation = Annotation()
    annotation[Segment(1, 5)] = 'Carol'
    annotation[Segment(6, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Carol'
    annotation[Segment(7, 20)] = 'Alice'
    continuum.add_annotation('liza', annotation)
    annotation = Annotation()
    annotation[Segment(2, 6)] = 'Carol'
    annotation[Segment(7, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Alice'
    annotation[Segment(8, 10)] = 'Alice'
    annotation[Segment(7, 19)] = 'Jeremy'
    continuum.add_annotation('pierrot', annotation)

    pos_dis = PositionalSporadicDissimilarity(delta_empty=0.5)

    list_dis = []
    for liza_unit in continuum['liza']:
        for pierrot_unit in continuum['pierrot']:
            unit_alignment = UnitaryAlignment([("liza", liza_unit),
                                               ("pierrot", pierrot_unit)])
            list_dis.append(unit_alignment.compute_disorder(pos_dis))
    assert list_dis == pytest.approx([
        0.03125, 1.62, 0.78125, 2.0, 2.88, 0.5, 0.05555555555555555,
        0.36734693877551017, 0.5, 2.0, 0.6245674740484429, 0.36734693877551017,
        0.0008, 0.26888888888888884, 0.06786703601108032, 2.4200000000000004,
        2.2959183673469385, 0.05555555555555555, 1.125, 0.0
    ], 0.001)

    liza_unit = Unit(Segment(45, 77), "")
    pierrot_unit = Unit(Segment(16, 64), "")
    unit_alignment_a = UnitaryAlignment([("liza", liza_unit),
                                         ("pierrot", pierrot_unit)])
    unit_alignment_b = UnitaryAlignment([("pierrot", pierrot_unit),
                                         ("liza", liza_unit)])
    assert (unit_alignment_a.compute_disorder(pos_dis)
            ==
            unit_alignment_b.compute_disorder(pos_dis))

    unit_alignment = UnitaryAlignment([("liza", liza_unit),
                                       ("pierrot", liza_unit)])
    assert unit_alignment.compute_disorder(pos_dis) == np.float32(0.0)


def test_positional_dissimilarity_figure10():
    pos_dis = PositionalSporadicDissimilarity(delta_empty=1.0)
    segments = {
        (Segment(4, 14), Segment(40, 44)): 22.2,
        (Segment(4, 14), Segment(4, 14)): 0.,
        (Segment(4, 14), Segment(20, 25)): 3.2,
        (Segment(4, 14), Segment(14, 24)): 1.,
        (Segment(20, 30), Segment(20, 25)): 0.11,
        (Segment(20, 25), Segment(14, 24)): 0.22,
    }
    for (seg_a, seg_b), value in segments.items():
        unit_alignment = UnitaryAlignment([("liza", Unit(seg_a, "cat_a")),
                                           ("pierrot", Unit(seg_b, "cat_b"))])
        assert unit_alignment.compute_disorder(pos_dis) == pytest.approx(value, 0.1)


def test_positional_dissimilarity_figure20_scale_effect():
    pos_dis = PositionalSporadicDissimilarity(delta_empty=1.0)
    unit_align_a = UnitaryAlignment([("pierrot", Unit(Segment(0, 7), "cat_a")),
                                     ("liza", Unit(Segment(0, 10), "cat_b"))])
    unit_align_b = UnitaryAlignment([("pierrot", Unit(Segment(0, 21), "cat_a")),
                                     ("liza", Unit(Segment(0, 30), "cat_b"))])

    assert (unit_align_a.compute_disorder(pos_dis)
            ==
            unit_align_b.compute_disorder(pos_dis))


def test_combi_categorical_dissimilarity():
    continuum = Continuum()
    annotation = Annotation()
    annotation[Segment(1, 5)] = 'Carol'
    annotation[Segment(6, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Carol'
    annotation[Segment(7, 20)] = 'Alice'
    continuum.add_annotation('liza', annotation)
    annotation = Annotation()
    annotation[Segment(2, 6)] = 'Carol'
    annotation[Segment(7, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Alice'
    annotation[Segment(8, 10)] = 'Alice'
    annotation[Segment(7, 19)] = 'Jeremy'
    continuum.add_annotation('pierrot', annotation)
    categories = SortedSet(['Carol', 'Bob', 'Alice', 'Jeremy'])

    cat = np.array([[0., 0.6, 0.3, 0.7],
                    [0.6, 0., 0.5, 0.4],
                    [0.3, 0.5, 0., 0.7],
                    [0.7, 0.4, 0.7, 0.]])

    combi_dis = CombinedCategoricalDissimilarity(
        delta_empty=0.5,
        cat_dissim=PrecomputedCategoricalDissimilarity(categories, cat, delta_empty=0.5),
        alpha=3, beta=1)
    list_dis = []
    for liza_unit in continuum['liza']:
        for pierrot_unit in continuum['pierrot']:
            unit_alignment = UnitaryAlignment([("liza", liza_unit),
                                               ("pierrot", pierrot_unit)])
            list_dis.append(unit_alignment.compute_disorder(combi_dis))

    assert list_dis == pytest.approx([
        0.09375, 5.11, 2.69375, 6.15, 8.790000000000001, 1.75,
        0.16666666666666666, 1.3020408163265305, 1.8, 6.3, 2.0237024221453286,
        1.4020408163265305, 0.3524, 0.8066666666666665, 0.20360110803324097,
        7.260000000000002, 7.137755102040815, 0.5166666666666666, 3.525, 0.15
    ], 0.001)

    unit_align_a = UnitaryAlignment([("liza", Unit(Segment(1, 5), "Carol")),
                                     ("pierrot", Unit(Segment(7, 19), "Jeremy"))])
    unit_align_b = UnitaryAlignment([("pierrot", Unit(Segment(7, 19), "Jeremy")),
                                     ("liza", Unit(Segment(1, 5), "Carol")),])
    assert (unit_align_a.compute_disorder(combi_dis)
            ==
            unit_align_b.compute_disorder(combi_dis))

    same_align = UnitaryAlignment([("liza", Unit(Segment(1, 5), "Carol")),
                                   ("pierrot", Unit(Segment(1, 5), "Carol"))])

    assert same_align.compute_disorder(combi_dis) == np.float32(0.0)


def test_custom_dissimilarity():
    from pygamma_agreement import AbstractDissimilarity, Unit
    from typing import Callable

    continuum = Continuum.from_csv("tests/data/AlexPaulSuzan.csv")

    class MyPositionalDissimilarity(AbstractDissimilarity):
        def __init__(self, p: int, delta_empty=1.0):
            self.p = p
            assert p > 0
            super().__init__(delta_empty=delta_empty)

        def d(self, unit1: Unit, unit2: Unit) -> float:
            return (abs(unit1.segment.start - unit2.segment.start) ** self.p
                    + abs(unit1.segment.end - unit2.segment.end) ** self.p) ** (1 / self.p)

        def compile_d_mat(self) -> Callable[[np.ndarray, np.ndarray], float]:
            # Calling self inside d_mat makes the compiler choke, so you need to copy attributes in locals.
            p = self.p
            delta_empty = self.delta_empty
            from pygamma_agreement import dissimilarity_dec

            @dissimilarity_dec  # This decorator specifies that this function will be compiled.
            def d_mat(unit1: np.ndarray, unit2: np.ndarray) -> float:
                # We're in numba environment here, which means that only python/numpy types and operations will work.
                return (abs(unit1[0] - unit2[0]) ** p + abs(unit1[1] - unit2[1]) ** p) ** (1 / p) * delta_empty
            return d_mat

    pos_dissim = MyPositionalDissimilarity(p=2, delta_empty=1.0)
    np.random.seed(4556)
    gamma_results_1 = continuum.compute_gamma(pos_dissim)
    np.random.seed(4556)
    gamma_results_2 = continuum.compute_gamma(PositionalSporadicDissimilarity())
    pos_dissim.p = 10
    pos_dissim = MyPositionalDissimilarity(p=10, delta_empty=1.0)
    np.random.seed(4556)
    gamma_results_3 = continuum.compute_gamma(pos_dissim)

    assert gamma_results_1.gamma != gamma_results_2.gamma
    assert gamma_results_2.gamma != gamma_results_3.gamma
    assert gamma_results_3.gamma != gamma_results_1.gamma

    class MyCombinedDissimilarity(AbstractDissimilarity):
        def __init__(self, alpha: float, beta: float,
                     pos_dissim: AbstractDissimilarity,
                     cat_dissim: CategoricalDissimilarity,
                     delta_empty=1.0):
            self.alpha, self.beta = alpha, beta
            self.pos_dissim, self.cat_dissim = pos_dissim, cat_dissim
            super().__init__(cat_dissim.categories, delta_empty=delta_empty)

        def compile_d_mat(self) -> Callable[[np.ndarray, np.ndarray], float]:
            alpha, beta = self.alpha, self.beta
            pos, cat = self.pos_dissim.d_mat, self.cat_dissim.d_mat
            # d_mat attribute contains the numba-compiled function

            from pygamma_agreement import dissimilarity_dec

            @dissimilarity_dec
            def d_mat(unit1: np.ndarray, unit2: np.ndarray) -> float:
                return pos(unit1, unit2) ** alpha * cat(unit1, unit2) ** beta

            return d_mat

        def d(self, unit1: Unit, unit2: Unit):
            return (self.pos_dissim.d(unit1, unit2) ** self.alpha *
                    self.cat_dissim.d(unit1, unit2) ** self.beta)

    class MyCategoricalDissimilarity(LambdaCategoricalDissimilarity):
        @staticmethod
        def cat_dissim_func(str1: str, str2: str) -> float:
            for i in range(min(len(str1), len(str2))):
                if str1[:i] != str2[:i]:
                    return 1 - 2*(i - 1) / (len(str1) + len(str2))
            return 1 if max(len(str1), len(str2)) > 0 else 0

    cat_dissim = MyCategoricalDissimilarity(continuum.categories)

    dissim1 = MyCombinedDissimilarity(3, 1, pos_dissim, cat_dissim)
    dissim2 = CombinedCategoricalDissimilarity(3, 1, pos_dissim=pos_dissim, cat_dissim=cat_dissim)

    np.random.seed(4556)
    gamma_results_1 = continuum.compute_gamma(dissim1)
    np.random.seed(4556)
    gamma_results_2 = continuum.compute_gamma(dissim2)

    assert gamma_results_1.gamma != gamma_results_2.gamma







