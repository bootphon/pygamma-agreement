"""Test of Units in the pygamma_agreement.continuum module"""

from pyannote.core import Segment
from sortedcontainers import SortedSet

from pygamma_agreement.continuum import Unit


def test_unit_equality():
    assert Unit(Segment(0, 1)) == Unit(Segment(0, 1))
    assert Unit(Segment(0, 1), "A") == Unit(Segment(0, 1), "A")
    assert Unit(Segment(0, 1), "A") != Unit(Segment(0, 1), "B")
    assert Unit(Segment(0, 1)) != Unit(Segment(0, 2))
    assert Unit(Segment(0, 1), None) != Unit(Segment(0, 2))
    assert Unit(Segment(0, 1), None) == Unit(Segment(0, 1), None)


def test_unit_ordering():
    assert Unit(Segment(0, 1)) < Unit(Segment(0, 2))
    assert Unit(Segment(1, 1)) > Unit(Segment(0, 2))
    assert Unit(Segment(0, 1)) < Unit(Segment(0, 1), "A")
    assert Unit(Segment(0, 1), "C") > Unit(Segment(0, 1))
    assert Unit(Segment(0, 1), "A") < Unit(Segment(0, 1), "B")
    assert Unit(Segment(0, 1), "B") > Unit(Segment(0, 1), "A")
    assert Unit(Segment(2, 3)) > Unit(Segment(0, 1), "A")
    assert Unit(Segment(3, 4), 'B') > Unit(Segment(0, 1))


def test_units_sets():
    units = [
        Unit(Segment(0, 1)),
        Unit(Segment(0, 1)),
        Unit(Segment(0, 2)),
        Unit(Segment(0, 2), None),
        Unit(Segment(3, 4), "A"),
        Unit(Segment(3, 4), "A"),
        Unit(Segment(3, 4), "B")
    ]
    units_set = {
        Unit(Segment(0, 1)),
        Unit(Segment(0, 2), None),
        Unit(Segment(3, 4), "A"),
        Unit(Segment(3, 4), "B")
    }
    assert set(units) == units_set


def test_units_ordered_set():
    units = [
        Unit(Segment(0, 2), None),
        Unit(Segment(3, 4), "A"),
        Unit(Segment(0, 1)),
        Unit(Segment(3, 4), "B"),
        Unit(Segment(3, 4), "A"),
        Unit(Segment(0, 1)),
        Unit(Segment(0, 2)),
    ]
    units_set = [
        Unit(Segment(0, 1)),
        Unit(Segment(0, 2), None),
        Unit(Segment(3, 4), "A"),
        Unit(Segment(3, 4), "B")
    ]
    assert list(SortedSet(units)) == units_set
