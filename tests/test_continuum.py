"""Test of the Continuum class in pygamma_agreement.continuum"""

from pyannote.core import Annotation, Segment

from pygamma_agreement.continuum import Continuum


def test_continuum_init():
    continuum = Continuum()
    annotation = Annotation()
    assert len(continuum) == 0
    annotation[Segment(1, 5)] = 'Carol'
    annotation[Segment(6, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Carol'
    annotation[Segment(7, 20)] = 'Alice'
    continuum.add_annotation("liza", annotation)
    annotation = Annotation()
    annotation[Segment(2, 6)] = 'Carol'
    annotation[Segment(7, 8)] = 'Bob'
    annotation[Segment(12, 18)] = 'Alice'
    annotation[Segment(8, 10)] = 'Alice'
    annotation[Segment(7, 19)] = 'Carol'
    continuum.add_annotation("pierrot", annotation)
    assert continuum
    assert len(continuum) == 2
    assert continuum.num_units == 9
    assert continuum.avg_num_annotations_per_annotator == 4.5


def test_continuum_from_elan():
    continuum = Continuum()
    continuum.add_elan("annotator1", "tests/data/MaureenMarvinRobin.eaf")

    assert continuum.num_units == 4
    assert set(continuum.categories) == {"S"}
    assert set(continuum.annotators) == {"annotator1"}

    continuum = Continuum()
    continuum.add_elan("annotator1", "tests/data/MaureenMarvinRobin.eaf",
                       selected_tiers=["Maureen", "Robin"],
                       use_tier_as_annotation=True)

    assert continuum.num_units == 3
    assert set(continuum.categories) == {"Maureen", "Robin"}


def test_continuum_from_textgrid():
    continuum = Continuum()
    continuum.add_textgrid("annotator1", "tests/data/MaureenMarvinRobin.TextGrid")

    assert continuum.num_units == 4
    assert set(continuum.categories) == {"S"}
    assert set(continuum.annotators) == {"annotator1"}

    continuum = Continuum()
    continuum.add_textgrid("annotator1", "tests/data/MaureenMarvinRobin.TextGrid",
                           selected_tiers=["Maureen", "Robin"],
                           use_tier_as_annotation=True)

    assert continuum.num_units == 3
    assert set(continuum.categories) == {"Maureen", "Robin"}


def test_continuum_merging():
    continuum = Continuum()
    continuum.add("marvin", Segment(0, 1), "A")
    continuum.add("marvin", Segment(2, 3), "A")
    continuum.add("robin", Segment(0, 2), "B")
    continuum.add("robin", Segment(3, 4), "C")
    other_cont = Continuum()
    other_cont.add("maureen", Segment(5, 6), "D")
    other_cont.add("maureen", Segment(9, 10), "D")
    other_cont.add_annotator("nick")

    merged_cont = continuum.merge(other_cont)
    assert merged_cont.num_units == 6
    assert merged_cont.num_annotators == 4

    continuum.merge(other_cont, in_place=True)
    assert continuum == merged_cont


def test_continua_eq():
    cont_a = Continuum()
    cont_a.add("marvin", Segment(0, 1), "A")
    cont_a.add("marvin", Segment(2, 3), "A")
    cont_a.add("robin", Segment(0, 2), "B")
    cont_a.add("robin", Segment(3, 4), "C")

    cont_b = Continuum()
    cont_b.add("marvin", Segment(0, 1), "A")
    cont_b.add("marvin", Segment(2, 3), "A")
    cont_b.add("robin", Segment(0, 2), "B")

    assert cont_a != cont_b

    cont_b.add("robin", Segment(3, 4), "C")
    assert cont_a == cont_b

    cont_b.add_annotator("maureen")
    assert cont_a != cont_b

    cont_a.add_annotator("maureen")
    assert cont_a == cont_b

    cont_a.add("maureen", Segment(0, 1))
    cont_b.add("maureen", Segment(0, 1), "C")
    assert cont_a != cont_b
