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
