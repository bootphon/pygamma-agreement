import pygamma_agreement as pa
from pyannote.core import Segment



def test_soft_alignment_check():
    unit_martin, unit_martino = pa.Unit(Segment(0, 10), "a"), pa.Unit(Segment(5, 15), "b")
    continuum = pa.Continuum()

    continuum.add("Martin", Segment(0, 10), "a")
    continuum.add("Martino", Segment(5, 15), "b")

    unitary_alignment_1 = pa.UnitaryAlignment([("Martin", unit_martin),
                                               ("Martino", None)])
    unitary_alignment_2 = pa.UnitaryAlignment([("Martin", unit_martin),
                                               ("Martino", unit_martino)])
    unitary_alignment_3 = pa.UnitaryAlignment([("Martin", None),
                                               ("Martino", unit_martino)])

    alignment = pa.alignment.SoftAlignment([unitary_alignment_3, unitary_alignment_1, unitary_alignment_2],
                                           [0.5, 0.5, 0.5],
                                           continuum,
                                           check_validity=True)