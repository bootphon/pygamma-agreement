from pyannote.core import Segment

from pygamma_agreement import (CombinedCategoricalDissimilarity,
                               Continuum,
                               show_alignment,
                               show_continuum)

continuum = Continuum()
continuum.add("Annotator1", Segment(2.5, 4.3), "Maureen")
continuum.add("Annotator1", Segment(4.6, 7.4), "Marvin")
continuum.add("Annotator1", Segment(8.2, 11.4), "Marvin")
continuum.add("Annotator1", Segment(13.5, 16.0), "Robin")

continuum.add("Annotator2", Segment(2.3, 4.5), "Maureen")
continuum.add("Annotator2", Segment(4.3, 7.2), "Marvin")
continuum.add("Annotator2", Segment(7.9, 11.2), "Robin")
continuum.add("Annotator2", Segment(13.0, 16.1), "Maureen")

continuum.add("Annotator3", Segment(2.5, 4.3), "Maureen")
continuum.add("Annotator3", Segment(4.6, 11.5), "Marvin")
continuum.add("Annotator3", Segment(13.1, 17.1), "Robin")

dissim = CombinedCategoricalDissimilarity(alpha=1, beta=2)

gamma_results = continuum.compute_gamma(dissim)

print(f"The gamma for that annotation is f{gamma_results.gamma}")
