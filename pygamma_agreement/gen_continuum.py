from .continuum import Continuum
from typing import Union, Iterable

class ContinuumGenerator:
    """
    Class for generating random continuua, using exponential distribution.
    """
    def __init__(self,
                 annotators: Union[int, Iterable[str]],
                 intensity_frequency: float,
                 intensity_duration: float,
                 categories: Union[int, Iterable[str]]):
        if isinstance(annotators, int):
            self.annotators = [f"annotator_{annotator}" for annotator in range(annotators)]
            self.nb_annotators = len(self.annotators)
            self.intensity_duration = intensity_duration
            self.intensity_frequency = intensity_frequency
