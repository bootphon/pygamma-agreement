import logging
from pathlib import Path
import numpy as np
import time

from pygamma import Continuum, CombinedCategoricalDissimilarity

class Timer:

    def start(self):
        self.start_time = time.time()
        self.lap_time = self.start_time

    def lap(self):
        last_time = self.lap_time
        self.lap_time = time.time()
        print("Done in ", self.lap_time - last_time)

    def total(self):
        print("All done in ", time.time() - self.start_time)

timer = Timer()
timer.start()
logging.getLogger().setLevel(logging.DEBUG)

print("Loading")
continuum = Continuum.from_csv(Path("tests/data/AlexPaulSuzan.csv"))
timer.lap()

dissim = CombinedCategoricalDissimilarity(list(continuum.categories),
                                          delta_empty=1,
                                          alpha=3,
                                          beta=1)
print("Computing disorder")
print(continuum.compute_disorders(dissim))
best_alignment = continuum.get_best_alignment(dissim)
print(f"Best align length: {len(best_alignment.unitary_alignments)}")
timer.lap()
print("Computing gamma")
gamma = continuum.compute_gamma(dissim, precision_level=0.01)
print(f"Gamma is {gamma.gamma}")
timer.lap()

timer.total()