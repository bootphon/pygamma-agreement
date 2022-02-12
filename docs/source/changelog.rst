#########
Changelog
#########

Version 0.5.6 (2022-02-12) (@hadware)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added support for Python 3.10
* Bumped cvxopt required version to 1.2.7

Version 0.5.4 (2021-11-29)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed a bug in the `Continuum.merge` function when a continuun had annotators with no annotations (@valentinoli)
* Fixed a bug in the `Continuum.reset_bounds()` function when a continuun had annotators with no annotations (@valentinoli)
* Added `__eq__` and `__ne__` comparison magic methods to enable == and != operators on `Continuum` instances

Version 0.5.0 (2021-09-17) (@lfavre)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added soft-gamma, an alternate inter-annotator agreement measure designed based on the gamma-agreement
    * Extensive documentation about this measure and its uses.
* Minor bug fixes

Version 0.4.1 (2021-08-30) (@lfavre)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Important bug fix : Some slicing error when the number of possible unitary alignments was to high.


Version 0.4.0 (2021-08-13) (@lfavre)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fast-gamma option
    - New algorithm that gives a satisfying approximation of the gamma-agreement, with a colossal gain in computing time and memory usage.
    - Detailed research, explanations and benchmarking about this algorithm are thoroughly detailed in the "Performances" section of the documentation
* Minor bug fixes


Version 0.3.0 (2021-08-06) (@lfavre)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* New interface for dissimilarities.
    - A real class structure, made with user-friendliness in mind
    - Making new or custom dissimilarities is now doable without copying huge chunks of numba code, supposedly without knowledge of the inner working of the library
    - The code for dissimilarities is overall clearer, more reliable and more maintainable
    - New natively available dissimilarities
* Bug fixes
    - Fixed many bugs that emerged from the sorted structures and a confusion when passing to a numba/numerical algorithmic environment
* Optimizations :
    - Memory usage of the gamma algorithm is now significantly lower than before
    - Unit-to-unit disorders are pre-computed during the best-alignment algorithm, which lowers the computation time
    - Multiprocessing has been replaced by multithreading, for additionnal memory usage, computation time and simplicity.
    - Some more parallelization in deeper code.
* Documentation
    - Extensive tutorials for dissimilarities, sampling and the corpus shuffling tool


Version 0.2.0 (2021-07-06) (@lfavre)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Numerous bug fixes (errors and mismatches with the original java implementation)
* Documentation of slight differences between the java implementation and our implementation that might impact results
* Various minor performance improvements
* Multiprocessing-based parallelization of the costly disorder computation
* Gamma-k and Gamma-cat implementation
* Levenshtein categorical dissimilarity
* New continuum sampler, and an API to create your own continuum sampler
* Addition of the Corpus Shuffling Tool for benchmarking the gamma family of agreement measures
* Rationalized the usage of various data structures to sortedcontainers to prevent any non-deterministic side effects
* Additionnal manipulation and creation methods for continuua
* New options for the command line tool, such as json output or seeding
* New visualization output for alignments in jupyter notebooks
* Changed the mixed integer programming solver from the unprecise ECOS_BB to GLPK_MI or CBC
