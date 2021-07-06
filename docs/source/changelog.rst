#########
Changelog
#########

Version 0.2.0 (2021-07-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~

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
