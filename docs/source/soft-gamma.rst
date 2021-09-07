.. _softgamma:

========================================================
The Soft-Gamma-Agreement : an alternate measure to gamma
========================================================

To complete the gamma-agreement, we have added an option called "soft" gamma.
It is a small tweak to the measure created with the goal of doing exactly what gamma
does, except it reduces the disagreement caused by splits in annotations.

The idea behind this concept is to make use of the gamma agreement with machine learning models,
since most of the existing ones are prone to produce of lot more splitted annotations than human annotators.

How to use soft-gamma
~~~~~~~~~~~~~~~~~~~~~

The soft-gamma measure, for the user at least, works exactly like gamma :

.. code-block:: python

    continuum = pa.Continuum.from_csv("tests/data/AlexPaulSuzan.csv")
    dissim = pa.CombinedCategoricalDissimilarity(delta_empty=1,
                                                 alpha=1,
                                                 beta=1)
    gamma_results = continuum.compute_gamma(dissim, soft=True)

    print(f"gamma = {gamma_results.gamma}")
    pa.show_alignment(gamma_results.best_alignment)


The only difference will be the look of the resulting best alignment, as well as the gamma value.
This new value can be lower than the normal gamma (it cannot be higher).
The more splitted annotations the input continuum contains, the wider the differences between the two measures will be.



