======
Issues
======

Gamma Software issues
---------------------

We observed early on in pygamma-agreement development that we weren't able to perfectly match [mathet2015]_'s results
from their closed-source `Java implementation <https://gamma.greyc.fr/>`_ (the "Gamma Software"). In an effort to
understand these discrepancies between our implementation of the gamma measure and theirs, we decompiled their
application and carefully studied its code. This allowed us to find a number of small (yet significant) implementation
details that were either undocumented or arbitrary.

In this section, we list off all the details that might make our calculation of the gamma-agreement deviate from the
Java implementation's calculation, and explain what our own implementation choice is.

.. warning::

    What we call "undocumented" are choices of implementation found in the Gamma Software that are not mentionned
    or explained in [mathet2015]_ or [mathet2018]_. We made the choice of replicating some of those in
    pygamma-agreement, and not others,

1. Average number of annotations per annotator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In [mathet2015]_, section 4.3, a value is defined as such:


    "let :math:`\bar{x}={\frac{\sum_{i=1}^{n}x_i}{n}}` be the average number of annotations per annotator"

This value is involved in the computation of the disorder of an alignment.

**In the Gamma Software:**
an int-instead-of-float division transforms this value into
:math:`\bar{x}=\lfloor{\frac{\sum_{i=1}^{n}x_i}{n}}\rfloor`.

**In pygamma-agreement:**
We chose not to replicate this small discrepancy as it seemed like a bug, and didn't
weight too much on the value of the gamma agreement.


Although it has no influence over which alignment will be considered the best alignment, it slighly changes the value
of the disorders, which tweaks the gamma agreement for small continua.


2. Minimal distance between pivots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In [mathet2015]_, section 5.2.1, *Mathet et Al.* explain their method of sampling continuua by shuffling the reference
continuum using random shift positions; and they specify a constraint on those positions :


    "To limit this phenomenon, we do not allow the distance between two shifts to be less than the average length of units."

**In the Gamma Software:**
The value used for this minimal distance is actually **half** the average length of units.

**In pygamma-agreement:**
We decided to include this discrepancy in the `ShuffleContinuumSampler` as it is designed to
mimic the java implementation's, as opposed to our `StatisticalContinuumSampler` used by default by ``pygamma-agreement``.

3. Pairing confidence
^^^^^^^^^^^^^^^^^^^^^

In [mathet2018]_, section 4.2.3, the pairing confidence of a pair of annotations is defined as such:


    "for   :math:`pair_i = (u_j, u_k)`,  :math:`p_i = max(0, 1 - d_{pos}(u_j, u_k))`"

**In the Gamma Software:**
Their implementation of this formula uses a combined dissimilarity
:math:`d_{\alpha, \beta} = \alpha d_{pos} + \beta d_{cat}`, which transforms the formula for the pairing confidence this
way: ":math:`pair_i = (u_j, u_k)`,  :math:`p_i = max(0, 1 - \alpha \times d_{pos}(u_j, u_k))`".

**In pygamma-agreement:**
Although it looked a lot like a bug, ignoring it makes the values of gamma-cat/k too different from those
of the gamma software. We chose to include the alpha factor, as setting it to `1.0` can remove the discrepancy :

.. code-block:: python

    dissimilarity = CombinedCategoricalDissimilarity(alpha=3.0, # Set any alpha value you want
                                                     beta=2.0,
                                                     delta_empty=1.0)

    gamma_results = continuum.compute_gamma(dissimilarity)
    dissimilarity.alpha = 1.0  # gamma_results stores the dissimilarity used for computing the
                               # best alignments, as it is needed for computing gamma-cat
    print(f"gamma-cat is {gamma_results.gamma_cat}")  # Gamma-k can also be influenced by alpha
    dissimilarity.alpha = 3.0  # Add this line if you want to reuse the dissimilarity with alpha = 3

4. Best alignment
^^^^^^^^^^^^^^^^^

The Mixed Integer Programming solvers used in `pygamma-agreement` not being the same as the one used by the
Gamma-Software, it is possible that the best alignments found by both software are different if multiple best
alignments with the same disorder exist.

**In the Gamma Software:**
The MIP solver used is ``liblpsolve``

**In pygamma-agreement:**
The MIP solver used is ``GLPK``, or the faster ``CBC`` if it is installed.

Although this doesn't weight on the value of gamma, it slightly does on gamma-cat and gamma-k's. Thus, there is no way
to obtain for sure the same results as the Gamma Software for gamma-cat/k.


How to obtain the results from the Gamma Software
-------------------------------------------------

This part explains how one can obtain an *almost* similar output as the Gamma Software using ``pygamma-agreement``.
The two main differences being :

Sampler
^^^^^^^
The sampler ``pygamma-agreement`` uses by default is **not** the one described in [mathet2015]_. Our sampler collects
statistical data about the input continuum (averages / standard deviation of several values such as length of
annotations), used then to generate the samples. We made this choice because we felt that their sampler, which simply
re-shuffles the input continuum, was unconvincing for the need of 'true' randomness.

To re-activate their sampler, you can use the ``--mathet-sampler`` (or ``-m``) option when using the command line, or
manually set the sampler used for computing the gamma agreement in python :

.. code-block:: python

    from pygamma_agreement import ShuffleContinuumSampler
    ...
    gamma_results = continuum.compute_gamma(sampler=ShuffleContinuumSampler(),
                                            precision_level=0.01)

Alpha value
^^^^^^^^^^^
The Gamma Software uses :math:`\alpha=3` in the combined categorical dissimilarity.

To set it in the command line interface, simply use the ``--alpha 3`` (or ``-a 3``) option.
In python, you need to manually create the combined categorical dissimilarity with the ``alpha=3`` parameter.

.. code-block:: python

    dissim = CombinedCategoricalDissimilarity(alpha=3)
    gamma_results = continuum.compute_gamma(dissim,
                                            sampler=ShuffleContinuumSampler(),
                                            precision_level=0.01)


Bugs in former versions of pygamma-agreement
--------------------------------------------

This section adresses fatal errors in release `0.1.6` of ``pygamma-agreement``, whose consequences were a wrong
output for gamma or other values. Those have been fixed in version `1.0.0`.

1. Average number of annotations per annotator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In [mathet2015]_, section 4.3, a value is defined as such:

    "let :math:`\bar{x}={\frac{\sum_{i=1}^{n}x_i}{n}}` be the average number of annotations per annotator"

A misreading made us interpret this value as the **total number of annotations** in the continuum. Thus, the values
calculated by ``pygamma-agreement`` were strongly impacted (a difference as big as *0.2* for small continua).

2. Minimal distance between pivots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In [mathet2015]_, section 5.2.1, *Mathet et Al.* explain their method of sampling continuua by shuffling the reference
continuum using random shift positions; and they specify a constraint on those positions :


    "To limit this phenomenon, we do not allow the distance between two shifts to be less than the average length of units."

In the previous version of the library, we overlooked this specificity of the sampling algorithm, which made the gamma
values slightly bigger than expected (even after correction of the previous, far more impactful error).


..  [mathet2015] Yann Mathet et Al.
    The Unified and Holistic Method Gamma (γ) for Inter-Annotator Agreement
    Measure and Alignment (Yann Mathet, Antoine Widlöcher, Jean-Philippe Métivier)

..  [mathet2018] Yann Mathet
    The Agreement Measure Gamma-Cat : a Complement to Gamma Focused on Categorization of a Continuum
    (Yann Mathet 2018)