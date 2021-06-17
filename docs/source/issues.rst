======
Issues
======


Gamma Software issues
---------------------


This section aims to explain the specificities of the ``pygamma-agreement`` library in comparison to the closed-source
java implementation by the original authors of [mathet2015]_ & [mathet2018]_ ("Gamma Software"), which can be downloaded
`here <https://gamma.greyc.fr/>`_.

We have reversed-engineered their program in an effort to find why some of its results were different from
``pygamma-agreement``'s. Although we found some undocumented implementations and bugs in there, it also helped
fixing fatal errors that made ``pygamma-agreement``'s ouputs wrong.

.. warning::

    What we call "undocumented" are choices of implementation found in the Gamma Software that are not mentionned
    or explained in [mathet2015]_ or [mathet2018]_. We made the choice of replicating those in
    pygamma-agreement as we suppose that Mathet et Al. being responsible for the theory behind the gamma agreement,
    their choices must have been carefully thought.

1. Average number of annotations per annotator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In [mathet2015]_, section 4.3, a value is defined as such:


    "let :math:`\bar{x}={\frac{\sum_{i=1}^{n}x_i}{n}}` be the average number of annotations per annotator"

This value is involved in the computation of the disorder of an alignment. In the Gamma Software, an
int-instead-of-float division transforms this value into :math:`\bar{x}=\lfloor{\frac{\sum_{i=1}^{n}x_i}{n}}\rfloor`.
Although it has no influence over which alignment will be considered the best alignment, it slighly changes the value
of the disorders, which tweaks the gamma agreement for small continua.
We chose not to replicate this as it seemed like a bug, and didn't weight too much on the value of the gamma agreement.

2. Minimal distance between pivots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In [mathet2015]_, section 5.2.1, *Mathet et Al.* explain their method of sampling continuua by shuffling the reference
continuum using random shift positions; and they specify a constraint on those positions :


    "To limit this phenomenon, we do not allow the distance between two shifts to be less than the average length of units."

It seems however that in the java implementation, this minimal distance is **half** the average length of units;
``pygamma-agreement`` 's `ShuffleContinuumSampler`, which tries to mimic the sampling method described in the paper,
takes the half average length of units just like the Gamma Software.

3. Pairing confidence
^^^^^^^^^^^^^^^^^^^^^

In [mathet2018]_, section 4.2.3, the pairing confidence of a pair of annotations is defined as such:


    "for   :math:`pair_i = (u_j, u_k)`,  :math:`p_i = max(0, 1 - d_{pos}(u_j, u_k))`"

However, their implementation of this formula uses a combined dissimilarity
:math:`d_{\alpha, \beta} = \alpha d_{pos} + \beta d_{cat}`, which transforms the formula for the pairing confidence this
way:


    "for   :math:`pair_i = (u_j, u_k)`,  :math:`p_i = max(0, 1 - \alpha \times d_{pos}(u_j, u_k))`"

Although it looked a lot like a bug, ignoring it makes the values of gamma-cat/k too different from those
of the gamma software, so we chose to replicate this formula for ``pygamma-agreement``. Here's how to replicate the
intended formula:

.. code-block:: python

    dissimilarity = CombinedCategoricalDissimilarity(continuum.categories,
                                                     alpha=3.0, # Set any alpha value you want
                                                     beta=2.0,
                                                     delta_empty=1.0)

    gamma_results = continuum.compute_gamma(dissimilarity)
    dissimilarity.alpha = 1.0  # gamma_results stores the dissimilarity used for computing the
                               # best alignments, as it is needed for computing gamma-cat
    print(f"gamma-cat is {gamma_results.gamma_cat}")  # Gamma-k can also be influenced by alpha
    dissimilarity.alpha = 3.0  # Add this line if you want to reuse the dissimilarity

Bugs in former versions of pygamma-agreement
--------------------------------------------

This section adresses fatal errors in release `0.1.6` of ``pygamma-agreement``, whose consequences were a wrong
output for gamma or other values. Those have been fixed in version `?`.

1. Average number of annotations per annotator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In [mathet2015]_, section 4.3, a value is defined as such:

.. pull-quote::

    let :math:`\bar{x}={\frac{\sum_{i=1}^{n}x_i}{n}}` be the average number of annotations per annotator

A misreading made us interpret this value as the ***total number of annotations*** in the continuum. Thus, the values
calculated by ``pygamma-agreement`` were strongly impacted, as those can differ a lot more than the intended one between
sampled continua.

2. Minimal distance between pivots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In [mathet2015]_, section 5.2.1, *Mathet et Al.* explain their method of sampling continuua by shuffling the reference
continuum using random shift positions; and they specify a constraint on those positions :


    "To limit this phenomenon, we do not allow the distance between two shifts to be less than the average length of units."

In the previous version of the library, we overlooked this specificity of the sampling algorithm, which made the gamma
values slightly bigger than expected (after correction of the previous, far more impactful bug).


..  [mathet2015] Yann Mathet et Al.
    The Unified and Holistic Method Gamma (γ) for Inter-Annotator Agreement
    Measure and Alignment (Yann Mathet, Antoine Widlöcher, Jean-Philippe Métivier)

..  [mathet2018] Yann Mathet
    The Agreement Measure Gamma-Cat : a Complement to Gamma Focused on Categorization of a Continuum
    (Yann Mathet 2018)