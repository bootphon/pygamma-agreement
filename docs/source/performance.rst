===========
Performance
===========

This section aims to describe and explain the performances of the ``pygamma-agreement``
library in terms of time and memory usage.


Computational complexity
~~~~~~~~~~~~~~~~~~~~~~~~
For a continuum with :

- :math:`p` annotators
- :math:`n` annotations per annotator
- :math:`N` random samples involved (depends on the required precision)

The computational complexity of ``Continuum.compute_gamma()`` is :

.. math::

    C(N, n, p) = O(s \times N \times n^p)

The factor :math:`s` depends on the continuum, for instance, lots of overlapping of annotations
means a higher factor. We're aware that for a high amount of annotators, the computation
takes a lot of time and cannot be viable for realistic input.

The theorical complexity cannot be reduced, however we have found a :ref:`workaround <fast_option>` that sacrifices
precision for a **significant** gain in complexity.

Moreover, the computation of the sample's dissimilarity is parellelized, which means
that the complexity can be reduced to at best :math:`O(s \times N \times n^p / c)`
with :math:`c` CPUs.

.. _fast_option:

Fast option
~~~~~~~~~~~

The ``Continuum.compute_gamma()`` method allows to set the *"fast"* option, which uses a different algorithm
for determining the best alignment of a disorder. Although there is no theory to back the precision of the algorithm,
we have found out that it gives the **exact** results for the best alignments, for real data.

It uses the fact that alignments are made using locality of annotations, so it is only precise with a positional
dissimilarity or a combined dissimilarity with :math:`\alpha > 2 \times \beta`.

Here are the performance comparisons between the two algorithms :

.. figure:: images/time2annotators.png
  :scale: 49%
  :alt: time to compute gamma (8 CPUs, 2 annotators)
  :align: right

  **2 annotators**


.. figure:: images/time3annotators.png
  :scale: 49%
  :alt: time to compute gamma (8 CPUs, 3 annotators)
  :align: left

  **3 annotators**

based on these graphs, we have decided that if unspecified, fast-gamma will be enabled by default when the number of
annotators is more than **3**, because otherwise,

